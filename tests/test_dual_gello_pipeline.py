import time

import numpy as np
import pytest

from gello_teleop.dual_gello_collect import DualGelloPipeline, SecondOrderPositionFollower
from gello_teleop.gello_hardware import GelloReader
from nero_collection.arms.base import ArmState


CONFIG = "gello_teleop/config/xarm7_gello_dual_dataset.yaml"


class FakeArm:
    def __init__(self, side, robot):
        self.name = side
        self.dof = len(robot.reset_q)
        self.q = np.asarray(robot.reset_q, dtype=float).copy()
        self.commands = []

    def connect(self):
        pass

    def enable(self):
        pass

    def move_to_reset(self, q, *, speed, acceleration):
        self.q = np.asarray(q, dtype=float).copy()

    def read_state(self):
        return ArmState(
            q=self.q.copy(), dq=np.zeros(self.dof), ddq=np.zeros(self.dof),
            ee_pose=np.eye(4), torque=np.zeros(self.dof),
            current=np.full(self.dof, np.nan), timestamp_us=time.time_ns() // 1000,
            dq_valid=True, torque_valid=True, current_valid=False,
        )

    def command_joint_positions(self, q):
        self.q = np.asarray(q, dtype=float).copy()
        self.commands.append(self.q.copy())

    def disconnect(self):
        pass


class FakeReader:
    def __init__(self, config):
        self.config = config
        self.raw = np.r_[np.asarray(config.leader_reference_q, dtype=float), 0.0]

    def read(self):
        return self.raw.copy()

    def prepare_alignment(self):
        pass

    def set_torque(self, *args, **kwargs):
        pass

    def close(self):
        pass


def make_pipeline():
    import tempfile
    from scripts.smoke_dual_gello_pipeline import write_simulation_config
    with tempfile.TemporaryDirectory() as directory:
        cfg = write_simulation_config(CONFIG, directory)
        import yaml
        data = yaml.safe_load(cfg.read_text())
        data['active_arms'] = ['left', 'right']
        data['gripper'] = dict(enabled=False, command_enabled=False)
        data['cameras'] = []
        data['require_cameras'] = False
        cfg.write_text(yaml.safe_dump(data))
        return DualGelloPipeline(cfg, arm_factory=FakeArm, reader_factory=FakeReader)


class ExitArm(FakeArm):
    """Observe controller mode/enable at disconnect without touching hardware."""

    def __init__(self, side, robot, events):
        super().__init__(side, robot)
        self.events = events
        self._enabled = False
        self.mode = 0

    def enable(self):
        self._enabled = True
        self.mode = 1
        self.events.append((self.name, 'enable'))

    def move_to_reset(self, q, *, speed, acceleration):
        assert self._enabled
        self.mode = 0
        self.events.append((self.name, 'reset', np.asarray(q).copy(), speed, acceleration))
        super().move_to_reset(q, speed=speed, acceleration=acceleration)

    def hold_position(self, **kwargs):
        assert self._enabled
        self.mode = 0
        self.events.append((self.name, 'hold', self.q.copy()))
        return self.q.copy()

    def control_status(self):
        return dict(mode=self.mode, state=2, error_code=0)

    def disable(self):
        pytest.fail('Exit must retain xArm motor enable')

    def disconnect(self):
        self.events.append((self.name, 'disconnect', self.mode, self._enabled, self.q.copy()))


def exit_pipeline():
    pipeline = make_pipeline()
    events = []
    pipeline._arm_factory = lambda side, robot: ExitArm(side, robot, events)
    pipeline.alignment.update(reset_samples=2, reset_speed_rad_s=.3, reset_acc_rad_s2=.4)
    return pipeline, events


@pytest.mark.parametrize('stage', ['reset', 'following'])
@pytest.mark.parametrize('key', ['q', 'ctrl_c'])
def test_q_and_ctrl_c_reset_both_then_disconnect_with_controller_hold(monkeypatch, stage, key):
    from unittest.mock import Mock
    import gello_teleop.dual_gello_collect as collect
    pipeline, events = exit_pipeline()
    interrupted = False
    def read_key(_timeout):
        nonlocal interrupted
        ready = pipeline.state == ('connected' if stage == 'reset' else 'following')
        if ready and not interrupted:
            interrupted = True
            if key == 'ctrl_c':
                raise KeyboardInterrupt
            return 'q'
        return None
    keys = Mock(is_tty=True, read_key=read_key)
    context = Mock()
    context.__enter__ = Mock(return_value=keys)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(collect, 'TerminalKeys', lambda: context)
    monkeypatch.setattr(collect, 'DualGelloPipeline', lambda _config: pipeline)
    original_reset = pipeline.reset_followers
    def observed_reset(*args, **kwargs):
        if kwargs.get('abort') is pipeline.exit_reset_abort:
            threads = [pipeline.control_thread, pipeline.sampling_thread]
            threads += [worker.thread for worker in pipeline.producers+pipeline.state_producers]
            assert all(thread is None or not thread.is_alive() for thread in threads)
        return original_reset(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'reset_followers', observed_reset)
    code = collect.main(['-c', CONFIG])
    assert code == (0 if key == 'q' and stage == 'following' else 130)
    assert pipeline.exit_reset_complete and pipeline.state == 'stopped'
    first_disconnect = next(i for i, event in enumerate(events) if event[1] == 'disconnect')
    assert len([event for event in events[:first_disconnect] if event[1] == 'reset']) == 4
    for (side, robot, _), arm in zip(pipeline.configs, pipeline.arms):
        side_events = [event for event in events if event[0] == side]
        reset = [event for event in side_events if event[1] == 'reset'][-1]
        np.testing.assert_array_equal(reset[2], robot.reset_q)
        assert reset[3:] == (.3, .4)
        assert sum(event[1] == 'hold' for event in side_events) == 1
        assert side_events[-1][1:4] == ('disconnect', 0, True)
        np.testing.assert_array_equal(side_events[-1][4], robot.reset_q)
        assert arm._enabled
    count = len(events)
    pipeline.close()
    assert len(events) == count  # Repeated cleanup must not send another target.


def test_exit_reset_timeout_holds_current_q_and_retains_enable(monkeypatch):
    pipeline, events = exit_pipeline()
    pipeline.connect()
    pipeline.state = 'holding'
    actuals = []
    for arm in pipeline.arms:
        arm.enable()
        arm.q += .2
        actuals.append(arm.q.copy())
        def stalled_reset(q, *, speed, acceleration, arm=arm):
            arm.mode = 0
            events.append((arm.name, 'reset', np.asarray(q).copy(), speed, acceleration))
        monkeypatch.setattr(arm, 'move_to_reset', stalled_reset)
    pipeline.alignment['reset_timeout_s'] = .04
    try:
        assert not pipeline.shutdown_reset_and_hold()
        assert not pipeline.exit_reset_complete and pipeline.exit_hold_complete
    finally:
        pipeline.close()
    for arm, actual in zip(pipeline.arms, actuals):
        np.testing.assert_array_equal(arm.q, actual)
        assert arm._enabled and arm.mode == 0
        assert sum(event[1] == 'hold' for event in events if event[0] == arm.name) == 2


def test_partial_exit_reset_failure_cancels_both_targets_before_disconnect(monkeypatch):
    pipeline, events = exit_pipeline()
    pipeline.connect()
    pipeline.state = 'holding'
    for arm in pipeline.arms:
        arm.enable()
        arm.q += .2
    def fail_reset(*_args, **_kwargs):
        raise RuntimeError('right reset command failed')
    monkeypatch.setattr(pipeline.arms[1], 'move_to_reset', fail_reset)
    try:
        assert not pipeline.shutdown_reset_and_hold()
        assert pipeline.exit_hold_complete and not pipeline.exit_reset_complete
    finally:
        pipeline.close()
    resets = [i for i, event in enumerate(events) if event[1] == 'reset']
    assert len(resets) == 1
    fallback_holds = [event for event in events[resets[0]+1:] if event[1] == 'hold']
    assert [event[0] for event in fallback_holds] == ['left', 'right']
    assert all(arm._enabled for arm in pipeline.arms)
    assert events[-2][1] == events[-1][1] == 'disconnect'


def test_exit_during_thread_startup_still_resets_and_holds():
    import threading
    from types import SimpleNamespace
    pipeline, events = exit_pipeline()
    pipeline.connect()
    for arm in pipeline.arms:
        arm.enable()
    pipeline.control_thread = threading.Thread(target=lambda: None)
    pipeline.state_producers = [SimpleNamespace(thread=threading.Thread(target=lambda: None))]
    try:
        assert pipeline.shutdown_reset_and_hold()
    finally:
        pipeline.close()
    assert all(arm.mode == 0 and arm._enabled for arm in pipeline.arms)


@pytest.mark.parametrize('invalid', ['invalid_flag', 'nan', 'stale'])
def test_exit_reset_rejects_invalid_or_stale_feedback(monkeypatch, invalid):
    from dataclasses import replace
    pipeline, events = exit_pipeline()
    pipeline.connect()
    pipeline.state = 'holding'
    for arm in pipeline.arms:
        arm.enable()
    read = pipeline.arms[0].read_state
    def bad_read():
        state = read()
        if invalid == 'invalid_flag':
            return replace(state, q_valid=False)
        if invalid == 'nan':
            return replace(state, q=np.full(7, np.nan))
        return replace(state, timestamp_us=state.timestamp_us-1_000_000)
    monkeypatch.setattr(pipeline.arms[0], 'read_state', bad_read)
    try:
        assert not pipeline.shutdown_reset_and_hold()
        assert pipeline.exit_hold_complete and not pipeline.exit_reset_complete
    finally:
        pipeline.close()
    assert all(arm._enabled for arm in pipeline.arms)


def test_exit_reset_waits_until_controller_is_no_longer_moving(monkeypatch):
    pipeline, events = exit_pipeline()
    pipeline.connect()
    pipeline.state = 'holding'
    for arm in pipeline.arms:
        arm.enable()
    statuses = iter([1, 1, 2])
    calls = []
    def moving_status():
        state = next(statuses, 2)
        calls.append(state)
        return dict(mode=0, state=state, error_code=0)
    monkeypatch.setattr(pipeline.arms[0], 'control_status', moving_status)
    try:
        assert pipeline.shutdown_reset_and_hold()
        assert calls == [1, 1, 2]
    finally:
        pipeline.close()


def test_second_ctrl_c_stops_exit_reset_and_preserves_hold(monkeypatch):
    import signal
    from unittest.mock import Mock
    import gello_teleop.dual_gello_collect as collect
    pipeline, events = exit_pipeline()
    def interactive(**_kwargs):
        pipeline.state = 'holding'
        for arm in pipeline.arms:
            arm.enable()
            arm.q += .2
        move = pipeline.arms[0].move_to_reset
        def interrupt_after_accepting_target(*args, **kwargs):
            move(*args, **kwargs)
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        monkeypatch.setattr(pipeline.arms[0], 'move_to_reset', interrupt_after_accepting_target)
        return 0
    monkeypatch.setattr(collect, 'DualGelloPipeline', lambda _: pipeline)
    monkeypatch.setattr(pipeline, 'interactive', interactive)
    previous_handler = signal.getsignal(signal.SIGINT)
    assert collect.main(['-c', CONFIG]) == 1
    assert signal.getsignal(signal.SIGINT) is previous_handler
    assert pipeline.exit_reset_abort.is_set() and not pipeline.exit_reset_complete
    assert len([event for event in events if event[1] == 'reset']) == 1
    assert all(arm._enabled and arm.mode == 0 for arm in pipeline.arms)
    assert len([event for event in events if event[1] == 'hold']) == 4


def test_fault_exit_holds_in_place_without_reset_or_disable(monkeypatch):
    import gello_teleop.dual_gello_collect as collect
    pipeline, events = exit_pipeline()
    def fault(**_kwargs):
        pipeline.state = 'following'
        for arm in pipeline.arms:
            arm.enable()
            arm.q += .2
        pipeline.stop_event.set()
        raise RuntimeError('stale GELLO target')
    monkeypatch.setattr(collect, 'DualGelloPipeline', lambda _: pipeline)
    monkeypatch.setattr(pipeline, 'interactive', fault)
    with pytest.raises(RuntimeError, match='stale GELLO target'):
        collect.main(['-c', CONFIG])
    assert not any(event[1] == 'reset' for event in events)
    assert len([event for event in events if event[1] == 'hold']) == 2
    assert all(arm._enabled and arm.mode == 0 for arm in pipeline.arms)


def test_controller_fault_exit_retains_protection_and_holds_other_arm(monkeypatch, caplog):
    import gello_teleop.dual_gello_collect as collect
    from ufactory_devices.robot.xarm_adapter import XArmControllerFault
    pipeline, events = exit_pipeline()
    def fault(**_kwargs):
        pipeline.state = 'following'
        for arm in pipeline.arms:
            arm.enable()
        error = XArmControllerFault('left', 31, operation='set_servo_angle_j', api_code=1)
        def protected_hold(**_kwargs):
            events.append(('left', 'protected_stop'))
            raise error
        monkeypatch.setattr(pipeline.arms[0], 'hold_position', protected_hold)
        pipeline.control_error = error
        pipeline.stop_event.set()
        pipeline.poll()
    monkeypatch.setattr(collect, 'DualGelloPipeline', lambda _: pipeline)
    monkeypatch.setattr(pipeline, 'interactive', fault)
    assert collect.main(['-c', CONFIG]) == 1
    assert not any(event[1] == 'reset' for event in events)
    assert [event[:2] for event in events if event[1] in {'protected_stop', 'hold'}] == [
        ('left', 'protected_stop'), ('right', 'hold')]
    assert 'C31' in caplog.text and 'SDK API code 1' in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_sampling_failure_retains_primary_controller_fault(monkeypatch):
    import gello_teleop.dual_gello_collect as collect
    from ufactory_devices.robot.xarm_adapter import XArmControllerFault
    from unittest.mock import Mock
    pipeline = make_pipeline()
    error = XArmControllerFault('left', 31, api_code=1)
    pipeline.control_error = error
    monkeypatch.setattr(collect.FixedRateTicker, 'wait', Mock(side_effect=RuntimeError('feedback unavailable')))
    pipeline._sampling_loop()
    assert pipeline.control_error is error
    assert pipeline.stop_event.is_set() and pipeline.sampling_stop.is_set()


def test_collision_config_reaches_active_adapters_and_recorded_metadata():
    pipeline = make_pipeline()
    pipeline.control['collision_sensitivity'] = 3
    pipeline.raw['arms']['left']['collision_sensitivity'] = 1
    pipeline.raw['arms']['right']['collision_sensitivity'] = None
    collection = pipeline._make_collection_config()
    for (side, robot, _), pair, expected in zip(pipeline.configs, collection.teleop.master_slave, [1, None]):
        adapter = pipeline._default_arm_factory(side, robot)
        assert adapter.config.config_kwargs['collision_sensitivity'] == expected
        assert pair.follower.config_kwargs['collision_sensitivity'] == expected
    del pipeline.raw['arms']['left']['collision_sensitivity']
    side, robot, _ = pipeline.configs[0]
    assert pipeline._default_arm_factory(side, robot).config.config_kwargs['collision_sensitivity'] == 3


def test_keyboard_interrupt_during_connect_does_not_enable_or_reset(monkeypatch):
    import gello_teleop.dual_gello_collect as collect
    pipeline, events = exit_pipeline()
    def interrupted_connect():
        pipeline.arms.append(ExitArm(*pipeline.configs[0][:2], events))
        raise KeyboardInterrupt
    monkeypatch.setattr(collect, 'DualGelloPipeline', lambda _: pipeline)
    monkeypatch.setattr(pipeline, 'connect', interrupted_connect)
    assert collect.main(['-c', CONFIG]) == 130
    assert [event[1] for event in events] == ['disconnect']
    assert not pipeline.arms[0]._enabled


def test_check_config_exit_has_no_motion_side_effects(monkeypatch):
    import gello_teleop.dual_gello_collect as collect
    pipeline, events = exit_pipeline()
    monkeypatch.setattr(collect, 'DualGelloPipeline', lambda _: pipeline)
    assert collect.main(['-c', CONFIG, '--check-config']) == 0
    assert events == []


def test_position_reference_is_bounded_and_takeover_starts_at_measured_pose():
    follower = SecondOrderPositionFollower(2, mode="second_order", kp=10, kd=1,
                                           max_velocity=0.5, max_acceleration=1, max_step=0.02,
                                           max_tracking_error=0.1)
    follower.initialize(np.zeros(2))
    first = follower.update(np.ones(2), np.zeros(2), 0.01)
    assert np.max(np.abs(first)) <= 0.02


def test_direct_reference_enforces_velocity_on_position_step():
    follower = SecondOrderPositionFollower(2, mode='direct', max_velocity=.3, max_step=.08)
    follower.initialize(np.zeros(2))
    q = follower.update(np.ones(2), np.zeros(2), .01)
    assert np.max(np.abs(q)) <= .003+1e-12


def test_full_pipeline_saves_cameras_grippers_and_causal_joint_commands(tmp_path):
    from scripts.smoke_dual_gello_pipeline import run_smoke
    import yaml
    source = yaml.safe_load(open(CONFIG))
    source['active_arms'] = ['left', 'right']
    source['cameras'] = [dict(name=f'{owner} wrist', backend='mock', width=64, height=48,
                             fps=30, output_size=[224, 224]) for owner in ('left', 'right')]
    source['gripper']['mode'] = 'follow'
    config = tmp_path/'follow.yaml'
    config.write_text(yaml.safe_dump(source))
    report = run_smoke(config, tmp_path/'full.h5', duration=.6)
    assert report['samples'] >= 10
    assert report['metadata']['synthetic_robots'] is True
    assert report['cameras']['left wrist']['frames'] >= 2
    assert report['cameras']['right wrist']['frames'] >= 2
    assert report['metadata']['queue_overflow_count'] == 0


def test_gripper_command_fault_stops_all_control(tmp_path):
    import yaml
    from scripts.smoke_dual_gello_pipeline import SimulatedArm, SimulatedLeader, ROOT, write_simulation_config
    cfg = yaml.safe_load(write_simulation_config(ROOT/'gello_teleop/config/xarm7_gello_dual_dataset.yaml', tmp_path).read_text())
    cfg['gripper']['mode'] = 'follow'
    cfg['cameras'] = []
    path = tmp_path/'fault.yaml'
    path.write_text(yaml.safe_dump(cfg))
    class BrokenGripper(SimulatedArm):
        def command_gripper(self, *_args, **_kwargs):
            raise RuntimeError('gripper disconnected')
    pipeline = DualGelloPipeline(path, arm_factory=BrokenGripper, reader_factory=SimulatedLeader)
    try:
        pipeline.connect()
        pipeline.reset()
        pipeline.confirm_alignment()
        pipeline.takeover()
        deadline = time.monotonic()+1
        while not pipeline.stop_event.is_set() and time.monotonic()<deadline:
            time.sleep(.005)
        assert pipeline.stop_event.is_set()
        with pytest.raises(RuntimeError, match='Gripper I/O failed'):
            pipeline.poll()
    finally:
        pipeline.close()


def test_damping_current_is_signed_filtered_and_limited_without_nm_claim():
    cfg = type("Damping", (), {
        "damping_gain": (2.0, 2.0), "damping_brake_gain": (1.0, 1.0),
        "damping_current_limit": (3, 3), "damping_velocity_threshold": 0.1,
        "damping_velocity_filter_alpha": 0.5, "weak_hold_enabled": True,
        "weak_hold_gain": (1.0, 1.0), "weak_hold_limit": (0.5, 0.5),
    })()
    current = GelloReader.compute_damping_current(np.array([1.0, -1.0]), np.zeros(2), 0.01, cfg,
                                                  np.zeros(2))
    assert np.all(np.abs(current) <= 3.0)
    assert current[0] < 0 and current[1] > 0


def test_unaligned_cannot_takeover_and_retakeover_checks_current_pose():
    pipeline = make_pipeline()
    pipeline.connect()
    pipeline.reset()
    with pytest.raises(RuntimeError, match="alignment"):
        pipeline.takeover()
    pipeline.confirm_alignment()
    pipeline.takeover()
    assert pipeline.state == "following"
    time.sleep(0.03)
    pipeline.start_episode()
    assert pipeline.state == "recording"
    pipeline.stop_episode(save=False)
    assert pipeline.state == 'following'
    pipeline.freeze_following()
    assert pipeline.state == "holding"
    pipeline.arms[0].q = pipeline.arms[0].q + 0.2
    with pytest.raises(RuntimeError, match="moved after alignment"):
        pipeline.takeover()
    pipeline.arms[0].q = np.asarray(pipeline.configs[0][1].reset_q, dtype=float)
    pipeline.confirm_alignment()
    pipeline.takeover()
    assert pipeline.state == "following"
    pipeline.close()


def test_recorded_q_cmd_is_causal_previous_successful_command():
    pipeline = make_pipeline()
    pipeline.connect()
    pipeline.reset()
    pipeline.confirm_alignment()
    pipeline.takeover()
    time.sleep(0.04)
    pipeline.start_episode()
    time.sleep(0.04)
    pipeline.poll()
    assert pipeline.buffer is not None and pipeline.buffer.sample_count > 0
    q_cmd = np.asarray(pipeline.buffer.teleop_data["q_cmd"][0])
    q_follower = np.asarray(pipeline.buffer.teleop_data["q_follower"][0])
    np.testing.assert_allclose(pipeline.buffer.teleop_data["delta_q"][0], q_cmd - q_follower)
    pipeline.stop_episode(save=False)
    pipeline.close()


def test_expired_leader_stops_both_arm_commands():
    pipeline = make_pipeline()
    pipeline.connect()
    pipeline.reset()
    pipeline.confirm_alignment()
    pipeline.takeover()
    pipeline.leader_max_age_s = 1.0e-9
    time.sleep(0.04)
    with pytest.raises(RuntimeError, match="expired"):
        pipeline.poll()
    assert pipeline.stop_event.is_set()
    pipeline.close()


def test_damping_opposes_encoder_motion_with_flipped_joint_mapping():
    from types import SimpleNamespace
    signs = np.array([1., -1.])
    cfg = SimpleNamespace(damping_gain=[8., 8.], damping_brake_gain=[8., 8.],
                          damping_current_limit=[15, 15], damping_velocity_threshold=.5,
                          damping_velocity_filter_alpha=.35, weak_hold_enabled=False,
                          joint_signs=signs)
    encoder_delta = np.array([.02, .02])
    current = GelloReader.compute_damping_current(encoder_delta*signs, np.zeros(2), .1, cfg)
    assert np.all(current*encoder_delta < 0)
    assert np.allclose(current, [-1.6, -1.6])
    assert np.allclose(GelloReader.compute_damping_current(np.zeros(2), None, .1, cfg), 0.)


def test_collection_damping_overrides_both_sides_without_changing_calibration(tmp_path):
    import yaml
    from scripts.smoke_dual_gello_pipeline import write_simulation_config
    path = write_simulation_config(CONFIG, tmp_path)
    data = yaml.safe_load(path.read_text())
    data['gripper']['mode'] = 'follow'
    original = yaml.safe_load(open(CONFIG))['gello_damping']
    data['active_arms'] = ['left', 'right']
    data['gello_damping'] = original
    data['gello_damping']['sides'] = {'right': {'damping_gain': [4]*7}}
    path.write_text(yaml.safe_dump(data))
    calibration_path = data['gello_config']
    before = open(calibration_path).read()
    pipeline = DualGelloPipeline(path)
    left, right = [leader for _, _, leader in pipeline.configs]
    assert left.damping_enabled and right.damping_enabled
    assert left.fps == right.fps == 30
    assert left.damping_gain == [8]*7 and right.damping_gain == [4]*7
    assert not left.weak_hold_enabled
    assert open(calibration_path).read() == before
    data['gello_damping']['sides']['right']['damping_current_limit'] = [-1]*7
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='damping_current_limit'):
        DualGelloPipeline(path)


def test_damping_io_failure_stops_following_and_releases_current():
    import threading
    from types import SimpleNamespace
    from gello_teleop.dual_gello_collect import _LeaderProducer
    stop = threading.Event()
    released = []
    reader = SimpleNamespace(read=lambda: (_ for _ in ()).throw(RuntimeError('USB lost')),
                             disable_current_damping=lambda: released.append(True))
    mapper = SimpleNamespace(config=SimpleNamespace(port='fake'))
    producer = _LeaderProducer(reader, mapper, .03, stop, damping_config=object())
    producer._run()
    assert stop.is_set() and released == [True]


def test_current_damping_configures_all_motors_before_enabling_watchdogs():
    import threading
    from types import SimpleNamespace
    # Model an EEPROM startup taking over 500ms across all seven motors.
    # The previous per-motor enable sequence expires motor 1 in this scenario.
    reader = GelloReader.__new__(GelloReader)
    reader.config = SimpleNamespace(joint_ids=tuple(range(1, 8)))
    reader.lock = threading.Lock()
    reader.detect_capabilities = lambda: {i: {'current_control': True} for i in range(1, 8)}
    clock = [0.]
    enabled_at = {}
    configured = set()
    goals = {}
    writes = []
    def write(motor, address, value, size=1):
        writes.append((motor, address, value))
        if address in (11, 38):
            clock[0] += .05
        for started in enabled_at.values():
            assert clock[0]-started < .5, 'startup watchdog expired'
        if address == 102:
            configured.add(motor)
            goals[motor] = value
        if address == 64 and value == 1:
            assert configured == set(range(1, 8))
            assert goals[motor] == 0
            enabled_at[motor] = clock[0]
    reader._write = write
    cfg = SimpleNamespace(damping_mode='current', damping_gain=[8]*7,
                          damping_current_limit=[15]*7, damping_watchdog_ms=500)
    assert reader.enable_current_damping(cfg)['enabled']
    assert set(enabled_at) == set(range(1, 8))
    last_configuration = max(i for i, (_, address, _) in enumerate(writes) if address in (11, 38, 102))
    first_enable = min(i for i, (_, address, value) in enumerate(writes) if address == 64 and value == 1)
    assert last_configuration < first_enable


def test_alignment_hold_uses_raw_current_80_and_continuous_current_pose():
    import threading
    from types import SimpleNamespace
    reader = GelloReader.__new__(GelloReader)
    reader.config = SimpleNamespace(joint_ids=(1, 2))
    reader.lock = threading.Lock()
    reader.detect_capabilities = lambda: {i: {'model_number': 1200} for i in (1, 2)}
    reader._read_register = lambda motor, address, size: {132: 8000+motor, 52: 0, 48: 4095, 36: 885}[address]
    writes = []
    reader._write = lambda motor, address, value, size=1: writes.append((motor, address, value))
    result = reader.enable_alignment_hold(80)
    assert result['current_limits_raw'] == {1: 1750, 2: 1750}
    for motor in (1, 2):
        assert (motor, 11, 5) in writes
        assert (motor, 102, 80) in writes
        assert (motor, 116, 8000+motor) in writes
    first_enable = next(i for i, (_, address, value) in enumerate(writes) if address == 64 and value == 1)
    assert all(address == 64 and value == 1 for _, address, value in writes[first_enable:])


def test_automatic_alignment_accepts_only_when_every_sample_passes(capsys):
    from types import SimpleNamespace
    from unittest.mock import Mock
    pipeline = DualGelloPipeline.__new__(DualGelloPipeline)
    pipeline.alignment = {'alignment_tolerance_rad': .05, 'alignment_samples': 2}
    good = {'left': np.zeros(7), 'right': np.zeros(7)}
    pipeline.reference_error_report = Mock(side_effect=[good, good])
    pipeline.confirm_alignment = Mock()
    pipeline._consume_samples = Mock()
    keys = SimpleNamespace(read_key=Mock(return_value=None))
    pipeline.wait_for_alignment(keys)
    assert keys.read_key.call_count == 2
    pipeline.confirm_alignment.assert_called_once()
    output = capsys.readouterr().out
    assert 'right:' in output and '自动对齐核对通过' in output
    assert '微调' not in output and '等待对齐' not in output


@pytest.mark.parametrize('passing_samples', [0, 1])
def test_automatic_alignment_stops_on_first_bad_sample_without_retry(passing_samples):
    from types import SimpleNamespace
    from unittest.mock import Mock
    pipeline = DualGelloPipeline.__new__(DualGelloPipeline)
    pipeline.alignment = {'alignment_tolerance_rad': .05, 'alignment_samples': 5}
    good = {'left': np.zeros(7), 'right': np.zeros(7)}
    bad = {'left': np.zeros(7), 'right': np.array([0., 0., .1, 0., 0., 0., 0.])}
    pipeline.reference_error_report = Mock(side_effect=[good]*passing_samples+[bad, good])
    pipeline.confirm_alignment = Mock()
    pipeline._consume_samples = Mock()
    with pytest.raises(RuntimeError, match='自动对齐失败.*right J3='):
        pipeline.wait_for_alignment(SimpleNamespace(read_key=Mock(return_value=None)))
    assert pipeline.reference_error_report.call_count == passing_samples+1
    pipeline.confirm_alignment.assert_not_called()


def test_automatic_alignment_read_failure_reports_original_cause_without_retry():
    from types import SimpleNamespace
    from unittest.mock import Mock
    pipeline = DualGelloPipeline.__new__(DualGelloPipeline)
    pipeline.alignment = {'alignment_tolerance_rad': .05, 'alignment_samples': 5}
    error = RuntimeError('Dynamixel 7 disconnected')
    pipeline.reference_error_report = Mock(side_effect=error)
    pipeline.confirm_alignment = Mock()
    pipeline._consume_samples = Mock()
    with pytest.raises(RuntimeError, match='自动对齐采样失败.*Dynamixel 7 disconnected') as failure:
        pipeline.wait_for_alignment(SimpleNamespace(read_key=Mock(return_value=None)))
    assert failure.value.__cause__ is error
    pipeline.reference_error_report.assert_called_once()
    pipeline.confirm_alignment.assert_not_called()


@pytest.mark.parametrize('device', ['gello', 'xarm'])
def test_automatic_alignment_rejects_nonfinite_feedback(device):
    from types import SimpleNamespace
    from unittest.mock import Mock
    pipeline = make_pipeline()
    try:
        pipeline.connect(); pipeline.reset()
        pipeline.alignment['alignment_samples'] = 1
        if device == 'gello':
            pipeline.readers[0].raw[0] = np.nan
        else:
            pipeline.arms[0].q[0] = np.nan
        with pytest.raises(RuntimeError, match='采样无效'):
            pipeline.wait_for_alignment(SimpleNamespace(read_key=Mock(return_value=None)))
        assert pipeline.state == 'aligning' and pipeline.control_thread is None
        assert all(not arm.commands for arm in pipeline.arms)
    finally:
        pipeline.close()


@pytest.mark.parametrize('entry', ['startup', 't'])
def test_startup_and_t_refuse_takeover_when_interpolation_misses_alignment(entry):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    pipeline = make_pipeline()
    try:
        pipeline.connect()
        if entry == 't':
            pipeline.reset(); pipeline.confirm_alignment(); pipeline.takeover()
            pipeline.start_episode()
            pipeline.freeze_following()
        pipeline.alignment.update(gello_hold_current_raw=80, gello_reset_steps=1,
                                  gello_reset_interval_s=.001)
        for i, reader in enumerate(pipeline.readers):
            reader.enable_alignment_hold = Mock(return_value={'mode': 'current_based_position'})
            reader.alignment_reset_plan = Mock(side_effect=lambda reference, *_: [reference.copy()])
            def command(goal, read=reader, index=i):
                read.raw[:7] = goal
                if index == 0:
                    # Below interpolation's 10 degree tracking limit, but
                    # above the final alignment threshold: no manual catch-up.
                    read.raw[0] += np.deg2rad(7)
            reader.command_alignment_positions = Mock(side_effect=command)
        pipeline.takeover = Mock(wraps=pipeline.takeover)
        keys = SimpleNamespace(is_tty=True, read_key=Mock(return_value=None))
        with pytest.raises(RuntimeError, match='自动对齐失败.*left J1='):
            if entry == 'startup':
                context = Mock()
                context.__enter__ = Mock(return_value=keys)
                context.__exit__ = Mock(return_value=False)
                with patch('gello_teleop.dual_gello_collect.TerminalKeys', return_value=context):
                    pipeline.interactive()
            else:
                pipeline.realign_and_takeover(keys)
        pipeline.takeover.assert_not_called()
        assert not pipeline.alignment_hold_active
        assert pipeline.recording is (entry == 't')
        assert pipeline.control_thread is None or not pipeline.control_thread.is_alive()
    finally:
        pipeline.close()


def test_live_reference_errors_use_saved_origin_and_mapping_without_recalibration():
    pipeline = make_pipeline()
    pipeline.connect(); pipeline.reset()
    offsets = [list(leader.joint_offsets) for _, _, leader in pipeline.configs]
    pipeline.readers[0].raw[1] += np.deg2rad(6)+2*np.pi
    report = pipeline.reference_error_report()
    assert abs(report['left'][1]) == pytest.approx(np.deg2rad(6))
    assert offsets == [list(leader.joint_offsets) for _, _, leader in pipeline.configs]
    pipeline.close()


def test_reset_enables_both_gello_holds_after_follower_reset():
    from unittest.mock import Mock
    pipeline = make_pipeline()
    pipeline.alignment['gello_hold_current_raw'] = 80
    pipeline.alignment['gello_reset_steps'] = 15
    pipeline.alignment['gello_reset_interval_s'] = .001
    pipeline.connect()
    for reader in pipeline.readers:
        reader.enable_alignment_hold = Mock(return_value={'current_limit_raw': 80})
        reader.alignment_reset_plan = Mock(return_value=[reader.raw[:7].copy() for _ in range(15)])
        reader.command_alignment_positions = Mock()
    pipeline.reset()
    assert pipeline.state == 'aligning'
    for reader in pipeline.readers:
        reader.enable_alignment_hold.assert_called_once_with(80)
        assert reader.command_alignment_positions.call_count == 15
    assert not pipeline.alignment_hold_active
    pipeline.confirm_alignment(); pipeline.takeover()
    assert not pipeline.alignment_hold_active
    pipeline.close()


def test_interactive_resets_without_enter_and_uses_live_gate():
    from unittest.mock import Mock, patch
    pipeline = DualGelloPipeline.__new__(DualGelloPipeline)
    pipeline.reset = Mock()
    pipeline.wait_for_alignment = Mock()
    pipeline.takeover = Mock()
    pipeline.poll = Mock()
    keys = Mock(); keys.is_tty = True; keys.read_key.return_value = 'q'
    manager = Mock(); manager.__enter__ = Mock(return_value=keys); manager.__exit__ = Mock(return_value=False)
    with patch('gello_teleop.dual_gello_collect.TerminalKeys', return_value=manager), \
         patch('gello_teleop.dual_gello_collect._wait_enter', side_effect=AssertionError('unexpected reset prompt')):
        assert pipeline.interactive() == 0
    pipeline.reset.assert_called_once()
    pipeline.wait_for_alignment.assert_called_once_with(keys)
    pipeline.takeover.assert_called_once()


def test_alignment_reset_plan_has_fifteen_linear_goals_and_keeps_nearest_turn():
    import threading
    from types import SimpleNamespace
    reader = GelloReader.__new__(GelloReader)
    reader.config = SimpleNamespace(joint_ids=(1, 2))
    reader.lock = threading.Lock()
    start = np.array([2*np.pi+.4, .8])
    reader.read = lambda: start.copy()
    reader._read_register = lambda motor, address, size: {11: 5, 52: 0, 48: 4095}[address]
    writes = []
    reader._write = lambda motor, address, value, size=1: writes.append((motor, address, value))
    plan = reader.alignment_reset_plan([.7, .5], 15, 90.)
    assert len(plan) == 15
    np.testing.assert_allclose(plan[-1], [2*np.pi+.7, .5])
    np.testing.assert_allclose(plan[0], start+(plan[-1]-start)/15)
    for goal in plan:
        reader.command_alignment_positions(goal)
    assert len(writes) == 30
    assert all(address == 116 for _, address, _ in writes)  # Goal Current remains unchanged.
    with pytest.raises(ValueError, match='travel'):
        reader.alignment_reset_plan([2.5, .5], 15, 90.)


def test_alignment_damping_runs_before_takeover_without_commanding_xarm():
    from dataclasses import replace
    from unittest.mock import Mock
    pipeline = make_pipeline()
    pipeline.connect(); pipeline.reset()
    runtime = []
    for i, (side, robot, leader) in enumerate(pipeline.configs):
        cfg = replace(leader, damping_enabled=True, damping_mode='current', fps=30,
                      damping_gain=[8]*7, damping_brake_gain=[0]*7,
                      damping_current_limit=[15]*7, weak_hold_enabled=False)
        runtime.append((side, robot, cfg))
        reader = pipeline.readers[i]
        reader.config = cfg
        reader.enable_current_damping = Mock(return_value={'enabled': True})
        reader.compute_damping_current = GelloReader.compute_damping_current
        reader.write_current_damping = Mock()
        reader.disable_current_damping = Mock()
    pipeline.configs = runtime
    pipeline.alignment_hold_active = True
    pipeline.start_alignment_damping()
    try:
        deadline = time.monotonic()+1.
        while not all(reader.write_current_damping.called for reader in pipeline.readers):
            if time.monotonic() >= deadline:
                raise AssertionError('alignment damping did not stream')
            time.sleep(.01)
        assert not pipeline.alignment_hold_active
        assert all(not arm.commands for arm in pipeline.arms)
        assert all(producer.thread.is_alive() for producer in pipeline.alignment_damping_producers)
        pipeline.confirm_alignment()
        pipeline.takeover()
        # Current mode remains active; takeover transfers streaming to mapped
        # producers without toggling torque/mode again after the error gate.
        for reader in pipeline.readers:
            reader.enable_current_damping.assert_called_once()
        assert not pipeline.alignment_damping_producers
        assert pipeline.state == 'following'
    finally:
        pipeline.close()


def test_takeover_waits_for_delayed_leader_samples_before_control_starts():
    import threading
    from unittest.mock import Mock
    pipeline = make_pipeline()
    pipeline.connect(); pipeline.reset(); pipeline.confirm_alignment()
    reads = [0, 0]
    for i, reader in enumerate(pipeline.readers):
        original = reader.read
        def delayed_read(index=i, read=original):
            if threading.current_thread().name.startswith('gello-reader-'):
                if reads[index] < 2:
                    assert all(not arm.commands for arm in pipeline.arms)
                reads[index] += 1
                time.sleep(.08 if reads[index] == 1 else .025)
            return read()
        reader.read = delayed_read
    for arm in pipeline.arms:
        def enter_servo():
            assert all(p.snapshot() is not None and p.snapshot().sequence >= 1 for p in pipeline.producers)
        arm.set_normal_mode = Mock(side_effect=enter_servo)
    try:
        pipeline.takeover()
        assert min(reads) >= 2
        assert all(producer.snapshot() is not None for producer in pipeline.producers)
        time.sleep(.04)
        pipeline.poll()
        assert pipeline.control_error is None
        assert pipeline.state == 'following'
        for arm in pipeline.arms:
            arm.set_normal_mode.assert_called_once()
    finally:
        pipeline.close()


def test_leader_read_failure_stops_before_following_and_cleans_up_threads():
    import threading
    from unittest.mock import Mock
    pipeline = make_pipeline()
    pipeline.leader_startup_timeout_s = .1
    pipeline.connect(); pipeline.reset(); pipeline.confirm_alignment()
    for reader in pipeline.readers:
        original = reader.read
        def missing_read(read=original):
            if threading.current_thread().name.startswith('gello-reader-'):
                raise RuntimeError('simulated unplug')
            return read()
        reader.read = missing_read
    for arm in pipeline.arms:
        arm.set_normal_mode = Mock()
    try:
        with pytest.raises(RuntimeError, match='启动采样失败'):
            pipeline.takeover()
        assert pipeline.control_thread is None
        assert pipeline.state != 'following'
        assert all(not arm.commands for arm in pipeline.arms)
        assert all(not arm.set_normal_mode.called for arm in pipeline.arms)
    finally:
        pipeline.close()
    assert all(not producer.thread.is_alive() for producer in pipeline.producers)
    assert all(not arm.commands for arm in pipeline.arms)


def test_missing_leader_feedback_times_out_before_following():
    import threading
    from unittest.mock import Mock
    pipeline = make_pipeline()
    gate = threading.Event()
    pipeline.leader_startup_timeout_s = .025
    pipeline.connect(); pipeline.reset(); pipeline.confirm_alignment()
    for reader in pipeline.readers:
        original = reader.read
        def delayed_read(read=original):
            if threading.current_thread().name.startswith('gello-reader-'):
                gate.wait(1.)
            return read()
        reader.read = delayed_read
    for arm in pipeline.arms:
        arm.set_normal_mode = Mock()
    try:
        with pytest.raises(TimeoutError, match='startup samples'):
            pipeline.takeover()
        assert all(producer.errors == 0 for producer in pipeline.producers)
        assert pipeline.control_thread is None
        assert all(not arm.commands and not arm.set_normal_mode.called for arm in pipeline.arms)
    finally:
        gate.set()
        pipeline.close()


@pytest.fixture
def sync_current_reader():
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock
    reader = GelloReader.__new__(GelloReader)
    reader.config = SimpleNamespace(joint_ids=tuple(range(1, 8)))
    reader.lock = threading.Lock()
    reader.closed = False
    values = {}
    def add(motor, data):
        values[motor] = data[0] | data[1] << 8
        return True
    reader.current_writer = Mock()
    reader.current_writer.addParam.side_effect = add
    reader.current_writer.clearParam.side_effect = values.clear
    reader.current_writer.txPacket.return_value = 0
    reader.current_feedback = Mock()
    reader.current_feedback.txRxPacket.return_value = 0
    reader.current_feedback.isAvailable.return_value = True
    reader.current_feedback.getData.side_effect = lambda motor, *_: values[motor]
    return reader


def test_damping_sync_write_encodes_signed_current_and_verifies_all_motors(sync_current_reader):
    from unittest.mock import call
    reader = sync_current_reader
    currents = [15, -15, 0, 1, -1, 80, -80]
    reader.write_current_damping(currents)
    assert reader.current_writer.addParam.call_args_list == [
        call(motor, [current & 0xff, (current & 0xffff) >> 8])
        for motor, current in enumerate(currents, 1)]
    reader.current_writer.txPacket.assert_called_once_with()
    reader.current_feedback.txRxPacket.assert_called_once_with()
    assert reader.current_feedback.getData.call_args_list == [call(motor, 102, 2) for motor in range(1, 8)]
    assert reader.current_writer.clearParam.call_count == 2


@pytest.mark.parametrize('failure', ['add', 'write', 'read', 'missing', 'mismatch', 'device'])
def test_damping_sync_write_rejects_failed_or_unverified_currents(sync_current_reader, failure):
    reader = sync_current_reader
    if failure == 'add':
        reader.current_writer.addParam.return_value = False
        reader.current_writer.addParam.side_effect = None
    elif failure == 'write':
        reader.current_writer.txPacket.return_value = -1001
    elif failure == 'read':
        reader.current_feedback.txRxPacket.return_value = -1001
    elif failure == 'missing':
        reader.current_feedback.isAvailable.side_effect = lambda motor, *_: motor != 4
    elif failure == 'device':
        reader.current_feedback.txRxPacket.side_effect = RuntimeError('Dynamixel 4 device error failed (code=7)')
    else:
        reader.current_feedback.getData.side_effect = None
        reader.current_feedback.getData.return_value = 0
    with pytest.raises(RuntimeError):
        reader.write_current_damping([15]*7)
    assert reader.current_writer.clearParam.call_count == 2


def test_freeze_keeps_gello_damping_alive_and_stops_arm_and_gripper_targets(tmp_path):
    from dataclasses import replace
    from unittest.mock import Mock
    from scripts.smoke_dual_gello_pipeline import SimulatedArm, write_simulation_config
    path = write_simulation_config(CONFIG, tmp_path)
    class TrackedArm(SimulatedArm):
        def __init__(self, *args):
            super().__init__(*args)
            self.joint_commands = 0
            self.gripper_commands = 0
        def command_joint_positions(self, q):
            super().command_joint_positions(q)
            self.joint_commands += 1
        def command_gripper(self, value, **kwargs):
            super().command_gripper(value, **kwargs)
            self.gripper_commands += 1
    pipeline = DualGelloPipeline(path, arm_factory=TrackedArm, reader_factory=FakeReader)
    try:
        pipeline.connect(); pipeline.reset(); pipeline.confirm_alignment()
        for i, (side, robot, leader) in enumerate(pipeline.configs):
            leader = replace(leader, damping_enabled=True, damping_mode='current', fps=60,
                             damping_gain=[8]*7, damping_brake_gain=[0]*7,
                             damping_current_limit=[15]*7, weak_hold_enabled=False)
            pipeline.configs[i] = side, robot, leader
            reader = pipeline.readers[i]
            reader.config = leader
            reader.enable_current_damping = Mock(return_value={'enabled': True})
            reader.compute_damping_current = GelloReader.compute_damping_current
            reader.write_current_damping = Mock()
            reader.disable_current_damping = Mock()
            reader.set_torque = Mock()
        pipeline.takeover()
        time.sleep(.04)
        pipeline.freeze_following()
        assert pipeline.state == 'holding' and not pipeline.stop_event.is_set()
        counts = [(arm.joint_commands, arm.gripper_commands) for arm in pipeline.arms]
        io_counts = [reader.write_current_damping.call_count for reader in pipeline.readers]
        poses = [arm.q.copy() for arm in pipeline.arms]
        for reader in pipeline.readers:
            reader.raw[:7] += .05
        time.sleep(.08)
        pipeline.poll()
        assert counts == [(arm.joint_commands, arm.gripper_commands) for arm in pipeline.arms]
        for arm, pose in zip(pipeline.arms, poses):
            np.testing.assert_allclose(arm.q, pose)
        for i, reader in enumerate(pipeline.readers):
            reader.enable_current_damping.assert_called_once()
            reader.disable_current_damping.assert_not_called()
            reader.set_torque.assert_not_called()
            assert reader.write_current_damping.call_count > io_counts[i]
            assert pipeline.producers[i].thread.is_alive()
    finally:
        pipeline.close()
    assert all(not producer.thread.is_alive() for producer in pipeline.producers)


def test_realign_uses_held_xarm_pose_and_keeps_recording_continuous(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from nero_collection.cameras import CameraFrame
    from nero_collection.time_utils import now_us
    pipeline = make_pipeline()
    pipeline.alignment.update(gello_hold_current_raw=80, gello_reset_steps=3, gello_reset_interval_s=.001)
    # Configure leader position I/O only AFTER the ordinary synthetic startup.
    pipeline.alignment.pop('gello_hold_current_raw')
    try:
        pipeline.connect(); pipeline.reset(); pipeline.confirm_alignment(); pipeline.takeover()
        pipeline.start_episode()
        time.sleep(.03); pipeline.poll()
        buffer = pipeline.buffer
        pipeline.freeze_following()
        assert pipeline.recording
        samples = buffer.sample_count
        hold_start = buffer.episode_metadata['teleop_events'][0]['timestamp_us']
        held = []
        for i, (reader, arm, (_, robot, leader)) in enumerate(zip(pipeline.readers, pipeline.arms, pipeline.configs)):
            # More than the original passive-start tolerance, to expose an
            # accidental return to reset_q or recalculation of saved offsets.
            arm.q = np.asarray(robot.reset_q)+.4*(i+1)
            held.append(arm.q.copy())
            arm.move_to_reset = Mock(side_effect=AssertionError('t must preserve the held xArm pose'))
            reader.raw[:7] += .9
            reader.enable_alignment_hold = Mock(return_value={'mode': 'current_based_position'})
            def plan(reference, steps, max_travel, read=reader):
                start = read.raw[:7].copy()
                target = reference+np.rint((start-reference)/(2*np.pi))*(2*np.pi)
                return [start+(target-start)*n/steps for n in range(1, steps+1)]
            reader.alignment_reset_plan = Mock(side_effect=plan)
            def command(q, read=reader):
                read.raw[:7] = q
            reader.command_alignment_positions = Mock(side_effect=command)
        time.sleep(.03); pipeline.poll()
        assert buffer.sample_count > samples
        pipeline.alignment['gello_hold_current_raw'] = 80
        pipeline.realign_and_takeover(SimpleNamespace(read_key=lambda timeout: None))
        assert pipeline.buffer is buffer and pipeline.recording
        for reader, arm, q, mapper, (_, _, leader) in zip(pipeline.readers, pipeline.arms, held, pipeline.mappers, pipeline.configs):
            expected_raw = np.asarray(leader.joint_offsets)+q/np.asarray(leader.joint_signs)
            np.testing.assert_allclose(reader.alignment_reset_plan.call_args.args[0], expected_raw)
            reader.enable_alignment_hold.assert_called_once_with(80)
            assert reader.command_alignment_positions.call_count == 3
            np.testing.assert_allclose(arm.q, q, atol=.001)
            np.testing.assert_allclose(mapper.offsets, leader.joint_offsets)
        takeover_time = buffer.episode_metadata['teleop_events'][-1]['timestamp_us']
        assert takeover_time > hold_start
        # Holding and alignment frames remain part of the same recording.
        buffer.append_camera = Mock()
        frame = np.zeros((8, 8, 3), np.uint8)
        pipeline.camera_manager.poll = Mock(return_value=[CameraFrame('hold', hold_start+1, frame),
                                                         CameraFrame('new', now_us(), frame)])
        pipeline._poll_cameras()
        assert buffer.append_camera.call_count == 2
        pipeline.camera_manager.poll.return_value = []
        time.sleep(.03); pipeline.poll()
        assert buffer.sample_count > samples
        timestamps = np.asarray(buffer.teleop_timestamps_us)
        assert np.any((timestamps > hold_start) & (timestamps < takeover_time))
        assert np.max(np.diff(timestamps)) < 60000
        pipeline.stop_episode(save=False)
        for key in ('q_cmd_sequence', 'q_leader_sequence', 'q_follower_sequence'):
            sequence = np.asarray(buffer.teleop_data[key])
            assert np.all(np.diff(sequence, axis=0) >= 0), key
        buffer.save(tmp_path/'resumed.h5')
    finally:
        pipeline.close()


def test_pipeline_creates_one_visualizer_for_selected_cameras(tmp_path):
    import yaml
    from scripts.smoke_dual_gello_pipeline import write_simulation_config
    path = write_simulation_config(CONFIG, tmp_path)
    data = yaml.safe_load(path.read_text())
    data['active_arms'] = ['left', 'right']
    data.setdefault('cameras', [dict(name=f'{owner} wrist', backend='mock', width=64,
                                    height=48, fps=30) for owner in ('left', 'right')])
    data['gripper']['mode'] = 'follow'
    for camera in data['cameras']:
        camera['visualize'] = True
    path.write_text(yaml.safe_dump(data))
    pipeline = DualGelloPipeline(path, arm_factory=FakeArm, reader_factory=FakeReader)
    try:
        assert pipeline.camera_manager.visualizer.camera_names == frozenset(c['name'] for c in data['cameras'])
        assert pipeline.camera_manager.visualizer._process is None
    finally:
        pipeline.close()


def test_camera_grid_labels_both_serials_and_keeps_rgb_dataset_unchanged():
    from nero_collection.cameras import _compose_camera_preview, _import_cv2
    names = ('left wrist', 'right wrist')
    frames = {names[0]: np.full((80, 160, 3), [255, 0, 0], np.uint8),
              names[1]: np.full((80, 160, 3), [0, 0, 255], np.uint8)}
    grid = _compose_camera_preview(names, frames, _import_cv2())
    assert grid.shape == (80, 320, 3)
    np.testing.assert_array_equal(grid[-1, 159], [0, 0, 255])
    np.testing.assert_array_equal(grid[-1, 319], [255, 0, 0])
    np.testing.assert_array_equal(frames[names[0]][0, 0], [255, 0, 0])


def test_interactive_f_and_t_use_freeze_and_automatic_realign_without_enter():
    from unittest.mock import Mock, patch
    pipeline = DualGelloPipeline.__new__(DualGelloPipeline)
    pipeline.state = 'following'
    pipeline.recording = False
    for name in ('reset', 'wait_for_alignment', 'takeover', 'poll'):
        setattr(pipeline, name, Mock())
    pipeline.freeze_following = Mock(side_effect=lambda: setattr(pipeline, 'state', 'holding'))
    pipeline.realign_and_takeover = Mock(side_effect=lambda keys: setattr(pipeline, 'state', 'following'))
    keys = Mock(is_tty=True); keys.read_key.side_effect = ['F', 't', 'q']
    context = Mock(); context.__enter__ = Mock(return_value=keys); context.__exit__ = Mock(return_value=False)
    with patch('gello_teleop.dual_gello_collect.TerminalKeys', return_value=context), \
         patch('gello_teleop.dual_gello_collect._wait_enter', side_effect=AssertionError('unexpected Enter')):
        assert pipeline.interactive() == 0
    pipeline.freeze_following.assert_called_once()
    pipeline.realign_and_takeover.assert_called_once_with(keys)


def test_current_pose_alignment_keeps_saved_offsets_across_encoder_turns():
    pipeline = make_pipeline()
    try:
        pipeline.connect()
        for mapper, reader, (_, robot, leader) in zip(pipeline.mappers, pipeline.readers, pipeline.configs):
            original = np.asarray(leader.joint_offsets).copy()
            target = np.asarray(robot.reset_q)+.4
            turns = np.array([1, -1, 0, 2, -2, 1, 0])
            raw = reader.read()
            raw[:7] = original+target/np.asarray(leader.joint_signs)+2*np.pi*turns
            mapper.align(raw, reference_q=target)
            np.testing.assert_allclose(mapper.target(raw)[0], target)
            np.testing.assert_allclose(mapper.offsets, original+2*np.pi*turns)
            np.testing.assert_array_equal(leader.joint_offsets, original)
    finally:
        pipeline.close()


def test_o_resets_xarms_keeps_gello_and_recording_and_waits_for_t():
    from types import SimpleNamespace
    from unittest.mock import Mock
    pipeline = make_pipeline()
    try:
        pipeline.connect(); pipeline.reset(); pipeline.confirm_alignment(); pipeline.takeover()
        pipeline.start_episode()
        time.sleep(.03); pipeline.poll()
        buffer = pipeline.buffer
        before = buffer.sample_count
        for reader in pipeline.readers:
            reader.prepare_alignment = Mock()
            reader.set_torque = Mock()
        pipeline.reset_and_hold(SimpleNamespace(read_key=lambda _: None))
        pipeline.poll()
        assert pipeline.state == 'holding' and pipeline.recording and pipeline.buffer is buffer
        assert pipeline.follow_stop.is_set() and not pipeline.control_thread.is_alive()
        assert pipeline.sampling_thread.is_alive()
        assert buffer.sample_count > before
        for arm, (_, robot, _), reader, producer in zip(pipeline.arms, pipeline.configs, pipeline.readers, pipeline.producers):
            np.testing.assert_allclose(arm.q, robot.reset_q)
            reader.prepare_alignment.assert_not_called()
            reader.set_torque.assert_not_called()
            assert producer.thread.is_alive()
        assert [event['event'] for event in buffer.episode_metadata['teleop_events']][-2:] == ['o_reset_begin', 'o_reset_done']
        pipeline.stop_episode(save=False)
        assert pipeline.state == 'holding' and not pipeline.recording
    finally:
        pipeline.close()


def test_gripper_absolute_opening_is_repeatable_and_releases_back_to_open():
    import threading
    from types import SimpleNamespace
    from gello_teleop.dual_gello_collect import _GripperWorker, _gripper_fraction
    stop = threading.Event()
    commands = []
    state = SimpleNamespace(value=.085, timestamp_us=0)
    arm = SimpleNamespace(name='fake', read_gripper_state=lambda: state,
                          command_gripper=lambda value, **_: commands.append(value))
    fraction = [0.]
    producer = SimpleNamespace(snapshot=lambda: SimpleNamespace(acquired_monotonic_s=time.monotonic(),
                                                                gripper_fraction=fraction[0]))
    worker = _GripperWorker(arm, producer, stop, period_s=.005, maximum_age_s=.15,
                            command_enabled=True, min_width=0., max_width=.085, max_speed=.0001)
    worker.start()
    try:
        for value, width in [(0., .085), (.5, .0425), (1., 0.), (.5, .0425), (0., .085)]:
            fraction[0] = value
            previous = len(commands)
            deadline = time.monotonic()+1.
            while len(commands) < previous+3:
                if time.monotonic() > deadline:
                    raise AssertionError(f'gripper worker stalled: {worker.error}')
                time.sleep(.002)
            np.testing.assert_allclose(commands[-2:], width)
        opening, closing = np.deg2rad([197.49, 155.49])
        for ratio in [0., .5, 1.]:
            assert _gripper_fraction(opening+(closing-opening)*ratio+2*np.pi, opening, closing) == pytest.approx(ratio)
    finally:
        stop.set(); worker.thread.join(timeout=1.)


@pytest.mark.parametrize('stop_key', [' ', '\r', '\n'])
def test_interactive_recording_r_starts_and_enter_or_space_stops(stop_key, capsys):
    from unittest.mock import Mock, patch
    pipeline = DualGelloPipeline.__new__(DualGelloPipeline)
    pipeline.state = 'following'
    pipeline.recording = False
    for name in ('reset', 'wait_for_alignment', 'takeover', 'poll'):
        setattr(pipeline, name, Mock())
    def start():
        pipeline.recording = True
        pipeline.state = 'recording'
    def stop(save, index, *, background=False):
        assert save is True and pipeline.recording
        assert background
        pipeline.recording = False
        pipeline.state = 'following'
        return None
    def hold(*args):
        assert pipeline.recording
        pipeline.state = 'holding'
    def takeover(*args):
        assert pipeline.recording
        pipeline.state = 'recording'
    pipeline.start_episode = Mock(side_effect=start)
    pipeline.stop_episode = Mock(side_effect=stop)
    pipeline.freeze_following = Mock(side_effect=hold)
    pipeline.reset_and_hold = Mock(side_effect=hold)
    pipeline.realign_and_takeover = Mock(side_effect=takeover)
    keys = Mock(is_tty=True); keys.read_key.side_effect = ['r', 'F', 'o', 't', stop_key, 'q']
    context = Mock(); context.__enter__ = Mock(return_value=keys); context.__exit__ = Mock(return_value=False)
    with patch('gello_teleop.dual_gello_collect.TerminalKeys', return_value=context):
        assert pipeline.interactive() == 0
    pipeline.start_episode.assert_called_once()
    pipeline.stop_episode.assert_called_once_with(True, None, background=True)
    pipeline.freeze_following.assert_called_once()
    pipeline.reset_and_hold.assert_called_once_with(keys)
    pipeline.realign_and_takeover.assert_called_once_with(keys)
    assert '停止录制，正在保存 episode' in capsys.readouterr().out


@pytest.mark.parametrize('stop_key', [' ', '\r', '\n'])
def test_stop_key_prints_when_no_episode_is_recording(stop_key, capsys):
    from unittest.mock import Mock
    pipeline = DualGelloPipeline.__new__(DualGelloPipeline)
    pipeline.recording = False
    pipeline.stop_episode = Mock()
    assert pipeline._recording_key(stop_key)
    pipeline.stop_episode.assert_not_called()
    assert '停止录制：当前没有正在录制的 episode' in capsys.readouterr().out


def test_episode_save_logs_length_sample_count_and_absolute_path(tmp_path, monkeypatch, caplog):
    import logging
    from dataclasses import replace
    from pathlib import Path
    from nero_collection.h5_writer import EpisodeBuffer
    pipeline = make_pipeline()
    pipeline.collection_config = replace(pipeline.collection_config,
        output=replace(pipeline.collection_config.output, directory=tmp_path, prefix='recording'))
    pipeline.buffer = EpisodeBuffer(pipeline.collection_config, pipeline.ARM_NAMES, enable_online_tau_ext=False)
    pipeline.buffer.teleop_timestamps_us.extend([1000000, 1010000])
    pipeline.recording = True
    pipeline.state = 'recording'
    pipeline.recording_started_t = time.monotonic()-1.25
    monkeypatch.setattr(pipeline, '_consume_samples', lambda: None)
    def save(path):
        assert '正在保存至' in caplog.text  # Visible before file writing starts.
        return path
    monkeypatch.setattr(pipeline.buffer, 'save', save)
    caplog.set_level(logging.INFO, logger='dual_gello_collect')
    assert pipeline.stop_episode(True, None)
    assert not pipeline.recording and pipeline.state == 'following'
    duration = pipeline.buffer.episode_metadata['recorded_duration_s']
    assert duration >= 1.25
    assert f'episode 保存完成：长度 {duration:.3f} s，2 个样本；保存路径：' in caplog.text
    assert str(Path(tmp_path).resolve()) in caplog.text
    assert '.h5' in caplog.text


@pytest.mark.parametrize('reset_q', [True, False])
def test_reference_hotkey_is_opt_in_and_keeps_recording_active(reset_q):
    from unittest.mock import Mock, patch
    pipeline = DualGelloPipeline.__new__(DualGelloPipeline)
    pipeline.state = 'recording'
    pipeline.recording = True
    for name in ('reset', 'wait_for_alignment', 'takeover', 'poll', 'save_reference_pose',
                 'freeze_following', 'stop_episode'):
        setattr(pipeline, name, Mock())
    keys = Mock(is_tty=True); keys.read_key.side_effect = ['s', 'S', 'q']
    context = Mock(); context.__enter__ = Mock(return_value=keys); context.__exit__ = Mock(return_value=False)
    with patch('gello_teleop.dual_gello_collect.TerminalKeys', return_value=context):
        assert pipeline.interactive(reset_q=reset_q) == 0
    assert pipeline.save_reference_pose.call_count == (2 if reset_q else 0)
    assert pipeline.state == 'recording' and pipeline.recording
    pipeline.freeze_following.assert_not_called()
    pipeline.stop_episode.assert_not_called()


def test_recording_keys_are_processed_inside_alignment_wait():
    from types import SimpleNamespace
    from unittest.mock import Mock
    pipeline = make_pipeline()
    try:
        pipeline.connect(); pipeline.reset(); pipeline.confirm_alignment(); pipeline.takeover()
        pipeline.start_episode()
        time.sleep(.03); pipeline.poll()
        pipeline.freeze_following()
        first_buffer = pipeline.buffer
        first_buffer.save = Mock(return_value='episode.h5')
        pipeline.alignment['alignment_samples'] = 3
        keys = SimpleNamespace(read_key=Mock(side_effect=[' ', 'r', None]))
        pipeline.wait_for_alignment(keys)
        for future, target in pipeline.pending_episode_saves:
            future.result(timeout=2.)
        first_buffer.save.assert_called_once()
        assert first_buffer.episode_metadata['recorded_sample_count'] > 0
        assert pipeline.recording and pipeline.buffer is not first_buffer
        assert pipeline.state == 'aligned' and pipeline.follow_stop.is_set()
        assert pipeline.sampling_thread.is_alive()
    finally:
        pipeline.close()


def test_headless_opencv_selects_tk_camera_preview():
    from unittest.mock import Mock, patch
    from nero_collection.cameras import _create_camera_preview_window
    cv2 = Mock()
    cv2.getBuildInformation.return_value = '  GUI:                           NONE\n'
    with patch('nero_collection.cameras._TkCameraPreviewWindow') as tk_window, \
         patch('nero_collection.cameras._OpenCVCameraPreviewWindow') as cv_window:
        assert _create_camera_preview_window(cv2) is tk_window.return_value
        tk_window.assert_called_once_with(cv2)
        cv_window.assert_not_called()
        cv2.getBuildInformation.return_value = '  GUI:                           QT5\n'
        assert _create_camera_preview_window(cv2) is cv_window.return_value


def test_camera_preview_stop_does_not_read_a_dead_workers_frame_pipe():
    import threading
    from unittest.mock import Mock
    from nero_collection.cameras import CameraVisualizer
    visualizer = CameraVisualizer(('left', 'right'))
    visualizer._process = Mock()
    visualizer._process.is_alive.return_value = False
    frames = Mock()
    frames.get_nowait.side_effect = AssertionError('a truncated image pipe can block even get_nowait')
    visualizer._queue = frames
    event = threading.Event()
    visualizer._stop_event = event
    visualizer.stop()
    assert event.is_set() and visualizer._process is None
    frames.get_nowait.assert_not_called()
    frames.put_nowait.assert_not_called()
    frames.close.assert_called_once()
