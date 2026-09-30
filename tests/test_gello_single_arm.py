"""Single-arm lifecycle and squeeze-triggered diagnostics, without hardware."""
from copy import deepcopy
import json
import time

import h5py
import pytest
import yaml

from gello_teleop.dual_gello_collect import DualGelloPipeline, _GripperPressDetector
from gello_teleop.inspect_gello_grippers import read_gripper_status
from gello_teleop.reference_pose import capture_current_pose, overwrite_reference_pose
from scripts.smoke_dual_gello_pipeline import SimulatedArm, SimulatedLeader, write_simulation_config

CONFIG = 'gello_teleop/config/xarm7_gello_dual_dataset.yaml'


@pytest.mark.parametrize('active_level', [0, 2.5, 3])
def test_collision_validation_uses_only_active_arm_effective_setting(tmp_path, active_level):
    path = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw['control']['collision_sensitivity'] = 6
    raw['arms']['left']['collision_sensitivity'] = active_level
    raw['arms']['right']['collision_sensitivity'] = 6
    path.write_text(yaml.safe_dump(raw))
    if active_level == 2.5:
        with pytest.raises(ValueError, match='collision_sensitivity'):
            DualGelloPipeline(path, active_arms=['left'])
    else:
        pipeline = DualGelloPipeline(path, active_arms=['left'])
        assert pipeline.collection_config.teleop.master_slave[0].follower.config_kwargs['collision_sensitivity'] == active_level
        pipeline.close()


@pytest.mark.parametrize('side', ['left', 'right'])
def test_arm_override_filters_devices_and_theoretical_plot_but_keeps_all_cameras(tmp_path, side):
    from pathlib import Path
    from unittest.mock import Mock
    from xarm_stack.torque_visualization import DEFAULT_URDF
    path = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw['active_arms'] = ['left', 'right']
    inactive = 'right' if side == 'left' else 'left'
    raw['torque_visualization'] = dict(enabled=True, urdf_path=str(DEFAULT_URDF),
                                     sides={inactive: {'urdf_path': '/missing/inactive.urdf'}})
    raw.setdefault('cameras', [dict(name=f'{owner} wrist', arm=owner, backend='mock',
                                    width=64, height=48) for owner in ('left', 'right')])
    raw['cameras'].append(dict(name='scene', backend='mock', width=64, height=48))
    path.write_text(yaml.safe_dump(raw))
    calibration = Path(raw['gello_config'])
    saved = yaml.safe_load(calibration.read_text())
    saved.pop(inactive)
    calibration.write_text(yaml.safe_dump(saved))
    pipeline = DualGelloPipeline(path, active_arms=[side], arm_factory=SimulatedArm,
                                 reader_factory=SimulatedLeader)
    assert set(pipeline.torque_visualizer.cfg['sides']) == {side}
    expected_cameras = {camera['name'] for camera in raw['cameras'] if camera.get('enabled', True)}
    assert {camera.name for camera in pipeline.collection_config.cameras} == expected_cameras
    assert yaml.safe_load(pipeline.collection_config.raw_yaml)['active_arms'] == [side]
    plot = pipeline.torque_visualizer = Mock()
    try:
        pipeline.connect()
        # Plot is fed before GELLO alignment or takeover, even without effort.
        assert plot.publish.called
        assert plot.publish.call_args.args[0] == (side,)
        assert [arm.name for arm in pipeline.arms] == [side]
        assert len(pipeline.readers) == 1
    finally:
        pipeline.close()
    plot.close.assert_called_once()


@pytest.mark.parametrize('active', [['left'], ['right'], ['left', 'right']])
def test_arm_selection_keeps_camera_rgb_depth_recording_and_preview(tmp_path, monkeypatch, active):
    import numpy as np
    from unittest.mock import Mock
    from nero_collection.cameras import CameraFrame
    from nero_collection.time_utils import now_us

    path = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw['active_arms'] = active
    raw['gripper'] = dict(enabled=False, command_enabled=False)
    raw['cameras'] = [
        dict(name='left wrist', arm='left', backend='mock', width=16, height=12,
             depth=True, visualize=True),
        dict(name='right wrist', arm='right', backend='mock', width=16, height=12,
             depth=True, visualize=True),
        dict(name='scene', backend='mock', width=16, height=12, visualize=False),
        dict(name='disabled', arm='left', backend='mock', enabled=False, visualize=True),
    ]
    path.write_text(yaml.safe_dump(raw))
    pipeline = DualGelloPipeline(path, arm_factory=SimulatedArm, reader_factory=SimulatedLeader)
    preview = pipeline.camera_manager.visualizer
    assert preview.camera_names == frozenset({'left wrist', 'right wrist'})
    monkeypatch.setattr(preview, 'start', Mock())
    monkeypatch.setattr(preview, 'stop', Mock())
    submitted = []
    def submit(frame):
        if frame.camera_name in preview.camera_names:
            submitted.append(frame)
    monkeypatch.setattr(preview, 'submit', submit)
    try:
        pipeline.connect()
        assert [arm.name for arm in pipeline.arms] == active
        assert {camera.name for camera in pipeline.camera_manager.cameras} == {
            'left wrist', 'right wrist', 'scene'}
        pipeline.reset()
        pipeline.confirm_alignment()
        pipeline.takeover()
        pipeline.start_episode()
        submitted.clear()
        expected = {}
        for index, camera in enumerate(pipeline.camera_manager.cameras, 1):
            rgb = np.full((12, 16, 3), index, dtype=np.uint8)
            depth = np.full((12, 16), index*100, dtype=np.uint16) if camera.config.depth else None
            frame = CameraFrame(camera.name, now_us(), rgb, depth=depth)
            expected[camera.name] = frame
            # Inject one distinct acquired frame per camera through the actual
            # manager, preview dispatch and episode recording paths.
            frames = iter((frame,))
            monkeypatch.setattr(camera, 'poll', Mock(side_effect=lambda source=frames: next(source, None)))
        pipeline._poll_cameras()
        assert {frame.camera_name for frame in submitted} == {'left wrist', 'right wrist'}
        pipeline._consume_samples()
        pipeline.stop_episode(save=False)
        output = tmp_path/'all_cameras.h5'
        pipeline.buffer.save(output)
        with h5py.File(output) as episode:
            assert set(episode['cameras']) == set(expected)
            for name, frame in expected.items():
                group = episode[f'cameras/{name}']
                np.testing.assert_array_equal(group['frames'][0], frame.frame)
                assert group['timestamp_us'][0] == frame.timestamp_us
                if frame.depth is not None:
                    np.testing.assert_array_equal(group['depth'][0], frame.depth)
                else:
                    assert 'depth' not in group
    finally:
        pipeline.close()


@pytest.mark.parametrize('flag,expected', [('--left', ['left']), ('--right', ['right']),
                                         ('--both', ['left', 'right'])])
def test_cli_arm_selection_is_read_only_for_check_config(tmp_path, monkeypatch, flag, expected):
    import gello_teleop.dual_gello_collect as collect
    path = write_simulation_config(CONFIG, tmp_path)
    constructed = []
    actual = collect.DualGelloPipeline
    def factory(config, **kwargs):
        constructed.append(kwargs)
        return actual(config, **kwargs)
    monkeypatch.setattr(collect, 'DualGelloPipeline', factory)
    assert collect.main(['-c', str(path), flag, '--check-config']) == 0
    assert constructed == [{'active_arms': expected}]


@pytest.mark.parametrize('gripper_mode', ['follow', 'trigger'])
@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('keep_inactive', [False, True])
def test_single_arm_lifecycle_recording_and_reference_save(tmp_path, side, keep_inactive, gripper_mode):
    path = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(active_arms=[side], cameras=[], require_cameras=False)
    raw['gripper']['mode'] = gripper_mode
    inactive = 'right' if side == 'left' else 'left'
    raw['arms'].pop(inactive)
    calibration = tmp_path/'synthetic_calibration.yaml'
    saved = yaml.safe_load(calibration.read_text())
    before = deepcopy(saved[inactive])
    if not keep_inactive:
        saved.pop(inactive)
    calibration.write_text(yaml.safe_dump(saved))
    path.write_text(yaml.safe_dump(raw))
    connected = []

    def arm_factory(name, robot):
        connected.append(name)
        assert name == side
        return SimulatedArm(name, robot)

    class Leader(SimulatedLeader):
        fully_pressed = False

        def read(self):
            values = super().read()
            if gripper_mode == 'trigger':
                import numpy as np
                opening = np.deg2rad(self.config.gripper_open_deg or 0.)
                closing = (opening+np.deg2rad(self.config.gripper_travel_deg)
                           if self.config.gripper_close_deg is None else np.deg2rad(self.config.gripper_close_deg))
                values[7] = closing if self.fully_pressed else opening
            return values

    pipeline = DualGelloPipeline(path, arm_factory=arm_factory, reader_factory=Leader)
    try:
        pipeline.connect()
        pipeline.reset()
        pipeline.confirm_alignment()
        pipeline.takeover()
        if gripper_mode == 'trigger':
            deadline = time.monotonic()+1.
            while pipeline.gripper_workers[0].press_detector.pressed:
                assert time.monotonic() < deadline
                time.sleep(.005)
            pipeline.readers[0].fully_pressed = True
        assert connected == [side]
        assert pipeline.ARM_NAMES == (side,)
        pipeline.start_episode()
        deadline = time.monotonic()+.3
        while time.monotonic() < deadline:
            pipeline.poll()
            time.sleep(.005)
        status = read_gripper_status(path)
        assert tuple(status) == (side,)
        with pytest.raises(RuntimeError, match='未启用'):
            read_gripper_status(path, (inactive,))
        pipeline.freeze_following()
        pose = capture_current_pose(pipeline.reference_pose_snapshot())
        overwrite_reference_pose(pose)
        updated = yaml.safe_load(calibration.read_text())
        if keep_inactive:
            assert updated[inactive] == before
        else:
            assert inactive not in updated
        pipeline.realign_and_takeover(None)
        pipeline.poll()
        pipeline.stop_episode(save=False)
        output = tmp_path/'single.h5'
        pipeline.buffer.save(output)
        with h5py.File(output) as episode:
            assert episode['teleop/q_follower'].shape[1] == 7
            assert episode['teleop/q_cmd'].shape[1] == 7
            assert episode['teleop/gripper_cmd'].shape[1] == 1
            assert episode['teleop/gripper_cmd_valid'][:].any()
            metadata = json.loads(episode['metadata/episode_json'][()])
            assert metadata['arm_names'] == [side]
            assert metadata['gripper_mode'] == gripper_mode
            assert len(metadata['leader_calibration']) == 1
    finally:
        pipeline.close()


@pytest.mark.parametrize('active', [[], ['left', 'left'], ['bad'], 'left', None])
def test_invalid_active_arm_selection(tmp_path, active):
    path = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw['active_arms'] = active
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='active_arms'):
        DualGelloPipeline(path)


def test_press_detector_debounces_and_ignores_direction_and_full_turns():
    detector = _GripperPressDetector()
    assert not detector.update(197.8)
    assert not detector.update(198.1)
    assert detector.update(202.)
    assert not detector.update(220.)
    assert not detector.update(220.)
    assert not detector.update(198.)  # release
    assert detector.update(192.)     # opposite encoder direction also works
    assert not detector.update(197.8+360.)
    assert detector.update(205.+360.)
    assert not detector.update(None)


@pytest.mark.parametrize('mode', ['follow', 'trigger'])
def test_press_prints_once_and_still_works_while_held(tmp_path, capsys, mode):
    path = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(cameras=[], require_cameras=False, active_arms=['left'])
    raw['gripper']['mode'] = mode
    path.write_text(yaml.safe_dump(raw))
    pipeline = DualGelloPipeline(path)
    grip = dict(gello_angle_deg=197.8, closure_fraction=0., desired_width_m=.085,
                command_width_m=.085, actual_width_m=.0817, open_deg=197.5,
                close_deg=155.5, following=False)
    pipeline.reference_pose_snapshot = lambda: dict(valid=True, arms={
        'left': dict(gripper=grip, identity={'gripper_id': 8})})
    pipeline._poll_gripper_press_status()
    grip['gello_angle_deg'] = 205.
    grip['closure_fraction'] = 1.
    pipeline._poll_gripper_press_status()
    pipeline._poll_gripper_press_status()
    out = capsys.readouterr().out
    assert out.count('left ID8') == 1
    assert '原始=205.0°' in out and '保持/未下发' in out
    grip['gello_angle_deg'] = 197.8
    grip['closure_fraction'] = 0.
    pipeline._poll_gripper_press_status()
    grip['gello_angle_deg'] = 210.
    grip['closure_fraction'] = 1.
    pipeline._poll_gripper_press_status()
    assert capsys.readouterr().out.count('left ID8') == 1


def test_trigger_toggles_once_per_press_and_discards_paused_presses():
    import threading
    from types import SimpleNamespace
    import numpy as np
    from gello_teleop.dual_gello_collect import _GripperWorker
    from nero_collection.arms.base import GripperState

    fraction = [0.]
    commands = []
    stop, pause = threading.Event(), threading.Event()
    arm = SimpleNamespace(name='left',
        read_gripper_state=lambda: GripperState(.0817, np.nan, time.time_ns()//1000, 'width'),
        command_gripper=lambda width, **_: commands.append(width))
    producer = SimpleNamespace(snapshot=lambda: SimpleNamespace(
        acquired_monotonic_s=time.monotonic(), gripper_fraction=fraction[0]))
    worker = _GripperWorker(arm, producer, stop, period_s=.005, maximum_age_s=.15,
        command_enabled=True, min_width=0., max_width=.085, max_speed=.02,
        mode='trigger', pause=pause)

    def wait_for(condition):
        deadline = time.monotonic()+1.
        while not condition():
            assert worker.error is None
            assert time.monotonic() < deadline
            time.sleep(.002)

    worker.start()
    try:
        wait_for(lambda: not worker.press_detector.pressed)
        assert commands == []  # startup preserves the physical gripper position
        fraction[0] = .5  # a partial squeeze must not command anything
        time.sleep(.03)
        assert commands == []
        fraction[0] = 1.
        wait_for(lambda: len(commands) == 1)
        assert commands == [0.]
        time.sleep(.03)
        assert commands == [0.]
        fraction[0] = 0.
        wait_for(lambda: not worker.press_detector.pressed)
        time.sleep(.03)
        assert commands == [0.]  # release keeps the follower closed
        fraction[0] = .8
        time.sleep(.03)
        assert commands == [0.]
        fraction[0] = 1.
        wait_for(lambda: len(commands) == 2)
        assert commands == [0., .085]
        pause.set()
        with worker.lock:
            pass
        fraction[0] = 0.
        wait_for(lambda: not worker.press_detector.pressed)
        fraction[0] = 1.
        wait_for(lambda: worker.press_detector.pressed)
        pause.clear()
        time.sleep(.03)
        assert commands == [0., .085]  # no delayed toggle when resuming
        fraction[0] = 0.
        wait_for(lambda: not worker.press_detector.pressed)
        fraction[0] = 1.
        wait_for(lambda: len(commands) == 3)
        assert commands == [0., .085, 0.]
    finally:
        stop.set()
        worker.thread.join(timeout=1.)
    assert worker.error is None


def test_invalid_gripper_mode(tmp_path):
    path = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw['gripper']['mode'] = 'invalid'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='gripper.mode'):
        DualGelloPipeline(path)


def test_trigger_requires_full_squeeze_and_releases_without_command():
    from gello_teleop.dual_gello_collect import _GripperTriggerDetector
    detector = _GripperTriggerDetector()
    assert not detector.update(1.)  # startup while squeezed
    assert not detector.update(0.)
    assert not detector.update(.5)
    assert not detector.update(.94)
    assert detector.update(.95)
    assert not detector.update(1.)
    assert not detector.update(.5)
    assert not detector.update(1.)  # insufficient release must not rearm
    assert not detector.update(.2)
    assert detector.update(1.)
