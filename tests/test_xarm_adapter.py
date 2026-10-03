import sys
import types

import numpy as np
import pytest

from nero_collection.config import ArmEndpointConfig


class FakeArm:
    calls = []
    command_code = 0

    def __init__(self, ip, **kwargs):
        self.connected = True
        self.calls.append(("init", ip, kwargs))

    def get_servo_angle(self, **kwargs):
        return 0, [0.0] * 6

    def get_position_aa(self, **kwargs):
        return 0, [100.0, 200.0, 300.0, 0.0, 0.0, 0.0]

    def get_joint_states(self, **kwargs):
        return 0, [[0.0] * 6, [0.0] * 6, [0.0] * 6]

    def get_err_warn_code(self):
        return 0, [0, 0]

    def motion_enable(self, *args, **kwargs):
        self.calls.append(("motion_enable", args, kwargs))
        return 0

    def set_mode(self, *args, **kwargs):
        self.calls.append(("set_mode", args, kwargs))
        return 0

    def set_state(self, *args, **kwargs):
        self.calls.append(("set_state", args, kwargs))
        return 0

    def set_servo_angle_j(self, *args, **kwargs):
        self.calls.append(("set_servo_angle_j", args, kwargs))
        return self.command_code

    def disconnect(self):
        self.calls.append(("disconnect",))


@pytest.fixture
def adapter(monkeypatch):
    FakeArm.calls = []
    FakeArm.command_code = 0
    package = types.ModuleType("xarm")
    wrapper = types.ModuleType("xarm.wrapper")
    wrapper.XArmAPI = FakeArm
    monkeypatch.setitem(sys.modules, "xarm", package)
    monkeypatch.setitem(sys.modules, "xarm.wrapper", wrapper)
    from ufactory_devices.robot.xarm_adapter import XArmAdapter

    value = XArmAdapter(
        ArmEndpointConfig(
            name="xarm6",
            rest_q=(0.0,) * 6,
            config_kwargs={"robot_ip": "127.0.0.1", "dof": 6, "execution_enabled": False},
        )
    )
    value.connect()
    return value


def test_connect_and_dry_run_have_no_motion_side_effects(adapter):
    state = adapter.read_state()
    assert state.q.shape == (6,)
    assert state.ee_pose[:3, 3].tolist() == [0.1, 0.2, 0.3]
    assert not any(call[0] in {"motion_enable", "set_mode", "set_state"} for call in FakeArm.calls)
    with pytest.raises(RuntimeError, match="execution is disabled"):
        adapter.enable()


def test_stop_motion_requires_execution_and_retains_motor_enable(adapter):
    with pytest.raises(RuntimeError, match="execution is disabled"):
        adapter.stop_motion()
    adapter.config.config_kwargs["execution_enabled"] = True
    adapter.enable()
    FakeArm.calls.clear()
    adapter.stop_motion()
    assert [call[0] for call in FakeArm.calls] == ["set_state"]
    assert FakeArm.calls[0][1] == (4,)
    assert adapter._enabled


@pytest.mark.parametrize('rotation_vector', [
    [0., 0., 0.],
    [1.e-12, -2.e-12, 3.e-12],
    [.4, -.7, 1.2],
    [np.pi / np.sqrt(3)] * 3,
    [0., 0., -np.pi / 2],
])
def test_ee_pose_uses_sdk_rotation_vector_and_one_pose_request(adapter, monkeypatch, rotation_vector):
    from unittest.mock import Mock
    from scipy.spatial.transform import Rotation
    axis_angle = Mock(return_value=(0, [125., -250., 375., *rotation_vector]))
    fallback = Mock(side_effect=AssertionError('successful axis-angle read must not add another RPC'))
    monkeypatch.setattr(adapter._arm, 'get_position_aa', axis_angle)
    monkeypatch.setattr(adapter._arm, 'get_position', fallback, raising=False)

    pose = adapter.read_state().ee_pose

    axis_angle.assert_called_once_with(is_radian=True)
    fallback.assert_not_called()
    np.testing.assert_array_equal(pose[:3, 3], [.125, -.250, .375])
    np.testing.assert_array_equal(pose[3], [0., 0., 0., 1.])
    np.testing.assert_allclose(pose[:3, :3], Rotation.from_rotvec(rotation_vector).as_matrix(),
                               atol=1.e-15, rtol=1.e-14)
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1.e-15)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.)


def test_non_axis_aligned_ee_rotation_is_not_treated_as_roll_pitch_yaw(adapter, monkeypatch):
    from nero_collection.arms.kinematics import pose6_to_matrix
    # A 120-degree rotation around (1,1,1) cycles x -> y -> z -> x.
    vector = np.full(3, 2. * np.pi / (3. * np.sqrt(3)))
    monkeypatch.setattr(adapter._arm, 'get_position_aa', lambda **kwargs: (0, [0., 0., 0., *vector]))

    pose = adapter.read_state().ee_pose

    np.testing.assert_allclose(pose[:3, :3], [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], atol=1.e-15)
    assert not np.allclose(pose[:3, :3], pose6_to_matrix(np.r_[np.zeros(3), vector])[:3, :3])


def test_ee_pose_euler_fallback_keeps_sdk_rpy_and_mm_to_m(adapter, monkeypatch):
    from unittest.mock import Mock
    from scipy.spatial.transform import Rotation
    angles = [.4, -.7, 1.2]
    fallback = Mock(return_value=(0, [125., -250., 375., *angles]))
    monkeypatch.setattr(adapter._arm, 'get_position_aa', None)
    monkeypatch.setattr(adapter._arm, 'get_position', fallback, raising=False)

    pose = adapter.read_state().ee_pose

    fallback.assert_called_once_with(is_radian=True)
    np.testing.assert_array_equal(pose[:3, 3], [.125, -.250, .375])
    np.testing.assert_allclose(pose[:3, :3], Rotation.from_euler('xyz', angles).as_matrix(), atol=1.e-15)
    assert not np.allclose(pose[:3, :3], Rotation.from_rotvec(angles).as_matrix())


@pytest.mark.parametrize('axis_angle', [True, False])
def test_ee_pose_rejects_failed_sdk_report_even_with_cached_finite_pose(adapter, monkeypatch, axis_angle):
    from unittest.mock import Mock
    failed = Mock(return_value=(5, [100., 200., 300., .4, -.7, 1.2]))
    if axis_angle:
        monkeypatch.setattr(adapter._arm, 'get_position_aa', failed)
        fallback = Mock(side_effect=AssertionError('failed pose read must not fetch an unrelated sample'))
        monkeypatch.setattr(adapter._arm, 'get_position', fallback, raising=False)
    else:
        monkeypatch.setattr(adapter._arm, 'get_position_aa', None)
        monkeypatch.setattr(adapter._arm, 'get_position', failed, raising=False)

    with pytest.raises(RuntimeError, match='failed with code 5'):
        adapter.read_state()

    failed.assert_called_once_with(is_radian=True)
    if axis_angle:
        fallback.assert_not_called()


def test_theoretical_only_state_requests_position_velocity_without_effort(adapter):
    from unittest.mock import Mock
    adapter._arm.get_joint_states = Mock(return_value=(0, [[0.0]*6, [0.1]*6]))
    state = adapter.read_state()
    adapter._arm.get_joint_states.assert_called_once_with(is_radian=True, num=2)
    assert state.dq_valid and not state.torque_valid and not state.current_valid
    np.testing.assert_allclose(state.dq, .1)
    assert np.isnan(state.torque).all()


@pytest.mark.parametrize('signal', ['none', 'torque', 'current'])
def test_joint_state_read_uses_one_reply_for_position_velocity_and_effort(adapter, monkeypatch, signal):
    from unittest.mock import Mock
    adapter.config.config_kwargs['feedback_signal'] = signal
    q, dq, effort = np.full(6, .25), np.full(6, .1), np.full(6, 2.)
    joint_states = Mock(return_value=(0, [q.tolist(), dq.tolist(), effort.tolist()]))
    angles = Mock(side_effect=AssertionError('redundant angle transaction'))
    pose = Mock(return_value=(0, [100., 200., 300., 0., 0., 0.]))
    monkeypatch.setattr(adapter._arm, 'get_joint_states', joint_states)
    monkeypatch.setattr(adapter._arm, 'get_servo_angle', angles)
    monkeypatch.setattr(adapter._arm, 'get_position_aa', pose)
    host_stamps = iter([100, 200])
    monkeypatch.setattr('ufactory_devices.robot.xarm_adapter.now_us', lambda: next(host_stamps))

    state = adapter.read_state()

    joint_states.assert_called_once_with(is_radian=True, num=2 if signal == 'none' else 3)
    angles.assert_not_called()
    pose.assert_called_once_with(is_radian=True)
    np.testing.assert_array_equal(state.q, q)
    np.testing.assert_array_equal(state.dq, dq)
    assert state.dq_valid
    assert state.torque_valid == (signal == 'torque')
    assert state.current_valid == (signal == 'current')
    np.testing.assert_array_equal(state.torque if signal == 'torque' else state.current,
                                  effort if signal != 'none' else np.full(6, np.nan))
    assert state.acquired_timestamp_us == state.q_acquired_timestamp_us == 100
    assert state.timestamp_us == state.q_timestamp_us == 200
    assert state.timestamp_source == 'host_receive'
    np.testing.assert_array_equal(state.q_component_timestamp_us, np.full(6, 200))


@pytest.mark.parametrize('invalid_q', [None, [0.] * 5, [np.nan] * 6])
def test_invalid_joint_state_position_falls_back_without_discarding_valid_feedback(adapter, monkeypatch, invalid_q):
    from unittest.mock import Mock
    adapter.config.config_kwargs['feedback_signal'] = 'torque'
    joint_states = Mock(return_value=(0, [invalid_q, [.1] * 6, [2.] * 6]))
    angles = Mock(return_value=(0, [.25] * 6))
    monkeypatch.setattr(adapter._arm, 'get_joint_states', joint_states)
    monkeypatch.setattr(adapter._arm, 'get_servo_angle', angles)
    host_stamps = iter([100, 200, 300])
    monkeypatch.setattr('ufactory_devices.robot.xarm_adapter.now_us', lambda: next(host_stamps))

    state = adapter.read_state()

    joint_states.assert_called_once_with(is_radian=True, num=3)
    angles.assert_called_once_with(is_radian=True)
    np.testing.assert_allclose(state.q, .25)
    np.testing.assert_allclose(state.dq, .1)
    np.testing.assert_allclose(state.torque, 2.)
    assert state.dq_valid and state.torque_valid and not state.current_valid
    assert state.acquired_timestamp_us == 100
    assert state.q_timestamp_us == 300


@pytest.mark.parametrize('reply', [None, (5, [[.9] * 6, [.2] * 6, [3.] * 6])])
def test_unavailable_or_failed_joint_state_reply_uses_angle_fallback_with_invalid_feedback(adapter, monkeypatch, reply):
    from unittest.mock import Mock
    adapter.config.config_kwargs['feedback_signal'] = 'torque'
    joint_states = None if reply is None else Mock(return_value=reply)
    angles = Mock(return_value=(0, [.25] * 6))
    monkeypatch.setattr(adapter._arm, 'get_joint_states', joint_states)
    monkeypatch.setattr(adapter._arm, 'get_servo_angle', angles)

    state = adapter.read_state()

    angles.assert_called_once_with(is_radian=True)
    np.testing.assert_allclose(state.q, .25)
    assert not state.dq_valid and not state.torque_valid and not state.current_valid
    assert np.isnan(state.torque).all() and np.isnan(state.current).all()


def test_position_difference_velocity_reuses_the_selected_position_sample(adapter, monkeypatch):
    from unittest.mock import Mock
    adapter.config.config_kwargs['velocity_source'] = 'position_difference'
    adapter._last_q, adapter._last_t = np.zeros(6), 10.
    joint_states = Mock(return_value=(0, [[.4] * 6]))
    angles = Mock(side_effect=AssertionError('velocity must use the same position sample'))
    monkeypatch.setattr(adapter._arm, 'get_joint_states', joint_states)
    monkeypatch.setattr(adapter._arm, 'get_servo_angle', angles)
    monkeypatch.setattr('ufactory_devices.robot.xarm_adapter.time.monotonic', lambda: 12.)

    state = adapter.read_state()

    joint_states.assert_called_once_with(is_radian=True, num=2)
    angles.assert_not_called()
    np.testing.assert_allclose(state.q, .4)
    np.testing.assert_allclose(state.dq, .2)
    assert state.dq_valid


def test_state_read_still_rejects_invalid_position_when_both_sources_fail(adapter, monkeypatch):
    monkeypatch.setattr(adapter._arm, 'get_joint_states', lambda **kwargs: (5, []))
    monkeypatch.setattr(adapter._arm, 'get_servo_angle', lambda **kwargs: (0, [np.nan] * 6))
    with pytest.raises(RuntimeError, match='invalid joint-angle report'):
        adapter.read_state()


def test_tcp_payload_reads_report_cache_and_converts_mm_to_m(adapter):
    before = list(FakeArm.calls)
    adapter._arm.tcp_load = [.79, [22., -3., 40.]]
    payload = adapter.reported_tcp_payload
    assert payload['mass_kg'] == .79
    np.testing.assert_allclose(payload['com_m'], [.022, -.003, .04])
    assert FakeArm.calls == before
    for invalid in [None, [0., [0., 0., 0.]], [.79, [np.nan, 0., 0.]], [.79, [1., 2.]]]:
        adapter._arm.tcp_load = invalid
        assert adapter.reported_tcp_payload is None


def test_command_updates_held_value_only_after_success(adapter):
    adapter.config.config_kwargs["execution_enabled"] = True
    adapter.enable()
    target = np.ones(6)
    FakeArm.command_code = 2
    with pytest.raises(RuntimeError, match="failed with code 2"):
        adapter.command_joint_positions(target)
    assert adapter.last_commanded_q is None
    FakeArm.command_code = 0
    adapter.command_joint_positions(target)
    np.testing.assert_allclose(adapter.last_commanded_q, target)
    call = next(call for call in reversed(FakeArm.calls) if call[0] == "set_servo_angle_j")
    assert "angles" in call[2]
    assert "angle" not in call[2]


def test_joint_state_effort_is_current_only_when_selected(monkeypatch):
    class CurrentArm(FakeArm):
        def set_report_tau_or_i(self, selector):
            self.calls.append(("set_report_tau_or_i", selector))
            return 0

        def get_joint_states(self, **kwargs):
            return 0, [[0.0] * 6, [0.1] * 6, [2.0] * 6]

    package = types.ModuleType("xarm")
    wrapper = types.ModuleType("xarm.wrapper")
    wrapper.XArmAPI = CurrentArm
    monkeypatch.setitem(sys.modules, "xarm", package)
    monkeypatch.setitem(sys.modules, "xarm.wrapper", wrapper)
    from ufactory_devices.robot.xarm_adapter import XArmAdapter

    value = XArmAdapter(ArmEndpointConfig(
        name="current",
        rest_q=(0.0,) * 6,
        config_kwargs={"robot_ip": "127.0.0.1", "dof": 6, "execution_enabled": True,
                       "feedback_signal": "current"},
    ))
    value.connect()
    value.enable()
    state = value.read_state()
    assert state.current_valid and not state.torque_valid
    np.testing.assert_allclose(state.current, 2.0)


def test_gripper_service_starts_without_enabling_hardware(monkeypatch):
    class FakePublisher:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def publish(self, value):
            del value

        def close(self):
            pass

    class FakeGripperArm:
        calls = []

        def __init__(self, ip):
            FakeGripperArm.calls.append(("init", ip))

        def get_gripper_position(self):
            return 0, 0

        def disconnect(self):
            FakeGripperArm.calls.append(("disconnect",))

    package = types.ModuleType("xarm")
    wrapper = types.ModuleType("xarm.wrapper")
    wrapper.XArmAPI = FakeGripperArm
    monkeypatch.setitem(sys.modules, "xarm", package)
    monkeypatch.setitem(sys.modules, "xarm.wrapper", wrapper)

    import xarm_stack.gripper_server as gripper_server

    monkeypatch.setattr(gripper_server, "StatePublisher", FakePublisher)
    service = gripper_server.GripperService(
        {"backend": "xarm", "robot_ip": "127.0.0.1", "execution_enabled": False},
        "inproc://test-gripper",
        rate_hz=1000,
    )
    service.start()
    service.stop()
    assert not any(call[0] in {"motion_enable", "set_gripper_enable", "set_gripper_mode"}
                   for call in FakeGripperArm.calls)


def test_reset_uses_planned_mode_and_fifteen_degree_speed(adapter, monkeypatch):
    from unittest.mock import Mock
    adapter.config.config_kwargs['execution_enabled'] = True
    adapter.enable()
    move = Mock(return_value=0)
    monkeypatch.setattr(adapter._arm, 'set_servo_angle', move, raising=False)
    begin = len(FakeArm.calls)
    adapter.move_to_reset(np.zeros(6), speed=np.deg2rad(15), acceleration=.3)
    modes = [call[1][0] for call in FakeArm.calls[begin:] if call[0] == 'set_mode']
    assert modes == [0]
    assert move.call_args.kwargs['speed'] == pytest.approx(np.deg2rad(15))
    assert move.call_args.kwargs['is_radian'] and not move.call_args.kwargs['wait']
    adapter.set_normal_mode()
    assert [call[1][0] for call in FakeArm.calls[begin:] if call[0] == 'set_mode'] == [0, 1]


def test_exit_hold_cancels_old_target_and_keeps_motor_enable_after_disconnect(adapter, monkeypatch):
    from unittest.mock import Mock
    adapter.config.config_kwargs['execution_enabled'] = True
    adapter.enable()
    adapter.command_joint_positions(np.ones(6))
    move = Mock(return_value=0)
    monkeypatch.setattr(adapter._arm, 'set_servo_angle', move, raising=False)
    begin = len(FakeArm.calls)
    held = adapter.hold_position(speed=.3, acceleration=.4)
    np.testing.assert_array_equal(held, np.zeros(6))
    np.testing.assert_array_equal(adapter.last_commanded_q, held)
    assert adapter._enabled
    assert [(call[0], call[1]) for call in FakeArm.calls[begin:]] == [
        ('set_state', (4,)), ('set_mode', (0,)), ('set_state', (0,))]
    assert move.call_args.kwargs == dict(angle=held.tolist(), speed=.3, mvacc=.4,
                                         is_radian=True, wait=False)
    adapter.disconnect()
    assert FakeArm.calls[-1] == ('disconnect',)
    assert not any(call[0] == 'motion_enable' and call[1] == (False,) for call in FakeArm.calls)


def test_exit_hold_retains_stop_and_enable_when_controller_reports_fault(adapter, monkeypatch):
    from unittest.mock import Mock
    adapter.config.config_kwargs['execution_enabled'] = True
    adapter.enable()
    move = Mock(return_value=0)
    monkeypatch.setattr(adapter._arm, 'set_servo_angle', move, raising=False)
    monkeypatch.setattr(adapter._arm, 'get_err_warn_code', lambda: (0, [11, 0]))
    begin = len(FakeArm.calls)
    with pytest.raises(RuntimeError, match='error code 11'):
        adapter.hold_position()
    assert adapter._enabled
    assert FakeArm.calls[begin:] == [('set_state', (4,), {})]
    move.assert_not_called()


def test_exit_hold_does_not_restart_motion_after_feedback_failure(adapter, monkeypatch):
    from unittest.mock import Mock
    adapter.config.config_kwargs['execution_enabled'] = True
    adapter.enable()
    monkeypatch.setattr(adapter._arm, 'get_servo_angle', Mock(side_effect=ConnectionError('lost feedback')))
    begin = len(FakeArm.calls)
    with pytest.raises(ConnectionError, match='lost feedback'):
        adapter.hold_position()
    assert FakeArm.calls[begin:] == [('set_state', (4,), {})]
    assert adapter._enabled


@pytest.mark.parametrize('value', [6, -1, True, False, 2.5, float('nan'), float('inf'), 'invalid'])
def test_invalid_collision_sensitivity_is_rejected_before_connection(value):
    from ufactory_devices.robot.xarm_adapter import XArmAdapter
    begin = len(FakeArm.calls)
    with pytest.raises(ValueError, match='collision_sensitivity'):
        XArmAdapter(ArmEndpointConfig(name='left', config_kwargs={'collision_sensitivity': value}))
    assert FakeArm.calls[begin:] == []


@pytest.mark.parametrize('level', [0, 3, 5])
def test_collision_sensitivity_is_applied_only_when_enabling(adapter, monkeypatch, caplog, level):
    from unittest.mock import Mock
    setter = Mock(return_value=0)
    monkeypatch.setattr(adapter._arm, 'set_collision_sensitivity', setter, raising=False)
    adapter.config.config_kwargs.update(execution_enabled=True, collision_sensitivity=level)
    def apply(level, **kwargs):
        FakeArm.calls.append(('set_collision_sensitivity', (level,), kwargs))
        return 0
    setter.side_effect = apply
    begin = len(FakeArm.calls)
    adapter.enable()
    setter.assert_called_once_with(level, wait=False)
    assert [call[0] for call in FakeArm.calls[begin:]] == [
        'set_collision_sensitivity', 'motion_enable', 'set_mode', 'set_state']
    assert FakeArm.calls[-1][1] == (0,)
    if level == 0:
        assert '控制器碰撞检测已关闭' in caplog.text


def test_null_collision_sensitivity_preserves_controller_setting(adapter, monkeypatch):
    from unittest.mock import Mock
    setter = Mock(return_value=0)
    monkeypatch.setattr(adapter._arm, 'set_collision_sensitivity', setter, raising=False)
    adapter.config.config_kwargs.update(execution_enabled=True, collision_sensitivity=None)
    adapter.enable()
    setter.assert_not_called()


def test_failed_collision_setting_prevents_motor_enable(adapter, monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr(adapter._arm, 'set_collision_sensitivity', Mock(return_value=1), raising=False)
    adapter.config.config_kwargs.update(execution_enabled=True, collision_sensitivity=3)
    begin = len(FakeArm.calls)
    with pytest.raises(RuntimeError, match='set_collision_sensitivity failed with code 1'):
        adapter.enable()
    assert not adapter._enabled
    assert FakeArm.calls[begin:] == []


def test_existing_controller_fault_prevents_collision_changes_and_enable(adapter, monkeypatch):
    from unittest.mock import Mock
    from ufactory_devices.robot.xarm_adapter import XArmControllerFault
    setter = Mock(return_value=0)
    monkeypatch.setattr(adapter._arm, 'set_collision_sensitivity', setter, raising=False)
    monkeypatch.setattr(adapter._arm, 'get_err_warn_code', lambda: (0, [31, 0]))
    adapter.config.config_kwargs.update(execution_enabled=True, collision_sensitivity=3)
    begin = len(FakeArm.calls)
    with pytest.raises(XArmControllerFault, match='C31'):
        adapter.enable()
    setter.assert_not_called()
    assert FakeArm.calls[begin:] == []
    assert not adapter._enabled


def test_contact_fault_reports_controller_cause_without_clearing_or_reenabling(adapter, monkeypatch):
    from ufactory_devices.robot.xarm_adapter import XArmControllerFault
    from unittest.mock import Mock
    adapter.config.config_kwargs['execution_enabled'] = True
    adapter.enable()
    previous_target = adapter.last_commanded_q
    getter = Mock(return_value=(0, [31, 0]))
    monkeypatch.setattr(adapter._arm, 'get_err_warn_code', getter)
    FakeArm.command_code = 1
    begin = len(FakeArm.calls)
    with pytest.raises(XArmControllerFault) as raised:
        adapter.command_joint_positions(np.full(6, .1))
    assert (raised.value.arm_name, raised.value.error_code, raised.value.api_code) == ('xarm6', 31, 1)
    assert raised.value.operation == 'set_servo_angle_j'
    getter.assert_called_once_with()
    assert [call[0] for call in FakeArm.calls[begin:]] == ['set_servo_angle_j']
    assert adapter.last_commanded_q is previous_target
    assert adapter._enabled


def test_successful_servo_does_not_add_error_query_to_control_loop(adapter, monkeypatch):
    from unittest.mock import Mock
    adapter.config.config_kwargs['execution_enabled'] = True
    adapter.enable()
    getter = Mock(return_value=(0, [0, 0]))
    monkeypatch.setattr(adapter._arm, 'get_err_warn_code', getter)
    adapter.command_joint_positions(np.full(6, .1))
    getter.assert_not_called()


def test_failed_fault_diagnostic_preserves_original_command_failure(adapter, monkeypatch):
    adapter.config.config_kwargs['execution_enabled'] = True
    adapter.enable()
    FakeArm.command_code = 1
    monkeypatch.setattr(adapter._arm, 'get_err_warn_code', lambda: (5, [31, 0]))
    with pytest.raises(RuntimeError, match='xarm6: xArm set_servo_angle_j failed with code 1'):
        adapter.command_joint_positions(np.full(6, .1))


def test_readonly_connection_cannot_request_exit_hold(adapter):
    begin = len(FakeArm.calls)
    with pytest.raises(RuntimeError, match='execution is disabled'):
        adapter.hold_position()
    assert FakeArm.calls[begin:] == []


def test_control_status_does_not_accept_failed_sdk_reads(adapter, monkeypatch):
    monkeypatch.setattr(adapter._arm, 'get_state', lambda: (5, 2), raising=False)
    with pytest.raises(RuntimeError, match='get_state failed with code 5'):
        adapter.control_status()


def test_mode_switch_waits_for_async_sdk_report_before_servo_command(adapter, monkeypatch):
    pending = [None]
    adapter._arm.mode = 0
    def set_mode(mode):
        pending[0] = mode
        return 0
    def deliver_report(_seconds):
        adapter._arm.mode = pending[0]
    monkeypatch.setattr(adapter._arm, 'set_mode', set_mode)
    monkeypatch.setattr('ufactory_devices.robot.xarm_adapter.time.sleep', deliver_report)
    adapter.config.config_kwargs['execution_enabled'] = True
    adapter.enable()
    assert adapter._enabled and adapter._arm.mode == 1
    adapter._arm.mode = 0
    adapter.set_normal_mode()
    assert adapter._arm.mode == 1
    adapter.command_joint_positions(np.zeros(6))


def test_mode_report_timeout_does_not_issue_motion_or_hide_enabled_state(adapter, monkeypatch):
    adapter.config.config_kwargs.update(execution_enabled=True, mode_switch_timeout_s=.01)
    adapter._arm.mode = 0
    with pytest.raises(TimeoutError, match='mode switch not reported'):
        adapter.enable()
    assert adapter._enabled  # cleanup must know that motion_enable succeeded
    assert not any(call[0] == 'set_servo_angle_j' for call in FakeArm.calls)
    adapter.disable()
    assert not adapter._enabled


def test_gripper_speed_is_applied_once_and_absolute_width_commands_remain_nonblocking(adapter, monkeypatch):
    from unittest.mock import Mock
    adapter.config.config_kwargs.update(execution_enabled=True, gripper_speed=5000)
    for name in ('set_gripper_enable', 'set_gripper_mode', 'set_gripper_speed', 'set_gripper_position'):
        monkeypatch.setattr(adapter._arm, name, Mock(return_value=0), raising=False)
    adapter.init_gripper()
    adapter.command_gripper(.085, force_n=0.)
    adapter.command_gripper(.0425, force_n=0.)
    adapter.command_gripper(0., force_n=0.)
    adapter._arm.set_gripper_speed.assert_called_once_with(5000)
    assert [(call.args[0], call.kwargs) for call in adapter._arm.set_gripper_position.call_args_list] == [
        (800, {'wait': False, 'wait_motion': False}),
        (400, {'wait': False, 'wait_motion': False}),
        (0, {'wait': False, 'wait_motion': False})]


def test_streaming_gripper_skips_arm_wait_and_preserves_sdk_error_and_baud_checks(adapter, monkeypatch):
    from unittest.mock import Mock
    adapter.config.config_kwargs['execution_enabled'] = True
    for name in ('set_gripper_enable', 'set_gripper_mode'):
        monkeypatch.setattr(adapter._arm, name, Mock(return_value=0), raising=False)
    def set_position(pos, *, wait=False, wait_motion=True, check_baud=True, check_err=True):
        assert not wait and not wait_motion
        assert check_baud and check_err
        return 19
    monkeypatch.setattr(adapter._arm, 'set_gripper_position', set_position, raising=False)
    with pytest.raises(RuntimeError, match='set_gripper_position failed with code 19'):
        adapter.command_gripper(.085, force_n=0.)


@pytest.mark.parametrize('speed', [None, -1])
def test_gripper_controller_speed_can_be_left_unchanged(adapter, monkeypatch, speed):
    from unittest.mock import Mock
    adapter.config.config_kwargs.update(execution_enabled=True, gripper_speed=speed)
    for name in ('set_gripper_enable', 'set_gripper_mode', 'set_gripper_speed'):
        monkeypatch.setattr(adapter._arm, name, Mock(return_value=0), raising=False)
    adapter.init_gripper()
    adapter._arm.set_gripper_speed.assert_not_called()


@pytest.mark.parametrize('speed', [0, -2, np.nan, np.inf, 1.5, True, 'invalid'])
def test_invalid_gripper_speed_cannot_enable_hardware(adapter, monkeypatch, speed):
    from unittest.mock import Mock
    adapter.config.config_kwargs.update(execution_enabled=True, gripper_speed=speed)
    enable = Mock(return_value=0)
    monkeypatch.setattr(adapter._arm, 'set_gripper_enable', enable, raising=False)
    with pytest.raises(ValueError, match='gripper_speed'):
        adapter.init_gripper()
    enable.assert_not_called()


def test_gripper_sdk_failure_is_reported_instead_of_accepting_bad_feedback(adapter, monkeypatch):
    from unittest.mock import Mock
    adapter.config.config_kwargs.update(execution_enabled=True, gripper_speed=5000)
    for name in ('set_gripper_enable', 'set_gripper_mode'):
        monkeypatch.setattr(adapter._arm, name, Mock(return_value=0), raising=False)
    monkeypatch.setattr(adapter._arm, 'set_gripper_speed', Mock(return_value=19), raising=False)
    with pytest.raises(RuntimeError, match='set_gripper_speed failed with code 19'):
        adapter.init_gripper()
    monkeypatch.setattr(adapter._arm, 'get_gripper_position', Mock(return_value=(19, 800)), raising=False)
    with pytest.raises(RuntimeError, match='get_gripper_position failed with code 19'):
        adapter.read_gripper_state()


def test_read_only_endpoint_cannot_apply_gripper_speed(adapter, monkeypatch):
    from unittest.mock import Mock
    adapter.config.config_kwargs['gripper_speed'] = 5000
    speed = Mock(return_value=0)
    monkeypatch.setattr(adapter._arm, 'set_gripper_speed', speed, raising=False)
    with pytest.raises(RuntimeError, match='execution is disabled'):
        adapter.init_gripper()
    speed.assert_not_called()
