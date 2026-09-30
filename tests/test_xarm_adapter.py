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
        (800, {'wait': False}), (400, {'wait': False}), (0, {'wait': False})]


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
