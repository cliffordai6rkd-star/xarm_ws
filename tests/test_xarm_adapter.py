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
