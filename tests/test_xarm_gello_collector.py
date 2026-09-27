"""Collector behavior with the SDK services replaced by in-memory endpoints."""

import time

import numpy as np
import pytest

from xarm_stack.collect import ThreeServiceCollector
from xarm_stack.force_feedback import TorqueFeedback


class State:
    def __init__(self, value):
        self.state = value

    def wait(self, timeout):
        del timeout
        return self.state


class Rpc:
    def __init__(self):
        self.calls = []

    def call(self, method, **kwargs):
        self.calls.append((method, kwargs))


def collector(*, feedback=False, valid=False):
    value = ThreeServiceCollector.__new__(ThreeServiceCollector)
    value.dof = 5
    value.states = {
        "xarm_state": State({
            "q": [0.1] * 5, "dq": [0.0] * 5,
            "torque": [1.0] * 5 if valid else [float("nan")] * 5,
            "current": [float("nan")] * 5,
            "dq_valid": True, "torque_valid": valid, "current_valid": False,
            "timestamp_us": 123,
        }),
        "gello_state": State({"q": [0.2] * 5, "dq": [0.0] * 5}),
        "gripper_state": State({}),
    }
    value.rpc = {"xarm_rpc": Rpc(), "gello_rpc": Rpc()}
    value.feedback = TorqueFeedback.from_config({"enabled": feedback, "dof": 5, "gain": 1.0,
                                                 "limit_nm": 2.0, "rate_limit_nm_s": 1e9,
                                                 "ramp_s": 0.0, "lowpass_hz": None})
    value.feedback_source = "torque"
    value.current_to_torque = np.zeros(5)
    value.rest_xarm = np.zeros(5)
    value.rest_gello = np.zeros(5)
    value.position_scale = np.ones(5)
    value.joint_signs = np.ones(5)
    value.last_command = np.zeros(5)
    value.max_step = 0.05
    value._last_loop_t = time.monotonic()
    return value


def test_disabled_feedback_preserves_missing_measurement_and_causal_command():
    value = collector()
    row, stamp = value._step()
    assert stamp == 123
    np.testing.assert_allclose(row["q_cmd"][1], np.zeros(5))
    np.testing.assert_allclose(row["delta_q"][1], [-0.1] * 5)
    np.testing.assert_allclose(value.last_command, [0.05] * 5)
    assert np.isnan(row["tau_follower"][1]).all()
    assert row["torque_valid_follower"][1].item() == 0
    np.testing.assert_allclose(value.rpc["gello_rpc"].calls[-1][1]["torque"], np.zeros(5))


def test_enabled_feedback_rejects_missing_torque_before_xarm_motion():
    value = collector(feedback=True)
    with pytest.raises(RuntimeError, match="torque feedback is unavailable"):
        value._step()
    assert value.rpc["xarm_rpc"].calls == []


def test_valid_torque_feedback_reaches_gello_with_position_only_xarm_command():
    value = collector(feedback=True, valid=True)
    value._step()
    assert [name for name, _ in value.rpc["xarm_rpc"].calls] == ["move_joints"]
    np.testing.assert_allclose(value.rpc["gello_rpc"].calls[-1][1]["torque"], np.ones(5))


def test_reset_rejects_disabled_xarm_before_hardware_rpc():
    value = collector()
    value.xarm_cfg = {"execution_enabled": False}
    value.gripper_cfg = {"backend": "xarm", "execution_enabled": False}
    with pytest.raises(RuntimeError, match="xarm.execution_enabled"):
        value.reset_and_check()
    assert all(not rpc.calls for rpc in value.rpc.values())
