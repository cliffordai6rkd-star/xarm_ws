"""ArmInterface proxy backed by the xArm state/RPC service."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from nero_collection.arms.base import ArmState, GripperState
from nero_collection.config import ArmEndpointConfig
from xarm_stack.protocol import RpcClient, StateSubscriber


class RemoteXArm:
    name = "xarm"
    dof = 7

    def __init__(self, config: ArmEndpointConfig) -> None:
        self.config = config
        options = config.config_kwargs
        self.name = config.name
        configured_dof = options.get("dof")
        if configured_dof is None and config.rest_q:
            configured_dof = len(config.rest_q)
        self.dof = int(configured_dof or 7)
        if self.dof not in {5, 6, 7}:
            raise ValueError("remote xArm dof must be 5, 6, or 7")
        self.rpc_endpoint = str(options.get("rpc_endpoint", options.get("rpc", "tcp://127.0.0.1:5551")))
        self.state_endpoint = str(options.get("state_endpoint", options.get("state", "tcp://127.0.0.1:5552")))
        self.rpc: RpcClient | None = None
        self.subscriber: StateSubscriber | None = None

    def connect(self) -> None:
        self.rpc = RpcClient(self.rpc_endpoint)
        self.subscriber = StateSubscriber(self.state_endpoint)
        self.rpc.call("ping")
        self.subscriber.wait(float(self.config.config_kwargs.get("state_timeout_s", 5.0)))

    def disconnect(self) -> None:
        if self.subscriber is not None:
            self.subscriber.close()
            self.subscriber = None
        if self.rpc is not None:
            self.rpc.close()
            self.rpc = None

    def enable(self) -> None:
        self._call("enable")

    def disable(self) -> None:
        self._call("disable")

    def set_leader_mode(self) -> None:
        self.set_normal_mode()

    def set_follower_mode(self) -> None:
        self._call("set_follower")

    def set_normal_mode(self) -> None:
        self._call("set_normal")

    def read_control_role(self, refresh: bool = False) -> str | None:
        del refresh
        return "follower"

    def read_state(self) -> ArmState:
        if self.subscriber is None:
            raise RuntimeError("remote xArm is not connected")
        state = self.subscriber.state
        if state is None:
            state = self.subscriber.wait(1.0)
        return _decode_state(state, self.dof)

    def read_leader_joint_positions(self) -> np.ndarray:
        return self.read_state().q.copy()

    def command_joint_positions(self, q: np.ndarray) -> None:
        self._call("move_joints", q=np.asarray(q, dtype=np.float64))

    def command_joint_impedance(self, q, v_des, kp, kd, t_ff) -> None:
        self._call("command_joint_impedance", q=q, v_des=v_des, kp=kp, kd=kd, t_ff=t_ff)

    def validate_joint_impedance_support(self) -> None:
        if not bool(self.config.config_kwargs.get("allow_impedance_position_fallback", False)):
            raise RuntimeError("remote xArm service does not advertise verified impedance control")

    def configure_joint_impedance_mode(self) -> None:
        self.validate_joint_impedance_support()

    def move_joints(self, q: np.ndarray) -> None:
        self.command_joint_positions(q)

    def wait_motion_done(self, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
        del poll_interval_s
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if np.max(np.abs(self.read_state().dq)) < 0.02:
                return True
            time.sleep(0.02)
        return False

    def init_gripper(self, effector: str = "xArmGripper") -> None:
        self._call("init_gripper", effector=effector)

    def reset_gripper(self) -> bool:
        self._call("command_gripper", value=0.0, force_n=0.0, mode="width")
        return True

    def read_gripper_state(self) -> GripperState:
        result = self._call("gripper_state")
        return GripperState(float(result.get("value", np.nan)), float(result.get("force", np.nan)), int(result.get("timestamp_us", time.time_ns() // 1000)), str(result.get("mode", "width")))

    def read_leader_gripper_state(self) -> GripperState:
        return self.read_gripper_state()

    def disable_gripper(self) -> None:
        return None

    def command_gripper(self, value: float, force_n: float, mode: str = "width") -> None:
        self._call("command_gripper", value=float(value), force_n=float(force_n), mode=str(mode))

    def _call(self, method: str, **params: Any) -> Any:
        if self.rpc is None:
            raise RuntimeError("remote xArm is not connected")
        return self.rpc.call(method, **params)


def _vector(state: dict[str, Any], key: str, dof: int, default: float = 0.0, *, allow_nonfinite: bool = False) -> np.ndarray:
    value = np.asarray(state.get(key, [default] * dof), dtype=np.float64).reshape(-1)
    if value.shape != (dof,) or (not allow_nonfinite and not np.isfinite(value).all()):
        raise RuntimeError(f"remote xArm state {key} is not a finite {dof}-vector")
    return value


def _decode_state(state: dict[str, Any], dof: int) -> ArmState:
    ee_pose = np.asarray(state.get("ee_pose", np.eye(4)), dtype=np.float64)
    if ee_pose.shape != (4, 4) or not np.isfinite(ee_pose).all():
        raise RuntimeError("remote xArm state ee_pose is not a finite 4x4 matrix")
    timestamp = int(state.get("timestamp_us", time.time_ns() // 1000))
    acquired = int(state.get("acquired_timestamp_us", timestamp))
    q_timestamp = int(state.get("q_timestamp_us", timestamp))
    q_acquired = int(state.get("q_acquired_timestamp_us", acquired))
    def _stamps(name: str, fallback: int) -> np.ndarray:
        values = np.asarray(state.get(name, [fallback] * dof), dtype=np.int64).reshape(-1)
        if values.shape != (dof,) or np.any(values <= 0):
            raise RuntimeError(f"remote xArm state {name} has invalid timestamps")
        return values
    q_components = _stamps("q_component_timestamp_us", q_timestamp)
    q_before = _stamps("q_source_before_timestamp_us", q_timestamp)
    q_after = _stamps("q_source_after_timestamp_us", q_timestamp)
    motor = _stamps("motor_timestamp_us", timestamp)
    motor_acquired = _stamps("motor_acquired_timestamp_us", acquired)
    return ArmState(
        q=_vector(state, "q", dof),
        dq=_vector(state, "dq", dof),
        ddq=_vector(state, "ddq", dof),
        ee_pose=ee_pose,
        torque=_vector(state, "torque", dof, default=float("nan"), allow_nonfinite=not bool(state.get("torque_valid", state.get("feedback_available", False)))),
        current=_vector(state, "current", dof, default=float("nan"), allow_nonfinite=not bool(state.get("current_valid", False))),
        timestamp_us=timestamp,
        acquired_timestamp_us=acquired,
        q_timestamp_us=q_timestamp,
        q_acquired_timestamp_us=q_acquired,
        q_component_timestamp_us=q_components,
        q_source_before_timestamp_us=q_before,
        q_source_after_timestamp_us=q_after,
        motor_timestamp_us=motor,
        motor_acquired_timestamp_us=motor_acquired,
        q_valid=bool(state.get("q_valid", True)),
        dq_valid=bool(state.get("dq_valid", True)),
        torque_valid=bool(state.get("torque_valid", state.get("feedback_available", False))),
        current_valid=bool(state.get("current_valid", False)),
        feedback_source=str(state.get("feedback_source", "hardware")),
        timestamp_source=str(state.get("timestamp_source", "host_receive")),
    )
