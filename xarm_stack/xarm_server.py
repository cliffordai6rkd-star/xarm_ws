from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nero_collection.arms.base import ArmState
from nero_collection.config import ArmEndpointConfig
from ufactory_devices.robot.xarm_adapter import XArmAdapter
from xarm_stack.protocol import RpcServer, StatePublisher

log = logging.getLogger(__name__)


class XArmService:
    def __init__(self, endpoint: ArmEndpointConfig, state_endpoint: str, rate_hz: float = 100.0) -> None:
        self.arm = XArmAdapter(endpoint)
        self.publisher = StatePublisher(state_endpoint)
        self.period_s = 1.0 / float(rate_hz)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.state_thread: threading.Thread | None = None

    def start(self) -> None:
        with self.lock:
            self.arm.connect()
        self.state_thread = threading.Thread(target=self._publish_loop, name="xarm-state", daemon=True)
        self.state_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.state_thread is not None:
            self.state_thread.join(timeout=2.0)
        with self.lock:
            self.arm.disconnect()
        self.publisher.close()

    def methods(self) -> dict[str, Any]:
        return {
            "ping": lambda: {"service": "xarm", "feedback_available": self.arm.feedback_available, "capabilities": self.arm.capabilities, "timing": self.arm.timing_stats},
            "state": self.state,
            "enable": self.enable,
            "disable": self.disable,
            "set_follower": self.set_follower,
            "set_normal": self.set_normal,
            "move_joints": self.move_joints,
            "move_interpolated": self.move_interpolated,
            "command_joint_impedance": self.command_joint_impedance,
            "init_gripper": self.init_gripper,
            "command_gripper": self.command_gripper,
            "gripper_state": self.gripper_state,
            "stop": self.stop_command,
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            return _state_dict(self.arm.read_state(), self.arm.feedback_available)

    def enable(self) -> None:
        with self.lock:
            self.arm.enable()

    def disable(self) -> None:
        with self.lock:
            self.arm.disable()

    def set_follower(self) -> None:
        with self.lock:
            self.arm.set_follower_mode()

    def set_normal(self) -> None:
        with self.lock:
            self.arm.set_normal_mode()

    def move_joints(self, q: list[float]) -> None:
        with self.lock:
            self.arm.command_joint_positions(np.asarray(q, dtype=np.float64))

    def move_interpolated(self, q: list[float], rate_hz: float = 50.0, max_step_rad: float = 0.02) -> None:
        target = np.asarray(q, dtype=np.float64).reshape(self.arm.dof)
        with self.lock:
            current = self.arm.read_state().q
            distance = float(np.max(np.abs(target - current)))
            steps = max(1, int(np.ceil(distance / max(float(max_step_rad), 1.0e-5))))
            period = 1.0 / max(float(rate_hz), 1.0)
            for index in range(1, steps + 1):
                alpha = index / steps
                self.arm.command_joint_positions(current + alpha * (target - current))
                time.sleep(period)

    def command_joint_impedance(self, q: list[float], v_des: list[float], kp: list[float], kd: list[float], t_ff: list[float]) -> None:
        with self.lock:
            self.arm.command_joint_impedance(q, v_des, kp, kd, t_ff)

    def init_gripper(self, effector: str = "xArmGripper") -> None:
        with self.lock:
            self.arm.init_gripper(effector)

    def command_gripper(self, value: float, force_n: float = 0.0, mode: str = "width") -> None:
        with self.lock:
            self.arm.command_gripper(value, force_n, mode)

    def gripper_state(self) -> dict[str, Any]:
        with self.lock:
            state = self.arm.read_gripper_state()
        return {"value": state.value, "force": state.force, "timestamp_us": state.timestamp_us, "mode": state.mode}

    def stop_command(self) -> None:
        with self.lock:
            state = self.arm.read_state()
            self.arm.command_joint_positions(state.q)

    def _publish_loop(self) -> None:
        next_t = time.monotonic()
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    state = _state_dict(self.arm.read_state(), self.arm.feedback_available)
                self.publisher.publish(state)
            except Exception:
                log.exception("xArm state publish failed")
            next_t += self.period_s
            time.sleep(max(0.0, next_t - time.monotonic()))


def _state_dict(state: ArmState, feedback_available: bool | None = None) -> dict[str, Any]:
    return {
        "timestamp_us": int(state.timestamp_us),
        "acquired_timestamp_us": int(state.acquired_timestamp_us),
        "q_timestamp_us": int(state.q_timestamp_us),
        "q_acquired_timestamp_us": int(state.q_acquired_timestamp_us),
        "q_component_timestamp_us": np.asarray(state.q_component_timestamp_us, dtype=np.int64),
        "q_source_before_timestamp_us": np.asarray(state.q_source_before_timestamp_us, dtype=np.int64),
        "q_source_after_timestamp_us": np.asarray(state.q_source_after_timestamp_us, dtype=np.int64),
        "motor_timestamp_us": np.asarray(state.motor_timestamp_us, dtype=np.int64),
        "motor_acquired_timestamp_us": np.asarray(state.motor_acquired_timestamp_us, dtype=np.int64),
        "q": np.asarray(state.q, dtype=np.float64),
        "dq": np.asarray(state.dq, dtype=np.float64),
        "ddq": np.asarray(state.ddq, dtype=np.float64),
        "ee_pose": np.asarray(state.ee_pose, dtype=np.float64),
        "torque": np.asarray(state.torque, dtype=np.float64),
        "current": np.asarray(state.current, dtype=np.float64),
        # A zero-valued but valid measurement must remain distinguishable from
        # an unavailable report.
        "feedback_available": bool(feedback_available) if feedback_available is not None else False,
        "q_valid": bool(state.q_valid),
        "dq_valid": bool(state.dq_valid),
        "torque_valid": bool(state.torque_valid),
        "current_valid": bool(state.current_valid),
        "feedback_source": str(state.feedback_source),
        "timestamp_source": str(state.timestamp_source),
    }


def _load_endpoint(path: str | Path, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data = yaml.safe_load(Path(path).expanduser().read_text())
    raw = dict(data["xarm"] if name == "xarm" else data[name])
    state = dict(data.get("services", {}))
    return raw, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="xArm hardware service")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rpc", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw, services = _load_endpoint(args.config, "xarm")
    endpoint = ArmEndpointConfig(
        name=str(raw.get("name", "xarm")),
        rest_q=tuple(raw.get("rest_q", ())),
        config_kwargs={k: v for k, v in raw.items() if k not in {"name", "rest_q"}},
    )
    rpc = args.rpc or services.get("xarm_rpc", "tcp://127.0.0.1:5551")
    state = args.state or services.get("xarm_state", "tcp://127.0.0.1:5552")
    service = XArmService(endpoint, state, float(services.get("xarm_state_hz", 100.0)))
    service.start()
    server = RpcServer(rpc, service.methods())
    log.info("xArm service ready rpc=%s state=%s", rpc, state)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
