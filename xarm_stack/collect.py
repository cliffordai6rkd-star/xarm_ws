"""Coordinate the three hardware services and collect joint/camera episodes."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nero_collection.cameras import CameraManager
from nero_collection.config import load_config
from nero_collection.episode_output import episode_path, next_episode_index
from nero_collection.fixed_rate import FixedRateTicker
from nero_collection.h5_writer import EpisodeBuffer
from nero_collection.keyboard import TerminalKeys
from xarm_stack.force_feedback import TorqueFeedback
from xarm_stack.protocol import RpcClient, StateSubscriber

log = logging.getLogger(__name__)


class ThreeServiceCollector:
    def __init__(self, raw: dict[str, Any], config_path: str | Path) -> None:
        self.raw = raw
        self.config_path = Path(config_path).expanduser().resolve()
        services = dict(raw.get("services", {}))
        self.rpc = {
            name: RpcClient(str(services[name]))
            for name in ("xarm_rpc", "gello_rpc", "gripper_rpc")
        }
        self.states = {
            name: StateSubscriber(str(services[name]))
            for name in ("xarm_state", "gello_state", "gripper_state")
        }
        self.config = load_config(self.config_path)
        self.xarm_cfg = dict(raw.get("xarm", {}))
        self.gello_cfg = dict(raw.get("gello", {}))
        self.gripper_cfg = dict(raw.get("gripper", {}))
        self.rest_xarm = _vector(self.xarm_cfg.get("rest_q"), "xarm.rest_q")
        self.rest_gello = _vector(self.gello_cfg.get("rest_q"), "gello.rest_q")
        self.dof = int(self.rest_xarm.size)
        if self.rest_gello.size != self.dof:
            raise ValueError("xArm and Gello joint vectors must have the same dimension")
        self.joint_signs = _vector(self.gello_cfg.get("joint_signs", [1.0] * self.dof), "gello.joint_signs")
        self.position_scale = _vector(self.gello_cfg.get("position_scale", [1.0] * self.dof), "gello.position_scale")
        self.reset_error = float(raw.get("startup", {}).get("reset_error_rad", 0.05))
        self.reset_samples = int(raw.get("startup", {}).get("reset_samples", 5))
        self.reset_timeout_s = float(raw.get("startup", {}).get("reset_timeout_s", 20.0))
        self.loop_rate_hz = float(raw.get("collection", {}).get("sample_rate_hz", 100.0))
        self.max_step = float(raw.get("collection", {}).get("max_step_rad", 0.08))
        feedback_raw = dict(raw.get("force_feedback", {}))
        feedback_raw.setdefault("dof", self.dof)
        self.feedback = TorqueFeedback.from_config(feedback_raw)
        feedback_cfg = dict(raw.get("force_feedback", {}))
        self.feedback_source = str(feedback_cfg.get("source", "torque")).lower()
        if self.feedback.enabled and self.feedback_source not in {"torque", "current"}:
            raise ValueError("enabled force feedback requires source=torque or source=current")
        if self.feedback.enabled and self.xarm_cfg.get("feedback_signal", "none") != self.feedback_source:
            raise ValueError("xarm.feedback_signal must match force_feedback.source")
        if self.feedback.enabled and self.feedback_source == "current" and "current_to_torque_nm_per_a" not in feedback_cfg:
            raise ValueError("current force feedback requires calibrated current_to_torque_nm_per_a")
        self.current_to_torque = _vector(
            feedback_cfg.get("current_to_torque_nm_per_a", [0.0] * self.dof),
            "force_feedback.current_to_torque_nm_per_a",
        )
        self.last_command = self.rest_xarm.copy()
        self._last_loop_t = time.monotonic()

    def close(self) -> None:
        for subscriber in self.states.values():
            subscriber.close()
        for client in self.rpc.values():
            client.close()

    def wait_ready(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        for name, client in self.rpc.items():
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                try:
                    response = client.call("ping")
                    log.info("%s ready: %s", name, response)
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(0.1)
            else:
                raise RuntimeError(f"{name} service is not ready: {last_error}")
        for name, subscriber in self.states.items():
            subscriber.wait(max(0.1, deadline - time.monotonic()))

    def reset_and_check(self) -> None:
        # The collector has no read-only dry-run: its reset sequence moves
        # both devices.  Reject the example's disabled endpoints before any
        # RPC that could enable or move hardware.
        if not bool(self.xarm_cfg.get("execution_enabled", False)):
            raise RuntimeError("xarm.execution_enabled must be true before collector reset")
        if self.gripper_cfg.get("backend", "xarm") == "xarm" and not bool(self.gripper_cfg.get("execution_enabled", False)):
            raise RuntimeError("gripper.execution_enabled must be true before collector reset")
        self.rpc["xarm_rpc"].call("enable")
        self.rpc["gello_rpc"].call("enable")
        if self.feedback.enabled or bool(self.raw.get("startup", {}).get("require_feedback", False)):
            xarm = self.rpc["xarm_rpc"].call("state")
            if self.feedback.enabled:
                self._measured_feedback(xarm)
            elif not bool(xarm.get("feedback_available", False)):
                raise RuntimeError("xArm did not report measured joint feedback")
        log.info("resetting xArm and Gello with interpolated trajectories")
        self.rpc["xarm_rpc"].call(
            "move_interpolated",
            q=self.rest_xarm,
            rate_hz=float(self.raw.get("startup", {}).get("xarm_rate_hz", 30.0)),
            max_step_rad=float(self.raw.get("startup", {}).get("xarm_max_step_rad", 0.03)),
        )
        self.rpc["gello_rpc"].call(
            "move_interpolated",
            q=self.rest_gello,
            duration_s=float(self.raw.get("startup", {}).get("gello_duration_s", 4.0)),
            rate_hz=float(self.raw.get("startup", {}).get("gello_rate_hz", 100.0)),
        )
        self.rpc["gripper_rpc"].call("reset")
        deadline = time.monotonic() + self.reset_timeout_s
        consecutive = 0
        while time.monotonic() < deadline:
            xarm = self.states["xarm_state"].wait(1.0)
            gello = self.states["gello_state"].wait(1.0)
            q_xarm = _vector(xarm.get("q"), "xarm state q")
            q_gello = _vector(gello.get("q"), "gello state q")
            mapped = self._map_gello_to_xarm(q_gello)
            error = float(np.max(np.abs(q_xarm - mapped)))
            log.info("reset check %d/%d max_error=%.5f rad", consecutive + 1, self.reset_samples, error)
            if error <= self.reset_error:
                consecutive += 1
                if consecutive >= self.reset_samples:
                    break
            else:
                consecutive = 0
            time.sleep(0.05)
        if consecutive < self.reset_samples:
            raise RuntimeError(
                f"reset alignment failed: required {self.reset_samples} consecutive checks "
                f"below {self.reset_error:.4f} rad"
            )
        self.rpc["xarm_rpc"].call("set_follower")
        xarm = self.rpc["xarm_rpc"].call("state")
        self.feedback.reset()
        self.last_command = _vector(xarm.get("q"), "xarm state q")
        # Establish an accepted position hold before the first H5 row uses
        # last_command as the command effective at its state timestamp.
        self.rpc["xarm_rpc"].call("move_joints", q=self.last_command)

    def run(self, dry_run_duration_s: float | None, auto_save: bool) -> int:
        cameras = CameraManager.from_config(self.config.cameras)
        output_dir = self.config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)
        episode_index = next_episode_index(output_dir, self.config.output.prefix)
        cameras.start()
        recording = False
        teleop_active = False
        buffer: EpisodeBuffer | None = None
        start_t = time.monotonic()
        ticker = FixedRateTicker(self.loop_rate_hz, 1.0 / self.loop_rate_hz * 2.0)
        try:
            with TerminalKeys() as keys:
                if not keys.is_tty and dry_run_duration_s is None:
                    raise RuntimeError("interactive keyboard is unavailable; use --dry-run-duration")
                if dry_run_duration_s is not None:
                    self._enter_teleop()
                    teleop_active = True
                    recording = True
                    buffer = EpisodeBuffer(self.config, ("arm",), enable_online_tau_ext=False)
                    self.feedback.reset()
                    start_t = time.monotonic()
                while True:
                    ticker.wait("three-service collection")
                    now = time.monotonic()
                    key = keys.read_key(0.0)
                    if key in {"q", "Q", "\x03"}:
                        break
                    if key in {"t", "T"}:
                        self._enter_teleop()
                        teleop_active = True
                        recording = False
                        log.info("teleoperation active without recording")
                    elif key in {"r", "R"}:
                        if recording:
                            log.warning("an episode is already recording; press SPACE to stop it first")
                            continue
                        if not teleop_active:
                            self._enter_teleop()
                            teleop_active = True
                        buffer = EpisodeBuffer(self.config, ("arm",), enable_online_tau_ext=False)
                        self.feedback.reset()
                        recording = True
                        start_t = now
                        log.info("recording started")
                    elif key == " ":
                        if recording and buffer is not None:
                            if self._finish_episode(buffer, output_dir, episode_index, auto_save, keys):
                                episode_index += 1
                        recording = False
                        buffer = None
                        self._hold()
                        teleop_active = False
                    if teleop_active:
                        values, timestamp_us = self._step()
                        frames = cameras.poll()
                        if recording and buffer is not None:
                            if now - start_t >= self.config.output.discard_initial_s:
                                buffer.append_teleop(timestamp_us, values)
                                for frame in frames:
                                    buffer.append_camera(frame.camera_name, frame.timestamp_us, frame.frame, frame.depth)
                            else:
                                buffer.append_teleop(timestamp_us, values, store=False)
                    else:
                        cameras.poll()
                    if dry_run_duration_s is not None and now - start_t >= dry_run_duration_s:
                        if recording and buffer is not None:
                            self._finish_episode(buffer, output_dir, episode_index, True, keys)
                        break
        finally:
            cameras.stop()
            if teleop_active:
                self._hold()
        return 0

    def _enter_teleop(self) -> None:
        self.rpc["xarm_rpc"].call("set_follower")
        self.rpc["xarm_rpc"].call("enable")
        self.rpc["gello_rpc"].call("enable")

    def _hold(self) -> None:
        try:
            self.rpc["gello_rpc"].call("set_torque", torque=np.zeros(self.dof))
            self.rpc["gello_rpc"].call("disable")
        finally:
            self.rpc["xarm_rpc"].call("stop")

    def _step(self) -> tuple[dict[str, tuple[str, np.ndarray]], int]:
        xarm = self.states["xarm_state"].wait(1.0)
        gello = self.states["gello_state"].wait(1.0)
        gripper = self.states["gripper_state"].state or {}
        q_xarm = _vector(xarm.get("q"), "xarm state q")
        q_gello = _vector(gello.get("q"), "gello state q")
        dq_xarm = _feedback_vector(xarm, "dq", self.dof)
        dq_gello = _vector(gello.get("dq", np.zeros(self.dof)), "gello state dq")
        torque = _feedback_vector(xarm, "torque", self.dof)
        current = _feedback_vector(xarm, "current", self.dof)
        measured_feedback = self._measured_feedback(xarm) if self.feedback.enabled else None
        held_at_sample = self.last_command.copy()
        target = self._map_gello_to_xarm(q_gello)
        target = self.last_command + np.clip(target - self.last_command, -self.max_step, self.max_step)
        self.rpc["xarm_rpc"].call("move_joints", q=target)
        self.last_command = target.copy()
        now = time.monotonic()
        feedback = (self.feedback.update(measured_feedback, now - self._last_loop_t)
                    if measured_feedback is not None else np.zeros(self.dof))
        self._last_loop_t = now
        # A zero command also clears the temporary reset hold target and puts
        # the leader into free-drag mode when force feedback is disabled.
        self.rpc["gello_rpc"].call("set_torque", torque=feedback)
        timestamp_us = int(xarm.get("timestamp_us", time.time_ns() // 1000))
        values: dict[str, tuple[str, np.ndarray]] = {
            "q_leader": ("q", q_gello),
            "q_follower": ("q", q_xarm),
            "q_cmd": ("q", held_at_sample),
            "delta_q": ("q_error", held_at_sample - q_xarm),
            "dq_leader": ("velocity", dq_gello),
            "dq_follower": ("velocity", dq_xarm),
            "dq_valid_follower": ("validity", np.asarray([bool(xarm.get("dq_valid", False))], dtype=np.uint8)),
            "ee_pose_follower": ("ee_pose", np.asarray(xarm.get("ee_pose", np.eye(4)), dtype=np.float64)),
            "tau_follower": ("torque", torque),
            "current_follower": ("current", current),
            "torque_valid_follower": ("validity", np.asarray([bool(xarm.get("torque_valid", False))], dtype=np.uint8)),
            "current_valid_follower": ("validity", np.asarray([bool(xarm.get("current_valid", False))], dtype=np.uint8)),
            "gripper_follower": ("gripper", np.asarray([float(gripper.get("value", np.nan))], dtype=np.float64)),
            "gripper_cmd": ("gripper", np.asarray([np.nan], dtype=np.float64)),
        }
        return values, timestamp_us

    def _measured_feedback(self, state: dict[str, Any]) -> np.ndarray:
        source = self.feedback_source
        if not bool(state.get(f"{source}_valid", False)):
            raise RuntimeError(f"xArm {source} feedback is unavailable; refusing force feedback")
        sample = _vector(state.get(source), f"xarm state {source}")
        if sample.size != self.dof:
            raise ValueError(f"xArm {source} feedback dimension does not match dof={self.dof}")
        return sample * self.current_to_torque if source == "current" else sample

    def _map_gello_to_xarm(self, q_gello: np.ndarray) -> np.ndarray:
        return self.rest_xarm + self.position_scale * self.joint_signs * (q_gello - self.rest_gello)

    @staticmethod
    def _finish_episode(
        buffer: EpisodeBuffer,
        output_dir: Path,
        index: int,
        auto_save: bool,
        keys: TerminalKeys,
    ) -> bool:
        if not auto_save:
            print("Press y to save the data or n to discard it.", flush=True)
            while True:
                answer = keys.read_key(0.1)
                if answer in {"y", "Y"}:
                    break
                if answer in {"n", "N"}:
                    return False
                if answer in {"q", "Q", "\x03"}:
                    raise KeyboardInterrupt
        path = episode_path(output_dir, buffer.config.output.prefix, index)
        buffer.save(path)
        log.info("saved episode to %s", path)
        return True


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size not in {5, 6, 7} or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite xArm5/6/7 vector")
    return result


def _feedback_vector(state: dict[str, Any], name: str, dof: int) -> np.ndarray:
    value = np.asarray(state.get(name, [np.nan] * dof), dtype=np.float64).reshape(-1)
    if value.shape != (dof,):
        raise ValueError(f"xArm {name} feedback must have {dof} values")
    if bool(state.get(f"{name}_valid", False)):
        if not np.isfinite(value).all():
            raise ValueError(f"xArm {name} feedback marked valid but contains nonfinite values")
        return value
    return np.full(dof, np.nan)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="three-service xArm/Gello data collection")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run-duration", type=float)
    parser.add_argument("--auto-save", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = Path(args.config).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    collector = ThreeServiceCollector(raw, path)
    try:
        collector.wait_ready()
        collector.reset_and_check()
        return collector.run(args.dry_run_duration, args.auto_save or args.dry_run_duration is not None)
    except KeyboardInterrupt:
        return 130
    finally:
        collector.close()


if __name__ == "__main__":
    raise SystemExit(main())
