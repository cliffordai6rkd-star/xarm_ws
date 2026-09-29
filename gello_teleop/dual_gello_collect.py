#!/usr/bin/env python3
"""双臂 GELLO 位置遥操和异步 HDF5 数采入口。

The hardware lifecycle is intentionally explicit:

``connect -> reset -> manual alignment -> takeover -> (r) record``.

GELLO readers run in their own producer threads.  A separate monotonic 100 Hz
controller samples both xArms, uses the latest valid GELLO target (ZOH), and
places completed state rows in a bounded queue.  HDF5 and camera work consume
that queue outside the controller loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from gello_teleop.gello_hardware import GelloReader
from gello_teleop.uf_robot_gello_teleop import JointMapper, load_configs
from nero_collection.cameras import CameraManager
from nero_collection.config import (
    ArmEndpointConfig,
    ArmPairConfig,
    CameraConfig,
    CollectionConfig,
    CommandConfig,
    OutputConfig,
    TeleopConfig,
)
from nero_collection.episode_output import episode_path, next_episode_index
from nero_collection.fixed_rate import FixedRateTicker
from nero_collection.h5_writer import EpisodeBuffer
from nero_collection.keyboard import TerminalKeys
from nero_collection.time_utils import now_us
from ufactory_devices.robot.xarm_adapter import XArmAdapter

log = logging.getLogger("dual_gello_collect")


@dataclass(frozen=True)
class LeaderSample:
    raw: np.ndarray
    mapped: np.ndarray
    timestamp_us: int
    acquired_timestamp_us: int
    acquired_monotonic_s: float
    sequence: int
    valid: bool


@dataclass(frozen=True)
class ControlSample:
    timestamp_us: int
    follower_states: tuple[Any, ...]
    leader_samples: tuple[LeaderSample, ...]
    q_cmd: np.ndarray
    q_cmd_timestamp_us: np.ndarray
    q_cmd_ok: np.ndarray
    q_cmd_sequence: np.ndarray
    q_follower_repeated: np.ndarray
    ddq_follower: np.ndarray
    gripper_follower: np.ndarray
    gripper_follower_valid: np.ndarray
    lateness_us: int


class SecondOrderPositionFollower:
    """Bounded independent reference state for xArm position servoing."""

    def __init__(self, dof: int, mode: str = "direct", *, kp=18.0, kd=3.0,
                 max_velocity=1.5, max_acceleration=8.0, max_step=0.08,
                 max_tracking_error=0.25, joint_limits=None):
        self.dof = int(dof)
        self.mode = str(mode).lower()
        if self.mode not in {"direct", "second_order"}:
            raise ValueError("control.position_mode must be direct or second_order")
        self.kp = _joint_vector(kp, self.dof, "kp")
        self.kd = _joint_vector(kd, self.dof, "kd")
        self.max_velocity = _joint_vector(max_velocity, self.dof, "max_velocity")
        self.max_acceleration = _joint_vector(max_acceleration, self.dof, "max_acceleration")
        self.max_step = _joint_vector(max_step, self.dof, "max_step")
        self.max_tracking_error = _joint_vector(max_tracking_error, self.dof, "max_tracking_error")
        self.joint_limits = None if joint_limits is None else np.asarray(joint_limits, dtype=float)
        if self.joint_limits is not None and self.joint_limits.shape != (self.dof, 2):
            raise ValueError("joint_limits must have shape (dof, 2)")
        self.q_ref: np.ndarray | None = None
        self.dq_ref = np.zeros(self.dof)

    def initialize(self, q_actual: np.ndarray) -> None:
        q = np.asarray(q_actual, dtype=float).reshape(self.dof)
        if not np.isfinite(q).all():
            raise ValueError("follower initialization is not finite")
        self.q_ref = q.copy()
        self.dq_ref.fill(0.0)

    def update(self, target: np.ndarray, q_actual: np.ndarray, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=float).reshape(self.dof)
        q_actual = np.asarray(q_actual, dtype=float).reshape(self.dof)
        if not np.isfinite(target).all() or not np.isfinite(q_actual).all():
            raise ValueError("position reference contains non-finite values")
        if self.q_ref is None:
            self.initialize(q_actual)
        dt = float(np.clip(dt, 1e-4, 0.1))
        if self.mode == "direct":
            q_next = self.q_ref + np.clip(target - self.q_ref, -self.max_step, self.max_step)
            q_next = np.minimum(q_next, q_actual + self.max_tracking_error)
            q_next = np.maximum(q_next, q_actual - self.max_tracking_error)
            self.dq_ref = np.clip((q_next - self.q_ref) / dt, -self.max_velocity, self.max_velocity)
            self.q_ref = q_next
        else:
            acceleration = np.clip(self.kp * (target - self.q_ref) - self.kd * self.dq_ref,
                                   -self.max_acceleration, self.max_acceleration)
            self.dq_ref = np.clip(self.dq_ref + acceleration * dt, -self.max_velocity, self.max_velocity)
            q_next = self.q_ref + self.dq_ref * dt
            step = np.clip(q_next - self.q_ref, -self.max_step, self.max_step)
            q_next = self.q_ref + step
            q_next = np.minimum(q_next, q_actual + self.max_tracking_error)
            q_next = np.maximum(q_next, q_actual - self.max_tracking_error)
            self.q_ref = q_next
        if self.joint_limits is not None:
            self.q_ref = np.clip(self.q_ref, self.joint_limits[:, 0], self.joint_limits[:, 1])
        return self.q_ref.copy()


class _LeaderProducer:
    def __init__(self, reader, mapper, period_s: float, stop: threading.Event, damping_config=None):
        self.reader, self.mapper = reader, mapper
        self.period_s, self.stop = float(period_s), stop
        self.lock = threading.Lock()
        self.latest: LeaderSample | None = None
        self.sequence = 0
        self.errors = 0
        self.thread: threading.Thread | None = None
        self.damping_config = damping_config
        self._previous_mapped: np.ndarray | None = None
        self._previous_t: float | None = None
        self._hold_reference: np.ndarray | None = None
        self._filtered_velocity: np.ndarray | None = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name=f"gello-reader-{self.mapper.config.port}", daemon=True)
        self.thread.start()

    def snapshot(self) -> LeaderSample | None:
        with self.lock:
            return self.latest

    def _run(self):
        next_t = time.monotonic()
        while not self.stop.is_set():
            started = time.monotonic()
            acquired = now_us()
            try:
                raw = np.asarray(self.reader.read(), dtype=float)
                mapped, _ = self.mapper.target(raw)
                sample = LeaderSample(raw[:self.mapper.n].copy(), mapped.copy(), now_us(), acquired,
                                     time.monotonic(), self.sequence, True)
                if self.damping_config is not None:
                    now = time.monotonic()
                    dt = max(now - (self._previous_t or now), 1e-4)
                    dq = np.zeros_like(mapped) if self._previous_mapped is None else (mapped - self._previous_mapped) / dt
                    cfg = self.damping_config
                    if self._hold_reference is None:
                        self._hold_reference = mapped.copy()
                    current = self.reader.compute_damping_current(
                        mapped, self._previous_mapped, dt, cfg, self._hold_reference,
                        self._filtered_velocity,
                    )
                    raw_dq = np.zeros_like(mapped) if self._previous_mapped is None else (mapped - self._previous_mapped) / dt
                    alpha = float(np.clip(getattr(cfg, "damping_velocity_filter_alpha", 1.0), 1e-6, 1.0))
                    self._filtered_velocity = raw_dq if self._filtered_velocity is None else alpha * raw_dq + (1.0 - alpha) * self._filtered_velocity
                    self.reader.write_current_damping(current)
                    if bool(cfg.weak_hold_enabled) and np.max(np.abs(dq)) < float(cfg.weak_hold_release_velocity):
                        self._hold_reference = 0.98 * self._hold_reference + 0.02 * mapped
                    self._previous_mapped, self._previous_t = mapped.copy(), now
                self.sequence += 1
                with self.lock:
                    self.latest = sample
            except Exception:
                self.errors += 1
                log.exception("GELLO sample failed on %s", self.mapper.config.port)
            next_t += self.period_s
            self.stop.wait(max(0.0, next_t - time.monotonic()))
            if time.monotonic() - started > max(self.period_s * 3, self.mapper.config.watchdog_timeout):
                next_t = time.monotonic()


class _FollowerStateProducer:
    """Continuously refresh one xArm state so control ticks never wait on a read."""

    def __init__(self, arm, period_s: float, stop: threading.Event, ddq_filter_alpha: float = 1.0,
                 read_gripper: bool = False):
        self.arm, self.period_s, self.stop = arm, float(period_s), stop
        self.lock = threading.Lock()
        self.latest = None
        self.errors = 0
        self.sequence = 0
        self.ddq_filter_alpha = float(np.clip(ddq_filter_alpha, 0.0, 1.0))
        self.filtered_ddq = None
        self.read_gripper = bool(read_gripper)
        self.latest_gripper = None
        self.thread: threading.Thread | None = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name=f"xarm-state-{self.arm.name}", daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.latest

    def _run(self):
        next_t = time.monotonic()
        while not self.stop.is_set():
            try:
                state = self.arm.read_state()
                self.sequence += 1
                raw_ddq = np.asarray(state.ddq, dtype=float)
                self.filtered_ddq = (raw_ddq.copy() if self.filtered_ddq is None else
                                     self.ddq_filter_alpha * raw_ddq + (1.0 - self.ddq_filter_alpha) * self.filtered_ddq)
                if self.read_gripper:
                    try:
                        self.latest_gripper = self.arm.read_gripper_state()
                    except Exception:
                        self.latest_gripper = None
                with self.lock:
                    self.latest = state
            except Exception:
                self.errors += 1
                log.exception("xArm state read failed on %s", self.arm.name)
            next_t += self.period_s
            self.stop.wait(max(0.0, next_t - time.monotonic()))
            if time.monotonic() - next_t > self.period_s * 3:
                next_t = time.monotonic()


class DualGelloPipeline:
    ARM_NAMES = ("left", "right")

    def __init__(self, config_path: str | Path, *, arm_factory: Callable | None = None,
                 reader_factory: Callable | None = None):
        self.config_path = Path(config_path).expanduser().resolve()
        self.raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        gello_path = Path(self.raw.get("gello_config", "xarm7_gello_teleop_dual_calibrated.yaml"))
        if not gello_path.is_absolute():
            gello_path = (self.config_path.parent / gello_path).resolve()
        self.configs = load_configs(gello_path, dual=True, require_calibrated=True)
        if tuple(name for name, _, _ in self.configs) != self.ARM_NAMES:
            raise ValueError("dual GELLO config must contain left and right in fixed order")
        self.control = dict(self.raw.get("control", {}))
        self.alignment = dict(self.raw.get("alignment", {}))
        self.sample_rate_hz = float(self.control.get("sample_rate_hz", 100.0))
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("control.sample_rate_hz must be positive")
        self.leader_max_age_s = float(self.control.get("leader_max_age_s", 0.15))
        self.state_max_age_s = float(self.control.get("state_max_age_s", 0.15))
        ddq_filter_alpha = float(self.control.get("ddq_filter_alpha", 1.0))
        if not np.isfinite(ddq_filter_alpha) or not 0.0 <= ddq_filter_alpha <= 1.0:
            raise ValueError("control.ddq_filter_alpha must be in [0, 1]")
        self.queue_size = int(self.control.get("sample_queue_size", 256))
        self.stop_event = threading.Event()
        self.state = "disconnected"
        self.arms = []
        self.readers = []
        self.mappers = []
        self.producers: list[_LeaderProducer] = []
        self.state_producers: list[_FollowerStateProducer] = []
        self.damping_capabilities: list[dict[str, Any]] = []
        self.followers: list[SecondOrderPositionFollower] = []
        self.last_q_cmd: list[np.ndarray | None] = [None, None]
        self.last_q_cmd_t = [0, 0]
        self.last_q_cmd_seq = [0, 0]
        self.last_q_cmd_ok = [True, True]
        self.command_history: list[deque] = [deque(maxlen=256), deque(maxlen=256)]
        self._last_follower_timestamp_us = [None, None]
        self.sample_queue: queue.Queue[ControlSample] = queue.Queue(maxsize=self.queue_size)
        self.control_thread: threading.Thread | None = None
        self.control_error: BaseException | None = None
        self.queue_overflow = 0
        self.stale_count = 0
        self._last_control_t: float | None = None
        self.control_tick_count = 0
        self.control_started_t: float | None = None
        self.recording_started_t: float | None = None
        self._arm_factory = arm_factory or self._default_arm_factory
        self._reader_factory = reader_factory or GelloReader
        self.collection_config = self._make_collection_config()
        self.buffer: EpisodeBuffer | None = None
        self.recording = False
        self.camera_manager = CameraManager.from_config(self.collection_config.cameras)
        self.gripper_enabled = bool(self.raw.get("gripper", {}).get("enabled", False))

    def _default_arm_factory(self, side, robot):
        arm_cfg = dict(self.raw.get("arms", {}).get(side, {}))
        kwargs = {
            "robot_ip": robot.robot_ip,
            "dof": len(robot.reset_q),
            "execution_enabled": bool(arm_cfg.get("execution_enabled", False)),
            "servo_api": "set_servo_angle_j",
            "joint_speed_rad_s": float(self.control.get("joint_speed_rad_s", 1.0)),
            "joint_acc_rad_s2": float(self.control.get("joint_acc_rad_s2", 5.0)),
            "sdk_timeout_s": float(self.control.get("sdk_timeout_s", 0.1)),
            "feedback_signal": str(arm_cfg.get("feedback_signal", "torque")),
            "velocity_source": str(arm_cfg.get("velocity_source", "hardware")),
        }
        kwargs.update({k: v for k, v in arm_cfg.items() if k not in {"robot_ip", "execution_enabled"}})
        return XArmAdapter(ArmEndpointConfig(name=side, rest_q=tuple(robot.reset_q), config_kwargs=kwargs))

    def _make_collection_config(self) -> CollectionConfig:
        output_raw = dict(self.raw.get("output", {}))
        output_dir = Path(output_raw.get("directory", "./episodes"))
        if not output_dir.is_absolute():
            output_dir = (self.config_path.parent / output_dir).resolve()
        pairs = []
        for side, robot, _ in self.configs:
            follower_kwargs = dict(self.raw.get("arms", {}).get(side, {}))
            follower_kwargs["robot_ip"] = robot.robot_ip
            pairs.append(ArmPairConfig(
                name=side,
                leader=ArmEndpointConfig(name=f"{side}_gello", rest_q=tuple(robot.reset_q)),
                follower=ArmEndpointConfig(name=side, rest_q=tuple(robot.reset_q), config_kwargs=follower_kwargs),
            ))
        cameras = tuple(_camera_config(item) for item in self.raw.get("cameras", []) if item.get("enabled", True))
        return CollectionConfig(
            teleop=TeleopConfig(master_slave=tuple(pairs), command=CommandConfig(control_mode="position", sample_rate_hz=self.sample_rate_hz)),
            output=OutputConfig(directory=output_dir, prefix=str(output_raw.get("prefix", "episode")), discard_initial_s=float(output_raw.get("discard_initial_s", 0.0))),
            cameras=cameras,
            raw_yaml=self.config_path.read_text(encoding="utf-8"),
        )

    def connect(self):
        if self.state != "disconnected":
            raise RuntimeError(f"connect is invalid in state {self.state}")
        for side, robot, leader in self.configs:
            arm = self._arm_factory(side, robot)
            arm.connect()
            arm.read_state()  # connection/feedback check before any motion
            reader = self._reader_factory(leader)
            mapper = JointMapper(robot, leader)
            # Only saved calibration is accepted.  JointMapper.align uses it
            # verbatim; it does not infer a new offset from the live pose.
            if not mapper.has_saved_calibration:
                raise RuntimeError(f"{side}: calibrated joint_offsets and leader_reference_q are required")
            self.arms.append(arm)
            self.readers.append(reader)
            self.mappers.append(mapper)
        self.camera_manager.start()
        self.state = "connected"

    def reset(self):
        if self.state != "connected":
            raise RuntimeError("both arms must pass connection check before reset")
        for arm in self.arms:
            arm.enable()
        reset_timeout = float(self.alignment.get("reset_timeout_s", 60.0))
        reset_speed = float(self.alignment.get("reset_speed_rad_s", 0.2))
        reset_acc = float(self.alignment.get("reset_acc_rad_s2", 0.5))
        for arm, (_, robot, _) in zip(self.arms, self.configs):
            if hasattr(arm, "move_to_reset"):
                arm.move_to_reset(np.asarray(robot.reset_q), speed=reset_speed, acceleration=reset_acc)
            else:  # small mock adapters used by offline tests
                arm.command_joint_positions(np.asarray(robot.reset_q))
        deadline = time.monotonic() + reset_timeout
        settled = [0, 0]
        tolerance = float(self.alignment.get("reset_tolerance_rad", 0.03))
        while time.monotonic() < deadline and min(settled) < int(self.alignment.get("reset_samples", 5)):
            for i, (arm, (_, robot, _)) in enumerate(zip(self.arms, self.configs)):
                state = arm.read_state()
                if np.max(np.abs(state.q - np.asarray(robot.reset_q))) <= tolerance:
                    settled[i] += 1
                else:
                    settled[i] = 0
            time.sleep(0.02)
        if min(settled) < int(self.alignment.get("reset_samples", 5)):
            raise TimeoutError(f"xArm reset did not settle: samples={settled}")
        for reader in self.readers:
            reader.prepare_alignment()
        self.state = "aligning"

    def alignment_report(self) -> dict[str, np.ndarray]:
        if self.state not in {"aligning", "aligned", "holding"}:
            raise RuntimeError("reset must complete before alignment check")
        report = {}
        for i, (side, robot, _) in enumerate(self.configs):
            raw = np.asarray(self.readers[i].read(), dtype=float)
            self.mappers[i].align(raw)
            mapped, _ = self.mappers[i].target(raw)
            actual = self.arms[i].read_state().q
            report[side] = mapped - actual
        return report

    def confirm_alignment(self) -> dict[str, np.ndarray]:
        report = self.alignment_report()
        tolerance = float(self.alignment.get("alignment_tolerance_rad", 0.05))
        bad = {side: error for side, error in report.items() if np.max(np.abs(error)) > tolerance}
        if bad:
            details = "; ".join(f"{side} max={np.max(np.abs(err)):.4f} rad" for side, err in bad.items())
            raise RuntimeError(f"GELLO/xArm alignment failed ({details}); offsets were not recalculated")
        self.state = "aligned"
        return report

    def takeover(self):
        if self.state not in {"aligned", "holding"}:
            raise RuntimeError("alignment must be confirmed before takeover")
        self.stop_event.clear()
        self.last_q_cmd_ok = [True, True]
        self.followers.clear()
        for i, arm in enumerate(self.arms):
            actual = arm.read_state().q
            mapped, _ = self.mappers[i].target(self.readers[i].read())
            tolerance = float(self.alignment.get("alignment_tolerance_rad", 0.05))
            if np.max(np.abs(mapped - actual)) > tolerance:
                raise RuntimeError(f"{self.ARM_NAMES[i]} moved after alignment confirmation; takeover refused")
            self.mappers[i].last_raw = mapped.copy()
            self.mappers[i].last_target = actual.copy()
            arm.command_joint_positions(actual)
            mode = str(self.control.get("position_mode", "direct"))
            follower = SecondOrderPositionFollower(
                arm.dof, mode=mode,
                kp=self.control.get("kp", 18.0), kd=self.control.get("kd", 3.0),
                max_velocity=self.control.get("max_velocity_rad_s", 1.5),
                max_acceleration=self.control.get("max_acceleration_rad_s2", 8.0),
                max_step=self.control.get("max_step_rad", 0.08),
                max_tracking_error=self.control.get("max_tracking_error_rad", 0.25),
                joint_limits=self.mappers[i].config.joint_limits,
            )
            follower.initialize(actual)
            self.followers.append(follower)
            self.last_q_cmd[i] = actual.copy()
            self.last_q_cmd_t[i] = now_us()
            self.last_q_cmd_seq[i] = 0
            self.command_history[i].clear()
            self.command_history[i].append((self.last_q_cmd_t[i], self.last_q_cmd[i].copy(), 0, True))
        self.producers = []
        self.damping_capabilities = []
        ddq_alpha = float(self.control.get("ddq_filter_alpha", 1.0))
        if self.gripper_enabled:
            for arm in self.arms:
                arm.init_gripper()
        self.state_producers = [_FollowerStateProducer(arm, 1.0 / self.sample_rate_hz, self.stop_event, ddq_alpha, self.gripper_enabled)
                                for arm in self.arms]
        for producer in self.state_producers:
            producer.start()
        deadline = time.monotonic() + 1.0
        while any(producer.snapshot() is None for producer in self.state_producers):
            if time.monotonic() >= deadline:
                raise RuntimeError("xArm state producers did not deliver initial feedback")
            time.sleep(0.001)
        for i, (reader, mapper) in enumerate(zip(self.readers, self.mappers)):
            damping_cfg = self.configs[i][2]
            damping = (damping_cfg if bool(getattr(damping_cfg, "damping_enabled", False)) and
                       str(getattr(damping_cfg, "damping_mode", "none")).lower() == "current" else None)
            if damping is not None:
                self.damping_capabilities.append(reader.enable_current_damping(damping))
                after_mode, _ = mapper.target(reader.read())
                current_actual = self.arms[i].read_state().q
                if np.max(np.abs(after_mode - current_actual)) > float(self.alignment.get("alignment_tolerance_rad", 0.05)):
                    raise RuntimeError(f"{self.ARM_NAMES[i]} GELLO position changed during damping mode switch")
            else:
                self.damping_capabilities.append({"enabled": False})
            self.producers.append(_LeaderProducer(reader, mapper, 1.0 / float(mapper.config.fps), self.stop_event, damping))
        for producer in self.producers:
            producer.start()
        self.state = "following"
        self._last_control_t = time.monotonic()
        self.control_thread = threading.Thread(target=self._control_loop, name="xarm-100hz-control", daemon=True)
        self.control_thread.start()

    def start_episode(self):
        if self.state != "following":
            raise RuntimeError("r is only accepted after alignment and takeover")
        self._drain_samples()
        # Drop camera frames queued before the operator pressed r; their
        # independent timestamps must not leak into this episode.
        while self.camera_manager.poll():
            pass
        self.buffer = EpisodeBuffer(self.collection_config, self.ARM_NAMES, enable_online_tau_ext=False)
        self.buffer.episode_metadata.update({"pipeline_state": "recording", "clock": "monotonic scheduler / unix host_receive",
                                             "arm_names": list(self.ARM_NAMES), "joint_order": "J1..J7 per arm",
                                             "leader_calibration": [self._calibration_snapshot(i) for i in range(2)],
                                             "gello_damping_capabilities": self.damping_capabilities,
                                             "xarm_sdk_version": _sdk_version(),
                                             "xarm_firmware": [self._firmware_snapshot(arm) for arm in self.arms],
                                             "xarm_feedback_source": [getattr(state, "feedback_source", "unknown")
                                                                       for state in [p.snapshot() for p in self.state_producers]]})
        self.recording = True
        self.recording_started_t = time.monotonic()
        self.state = "recording"

    def stop_episode(self, save: bool, index: int | None = None) -> Path | None:
        if self.state != "recording":
            raise RuntimeError("no recording episode is active")
        # Stop producers first, then drain the final causal rows while the
        # episode is still marked recording.
        self._hold_both()
        self._consume_samples()
        self.recording = False
        if self.buffer is not None:
            elapsed = max(time.monotonic() - (self.recording_started_t or time.monotonic()), 1e-6)
            self.buffer.episode_metadata.update({
                "actual_recorded_hz": self.buffer.sample_count / elapsed,
                "control_tick_count": self.control_tick_count,
                "gello_reader_errors": [producer.errors for producer in self.producers],
                "xarm_state_reader_errors": [producer.errors for producer in self.state_producers],
                "queue_overflow_count": self.queue_overflow,
                "expired_target_count": self.stale_count,
            })
        self.state = "holding"
        if not save or self.buffer is None or self.buffer.sample_count == 0:
            return None
        output_dir = self.collection_config.output.directory
        index = next_episode_index(output_dir, self.collection_config.output.prefix) if index is None else index
        return self.buffer.save(episode_path(output_dir, self.collection_config.output.prefix, index))

    def _control_loop(self):
        ticker = FixedRateTicker(self.sample_rate_hz, float(self.control.get("maximum_lateness_s", 0.03)))
        try:
            self.control_started_t = time.monotonic()
            while not self.stop_event.is_set() and self.state in {"following", "recording"}:
                _, lateness = ticker.wait("dual xArm position control")
                self.control_tick_count += 1
                now = time.monotonic()
                dt = max(now - (self._last_control_t or now), 1e-4)
                self._last_control_t = now
                leaders = tuple(p.snapshot() for p in self.producers)
                if any(s is None or now - s.acquired_monotonic_s > self.leader_max_age_s for s in leaders):
                    self.stale_count += 1
                    raise RuntimeError("GELLO target expired/disconnected; both arms stopped accepting new targets")
                states = [producer.snapshot() for producer in self.state_producers]
                if any(state is None for state in states):
                    raise RuntimeError("xArm state producer has not delivered an initial sample")
                state_age_s = []
                for state in states:
                    acquired_us = int(getattr(state, "acquired_timestamp_us", 0) or 0)
                    if acquired_us <= 0:
                        acquired_us = int(getattr(state, "timestamp_us", now_us()))
                    state_age_s.append(max(0.0, (now_us() - acquired_us) * 1e-6))
                if any(age > self.state_max_age_s for age in state_age_s):
                    raise RuntimeError(f"xArm feedback expired: ages_s={state_age_s}")
                repeated_feedback = np.asarray([
                    self._last_follower_timestamp_us[i] == int(getattr(state, "timestamp_us", 0))
                    for i, state in enumerate(states)
                ], dtype=np.uint8)
                self._last_follower_timestamp_us = [int(getattr(state, "timestamp_us", 0)) for state in states]
                filtered_ddq = np.concatenate([
                    producer.filtered_ddq if producer.filtered_ddq is not None else np.asarray(state.ddq, dtype=float)
                    for producer, state in zip(self.state_producers, states)
                ])
                for i, (arm, state, leader, follower) in enumerate(zip(self.arms, states, leaders, self.followers)):
                    target = follower.update(leader.mapped, state.q, dt)
                    try:
                        arm.command_joint_positions(target)
                    except Exception:
                        self.last_q_cmd_ok[i] = False
                        raise
                    self.last_q_cmd[i] = target.copy()
                    self.last_q_cmd_t[i] = now_us()
                    self.last_q_cmd_seq[i] += 1
                    self.last_q_cmd_ok[i] = True
                    self.command_history[i].append((self.last_q_cmd_t[i], self.last_q_cmd[i].copy(),
                                                    self.last_q_cmd_seq[i], True))
                # State was acquired before this tick's command. Pair it with
                # the prior successful command (causal ZOH), not the target
                # that was just calculated.
                selected_commands = []
                for i, state in enumerate(states):
                    state_time = int(getattr(state, "acquired_timestamp_us", 0) or getattr(state, "timestamp_us", 0))
                    candidates = [item for item in self.command_history[i] if item[0] <= state_time]
                    selected_commands.append(candidates[-1] if candidates else
                                             (self.last_q_cmd_t[i], self.last_q_cmd[i].copy(), self.last_q_cmd_seq[i], False))
                q_cmd = np.concatenate([item[1] for item in selected_commands])
                selected_ts = np.asarray([item[0] for item in selected_commands], dtype=np.int64)
                selected_ok = np.asarray([item[3] for item in selected_commands], dtype=np.uint8)
                selected_seq = np.asarray([item[2] for item in selected_commands], dtype=np.int64)
                row = ControlSample(now_us(), tuple(states), tuple(leaders), q_cmd,
                                    selected_ts, selected_ok, selected_seq, repeated_feedback, filtered_ddq,
                                    np.asarray([producer.latest_gripper.value if producer.latest_gripper is not None else np.nan
                                                for producer in self.state_producers], dtype=np.float64),
                                    np.asarray([producer.latest_gripper is not None for producer in self.state_producers], dtype=np.uint8),
                                    int(lateness * 1e6))
                try:
                    self.sample_queue.put_nowait(row)
                except queue.Full:
                    self.queue_overflow += 1
                    try:
                        self.sample_queue.get_nowait()
                        self.sample_queue.put_nowait(row)
                    except queue.Empty:
                        pass
        except BaseException as exc:
            self.control_error = exc
            self.stop_event.set()
            log.exception("dual xArm control loop stopped")

    def _consume_samples(self):
        if self.buffer is None:
            return
        while True:
            try:
                sample = self.sample_queue.get_nowait()
            except queue.Empty:
                break
            self.buffer.append_teleop(sample.timestamp_us, self._values(sample), store=self.recording)
            for frame in self.camera_manager.poll():
                self.buffer.append_camera(frame.camera_name, frame.timestamp_us, frame.frame, frame.depth)

    def poll(self):
        self._consume_samples()
        if self.control_error is not None:
            raise RuntimeError(str(self.control_error)) from self.control_error

    def _values(self, sample: ControlSample):
        states = sample.follower_states
        leaders = sample.leader_samples
        q_follower = np.concatenate([s.q for s in states])
        q_cmd = sample.q_cmd.copy()
        values = {
            "q_follower": ("q", q_follower),
            "q_cmd": ("q", q_cmd),
            "delta_q": ("q_error", q_cmd - q_follower),
            "dq_follower": ("velocity", np.concatenate([s.dq for s in states])),
            "ddq_follower_raw": ("acceleration_raw", np.concatenate([s.ddq for s in states])),
            "ddq_follower": ("acceleration", sample.ddq_follower),
            "dq_valid_follower": ("validity", np.asarray([all(s.dq_valid for s in states)], dtype=np.uint8)),
            "ddq_valid_follower": ("validity", np.asarray([all(s.dq_valid for s in states)], dtype=np.uint8)),
            "tau_follower": ("torque", np.concatenate([s.torque for s in states])),
            "current_follower": ("current", np.concatenate([s.current for s in states])),
            "torque_valid_follower": ("validity", np.asarray([all(s.torque_valid for s in states)], dtype=np.uint8)),
            "current_valid_follower": ("validity", np.asarray([all(s.current_valid for s in states)], dtype=np.uint8)),
            "q_leader_raw": ("q_raw", np.concatenate([s.raw for s in leaders])),
            "q_leader_mapped": ("q", np.concatenate([s.mapped for s in leaders])),
            "q_leader_valid": ("validity", np.asarray([all(s.valid for s in leaders)], dtype=np.uint8)),
            "q_leader_timestamp_us": ("timestamp", np.asarray([s.timestamp_us for s in leaders], dtype=np.int64)),
            "q_leader_acquired_timestamp_us": ("timestamp", np.asarray([s.acquired_timestamp_us for s in leaders], dtype=np.int64)),
            "q_leader_sequence": ("sequence", np.asarray([s.sequence for s in leaders], dtype=np.int64)),
            "q_leader_age_us": ("duration", np.asarray([max(0, sample.timestamp_us - s.timestamp_us) for s in leaders], dtype=np.int64)),
            "q_follower_timestamp_us": ("timestamp", np.asarray([s.timestamp_us for s in states], dtype=np.int64)),
            "q_follower_acquired_timestamp_us": ("timestamp", np.asarray([s.acquired_timestamp_us for s in states], dtype=np.int64)),
            "q_follower_sequence": ("sequence", np.asarray([max(0, p.sequence - 1) for p in self.state_producers], dtype=np.int64)),
            "q_follower_age_us": ("duration", np.asarray([max(0, sample.timestamp_us - int(getattr(s, "timestamp_us", sample.timestamp_us))) for s in states], dtype=np.int64)),
            "q_follower_valid": ("validity", np.asarray([all(s.q_valid for s in states)], dtype=np.uint8)),
            "q_follower_repeated": ("validity", sample.q_follower_repeated),
            "q_cmd_timestamp_us": ("timestamp", sample.q_cmd_timestamp_us),
            "q_cmd_send_ok": ("validity", sample.q_cmd_ok),
            "q_cmd_sequence": ("sequence", sample.q_cmd_sequence),
            "sample_lateness_us": ("duration", np.asarray([sample.lateness_us], dtype=np.int64)),
            "gripper_follower": ("gripper", sample.gripper_follower),
            "gripper_cmd": ("gripper", np.full((2,), np.nan, dtype=np.float64)),
            "gripper_follower_valid": ("validity", sample.gripper_follower_valid),
            "gripper_cmd_valid": ("validity", np.zeros((2,), dtype=np.uint8)),
        }
        return values

    def _drain_samples(self):
        while True:
            try:
                self.sample_queue.get_nowait()
            except queue.Empty:
                return

    def _hold_both(self):
        self.stop_event.set()
        for producer in self.producers:
            if producer.thread is not None:
                producer.thread.join(timeout=1.0)
        for producer in self.state_producers:
            if producer.thread is not None:
                producer.thread.join(timeout=1.0)
        if self.control_thread is not None:
            self.control_thread.join(timeout=1.0)
        for arm in self.arms:
            try:
                state = arm.read_state()
                arm.command_joint_positions(state.q)
            except Exception:
                log.exception("failed to hold %s", arm.name)
        for reader in self.readers:
            try:
                cfg = reader.config
                if (bool(getattr(cfg, "damping_enabled", False)) and
                        str(getattr(cfg, "damping_mode", "none")).lower() == "current"):
                    reader.disable_current_damping()
                reader.set_torque(False, ids=list(reader.config.joint_ids))
            except Exception:
                pass

    def close(self):
        if self.state in {"following", "recording"}:
            # A fault, stale target, or command error must leave both xArms in
            # a local position hold before their SDK connections are closed.
            self._hold_both()
        self.stop_event.set()
        if self.control_thread is not None:
            self.control_thread.join(timeout=1.0)
        try:
            self.camera_manager.stop()
        except Exception:
            pass
        for reader in self.readers:
            try:
                reader.close()
            except Exception:
                log.exception("failed to close GELLO reader")
        for arm in self.arms:
            try:
                if bool(getattr(arm, "_enabled", False)):
                    arm.disable()
                arm.disconnect()
            except Exception:
                log.exception("failed to disconnect xArm %s", arm.name)
        self.state = "stopped"

    def _calibration_snapshot(self, i):
        mapper = self.mappers[i]
        return {"side": self.ARM_NAMES[i], "joint_signs": list(mapper.config.joint_signs),
                "joint_offsets": list(mapper.config.joint_offsets),
                "leader_reference_q": list(mapper.config.leader_reference_q)}

    @staticmethod
    def _firmware_snapshot(arm):
        try:
            result = arm._call("get_version", required=False)
            return str(result) if result is not None else "unavailable"
        except Exception:
            return "unavailable"

    def interactive(self, auto_save=False):
        with TerminalKeys() as keys:
            if not keys.is_tty:
                raise RuntimeError("interactive keyboard is unavailable; use --dry-run with mocks")
            _wait_enter(keys, "连接检查完成。按 Enter 让两台 xArm 低速回到各自 reset_q：")
            self.reset()
            _wait_enter(keys, "两台 xArm 已到位。请分别手动对齐两条 GELLO，按 Enter 检查：")
            report = self.confirm_alignment()
            for side, error in report.items():
                print(f"{side} 对齐误差(rad): {np.array2string(error, precision=4)}", flush=True)
            _wait_enter(keys, "两侧均在阈值内。托住主手，按 Enter 接管遥操：")
            self.takeover()
            print("遥操已接管；r 开始采集，空格停止并保持，q 退出。", flush=True)
            episode_index = None
            while True:
                self.poll()
                key = keys.read_key(0.01)
                if key in {"q", "Q", "\x03"}:
                    break
                if key in {"t", "T"} and self.state == "holding":
                    _wait_enter(keys, "请重新确认两侧 GELLO 与 xArm 当前姿态一致，按 Enter 检查：")
                    self.confirm_alignment()
                    _wait_enter(keys, "检查通过。按 Enter 重新接管：")
                    self.takeover()
                    print("遥操已重新接管；按 r 开始新的 episode。", flush=True)
                    continue
                if key in {"r", "R"}:
                    if self.state != "following":
                        print("当前未处于可采集的遥操状态；停止后需重新接管。", flush=True)
                    else:
                        self.start_episode()
                        print("episode recording", flush=True)
                elif key == " ":
                    if self.state == "recording":
                        save = bool(auto_save)
                        if not save:
                            print("保存本 episode 吗？按 y 保存，n 丢弃，q 退出：", flush=True)
                            while True:
                                answer = keys.read_key(0.1)
                                if answer in {"y", "Y"}:
                                    save = True
                                    break
                                if answer in {"n", "N"}:
                                    break
                                if answer in {"q", "Q", "\x03"}:
                                    raise KeyboardInterrupt
                        path = self.stop_episode(save, episode_index)
                        print(f"episode {'saved to ' + str(path) if path else 'discarded'}", flush=True)
                        self.state = "holding"
        return 0


def _wait_enter(keys: TerminalKeys, prompt: str) -> None:
    print(prompt, end=" ", flush=True)
    while True:
        key = keys.read_key(0.1)
        if key in {"\r", "\n"}:
            return
        if key in {"q", "Q", "\x03"}:
            raise KeyboardInterrupt


def _joint_vector(value, dof, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(dof, float(array))
    array = array.reshape(-1)
    if array.shape != (dof,) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must be a non-negative scalar or {dof}-vector")
    return array


def _camera_config(item):
    value = dict(item)
    return CameraConfig(name=str(value.pop("name")), **value)


def _sdk_version():
    try:
        import xarm
        return str(getattr(xarm, "__version__", getattr(xarm, "SDK_VERSION", "unknown")))
    except Exception:
        return "unavailable"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--auto-save", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pipeline = DualGelloPipeline(args.config)
    try:
        if args.check_config:
            print(json.dumps({"state": pipeline.state, "arm_names": list(pipeline.ARM_NAMES),
                              "sample_rate_hz": pipeline.sample_rate_hz,
                              "gello_config": pipeline.raw.get("gello_config")}, indent=2))
            return 0
        pipeline.connect()
        return pipeline.interactive(auto_save=args.auto_save)
    except KeyboardInterrupt:
        return 130
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
