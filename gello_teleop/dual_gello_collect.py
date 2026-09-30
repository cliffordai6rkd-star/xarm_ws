#!/usr/bin/env python3
"""双臂 GELLO 位置遥操和异步 HDF5 数采入口。

The hardware lifecycle is intentionally explicit:

``connect -> reset -> automatic alignment check -> takeover -> (r) record``.

GELLO and xArm feedback use independent producer threads.  The 100 Hz controller
tracks the latest valid GELLO target; a separate 100 Hz recorder keeps capturing
state while following is held or being realigned.  Camera preview runs in its
own process, and episode storage consumes the bounded state queue.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import queue
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from gello_teleop.gello_hardware import GelloReader
from gello_teleop.gello_damping import damping_config
from gello_teleop.inspect_gello_grippers import format_status
from gello_teleop.uf_robot_gello_teleop import JointMapper, load_configs
from gello_teleop.reference_pose import ReferencePosePublisher, capture_current_pose, overwrite_reference_pose
from nero_collection.cameras import CameraManager, CameraVisualizer
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
    gripper_fraction: float = np.nan
    gripper_angle_rad: float = np.nan


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
    gripper_cmd: np.ndarray
    gripper_cmd_valid: np.ndarray
    gripper_cmd_timestamp_us: np.ndarray
    gripper_follower_timestamp_us: np.ndarray


class _GripperPressDetector:
    """One event per squeeze, using raw motion even with incorrect calibration."""
    def __init__(self, press_deg=2.0, release_deg=1.0):
        self.press_deg, self.release_deg = press_deg, release_deg
        self.open_deg = None
        self.pressed = False

    def update(self, angle_deg):
        if angle_deg is None or not np.isfinite(angle_deg):
            return False
        if self.open_deg is None:
            self.open_deg = angle_deg
            return False
        travel = abs((angle_deg-self.open_deg+180.) % 360.-180.)
        if self.pressed:
            if travel <= self.release_deg:
                self.pressed = False
            return False
        if travel >= self.press_deg:
            self.pressed = True
            return True
        return False


class _GripperTriggerDetector:
    """Latch a full squeeze until released; releasing never issues a command."""
    def __init__(self):
        # Require an initial released sample, avoiding a command during takeover.
        self.pressed = True

    def update(self, fraction):
        if fraction is None or not np.isfinite(fraction):
            return False
        if self.pressed:
            if fraction <= 0.2:
                self.pressed = False
            return False
        if fraction >= 0.95:
            self.pressed = True
            return True
        return False


class _GripperWorker:
    """Low-rate gripper I/O; keep it outside the joint-control scheduler."""

    def __init__(self, arm, producer, stop, *, period_s, maximum_age_s,
                 command_enabled, min_width, max_width, max_speed, pause=None,
                 control_mode='absolute_position', mode='follow'):
        self.arm, self.producer, self.stop = arm, producer, stop
        self.period_s, self.maximum_age_s = period_s, maximum_age_s
        self.command_enabled = command_enabled
        self.pause = pause if pause is not None else threading.Event()
        self.control_mode = control_mode
        if mode not in {'trigger', 'follow'}:
            raise ValueError('gripper.mode must be trigger or follow')
        self.mode = mode
        self.press_detector = _GripperTriggerDetector()
        self.trigger_closed = False
        self.min_width, self.max_width, self.max_speed = min_width, max_width, max_speed
        self.lock = threading.Lock()
        self.latest = None
        self.history = deque(maxlen=256)
        self.error = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name=f'gripper-{self.arm.name}', daemon=True)
        self.thread.start()

    def snapshot(self, state_time):
        with self.lock:
            candidates = [row for row in self.history if row[0] <= state_time]
            ts, cmd = candidates[-1] if candidates else (0, np.nan)
            actual = self.latest
        value = actual.value if actual is not None else np.nan
        return value, bool(np.isfinite(value)), cmd, bool(candidates), ts, actual.timestamp_us if actual is not None else 0

    def _run(self):
        command, previous = None, time.monotonic()
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                state = self.arm.read_gripper_state()
                with self.lock:
                    self.latest = state
                leader = self.producer.snapshot()
                pressed = False
                if (self.mode == 'trigger' and leader is not None
                        and 0 <= time.monotonic()-leader.acquired_monotonic_s <= self.maximum_age_s):
                    # Consume presses even while paused so they cannot execute on resume.
                    pressed = self.press_detector.update(leader.gripper_fraction)
                if self.command_enabled and not self.pause.is_set():
                    if leader is None:
                        self.stop.wait(self.period_s)
                        continue
                    input_value = leader.gripper_fraction
                    if time.monotonic()-leader.acquired_monotonic_s > self.maximum_age_s or not np.isfinite(input_value):
                        raise RuntimeError('GELLO gripper target expired or unavailable')
                    if self.mode == 'trigger' and not pressed:
                        self.stop.wait(max(0., self.period_s-(time.monotonic()-started)))
                        continue
                    if command is None:
                        if not np.isfinite(state.value):
                            raise RuntimeError('Initial gripper width feedback is unavailable')
                        command = float(state.value)
                    target = (self.max_width if self.trigger_closed else self.min_width) if self.mode == 'trigger' else (
                        self.max_width-leader.gripper_fraction*(self.max_width-self.min_width))
                    if self.mode == 'follow' and self.control_mode == 'rate_limited_position':
                        step = self.max_speed*min(max(started-previous, 0.), self.period_s)
                        target = command+np.clip(target-command, -step, step)
                    target = float(np.clip(target, self.min_width, self.max_width))
                    if self.stop.is_set():
                        break
                    with self.lock:
                        if self.pause.is_set():
                            continue
                        self.arm.command_gripper(target, force_n=0., mode='width')
                        command = target
                        if self.mode == 'trigger':
                            self.trigger_closed = not self.trigger_closed
                        self.history.append((now_us(), command))
                previous = started
                self.stop.wait(max(0., self.period_s-(time.monotonic()-started)))
        except Exception as exc:
            self.error = exc
            self.stop.set()
            log.exception('gripper I/O failed on %s', self.arm.name)


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
            step_limit = np.minimum(self.max_step, self.max_velocity*dt)
            q_next = self.q_ref + np.clip(target - self.q_ref, -step_limit, step_limit)
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
    def __init__(self, reader, mapper, period_s: float, stop: threading.Event, damping_config=None, failure_stop=None):
        self.reader, self.mapper = reader, mapper
        self.period_s, self.stop = float(period_s), stop
        self.failure_stop = failure_stop if failure_stop is not None else stop
        self.lock = threading.Lock()
        self.latest: LeaderSample | None = None
        self.sequence = 0
        self.errors = 0
        self.last_error = None
        self.last_read_s = 0.
        self.last_io_s = 0.
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
                self.last_read_s = time.monotonic()-started
                mapped, _ = self.mapper.target(raw)
                fraction = np.nan
                if self.mapper.config.gripper_id >= 0:
                    fraction = _gripper_fraction(raw[self.mapper.n], self.mapper.gripper_open, self.mapper.gripper_close)
                sample = LeaderSample(raw[:self.mapper.n].copy(), mapped.copy(), now_us(), acquired,
                                     time.monotonic(), self.sequence, True, fraction,
                                      raw[self.mapper.n] if self.mapper.config.gripper_id >= 0 else np.nan)
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
                self.last_io_s = time.monotonic()-started
                with self.lock:
                    self.latest = sample
                self.last_error = None
            except Exception as exc:
                self.errors += 1
                self.last_error = str(exc)
                log.exception("GELLO sample failed on %s", self.mapper.config.port)
                if self.damping_config is not None:
                    self.stop.set()
                    self.failure_stop.set()
                    try:
                        self.reader.disable_current_damping()
                    except Exception:
                        log.exception("Failed to release GELLO damping")
                    return
            next_t += self.period_s
            self.stop.wait(max(0.0, next_t - time.monotonic()))
            if time.monotonic() - started > max(self.period_s * 3, self.mapper.config.watchdog_timeout):
                next_t = time.monotonic()


class _EncoderDampingMapper:
    """Damp actual encoder motion while waiting; never command followers."""
    def __init__(self, config):
        self.config = replace(config, joint_signs=tuple([1]*len(config.joint_ids)),
                              gripper_id=-1, weak_hold_enabled=False)
        self.n = len(config.joint_ids)

    def target(self, raw):
        angles = np.asarray(raw, float)[:self.n]
        if angles.shape != (self.n,) or not np.isfinite(angles).all():
            raise ValueError('Invalid alignment damping encoder sample')
        return angles, raw


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
        active = self.raw.get("active_arms", list(type(self).ARM_NAMES))
        if (not isinstance(active, list) or not active
                or any(side not in type(self).ARM_NAMES for side in active)
                or len(set(active)) != len(active)):
            raise ValueError('active_arms must be a nonempty list of unique left/right sides')
        # Preserve the dataset's canonical side order, including right-only mode.
        self.ARM_NAMES = tuple(side for side in type(self).ARM_NAMES if side in active)
        gello_path = Path(self.raw.get("gello_config", "xarm7_gello_calibration.yaml"))
        if not gello_path.is_absolute():
            gello_path = (self.config_path.parent / gello_path).resolve()
        if not gello_path.exists():
            raise ValueError('尚无标定结果；先运行 calibrate_joint_directions.py --left，再运行 --right')
        self.gello_config_path = gello_path.resolve()
        self.configs = load_configs(gello_path, dual=True, require_calibrated=True,
                                    side=self.ARM_NAMES[0] if len(self.ARM_NAMES) == 1 else None)
        self.configs = [(side, robot, damping_config(robot, leader, side, self.raw.get('gello_damping')))
                        for side, robot, leader in self.configs]
        if tuple(name for name, _, _ in self.configs) != self.ARM_NAMES:
            raise ValueError("GELLO config must contain the active arms in fixed order")
        self.control = dict(self.raw.get("control", {}))
        self.alignment = dict(self.raw.get("alignment", {}))
        for key in ('alignment_tolerance_rad', 'alignment_samples'):
            value = self.alignment.get(key, .05 if key.endswith('rad') else 5)
            if not np.isfinite(value) or value <= 0 or (key.endswith('samples') and int(value) != value):
                raise ValueError(f'alignment.{key} must be positive')
        if 'gello_hold_current_fraction' in self.alignment:
            raise ValueError('Replace gello_hold_current_fraction with gello_hold_current_raw: 80')
        current = self.alignment.get('gello_hold_current_raw')
        if current is not None and (type(current) is not int or current <= 0):
            raise ValueError('alignment.gello_hold_current_raw must be a positive integer')
        for key, default in [('gello_reset_steps', 15), ('gello_reset_interval_s', .2),
                             ('gello_reset_max_travel_deg', 90.), ('gello_reset_tracking_error_deg', 10.),
                             ('reset_speed_rad_s', .2), ('reset_acc_rad_s2', .5),
                             ('reset_timeout_s', 60.), ('reset_tolerance_rad', .03), ('reset_samples', 5)]:
            value = self.alignment.get(key, default)
            if (not np.isfinite(value) or value <= 0
                    or (key.endswith(('steps', 'samples')) and int(value) != value)):
                raise ValueError(f'alignment.{key} must be positive')
        self.alignment_hold_capabilities = []
        self.alignment_hold_active = False
        self.alignment_damping_stop = threading.Event()
        self.alignment_damping_producers = []
        self.alignment_damping_capabilities = []
        self.sample_rate_hz = float(self.control.get("sample_rate_hz", 100.0))
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("control.sample_rate_hz must be positive")
        self.leader_max_age_s = float(self.control.get("leader_max_age_s", 0.15))
        self.leader_startup_timeout_s = float(self.control.get('leader_startup_timeout_s', 2.))
        if not np.isfinite(self.leader_startup_timeout_s) or self.leader_startup_timeout_s <= 0:
            raise ValueError('control.leader_startup_timeout_s must be positive')
        self.state_max_age_s = float(self.control.get("state_max_age_s", 0.15))
        ddq_filter_alpha = float(self.control.get("ddq_filter_alpha", 1.0))
        if not np.isfinite(ddq_filter_alpha) or not 0.0 <= ddq_filter_alpha <= 1.0:
            raise ValueError("control.ddq_filter_alpha must be in [0, 1]")
        self.queue_size = int(self.control.get("sample_queue_size", 256))
        self.stop_event = threading.Event()
        self.follow_stop = threading.Event()
        self.leader_stop = threading.Event()
        self.sampling_stop = threading.Event()
        self.sampling_thread = None
        self.raw_sample_lock = threading.Lock()
        self.raw_samples = [None] * len(self.ARM_NAMES)
        self.raw_sequences = [0] * len(self.ARM_NAMES)
        self.command_lock = threading.RLock()
        self.alignment_target_q = [np.asarray(robot.reset_q, float) for _, robot, _ in self.configs]
        self.alignment_reference_raw = [np.asarray(leader.leader_reference_q, float) for _, _, leader in self.configs]
        self.state = "disconnected"
        self.exit_reset_attempted = False
        self.exit_reset_complete = False
        self.exit_hold_complete = False
        self.exit_reset_abort = threading.Event()
        self.arms = []
        self.readers = []
        self.mappers = []
        self.producers: list[_LeaderProducer] = []
        self.state_producers: list[_FollowerStateProducer] = []
        self.damping_capabilities: list[dict[str, Any]] = []
        self.followers: list[SecondOrderPositionFollower] = []
        self.last_q_cmd: list[np.ndarray | None] = [None] * len(self.ARM_NAMES)
        self.last_q_cmd_t = [0] * len(self.ARM_NAMES)
        self.last_q_cmd_seq = [0] * len(self.ARM_NAMES)
        self.last_q_cmd_ok = [True] * len(self.ARM_NAMES)
        self.command_history: list[deque] = [deque(maxlen=256) for _ in self.ARM_NAMES]
        self._last_follower_timestamp_us = [None] * len(self.ARM_NAMES)
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
        self.recording_segment_start_us = 0
        self.reference_capture_busy = False
        self.reference_publisher = ReferencePosePublisher(self.config_path, self.reference_pose_snapshot)
        self.reference_save_executor = None
        self.reference_save_future = None
        self.reference_save_pose = None
        visualizer = CameraVisualizer.from_config(self.collection_config.cameras)
        self.camera_manager = CameraManager.from_config(
            self.collection_config.cameras,
            visualizer=visualizer if visualizer.camera_names else None,
        )
        self.gripper_press_log_enabled = bool(self.raw.get('gripper', {}).get('print_on_press', True))
        self.gripper_enabled = bool(self.raw.get("gripper", {}).get("enabled", False))
        self.gripper_workers = []
        grip = self.raw.get('gripper', {})
        self.gripper_command_enabled = bool(grip.get('command_enabled', False))
        self.gripper_hz = float(grip.get('sample_rate_hz', 20.))
        self.gripper_speed = float(grip.get('max_speed_m_s', .02))
        self.gripper_mode = str(grip.get('mode', 'follow'))
        if self.gripper_mode not in {'trigger', 'follow'}:
            raise ValueError('gripper.mode must be trigger or follow')
        self.gripper_press_detectors = {side: (_GripperTriggerDetector() if self.gripper_mode == 'trigger'
                                               else _GripperPressDetector()) for side in self.ARM_NAMES}
        self.gripper_control_mode = str(grip.get('control_mode', 'absolute_position'))
        if self.gripper_control_mode not in {'absolute_position', 'rate_limited_position'}:
            raise ValueError('gripper.control_mode must be absolute_position or rate_limited_position')
        if not np.isfinite([self.gripper_hz, self.gripper_speed]).all() or not (
                0 < self.gripper_hz <= 30 and self.gripper_speed > 0):
            raise ValueError('gripper sample_rate_hz must be in (0,30] and max_speed_m_s positive')
        if self.gripper_command_enabled and (not self.gripper_enabled or any(
                leader.gripper_id < 0 for _, _, leader in self.configs)):
            raise ValueError('Gripper commands require enabled grippers and all active GELLO gripper IDs')
        if self.gripper_command_enabled and any(robot.gripper_type != 1 for _, robot, _ in self.configs):
            raise ValueError('This pipeline gripper command path currently supports xArm Gripper G1 only')

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
            self.arms.append(arm)
            arm.connect()
            arm.read_state()  # connection/feedback check before any motion
            reader = self._reader_factory(leader)
            mapper = JointMapper(robot, leader)
            # Only saved calibration is accepted.  JointMapper.align uses it
            # verbatim; it does not infer a new offset from the live pose.
            if not mapper.has_saved_calibration:
                raise RuntimeError(f"{side}: calibrated joint_offsets and leader_reference_q are required")
            self.readers.append(reader)
            self.mappers.append(mapper)
            self._observe_reader(len(self.readers)-1, reader)
        expected = {c.name for c in self.collection_config.cameras}
        available = {c.name for c in self.camera_manager.cameras}
        if self.raw.get('require_cameras', False) and expected != available:
            raise RuntimeError(f'Required cameras unavailable: {sorted(expected-available)}')
        self.camera_manager.start()
        if self.raw.get('require_cameras', False):
            received = set()
            deadline = time.monotonic()+5.
            while received != expected:
                received.update(f.camera_name for f in self.camera_manager.poll())
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Required cameras did not deliver frames: {sorted(expected-received)}')
                time.sleep(.01)
        self.state = "connected"

    def _observe_reader(self, index, reader):
        """Cache actual reads for the recorder, including reads during alignment."""
        original_read = reader.read
        def observed_read(*args, **kwargs):
            acquired_us = now_us()
            result = original_read(*args, **kwargs)
            values = np.asarray(result[0] if isinstance(result, tuple) else result, float).copy()
            with self.raw_sample_lock:
                self.raw_sequences[index] += 1
                self.raw_samples[index] = (values, acquired_us, now_us(), time.monotonic(), self.raw_sequences[index])
            return result
        reader.read = observed_read

    def reset(self, keys=None):
        if self.state != "connected":
            raise RuntimeError("both arms must pass connection check before reset")
        self.reset_followers(keys)
        self.align_gello_to_xarm([np.asarray(robot.reset_q) for _, robot, _ in self.configs], keys=keys)

    def reset_followers(self, keys=None, *, enable=True, consume_samples=True, abort=None):
        """Planned xArm reset; never touches GELLO or the recording flag."""
        def check_abort():
            if abort is not None and abort.is_set():
                raise InterruptedError('退出复位已取消')

        for arm in self.arms:
            check_abort()
            if enable:
                print(f'{arm.name} xArm：开始使能。', flush=True)
                arm.enable()
                print(f'{arm.name} xArm：使能完成。', flush=True)
            if consume_samples:
                self._consume_samples()
        reset_timeout = float(self.alignment.get("reset_timeout_s", 60.0))
        reset_speed = float(self.alignment.get("reset_speed_rad_s", 0.2))
        reset_acc = float(self.alignment.get("reset_acc_rad_s2", 0.5))
        for arm, (_, robot, _) in zip(self.arms, self.configs):
            check_abort()
            print(f'{arm.name} xArm：下发复位，速度 {np.rad2deg(reset_speed):.1f}°/s，'
                  f'目标°={np.round(np.rad2deg(robot.reset_q), 2).tolist()}。', flush=True)
            if hasattr(arm, "move_to_reset"):
                arm.move_to_reset(np.asarray(robot.reset_q), speed=reset_speed, acceleration=reset_acc)
            else:  # small mock adapters used by offline tests
                arm.command_joint_positions(np.asarray(robot.reset_q))
            if self.sampling_thread is not None:
                i = self.arms.index(arm)
                with self.command_lock:
                    self.last_q_cmd[i] = np.asarray(robot.reset_q).copy()
                    self.last_q_cmd_t[i] = now_us()
                    self.last_q_cmd_seq[i] += 1
                    self.command_history[i].append((self.last_q_cmd_t[i], self.last_q_cmd[i].copy(),
                                                    self.last_q_cmd_seq[i], True))
            print(f'{arm.name} xArm：复位指令已下发，等待到位。', flush=True)
        deadline = time.monotonic() + reset_timeout
        settled = [0] * len(self.ARM_NAMES)
        previous_timestamps = [None] * len(self.ARM_NAMES)
        tolerance = float(self.alignment.get("reset_tolerance_rad", 0.03))
        next_report = 0.
        errors = {}
        while time.monotonic() < deadline and min(settled) < int(self.alignment.get("reset_samples", 5)):
            check_abort()
            if consume_samples:
                self._consume_samples()
            if keys is not None and self._transition_key(keys, 0.) in {'q', 'Q', '\x03'}:
                raise KeyboardInterrupt
            for i, (arm, (_, robot, _)) in enumerate(zip(self.arms, self.configs)):
                state = arm.read_state()
                q = np.asarray(state.q, float)
                timestamp = int(getattr(state, 'q_timestamp_us', 0) or state.timestamp_us)
                if (q.shape != (arm.dof,) or not np.isfinite(q).all() or not state.q_valid
                        or not 0 <= now_us()-timestamp <= self.state_max_age_s*1e6):
                    raise RuntimeError(f'{arm.name} xArm 复位反馈采样无效或已过期')
                errors[arm.name] = np.round(np.rad2deg(q-np.asarray(robot.reset_q)), 2).tolist()
                if np.max(np.abs(q - np.asarray(robot.reset_q))) > tolerance:
                    settled[i] = 0
                elif timestamp != previous_timestamps[i]:
                    settled[i] += 1
                previous_timestamps[i] = timestamp
            report_due = time.monotonic() >= next_report
            if report_due or min(settled) >= int(self.alignment.get('reset_samples', 5)):
                statuses = {arm.name: arm.control_status() for arm in self.arms if hasattr(arm, 'control_status')}
                if report_due:
                    print(f'xArm 复位误差° J1..J7：{errors}；控制器状态：{statuses}', flush=True)
                for side, status in statuses.items():
                    if (status.get('error_code', 0)
                            or (status.get('state') is not None and status['state'] >= 3)
                            or status.get('mode') not in {None, 0}):
                        raise RuntimeError(f'{side} xArm 复位停止：{status}')
                    if status.get('state') == 1:
                        settled[self.ARM_NAMES.index(side)] = 0
                if report_due:
                    next_report = time.monotonic()+.5
            time.sleep(0.02)
        if min(settled) < int(self.alignment.get("reset_samples", 5)):
            raise TimeoutError(f"xArm reset did not settle: samples={settled}, errors_deg={errors}")
        check_abort()
        print('所选 xArm 已完成复位。', flush=True)

    def align_gello_to_xarm(self, target_q, keys=None):
        """Use the saved mapping to bring leaders to stationary followers."""
        if self.state not in {'connected', 'holding'}:
            raise RuntimeError('xArm must be stationary before GELLO alignment')
        self.alignment_target_q = [np.asarray(q, float).copy() for q in target_q]
        if len(self.alignment_target_q) != len(self.ARM_NAMES) or any(q.shape != (arm.dof,) or not np.isfinite(q).all()
                for q, arm in zip(self.alignment_target_q, self.arms)):
            raise ValueError('Invalid xArm alignment targets')
        self.alignment_reference_raw = [np.asarray(leader.joint_offsets) + q / np.asarray(leader.joint_signs)
                                       for q, (_, _, leader) in zip(self.alignment_target_q, self.configs)]
        for reader in self.readers:
            reader.prepare_alignment()
        current = self.alignment.get('gello_hold_current_raw')
        self.state = "aligning"
        if current is not None:
            self.alignment_hold_active = True
            self.alignment_hold_capabilities = []
            for reader in self.readers:
                self.alignment_hold_capabilities.append(reader.enable_alignment_hold(current))
                self._consume_samples()
            steps = int(self.alignment.get('gello_reset_steps', 15))
            interval = float(self.alignment.get('gello_reset_interval_s', .2))
            tracking_limit = np.deg2rad(self.alignment.get('gello_reset_tracking_error_deg', 10.))
            # Validate BOTH paths before advancing either leader.
            plans = []
            for reader, reference in zip(self.readers, self.alignment_reference_raw):
                plans.append(reader.alignment_reset_plan(reference, steps,
                             float(self.alignment.get('gello_reset_max_travel_deg', 90.))))
                self._consume_samples()
            print(f'GELLO 电流位置保持：{current} 原始单位；{steps} 次插值对齐 xArm 当前保持姿态。', flush=True)
            for step in range(steps):
                if keys is not None and self._transition_key(keys, 0.) in {'q', 'Q', '\x03'}:
                    raise KeyboardInterrupt
                for reader, plan in zip(self.readers, plans):
                    reader.command_alignment_positions(plan[step])
                deadline = time.monotonic()+interval
                while time.monotonic() < deadline:
                    self._consume_samples()
                    if keys is not None:
                        if self._transition_key(keys, min(.05, max(0., deadline-time.monotonic()))) in {'q', 'Q', '\x03'}:
                            raise KeyboardInterrupt
                    else:
                        time.sleep(min(.05, max(0., deadline-time.monotonic())))
                for reader, plan in zip(self.readers, plans):
                    actual = np.asarray(reader.read())[:len(reader.config.joint_ids)]
                    if np.max(np.abs(actual-plan[step])) > tracking_limit:
                        raise RuntimeError('GELLO 插值复位跟踪误差过大；检查支撑、线缆或电流保持能力')
                report = self.reference_error_report()
                errors = ' | '.join(side+': '+str(np.round(np.rad2deg(error), 1).tolist())
                                    for side, error in report.items())
                print(f'\r\033[2K复位 {step+1}/{steps} 误差° J1..J7 | {errors}', end='', flush=True)
            print(flush=True)
            self.start_alignment_damping()

    def start_alignment_damping(self):
        """Release reset position goals before automatically checking alignment."""
        for reader in self.readers:
            reader.set_torque(False, ids=list(reader.config.joint_ids))
        self.alignment_hold_active = False
        self.alignment_damping_stop.clear()
        self.alignment_damping_producers = []
        self.alignment_damping_capabilities = []
        for reader, (_, _, leader) in zip(self.readers, self.configs):
            if leader.damping_enabled and leader.damping_mode == 'current':
                mapper = _EncoderDampingMapper(leader)
                self.alignment_damping_capabilities.append(reader.enable_current_damping(mapper.config))
                producer = _LeaderProducer(reader, mapper, 1/leader.fps,
                                           self.alignment_damping_stop, mapper.config)
                self.alignment_damping_producers.append(producer)
                # Keep this bus alive while the other side is configured.
                producer.start()
            else:
                self.alignment_damping_capabilities.append({'enabled': False})
            self._consume_samples()
        print('插值复位结束，主手位置保持已解除，已恢复阻尼；开始自动采样核对对齐。', flush=True)

    def stop_alignment_damping_workers(self):
        self.alignment_damping_stop.set()
        for producer in self.alignment_damping_producers:
            if producer.thread is not None:
                producer.thread.join(timeout=1.)
                if producer.thread.is_alive():
                    raise RuntimeError('Alignment damping worker did not stop')
        self.alignment_damping_producers = []

    def alignment_report(self) -> dict[str, np.ndarray]:
        if self.state not in {"aligning", "aligned", "holding"}:
            raise RuntimeError("reset must complete before alignment check")
        report = {}
        for i, (side, robot, _) in enumerate(self.configs):
            raw = np.asarray(self.readers[i].read(), dtype=float)
            self.mappers[i].align(raw, reference_q=self.alignment_target_q[i])
            mapped, _ = self.mappers[i].target(raw)
            state = self.arms[i].read_state()
            actual = np.asarray(state.q, dtype=float)
            if (actual.shape != mapped.shape or not np.isfinite(actual).all()
                    or not getattr(state, 'q_valid', True)):
                raise RuntimeError(f'{side} xArm 对齐反馈采样无效')
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

    def reference_error_report(self):
        """Report errors to the current alignment target using the saved mapping."""
        if self.state not in {'aligning', 'aligned', 'holding'}:
            raise RuntimeError('reset must complete before reference error check')
        if any(p.errors for p in self.alignment_damping_producers):
            raise RuntimeError('主手对齐阻尼读写失败，停止接管')
        report = {}
        for i, (side, _, leader) in enumerate(self.configs):
            raw = np.asarray(self.readers[i].read(), dtype=float)[:len(leader.joint_ids)]
            if raw.shape != (len(leader.joint_ids),) or not np.isfinite(raw).all():
                raise ValueError(f'{side} GELLO 对齐关节采样无效')
            delta = raw-self.alignment_reference_raw[i]
            delta = (delta+np.pi) % (2*np.pi)-np.pi
            if not np.isfinite(delta).all():
                raise ValueError(f'{side} GELLO reference error is not finite')
            report[side] = delta*np.asarray(leader.joint_signs)
        return report

    def wait_for_alignment(self, keys=None):
        """Check a fixed batch of fresh reads; never wait for manual correction."""
        tolerance = float(self.alignment.get('alignment_tolerance_rad', .05))
        required = int(self.alignment.get('alignment_samples', 5))
        print(f'自动对齐核对：采样 {required} 次；误差超限或采样失败立即停止接管；q 退出。', flush=True)
        try:
            for sample_index in range(required):
                self._consume_samples()
                try:
                    report = self.reference_error_report()
                except Exception as exc:
                    raise RuntimeError(f'自动对齐采样失败（{sample_index+1}/{required}）：{exc}') from exc
                bad = {side: error for side, error in report.items()
                       if not np.isfinite(error).all() or np.any(np.abs(error) > tolerance)}
                values = ' | '.join(side+': ['+', '.join(f'{v:+.1f}' for v in np.rad2deg(error))+']'
                                    for side, error in report.items())
                status = '对齐失败' if bad else f'自动采样 {sample_index+1}/{required}'
                sys.stdout.write(f'\r\033[2K误差° J1..J7 | {values} | 阈值 {np.rad2deg(tolerance):.2f}° | {status}')
                sys.stdout.flush()
                if bad:
                    details = '; '.join(f'{side} J{axis+1}={np.rad2deg(error[axis]):+.2f}°'
                                        for side, error in bad.items()
                                        for axis in np.flatnonzero(~np.isfinite(error) | (np.abs(error) > tolerance)))
                    raise RuntimeError(f'自动对齐失败（采样 {sample_index+1}/{required}，'
                                       f'阈值 {np.rad2deg(tolerance):.2f}°）：{details}；停止接管')
                delay = 0. if sample_index+1 == required else .1
                if keys is None:
                    time.sleep(delay)
                    key = None
                else:
                    key = self._transition_key(keys, delay)
                if key in {'q', 'Q', '\x03'}:
                    raise KeyboardInterrupt
            # Recheck current actual follower positions before takeover.
            self.confirm_alignment()
            print('\n自动对齐核对通过，准备接管。', end='', flush=True)
        finally:
            print(flush=True)

    def takeover(self, keys=None):
        if self.state not in {"aligned", "holding"}:
            raise RuntimeError("alignment must be confirmed before takeover")
        if any(p.thread is not None and p.thread.is_alive() for p in self.producers):
            self.leader_stop.set()
            for producer in self.producers:
                producer.thread.join(timeout=1.)
                if producer.thread.is_alive():
                    raise RuntimeError('GELLO worker did not stop before takeover')
        self.stop_event.clear()
        self.leader_stop.clear()
        self.control_error = None
        self.last_q_cmd_ok = [True] * len(self.ARM_NAMES)
        self.followers.clear()
        for i, arm in enumerate(self.arms):
            actual = arm.read_state().q
            mapped, _ = self.mappers[i].target(self.readers[i].read())
            tolerance = float(self.alignment.get("alignment_tolerance_rad", 0.05))
            if np.max(np.abs(mapped - actual)) > tolerance:
                raise RuntimeError(f"{self.ARM_NAMES[i]} moved after alignment confirmation; takeover refused")
            self.mappers[i].last_raw = mapped.copy()
            self.mappers[i].last_target = actual.copy()
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
        self.producers = []
        self.damping_capabilities = []
        ddq_alpha = float(self.control.get("ddq_filter_alpha", 1.0))
        if self.gripper_enabled and not self.gripper_workers:
            for arm in self.arms:
                arm.init_gripper()
        if not self.state_producers:
            self.state_producers = [_FollowerStateProducer(arm, 1.0 / self.sample_rate_hz, self.stop_event, ddq_alpha, False)
                                    for arm in self.arms]
            for producer in self.state_producers:
                producer.start()
        deadline = time.monotonic() + 1.0
        while any(producer.snapshot() is None for producer in self.state_producers):
            if time.monotonic() >= deadline:
                raise RuntimeError("xArm state producers did not deliver initial feedback")
            time.sleep(0.001)
        if self.alignment_hold_active:
            # Release both position loops before switching either side to damping.
            for reader in self.readers:
                reader.set_torque(False, ids=list(reader.config.joint_ids))
            self.alignment_hold_active = False
        standby_active = bool(self.alignment_damping_producers)
        if standby_active:
            self.stop_alignment_damping_workers()
        for i, (reader, mapper) in enumerate(zip(self.readers, self.mappers)):
            damping_cfg = self.configs[i][2]
            damping = (damping_cfg if bool(getattr(damping_cfg, "damping_enabled", False)) and
                       str(getattr(damping_cfg, "damping_mode", "none")).lower() == "current" else None)
            if damping is not None:
                capability = (self.alignment_damping_capabilities[i] if standby_active
                              else reader.enable_current_damping(damping))
                self.damping_capabilities.append(capability)
                after_mode, _ = mapper.target(reader.read())
                current_actual = self.arms[i].read_state().q
                if np.max(np.abs(after_mode - current_actual)) > float(self.alignment.get("alignment_tolerance_rad", 0.05)):
                    error = np.round(np.rad2deg(after_mode-current_actual), 2).tolist()
                    raise RuntimeError(f"{self.ARM_NAMES[i]} 主从对齐偏差超限（非 xArm 控制器错误）："
                                       f"各轴误差°={error}；停止接管")
            else:
                self.damping_capabilities.append({"enabled": False})
            self.producers.append(_LeaderProducer(reader, mapper, 1.0 / float(mapper.config.fps),
                                                 self.leader_stop, damping, failure_stop=self.stop_event))
            self.producers[-1].start()
        self.wait_for_leader_samples(keys)
        # Keep the followers in their planned-position hold until leader I/O
        # is ready. Mode reports are asynchronous, so wait for both switches
        # before validating the final poses and sending any servo target.
        print('主手数据已就绪，切换所选 xArm 到遥操模式。', flush=True)
        for arm in self.arms:
            if hasattr(arm, 'set_normal_mode'):
                arm.set_normal_mode()
        actuals = [arm.read_state().q for arm in self.arms]
        samples = [producer.snapshot() for producer in self.producers]
        now = time.monotonic()
        if self.stop_event.is_set() or any(s is None or not s.valid or
                now-s.acquired_monotonic_s > self.leader_max_age_s for s in samples):
            raise RuntimeError(f'GELLO data unavailable before first servo target: {self.leader_sample_status()}')
        for i, (sample, actual) in enumerate(zip(samples, actuals)):
            if np.max(np.abs(sample.mapped-actual)) > tolerance:
                raise RuntimeError(f'{self.ARM_NAMES[i]} moved while waiting for takeover; takeover refused')
        for i, (arm, actual, follower) in enumerate(zip(self.arms, actuals, self.followers)):
            follower.initialize(actual)
            arm.command_joint_positions(actual)
            self.last_q_cmd[i] = actual.copy()
            self.last_q_cmd_t[i] = now_us()
            self.last_q_cmd_seq[i] = self.last_q_cmd_seq[i]+1 if self.command_history[i] else 0
            with self.command_lock:
                self.command_history[i].append((self.last_q_cmd_t[i], actual.copy(), self.last_q_cmd_seq[i], True))
        if self.gripper_enabled and self.gripper_workers:
            for worker, producer in zip(self.gripper_workers, self.producers):
                worker.producer = producer
        elif self.gripper_enabled:
            for side, arm, producer in zip(self.ARM_NAMES, self.arms, self.producers):
                cfg = self.raw.get('arms', {}).get(side, {})
                minimum = float(cfg.get('gripper_min_width_m', 0.))
                maximum = float(cfg.get('gripper_max_width_m', .085))
                if not np.isfinite([minimum, maximum]).all() or not 0 <= minimum < maximum:
                    raise ValueError('Invalid physical gripper width calibration')
                worker = _GripperWorker(arm, producer, self.stop_event, period_s=1/self.gripper_hz,
                                        maximum_age_s=self.leader_max_age_s,
                                        command_enabled=self.gripper_command_enabled,
                                        min_width=minimum, max_width=maximum, max_speed=self.gripper_speed,
                                        pause=self.follow_stop, control_mode=self.gripper_control_mode,
                                        mode=self.gripper_mode)
                self.gripper_workers.append(worker)
                worker.start()
        self.state = 'recording' if self.recording else 'following'
        self.follow_stop.clear()
        if self.sampling_thread is None:
            self.sampling_thread = threading.Thread(target=self._sampling_loop, name='teleop-data-100hz', daemon=True)
            self.sampling_thread.start()
        self.reference_publisher.start()
        self._last_control_t = time.monotonic()
        self.control_thread = threading.Thread(target=self._control_loop, name="xarm-100hz-control", daemon=True)
        self.control_thread.start()

    def leader_sample_status(self):
        now = time.monotonic()
        return {side: {'age_ms': None if (sample := producer.snapshot()) is None else
                        round((now-sample.acquired_monotonic_s)*1000, 1),
                       'sequence': None if sample is None else sample.sequence,
                       'read_ms': round(producer.last_read_s*1000, 1),
                       'io_ms': round(producer.last_io_s*1000, 1),
                       'errors': producer.errors, 'last_error': producer.last_error}
                for side, producer in zip(self.ARM_NAMES, self.producers)}

    def wait_for_leader_samples(self, keys=None):
        """Do not start xArm streaming until BOTH readers have fresh data."""
        deadline = time.monotonic()+self.leader_startup_timeout_s
        next_report = time.monotonic()+.5
        print('等待所选 GELLO 连续新样本，准备启动遥操。', flush=True)
        while True:
            if keys is not None and self._transition_key(keys, 0.) in {'q', 'Q', '\x03'}:
                raise KeyboardInterrupt
            now = time.monotonic()
            self._consume_samples()
            samples = [producer.snapshot() for producer in self.producers]
            if self.stop_event.is_set() or any(producer.errors for producer in self.producers):
                raise RuntimeError(f'GELLO 启动采样失败：{self.leader_sample_status()}')
            if len(samples) == len(self.ARM_NAMES) and all(s is not None and producer.sequence >= 2 and s.valid
                    and now-s.acquired_monotonic_s <= self.leader_max_age_s
                    for s, producer in zip(samples, self.producers)):
                print(f'所选 GELLO 采样已就绪：{self.leader_sample_status()}', flush=True)
                return
            if now >= deadline:
                raise TimeoutError(f'GELLO startup samples unavailable/stale: {self.leader_sample_status()}')
            if now >= next_report:
                print(f'仍在等待主手采样：{self.leader_sample_status()}', flush=True)
                next_report = now+.5
            time.sleep(.005)

    def start_episode(self):
        if self.recording:
            raise RuntimeError('An episode is already recording')
        if self.state not in {'following', 'holding', 'aligning', 'aligned'} or self.sampling_thread is None:
            raise RuntimeError("r is only accepted after alignment and takeover")
        self._drain_samples()
        # Drop camera frames queued before the operator pressed r; their
        # independent timestamps must not leak into this episode.
        while self.camera_manager.poll():
            pass
        self.buffer = EpisodeBuffer(self.collection_config, self.ARM_NAMES, enable_online_tau_ext=False)
        self.buffer.episode_metadata.update({"pipeline_state": "recording", "clock": "monotonic scheduler / unix host_receive",
                                             'teleop_events': [], 'gripper_control_mode': self.gripper_control_mode,
                                             'gripper_mode': self.gripper_mode,
                                             "arm_names": list(self.ARM_NAMES), "joint_order": "J1..J7 per arm",
                                             "leader_calibration": [self._calibration_snapshot(i) for i in range(len(self.ARM_NAMES))],
                                             "gello_damping_capabilities": self.damping_capabilities,
                                             "gello_alignment_hold": self.alignment_hold_capabilities,
                                             "gello_damping_settings": [{k: getattr(leader, k) for k in (
                                                 'damping_enabled', 'damping_mode', 'damping_gain',
                                                 'damping_brake_gain', 'damping_current_limit',
                                                 'damping_velocity_threshold', 'damping_velocity_filter_alpha',
                                                 'damping_watchdog_ms', 'weak_hold_enabled', 'weak_hold_gain',
                                                 'weak_hold_limit', 'weak_hold_release_velocity', 'fps')}
                                                 for _, _, leader in self.configs],
                                             "xarm_sdk_version": _sdk_version(),
                                             "xarm_firmware": [self._firmware_snapshot(arm) for arm in self.arms],
                                             "xarm_feedback_source": [getattr(state, "feedback_source", "unknown")
                                                                       for state in [p.snapshot() for p in self.state_producers]]})
        self.recording = True
        self.recording_started_t = time.monotonic()
        self.recording_segment_start_us = now_us()
        if self.state == 'following':
            self.state = 'recording'

    def _record_event(self, name, **details):
        if self.recording and self.buffer is not None:
            self.buffer.episode_metadata.setdefault('teleop_events', []).append({
                'event': name, 'timestamp_us': now_us(), 'state': self.state, **details})

    def freeze_following(self):
        """Stop follower/gripper targets, leaving GELLO sampling and damping intact."""
        if self.state not in {'following', 'recording'}:
            return
        self.follow_stop.set()
        for worker in self.gripper_workers:
            with worker.lock:
                pass  # wait for an in-flight gripper command; keep its feedback loop
        threads = [self.control_thread]
        for thread in threads:
            if thread is not None:
                thread.join(timeout=1.)
                if thread.is_alive():
                    self.stop_event.set()
                    raise RuntimeError('Follower control worker did not stop before position hold')
        actuals = [arm.read_state().q.copy() for arm in self.arms]
        for i, (arm, actual) in enumerate(zip(self.arms, actuals)):
            with self.command_lock:
                arm.command_joint_positions(actual)
                self.last_q_cmd[i] = actual.copy()
                self.last_q_cmd_t[i] = now_us()
                self.last_q_cmd_seq[i] += 1
                self.command_history[i].append((self.last_q_cmd_t[i], actual.copy(), self.last_q_cmd_seq[i], True))
        self._consume_samples()
        self.state = 'holding'
        self._record_event('f_hold')

    def realign_and_takeover(self, keys):
        """Reuse startup leader interpolation at the followers' CURRENT pose."""
        if self.state != 'holding':
            raise RuntimeError('Stop following before realignment')
        self.reference_capture_busy = True
        try:
            self.follow_stop.set()
            self.leader_stop.set()
            self.stop_alignment_damping_workers()
            threads = [self.control_thread] + [producer.thread for producer in self.producers]
            for thread in threads:
                if thread is not None:
                    thread.join(timeout=1.)
                    if thread.is_alive():
                        raise RuntimeError('Control worker did not stop before GELLO realignment')
            target_q = [arm.read_state().q.copy() for arm in self.arms]
            self._record_event('t_align_begin')
            self.align_gello_to_xarm(target_q, keys=keys)
            self.wait_for_alignment(keys)
            self.takeover(keys)
            self._record_event('t_takeover')
        finally:
            self.reference_capture_busy = False

    def reset_and_hold(self, keys):
        if self.state in {'following', 'recording'}:
            self.freeze_following()
        if self.state != 'holding':
            raise RuntimeError('Stop following before xArm reset')
        self.reference_capture_busy = True
        try:
            self._record_event('o_reset_begin')
            self.reset_followers(keys)
            self.state = 'holding'
            self._record_event('o_reset_done')
        finally:
            self.reference_capture_busy = False

    def reference_pose_snapshot(self):
        """Copy actual cached q for the read script without any hardware transaction."""
        pose = dict(valid=False, active_arms=list(self.ARM_NAMES), calibration_path=str(self.gello_config_path), state=self.state,
                    leader_max_age_s=self.leader_max_age_s, state_max_age_s=self.state_max_age_s)
        if self.stop_event.is_set() or self.control_error is not None:
            return dict(pose, reason='遥操进程已故障或停止')
        if self.reference_capture_busy or self.state not in {'following', 'recording', 'holding'}:
            return dict(pose, reason='请在遥操或 F 保持状态采样；当前正在复位或对齐')
        with self.raw_sample_lock:
            raw_samples = list(self.raw_samples)
        states = [producer.snapshot() for producer in self.state_producers]
        if len(states) != len(self.ARM_NAMES) or any(item is None for item in raw_samples+states):
            return dict(pose, reason='所选主从臂尚未提供完整关节反馈')
        arms = {}
        now = now_us()
        for index, ((side, robot, leader), raw_sample, state) in enumerate(zip(self.configs, raw_samples, states)):
            raw, acquired_us, _, acquired_t, sequence = raw_sample
            n = len(leader.joint_ids)
            q = np.asarray(state.q)
            raw_all = np.asarray(raw)
            raw = raw_all[:n]
            q_timestamp = int(getattr(state, 'q_timestamp_us', 0) or state.timestamp_us)
            if (q.shape != (n,) or raw.shape != (n,) or not np.isfinite(np.r_[q, raw]).all()
                    or not state.q_valid):
                return dict(pose, reason=f'{side} 关节反馈无效')
            if (time.monotonic()-acquired_t > self.leader_max_age_s or
                    now-q_timestamp > self.state_max_age_s*1e6):
                return dict(pose, reason=f'{side} 关节反馈已过期')
            arms[side] = dict(
                xarm_q=q.tolist(), gello_q=raw.tolist(), xarm_timestamp_us=q_timestamp,
                gello_timestamp_us=int(acquired_us), gello_sequence=int(sequence),
                identity=dict(robot_ip=robot.robot_ip, port=leader.port, baudrate=leader.baudrate,
                              joint_ids=list(leader.joint_ids), joint_signs=list(leader.joint_signs),
                              gripper_id=leader.gripper_id))
            mapper = self.mappers[index]
            fraction = (None if leader.gripper_id < 0 or raw_all.size <= n or
                        not np.isfinite(raw_all[n]) else
                        _gripper_fraction(raw_all[n], mapper.gripper_open, mapper.gripper_close))
            arm_cfg = self.raw.get('arms', {}).get(side, {})
            minimum = float(arm_cfg.get('gripper_min_width_m', 0.))
            maximum = float(arm_cfg.get('gripper_max_width_m', .085))
            grip = dict(mode=self.gripper_mode, gello_angle_deg=None if fraction is None else float(np.rad2deg(raw_all[n])),
                        open_deg=None if mapper.gripper_open is None else float(np.rad2deg(mapper.gripper_open)),
                        close_deg=None if mapper.gripper_close is None else float(np.rad2deg(mapper.gripper_close)),
                        closure_fraction=fraction,
                        desired_width_m=None if fraction is None else maximum-fraction*(maximum-minimum),
                        command_width_m=None, actual_width_m=None,
                        command_timestamp_us=0, actual_timestamp_us=0,
                        configured_speed_rpm=arm_cfg.get('gripper_speed'),
                        following=self.gripper_command_enabled and not self.follow_stop.is_set())
            if index < len(self.gripper_workers):
                actual, actual_valid, command, command_valid, command_us, actual_us = self.gripper_workers[index].snapshot(now)
                actual_fresh = 0 <= now-actual_us <= max(self.state_max_age_s, 2/self.gripper_hz)*1e6
                grip.update(actual_width_m=float(actual) if actual_valid and actual_fresh else None,
                            command_width_m=float(command) if command_valid and np.isfinite(command) else None,
                            actual_timestamp_us=int(actual_us), command_timestamp_us=int(command_us))
            if self.gripper_mode == 'trigger':
                grip['desired_width_m'] = grip['command_width_m']
            arms[side]['gripper'] = grip
        if self.reference_capture_busy or self.state not in {'following', 'recording', 'holding'}:
            return dict(pose, reason='采样期间复位或对齐流程已启动')
        return dict(pose, valid=True, arms=arms)

    def save_reference_pose(self):
        """Snapshot all four arms now; write in the background while teleoperation continues."""
        self._poll_reference_save()
        if self.reference_save_future is not None:
            print('上一份所选主从臂参考正在保存；完成后可再次按 s。', flush=True)
            return False
        pose = capture_current_pose(self.reference_pose_snapshot())
        if self.reference_save_executor is None:
            self.reference_save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='save-reference-pose')
        self.reference_save_pose = pose
        self.reference_save_future = self.reference_save_executor.submit(overwrite_reference_pose, pose, wait_for_lock=False)
        self._record_event('reference_pose_save_requested', captured_timestamp_us=pose['captured_timestamp_us'])
        print('已读取所选主从臂当前七轴 q，正在覆盖保存参考；遥操和录制继续。', flush=True)
        return True

    def _poll_reference_save(self):
        future = self.reference_save_future
        if future is None or not future.done():
            return
        pose = self.reference_save_pose
        self.reference_save_future = self.reference_save_pose = None
        try:
            destination = future.result()
        except Exception as exc:
            print(f'所选主从臂参考未保存：{exc}；遥操继续。', flush=True)
            self._record_event('reference_pose_save_failed', error=str(exc))
            return
        for side in self.ARM_NAMES:
            for device in ('xarm', 'gello'):
                q = pose['arms'][side][device+'_q']
                print(f'{side} {device} q_rad = {json.dumps(q)}', flush=True)
        print(f'已覆盖保存所选主从臂参考：{destination}；当前遥操继续，下次启动使用新位姿。', flush=True)
        self._record_event('reference_pose_saved', calibration_path=str(destination),
                           captured_timestamp_us=pose['captured_timestamp_us'],
                           reference_q={side: {device: pose['arms'][side][device+'_q'] for device in ('xarm', 'gello')}
                                        for side in self.ARM_NAMES})

    def _recording_key(self, key):
        if key in {'r', 'R'}:
            if self.recording:
                print('当前 episode 已在录制；空格停止并保存。', flush=True)
            elif self.state in {'following', 'holding'} or (
                    self.state in {'aligning', 'aligned'} and self.sampling_thread is not None):
                self.start_episode()
                print('episode recording', flush=True)
            else:
                print('请在启动接管完成后按 r 录制。', flush=True)
            return True
        if key == ' ':
            if self.recording:
                path = self.stop_episode(True, None)
                print(f'episode saved to {path}' if path else 'episode stopped (no samples)', flush=True)
            return True
        return False

    def _transition_key(self, keys, timeout):
        key = keys.read_key(timeout)
        return None if self._recording_key(key) else key

    def stop_episode(self, save: bool, index: int | None = None) -> Path | None:
        if not self.recording:
            raise RuntimeError("no recording episode is active")
        # Recording is independent of follower control, including F/T/O.
        self._consume_samples()
        self.recording = False
        if self.buffer is not None:
            elapsed = max(time.monotonic()-self.recording_started_t, 1e-6)
            self.buffer.episode_metadata.update({
                'recorded_duration_s': elapsed,
                "actual_recorded_hz": self.buffer.sample_count / elapsed,
                "control_tick_count": self.control_tick_count,
                "gello_reader_errors": [producer.errors for producer in self.producers],
                "xarm_state_reader_errors": [producer.errors for producer in self.state_producers],
                "queue_overflow_count": self.queue_overflow,
                "expired_target_count": self.stale_count,
            })
        if self.state == 'recording':
            self.state = 'following'
        if not save or self.buffer is None or self.buffer.sample_count == 0:
            return None
        output_dir = self.collection_config.output.directory
        index = next_episode_index(output_dir, self.collection_config.output.prefix) if index is None else index
        return self.buffer.save(episode_path(output_dir, self.collection_config.output.prefix, index))

    def _control_loop(self):
        ticker = FixedRateTicker(self.sample_rate_hz, float(self.control.get("maximum_lateness_s", 0.03)))
        try:
            self.control_started_t = time.monotonic()
            while not self.stop_event.is_set() and not self.follow_stop.is_set() and self.state in {"following", "recording"}:
                _, lateness = ticker.wait("dual xArm position control")
                if self.stop_event.is_set() or self.follow_stop.is_set():
                    break
                self.control_tick_count += 1
                now = time.monotonic()
                dt = max(now - (self._last_control_t or now), 1e-4)
                self._last_control_t = now
                leaders = tuple(p.snapshot() for p in self.producers)
                if any(s is None or now - s.acquired_monotonic_s > self.leader_max_age_s for s in leaders):
                    self.stale_count += 1
                    raise RuntimeError('GELLO target expired/disconnected; all active arms stopped accepting new targets; '
                                       f'details={self.leader_sample_status()}')
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
                for i, (arm, state, leader, follower) in enumerate(zip(self.arms, states, leaders, self.followers)):
                    if self.stop_event.is_set() or self.follow_stop.is_set():
                        break
                    target = follower.update(leader.mapped, state.q, dt)
                    try:
                        arm.command_joint_positions(target)
                    except Exception:
                        self.last_q_cmd_ok[i] = False
                        raise
                    with self.command_lock:
                        self.last_q_cmd[i] = target.copy()
                        self.last_q_cmd_t[i] = now_us()
                        self.last_q_cmd_seq[i] += 1
                        self.last_q_cmd_ok[i] = True
                        self.command_history[i].append((self.last_q_cmd_t[i], self.last_q_cmd[i].copy(),
                                                        self.last_q_cmd_seq[i], True))
        except BaseException as exc:
            self.control_error = exc
            self.stop_event.set()
            log.exception("dual xArm control loop stopped")

    def _sampling_loop(self):
        """Independent state capture; F/T/O never start or stop recording."""
        ticker = FixedRateTicker(self.sample_rate_hz, float(self.control.get('maximum_lateness_s', .03)))
        try:
            while not self.sampling_stop.is_set():
                _, lateness = ticker.wait('dual teleop data capture')
                with self.raw_sample_lock:
                    raw_samples = list(self.raw_samples)
                states = [producer.snapshot() for producer in self.state_producers]
                if any(s is None for s in raw_samples+states):
                    continue
                leaders = []
                now = time.monotonic()
                for raw_sample, mapper in zip(raw_samples, self.mappers):
                    raw, acquired_us, timestamp_us, acquired_t, sequence = raw_sample
                    mapped = (raw[:mapper.n]-mapper.offsets)*np.asarray(mapper.config.joint_signs)
                    fraction = (_gripper_fraction(raw[mapper.n], mapper.gripper_open, mapper.gripper_close)
                                if mapper.config.gripper_id >= 0 else np.nan)
                    valid = now-acquired_t <= self.leader_max_age_s and self.state not in {'aligning', 'aligned'}
                    leaders.append(LeaderSample(raw[:mapper.n].copy(), mapped, timestamp_us, acquired_us,
                                                acquired_t, sequence, valid, fraction,
                                                raw[mapper.n] if mapper.config.gripper_id >= 0 else np.nan))
                repeated_feedback = np.asarray([
                    self._last_follower_timestamp_us[i] == int(state.timestamp_us)
                    for i, state in enumerate(states)], dtype=np.uint8)
                self._last_follower_timestamp_us = [int(state.timestamp_us) for state in states]
                filtered_ddq = np.concatenate([
                    producer.filtered_ddq if producer.filtered_ddq is not None else np.asarray(state.ddq, float)
                    for producer, state in zip(self.state_producers, states)])
                # Select only successful commands sent before state acquisition.
                selected_commands = []
                with self.command_lock:
                    for i, state in enumerate(states):
                        state_time = int(getattr(state, 'acquired_timestamp_us', 0) or state.timestamp_us)
                        candidates = [item for item in self.command_history[i] if item[0] <= state_time]
                        selected_commands.append(candidates[-1] if candidates else
                                                 (self.last_q_cmd_t[i], self.last_q_cmd[i].copy(), self.last_q_cmd_seq[i], False))
                q_cmd = np.concatenate([item[1] for item in selected_commands])
                selected_ts = np.asarray([item[0] for item in selected_commands], dtype=np.int64)
                selected_ok = np.asarray([item[3] for item in selected_commands], dtype=np.uint8)
                selected_seq = np.asarray([item[2] for item in selected_commands], dtype=np.int64)
                grippers = [worker.snapshot(int(getattr(state, 'acquired_timestamp_us', 0) or state.timestamp_us))
                            for worker, state in zip(self.gripper_workers, states)] if self.gripper_enabled else [
                                (np.nan, False, np.nan, False, 0, 0)]*len(self.ARM_NAMES)
                row = ControlSample(now_us(), tuple(states), tuple(leaders), q_cmd,
                                    selected_ts, selected_ok, selected_seq, repeated_feedback, filtered_ddq,
                                    np.asarray([g[0] for g in grippers]), np.asarray([g[1] for g in grippers], dtype=np.uint8),
                                    int(lateness * 1e6), np.asarray([g[2] for g in grippers]),
                                    np.asarray([g[3] for g in grippers], dtype=np.uint8),
                                    np.asarray([g[4] for g in grippers], dtype=np.int64),
                                    np.asarray([g[5] for g in grippers], dtype=np.int64))
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
            self.sampling_stop.set()
            log.exception('dual teleop sampling loop stopped')

    def _consume_samples(self):
        while True:
            try:
                sample = self.sample_queue.get_nowait()
            except queue.Empty:
                break
            if self.buffer is not None:
                self.buffer.append_teleop(sample.timestamp_us, self._values(sample), store=self.recording)
        self._poll_cameras()

    def _poll_cameras(self):
        # Preview and draining continue even without recording/control rows.
        for frame in self.camera_manager.poll():
            if self.recording and self.buffer is not None and frame.timestamp_us >= self.recording_segment_start_us:
                self.buffer.append_camera(frame.camera_name, frame.timestamp_us, frame.frame, frame.depth)

    def _poll_gripper_press_status(self):
        if not self.gripper_enabled or not self.gripper_press_log_enabled:
            return
        pose = self.reference_pose_snapshot()
        if not pose.get('valid'):
            return
        for side in self.ARM_NAMES:
            grip = pose['arms'][side]['gripper']
            press_value = grip['closure_fraction'] if self.gripper_mode == 'trigger' else grip['gello_angle_deg']
            if self.gripper_press_detectors[side].update(press_value):
                status = dict(grip, gello_motor_id=pose['arms'][side]['identity']['gripper_id'])
                print(format_status({side: status}), flush=True)

    def poll(self):
        self._consume_samples()
        self._poll_gripper_press_status()
        self._poll_reference_save()
        for worker in self.gripper_workers:
            if worker.error is not None:
                raise RuntimeError(f'Gripper I/O failed: {worker.error}') from worker.error
        if self.control_error is not None:
            raise RuntimeError(str(self.control_error)) from self.control_error
        if self.stop_event.is_set() and any(producer.last_error for producer in self.producers):
            raise RuntimeError(f'GELLO I/O failed: {self.leader_sample_status()}')

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
            "q_follower_valid": ("validity", np.asarray([all(s.q_valid and sample.timestamp_us-
                int(getattr(s, 'acquired_timestamp_us', 0) or s.timestamp_us) <= self.state_max_age_s*1e6
                for s in states)], dtype=np.uint8)),
            "q_follower_repeated": ("validity", sample.q_follower_repeated),
            "q_cmd_timestamp_us": ("timestamp", sample.q_cmd_timestamp_us),
            "q_cmd_send_ok": ("validity", sample.q_cmd_ok),
            "q_cmd_sequence": ("sequence", sample.q_cmd_sequence),
            "sample_lateness_us": ("duration", np.asarray([sample.lateness_us], dtype=np.int64)),
            "gripper_follower": ("gripper", sample.gripper_follower),
            "gripper_cmd": ("gripper", sample.gripper_cmd),
            "gripper_follower_valid": ("validity", sample.gripper_follower_valid),
            "gripper_cmd_valid": ("validity", sample.gripper_cmd_valid),
            "gripper_cmd_timestamp_us": ("timestamp", sample.gripper_cmd_timestamp_us),
            "gripper_follower_timestamp_us": ("timestamp", sample.gripper_follower_timestamp_us),
            "gripper_leader_fraction": ("gripper_fraction", np.asarray([s.gripper_fraction for s in leaders])),
        }
        return values

    def _drain_samples(self):
        while True:
            try:
                self.sample_queue.get_nowait()
            except queue.Empty:
                return

    def _hold_both(self, *, hold_followers=True):
        """Stop all I/O workers and retain xArm enable during local hold."""
        success = True
        self.follow_stop.set()
        self.stop_event.set()
        self.leader_stop.set()
        self.sampling_stop.set()
        self.alignment_damping_stop.set()
        threads = [self.control_thread, self.sampling_thread]
        threads += [worker.thread for worker in (self.gripper_workers + self.producers
                    + self.state_producers + self.alignment_damping_producers)]
        for thread in threads:
            if thread is not None:
                # Ctrl+C may arrive after Thread() but before start().
                if thread.ident is not None:
                    thread.join(timeout=1.)
                if thread.is_alive():
                    success = False
                    log.error('退出前工作线程未停止：%s', thread.name)
        if all(p.thread is None or not p.thread.is_alive() for p in self.alignment_damping_producers):
            self.alignment_damping_producers = []
        if hold_followers:
            for i, arm in enumerate(self.arms):
                if not bool(getattr(arm, '_enabled', bool(self.command_history[i]))):
                    continue  # Read-only/partial connections must not enable motion.
                try:
                    if hasattr(arm, 'hold_position'):
                        arm.hold_position(speed=float(self.alignment.get('reset_speed_rad_s', .2)),
                                          acceleration=float(self.alignment.get('reset_acc_rad_s2', .5)))
                    else:  # offline adapters
                        state = arm.read_state()
                        if not state.q_valid or not np.isfinite(state.q).all():
                            raise RuntimeError('Position hold feedback is invalid')
                        arm.command_joint_positions(state.q)
                except Exception:
                    success = False
                    log.exception("failed to hold %s; motor enable was not cancelled", arm.name)
        for reader in self.readers:
            try:
                cfg = reader.config
                if (bool(getattr(cfg, "damping_enabled", False)) and
                        str(getattr(cfg, "damping_mode", "none")).lower() == "current"):
                    reader.disable_current_damping()
                reader.set_torque(False, ids=list(reader.config.joint_ids))
            except Exception:
                pass
        return success

    def shutdown_reset_and_hold(self):
        """Orderly q/Ctrl+C exit: reset both followers, then leave mode 0 active."""
        if self.exit_reset_attempted:
            return self.exit_reset_complete
        self.exit_reset_attempted = True
        if self.state in {'disconnected', 'stopped'} or len(self.arms) != len(self.ARM_NAMES):
            return True  # Connection/check-only exit: no motion has been authorized yet.
        faulted = self.stop_event.is_set()
        self.reference_capture_busy = True
        self._record_event('exit_reset_begin')
        print('退出：停止遥操，所选 xArm 返回 reset_q 后保留位置保持。', flush=True)
        try:
            self.exit_hold_complete = self._hold_both()
            if (faulted or self.control_error is not None or any(w.error for w in self.gripper_workers)
                    or not self.exit_hold_complete):
                raise RuntimeError('工作线程或设备已有故障，不能执行退出复位')
            if self.exit_reset_abort.is_set():
                raise InterruptedError('退出复位已取消')
            self.state = 'exiting'
            # Re-enable only if startup was interrupted before enabling both arms.
            enable = any(not bool(getattr(arm, '_enabled', True)) for arm in self.arms)
            self.exit_hold_complete = False
            self.reset_followers(enable=enable, consume_samples=False, abort=self.exit_reset_abort)
            self.exit_reset_complete = self.exit_hold_complete = True
            self._record_event('exit_reset_done')
            print('所选 xArm 已到 reset_q，控制器位置保持已保留；正在关闭连接。', flush=True)
            return True
        except (Exception, KeyboardInterrupt) as exc:
            self._record_event('exit_reset_failed', error=str(exc))
            print(f'退出复位未完成：{exc}；尝试在当前位置保持，不取消 xArm 使能。', flush=True)
            self.exit_hold_complete = self._hold_both()
            return False

    def close(self):
        if self.state == 'stopped':
            return
        try:
            self.reference_publisher.close()
        except Exception:
            log.exception('关节反馈缓存停止失败；继续关闭设备')
        # A successful exit reset already left the controller at its final
        # mode-0 target. Do not overwrite it with a mode-1 servo command.
        self._hold_both(hold_followers=not self.exit_hold_complete)
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
                # Closing transport retains the controller's position hold.
                # motion_enable(False) here would undo that hold.
                arm.disconnect()
            except Exception:
                log.exception("failed to disconnect xArm %s", arm.name)
        self.state = "stopped"
        if self.reference_save_executor is not None:
            self.reference_save_executor.shutdown(wait=True)
            self._poll_reference_save()
            self.reference_save_executor = None

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

    def interactive(self, auto_save=False, reset_q=False):
        with TerminalKeys() as keys:
            if not keys.is_tty:
                raise RuntimeError("interactive keyboard is unavailable; use --dry-run with mocks")
            print('连接检查完成，所选 xArm 自动低速返回各自 reset_q。', flush=True)
            self.reset(keys=keys)
            self.wait_for_alignment(keys)
            self.takeover()
            print("遥操已接管；r 录制，空格停止保存；F 保持，o 复位，t 对齐接管；q/Ctrl+C 复位保持后退出。", flush=True)
            if reset_q:
                print('参考位姿模式：遥操到希望的位置后按 s，覆盖保存所选主从臂当前 q；无需 F 或 Enter，可重复保存。', flush=True)
            while True:
                self.poll()
                key = keys.read_key(0.01)
                if key in {"q", "Q", "\x03"}:
                    break
                if self._recording_key(key):
                    continue
                if key in {'s', 'S'}:
                    if not reset_q:
                        print('保存所选主从臂参考需启动时添加 --reset-q。', flush=True)
                        continue
                    try:
                        self.save_reference_pose()
                    except (OSError, ValueError, RuntimeError, KeyError) as exc:
                        print(f'所选主从臂参考未保存：{exc}；遥操继续。', flush=True)
                    continue
                if key in {'f', 'F'} and self.state in {'following', 'recording'}:
                    self.freeze_following()
                    print('xArm 已保持当前位置；GELLO 阻尼、录制状态保持不变；t 对齐后接管。', flush=True)
                    continue
                if key in {'o', 'O'} and self.state in {'following', 'recording', 'holding'}:
                    self.reset_and_hold(keys)
                    print('xArm 已复位；GELLO 和录制状态保持不变；按 t 对齐接管。', flush=True)
                    continue
                if key in {"t", "T"} and self.state == "holding":
                    self.realign_and_takeover(keys)
                    print('遥操已重新接管；录制持续进行。' if self.recording else
                          '遥操已重新接管；按 r 开始新的 episode。', flush=True)
                    continue
        return 0


def _gripper_fraction(raw, opening, closing):
    delta = (float(raw)-float(opening)+np.pi) % (2*np.pi)-np.pi
    return float(np.clip(delta/(float(closing)-float(opening)), 0., 1.))


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


@contextmanager
def _exit_interrupt_handler(pipeline):
    """A second Ctrl+C cancels reset without interrupting hold/cleanup midway."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    def cancel_reset(_signum, _frame):
        pipeline.exit_reset_abort.set()
    signal.signal(signal.SIGINT, cancel_reset)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--auto-save", action="store_true")
    parser.add_argument("--reset-q", action="store_true", help="照常启动遥操；按 s 覆盖保存所选主从臂当前 q 为下次启动参考，保存后继续遥操")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pipeline = DualGelloPipeline(args.config)
    user_exit = False
    exit_code = 0
    try:
        if args.check_config:
            print(json.dumps({"state": pipeline.state, "arm_names": list(pipeline.ARM_NAMES),
                              "sample_rate_hz": pipeline.sample_rate_hz,
                              "gello_config": pipeline.raw.get("gello_config")}, indent=2))
            return 0
        pipeline.connect()
        exit_code = pipeline.interactive(auto_save=args.auto_save, reset_q=args.reset_q)
        user_exit = True
    except KeyboardInterrupt:
        user_exit = True
        exit_code = 130
    finally:
        with _exit_interrupt_handler(pipeline):
            try:
                if user_exit and not pipeline.shutdown_reset_and_hold():
                    exit_code = 1
            finally:
                pipeline.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
