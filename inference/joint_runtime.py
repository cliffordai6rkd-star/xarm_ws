"""Camera acquisition, asynchronous policy inference and xArm q servo execution."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from inference.core.contracts import ActionChunk
from nero_collection.cameras import CameraManager, CameraVisualizer
from nero_collection.time_utils import now_us
from ufactory_devices.robot.position_reference import SecondOrderPositionFollower

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JointObservation:
    timestamp_us: int
    q: np.ndarray
    images: dict
    image_timestamps_us: dict
    metadata: dict = field(default_factory=dict)


def state_time(state):
    return int(state.q_acquired_timestamp_us or state.acquired_timestamp_us or state.timestamp_us)


class StateReader:
    def __init__(self, arm, period_s):
        self.arm, self.period_s = arm, period_s
        self.history = deque(maxlen=512)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name=f"policy-state-{arm.name}", daemon=True)

    def _run(self):
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                state = self.arm.read_state()
                if not state.q_valid or np.asarray(state.q).shape != (7,) or not np.isfinite(state.q).all():
                    raise RuntimeError(f"Invalid xArm joint feedback on {self.arm.name}")
                with self.lock:
                    self.history.append(state)
                self.stop.wait(max(0., self.period_s - (time.monotonic() - started)))
        except Exception as exc:
            self.error = exc

    def snapshot(self):
        if self.error:
            raise RuntimeError(f"xArm state read failed on {self.arm.name}") from self.error
        with self.lock:
            return tuple(self.history)


class PolicyWorker:
    """One running prediction and one replaceable pending observation window."""
    def __init__(self, policy):
        self.policy = policy
        self.condition = threading.Condition()
        self.pending = self.result = self.error = None
        self.generation = 0
        self.stopped = False
        self.thread = threading.Thread(target=self._run, name="joint-policy", daemon=True)

    def submit(self, observations):
        with self.condition:
            self.pending = (self.generation, observations)
            self.condition.notify()

    def cancel(self):
        with self.condition:
            self.generation += 1
            self.pending = self.result = self.error = None

    def take_result(self):
        with self.condition:
            if self.error:
                raise RuntimeError("Policy inference failed") from self.error
            result, self.result = self.result, None
            return result

    def _run(self):
        previous_generation = None
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or self.pending is not None)
                if self.stopped:
                    return
                generation, observations = self.pending
                self.pending = None
            try:
                if generation != previous_generation:
                    self.policy.reset_episode()
                    previous_generation = generation
                started = time.monotonic()
                chunk = self.policy.predict_observations(observations)
                elapsed_ms = (time.monotonic() - started) * 1000
                with self.condition:
                    if generation == self.generation and not self.stopped:
                        self.result = (chunk, elapsed_ms)
            except Exception as exc:
                with self.condition:
                    if generation == self.generation and not self.stopped:
                        self.error = exc

    def close(self):
        with self.condition:
            self.stopped = True
            self.pending = None
            self.condition.notify()
        if self.thread.ident is not None:
            self.thread.join(timeout=2.)


class JointInferenceRuntime:
    def __init__(self, config, policy, camera_map, *, backend="xarm", command_enabled=False,
                 arms=None, camera_manager=None):
        if backend not in {"xarm", "mock"}:
            raise ValueError("backend must be xarm or mock")
        self.config, self.policy, self.camera_map = config, policy, dict(camera_map)
        self.command_enabled = bool(command_enabled)
        if arms is None:
            from nero_collection.arms.mock import MockArm
            from ufactory_devices.robot.xarm_adapter import XArmAdapter

            adapter = MockArm if backend == "mock" else XArmAdapter
            arms = {side: adapter(replace(endpoint, config_kwargs={
                **endpoint.config_kwargs, "execution_enabled": self.command_enabled}))
                for side, endpoint in config.endpoints.items()}
        self.arms = arms
        required = set(self.camera_map.values())
        camera_configs = tuple(replace(camera, backend="mock", visualize=False) if backend == "mock"
                               else camera for camera in config.cameras if camera.name in required)
        self.cameras = camera_manager or CameraManager.from_config(
            camera_configs, visualizer=CameraVisualizer.from_config(camera_configs))
        if {camera.name for camera in self.cameras.cameras} != required:
            raise ValueError("Cannot construct all checkpoint camera sources")
        period = 1. / float(config.execution["control_hz"])
        self.readers = {side: StateReader(arm, period) for side, arm in self.arms.items()}
        self.frames = {name: deque(maxlen=256) for name in required}
        self.observations = deque(maxlen=max(256, policy.n_obs_steps * 4))
        self.worker = PolicyWorker(policy)
        control = dict(config.hardware.get("control", {}))
        control.update(config.execution.get("position_reference", {}))
        self.references = {side: SecondOrderPositionFollower(
            7, mode=control.get("position_mode", "direct"), kp=control.get("kp", 18.),
            kd=control.get("kd", 3.), max_velocity=control.get("max_velocity_rad_s", 1.5),
            max_acceleration=control.get("max_acceleration_rad_s2", 8.),
            max_step=control.get("max_step_rad", .01),
            max_tracking_error=control.get("max_tracking_error_rad", .1),
            joint_limits=config.joint_limits.get(side)) for side in self.arms}
        self.connected = []
        self.enabled = set()
        self.plan = None
        self.plan_started = 0.
        self.predictions = self.commands = 0
        self.last_target = None
        self.inference_ms = None

    def _states(self, current_us):
        histories = {side: reader.snapshot() for side, reader in self.readers.items()}
        current_us = max(current_us, now_us())
        for side, history in histories.items():
            if history and not 0 <= current_us - state_time(history[-1]) <= self.config.observation["maximum_state_age_s"] * 1e6:
                raise RuntimeError(f"xArm feedback expired on {side}")
        return histories

    def _observe(self, histories, current_us):
        for frame in self.cameras.poll():
            if frame.camera_name in self.frames:
                history = self.frames[frame.camera_name]
                if not history or frame.timestamp_us > history[-1].timestamp_us:
                    history.append(frame)
        current_us = max(current_us, now_us())
        settings = self.config.observation
        for name, history in self.frames.items():
            if history and not 0 <= current_us - history[-1].timestamp_us <= settings["maximum_camera_age_s"] * 1e6:
                raise RuntimeError(f"Checkpoint camera expired: {name}")
        if any(not history for history in (*self.frames.values(), *histories.values())):
            return None
        anchor = min(history[-1].timestamp_us for history in self.frames.values())
        if self.observations and anchor <= self.observations[-1].timestamp_us:
            return None
        selected_frames = {}
        for key, name in self.camera_map.items():
            frame = next((frame for frame in reversed(self.frames[name]) if frame.timestamp_us <= anchor), None)
            if frame is None or anchor - frame.timestamp_us > settings["maximum_camera_skew_s"] * 1e6:
                return None
            selected_frames[key] = frame
        selected_states = {}
        for side in self.config.active_arms:
            state = next((state for state in reversed(histories[side]) if state_time(state) <= anchor), None)
            if state is None or anchor - state_time(state) > settings["maximum_alignment_gap_s"] * 1e6:
                return None
            selected_states[side] = state
        low_dim = {}
        for key, source in self.config.observation.get("low_dim_sources", {}).items():
            if key not in getattr(self.policy, "low_dim_shapes", {}):
                continue
            values = []
            for state in selected_states.values():
                valid = getattr(state, "torque_valid" if source == "torque" else f"{source}_valid", True)
                if not valid or not np.isfinite(getattr(state, source)).all():
                    raise RuntimeError(f"Required policy state {key!r} is unavailable")
                values.append(getattr(state, source))
            low_dim[key] = np.concatenate(values)
        observation = JointObservation(
            anchor, np.concatenate([state.q for state in selected_states.values()]),
            {key: frame.frame for key, frame in selected_frames.items()},
            {key: frame.timestamp_us for key, frame in selected_frames.items()},
            {"policy_state": low_dim})
        self.observations.append(observation)
        # Select causal samples at the checkpoint's training timestep, then pad startup.
        window = []
        for offset in range(self.policy.n_obs_steps - 1, -1, -1):
            timestamp = anchor - round(offset * self.policy.step_s * 1e6)
            window.append(next((item for item in reversed(self.observations) if item.timestamp_us <= timestamp),
                               self.observations[0]))
        return tuple(window)

    def _install(self, chunk, inference_ms, current_us, monotonic_s):
        if not isinstance(chunk, ActionChunk) or chunk.semantic != "joint":
            raise ValueError("xArm executor accepts absolute joint ActionChunk only")
        if chunk.values.shape[1] != len(self.config.active_arms) * 7:
            raise ValueError(f"Wrong joint action shape: {chunk.values.shape}")
        if chunk.step_s is None or not np.isclose(chunk.step_s, self.policy.step_s):
            raise ValueError("Action timing does not match the checkpoint contract")
        for i, side in enumerate(self.config.active_arms):
            limits = self.config.joint_limits.get(side)
            if limits is not None and (np.any(chunk.values[:, 7*i:7*i+7] < limits[:, 0])
                                       or np.any(chunk.values[:, 7*i:7*i+7] > limits[:, 1])):
                raise ValueError(f"Policy q is outside joint limits on {side}")
        if not 0 <= current_us - chunk.timestamp_us <= self.config.execution["maximum_prediction_age_s"] * 1e6:
            raise RuntimeError("Policy result expired before execution")
        self.plan, self.plan_started = chunk, monotonic_s
        self.predictions += 1
        self.inference_ms = inference_ms

    def _execute(self, histories, current_us, monotonic_s, dt):
        if self.plan is None:
            return
        if not 0 <= current_us - self.plan.timestamp_us <= self.config.execution["maximum_prediction_age_s"] * 1e6:
            raise RuntimeError("Policy target expired; stopping xArm")
        index = min(int(max(0., monotonic_s - self.plan_started) / self.plan.step_s),
                    len(self.plan.values) - 1)
        target = self.plan.values[index]
        self.last_target = target.copy()
        if not self.command_enabled:
            return
        # Check every side before sending any command.
        if any(not histories[side] for side in self.config.active_arms):
            raise RuntimeError("Missing joint feedback before command")
        just_enabled = False
        for side in self.config.active_arms:
            if side not in self.enabled:
                self.enabled.add(side)  # Cleanup also covers a partially failed enable().
                self.arms[side].enable()
                just_enabled = True
        if just_enabled:
            # Mode switching may wait for SDK reports. Refresh feedback before the first q.
            current_us = now_us()
            histories = self._states(current_us)
            self._observe(histories, current_us)
            if current_us - self.plan.timestamp_us > self.config.execution["maximum_prediction_age_s"] * 1e6:
                raise RuntimeError("Policy target expired during xArm enable")
            for side in self.config.active_arms:
                self.references[side].initialize(histories[side][-1].q)
        commands = {}
        for i, side in enumerate(self.config.active_arms):
            actual = histories[side][-1].q
            commands[side] = self.references[side].update(target[7*i:7*i+7], actual, dt)
        for side, q in commands.items():
            self.arms[side].command_joint_positions(q)
        self.commands += 1

    def _stop_motion(self):
        first_error = None
        for side in tuple(self.enabled):
            try:
                stop = getattr(self.arms[side], "stop_motion", None)
                if stop:
                    stop()
                else:  # Mock adapter: freeze at its measured current position.
                    self.arms[side].command_joint_positions(self.arms[side].read_state().q)
            except Exception as exc:
                log.exception("Failed to stop xArm %s", side)
                first_error = first_error or exc
        self.enabled.clear()
        if first_error:
            raise RuntimeError("Failed to stop one or more xArms") from first_error

    def pause(self):
        self.worker.cancel()
        self.plan = None
        self.observations.clear()
        self._stop_motion()

    def run(self, duration=None, *, read_key=None, single_step=False):
        paused, one_chunk, submitted_once = bool(single_step), False, False
        in_flight, next_plan, latest_window = False, None, None
        period = 1. / self.config.execution["control_hz"]
        started = previous = next_tick = time.monotonic()
        last_report = started
        active_started = started
        try:
            self.policy.start()
            self.cameras.start()
            for side, arm in self.arms.items():
                self.connected.append(side)
                arm.connect()
                self.readers[side].thread.start()
            self.worker.thread.start()
            started = previous = next_tick = active_started = time.monotonic()
            while duration is None or time.monotonic() - started < duration:
                wall = time.monotonic()
                current_us = now_us()
                dt, previous = max(1e-4, min(wall - previous, period * 2)), wall
                key = read_key(0.) if read_key else None
                if key in {"q", "\x03"}:
                    break
                if key in {"p", "s", "c"}:
                    self.pause()
                    paused, one_chunk = key == "p", key == "s"
                    submitted_once = False
                    in_flight, next_plan, latest_window = False, None, None
                    active_started = wall
                    log.info("%s", "Paused" if paused else "One chunk" if one_chunk else "Continuous inference")
                histories = self._states(current_us)
                window = self._observe(histories, current_us)
                if window:
                    latest_window = window
                if not paused:
                    remaining = (0. if self.plan is None else len(self.plan.values) * self.plan.step_s
                                 - (wall - self.plan_started))
                    prefetch_s = max(.05, (self.inference_ms or 0.) / 1000. + period)
                    if (latest_window and not in_flight and next_plan is None
                            and remaining <= prefetch_s and (not one_chunk or not submitted_once)):
                        self.worker.submit(latest_window)
                        in_flight = True
                        submitted_once = True
                    result = self.worker.take_result()
                    if result:
                        in_flight = False
                        next_plan = result
                    if next_plan and remaining <= 0.:
                        self._install(*next_plan, current_us, wall)
                        next_plan = None
                    if self.plan is None:
                        if wall - active_started > self.config.execution["startup_timeout_s"]:
                            raise RuntimeError("Timed out waiting for cameras, aligned states and first policy output")
                    else:
                        self._execute(histories, current_us, wall, dt)
                        if one_chunk and wall - self.plan_started >= len(self.plan.values) * self.plan.step_s:
                            self.pause()
                            paused = True
                if wall - last_report >= 1.:
                    log.info("policy=%s predictions=%d command_ticks=%d infer_ms=%s target_q=%s",
                             self.config.policy["type"], self.predictions, self.commands,
                             None if self.inference_ms is None else round(self.inference_ms, 1),
                             None if self.last_target is None else np.round(self.last_target, 3).tolist())
                    last_report = wall
                next_tick += period
                if next_tick < time.monotonic():
                    next_tick = time.monotonic()
                time.sleep(max(0., next_tick - time.monotonic()))
        finally:
            # Stop commands first, including when only one side of a dual command failed.
            try:
                self._stop_motion()
            finally:
                self.worker.close()
                for reader in self.readers.values():
                    reader.stop.set()
                for reader in self.readers.values():
                    if reader.thread.ident is not None:
                        reader.thread.join(timeout=1.)
                for side in reversed(self.connected):
                    self.arms[side].disconnect()
                self.cameras.stop()
                self.policy.close()
        return self.predictions
