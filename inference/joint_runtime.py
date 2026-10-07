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
from inference.policies.errors import JointActionContractError
from inference.interactive import InferenceCommand, decode_inference_key
from nero_collection.cameras import CameraManager, CameraVisualizer
from nero_collection.time_utils import now_us
from ufactory_devices.robot.position_reference import SecondOrderPositionFollower

log = logging.getLogger(__name__)

# Device-to-host clock mapping can place an acquired frame slightly ahead of
# wall time. Wait for these frames; never submit a future policy observation.
CAMERA_CLOCK_LEAD_TOLERANCE_US = 5_000


class PolicyResultExpired(RuntimeError):
    """A valid policy result arrived too late for its source observation."""


@dataclass(frozen=True)
class JointObservation:
    timestamp_us: int
    q: np.ndarray
    images: dict
    image_timestamps_us: dict
    metadata: dict = field(default_factory=dict)


def state_time(state):
    # xArm reports a host-receive timestamp as well as the SDK request start.
    # A sample becomes available only when the joint reply arrives; using the
    # request start counts SDK latency as if newly received q were already old
    # and can select a reply that had not arrived at the camera anchor.
    if getattr(state, "timestamp_source", None) == "host_receive":
        stamp = getattr(state, "q_timestamp_us", 0) or state.timestamp_us
        if stamp:
            return int(stamp)
    return int(state.q_acquired_timestamp_us or state.acquired_timestamp_us or state.timestamp_us)


class StateReader:
    def __init__(self, arm, period_s):
        self.arm, self.period_s = arm, period_s
        self.history = deque(maxlen=512)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.read_started_s = self.read_finished_s = self.last_read_ms = None
        self.reads = 0
        self.thread = threading.Thread(target=self._run, name=f"policy-state-{arm.name}", daemon=True)

    def _run(self):
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                with self.lock:
                    self.read_started_s = started
                read = getattr(self.arm, "read_joint_state", None)
                state = read() if callable(read) else self.arm.read_state()
                if not state.q_valid or np.asarray(state.q).shape != (7,) or not np.isfinite(state.q).all():
                    raise RuntimeError(f"Invalid xArm joint feedback on {self.arm.name}")
                with self.lock:
                    self.read_finished_s = time.monotonic()
                    self.last_read_ms = (self.read_finished_s - started) * 1000.
                    self.reads += 1
                    self.history.append(state)
                self.stop.wait(max(0., self.period_s - (time.monotonic() - started)))
        except Exception as exc:
            self.error = exc

    def snapshot(self):
        if self.error:
            raise RuntimeError(f"xArm state read failed on {self.arm.name}") from self.error
        with self.lock:
            return tuple(self.history)

    def diagnostics(self):
        with self.lock:
            in_flight = (self.read_started_s is not None and
                         (self.read_finished_s is None or self.read_started_s > self.read_finished_s))
            return {"reads": self.reads,
                    "last_read_ms": None if self.last_read_ms is None else round(self.last_read_ms, 1),
                    "read_in_flight_ms": round((time.monotonic() - self.read_started_s) * 1000., 1)
                                         if in_flight else None,
                    "reader_alive": self.thread.is_alive()}


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
                if isinstance(self.error, JointActionContractError):
                    raise self.error
                raise RuntimeError(f"Policy inference failed: {self.error}") from self.error
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
        self.reset_progress = None
        self.plan = None
        self.plan_started = 0.
        self.predictions = self.commands = 0
        self.requests = 0
        self.last_target = None
        self.inference_ms = None
        # Exclude cold-start warmup, but retain recent end-to-end request times
        # so a fast response does not erase the budget for network/GPU jitter.
        self.prediction_latencies_s = deque(maxlen=8)
        self.observation_wait = "waiting for camera and xArm samples"
        self.execution_observation_after_us = 0
        self.prediction_status = {}
        self.cycle_timing = None

    @property
    def open_loop(self):
        return self.config.execution.get("open_loop", False)

    def _states(self, current_us):
        histories = {side: reader.snapshot() for side, reader in self.readers.items()}
        current_us = max(current_us, now_us())
        for side, history in histories.items():
            if history and not 0 <= current_us - state_time(history[-1]) <= self.config.observation["maximum_state_age_s"] * 1e6:
                state = history[-1]
                raise RuntimeError(f"xArm feedback expired on {side}: "
                                   f"age_ms={(current_us - state_time(state)) / 1000.:.1f}, "
                                   f"maximum_age_ms={self.config.observation['maximum_state_age_s'] * 1000.:.1f}, "
                                   f"timestamp_source={getattr(state, 'timestamp_source', 'unknown')}, "
                                   f"reader={self.readers[side].diagnostics()}")
        return histories

    def _observe(self, histories, current_us):
        for frame in self.cameras.poll():
            if frame.camera_name in self.frames:
                if frame.timestamp_us < self.execution_observation_after_us:
                    # Acquisition continues during SDK setup. IPC can still
                    # deliver frames queued before startup or a mode switch.
                    continue
                history = self.frames[frame.camera_name]
                if not history or frame.timestamp_us > history[-1].timestamp_us:
                    history.append(frame)
        current_us = max(current_us, now_us())
        settings = self.config.observation
        future_cameras = []
        for name, history in self.frames.items():
            if not history:
                continue
            age_us = current_us - history[-1].timestamp_us
            if age_us < -CAMERA_CLOCK_LEAD_TOLERANCE_US or age_us > settings["maximum_camera_age_s"] * 1e6:
                age_ms = age_us / 1000.
                raise RuntimeError(
                    f"Checkpoint camera expired: {name}; age_ms={age_ms:.1f}, "
                    f"maximum_age_ms={settings['maximum_camera_age_s'] * 1000.:.1f}")
            if age_us < 0:
                future_cameras.append(name)
        if future_cameras:
            self.observation_wait = f"waiting for host clock to reach camera timestamp: {sorted(future_cameras)}"
            return None
        if any(not history for history in (*self.frames.values(), *histories.values())):
            missing_cameras = [name for name, history in self.frames.items() if not history]
            missing_arms = [side for side, history in histories.items() if not history]
            self.observation_wait = f"missing cameras={missing_cameras}, xArm states={missing_arms}"
            return None
        anchor = min(history[-1].timestamp_us for history in self.frames.values())
        if anchor < self.execution_observation_after_us:
            self.observation_wait = "waiting for a camera frame after xArm mode switch"
            return None
        if self.observations and anchor <= self.observations[-1].timestamp_us:
            self.observation_wait = "waiting for a new synchronized camera frame"
            return None
        selected_frames = {}
        for key, name in self.camera_map.items():
            frame = next((frame for frame in reversed(self.frames[name]) if frame.timestamp_us <= anchor), None)
            if frame is None or anchor - frame.timestamp_us > settings["maximum_camera_skew_s"] * 1e6:
                self.observation_wait = f"waiting for camera alignment: {name}"
                return None
            if frame.timestamp_us < self.execution_observation_after_us:
                self.observation_wait = f"waiting for post-enable camera frame: {name}"
                return None
            selected_frames[key] = frame
        selected_states = {}
        for side in self.config.active_arms:
            state = next((state for state in reversed(histories[side]) if state_time(state) <= anchor), None)
            if state is None or anchor - state_time(state) > settings["maximum_alignment_gap_s"] * 1e6:
                self.observation_wait = f"waiting for xArm/camera alignment: {side}"
                return None
            if state_time(state) < self.execution_observation_after_us:
                self.observation_wait = f"waiting for post-enable xArm state: {side}"
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
        self.observation_wait = "aligned observation ready"
        # Select causal samples at the checkpoint's training timestep, then pad startup.
        window = []
        for offset in range(self.policy.n_obs_steps - 1, -1, -1):
            timestamp = anchor - round(offset * self.policy.step_s * 1e6)
            window.append(next((item for item in reversed(self.observations) if item.timestamp_us <= timestamp),
                               self.observations[0]))
        return tuple(window)

    def _validate_chunk(self, chunk, inference_ms, current_us):
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
        # Polling cameras and collecting a fast worker result can advance the
        # observation timestamp beyond the clock sampled at the loop's start.
        current_us = max(current_us, now_us())
        age_ms = (current_us - chunk.timestamp_us) / 1000.
        limit_ms = self.config.execution["maximum_prediction_age_s"] * 1000.
        if age_ms < 0 or (not self.open_loop and age_ms > limit_ms):
            error = PolicyResultExpired if age_ms > limit_ms else RuntimeError
            raise error("Policy result expired before execution: "
                        f"age_ms={age_ms:.1f}, maximum_age_ms={limit_ms:.1f}, "
                        f"infer_ms={inference_ms:.1f}")

    def _install(self, chunk, inference_ms, current_us, monotonic_s, *, timing=None):
        self._validate_chunk(chunk, inference_ms, current_us)
        self._finish_cycle("replaced", time.monotonic())
        self.plan, self.plan_started = chunk, max(monotonic_s, time.monotonic())
        self.predictions += 1
        self.inference_ms = inference_ms
        self.cycle_timing = timing

    def _log_timing(self, phase, timing, chunk, inference_ms, *, execute_ms=0., setup_ms=0., status="complete"):
        if not self.config.execution.get("log_timing", False) or timing is None:
            return
        details = chunk.metadata.get("timing", {})
        server_ms = details.get("server_ms")
        log.info("%s phase=%s cycle=%d status=%s request=%d observe_ms=%.1f prepare_ms=%.1f connect_ms=%.1f "
                 "remote_ms=%.1f server_ms=%s postprocess_ms=%.1f setup_ms=%.1f request_ms=%.1f "
                 "execute_ms=%.1f total_ms=%.1f observation_age_ms=%.1f",
                 self.config.policy['type'], phase, self.predictions, status, timing['request'],
                 timing['observe_ms'], details.get('prepare_ms', 0.), details.get('connect_ms', 0.),
                 details.get('remote_ms', inference_ms),
                 'unknown' if server_ms is None else f'{float(server_ms):.1f}',
                 details.get('postprocess_ms', 0.), setup_ms, inference_ms,
                 execute_ms, (time.monotonic() - timing['started']) * 1000., timing['observation_age_ms'])

    def _finish_cycle(self, status, monotonic_s):
        if self.cycle_timing is None:
            return
        self._log_timing("execute", self.cycle_timing, self.plan, self.inference_ms,
                         execute_ms=max(0., monotonic_s - self.plan_started) * 1000., status=status)
        self.cycle_timing = None

    def _reset_observations(self, reason):
        """Wait for newly acquired camera/state samples after blocking setup."""
        self.execution_observation_after_us = now_us()
        for history in self.frames.values():
            history.clear()
        self.observations.clear()
        self.observation_wait = reason

    def _prepare_arms(self):
        """Finish SDK mode switching before collecting an executable prediction."""
        started = time.monotonic()
        for side in self.config.active_arms:
            if side not in self.enabled:
                self.enabled.add(side)  # Cleanup also covers a partially failed enable().
                self.arms[side].enable()
        histories = self._states(now_us())
        if any(not histories[side] for side in self.config.active_arms):
            raise RuntimeError("Missing joint feedback after xArm enable")
        # Cancel any previous servo target by holding measured q while waiting
        # for the post-enable prediction. Do not send the pre-enable model q.
        positions = {side: histories[side][-1].q.copy() for side in self.config.active_arms}
        for side, q in positions.items():
            self.references[side].initialize(q)
        for side, q in positions.items():
            self.arms[side].command_joint_positions(q)
        self._reset_observations("waiting for a camera frame after xArm mode switch")
        if not self.config.execution.get("quiet", False):
            log.info("xArm preparation completed: enable_ms=%.1f; holding measured q, "
                     "discarding pre-enable prediction and waiting for a new observation",
                     (time.monotonic() - started) * 1000.)

    def _remaining(self, current_us, monotonic_s):
        if self.plan is None:
            return 0.
        chunk_remaining = len(self.plan.values) * self.plan.step_s - (monotonic_s - self.plan_started)
        if self.open_loop:
            return chunk_remaining
        # A tail that would exceed the source observation's age budget is not
        # executable. Allow a ready new prediction to replace it before expiry.
        age_remaining = (self.config.execution["maximum_prediction_age_s"]
                         - (current_us - self.plan.timestamp_us) / 1e6
                         - 1. / self.config.execution["control_hz"])
        return min(chunk_remaining, age_remaining)

    def _prefetch_s(self):
        if self.open_loop:
            return 0.  # Observe and request again only after the chunk finishes.
        latency_s = max(self.prediction_latencies_s,
                        default=(self.inference_ms or 0.) / 1000.)
        margin_s = self.config.execution.get("prediction_prefetch_margin_s", .1)
        return max(.05, latency_s + margin_s + 1. / self.config.execution["control_hz"])

    def _execute(self, histories, current_us, monotonic_s, dt):
        if self.plan is None:
            return
        current_us = max(current_us, now_us())
        monotonic_s = max(monotonic_s, time.monotonic())
        age_us = current_us - self.plan.timestamp_us
        if age_us < 0 or (not self.open_loop and age_us > self.config.execution["maximum_prediction_age_s"] * 1e6):
            raise RuntimeError("Policy target expired; stopping xArm: "
                               f"age_ms={(current_us - self.plan.timestamp_us) / 1000.:.1f}, "
                               f"maximum_age_ms={self.config.execution['maximum_prediction_age_s'] * 1000.:.1f}, "
                               f"infer_ms={self.inference_ms}, "
                               f"plan_elapsed_ms={(monotonic_s - self.plan_started) * 1000.:.1f}, "
                               f"prediction_status={self.prediction_status}")
        index = min(int(max(0., monotonic_s - self.plan_started) / self.plan.step_s),
                    len(self.plan.values) - 1)
        target = self.plan.values[index]
        self.last_target = target.copy()
        if not self.command_enabled:
            return
        # Check every side before sending any command.
        if any(not histories[side] for side in self.config.active_arms):
            raise RuntimeError("Missing joint feedback before command")
        if not set(self.config.active_arms) <= self.enabled:
            raise RuntimeError("xArm mode must be prepared before executing policy q")
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
        self._finish_cycle("interrupted", time.monotonic())
        self.plan = None
        self.reset_progress = None
        self.last_target = None
        self.observations.clear()
        self._stop_motion()

    def _begin_reset(self):
        """Issue a planned return to rest_q, then poll completion in the loop."""
        if not self.command_enabled:
            self._reset_observations("paused; press s to start inference")
            log.info("Inference stopped; read-only mode leaves xArm positions unchanged")
            return
        reset_started = time.monotonic()
        from nero_collection.arms.mock import MockArm

        settings = self.config.hardware.get("alignment", {})
        defaults = {"reset_speed_rad_s": .2, "reset_acc_rad_s2": .5,
                    "reset_timeout_s": 60., "reset_tolerance_rad": .03}
        parameters = {key: float(settings.get(key, value)) for key, value in defaults.items()}
        if any(not np.isfinite(value) or value <= 0 for value in parameters.values()):
            raise ValueError("Reset speed, acceleration, timeout and tolerance must be positive and finite")
        samples = settings.get("reset_samples", 5)
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("alignment.reset_samples must be a positive integer")
        targets = {}
        # Validate every side before enabling or moving any of them.
        for side in self.config.active_arms:
            target = np.asarray(self.config.endpoints[side].rest_q, dtype=float)
            if target.shape != (7,) or not np.isfinite(target).all():
                raise ValueError(f"Invalid reset_q on {side}: expected seven finite joint angles")
            limits = self.config.joint_limits.get(side)
            if limits is not None and (np.any(target < limits[:, 0]) or np.any(target > limits[:, 1])):
                raise ValueError(f"reset_q is outside joint limits on {side}")
            arm = self.arms[side]
            if not callable(getattr(arm, "move_to_reset", None)) and not isinstance(arm, MockArm):
                raise RuntimeError(f"Arm {side} does not support planned reset motion")
            targets[side] = target.copy()
        for side in self.config.active_arms:
            self.enabled.add(side)  # Include partial enable/reset failures in cleanup.
            self.arms[side].enable()
        for side, target in targets.items():
            arm = self.arms[side]
            if callable(getattr(arm, "move_to_reset", None)):
                arm.move_to_reset(target, speed=parameters["reset_speed_rad_s"],
                                  acceleration=parameters["reset_acc_rad_s2"])
            else:
                arm.move_joints(target)  # Offline MockArm only.
        self._reset_observations("resetting xArm; inference paused")
        self.reset_progress = {
            "started": reset_started,
            "targets": targets, "parameters": parameters, "samples": samples,
            "deadline": time.monotonic() + parameters["reset_timeout_s"],
            "after_us": self.execution_observation_after_us,
            "settled": {side: 0 for side in targets},
            "timestamps": {side: None for side in targets},
            "statuses": {}, "next_status_check": 0.,
        }
        log.info("Inference stopped; returning xArm to reset_q at %.3f rad/s; "
                 "q interrupts reset", parameters["reset_speed_rad_s"])

    def _poll_reset(self, histories, current_us, *, resume=False):
        progress = self.reset_progress
        if progress is None:
            return False
        if time.monotonic() >= progress["deadline"]:
            raise TimeoutError("xArm reset did not settle before alignment.reset_timeout_s")
        for side, target in progress["targets"].items():
            history = histories[side]
            if not history:
                continue
            state = history[-1]
            stamp = state_time(state)
            if stamp < progress["after_us"] or stamp == progress["timestamps"][side]:
                continue
            progress["timestamps"][side] = stamp
            error = np.max(np.abs(np.asarray(state.q) - target))
            progress["settled"][side] = (progress["settled"][side] + 1
                                         if error <= progress["parameters"]["reset_tolerance_rad"] else 0)
        candidates_ready = all(count >= progress["samples"] for count in progress["settled"].values())
        if time.monotonic() >= progress["next_status_check"] or candidates_ready:
            for side, arm in self.arms.items():
                if side not in progress["targets"]:
                    continue
                if callable(getattr(arm, "control_status", None)):
                    status = arm.control_status()
                    if (status.get("error_code", 0) or status.get("mode") not in {None, 0}
                            or (status.get("state") is not None and status["state"] >= 3)):
                        raise RuntimeError(f"xArm reset stopped on {side}: {status}")
                    progress["statuses"][side] = status
                    if status.get("state") == 1:
                        progress["settled"][side] = 0
            progress["next_status_check"] = time.monotonic() + .1
        if not all(count >= progress["samples"] for count in progress["settled"].values()):
            return False
        # Reset uses controller mode 0. Clear the prepared servo state so the
        # next s must enable mode 1 again before accepting a new policy target.
        self._stop_motion()
        if self.config.execution.get("log_timing", False):
            log.info("%s reset_ms=%.1f", self.config.policy['type'],
                     (time.monotonic() - progress['started']) * 1000.)
        self.reset_progress = None
        self._reset_observations("reset complete; waiting for fresh observations" if resume
                                 else "reset complete; press s to start inference")
        log.info("xArm reset complete; %s", "starting inference" if resume
                 else "inference paused, press s to start")
        return True

    def run(self, duration=None, *, read_key=None, single_step=False, start_paused=False):
        paused, one_chunk, submitted_once = bool(single_step or start_paused), bool(single_step), False
        in_flight, next_plan, latest_window = False, None, None
        next_plan_latency_s = None
        expired_startup_predictions, last_startup_expiry = 0, None
        warmup_remaining = self.config.execution.get("warmup_steps", 1)
        last_submitted_us = None
        submitted_at = None
        submitted_timing = next_plan_timing = None
        latest_observe_ms = 0.
        startup_reset_pending = self.config.execution.get("reset_on_start", False) and self.command_enabled
        resume_after_startup_reset = not paused
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
            self._reset_observations("waiting for a camera frame after startup")
            last_report = started
            if startup_reset_pending:
                paused = True
                if not self.config.execution.get("quiet", False):
                    log.info("Startup: resetting active xArm(s) before policy inference")
                self._begin_reset()
            if self.open_loop and not self.config.execution.get("quiet", False):
                log.info("Open-loop execution: consume each chunk, then request again; "
                         "hold the last q while waiting, with no prediction-age deadline")
            if paused and not startup_reset_pending:
                log.info("Inference paused; s starts inference, i stops and resets, q exits")
            elif not paused and warmup_remaining and not self.config.execution.get("quiet", False):
                log.info("Policy warmup: %d prediction(s); outputs will not be sent to xArm", warmup_remaining)
            while duration is None or time.monotonic() - started < duration:
                wall = time.monotonic()
                current_us = now_us()
                dt, previous = max(1e-4, min(wall - previous, period * 2)), wall
                command = decode_inference_key(read_key(0.) if read_key else None,
                                               single_step=single_step)
                if command is InferenceCommand.EXIT:
                    break
                if command in {InferenceCommand.PAUSE, InferenceCommand.STOP_RESET}:
                    resume_after_startup_reset = False
                    if command is InferenceCommand.STOP_RESET and self.reset_progress is not None:
                        log.info("xArm is already resetting; inference remains paused")
                        command = None
                    else:
                        self.pause()
                        paused, submitted_once = True, False
                        in_flight, next_plan, latest_window = False, None, None
                        last_submitted_us = submitted_at = None
                        if command is InferenceCommand.STOP_RESET:
                            self._begin_reset()
                        else:
                            log.info("Inference paused; press s to start")
                if command in {InferenceCommand.START, InferenceCommand.SINGLE_STEP} and self.reset_progress is not None:
                    log.info("Wait for xArm reset to complete before starting inference")
                elif command in {InferenceCommand.START, InferenceCommand.SINGLE_STEP} and paused:
                    self.pause()
                    self._reset_observations("waiting for a fresh observation after s")
                    paused, one_chunk = False, command is InferenceCommand.SINGLE_STEP
                    submitted_once = False
                    in_flight, next_plan, latest_window = False, None, None
                    last_submitted_us = submitted_at = None
                    warmup_remaining = self.config.execution.get("warmup_steps", 1)
                    expired_startup_predictions, last_startup_expiry = 0, None
                    active_started = time.monotonic()
                    log.info("%s", "One chunk" if one_chunk else "Continuous inference")
                observe_started = time.monotonic()
                histories = self._states(current_us)
                window = self._observe(histories, current_us)
                if window:
                    latest_window = window
                    latest_observe_ms = (time.monotonic() - observe_started) * 1000.
                # Camera polling happens after the loop-start timestamp. Use
                # clocks sampled after acquisition for scheduling and expiry.
                current_us, wall = now_us(), time.monotonic()
                if self.reset_progress is not None:
                    reset_done = self._poll_reset(histories, current_us,
                                                  resume=startup_reset_pending and resume_after_startup_reset)
                    if reset_done:
                        latest_window = None
                    if reset_done and startup_reset_pending:
                        startup_reset_pending = False
                        if resume_after_startup_reset:
                            paused = False
                            active_started = time.monotonic()
                if not paused:
                    if self.plan is not None and wall - self.plan_started >= len(self.plan.values) * self.plan.step_s:
                        self._finish_cycle("complete", wall)
                    remaining = self._remaining(current_us, wall)
                    prefetch_s = self._prefetch_s()
                    fresh_window = (latest_window is not None
                                    and (last_submitted_us is None or latest_window[-1].timestamp_us > last_submitted_us)
                                    and 0 <= current_us - latest_window[-1].timestamp_us
                                    <= self.config.execution["maximum_prediction_age_s"] * 1e6)
                    if (fresh_window and not in_flight and next_plan is None
                            and remaining <= prefetch_s and (warmup_remaining or not one_chunk or not submitted_once)):
                        self.worker.submit(latest_window)
                        in_flight = True
                        submitted_at = wall
                        self.requests += 1
                        submitted_timing = {"started": wall - latest_observe_ms / 1000.,
                                            "request": self.requests,
                                            "observe_ms": latest_observe_ms,
                                            "observation_age_ms": (current_us - latest_window[-1].timestamp_us) / 1000.}
                        last_submitted_us = latest_window[-1].timestamp_us
                        if not warmup_remaining:
                            submitted_once = True
                    result = self.worker.take_result()
                    if result:
                        in_flight = False
                        if warmup_remaining:
                            chunk, elapsed_ms = result
                            self._log_timing("warmup", submitted_timing, chunk, elapsed_ms)
                            warmup_remaining -= 1
                            if not self.config.execution.get("quiet", False):
                                log.info("Policy warmup completed: infer_ms=%.1f, observation_age_ms=%.1f, "
                                         "remaining=%d; discarding output and waiting for a new observation",
                                         elapsed_ms, (now_us() - chunk.timestamp_us) / 1000., warmup_remaining)
                            # Reset history/RNN state in the worker before real
                            # inference, and never reuse the warmup observation.
                            self.worker.cancel()
                            latest_window = None
                            if not warmup_remaining:
                                active_started = time.monotonic()
                        else:
                            next_plan_latency_s = max(result[1] / 1000., time.monotonic() - submitted_at)
                            next_plan_timing = submitted_timing
                            next_plan = result
                    current_us, wall = now_us(), time.monotonic()
                    remaining = self._remaining(current_us, wall)
                    if next_plan and remaining <= 0.:
                        try:
                            self._validate_chunk(*next_plan, current_us)
                            if self.command_enabled and not set(self.config.active_arms) <= self.enabled:
                                # Never execute a prediction made before a
                                # blocking SDK mode switch, even if it was fresh.
                                setup_started = time.monotonic()
                                self._prepare_arms()
                                self._log_timing("prepare", next_plan_timing, next_plan[0], next_plan[1],
                                                 setup_ms=(time.monotonic() - setup_started) * 1000.)
                                histories = self._states(now_us())
                                self.worker.cancel()
                                latest_window = None
                                last_submitted_us = submitted_at = None
                                submitted_once = False
                                active_started = time.monotonic()
                            else:
                                self._install(*next_plan, current_us, wall, timing=next_plan_timing)
                        except PolicyResultExpired as exc:
                            if self.plan is not None:
                                raise  # Running execution retains strict expiry.
                            expired_startup_predictions += 1
                            last_startup_expiry = str(exc)
                            log.warning("Discarding expired startup prediction #%d: %s; "
                                        "retrying with a fresh observation, startup_remaining_s=%.1f",
                                        expired_startup_predictions, exc,
                                        max(0., self.config.execution['startup_timeout_s']
                                            - (time.monotonic() - active_started)))
                            self.worker.cancel()
                            latest_window = None
                            submitted_once = False
                            submitted_at = None
                            self._reset_observations("waiting for a fresh observation after expired startup prediction")
                            # Do not reset the startup timeout or include this
                            # unusable slow response in steady-state prefetch.
                        else:
                            self.prediction_latencies_s.append(next_plan_latency_s)
                        next_plan = None
                        next_plan_latency_s = None
                        next_plan_timing = None
                    current_us, wall = now_us(), time.monotonic()
                    self.prediction_status = {
                        "open_loop": self.open_loop,
                        "in_flight": in_flight,
                        "in_flight_ms": round((wall - submitted_at) * 1000., 1) if in_flight else None,
                        "next_plan_ready": next_plan is not None,
                        "prefetch_ms": round(self._prefetch_s() * 1000., 1),
                        "observation_age_ms": (None if not self.observations else
                                               round((current_us - self.observations[-1].timestamp_us) / 1000., 1)),
                        "observation_wait": self.observation_wait,
                        "camera_age_ms": {name: round((current_us - history[-1].timestamp_us) / 1000., 1)
                                          if history else None for name, history in self.frames.items()},
                        "state_age_ms": {side: round((current_us - state_time(history[-1])) / 1000., 1)
                                         if history else None for side, history in histories.items()},
                    }
                    if self.plan is None:
                        if wall - active_started > self.config.execution["startup_timeout_s"]:
                            raise RuntimeError("Timed out waiting for cameras, aligned states and first policy output: "
                                               f"warmup_remaining={warmup_remaining}, in_flight={in_flight}, "
                                               f"expired_startup_predictions={expired_startup_predictions}, "
                                               f"last_startup_expiry={last_startup_expiry}, {self.observation_wait}")
                    else:
                        self._execute(histories, current_us, wall, dt)
                        if one_chunk and wall - self.plan_started >= len(self.plan.values) * self.plan.step_s:
                            self.pause()
                            paused = True
                if not self.config.execution.get("quiet", False) and wall - last_report >= 1.:
                    status = ("resetting" if self.reset_progress is not None else "paused" if paused
                              else "warming_policy" if warmup_remaining and in_flight
                              else "inferencing" if in_flight else "executing" if self.plan is not None
                              else self.observation_wait)
                    observation_age_ms = (None if not self.observations else
                                          round((now_us() - self.observations[-1].timestamp_us) / 1000., 1))
                    log.info("policy=%s status=%s predictions=%d command_ticks=%d infer_ms=%s "
                             "in_flight_ms=%s prefetch_ms=%.1f observation_age_ms=%s target_q=%s",
                             self.config.policy["type"], status, self.predictions, self.commands,
                             None if self.inference_ms is None else round(self.inference_ms, 1),
                             round((wall - submitted_at) * 1000., 1) if in_flight else None,
                             self._prefetch_s() * 1000.,
                             observation_age_ms,
                             None if self.last_target is None else np.round(self.last_target, 3).tolist())
                    last_report = wall
                next_tick += period
                if next_tick < time.monotonic():
                    next_tick = time.monotonic()
                time.sleep(max(0., next_tick - time.monotonic()))
        finally:
            # Stop commands first, including when only one side of a dual command failed.
            try:
                self._finish_cycle("interrupted", time.monotonic())
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
