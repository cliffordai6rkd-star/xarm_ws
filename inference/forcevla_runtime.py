"""Causal training-matched xArm torque features and remote ForceVLA execution."""
from __future__ import annotations

from dataclasses import dataclass, fields
import logging
import time

import numpy as np

from inference.core.contracts import ActionChunk
from inference.joint_runtime import JointInferenceRuntime, StateReader
from nero_collection.arms.base import ArmState
from xarm_stack.causal_tau_ext import CausalTauExt, TauExtConfig

log = logging.getLogger(__name__)


@dataclass
class TauExtState(ArmState):
    tau_ext: np.ndarray | None = None
    tau_ext_valid: bool = False
    observation_timestamp_us: int = 0


class TauExtStateReader(StateReader):
    """Resample feedback causally at the saved FPS, independently of policy requests."""

    def __init__(self, arm, period_s, settings):
        super().__init__(arm, period_s)
        self.settings = settings
        self.processor = CausalTauExt(settings["urdf_path"], TauExtConfig(**settings["processing_config"]))
        self.origin_us = self.previous_state = None
        self.sample_index = 0

    def reset_episode(self):
        with self.lock:
            self.processor.reset()
            self.origin_us = self.previous_state = None
            self.sample_index = 0
            self.history.clear()

    def accept_state(self, state):
        # Completion time avoids using feedback that arrived after the image.
        timestamp_us = int(state.q_timestamp_us or state.timestamp_us)
        if (not all((state.q_valid, state.dq_valid, state.torque_valid))
                or any(np.shape(v) != (7,) or not np.isfinite(v).all()
                       for v in (state.q, state.dq, state.torque))):
            raise RuntimeError("ForceVLA requires measured finite q/dq/torque (rad, rad/s, Nm)")
        with self.lock:
            previous_us = (int(self.previous_state.q_timestamp_us or self.previous_state.timestamp_us)
                           if self.previous_state is not None else None)
            if previous_us is not None:
                if timestamp_us == previous_us:
                    return
                if not 0 < timestamp_us - previous_us <= self.settings["max_gap_s"] * 1e6:
                    raise RuntimeError("tau_ext feedback timeline lost; stop and restart inference")
            if self.origin_us is None:
                self.origin_us = timestamp_us
            while True:
                sample_us = self.origin_us + round(self.sample_index * 1e6 / self.processor.config.fps)
                if sample_us > timestamp_us:
                    break
                # Zero-order hold: select only a report already available at this
                # grid time, and never interpolate using the later report.
                selected = state if sample_us == timestamp_us else self.previous_state
                result = self.processor.update(selected.q, selected.dq, selected.torque,
                                               self.sample_index / self.processor.config.fps)
                observed = TauExtState(**{f.name: getattr(selected, f.name) for f in fields(ArmState)},
                                       tau_ext=result["tau_ext"], tau_ext_valid=True,
                                       observation_timestamp_us=sample_us)
                self.history.append(observed)
                self.sample_index += 1
            self.previous_state = state

    def _run(self):
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                self.accept_state(self.arm.read_state())
                self.stop.wait(max(0., self.period_s - (time.monotonic() - started)))
        except Exception as exc:
            self.error = exc


def resize_rgb(image, size):
    """Match openpi-client's PIL bilinear resize-with-pad, retaining RGB uint8."""
    from PIL import Image

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("ForceVLA cameras must provide RGB uint8 HxWx3 images")
    height, width = size
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    current_height, current_width = image.shape[:2]
    ratio = max(current_width / width, current_height / height)
    resized = Image.fromarray(image).resize((int(current_width / ratio), int(current_height / ratio)),
                                            Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (width, height), 0)
    padded.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return np.asarray(padded)


class RemoteForceVLAPolicy:
    n_obs_steps = 1
    low_dim_shapes = {"tau_ext": (7,)}

    def __init__(self, config, client):
        self.config, self.client = config, client
        self.step_s = 1. / config.policy["action_rate_hz"]

    def start(self):
        self.client.connect()
        log.info("Remote ForceVLA metadata: %s", self.client.metadata)

    def reset_episode(self):
        pass  # The existing server is stateless and defines no reset RPC.

    def predict_observations(self, observations):
        observation = observations[-1]
        tau_ext = np.asarray(observation.metadata["policy_state"]["tau_ext"], dtype=np.float32)
        q = np.asarray(observation.q, dtype=np.float32)
        if q.shape != (7,) or tau_ext.shape != (7,) or not np.isfinite([q, tau_ext]).all():
            raise ValueError("ForceVLA state and tau_ext must be finite seven-vectors")
        request = {"state": q, "tau_ext": tau_ext, "prompt": self.config.policy["prompt"],
                   **{key: resize_rgb(frame, self.config.policy["image_size"])
                      for key, frame in observation.images.items()}}
        response = self.client.infer(request)
        actions = np.asarray(response["actions"], dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError(f"ForceVLA returned invalid absolute joint actions: {actions.shape}")
        steps = self.config.policy["execute_steps"]
        if len(actions) < steps:
            raise ValueError(f"ForceVLA returned {len(actions)} actions; execute_steps={steps}")
        # Keep the full server horizon. The executor selects the still-future
        # prefix when the result is installed, including time spent prefetching.
        return ActionChunk(actions, "joint", None, observation.timestamp_us, self.step_s,
                           {"server_timing": response.get("server_timing", {})})

    def close(self):
        self.client.close()


class ForceVLAInferenceRuntime(JointInferenceRuntime):
    def __init__(self, config, policy, *, command_enabled=False, backend="xarm", arms=None,
                 camera_manager=None):
        super().__init__(config, policy, config.observation["camera_map"], backend=backend,
                         command_enabled=command_enabled, arms=arms, camera_manager=camera_manager)
        period = 1. / config.execution["control_hz"]
        self.readers = {side: TauExtStateReader(arm, period, config.tau_ext) for side, arm in self.arms.items()}

    def pause(self):
        super().pause()
        for reader in self.readers.values():
            reader.reset_episode()

    def _connect_arm(self, arm):
        arm.connect()
        configure = getattr(arm, "configure_feedback_report", None)
        if configure:
            configure()

    def _install(self, chunk, inference_ms, current_us, monotonic_s):
        # JAX's first compilation can outlive the observation before any motion starts.
        # Discard that chunk and request a fresh observation once the model is warm.
        age_s = (current_us - chunk.timestamp_us) / 1e6
        if self.plan is None and age_s > self.config.execution["maximum_prediction_age_s"]:
            log.info("Discarding first stale ForceVLA result (%.2fs); requesting a fresh observation", age_s)
            return
        if age_s < 0:
            raise RuntimeError("ForceVLA observation timestamp is in the future")
        elapsed_steps = int(age_s / self.policy.step_s)
        if elapsed_steps >= len(chunk.values):
            if self.plan is None:
                log.info("Discarding first ForceVLA result: its complete action horizon has elapsed")
                return
            raise RuntimeError("ForceVLA action horizon expired before execution")
        end = elapsed_steps + self.config.policy["execute_steps"]
        selected = ActionChunk(chunk.values[elapsed_steps:end], chunk.semantic, chunk.frame_name,
                               chunk.timestamp_us, chunk.step_s,
                               {**chunk.metadata, "elapsed_action_steps": elapsed_steps})
        super()._install(selected, inference_ms, current_us, monotonic_s)
        # Maintain the training timeline: action[k] belongs to observation_time
        # + k*step_s. Keep its fractional phase when converting to monotonic time.
        phase_s = age_s - elapsed_steps * self.policy.step_s
        self.plan_started = monotonic_s - phase_s
