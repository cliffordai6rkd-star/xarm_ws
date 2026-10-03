"""Native ACT checkpoint adapter; imports training dependencies only on load."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np

from inference.core.contracts import ActionChunk
from inference.policies.dp.policy import DiffusionPolicy


class ACTPolicy:
    def __init__(self, model: Any, config: Mapping, stats: Mapping, *,
                 image_shapes: Mapping, device="cpu", step_s=None, action_steps=None):
        self.model, self.device = model, device
        self.config = dict(config)
        self.image_keys = tuple(config["camera_names"])  # Training order matters.
        self.image_shapes = {key: tuple(image_shapes[key]) for key in self.image_keys}
        self.state_dim = int(config["state_dim"])
        self.action_dim = int(config["action_dim"])
        self.n_obs_steps = 1
        self.n_action_steps = int(config["num_queries"])
        self.action_steps = self.n_action_steps if action_steps is None else int(action_steps)
        self.step_s = step_s
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("ACT camera_names must be nonempty and unique")
        if not 1 <= self.action_steps <= self.n_action_steps:
            raise ValueError("ACT action_steps must be between 1 and num_queries")
        # A quaternion-normalized training target cannot be recovered as joint q.
        if config.get("quaternion_indices"):
            raise ValueError("ACT checkpoint quaternion_indices is set: this checkpoint "
                             "is incompatible with joint q execution. Train with action.joint "
                             "and without --quaternion_indices; do not edit the saved config.")
        action_key = str(config.get("action_key", "action.joint")).lower()
        if config.get("action_semantic", "joint") != "joint" or any(
                word in action_key for word in ("pose", "torque", "delta")):
            raise ValueError(f"ACT checkpoint action {action_key!r} is not absolute joint q")
        for key, shape in self.image_shapes.items():
            if len(shape) != 3 or shape[0] != 3 or min(shape) < 1:
                raise ValueError(f"ACT image shape {key}: expected [3,H,W], got {shape}")
        self.stats = {}
        for name, size in (("qpos_mean", self.state_dim), ("qpos_std", self.state_dim),
                           ("action_mean", self.action_dim), ("action_std", self.action_dim)):
            value = np.asarray(stats[name], dtype=np.float32)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"ACT {name} must contain {size} finite values")
            if name.endswith("std") and np.any(value <= 0):
                raise ValueError(f"ACT {name} must be positive")
            self.stats[name] = value

    @classmethod
    def from_checkpoint(cls, checkpoint_path, *, image_shapes, device="cpu",
                        source_path=None, config_path=None, stats_path=None,
                        step_s=None, action_steps=None):
        import torch

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        config_path = Path(config_path or checkpoint.parent / "policy_config.pkl")
        stats_path = Path(stats_path or checkpoint.parent / "dataset_stats.pkl")
        with config_path.open("rb") as handle:
            config = pickle.load(handle)
        with stats_path.open("rb") as handle:
            stats = pickle.load(handle)
        # Validate semantics and normalization before importing/building the model.
        policy = cls(None, config, stats, image_shapes=image_shapes, device=device,
                     step_s=step_s, action_steps=action_steps)
        source = Path(source_path or Path(__file__).resolve().parents[2].parent / "act").resolve()
        if not (source / "detr" / "main.py").is_file():
            raise FileNotFoundError(f"ACT source not found: {source}; set policy.source_path")
        for directory in (source, source / "detr"):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        from detr.main import get_args_parser
        from detr.models import build_ACT_model

        args = get_args_parser().parse_args([])
        for key, value in config.items():
            setattr(args, key, value)
        # All backbone weights come from the checkpoint; never download pretrained weights.
        with patch("detr.models.backbone.is_main_process", return_value=False):
            model = build_ACT_model(args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("model."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        policy.model = model.to(device).eval()
        return policy

    def start(self):
        self.model.eval()

    def close(self):
        pass

    def reset_episode(self):
        pass

    def predict_observations(self, observations):
        return self.predict(observations[-1])

    def predict(self, observation):
        import torch

        missing = set(self.image_keys) - observation.images.keys()
        if missing:
            raise KeyError(f"ACT missing cameras: {sorted(missing)}")
        # Use the same RGB resizing as DP, then ACT's ImageNet normalization.
        preparer = object.__new__(DiffusionPolicy)
        preparer._image_shapes = self.image_shapes
        images = np.stack([preparer._prepare_image(observation.images[key], key)
                           for key in self.image_keys])
        images = (images - np.array([.485, .456, .406], np.float32)[None, :, None, None])
        images /= np.array([.229, .224, .225], np.float32)[None, :, None, None]
        if self.config.get("image_only", False):
            state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            state = np.asarray(observation.q, dtype=np.float32)
            if state.shape != (self.state_dim,) or not np.isfinite(state).all():
                raise ValueError(f"ACT requires {self.state_dim} finite joint observations")
            state = (state - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        with torch.inference_mode():
            output = self.model(torch.from_numpy(state[None]).to(self.device),
                                torch.from_numpy(images[None]).to(self.device), None)
        values = output[0] if isinstance(output, tuple) else output
        values = values.detach().cpu().numpy()
        if values.shape != (1, self.n_action_steps, self.action_dim):
            raise ValueError(f"ACT returned unexpected action shape {values.shape}")
        values = values[0, :self.action_steps] * self.stats["action_std"] + self.stats["action_mean"]
        return ActionChunk(values, "joint", None, observation.timestamp_us, self.step_s,
                           {"policy": "act"})


__all__ = ["ACTPolicy"]
