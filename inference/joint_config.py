"""Deployment configuration and checkpoint contracts for xArm joint policies."""
from __future__ import annotations

import json
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from nero_collection.config import ArmEndpointConfig, CameraConfig


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def resolve_checkpoint(path, algorithm):
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if algorithm == "dp":
        for candidate in (path / "checkpoints/latest.ckpt", path / "latest.ckpt"):
            if candidate.is_file():
                return candidate
        if (path / "config.json").is_file():
            return path  # LeRobot pretrained_model export.
    else:
        if (path / "policy_best.ckpt").is_file():
            return path / "policy_best.ckpt"
        epochs = [(int(match.group(1)), candidate) for candidate in path.glob("policy_epoch_*_seed_*.ckpt")
                  if (match := re.fullmatch(r"policy_epoch_(\d+)_seed_\d+\.ckpt", candidate.name))]
        if epochs:
            return max(epochs, key=lambda pair: (pair[0], pair[1].name))[1]
    raise FileNotFoundError(f"No {algorithm} checkpoint in {path}; supply an explicit file")


@dataclass
class JointDeployment:
    path: Path
    policy: dict
    hardware_config: Path
    hardware: dict
    active_arms: tuple[str, ...]
    endpoints: dict[str, ArmEndpointConfig]
    cameras: tuple[CameraConfig, ...]
    observation: dict = field(default_factory=dict)
    execution: dict = field(default_factory=dict)
    joint_limits: dict = field(default_factory=dict)
    ik: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path, *, checkpoint=None, device=None, active_arms=None):
        path = Path(path).expanduser().resolve()
        data = yaml.safe_load(path.read_text())
        if data.get('pipeline') in ('pi0_contact_wm', 'pi0_latent_wm') and data.get('wm', {}).get('enabled') is False:
            # Reuse the same deployment YAML when the world model is disabled.
            observation = data['observation']
            data.setdefault('hardware_config', observation['hardware_config'])
            data.setdefault('active_arms', [observation['arm']])
            for source, target in (('maximum_age_ms', 'maximum_camera_age_s'),
                                   ('maximum_age_ms', 'maximum_state_age_s'),
                                   ('maximum_camera_skew_ms', 'maximum_camera_skew_s')):
                if source in observation:
                    observation.setdefault(target, observation[source]/1000.)
            execution = data.setdefault('execution', {})
            execution.setdefault('open_loop', True)
            execution.setdefault('startup_timeout_s', data['pi0'].get('request_timeout_s', 180.) + 60.)
            execution.setdefault('quiet', True)
            execution.setdefault('log_timing', True)
        policy = dict(data["policy"])
        algorithm = str(policy["type"]).lower()
        if algorithm not in {"dp", "act", "pi0"}:
            raise ValueError("policy.type must be dp, act or pi0")
        policy["type"] = algorithm
        if algorithm == "pi0":
            if checkpoint is not None:
                raise ValueError("pi0 checkpoints are loaded on the inference server")
            if data.get("wm", {}).get("enabled", False):
                raise ValueError("The direct pi0 adapter requires wm.enabled: false")
            policy["pi0"] = dict(data["pi0"])
            for key in ("local_checkpoint", "local_source_path"):
                if policy["pi0"].get(key):
                    policy["pi0"][key] = resolve_path(policy["pi0"][key], path.parent)
            policy["checkpoint"] = None
        else:
            policy["checkpoint"] = resolve_checkpoint(
                resolve_path(checkpoint, Path.cwd()) if checkpoint else
                resolve_path(policy["checkpoint"], path.parent), algorithm)
        for key in ("dataset_path", "source_path", "config_path", "stats_path", "client_source_path"):
            if policy.get(key):
                policy[key] = resolve_path(policy[key], path.parent)
        if device:
            policy["device"] = device
        hardware_path = resolve_path(data["hardware_config"], path.parent)
        hardware = yaml.safe_load(hardware_path.read_text())
        calibration = yaml.safe_load(resolve_path(hardware["gello_config"], hardware_path.parent).read_text())
        sides = active_arms if active_arms is not None else data.get("active_arms")
        sides = tuple(hardware.get("active_arms", ["left", "right"]) if sides is None else sides)
        if not sides or len(set(sides)) != len(sides) or not set(sides) <= {"left", "right"}:
            raise ValueError("active_arms must select left, right or both without duplicates")
        # An explicit side order is the action/qpos concatenation order.
        control = hardware.get("control", {})
        initial_q = data.get("execution", {}).get("initial_q", {})
        if not isinstance(initial_q, dict) or set(initial_q) - {"left", "right"}:
            raise ValueError("execution.initial_q must map left/right to seven joint angles in radians")
        for side, values in initial_q.items():
            try:
                target = np.asarray(values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"execution.initial_q.{side} must contain seven finite joint angles") from exc
            if target.shape != (7,) or not np.isfinite(target).all():
                raise ValueError(f"execution.initial_q.{side} must contain seven finite joint angles")
            initial_q[side] = target.tolist()
        endpoints = {}
        for side in sides:
            robot = calibration[side]["RobotConfig"]
            kwargs = dict(hardware.get("arms", {}).get(side, {}))
            kwargs.update(robot_ip=robot["robot_ip"], dof=7,
                          sdk_timeout_s=control.get("sdk_timeout_s", .1),
                          execution_enabled=False)
            kwargs.setdefault("collision_sensitivity", control.get("collision_sensitivity"))
            endpoints[side] = ArmEndpointConfig(
                side, rest_q=tuple(initial_q.get(side, robot["reset_q"])), config_kwargs=kwargs)
        cameras = []
        for item in hardware.get("cameras", []):
            item = dict(item)
            item.pop("arm", None)
            if item.get("enabled", True):
                item["preview_output_size"] = (224, 224)
                cameras.append(CameraConfig(**item))
        names = [camera.name for camera in cameras]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate hardware camera names")
        execution = dict(data.get("execution", {}))
        observation = dict(data.get("observation", {}))
        defaults = {"control_hz": control.get("sample_rate_hz", 100),
                    "maximum_prediction_age_s": 1., "startup_timeout_s": 15., "warmup_steps": 1,
                    "prediction_prefetch_margin_s": .1, "open_loop": False,
                    "reset_on_start": False, "quiet": False, "log_timing": False}
        for key, value in defaults.items():
            execution.setdefault(key, value)
        warmup_steps = execution["warmup_steps"]
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        for key in ("open_loop", "reset_on_start", "quiet", "log_timing"):
            if not isinstance(execution[key], bool):
                raise ValueError(f"{key} must be a boolean")
        for key, value in {"maximum_camera_age_s": .3, "maximum_camera_skew_s": .06,
                           "maximum_state_age_s": control.get("state_max_age_s", .15),
                           "maximum_alignment_gap_s": .15}.items():
            observation.setdefault(key, value)
        for settings in (execution, observation):
            for key, value in settings.items():
                if key.endswith(("_s", "_hz")) and value is not None:
                    if not np.isfinite(float(value)) or float(value) <= 0:
                        raise ValueError(f"{key} must be positive and finite")
        joint_limits = {}
        for side in sides:
            leader = calibration[side].get("TeleoperatorConfig", {})
            limits = execution.get("joint_limits", {}).get(side) or leader.get("joint_limits")
            if limits is None and leader.get("dynamics_urdf"):
                urdf = resolve_path(leader["dynamics_urdf"], hardware_path.parent)
                joints = {joint.attrib["name"]: joint for joint in ET.parse(urdf).getroot().findall("joint")}
                limits = [[float(joints[name].find("limit").attrib[bound]) for bound in ("lower", "upper")]
                          for name in leader.get("dynamics_joint_names", [f"joint{i}" for i in range(1, 8)])]
            if limits is not None:
                value = np.asarray(limits, dtype=float)
                if value.shape != (7, 2) or not np.isfinite(value).all() or np.any(value[:, 0] >= value[:, 1]):
                    raise ValueError(f"Invalid joint limits on {side}")
                joint_limits[side] = value
                if side in initial_q:
                    target = np.asarray(initial_q[side])
                    if np.any(target < value[:, 0]) or np.any(target > value[:, 1]):
                        raise ValueError(f"execution.initial_q.{side} is outside joint limits")
        ik = dict(data.get('ik', {}))
        if ik.get('urdf_path'):
            ik['urdf_path'] = resolve_path(ik['urdf_path'], path.parent)
        return cls(path, policy, hardware_path, hardware, sides, endpoints,
                   tuple(cameras), observation, execution, joint_limits, ik)


def _dataset_info(path):
    if path is None:
        return {}
    info = Path(path) / "meta/info.json"
    if not info.is_file():
        raise FileNotFoundError(f"Training dataset metadata not found: {info}")
    return json.loads(info.read_text())


def load_joint_policy(config: JointDeployment):
    options = config.policy
    algorithm, checkpoint = options["type"], options["checkpoint"]
    if options.get("source_path") and algorithm == "dp":
        sys.path.insert(0, str(options["source_path"]))
    device = options.get("device", "cuda:0")
    if algorithm == "pi0":
        from inference.policies.pi0_joint import Pi0JointPolicy

        policy = Pi0JointPolicy(options["pi0"], client_source_path=options.get("client_source_path"))
        if policy.action_type == 'eepose':
            from inference.pose_action_adapter import PoseActionAdapter

            if len(config.active_arms) != 1:
                raise ValueError('A seven-dimensional pi0 EEP pose selects exactly one active arm')
            policy.pose_adapter = PoseActionAdapter(config.ik, policy.interface,
                                                    joint_limits=config.joint_limits.get(config.active_arms[0]))
        info = {}
    elif algorithm == "dp":
        from inference.policies.dp.policy import DiffusionPolicy

        policy = DiffusionPolicy.from_checkpoint(
            checkpoint, device=device, use_ema=options.get("use_ema", True),
            sampling_method=options.get("sampling_method", "ddim"),
            num_inference_steps=options.get("num_inference_steps", 8),
            action_steps=options.get("action_steps"))
        if not hasattr(policy, "predict_observations"):
            raise ValueError("This xArm entry point requires a native Diffusion Policy Hydra .ckpt; "
                             "the LeRobot adapter uses the legacy inference configuration")
        cfg = getattr(policy.model, "_inference_checkpoint_config", {})
        dataset = cfg.get("task", {}).get("dataset", {}) if isinstance(cfg, Mapping) else {}
        action_key = dataset.get("action_key", options.get("action_key", "action.joint"))
        if dataset.get("quaternion_indices") or any(word in str(action_key).lower()
                                                    for word in ("pose", "torque", "delta")):
            raise ValueError(f"DP checkpoint action {action_key!r} is not absolute joint q")
        dataset_path = options.get("dataset_path") or dataset.get("dataset_path")
        if options.get("dataset_path") or (dataset_path and (Path(dataset_path) / "meta/info.json").is_file()):
            info = _dataset_info(dataset_path)
        else:
            info = {}  # Moved checkpoints can explicitly supply their training action rate.
        policy.image_shapes = dict(policy._image_shapes)
    else:
        from inference.policies.act import ACTPolicy

        with Path(options.get("config_path") or checkpoint.parent / "policy_config.pkl").open("rb") as handle:
            saved = pickle.load(handle)
        action_key = saved.get("action_key", options.get("action_key", "action.joint"))
        if any(word in str(action_key).lower() for word in ("pose", "torque", "delta")):
            raise ValueError(f"ACT action {action_key!r} is not absolute joint q")
        info = _dataset_info(options.get("dataset_path"))
        shapes = dict(saved.get("image_shapes", {}))
        for key in saved["camera_names"]:
            if key in shapes:
                continue
            feature = info.get("features", {}).get(key, {})
            if feature:
                shape = feature["shape"]
                shapes[key] = (3, int(shape[0]), int(shape[1]))
            elif key not in shapes:
                if not options.get("image_shape"):
                    raise ValueError("ACT has no saved image size; provide policy.dataset_path "
                                     "or policy.image_shape: [3,H,W]")
                shapes[key] = tuple(options["image_shape"])
        policy = ACTPolicy.from_checkpoint(
            checkpoint, device=device, image_shapes=shapes,
            source_path=options.get("source_path"), config_path=options.get("config_path"),
            stats_path=options.get("stats_path"), action_steps=options.get("action_steps"),
            brightness_jitter=options.get("brightness_jitter", 0.),
            brightness_seed=options.get("brightness_seed"))
    expected = 7 * len(config.active_arms)
    if policy.action_dim != expected:
        raise ValueError(f"Checkpoint outputs {policy.action_dim} values, but active_arms="
                         f"{list(config.active_arms)} requires {expected} joint angles")
    if algorithm == "act" and not policy.config.get("image_only") and policy.state_dim != expected:
        raise ValueError(f"ACT state_dim={policy.state_dim}, expected {expected} joint angles")
    fps = config.execution.get("action_rate_hz")
    if fps is None and policy.step_s is None:
        fps = info.get("fps")
    if fps is not None:
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Training action FPS must be positive and finite")
        if algorithm == "pi0" and not np.isclose(fps, 1. / policy.step_s):
            raise ValueError("execution.action_rate_hz differs from the pi0 training interface")
        policy.step_s = 1. / fps
    if policy.step_s is None:
        raise ValueError("Cannot determine action timing; set execution.action_rate_hz to training FPS")
    required_keys = tuple(policy.image_keys)
    shapes = policy.image_shapes
    if set(shapes) != set(required_keys):
        raise ValueError("Checkpoint must describe every required camera's [3,H,W] shape")
    camera_map = config.observation.get("camera_map", {})
    policy_camera_map = getattr(policy, "camera_map", {})
    mapping = {key: camera_map.get(key, policy_camera_map.get(key, key.removeprefix("observation.images.")))
               for key in required_keys}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Each checkpoint camera must map to a separate hardware camera")
    names = {camera.name for camera in config.cameras}
    if missing := set(mapping.values()) - names:
        raise ValueError(f"Checkpoint cameras missing from hardware config: {sorted(missing)}")
    # Low-dimensional inputs are explicitly mapped; unavailable effort is never filled with zero.
    sources = config.observation.setdefault("low_dim_sources", {})
    aliases = {"observation.joint": "q", "observation.state": "q", "qpos": "q",
               "observation.velocity": "dq", "observation.torque": "torque"}
    for key, shape in getattr(policy, "low_dim_shapes", {}).items():
        sources.setdefault(key, aliases.get(key))
        if sources[key] not in {"q", "dq", "ddq", "torque"} or tuple(shape) != (expected,):
            raise ValueError(f"Unsupported checkpoint low-dimensional input {key}: {shape}; "
                             "configure observation.low_dim_sources")
    return policy, mapping


def policy_report(config, policy, mapping):
    report = {"status": "ok", "policy": config.policy["type"],
            "checkpoint": str(config.policy["checkpoint"]), "active_arms": list(config.active_arms),
            "action": "absolute joint q (rad)", "action_dim": policy.action_dim,
            "action_steps": policy.action_steps, "action_rate_hz": 1. / policy.step_s,
            "n_obs_steps": policy.n_obs_steps, "camera_map": mapping,
            "image_shapes": policy.image_shapes, "hardware_config": str(config.hardware_config)}
    if config.policy["type"] == "act":
        report["brightness_jitter"] = policy.brightness_jitter
        report["brightness_seed"] = policy.brightness_seed
    if config.policy["type"] == "pi0":
        report["server"] = policy.config["host"] + ":" + str(policy.config["port"])
        report["interface"] = policy.interface
        report["server_contract_verified"] = policy.server_verified
        report["checkpoint"] = "loaded by openpi server"
        if policy.config.get("local", False):
            report["server"] = None
            report["checkpoint"] = str(policy.config["local_checkpoint"])
            report["local"] = True
            report["server_contract_verified"] = False
        report['model_action_type'] = policy.action_type
        if policy.action_type == 'eepose':
            report['action'] = 'absolute EEP pose -> local IK -> joint q (rad)'
            report['ik'] = {key: str(value) if isinstance(value, Path) else value
                            for key, value in config.ik.items()}
    return report
