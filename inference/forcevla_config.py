"""Local deployment manifest for the remote ForceVLA seven-joint policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from inference.joint_config import resolve_path
from nero_collection.config import ArmEndpointConfig, CameraConfig
from xarm_stack.causal_tau_ext import load_processing_record


@dataclass
class ForceVLADeployment:
    path: Path
    server: dict
    policy: dict
    tau_ext: dict
    hardware_config: Path
    hardware: dict
    active_arms: tuple[str, ...]
    endpoints: dict
    cameras: tuple
    observation: dict
    execution: dict
    joint_limits: dict = field(default_factory=dict)

    @classmethod
    def load(cls, filename):
        path = Path(filename).expanduser().resolve()
        data = yaml.safe_load(path.read_text())
        server, policy = dict(data["server"]), dict(data["policy"])
        # The upstream metadata greeting is empty: action semantics cannot be discovered over WS.
        if policy["training_config"] not in {"pi0_force_lora_board", "pi0_force_lora_huanggua"}:
            raise ValueError("Use a v3 seven-joint ForceVLA config: pi0_force_lora_board or "
                             "pi0_force_lora_huanggua; legacy forcevla_lora has a different action space")
        if not isinstance(policy.get("prompt"), str) or not policy["prompt"].strip():
            raise ValueError("policy.prompt must be a nonempty task instruction")
        policy["type"] = "forcevla"
        rate = float(policy.get("action_rate_hz", 25.))
        steps = policy.get("execute_steps", 10)
        if not np.isfinite(rate) or rate <= 0 or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("policy.action_rate_hz must be positive; execute_steps must be a positive integer")
        policy.update(action_rate_hz=rate, execute_steps=steps)
        size = policy.setdefault("image_size", [224, 224])
        if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size):
            raise ValueError("policy.image_size must be [height, width] with positive integers")
        if not isinstance(server.get("host"), str) or not server["host"].strip():
            raise ValueError("server.host must be the server Tailscale IP or MagicDNS name")
        port = server.setdefault("port", 8000)
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("server.port must be an integer in 1..65535")
        server.setdefault("connect_timeout_s", 10.)
        server.setdefault("request_timeout_s", 120.)
        server.setdefault("startup_timeout_s", 300.)

        hardware_path = resolve_path(data["hardware_config"], path.parent)
        hardware = yaml.safe_load(hardware_path.read_text())
        calibration = yaml.safe_load(resolve_path(hardware["gello_config"], hardware_path.parent).read_text())
        side = data.get("arm", "right")
        if side not in {"left", "right"}:
            raise ValueError("arm must be left or right (one seven-joint arm per policy)")
        robot = calibration[side]["RobotConfig"]
        control = hardware.get("control", {})
        options = dict(hardware.get("arms", {}).get(side, {}))
        options.update(data.get("arm_options", {}))
        options.setdefault("robot_ip", robot["robot_ip"])
        options.update(dof=7, feedback_signal="torque", velocity_source="hardware", execution_enabled=False)
        options.setdefault("sdk_timeout_s", control.get("sdk_timeout_s", .1))
        endpoint = ArmEndpointConfig(side, rest_q=tuple(robot["reset_q"]), config_kwargs=options)

        cameras = []
        for item in data.get("cameras", hardware.get("cameras", [])):
            item = dict(item)
            item.pop("arm", None)
            if item.get("enabled", True):
                cameras.append(CameraConfig(**item))
        names = [camera.name for camera in cameras]
        observation = {"maximum_camera_age_s": .3, "maximum_camera_skew_s": .06,
                       "maximum_state_age_s": .15, "maximum_alignment_gap_s": .1,
                       **data.get("observation", {})}
        mapping = observation.setdefault("camera_map", {"image": "front", "right_wrist_image": "right_wrist"})
        if "wrist_image" in mapping:
            raise ValueError("camera_map must not include wrist_image: the server fills the "
                             "left-wrist input from the training config and sets mask=False")
        if (not {"image", "right_wrist_image"} <= mapping.keys()
                or not set(mapping) <= {"image", "right_wrist_image"}
                or len(set(mapping.values())) != len(mapping)
                or len(set(names)) != len(names) or not set(mapping.values()) <= set(names)):
            raise ValueError("camera_map must map image (front) and right_wrist_image "
                             "(right wrist) to distinct configured cameras")
        observation["low_dim_sources"] = {"tau_ext": "tau_ext"}
        execution = {"control_hz": control.get("sample_rate_hz", 100.), "startup_timeout_s": 150.,
                     "maximum_prediction_age_s": 2., **data.get("execution", {})}
        if "enable_command" in execution and not isinstance(execution["enable_command"], bool):
            raise ValueError("execution.enable_command must be true or false")
        tau_ext = {"processing_record": "tau_ext_dp_hirol_clean.json",
                   "urdf_path": "../../gello_teleop/models/xarm7_dynamics.urdf", "max_gap_s": .15,
                   **data.get("tau_ext", {})}
        for key in ("processing_record", "urdf_path"):
            tau_ext[key] = resolve_path(tau_ext[key], path.parent)
        processing = load_processing_record(tau_ext["processing_record"], tau_ext["urdf_path"])
        tau_ext["processing_config"] = processing.to_dict()
        if not np.isclose(processing.fps, rate):
            raise ValueError("tau_ext saved FPS must match policy.action_rate_hz")
        if execution["control_hz"] < rate:
            raise ValueError("control_hz must be at least the tau_ext saved FPS")
        for settings in (server, execution, observation, tau_ext):
            for key, value in settings.items():
                if key.endswith(("_s", "_hz")) and value is not None:
                    values = np.asarray(value, dtype=float)
                    if not np.isfinite(values).all() or np.any(values <= 0):
                        raise ValueError(f"{key} must be positive and finite")
        joints = {j.attrib["name"]: j for j in ET.parse(tau_ext["urdf_path"]).getroot().findall("joint")}
        limits = np.asarray([[float(joints[f"joint{i}"].find("limit").attrib[k])
                              for k in ("lower", "upper")] for i in range(1, 8)])
        return cls(path, server, policy, tau_ext, hardware_path, hardware, (side,), {side: endpoint},
                   tuple(cameras), observation, execution, {side: limits})

    def server_command(self):
        """Shell command for an existing remote checkout; never edits that checkout."""
        repo = self.server.get("repo_path")
        checkpoint = self.server.get("checkpoint")
        runner = self.server.get("python_command", ["uv", "run"])
        if not repo or not checkpoint or not isinstance(runner, list) or not runner:
            raise ValueError("Set server.repo_path, server.checkpoint and server.python_command in config")
        args = [*runner, "scripts/serve_policy.py", "--port", str(self.server["port"]),
                "policy:checkpoint", f"--policy.config={self.policy['training_config']}",
                f"--policy.dir={checkpoint}"]
        return f"cd -- {shlex.quote(str(repo))} && exec {shlex.join(args)}"
