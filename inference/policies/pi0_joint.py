"""Remote openpi joint/EEP pose policy for the shared xArm execution loop.

The server owns ZeroObservationState, normalization and LiberoOutputs. Keys
containing '/' are literal keys, as required by the training LiberoInputs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from inference.core.contracts import ActionChunk
from inference.policies.errors import JointActionContractError


class OpenPiConnection:
    """Official openpi NumPy wire format with bounded connection/receive waits."""

    def __init__(self, config):
        try:
            from openpi_client import msgpack_numpy
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError("Install requirements-pi0-joint.txt and set policy.client_source_path "
                               "to openpi/packages/openpi-client/src") from exc
        host = str(config["host"])
        uri = host if host.startswith(("ws://", "wss://")) else "ws://" + host
        uri += ":" + str(config["port"])
        self.timeout = float(config.get("request_timeout_s", 30.))
        self.codec = msgpack_numpy
        # openpi runs synchronous JAX inference in its server event loop. First
        # compilation may delay pong replies; recv(timeout=...) still bounds waits.
        self.socket = connect(uri, compression=None, max_size=None, open_timeout=self.timeout,
                              close_timeout=1., ping_interval=None)
        try:
            self.metadata = self.codec.unpackb(self.socket.recv(timeout=self.timeout))
        except BaseException:
            self.socket.close()
            raise

    def get_server_metadata(self):
        return self.metadata

    def infer(self, observation):
        self.socket.send(self.codec.packb(observation))
        response = self.socket.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError(f"openpi server inference failed: {response}")
        return self.codec.unpackb(response)

    def close(self):
        self.socket.close()


def local_norm_stats_path(config):
    """Optional explicit stats file, relative to the selected checkpoint."""
    value = config.get("local_norm_stats")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["local_checkpoint"]) / path
    if path.name != "norm_stats.json" or not path.is_file():
        raise JointActionContractError(f"Local norm_stats.json not found: {path}")
    return path


class LocalPiConnection:
    """Load openpi lazily in the policy worker, using checkpoint normalization."""

    def __init__(self, config):
        stats_path = local_norm_stats_path(config)
        source = Path(config["local_source_path"])
        if not (source / "openpi/training/config.py").is_file():
            raise JointActionContractError(f"openpi source not found: {source}")
        sys.path.insert(0, str(source))
        try:
            from openpi.training import config as training_config
            from openpi.policies import policy_config
        except ImportError as exc:
            raise RuntimeError("Local pi0 requires the openpi training Python environment") from exc
        train = training_config.get_config(config["interface"]["training_config"])
        if train.model.action_horizon != config["action_horizon"]:
            raise JointActionContractError("Local training action_horizon differs from pi0.action_horizon")
        self.config = config
        norm_stats = None
        if stats_path is not None:
            from openpi.shared import normalize

            norm_stats = normalize.load(stats_path.parent)
        self.policy = policy_config.create_trained_policy(
            train, Path(config["local_checkpoint"]), default_prompt=config["prompt"],
            norm_stats=norm_stats)

    def get_server_metadata(self):
        # Local interface is configured here; there is no independent server handshake.
        return {"pi0_wm": self.config["interface"]}

    def infer(self, observation):
        response = self.policy.infer(observation)
        output_key = self.config.get("output_key", "actions")
        if output_key != "actions":
            response[output_key] = response.pop("actions")
        return response

    def close(self):
        self.policy = None


class MockPiConnection:
    """Explicit offline transport; no model, network, cameras or robot required."""

    def __init__(self, config, *, action=None):
        self.config = config
        self.calls = 0
        self.action = np.zeros(7, dtype=np.float32) if action is None else np.asarray(action, np.float32)

    def get_server_metadata(self):
        return {"pi0_wm": self.config["interface"], "mock_policy": True}

    def infer(self, observation):
        self.calls += 1
        return {self.config.get("output_key", "actions"):
                np.tile(self.action, (int(self.config["action_horizon"]), 1))}

    def close(self):
        pass


class Pi0JointPolicy:
    n_obs_steps = 1
    action_dim = 7
    low_dim_shapes = {}  # Training ZeroObservationState makes this image-only.

    def __init__(self, config, *, client_source_path=None, client_factory=None):
        self.config = dict(config)
        self.interface = dict(self.config["interface"])
        self.action_type = self.config.get('action_type', 'joint')
        self.pose_adapter = None
        if self.action_type not in {'joint', 'eepose'}:
            raise JointActionContractError('pi0.action_type must be joint or eepose')
        expected = {"state_key": "observation/state", "prompt_key": "prompt",
                    "state_semantic": "zero",
                    "image_format": "uint8_hwc_rgb", "output_space": "physical_absolute"}
        if self.action_type == 'joint':
            expected.update(action_semantic='action.joint', coordinate_frame=None,
                            representation='joint_positions', units='rad',
                            joint_names=[f'joint{i}' for i in range(1, 8)])
        else:
            expected.update(action_semantic='action.ee_pose', representation='xyz_quaternion', units='m')
            if not isinstance(self.interface.get('coordinate_frame'), str) or not self.interface['coordinate_frame'].strip():
                raise JointActionContractError('EEP pose interface.coordinate_frame is required')
            if self.interface.get('quaternion_order') not in {'xyzw', 'wxyz'}:
                raise JointActionContractError('EEP pose interface.quaternion_order must be xyzw or wxyz')
        for key, value in expected.items():
            if self.interface.get(key) != value:
                raise JointActionContractError(f"pi0 interface.{key} must be {value!r}; "
                                               f"selected action_type={self.action_type}")
        if not self.interface.get("training_config"):
            raise JointActionContractError("pi0 interface.training_config is required")
        images = self.interface.get("images", {})
        if not images or len(set(images.values())) != len(images):
            raise JointActionContractError("pi0 interface.images must map unique camera input keys")
        required_images = {"observation/image", "observation/wrist_image"}
        if set(images.values()) not in (required_images, required_images | {"observation/right_wrist_image"}):
            raise JointActionContractError("Expected two or three LiberoInputs camera keys")
        self.image_keys = tuple(images.values())
        self.camera_map = {key: name for name, key in images.items()}
        self.image_shapes = {key: (3, 224, 224) for key in self.image_keys}
        hz = float(self.interface["action_hz"])
        if not np.isfinite(hz) or hz <= 0 or hz != float(self.config["action_hz"]):
            raise JointActionContractError("pi0.action_hz must match the positive training interface FPS")
        self.step_s = 1. / hz
        self.horizon = self.config["action_horizon"]
        self.action_steps = self.config.get("consume_steps", self.horizon)
        for name, value in (("action_horizon", self.horizon), ("consume_steps", self.action_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise JointActionContractError(f"pi0.{name} must be a positive integer")
        if self.action_steps > self.horizon:
            raise JointActionContractError("consume_steps exceeds the trained action_horizon")
        timeout = float(self.config.get("request_timeout_s", 30.))
        if not np.isfinite(timeout) or timeout <= 0:
            raise JointActionContractError("pi0.request_timeout_s must be positive and finite")
        local = self.config.get("local", False)
        if not isinstance(local, bool):
            raise JointActionContractError("pi0.local must be a boolean")
        if local:
            for key in ("local_checkpoint", "local_source_path"):
                if not self.config.get(key):
                    raise JointActionContractError(f"pi0.{key} is required for local mode")
            checkpoint = Path(self.config["local_checkpoint"])
            if not (checkpoint / "params").is_dir() or not (checkpoint / "assets").is_dir():
                raise JointActionContractError(f"Local checkpoint requires params/ and assets/: {checkpoint}")
            local_norm_stats_path(self.config)
        if client_source_path and not local:
            source = Path(client_source_path).expanduser().resolve()
            if not (source / "openpi_client/msgpack_numpy.py").is_file():
                raise JointActionContractError(f"openpi client source not found: {source}")
            sys.path.insert(0, str(source))
        self.client_factory = client_factory or (LocalPiConnection if local else OpenPiConnection)
        self.client = None
        self.server_verified = False
        self.allow_mock_server = False

    def verify_server(self):
        if self.client is None:
            client = self.client_factory(self.config)
            try:
                metadata = client.get_server_metadata()
                if metadata.get("mock_policy") and not self.allow_mock_server:
                    raise JointActionContractError("A mock pi0 server requires --backend mock")
                actual = metadata.get("pi0_wm")
                if actual != self.interface:
                    if isinstance(actual, dict):
                        differences = '; '.join(
                            f'{key}: server={actual.get(key, "<missing>")!r}, '
                            f'local={self.interface.get(key, "<missing>")!r}'
                            for key in sorted(set(actual) | set(self.interface))
                            if actual.get(key, '<missing>') != self.interface.get(key, '<missing>'))
                    else:
                        differences = f'server pi0_wm metadata={actual!r}'
                    raise JointActionContractError(f"pi0 server interface mismatch: "
                                                   f"{differences}. Synchronize the server YAML pi0.interface "
                                                   "with the local config and restart serve_pi0_wm.py.")
            except BaseException:
                client.close()
                raise
            self.client = client
            self.server_verified = True

    def start(self):
        pass  # Connect in the policy worker, so s/i/q remain responsive.

    def reset_episode(self):
        pass  # openpi websocket reset is a no-op; cancelled results are dropped by the worker.

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None
        self.server_verified = False

    def predict_observations(self, observations, *, plan_steps=None):
        started = time.monotonic()
        if not observations:
            raise ValueError("pi0 observation window is empty")
        observation = observations[-1]
        request = {self.interface["state_key"]: np.zeros(7, dtype=np.float32),
                   self.interface["prompt_key"]: str(self.config["prompt"])}
        for key in self.image_keys:
            image = np.asarray(observation.images[key])
            if image.dtype != np.uint8 or image.shape != (224, 224, 3):
                raise ValueError(f"pi0 camera {key} must be uint8 RGB [224,224,3], got "
                                 f"{image.shape} {image.dtype}")
            request[key] = np.ascontiguousarray(image)
        prepared = time.monotonic()
        self.verify_server()
        connected = time.monotonic()
        response = self.client.infer(request)
        received = time.monotonic()
        actions = np.asarray(response[self.config.get("output_key", "actions")])
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise JointActionContractError(f"pi0 server must return post-transform physical absolute "
                                           f"{self.action_type} actions [{self.horizon},7], got {actions.shape}")
        steps = self.action_steps if plan_steps is None else plan_steps
        if type(steps) is not int or not 1 <= steps <= self.horizon:
            raise JointActionContractError("plan_steps must fit the trained pi0 horizon")
        values = actions[:steps]
        metadata = {"training_config": self.interface["training_config"],
                    "source": "openpi_local" if self.config.get("local", False) else "openpi_server",
                    "model_action_type": self.action_type}
        if self.action_type == 'eepose':
            if self.pose_adapter is None:
                raise JointActionContractError('EEP pose policy requires a configured local IK adapter')
            values, diagnostics = self.pose_adapter.convert(values, observation.q)
            metadata.update(ik=diagnostics, target_frame=self.interface['coordinate_frame'])
        metadata['timing'] = {
            'prepare_ms': (prepared - started) * 1000.,
            'connect_ms': (connected - prepared) * 1000.,
            'remote_ms': (received - connected) * 1000.,
            'postprocess_ms': (time.monotonic() - received) * 1000.,
            'server_ms': response.get('server_timing', {}).get('infer_ms'),
        }
        return ActionChunk(values, semantic="joint", frame_name=None,
                           timestamp_us=observation.timestamp_us, step_s=self.step_s,
                           metadata=metadata)
