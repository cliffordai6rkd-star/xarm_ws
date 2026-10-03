import pickle
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from inference.core.contracts import ActionChunk
from inference.joint_config import JointDeployment, load_joint_policy, resolve_checkpoint
from inference.joint_runtime import JointInferenceRuntime, JointObservation, PolicyWorker
from inference.policies.act import ACTPolicy
from inference.policies.dp.policy import DiffusionPolicy
from nero_collection.arms.mock import MockArm
from nero_collection.cameras import CameraFrame
from nero_collection.config import ArmEndpointConfig, CameraConfig
from nero_collection.time_utils import now_us


def observation(timestamp=100, q=None, red=255, blue=0):
    images = {"observation.images.left_wrist": np.full((8, 10, 3), red, np.uint8),
              "observation.images.right_wrist": np.full((8, 10, 3), blue, np.uint8)}
    return JointObservation(timestamp, np.arange(7.) if q is None else q, images,
                            {key: timestamp for key in images}, {"policy_state": {"observation.joint": np.arange(7.)}})


class FakeDP:
    n_obs_steps, horizon, n_action_steps, action_dim = 2, 6, 3, 7
    def __init__(self):
        self.obs_encoder = SimpleNamespace(rgb_keys=list(observation().images), key_shape_map={
            **{key: (3, 4, 5) for key in observation().images}, "observation.joint": (7,)})

    def predict_action(self, inputs):
        self.inputs = inputs
        return {"action": torch.arange(21.).reshape(1, 3, 7)}


def test_dp_uses_acquisition_window_and_low_dim_history():
    model = FakeDP()
    policy = DiffusionPolicy(model, device="cpu", step_s=1/30)
    first, second = observation(1, red=0, blue=255), observation(2)
    chunk = policy.predict_observations((first, second))
    assert chunk.values.shape == (3, 7)
    assert chunk.timestamp_us == 2
    left = model.inputs["observation.images.left_wrist"]
    assert tuple(left.shape) == (1, 2, 3, 4, 5)
    assert torch.all(left[0, 0] == 0) and torch.all(left[0, 1] == 1)
    assert tuple(model.inputs["observation.joint"].shape) == (1, 2, 7)
    # Skipped inference calls must not substitute the prior inference's old frame.
    policy.predict_observations((observation(3, red=51), observation(4, red=102)))
    assert torch.allclose(model.inputs["observation.images.left_wrist"][0, :, 0, 0, 0], torch.tensor([.2, .4]))
    policy.reset_episode()
    assert not any(policy._state_history.values())


class FakeACT(torch.nn.Module):
    def forward(self, q, image, env_state):
        self.q, self.image = q, image
        return torch.ones((1, 3, 7)), None, (None, None)


def act_fixture():
    config = {"camera_names": list(observation().images), "state_dim": 7,
              "action_dim": 7, "num_queries": 3, "image_only": False}
    stats = {"qpos_mean": np.ones(7), "qpos_std": np.full(7, 2.),
             "action_mean": np.arange(7.), "action_std": np.full(7, .5)}
    shapes = {key: (3, 4, 5) for key in config["camera_names"]}
    return config, stats, shapes


def test_act_matches_training_camera_order_normalization_and_denormalizes_q():
    config, stats, shapes = act_fixture()
    config["camera_names"].reverse()
    model = FakeACT()
    policy = ACTPolicy(model, config, stats, image_shapes=shapes, step_s=1/30)
    chunk = policy.predict(observation())
    assert tuple(model.image.shape) == (1, 2, 3, 4, 5)
    assert torch.allclose(model.image[0, 0, :, 0, 0], -torch.tensor([.485, .456, .406]) / torch.tensor([.229, .224, .225]))
    assert torch.allclose(model.q[0], (torch.arange(7.) - 1) / 2)
    np.testing.assert_allclose(chunk.values, np.tile(np.arange(7.) + .5, (3, 1)))
    assert chunk.semantic == "joint" and chunk.step_s == 1/30


def test_act_image_only_does_not_require_state_dim_to_match_q():
    config, stats, shapes = act_fixture()
    config["image_only"] = True
    model = FakeACT()
    ACTPolicy(model, config, stats, image_shapes=shapes).predict(observation(q=np.zeros(14)))
    assert torch.all(model.q == 0)


def test_act_pose_quaternion_target_is_rejected_before_model_import(tmp_path):
    config, stats, shapes = act_fixture()
    config["quaternion_indices"] = [3, 4, 5, 6]
    for name, value in (("policy_config.pkl", config), ("dataset_stats.pkl", stats)):
        (tmp_path / name).write_bytes(pickle.dumps(value))
    with pytest.raises(ValueError, match="quaternion_indices"):
        ACTPolicy.from_checkpoint(tmp_path / "unused.ckpt", image_shapes=shapes, source_path=tmp_path)


def test_act_rejects_invalid_normalizer_and_pose_targets():
    config, stats, shapes = act_fixture()
    stats["action_std"][2] = 0
    with pytest.raises(ValueError, match="action_std"):
        ACTPolicy(FakeACT(), config, stats, image_shapes=shapes)
    stats["action_std"][2] = .5
    config["action_key"] = "action.eepose"
    with pytest.raises(ValueError, match="not absolute joint q"):
        ACTPolicy(FakeACT(), config, stats, image_shapes=shapes)


def test_act_nonfinite_action_is_rejected():
    config, stats, shapes = act_fixture()
    model = FakeACT()
    model.forward = lambda *args: (torch.full((1, 3, 7), float("nan")), None, (None, None))
    policy = ACTPolicy(model, config, stats, image_shapes=shapes)
    with pytest.raises(ValueError, match="finite"):
        policy.predict(observation())


class FakePolicy:
    image_keys = ("observation.images.left_wrist", "observation.images.right_wrist")
    n_obs_steps, n_action_steps, action_steps, step_s = 2, 3, 3, .04
    low_dim_shapes = {}
    def __init__(self, arms=1, delay=0.):
        self.action_dim = arms * 7
        self.delay = delay
        self.windows = []

    def start(self): pass
    def close(self): pass
    def reset_episode(self): pass
    def predict_observations(self, window):
        if self.delay:
            time.sleep(self.delay)
        self.windows.append(window)
        q = np.stack([np.concatenate([np.full(7, (side+1) * (step+1) * .01)
                                     for side in range(self.action_dim // 7)]) for step in range(3)])
        return ActionChunk(q, "joint", None, window[-1].timestamp_us, self.step_s)


class FakeCameras:
    def __init__(self, outage_after=None):
        self.cameras = [SimpleNamespace(name=name) for name in ("left_wrist", "right_wrist")]
        self.count = 0
        self.outage_after = outage_after
        self.stopped = False
    def start(self): pass
    def stop(self): self.stopped = True
    def poll(self):
        self.count += 1
        if self.outage_after and self.count > self.outage_after:
            return []
        stamp = now_us()
        return [CameraFrame(camera.name, stamp, np.zeros((4, 5, 3), np.uint8)) for camera in self.cameras]


class TrackingArm(MockArm):
    def __post_init__(self):
        super().__post_init__()
        self.commands, self.enables, self.stops = [], 0, 0
    def enable(self):
        super().enable()
        self.enables += 1
    def read_state(self):
        state = super().read_state()
        state.torque[:] = np.nan  # Image/joint policies don't require force observations.
        state.torque_valid = False
        return state
    def command_joint_positions(self, q):
        self.commands.append(q.copy())
        super().command_joint_positions(q)
    def stop_motion(self):
        self.stops += 1


def runtime_fixture(sides=("right",), *, enabled=True, delay=0., outage_after=None):
    endpoints = {side: ArmEndpointConfig(side, rest_q=(0.,)*7) for side in sides}
    config = JointDeployment(Path("fixture"), {"type": "dp"}, Path("fixture"),
        {"control": {"position_mode": "direct", "max_step_rad": .1,
                     "max_velocity_rad_s": 100., "max_tracking_error_rad": 1.}},
        sides, endpoints, tuple(CameraConfig(name=name, backend="mock") for name in ("left_wrist", "right_wrist")),
        {"maximum_camera_age_s": .1, "maximum_camera_skew_s": .06,
         "maximum_state_age_s": .15, "maximum_alignment_gap_s": .15},
        {"control_hz": 100., "maximum_prediction_age_s": .5, "startup_timeout_s": .5})
    policy = FakePolicy(len(sides), delay)
    arms = {side: TrackingArm(endpoint) for side, endpoint in endpoints.items()}
    cameras = FakeCameras(outage_after)
    runtime = JointInferenceRuntime(config, policy, {key: key.removeprefix("observation.images.")
        for key in policy.image_keys}, arms=arms, camera_manager=cameras, command_enabled=enabled)
    return runtime, arms, policy, cameras


def test_dual_q_execution_consumes_whole_chunk_and_stops_both_arms():
    runtime, arms, policy, cameras = runtime_fixture(("left", "right"))
    runtime.run(.22)
    assert runtime.predictions >= 1
    for side, scale in (("left", 1), ("right", 2)):
        targets = {round(float(command[0]), 3) for command in arms[side].commands}
        assert {scale * .01, scale * .02, scale * .03} <= targets
        assert arms[side].enables == 1 and arms[side].stops == 1
        assert not arms[side]._connected
    assert policy.windows[0][-1].q.shape == (14,)
    assert cameras.stopped


def test_read_only_never_enables_or_sends_targets():
    runtime, arms, _, _ = runtime_fixture(enabled=False)
    runtime.run(.13)
    assert runtime.predictions
    assert arms["right"].enables == arms["right"].stops == 0
    assert not arms["right"].commands


def test_single_step_executes_one_full_chunk_then_pauses():
    runtime, arms, _, _ = runtime_fixture()
    keys = iter(["s"])
    runtime.run(.3, read_key=lambda _: next(keys, None), single_step=True)
    assert runtime.predictions == 1
    assert arms["right"].stops == 1
    assert {round(float(q[0]), 2) for q in arms["right"].commands} >= {.01, .02, .03}


def test_camera_loss_stops_every_enabled_arm():
    runtime, arms, _, _ = runtime_fixture(("left", "right"), outage_after=8)
    runtime.config.observation["maximum_camera_age_s"] = .03
    with pytest.raises(RuntimeError, match="camera expired"):
        runtime.run(.3)
    assert all(arm.stops == 1 and not arm._connected for arm in arms.values())


def test_expired_inference_result_is_never_sent():
    runtime, arms, _, _ = runtime_fixture(delay=.08)
    runtime.config.execution["maximum_prediction_age_s"] = .025
    with pytest.raises(RuntimeError, match="result expired"):
        runtime.run(.3)
    assert arms["right"].enables == 0 and not arms["right"].commands


def test_invalid_joint_limits_reject_entire_chunk_before_either_arm_commands():
    runtime, arms, _, _ = runtime_fixture(("left", "right"))
    runtime.config.joint_limits["right"] = np.tile([-.005, .005], (7, 1))
    with pytest.raises(ValueError, match="outside joint limits on right"):
        runtime.run(.15)
    assert all(arm.enables == 0 and not arm.commands for arm in arms.values())


def test_joint_send_failure_stops_both_sides():
    runtime, arms, _, _ = runtime_fixture(("left", "right"))
    def fail(q): raise RuntimeError("SDK failure")
    arms["right"].command_joint_positions = fail
    with pytest.raises(RuntimeError, match="SDK failure"):
        runtime.run(.15)
    assert all(arm.stops == 1 and not arm._connected for arm in arms.values())


def test_cancelled_inflight_prediction_cannot_reappear_after_pause():
    entered, release = threading.Event(), threading.Event()
    policy = FakePolicy()
    predict = policy.predict_observations
    def blocked(window):
        entered.set()
        release.wait(timeout=1.)
        return predict(window)
    policy.predict_observations = blocked
    worker = PolicyWorker(policy)
    worker.thread.start()
    worker.submit((observation(),))
    assert entered.wait(timeout=1.)
    worker.cancel()
    release.set()
    worker.close()
    assert worker.take_result() is None


def test_checkpoint_directory_selection(tmp_path):
    for name in ("policy_epoch_2_seed_0.ckpt", "policy_epoch_100_seed_0.ckpt"):
        (tmp_path / name).touch()
    assert resolve_checkpoint(tmp_path, "act").name == "policy_epoch_100_seed_0.ckpt"
    (tmp_path / "policy_best.ckpt").touch()
    assert resolve_checkpoint(tmp_path, "act").name == "policy_best.ckpt"


def test_camera_and_joint_alignment_never_use_future_samples():
    runtime, _, _, cameras = runtime_fixture()
    stamp = now_us()
    anchor = stamp - 50_000
    old_right = CameraFrame("right_wrist", anchor - 30_000, np.full((4, 5, 3), 5, np.uint8))
    runtime.frames["right_wrist"].append(old_right)
    frames = [CameraFrame("left_wrist", anchor, np.zeros((4, 5, 3), np.uint8)),
              CameraFrame("right_wrist", anchor + 1_000, np.full((4, 5, 3), 9, np.uint8))]
    cameras.poll = lambda: frames
    states = [SimpleNamespace(q_acquired_timestamp_us=anchor + offset, acquired_timestamp_us=0,
                              timestamp_us=anchor + offset, q=np.full(7, value))
              for offset, value in ((-1_000, 1.), (1_000, 9.))]
    window = runtime._observe({"right": states}, stamp)
    assert window[-1].timestamp_us == anchor
    np.testing.assert_array_equal(window[-1].q, np.ones(7))
    assert np.all(window[-1].images["observation.images.right_wrist"] == 5)
    assert all(timestamp <= anchor for timestamp in window[-1].image_timestamps_us.values())


def deployment_file(tmp_path):
    checkpoint = tmp_path / "test.ckpt"
    checkpoint.touch()
    urdf = Path(__file__).resolve().parents[1] / "gello_teleop/models/xarm7_dynamics.urdf"
    calibration = {side: {"RobotConfig": {"robot_ip": f"192.168.1.{suffix}", "reset_q": [0.] * 7},
                          "TeleoperatorConfig": {"dynamics_urdf": str(urdf)}}
                   for side, suffix in (("left", 203), ("right", 196))}
    (tmp_path / "calibration.yaml").write_text(yaml.safe_dump(calibration))
    hardware = {"gello_config": "calibration.yaml", "active_arms": ["right"],
                "cameras": [{"name": name, "backend": "mock"} for name in ("left_wrist", "right_wrist")]}
    (tmp_path / "hardware.yaml").write_text(yaml.safe_dump(hardware))
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump({"policy": {"type": "dp", "checkpoint": "test.ckpt"},
                                    "hardware_config": "hardware.yaml"}))
    return path


def test_hardware_config_is_loaded_without_connecting(tmp_path):
    config = JointDeployment.load(deployment_file(tmp_path))
    assert config.active_arms == ("right",)
    assert config.endpoints["right"].config_kwargs["robot_ip"] == "192.168.1.196"
    assert not config.endpoints["right"].config_kwargs["execution_enabled"]
    assert config.joint_limits["right"].shape == (7, 2)


def test_model_arm_dimension_and_missing_cameras_checked_before_hardware(monkeypatch, tmp_path):
    config = JointDeployment.load(deployment_file(tmp_path))
    policy = FakePolicy()
    policy.model = SimpleNamespace(_inference_checkpoint_config={})
    policy._image_shapes = {key: (3, 4, 5) for key in policy.image_keys}
    monkeypatch.setattr(DiffusionPolicy, "from_checkpoint", lambda *args, **kwargs: policy)
    config.execution["action_rate_hz"] = 30
    config.active_arms = ("left", "right")
    with pytest.raises(ValueError, match="requires 14 joint angles"):
        load_joint_policy(config)
    config.active_arms = ("right",)
    config.cameras = (CameraConfig("right_wrist"),)
    with pytest.raises(ValueError, match="missing from hardware config"):
        load_joint_policy(config)
