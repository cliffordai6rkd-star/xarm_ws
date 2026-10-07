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
from inference.joint_runtime import JointInferenceRuntime, JointObservation, PolicyWorker, StateReader, state_time
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


def test_act_brightness_jitter_reaches_model_before_normalization_without_mutating_frames():
    config, stats, shapes = act_fixture()
    obs = observation()
    for key, rgb in zip(config["camera_names"], ([0, 200, 255], [50, 128, 255])):
        obs.images[key][:] = rgb
    originals = {key: image.copy() for key, image in obs.images.items()}
    model = FakeACT()
    policy = ACTPolicy(model, config, stats, image_shapes=shapes,
                       brightness_jitter=.2, brightness_seed=0)
    factors = []
    for _ in range(10):
        chunk = policy.predict(obs)
        factor = chunk.metadata["brightness_factor"]
        factors.append(factor)
        assert .8 <= factor <= 1.2
        for index, key in enumerate(config["camera_names"]):
            rgb = torch.tensor(originals[key][0, 0], dtype=torch.float32) / 255
            expected = (torch.clamp(rgb * factor, 0, 1) -
                        torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
            torch.testing.assert_close(model.image[0, index, :, 0, 0], expected)
            np.testing.assert_array_equal(obs.images[key], originals[key])
        assert model.image.dtype == torch.float32
        assert tuple(model.image.shape) == (1, 2, 3, 4, 5)
    assert min(factors) < 1 < max(factors)


def test_act_brightness_seed_replays_input_perturbations():
    config, stats, shapes = act_fixture()
    policies = [ACTPolicy(FakeACT(), config, stats, image_shapes=shapes,
                          brightness_jitter=.2, brightness_seed=42) for _ in range(2)]
    sequences = [[policy.predict(observation(red=128)).metadata["brightness_factor"]
                  for _ in range(3)] for policy in policies]
    assert sequences[0] == sequences[1]
    assert len(set(sequences[0])) == 3
    torch.testing.assert_close(policies[0].model.image, policies[1].model.image)


@pytest.mark.parametrize("strength", [-.1, 1.1, float("nan"), float("inf")])
def test_act_invalid_brightness_jitter_is_rejected(strength):
    config, stats, shapes = act_fixture()
    with pytest.raises(ValueError, match="brightness_jitter"):
        ACTPolicy(FakeACT(), config, stats, image_shapes=shapes, brightness_jitter=strength)


def test_act_pose_quaternion_target_is_rejected_before_model_import(tmp_path):
    config, stats, shapes = act_fixture()
    config["quaternion_indices"] = [3, 4, 5, 6]
    for name, value in (("policy_config.pkl", config), ("dataset_stats.pkl", stats)):
        (tmp_path / name).write_bytes(pickle.dumps(value))
    with pytest.raises(ValueError, match="quaternion_indices"):
        ACTPolicy.from_checkpoint(tmp_path / "unused.ckpt", image_shapes=shapes, source_path=tmp_path)


def test_cli_reports_contract_error_without_starting_hardware(monkeypatch, tmp_path, capsys):
    from inference import joint_cli
    from inference.policies.errors import JointActionContractError

    def reject(config):
        raise JointActionContractError("quaternion_indices invalid; retrain action.joint")
    monkeypatch.setattr(joint_cli, "load_joint_policy", reject)
    with pytest.raises(SystemExit) as error:
        joint_cli.main(["--config", str(deployment_file(tmp_path)), "--run", "--enable-command"])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "quaternion_indices invalid" in stderr
    assert "Traceback" not in stderr


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


@pytest.mark.parametrize("enable_delay", [.11, .2])
def test_slow_mode_switch_uses_new_prediction_after_enable(enable_delay):
    runtime, arms, policy, _ = runtime_fixture(delay=.03)
    runtime.config.execution.update(warmup_steps=0, maximum_prediction_age_s=.16)
    policy.step_s = .02
    enable = arms["right"].enable
    ready_stamps = []
    def slow_enable():
        time.sleep(enable_delay)
        enable()
        ready_stamps.append(now_us())
    arms["right"].enable = slow_enable
    runtime.run(.45)
    assert runtime.predictions >= 2
    assert arms["right"].enables == arms["right"].stops == 1
    assert policy.windows[0][-1].timestamp_us < ready_stamps[0]
    assert any(window[-1].timestamp_us >= ready_stamps[0] for window in policy.windows[1:])
    assert {round(float(q[0]), 2) for q in arms["right"].commands} >= {.01, .02, .03}


def test_camera_loss_stops_every_enabled_arm():
    runtime, arms, _, _ = runtime_fixture(("left", "right"), outage_after=8)
    runtime.config.observation["maximum_camera_age_s"] = .03
    with pytest.raises(RuntimeError, match="camera expired"):
        runtime.run(.3)
    assert all(arm.stops == 1 and not arm._connected for arm in arms.values())


def test_expired_startup_results_retry_but_timeout_without_enabling(caplog):
    runtime, arms, policy, _ = runtime_fixture(delay=.08)
    runtime.config.execution.update(warmup_steps=0, maximum_prediction_age_s=.025,
                                    startup_timeout_s=.24)
    with caplog.at_level("WARNING", logger="inference.joint_runtime"):
        with pytest.raises(RuntimeError, match=r"Timed out.*expired_startup_predictions=[1-9].*last_startup_expiry=Policy result expired"):
            runtime.run(.6)
    assert len(policy.windows) >= 2
    assert "Discarding expired startup prediction" in caplog.text
    assert runtime.predictions == 0 and not runtime.prediction_latencies_s
    assert arms["right"].enables == 0 and not arms["right"].commands


@pytest.mark.parametrize("slow_request", [1, 2])
def test_slow_result_after_warmup_or_mode_switch_retries_fresh_then_executes(slow_request, caplog):
    runtime, arms, policy, _ = runtime_fixture()
    runtime.config.execution.update(warmup_steps=1, maximum_prediction_age_s=.12,
                                    startup_timeout_s=.4)
    policy.step_s = .02
    predict = policy.predict_observations
    discarded = []
    def slow_once(window):
        if len(policy.windows) == slow_request:
            time.sleep(.15)
            assert arms["right"].enables == (0 if slow_request == 1 else 1)
            # Before actual execution only a measured-q hold is permitted.
            assert all(np.all(q == 0.) for q in arms["right"].commands)
            chunk = predict(window)
            discarded.append((chunk.timestamp_us, now_us()))
            return ActionChunk(timestamp_us=chunk.timestamp_us, semantic="joint", frame_name=None,
                               values=np.full_like(chunk.values, .8), step_s=chunk.step_s)
        return predict(window)
    policy.predict_observations = slow_once
    keys = iter(["s"])
    with caplog.at_level("WARNING", logger="inference.joint_runtime"):
        runtime.run(.45, read_key=lambda _: next(keys, None), single_step=True)
    assert "Discarding expired startup prediction #1" in caplog.text
    assert runtime.predictions == 1 and len(policy.windows) == 4
    assert policy.windows[slow_request + 1][-1].timestamp_us > discarded[0][1]
    assert runtime.plan is None  # Single-step completes and pauses.
    assert max(runtime.prediction_latencies_s) < .12
    assert arms["right"].enables == arms["right"].stops == 1
    assert {round(float(q[0]), 2) for q in arms["right"].commands} >= {.01, .02, .03}
    assert all(np.all(q < .8) for q in arms["right"].commands)


@pytest.mark.parametrize("open_loop", [False, True])
def test_future_startup_result_is_rejected_without_retry(open_loop):
    runtime, arms, policy, _ = runtime_fixture()
    runtime.config.execution.update(warmup_steps=0, open_loop=open_loop)
    predict = policy.predict_observations
    def future_result(window):
        chunk = predict(window)
        return ActionChunk(timestamp_us=now_us() + 100_000, semantic="joint", frame_name=None,
                           values=chunk.values, step_s=chunk.step_s)
    policy.predict_observations = future_result
    with pytest.raises(RuntimeError, match="Policy result expired.*age_ms=-"):
        runtime.run(.3)
    assert len(policy.windows) == 1
    assert arms["right"].enables == 0 and not arms["right"].commands


def test_open_loop_executes_late_chunks_in_order_and_holds_while_requesting(caplog):
    runtime, arms, policy, _ = runtime_fixture(delay=.1)
    runtime.config.execution.update(warmup_steps=0, open_loop=True, quiet=True,
                                    log_timing=True, maximum_prediction_age_s=.03)
    request_starts, installed = [], []
    predict = policy.predict_observations
    install = runtime._install
    def tracked_predict(window):
        request_starts.append(time.monotonic())
        return predict(window)
    def tracked_install(*args, **kwargs):
        install(*args, **kwargs)
        installed.append(runtime.plan_started)
    policy.predict_observations = tracked_predict
    runtime._install = tracked_install
    with caplog.at_level("INFO", logger="inference.joint_runtime"):
        runtime.run(.85)
    assert runtime.predictions >= 2 and runtime._prefetch_s() == 0.
    # Request 0 is the pre-enable preview, 1 is the first executed plan.
    # Every later request begins after all three steps of the old plan.
    for start, previous_install in zip(request_starts[2:], installed):
        assert start - previous_install >= 3 * policy.step_s
    assert len(installed) >= 2 and installed[1] - installed[0] >= 3 * policy.step_s + .1
    targets = [round(float(q[0]), 2) for q in arms["right"].commands]
    assert targets.count(.03) >= 5  # Last target continues during the slow request.
    assert {.01, .02, .03} <= set(targets)
    assert "cycle=1 status=complete" in caplog.text
    assert "request_ms=" in caplog.text and "execute_ms=" in caplog.text and "total_ms=" in caplog.text
    assert "command_ticks=" not in caplog.text  # Quiet mode suppresses status spam.
    assert arms["right"].stops == 1


def test_open_loop_still_stops_if_camera_disappears():
    runtime, arms, _, _ = runtime_fixture(delay=.01, outage_after=15)
    runtime.config.execution.update(warmup_steps=0, open_loop=True)
    runtime.config.observation["maximum_camera_age_s"] = .03
    with pytest.raises(RuntimeError, match="Checkpoint camera expired"):
        runtime.run(.4)
    assert arms["right"].stops == 1


def test_cold_start_output_is_discarded_and_new_observation_is_used(caplog):
    runtime, arms, policy, _ = runtime_fixture()
    runtime.config.execution["maximum_prediction_age_s"] = .12
    runtime.config.execution["startup_timeout_s"] = .4
    policy.step_s = .01
    predict = policy.predict_observations
    def cold_start(window):
        if not policy.windows:
            time.sleep(.15)
            assert not arms["right"].commands and arms["right"].enables == 0
        return predict(window)
    policy.predict_observations = cold_start
    # The cold prediction exceeds the real-action deadline; execute only the
    # next prediction from a newer observation, preserving one-chunk mode.
    keys = iter(["s"])
    with caplog.at_level("INFO", logger="inference.joint_runtime"):
        runtime.run(.3, read_key=lambda _: next(keys, None), single_step=True)
    assert "Policy warmup completed" in caplog.text
    assert runtime.predictions == 1 and len(policy.windows) == 3
    assert policy.windows[1][-1].timestamp_us > policy.windows[0][-1].timestamp_us
    assert arms["right"].commands and arms["right"].enables == 1


def test_warmup_timeout_does_not_enable_either_arm():
    runtime, arms, _, _ = runtime_fixture(("left", "right"), delay=.15)
    runtime.config.execution["startup_timeout_s"] = .04
    with pytest.raises(RuntimeError, match=r"Timed out.*warmup_remaining=1, in_flight=True"):
        runtime.run(.3)
    assert all(arm.enables == 0 and not arm.commands and not arm._connected for arm in arms.values())


def test_install_refreshes_clock_after_acquiring_new_observation(monkeypatch):
    runtime, _, policy, _ = runtime_fixture()
    stamp = now_us()
    monkeypatch.setattr("inference.joint_runtime.now_us", lambda: stamp + 1_000)
    chunk = policy.predict_observations((observation(stamp),))
    # The iteration clock preceded camera polling, but the result is fresh.
    runtime._install(chunk, .5, stamp - 10_000, time.monotonic() - 1.)
    assert runtime.plan is chunk
    assert time.monotonic() - runtime.plan_started < .02


def test_install_checks_actual_result_age_instead_of_old_loop_clock(monkeypatch):
    runtime, arms, policy, _ = runtime_fixture()
    stamp = now_us()
    monkeypatch.setattr("inference.joint_runtime.now_us", lambda: stamp + 600_000)
    chunk = policy.predict_observations((observation(stamp),))
    with pytest.raises(RuntimeError, match=r"age_ms=600.0.*maximum_age_ms=500.0.*infer_ms=550.0"):
        runtime._install(chunk, 550., stamp + 10_000, time.monotonic())
    assert runtime.plan is None and not arms["right"].commands


def test_clock_refresh_does_not_allow_genuinely_future_result(monkeypatch):
    runtime, _, policy, _ = runtime_fixture()
    stamp = now_us()
    monkeypatch.setattr("inference.joint_runtime.now_us", lambda: stamp)
    chunk = policy.predict_observations((observation(stamp + 10_000),))
    with pytest.raises(RuntimeError, match="age_ms=-10.0"):
        runtime._install(chunk, .5, stamp, time.monotonic())
    assert runtime.plan is None


def test_no_new_aligned_observations_are_not_resubmitted():
    runtime, arms, policy, _ = runtime_fixture()
    runtime.config.execution["warmup_steps"] = 0
    runtime.config.execution["maximum_prediction_age_s"] = .16
    runtime.config.observation["maximum_alignment_gap_s"] = .001
    # Model an aligner which produces one observation then cannot align new
    # frames. Its last observation must not be submitted again indefinitely.
    post_enable_observed = False
    def observe_once(*args):
        nonlocal post_enable_observed
        # One preview before enable, one actual prediction after enable.
        if not runtime.enabled or not post_enable_observed:
            post_enable_observed = bool(runtime.enabled)
            item = observation(now_us())
            runtime.observations.append(item)
            return (item,)
        return None
    runtime._observe = observe_once
    with pytest.raises(RuntimeError, match="Policy target expired"):
        runtime.run(.3)
    assert len(policy.windows) == 2
    assert arms["right"].stops == 1


def test_ready_prediction_replaces_tail_before_source_age_budget_expires():
    runtime, arms, policy, _ = runtime_fixture(delay=.01)
    runtime.config.execution.update(warmup_steps=0, maximum_prediction_age_s=.18)
    # A whole chunk lasts .24 s, longer than the observation age budget.
    # A ready fresh plan must replace its otherwise expired tail.
    policy.step_s = .08
    runtime.run(.4)
    assert runtime.predictions >= 2
    assert arms["right"].commands and arms["right"].stops == 1


@pytest.mark.parametrize("margin_s, expires", [(0.000001, True), (.1, False)])
def test_prefetch_margin_handles_latency_increase_before_target_expires(margin_s, expires):
    runtime, arms, policy, _ = runtime_fixture(delay=.12)
    runtime.config.execution.update(warmup_steps=0, maximum_prediction_age_s=.6,
                                    prediction_prefetch_margin_s=margin_s)
    policy.step_s = .15  # 450 ms of actions; source age includes the 120 ms request.
    predict = policy.predict_observations
    def jitter(window):
        # Preview and first executable request take 120 ms. The next request
        # takes 180 ms, exhausting the old scheduler's one-tick slack.
        if len(policy.windows) == 2:
            time.sleep(.06)
        return predict(window)
    policy.predict_observations = jitter
    if expires:
        with pytest.raises(RuntimeError, match="Policy target expired"):
            runtime.run(1.7)
        assert runtime.predictions == 1
    else:
        runtime.run(1.7)
        assert runtime.predictions >= 3
    assert arms["right"].commands and arms["right"].stops == 1
    assert not arms["right"]._connected


def test_prefetch_retains_recent_slow_request_then_recovers():
    runtime, _, _, _ = runtime_fixture()
    runtime.config.execution["prediction_prefetch_margin_s"] = .15
    runtime.inference_ms = 283.
    assert runtime._prefetch_s() == pytest.approx(.443)
    runtime.prediction_latencies_s.extend([.283, .35, .2])
    assert runtime._prefetch_s() == pytest.approx(.51)
    # Eight subsequent fast responses let the old spike leave the budget.
    runtime.prediction_latencies_s.extend([.2] * 8)
    assert runtime._prefetch_s() == pytest.approx(.36)


def test_expiry_error_contains_prediction_and_alignment_diagnostics():
    runtime, arms, policy, _ = runtime_fixture(delay=.01)
    runtime.config.execution.update(warmup_steps=0, maximum_prediction_age_s=.16)
    policy.step_s = .02
    predict = policy.predict_observations
    def stall_next_prediction(window):
        # Preview and first real prediction complete normally, then stall.
        if len(policy.windows) >= 2:
            time.sleep(.2)
        return predict(window)
    policy.predict_observations = stall_next_prediction
    with pytest.raises(RuntimeError, match=r"Policy target expired.*infer_ms=.*in_flight.*observation_age_ms"):
        runtime.run(.4)
    assert arms["right"].stops == 1 and not arms["right"]._connected


@pytest.mark.parametrize("open_loop", [False, True])
def test_invalid_joint_limits_reject_entire_chunk_before_either_arm_commands(open_loop):
    runtime, arms, _, _ = runtime_fixture(("left", "right"))
    runtime.config.execution["open_loop"] = open_loop
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


def test_host_receive_feedback_age_does_not_include_sdk_request_wait(monkeypatch):
    runtime, _, _, _ = runtime_fixture()
    stamp = now_us()
    monkeypatch.setattr('inference.joint_runtime.now_us', lambda: stamp)
    state = SimpleNamespace(q_acquired_timestamp_us=stamp - 200_000,
                            acquired_timestamp_us=stamp - 200_000, timestamp_us=stamp - 1_000,
                            q_timestamp_us=stamp - 1_000, timestamp_source='host_receive', q=np.zeros(7))
    runtime.readers['right'].history.append(state)
    assert runtime._states(stamp)['right'][-1] is state
    assert state_time(state) == stamp - 1_000


@pytest.mark.parametrize('offset,age_ms', [(-160_000, 160.), (1_000, -1.)])
def test_feedback_expiry_reports_age_and_reader_diagnostics(monkeypatch, offset, age_ms):
    runtime, _, _, _ = runtime_fixture()
    stamp = now_us()
    monkeypatch.setattr('inference.joint_runtime.now_us', lambda: stamp)
    state = SimpleNamespace(q_acquired_timestamp_us=stamp + offset, acquired_timestamp_us=stamp + offset,
                            timestamp_us=stamp + offset, q_timestamp_us=stamp + offset,
                            timestamp_source='host_receive', q=np.zeros(7))
    runtime.readers['right'].history.append(state)
    with pytest.raises(RuntimeError, match='xArm feedback expired on right') as error:
        runtime._states(stamp)
    assert f'age_ms={age_ms:.1f}' in str(error.value)
    assert 'maximum_age_ms=150.0' in str(error.value)
    assert 'last_read_ms' in str(error.value) and 'read_in_flight_ms' in str(error.value)


def test_host_receive_alignment_waits_for_reply_arrival_not_request_start(monkeypatch):
    runtime, _, _, cameras = runtime_fixture()
    stamp = now_us()
    anchor = stamp - 10_000
    monkeypatch.setattr('inference.joint_runtime.now_us', lambda: stamp)
    cameras.poll = lambda: [CameraFrame(camera.name, anchor, np.zeros((4, 5, 3), np.uint8))
                            for camera in cameras.cameras]
    states = [SimpleNamespace(q_acquired_timestamp_us=anchor - 20_000,
                              acquired_timestamp_us=anchor - 20_000,
                              timestamp_us=anchor + offset, q_timestamp_us=anchor + offset,
                              timestamp_source='host_receive', q=np.full(7, value))
              for offset, value in ((-1_000, 1.), (1_000, 9.))]
    window = runtime._observe({'right': states}, stamp)
    np.testing.assert_array_equal(window[-1].q, np.ones(7))


def test_state_reader_uses_joint_only_path_when_available():
    entered = threading.Event()
    stamp = now_us()
    state = SimpleNamespace(q_valid=True, q=np.zeros(7), timestamp_us=stamp,
                            acquired_timestamp_us=stamp, q_acquired_timestamp_us=stamp)
    arm = SimpleNamespace(name='offline')
    def joint_state():
        entered.set()
        return state
    def full_state():
        raise AssertionError('unnecessary full SDK state read')
    arm.read_joint_state, arm.read_state = joint_state, full_state
    reader = StateReader(arm, .01)
    reader.thread.start()
    try:
        assert entered.wait(timeout=1.)
    finally:
        reader.stop.set()
        reader.thread.join(timeout=1.)
    assert reader.snapshot()[-1] is state
    assert reader.diagnostics()['reads'] >= 1


def test_feedback_disconnection_stops_open_loop_commands():
    runtime, arms, _, _ = runtime_fixture()
    runtime.config.execution.update(open_loop=True, warmup_steps=0)
    runtime.config.observation['maximum_state_age_s'] = .04
    observed_active_plan = False
    def keys(_):
        nonlocal observed_active_plan
        if runtime.commands >= 3 and not observed_active_plan:
            observed_active_plan = True
            runtime.readers['right'].stop.set()  # Freeze the real mock reader's last sample.
        return None
    with pytest.raises(RuntimeError, match='xArm feedback expired on right'):
        runtime.run(.35, read_key=keys)
    assert observed_active_plan and arms['right'].stops == 1


def test_post_enable_observation_requires_new_frames_and_states_from_every_source():
    runtime, _, _, cameras = runtime_fixture()
    stamp = now_us()
    runtime.execution_observation_after_us = stamp - 20_000
    anchor = stamp - 10_000
    cameras.poll = lambda: []
    rgb = np.zeros((4, 5, 3), np.uint8)
    runtime.frames["left_wrist"].extend([
        CameraFrame("left_wrist", stamp - 25_000, rgb),
        CameraFrame("left_wrist", stamp - 5_000, rgb)])
    runtime.frames["right_wrist"].append(CameraFrame("right_wrist", anchor, rgb))
    state = SimpleNamespace(q_acquired_timestamp_us=stamp - 21_000, acquired_timestamp_us=0,
                            timestamp_us=stamp - 21_000, q=np.zeros(7))
    assert runtime._observe({"right": [state]}, stamp) is None
    assert "post-enable camera frame: left_wrist" in runtime.observation_wait
    runtime.frames["left_wrist"].insert(1, CameraFrame("left_wrist", stamp - 15_000, rgb))
    assert runtime._observe({"right": [state]}, stamp) is None
    assert "post-enable xArm state: right" in runtime.observation_wait
    state.q_acquired_timestamp_us = anchor - 1_000
    window = runtime._observe({"right": [state]}, stamp)
    assert window[-1].timestamp_us == anchor
    assert all(value >= runtime.execution_observation_after_us
               for value in window[-1].image_timestamps_us.values())


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
    assert config.execution["prediction_prefetch_margin_s"] == .1
    assert config.execution["open_loop"] is False
    assert config.execution["reset_on_start"] is False


@pytest.mark.parametrize("margin", [-.1, 0., float("nan"), float("inf")])
def test_invalid_prediction_prefetch_margin_is_rejected(tmp_path, margin):
    path = deployment_file(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["execution"] = {"prediction_prefetch_margin_s": margin}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="prediction_prefetch_margin_s"):
        JointDeployment.load(path)


@pytest.mark.parametrize("key", ["open_loop", "reset_on_start", "quiet", "log_timing"])
def test_execution_flags_require_boolean(tmp_path, key):
    path = deployment_file(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["execution"] = {key: "false"}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=key):
        JointDeployment.load(path)


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


def test_mode_switch_discards_cached_and_queued_frames_then_recovers(monkeypatch):
    runtime, arms, _, cameras = runtime_fixture()
    runtime.config.observation["maximum_camera_age_s"] = .3
    clock = {"stamp": now_us()}
    monkeypatch.setattr("inference.joint_runtime.now_us", lambda: clock["stamp"])
    rgb = np.zeros((4, 5, 3), np.uint8)
    before_enable = clock["stamp"]
    for name, history in runtime.frames.items():
        history.append(CameraFrame(name, before_enable, rgb))
    runtime.observations.append(observation(before_enable))

    def state(stamp):
        return SimpleNamespace(q_acquired_timestamp_us=stamp,
                               acquired_timestamp_us=stamp, timestamp_us=stamp,
                               q=np.zeros(7))

    runtime.readers["right"].history.append(state(before_enable))
    enable = arms["right"].enable

    def slow_enable():
        enable()
        # Model a 400 ms SDK switch without sleeping or using real hardware.
        clock["stamp"] += 400_000
        runtime.readers["right"].history.append(state(clock["stamp"] - 1_000))

    arms["right"].enable = slow_enable
    arms["right"].connect()
    try:
        runtime._prepare_arms()
        assert runtime.execution_observation_after_us == clock["stamp"]
        assert not runtime.observations
        assert all(not history for history in runtime.frames.values())
        histories = {"right": runtime.readers["right"].snapshot()}
        cameras.poll = lambda: []
        assert runtime._observe(histories, clock["stamp"]) is None

        # IPC may still deliver a packet captured before the blocking switch.
        cameras.poll = lambda: [CameraFrame(name, before_enable, rgb)
                               for name in runtime.frames]
        assert runtime._observe(histories, clock["stamp"]) is None
        assert all(not history for history in runtime.frames.values())

        clock["stamp"] += 20_000
        acquired = clock["stamp"] - 1_000
        cameras.poll = lambda: [CameraFrame(name, acquired, rgb)
                               for name in runtime.frames]
        window = runtime._observe({"right": [state(acquired)]}, clock["stamp"])
        assert window[-1].timestamp_us == acquired
        assert all(stamp >= runtime.execution_observation_after_us
                   for stamp in window[-1].image_timestamps_us.values())
    finally:
        runtime._stop_motion()
        arms["right"].disconnect()


def test_startup_ignores_queued_old_frames_and_waits_for_fresh_samples():
    runtime, arms, policy, cameras = runtime_fixture(enabled=False)
    runtime.config.execution["warmup_steps"] = 0
    runtime.config.observation["maximum_camera_age_s"] = .3
    poll = cameras.poll
    first_poll = True

    def initially_queued_frames():
        nonlocal first_poll
        if first_poll:
            first_poll = False
            stale = now_us() - 400_000
            return [CameraFrame(camera.name, stale, np.zeros((4, 5, 3), np.uint8))
                    for camera in cameras.cameras]
        return poll()

    cameras.poll = initially_queued_frames
    runtime.run(.15)
    assert runtime.predictions >= 1
    assert policy.windows
    assert all(window[-1].timestamp_us >= runtime.execution_observation_after_us
               for window in policy.windows)
    assert arms["right"].enables == 0 and not arms["right"].commands


def test_no_post_enable_frames_waits_for_startup_timeout_and_stops_arm():
    runtime, arms, _, cameras = runtime_fixture()
    runtime.config.execution.update(warmup_steps=0, startup_timeout_s=.12)
    runtime.config.observation["maximum_camera_age_s"] = .3
    enable = arms["right"].enable
    poll = cameras.poll

    def slow_enable():
        time.sleep(.32)
        enable()

    arms["right"].enable = slow_enable
    cameras.poll = lambda: [] if arms["right"].enables else poll()
    with pytest.raises(RuntimeError, match=r"Timed out waiting for cameras"):
        runtime.run(.7)
    assert runtime.predictions == 0
    assert arms["right"].enables == arms["right"].stops == 1
    assert not arms["right"]._connected and cameras.stopped
    # Preparation holds measured q once; an expired pre-switch model target
    # must never reach the mock command sink.
    assert len(arms["right"].commands) == 1
    np.testing.assert_array_equal(arms["right"].commands[0], np.zeros(7))


@pytest.mark.parametrize("timestamp_offset,age_ms", [(10_000, -10.0), (-400_000, 400.0)])
def test_camera_expiry_reports_actual_age_and_limit(monkeypatch, timestamp_offset, age_ms):
    runtime, _, _, cameras = runtime_fixture()
    runtime.config.observation["maximum_camera_age_s"] = .3
    stamp = now_us()
    monkeypatch.setattr("inference.joint_runtime.now_us", lambda: stamp)
    cameras.poll = lambda: [CameraFrame(camera.name, stamp + timestamp_offset,
                                       np.zeros((4, 5, 3), np.uint8))
                            for camera in cameras.cameras]
    with pytest.raises(RuntimeError, match="Checkpoint camera expired") as error:
        runtime._observe({"right": []}, stamp)
    assert f"age_ms={age_ms:.1f}" in str(error.value)
    assert "maximum_age_ms=300.0" in str(error.value)


@pytest.mark.parametrize("lead_us", [1, 100, 5_000])
def test_small_camera_clock_lead_waits_until_frame_is_causal(monkeypatch, lead_us):
    runtime, _, _, cameras = runtime_fixture()
    stamp = now_us()
    clock = {"now": stamp}
    monkeypatch.setattr("inference.joint_runtime.now_us", lambda: clock["now"])
    frames = [CameraFrame(camera.name, stamp + lead_us,
                          np.zeros((4, 5, 3), np.uint8)) for camera in cameras.cameras]
    cameras.poll = lambda: frames
    state = SimpleNamespace(q_acquired_timestamp_us=stamp,
                            acquired_timestamp_us=stamp, timestamp_us=stamp,
                            q=np.zeros(7))
    histories = {"right": [state]}
    assert runtime._observe(histories, stamp) is None
    assert "waiting for host clock" in runtime.observation_wait
    assert not runtime.observations
    # Retain the original device-mapped timestamp, not a clamped replacement.
    assert all(history[-1].timestamp_us == stamp + lead_us for history in runtime.frames.values())
    cameras.poll = lambda: []
    clock["now"] += lead_us
    window = runtime._observe(histories, clock["now"])
    assert window[-1].timestamp_us == stamp + lead_us
    assert window[-1].timestamp_us <= clock["now"]


def test_clock_refresh_after_camera_poll_accepts_newly_acquired_frame(monkeypatch):
    runtime, _, _, cameras = runtime_fixture()
    stamp = now_us()
    clock = {"now": stamp}
    monkeypatch.setattr("inference.joint_runtime.now_us", lambda: clock["now"])
    def poll():
        clock["now"] += 10_000
        return [CameraFrame(camera.name, clock["now"], np.zeros((4, 5, 3), np.uint8))
                for camera in cameras.cameras]
    cameras.poll = poll
    state = SimpleNamespace(q_acquired_timestamp_us=stamp,
                            acquired_timestamp_us=stamp, timestamp_us=stamp,
                            q=np.zeros(7))
    window = runtime._observe({"right": [state]}, stamp)
    assert window[-1].timestamp_us == clock["now"]


def test_small_clock_lead_does_not_hide_expired_other_camera(monkeypatch):
    runtime, _, _, cameras = runtime_fixture()
    stamp = now_us()
    monkeypatch.setattr("inference.joint_runtime.now_us", lambda: stamp)
    rgb = np.zeros((4, 5, 3), np.uint8)
    cameras.poll = lambda: [CameraFrame("left_wrist", stamp + 100, rgb),
                            CameraFrame("right_wrist", stamp - 200_000, rgb)]
    with pytest.raises(RuntimeError, match="Checkpoint camera expired: right_wrist"):
        runtime._observe({"right": []}, stamp)
    assert not runtime.observations
