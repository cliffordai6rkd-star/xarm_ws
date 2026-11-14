"""Remote-policy integration with synthetic cameras, arms and dynamics."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from inference.forcevla_config import ForceVLADeployment
from inference.forcevla_runtime import ForceVLAInferenceRuntime, TauExtState, RemoteForceVLAPolicy
from inference.core.contracts import ActionChunk
from xarm_stack.causal_tau_ext import CausalTauExt, TauExtConfig
from nero_collection.arms.mock import MockArm
from nero_collection.cameras import CameraFrame
from nero_collection.time_utils import now_us


DEPLOYMENT = Path(__file__).resolve().parents[1] / "inference/configs/forcevla_xarm.yaml"


class TrackingArm(MockArm):
    def __post_init__(self):
        super().__post_init__()
        self.enables = self.stops = self.reads = self.report_configurations = 0
        self.commands = []
        self.invalidate_after_command = False

    def configure_feedback_report(self):
        self.report_configurations += 1

    def read_state(self):
        self.reads += 1
        state = super().read_state()
        if self.invalidate_after_command and self.commands:
            state.torque_valid = False
        return state

    def enable(self):
        super().enable()
        self.enables += 1

    def command_joint_positions(self, q):
        self.commands.append(np.asarray(q).copy())
        super().command_joint_positions(q)

    def stop_motion(self):
        self.stops += 1


class FakeCameras:
    colors = {"front": (230, 10, 20), "right_wrist": (2, 3, 200)}

    def __init__(self):
        self.cameras = [SimpleNamespace(name=name) for name in self.colors]
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def poll(self):
        stamp = now_us()
        return [CameraFrame(name, stamp, np.full((6, 8, 3), color, np.uint8))
                for name, color in self.colors.items()]


class FakeClient:
    metadata = {}

    def __init__(self, *, delay=0., fail_on=None, delays=None):
        self.delay, self.fail_on = delay, fail_on
        self.delays = delays
        self.requests = []
        self.closed = False
        self.arm = None
        self.samples_during_inference = []

    def connect(self):
        return self.metadata

    def infer(self, request):
        self.requests.append(request)
        samples_before = self.arm.reads
        delay = self.delays[min(len(self.requests)-1, len(self.delays)-1)] if self.delays else self.delay
        if delay:
            time.sleep(delay)
        self.samples_during_inference.append(self.arm.reads - samples_before)
        if self.fail_on == len(self.requests):
            raise RuntimeError("remote model failure")
        # Small absolute-q targets exercise the real limiter within URDF limits.
        actions = np.tile(request["state"], (50, 1))
        actions[:, 0] += .005
        return {"actions": actions, "server_timing": {"infer_ms": delay*1000}}

    def close(self):
        self.closed = True


def runtime_fixture(*, enabled=False, delay=0., fail_on=None, delays=None):
    config = ForceVLADeployment.load(DEPLOYMENT)
    config.policy.update(image_size=[6, 8], execute_steps=3)
    config.execution.update(startup_timeout_s=1., maximum_prediction_age_s=1.)
    config.observation.update(maximum_state_age_s=.3, maximum_camera_age_s=.3)
    arm = TrackingArm(config.endpoints["right"])
    cameras = FakeCameras()
    client = FakeClient(delay=delay, fail_on=fail_on, delays=delays)
    client.arm = arm
    policy = RemoteForceVLAPolicy(config, client)
    runtime = ForceVLAInferenceRuntime(config, policy, backend="mock", command_enabled=enabled,
                                      arms={"right": arm}, camera_manager=cameras)
    return runtime, arm, cameras, client


def test_shipped_config_loads_hardware_contract_without_connecting():
    config = ForceVLADeployment.load(DEPLOYMENT)
    options = config.endpoints["right"].config_kwargs
    assert config.active_arms == ("right",)
    assert options["robot_ip"] == "192.168.1.196"
    assert options["feedback_signal"] == "torque"
    assert options["velocity_source"] == "hardware"
    assert not options["execution_enabled"]
    assert config.joint_limits["right"].shape == (7, 2)
    assert config.observation["low_dim_sources"] == {"tau_ext": "tau_ext"}


def test_read_only_sends_rgb_and_training_matched_tau_without_enabling_arm():
    runtime, arm, cameras, client = runtime_fixture()
    runtime.run(.24)
    assert runtime.predictions >= 1
    assert arm.report_configurations == 1
    assert arm.enables == arm.stops == 0
    assert not arm.commands
    request = client.requests[0]
    assert set(request) == {"state", "tau_ext", "prompt", "image", "right_wrist_image"}
    assert request["state"].dtype == request["tau_ext"].dtype == np.float32
    np.testing.assert_allclose(request["state"], arm.config.rest_q, atol=1e-6)
    reference = CausalTauExt(runtime.config.tau_ext["urdf_path"],
                             TauExtConfig(**runtime.config.tau_ext["processing_config"]))
    expected = reference.update(arm.config.rest_q, np.zeros(7), .05*np.sin(arm.config.rest_q), 0.)
    np.testing.assert_array_equal(request["tau_ext"], expected["tau_ext"])
    for key, name in (("image", "front"), ("right_wrist_image", "right_wrist")):
        image = request[key]
        assert image.dtype == np.uint8 and image.shape == (6, 8, 3)
        np.testing.assert_array_equal(image[0, 0], FakeCameras.colors[name])
    assert cameras.stopped and client.closed and not arm._connected


def test_fixed_rate_tau_sampling_continues_while_remote_inference_is_delayed():
    runtime, arm, _, client = runtime_fixture(delay=.12)
    runtime.run(.34)
    assert runtime.predictions >= 1
    assert client.samples_during_inference[0] >= 5
    history = runtime.readers["right"].snapshot()
    assert len(history) >= 5
    np.testing.assert_array_equal(np.diff([s.observation_timestamp_us for s in history]), 40_000)
    assert history[-1].timestamp_us - history[0].timestamp_us >= 150_000
    assert all(state.tau_ext_valid and np.isfinite(state.tau_ext).all() for state in history)
    assert arm.enables == 0


def test_joint_execution_respects_limits_and_stops_on_exit():
    runtime, arm, cameras, client = runtime_fixture(enabled=True)
    runtime.run(.24)
    assert runtime.predictions >= 1 and arm.commands
    assert arm.enables == arm.stops == 1
    commands = np.stack(arm.commands)
    limits = runtime.config.joint_limits["right"]
    assert np.all(commands >= limits[:, 0]) and np.all(commands <= limits[:, 1])
    assert commands[-1, 0] > arm.config.rest_q[0]
    assert cameras.stopped and client.closed and not arm._connected


def test_remote_error_after_execution_stops_motion_and_cleans_up():
    runtime, arm, cameras, client = runtime_fixture(enabled=True, fail_on=2)
    with pytest.raises(RuntimeError, match="Policy inference failed"):
        runtime.run(.5)
    assert arm.commands and arm.enables == arm.stops == 1
    assert cameras.stopped and client.closed and not arm._connected


def test_loss_of_measured_torque_stops_motion():
    runtime, arm, cameras, client = runtime_fixture(enabled=True)
    arm.invalidate_after_command = True
    with pytest.raises(RuntimeError, match="state read failed") as error:
        runtime.run(.5)
    assert "measured finite q/dq/torque" in str(error.value.__cause__)
    assert arm.commands and arm.enables == arm.stops == 1
    assert cameras.stopped and client.closed and not arm._connected


def test_repeated_stale_initial_results_never_enable_or_command_arm():
    runtime, arm, cameras, client = runtime_fixture(enabled=True, delay=.08)
    runtime.config.execution["maximum_prediction_age_s"] = .025
    runtime.run(.25)
    assert client.requests and runtime.predictions == 0
    assert arm.enables == arm.stops == 0 and not arm.commands
    assert cameras.stopped and client.closed and not arm._connected


def test_stale_initial_compilation_result_is_discarded_then_fresh_result_executes(caplog):
    runtime, arm, _, client = runtime_fixture(enabled=True, delays=[.12, 0.])
    runtime.config.execution["maximum_prediction_age_s"] = .08
    runtime.config.policy["execute_steps"] = 1
    with caplog.at_level("INFO", logger="inference.forcevla_runtime"):
        runtime.run(.32)
    assert "Discarding first stale ForceVLA result" in caplog.text
    assert len(client.requests) >= 2 and runtime.predictions >= 1
    assert arm.commands and arm.enables == arm.stops == 1


def test_network_stall_after_execution_expires_target_and_stops_motion():
    runtime, arm, cameras, client = runtime_fixture(enabled=True, delays=[0., .18])
    runtime.config.execution["maximum_prediction_age_s"] = .12
    runtime.config.policy["execute_steps"] = 2
    with pytest.raises(RuntimeError, match="target expired"):
        runtime.run(.5)
    assert arm.commands and arm.enables == arm.stops == 1
    assert cameras.stopped and client.closed and not arm._connected


def test_camera_alignment_uses_q_and_tau_from_the_same_causal_state():
    runtime, arm, cameras, client = runtime_fixture()
    stamp = now_us()
    anchor = stamp - 50_000
    old_wrist = CameraFrame("right_wrist", anchor - 20_000, np.full((6, 8, 3), 17, np.uint8))
    runtime.frames["right_wrist"].append(old_wrist)
    cameras.poll = lambda: [
        CameraFrame("front", anchor, np.full((6, 8, 3), 11, np.uint8)),
        CameraFrame("right_wrist", anchor+1_000, np.full((6, 8, 3), 99, np.uint8)),
    ]
    arm.connect()
    base = arm.read_state()
    arm.disconnect()
    states = []
    for offset, value in ((-1_000, 1.), (1_000, 9.)):
        state = TauExtState(**base.__dict__, tau_ext=np.full(7, 2*value), tau_ext_valid=True)
        states.append(replace(state, q=np.full(7, value), timestamp_us=anchor+offset,
                              q_acquired_timestamp_us=anchor+offset, acquired_timestamp_us=anchor+offset))
    window = runtime._observe({"right": states}, stamp)
    observation = window[-1]
    assert observation.timestamp_us == anchor
    np.testing.assert_array_equal(observation.q, np.ones(7))
    np.testing.assert_array_equal(observation.metadata["policy_state"]["tau_ext"], np.full(7, 2.))
    assert np.all(observation.images["right_wrist_image"] == 17)
    assert all(timestamp <= anchor for timestamp in observation.image_timestamps_us.values())
    chunk = runtime.policy.predict_observations(window)
    assert chunk.timestamp_us == anchor and chunk.values.shape == (50, 7)
    np.testing.assert_array_equal(client.requests[0]["tau_ext"], np.full(7, 2., np.float32))


def test_reply_straddling_image_capture_is_excluded_by_observation_timestamp():
    runtime, arm, cameras, _ = runtime_fixture()
    stamp = now_us()
    anchor = stamp - 50_000
    cameras.poll = lambda: [CameraFrame(camera.name, anchor, np.zeros((6, 8, 3), np.uint8))
                            for camera in cameras.cameras]
    arm.connect()
    base = arm.read_state()
    arm.disconnect()
    observed = TauExtState(**base.__dict__, tau_ext=np.full(7, 2.), tau_ext_valid=True)
    old = replace(observed, q=np.ones(7), timestamp_us=anchor-10_000,
                  q_timestamp_us=anchor-10_000, observation_timestamp_us=anchor-10_000,
                  q_acquired_timestamp_us=anchor-20_000, acquired_timestamp_us=anchor-20_000)
    # The next query starts before image capture but its reply arrives after.
    # Acquisition-start metadata must remain intact while alignment excludes it.
    straddling = replace(observed, q=np.full(7, 9.), tau_ext=np.full(7, 18.),
                         timestamp_us=anchor+1_000, q_timestamp_us=anchor+1_000,
                         observation_timestamp_us=anchor+1_000,
                         q_acquired_timestamp_us=anchor-5_000, acquired_timestamp_us=anchor-5_000)
    window = runtime._observe({"right": [old, straddling]}, stamp)
    np.testing.assert_array_equal(window[-1].q, old.q)
    np.testing.assert_array_equal(window[-1].metadata["policy_state"]["tau_ext"], old.tau_ext)
    assert straddling.q_acquired_timestamp_us < anchor < straddling.observation_timestamp_us


def test_delayed_prefetched_actions_follow_observation_time_instead_of_replaying_prefix():
    runtime, arm, _, _ = runtime_fixture()
    actions = np.tile(arm.config.rest_q, (50, 1))
    actions[:, 0] += .001 * np.arange(50)
    chunk = ActionChunk(actions, "joint", None, 1_000_000, .04)
    # 215 ms includes network time and waiting for the preceding plan to finish.
    runtime._install(chunk, 90., 1_215_000, 10.)
    assert runtime.plan.values.shape == (3, 7)
    np.testing.assert_array_equal(runtime.plan.values, actions[5:8])
    assert runtime.plan.metadata["elapsed_action_steps"] == 5
    assert runtime.plan_started == pytest.approx(9.985)
    runtime._execute({}, 1_215_000, 10., .01)
    np.testing.assert_array_equal(runtime.last_target, actions[5])
    runtime._execute({}, 1_241_000, 10.026, .01)
    np.testing.assert_array_equal(runtime.last_target, actions[6])


def test_elapsed_full_horizon_is_discarded_before_motion_and_stops_during_execution():
    runtime, arm, _, _ = runtime_fixture()
    runtime.config.execution["maximum_prediction_age_s"] = 3.
    actions = np.tile(arm.config.rest_q, (50, 1))
    chunk = ActionChunk(actions, "joint", None, 1_000_000, .04)
    runtime._install(chunk, 2100., 3_100_000, 10.)
    assert runtime.plan is None and runtime.predictions == 0 and not arm.commands
    runtime._install(chunk, 10., 1_010_000, 10.)
    with pytest.raises(RuntimeError, match="action horizon expired"):
        runtime._install(chunk, 2100., 3_100_000, 12.1)


def test_initial_enable_wait_advances_the_first_sent_target(monkeypatch):
    runtime, arm, _, _ = runtime_fixture(enabled=True)
    runtime.config.policy["execute_steps"] = 10
    actions = np.tile(arm.config.rest_q, (50, 1))
    actions[:, 0] += .001 * np.arange(50)
    runtime._install(ActionChunk(actions, "joint", None, 1_000_000, .04), 90., 1_215_000, 10.)
    arm.connect()
    state = arm.read_state()
    histories = {"right": [state]}
    # The enable/report wait consumes a further three policy frames.
    monkeypatch.setattr("inference.joint_runtime.now_us", lambda: 1_335_000)
    monkeypatch.setattr("inference.joint_runtime.time.monotonic", lambda: 10.12)
    monkeypatch.setattr(runtime, "_states", lambda _: histories)
    monkeypatch.setattr(runtime, "_observe", lambda *_: None)
    runtime._execute(histories, 1_215_000, 10., .01)
    np.testing.assert_array_equal(runtime.last_target, actions[8])
    assert arm.commands
    arm.disconnect()


def test_fixed_rate_sampler_holds_only_past_raw_feedback_and_resets_each_rollout():
    runtime, arm, _, _ = runtime_fixture()
    reader = runtime.readers["right"]
    arm.connect()
    base = arm.read_state()
    arm.disconnect()

    def report(offset_us, joint0):
        q = base.q.copy()
        q[0] = joint0
        return replace(base, q=q, dq=np.full(7, joint0), torque=np.full(7, joint0 * 20),
                       timestamp_us=1_000_000 + offset_us, q_timestamp_us=1_000_000 + offset_us)

    first, past, future = report(0, 0.), report(30_000, .1), report(41_000, .9)
    reader.accept_state(first)
    reader.accept_state(past)
    reader.accept_state(future)
    history = reader.snapshot()
    assert [s.observation_timestamp_us for s in history] == [1_000_000, 1_040_000]
    assert history[-1].q_timestamp_us == 1_030_000
    np.testing.assert_array_equal(history[-1].q, past.q)
    reference = CausalTauExt(runtime.config.tau_ext["urdf_path"],
                             TauExtConfig(**runtime.config.tau_ext["processing_config"]))
    reference.update(first.q, first.dq, first.torque, 0.)
    expected = reference.update(past.q, past.dq, past.torque, .04)
    np.testing.assert_array_equal(history[-1].tau_ext, expected["tau_ext"])
    reader.accept_state(future)  # A duplicated SDK report cannot advance filters.
    assert len(reader.snapshot()) == 2
    reader.reset_episode()
    assert not reader.snapshot()
    reader.accept_state(future)
    assert reader.processor.last_time == 0.
    np.testing.assert_array_equal(reader.snapshot()[0].tau_ext,
                                  CausalTauExt(runtime.config.tau_ext["urdf_path"], reference.config)
                                  .update(future.q, future.dq, future.torque, 0.)["tau_ext"])


def test_feedback_gap_stops_fixed_rate_tau_processing():
    runtime, arm, _, _ = runtime_fixture()
    reader = runtime.readers["right"]
    arm.connect()
    base = arm.read_state()
    arm.disconnect()
    reader.accept_state(replace(base, timestamp_us=1_000_000, q_timestamp_us=1_000_000))
    with pytest.raises(RuntimeError, match="feedback timeline lost"):
        reader.accept_state(replace(base, timestamp_us=1_160_000, q_timestamp_us=1_160_000))
