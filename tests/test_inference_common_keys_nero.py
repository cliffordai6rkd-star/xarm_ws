"""Nero keyboard transitions with fake models, cameras, and an in-memory arm."""

from collections import deque
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from inference.runtime import NeroInferenceRuntime


class FakeControlWorker:
    def __init__(self):
        self.running = False
        self.fault = None
        self._name = "control"
        self.manual_mode = False
        self.pending_steps = 0
        self.cycles = 0
        self.last_output = None

    def set_manual_mode(self, enabled):
        self.manual_mode = bool(enabled)
        self.pending_steps = 0

    def request_step(self):
        self.manual_mode = True
        self.pending_steps += 1


class FakeAsyncRuntime:
    """Starting a worker performs model work, making premature starts visible."""

    def __init__(self, *, world_model=True):
        self.dp_worker = SimpleNamespace(
            running=False, fault=None, _name="dp", updates=0,
        )
        self.wm_worker = (
            SimpleNamespace(running=False, fault=None, _name="wm",
                            inference_count=0, last_latency_s=0.)
            if world_model else None
        )
        self.control_worker = FakeControlWorker()
        self.starts = 0

    @property
    def cycles(self):
        return self.control_worker.cycles

    @property
    def last_output(self):
        return self.control_worker.last_output

    @property
    def model_calls(self):
        return (self.dp_worker.updates,
                0 if self.wm_worker is None else self.wm_worker.inference_count)

    def start(self):
        self.starts += 1
        self.dp_worker.running = self.control_worker.running = True
        self.dp_worker.updates += 1
        if self.wm_worker is not None:
            self.wm_worker.running = True
            self.wm_worker.inference_count += 1

    def stop(self):
        self.dp_worker.running = self.control_worker.running = False
        if self.wm_worker is not None:
            self.wm_worker.running = False

    def advance(self):
        if not self.control_worker.running:
            return
        self.dp_worker.updates += 1
        if self.wm_worker is not None:
            self.wm_worker.inference_count += 1
        if not self.control_worker.manual_mode or self.control_worker.pending_steps:
            self.control_worker.cycles += 1
            if self.control_worker.pending_steps:
                self.control_worker.pending_steps -= 1


def runtime_fixture(*, enabled=False, asynchronous=False, world_model=True,
                    unavailable_samples=0):
    """Retain runtime lifecycle methods so startup cannot hide model work."""
    runtime = object.__new__(NeroInferenceRuntime)
    runtime.config = SimpleNamespace(
        runtime=SimpleNamespace(maximum_inference_steps=None),
        predictor=SimpleNamespace(enabled=asynchronous and world_model,
                                  mode="contact_world_model"),
        execution=SimpleNamespace(mode="q"),
        observation_protection=SimpleNamespace(enabled=False),
    )
    runtime.command_enabled = enabled
    runtime._started = False
    runtime._modular_mode = False
    runtime._timestamp_async_enabled = asynchronous
    runtime._continuous_state_stream_enabled = asynchronous
    runtime._async_runtime = (
        FakeAsyncRuntime(world_model=world_model) if asynchronous else None
    )
    runtime.arm = Mock()
    state = SimpleNamespace(q=np.arange(7, dtype=np.float64) / 10,
                            q_timestamp_us=1, timestamp_us=1)
    runtime.arm.read_state.return_value = state
    runtime._wait_for_finite_arm_state = Mock(return_value=state)
    runtime._prepare_inference_control_mode = Mock()
    runtime._set_q_cmd = Mock()
    runtime._q_cmd_lock = threading.Lock()
    runtime._q_cmd_history = deque()
    runtime._state_stream = Mock(history_size=50, poll_interval_s=.01,
                                 history_rollover_count=0)
    runtime.state_history_buffer = Mock()
    runtime.action_plan_buffer = Mock()
    runtime.action_plan_buffer.snapshot.return_value = []
    runtime.wm_target_buffer = Mock()
    runtime.wm_target_buffer.snapshot.return_value = []
    runtime.pipeline = Mock()
    runtime._observation_sampler = Mock()
    runtime.online_tau_ext = Mock()
    runtime.cameras = Mock()
    runtime.wrench_plotter = Mock()
    runtime.tau_ext_plotter = Mock()
    runtime.mujoco_visualizer = Mock()
    runtime._wrench_filter = Mock()
    runtime._latest_frames = {}
    runtime._async_last_image_timestamps = None
    record = SimpleNamespace(attempts=0, predictions=0, reset_reasons=[])

    def reset(reason):
        record.reset_reasons.append(reason)
        runtime.arm.enable()
        runtime.arm.move_joints(np.zeros(7))

    def step():
        record.attempts += 1
        if record.attempts <= unavailable_samples:
            return None
        record.predictions += 1
        return object()

    runtime._reset_arm_to_rest = reset
    runtime.step = step
    return runtime, record


def test_start_paused_does_not_predict_until_s():
    runtime, record = runtime_fixture()
    keys = iter([None, None, None, "q"])
    assert runtime.run(.5, read_key=lambda _: next(keys), start_paused=True) == 0
    assert record.attempts == record.predictions == 0
    runtime.arm.enable.assert_not_called()
    runtime.arm.move_joints.assert_not_called()


def test_s_starts_continuous_inference_by_default():
    runtime, record = runtime_fixture()
    keys = iter([None, "s", None, None, None, "q"])

    def read_key(_):
        key = next(keys)
        if key == "s":
            assert record.predictions == 0
        return key

    assert runtime.run(.5, read_key=read_key, start_paused=True) == 4
    assert record.predictions == 4


def test_programmatic_run_remains_continuous_by_default():
    runtime, record = runtime_fixture()
    keys = iter([None, None, "q"])
    assert runtime.run(.5, read_key=lambda _: next(keys)) == 2
    assert record.predictions == 2


def test_i_resets_once_then_waits_for_s_and_retains_final_reset():
    runtime, record = runtime_fixture(enabled=True)
    keys = iter(["s", None, "i", None, None, None, "s", None, "q"])
    paused_predictions = None

    def read_key(_):
        nonlocal paused_predictions
        key = next(keys)
        if key == "i":
            paused_predictions = record.predictions
        elif paused_predictions is not None:
            assert record.predictions == paused_predictions
            assert record.reset_reasons.count("episode boundary") == 1
            if key == "s":
                paused_predictions = None
        return key

    assert runtime.run(.5, read_key=read_key, start_paused=True) == 4
    assert record.reset_reasons == ["startup", "episode boundary", "shutdown"]


def test_p_holds_without_reset_and_s_resumes():
    runtime, record = runtime_fixture(enabled=True)
    keys = iter(["s", None, "p", None, None, None, "s", None, "q"])
    paused_predictions = None

    def read_key(_):
        nonlocal paused_predictions
        key = next(keys)
        if key == "p":
            paused_predictions = record.predictions
        elif paused_predictions is not None:
            assert record.predictions == paused_predictions
            assert record.reset_reasons == ["startup"]
            if key == "s":
                paused_predictions = None
        return key

    assert runtime.run(.5, read_key=read_key, start_paused=True) == 4
    assert record.reset_reasons == ["startup", "shutdown"]
    assert any(np.array_equal(call.args[0], runtime.arm.read_state.return_value.q)
               for call in runtime.arm.move_joints.call_args_list)


@pytest.mark.parametrize("stop_key", ["i", "p"])
def test_read_only_stop_does_not_enable_or_move_arm(stop_key):
    runtime, record = runtime_fixture()
    keys = iter(["s", stop_key, None, None, "q"])
    assert runtime.run(.5, read_key=lambda _: next(keys), start_paused=True) == 1
    assert record.predictions == 1
    assert record.reset_reasons == []
    runtime.arm.enable.assert_not_called()
    runtime.arm.move_joints.assert_not_called()


def test_single_step_waits_for_one_valid_sample_then_c_runs_continuously():
    runtime, record = runtime_fixture(unavailable_samples=1)
    keys = iter([None, "s", None, None, None, "c", None, "q"])

    def read_key(_):
        key = next(keys)
        if key == "s":
            assert record.attempts == 0
        if key == "c":
            assert record.attempts == 2
            assert record.predictions == 1
        return key

    assert runtime.run(.5, read_key=read_key, single_step=True,
                       start_paused=True) == 3
    assert record.predictions == 3


@pytest.mark.parametrize("world_model", [True, False])
def test_timestamp_workers_and_wm_warmup_wait_for_s(world_model):
    runtime, record = runtime_fixture(asynchronous=True, world_model=world_model)
    asynchronous = runtime._async_runtime
    keys = iter([None, None, None, "s", None, "q"])
    running = False

    def read_key(_):
        nonlocal running
        asynchronous.advance()
        key = next(keys)
        if not running:
            assert asynchronous.starts == 0
            assert asynchronous.model_calls == (0, 0)
            runtime.pipeline.warm_up_contact_reference.assert_not_called()
        if key == "s":
            running = True
        elif running:
            assert asynchronous.dp_worker.updates > 0
            if world_model:
                assert asynchronous.wm_worker.inference_count > 0
        return key

    runtime.run(.5, read_key=read_key, start_paused=True)
    assert asynchronous.starts == 1
    assert record.attempts == 0


@pytest.mark.parametrize("stop_key", ["i", "p"])
def test_timestamp_stop_suspends_model_workers_until_s(stop_key):
    runtime, record = runtime_fixture(asynchronous=True)
    asynchronous = runtime._async_runtime
    keys = iter(["s", None, stop_key, None, None, None, "s", None, "q"])
    stopped_calls = None

    def read_key(_):
        nonlocal stopped_calls
        asynchronous.advance()
        key = next(keys)
        if key == stop_key:
            stopped_calls = asynchronous.model_calls
        elif stopped_calls is not None:
            assert asynchronous.model_calls == stopped_calls
            assert not asynchronous.dp_worker.running
            assert not asynchronous.wm_worker.running
            if key == "s":
                stopped_calls = None
        return key

    runtime.run(.5, read_key=read_key, start_paused=True)
    assert asynchronous.starts == 2
    assert record.attempts == 0
    assert record.reset_reasons == []


def test_timestamp_single_step_runs_one_cycle_then_c_resumes():
    runtime, record = runtime_fixture(asynchronous=True)
    asynchronous = runtime._async_runtime
    keys = iter([None, "s", None, None, None, "c", None, None, "q"])
    phase = "waiting"
    checked_paused = False

    def read_key(_):
        nonlocal phase, checked_paused
        asynchronous.advance()
        key = next(keys)
        if phase == "waiting":
            assert asynchronous.starts == asynchronous.cycles == 0
            if key == "s":
                phase = "single"
        elif phase == "single":
            assert asynchronous.cycles == 1
            if not asynchronous.control_worker.running:
                checked_paused = True
                assert not asynchronous.dp_worker.running
                assert not asynchronous.wm_worker.running
            if key == "c":
                assert checked_paused
                phase = "continuous"
        else:
            assert asynchronous.cycles > 1
            assert not asynchronous.control_worker.manual_mode
        return key

    runtime.run(.5, read_key=read_key, single_step=True, start_paused=True)
    assert asynchronous.starts == 2
    assert record.attempts == 0
    assert phase == "continuous"


def stop_with_live_dp(runtime, *, control_running=False):
    """A stuck model survives stop; later cleanup failures must not hide it."""
    asynchronous = runtime._async_runtime
    first_error = TimeoutError("original DP stop timeout")
    calls = []

    def stop():
        asynchronous.control_worker.running = control_running
        asynchronous.dp_worker.running = True
        if asynchronous.wm_worker is not None:
            asynchronous.wm_worker.running = False
        error = first_error if not calls else TimeoutError("cleanup DP stop timeout")
        calls.append(error)
        raise error

    asynchronous.stop = stop
    return first_error


def assert_resources_stopped(runtime):
    runtime._state_stream.stop.assert_called()
    runtime._state_stream.clear.assert_called()
    runtime.cameras.stop.assert_called_once()
    runtime.wrench_plotter.close.assert_called_once()
    runtime.tau_ext_plotter.close.assert_called_once()
    runtime.mujoco_visualizer.close.assert_called_once()
    runtime.arm.disconnect.assert_called_once()
    assert not runtime._started


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("stop_key", ["i", "q"])
def test_worker_stop_timeout_cleans_up_and_preserves_original_error(enabled, stop_key):
    runtime, record = runtime_fixture(enabled=enabled, asynchronous=True)
    original_error = stop_with_live_dp(runtime)
    keys = iter(["s", stop_key])

    def read_key(_):
        key = next(keys)
        if key == stop_key:
            # Startup may reset the model; it must remain untouched once a live
            # worker cannot be joined at the requested episode/shutdown boundary.
            runtime.pipeline.reset.reset_mock()
        return key

    with pytest.raises(TimeoutError) as error:
        runtime.run(.5, read_key=read_key, start_paused=True)

    assert error.value is original_error
    assert runtime._async_runtime.dp_worker.running
    assert not runtime._async_runtime.control_worker.running
    assert_resources_stopped(runtime)
    runtime.pipeline.reset.assert_not_called()
    runtime.pipeline.close.assert_not_called()
    runtime.arm.disable.assert_not_called()
    if enabled:
        assert record.reset_reasons == ["startup"]
        np.testing.assert_array_equal(runtime.arm.move_joints.call_args.args[0],
                                      runtime.arm.read_state.return_value.q)
    else:
        assert record.reset_reasons == []
        runtime.arm.enable.assert_not_called()
        runtime.arm.move_joints.assert_not_called()


def test_stop_state_stream_still_stops_sensing_when_model_stop_times_out():
    runtime, _ = runtime_fixture(asynchronous=True)
    runtime.start()
    original_error = stop_with_live_dp(runtime)
    runtime._state_stream.stop.reset_mock()
    runtime._state_stream.clear.reset_mock()

    with pytest.raises(TimeoutError) as error:
        runtime._stop_state_stream(clear=True)

    assert error.value is original_error
    runtime._state_stream.stop.assert_called_once()
    runtime._state_stream.clear.assert_called_once()
    assert runtime._async_runtime.dp_worker.running
    assert not runtime._async_runtime.control_worker.running


def test_direct_stop_cleans_up_and_failed_worker_cannot_resume_or_start():
    runtime, record = runtime_fixture(asynchronous=True)
    runtime.start()
    original_error = stop_with_live_dp(runtime)
    runtime.pipeline.reset.reset_mock()

    with pytest.raises(TimeoutError) as error:
        runtime.stop()

    assert error.value is original_error
    assert_resources_stopped(runtime)
    runtime.pipeline.reset.assert_not_called()
    runtime.pipeline.close.assert_not_called()
    assert record.reset_reasons == []
    runtime.arm.enable.assert_not_called()
    runtime.arm.move_joints.assert_not_called()

    runtime.pipeline.warm_up_contact_reference.reset_mock()
    runtime._contact_reference_warmed_up = False
    runtime.arm.connect.reset_mock()
    starts = runtime._async_runtime.starts
    for resume in (runtime._resume_inference, runtime.start):
        with pytest.raises((RuntimeError, TimeoutError)):
            resume()
    assert runtime._async_runtime.starts == starts
    runtime.pipeline.warm_up_contact_reference.assert_not_called()
    runtime.arm.connect.assert_not_called()


def test_live_control_worker_prevents_all_post_failure_arm_motion():
    runtime, record = runtime_fixture(enabled=True, asynchronous=True)
    original_error = stop_with_live_dp(runtime, control_running=True)
    keys = iter(["s", "i"])

    def read_key(_):
        key = next(keys)
        if key == "i":
            runtime.arm.enable.reset_mock()
            runtime.arm.move_joints.reset_mock()
            runtime.pipeline.reset.reset_mock()
        return key

    with pytest.raises(TimeoutError) as error:
        runtime.run(.5, read_key=read_key, start_paused=True)

    assert error.value is original_error
    assert runtime._async_runtime.dp_worker.running
    assert runtime._async_runtime.control_worker.running
    assert_resources_stopped(runtime)
    assert record.reset_reasons == ["startup"]
    runtime.arm.enable.assert_not_called()
    runtime.arm.move_joints.assert_not_called()
    runtime.arm.disable.assert_not_called()
    runtime.pipeline.reset.assert_not_called()
    runtime.pipeline.close.assert_not_called()
