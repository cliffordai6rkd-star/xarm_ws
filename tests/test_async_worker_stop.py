"""Cancelled slow predictions cannot republish plans or overlap a new run."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from inference.async_fast_slow import (
    ActionPlanBuffer,
    DPWorker,
    StateHistoryBuffer,
    TimestampFastSlowRuntime,
    WMTargetBuffer,
    WMWorker,
    monotonic_time,
)


@pytest.mark.parametrize("old_prediction_raises", [False, True])
def test_dp_timeout_retains_thread_discards_old_work_and_prevents_parallel_restart(
    old_prediction_raises,
):
    entered, release, next_entered, published = (threading.Event() for _ in range(4))
    calls, publications = [], []
    active = {"count": 0, "maximum": 0}
    lock = threading.Lock()
    buffer = ActionPlanBuffer()

    def infer(observation):
        with lock:
            calls.append(observation)
            active["count"] += 1
            active["maximum"] = max(active["maximum"], active["count"])
        try:
            if observation == "old":
                entered.set()
                assert release.wait(timeout=1.)
                if old_prediction_raises:
                    raise RuntimeError("cancelled old prediction failure")
            else:
                next_entered.set()
            return np.full((3, 7), 2. if observation == "new" else 1.)
        finally:
            with lock:
                active["count"] -= 1

    def on_publish(plan):
        publications.append(plan)
        published.set()

    worker = DPWorker(infer, buffer, on_publish=on_publish)
    try:
        worker.submit("old")  # Preserve the existing submit-before-start contract.
        worker.start()
        assert entered.wait(timeout=1.)
        old_thread = worker._thread
        worker.submit("queued old observation")
        with pytest.raises(TimeoutError, match="Timed out stopping worker"):
            worker.stop(timeout_s=.01)
        assert worker.running and worker._thread is old_thread
        with pytest.raises(RuntimeError, match="still stopping"):
            worker.start()
        worker.submit("observation submitted while stopped")
        release.set()
        worker.stop(timeout_s=.5)
        assert not worker.running
        assert not buffer.snapshot() and not publications
        assert worker.updates == 0 and worker.fault is None
        assert calls == ["old"]

        worker.start()
        assert not next_entered.wait(timeout=.02)
        worker.submit("new")
        assert published.wait(timeout=1.)
        worker.stop(timeout_s=.5)
        assert calls == ["old", "new"]
        assert active["maximum"] == 1
        assert len(buffer.snapshot()) == len(publications) == 1
        np.testing.assert_array_equal(buffer.snapshot()[0].values, np.full((3, 7), 2.))
    finally:
        release.set()
        worker.stop(timeout_s=.5)


def test_dp_stop_during_output_conversion_also_prevents_publication(monkeypatch):
    converting, release = threading.Event(), threading.Event()
    buffer = ActionPlanBuffer()
    publications = []
    worker = DPWorker(lambda observation: np.ones((3, 7)), buffer,
                      on_publish=publications.append)
    extract = worker._extract_action

    def blocked_conversion(result):
        converting.set()
        assert release.wait(timeout=1.)
        return extract(result)

    monkeypatch.setattr(worker, "_extract_action", blocked_conversion)
    try:
        worker.start()
        worker.submit("old")
        assert converting.wait(timeout=1.)
        with pytest.raises(TimeoutError):
            worker.stop(timeout_s=.01)
        release.set()
        worker.stop(timeout_s=.5)
        assert not buffer.snapshot() and not publications
        assert worker.updates == 0
    finally:
        release.set()
        worker.stop(timeout_s=.5)


def test_dp_stop_timeout_also_bounds_a_running_publication_callback():
    callback_entered, release = threading.Event(), threading.Event()
    buffer = ActionPlanBuffer()

    def blocked_callback(plan):
        callback_entered.set()
        assert release.wait(timeout=.5)

    worker = DPWorker(lambda observation: np.ones((3, 7)), buffer,
                      on_publish=blocked_callback)
    try:
        worker.start()
        worker.submit("prediction before stop")
        assert callback_entered.wait(timeout=1.)
        # Its plan was committed before cancellation. The callback must not
        # prevent stop from enforcing the requested join timeout.
        assert len(buffer.snapshot()) == 1
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            worker.stop(timeout_s=.01)
        assert time.monotonic() - started < .2
        assert worker.running
        with pytest.raises(RuntimeError, match="still stopping"):
            worker.start()
        release.set()
        worker.stop(timeout_s=.5)
        assert not worker.running and worker.fault is None
        assert len(buffer.snapshot()) == 1
    finally:
        release.set()
        worker.stop(timeout_s=.5)


def test_wm_timeout_discards_old_targets_and_pending_requests(monkeypatch):
    entered, release, next_entered, published = (threading.Event() for _ in range(4))
    state_buffer = StateHistoryBuffer(rate_hz=10., horizon_s=.1)
    action_buffer = ActionPlanBuffer()
    target_buffer = WMTargetBuffer()
    stamp = monotonic_time()
    state_buffer.append(stamp, np.zeros(7), np.zeros(7), tau=np.zeros(7))
    action_buffer.append(np.zeros((20, 7)), start_time_s=stamp - .1, step_s=.1)
    calls = []

    def infer(history, action):
        calls.append(history.timestamps_s[-1])
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=1.)
        else:
            next_entered.set()
        return np.ones((10, 7)), np.zeros((10, 7))

    append = target_buffer.append

    def record_target(*args, **kwargs):
        result = append(*args, **kwargs)
        published.set()
        return result

    monkeypatch.setattr(target_buffer, "append", record_target)
    worker = WMWorker(
        state_buffer, action_buffer, target_buffer, infer,
        prediction_horizon=10, prediction_dt_s=.1,
        state_horizon_s=.1, action_horizon_s=.2, auto_request=False,
    )
    try:
        assert worker.request(stamp)
        worker.start()
        assert entered.wait(timeout=1.)
        old_thread = worker._thread
        assert worker.request(stamp)
        with pytest.raises(TimeoutError):
            worker.stop(timeout_s=.01)
        assert worker._thread is old_thread and worker.running
        assert not worker.request(stamp)
        with pytest.raises(RuntimeError, match="still stopping"):
            worker.start()
        release.set()
        worker.stop(timeout_s=.5)
        assert not target_buffer.snapshot()
        assert worker.inference_count == 0 and worker.last_latency_s is None
        assert worker.fault is None

        worker.start()
        assert not next_entered.wait(timeout=.02)
        stamp = monotonic_time()
        state_buffer.append(stamp, np.zeros(7), np.zeros(7), tau=np.zeros(7))
        assert worker.request(stamp)
        assert published.wait(timeout=1.)
        worker.stop(timeout_s=.5)
        assert len(calls) == 2 and worker.inference_count == 1
        assert len(target_buffer.snapshot()) == 1
    finally:
        release.set()
        worker.stop(timeout_s=.5)


def test_coordinator_stops_every_worker_and_preserves_buffers_after_stop_failure():
    events = []
    first_failure = RuntimeError("control worker did not stop")

    class Worker:
        def __init__(self, name, error=None):
            self.name, self.error = name, error

        def stop(self):
            events.append(f"stop {self.name}")
            if self.error:
                raise self.error

        def start(self):
            events.append(f"start {self.name}")

    buffer = SimpleNamespace(clear=lambda: events.append("clear buffer"))
    runtime = TimestampFastSlowRuntime(
        state_buffer=buffer, action_buffer=buffer, target_buffer=buffer,
        control_worker=Worker("control", first_failure),
        wm_worker=Worker("wm", TimeoutError("wm worker still inferring")),
        dp_worker=Worker("dp"),
    )
    with pytest.raises(RuntimeError) as error:
        runtime.reset()
    assert error.value is first_failure
    assert events == ["stop control", "stop wm", "stop dp"]
