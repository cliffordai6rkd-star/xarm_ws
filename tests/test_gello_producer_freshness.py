import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import gello_teleop.dual_gello_collect as collect


class FakeClock:
    def __init__(self):
        self.now = 1.

    def monotonic(self):
        return self.now

    def advance(self, duration):
        self.now += duration


class ClockStop:
    def __init__(self, clock):
        self.clock = clock
        self.stopped = False
        self.waits = []

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, duration):
        self.waits.append(duration)
        if not self.stopped:
            self.clock.advance(duration)
        return self.stopped


@pytest.fixture
def clock(monkeypatch):
    clock = FakeClock()
    # Replace the module reference, leaving threading/pytest's clocks intact.
    monkeypatch.setattr(collect, 'time', SimpleNamespace(monotonic=clock.monotonic, sleep=clock.advance))
    monkeypatch.setattr(collect, 'now_us', lambda: 1_000_000_000 + round(clock.now*1e6))
    return clock


def make_producer(clock, write, *, stop=None, feedback=None, read=None):
    stop = ClockStop(clock) if stop is None else stop
    cfg = SimpleNamespace(port='fake', gripper_id=-1, joint_signs=(1, 1),
                          damping_current_limit=(10, 10), weak_hold_enabled=False)
    mapper = SimpleNamespace(config=cfg, n=2, target=lambda raw: (raw.copy(), raw))
    def default_read():
        clock.advance(.005)
        return np.array([.1, -.2])
    reader = SimpleNamespace(read=default_read if read is None else read,
                             compute_damping_current=lambda *_args: np.zeros(2),
                             write_current_damping=write, disable_current_damping=Mock())
    producer = collect._LeaderProducer(reader, mapper, .1, stop, cfg,
                                      failure_stop=ClockStop(clock), feedback_worker=feedback,
                                      feedback_active=lambda: True)
    return producer


def startup_pipeline(producer):
    pipeline = object.__new__(collect.DualGelloPipeline)
    pipeline.ARM_NAMES = ('right',)
    pipeline.producers = [producer]
    pipeline.stop_event = producer.failure_stop
    pipeline.leader_max_age_s = .15
    pipeline.leader_startup_timeout_s = .01
    pipeline._consume_samples = lambda: None
    return pipeline


def test_position_is_visible_during_blocked_damping_readback_with_original_timestamp(clock):
    entered, release = threading.Event(), threading.Event()
    stop = threading.Event()
    def write(_current):
        entered.set()
        assert release.wait(1.)
        clock.advance(.025)
        stop.set()
    producer = make_producer(clock, write, stop=stop)
    producer.last_io_s = .059  # The last completed cycle is a separate metric.
    producer.start()
    try:
        assert entered.wait(1.)
        sample = producer.snapshot()
        assert sample is not None and sample.valid
        assert sample.sequence == producer.sequence == 0
        assert sample.acquired_monotonic_s == pytest.approx(1.005)
        assert sample.timestamp_us == 1_001_005_000
        assert sample.acquired_timestamp_us == 1_001_000_000
        np.testing.assert_array_equal(sample.mapped, [.1, -.2])
        clock.advance(.05)
        status = startup_pipeline(producer).leader_sample_status()['right']
        assert status['io_stage'] == 'damping_write_readback'
        assert status['in_flight_ms'] == pytest.approx(55.)
        assert status['read_ms'] == pytest.approx(5.)
        assert status['io_ms'] == pytest.approx(59.)
        assert status['age_ms'] == pytest.approx(50.)
        assert status['completed_cycles'] == 0
    finally:
        release.set()
        producer.thread.join(timeout=1.)
    assert not producer.thread.is_alive()
    assert producer.errors == 0 and producer.sequence == 1
    assert producer.snapshot() is sample
    assert producer.snapshot().acquired_monotonic_s == pytest.approx(1.005)
    assert producer.io_stage == 'idle' and producer.io_started_monotonic_s is None


def test_damping_failure_withdraws_early_sample_and_invalidates_feedback(clock):
    feedback = SimpleNamespace(
        snapshot=lambda _index: SimpleNamespace(valid=False, tau_ext=np.zeros(2)),
        controllers=[SimpleNamespace(update=lambda *_args, **_kwargs: np.zeros(2))],
        invalidate=Mock(),
    )
    def write(_current):
        assert producer.snapshot() is not None
        assert producer.sequence == 0
        clock.advance(.04)
        raise RuntimeError('Dynamixel 1 response COMM_RX_TIMEOUT')
    producer = make_producer(clock, write, feedback=feedback)
    producer.feedback_current[:] = 7.
    producer.current_command[:] = 8.
    producer.feedback_valid = True
    producer._run()
    assert producer.stop.is_set() and producer.failure_stop.is_set()
    assert producer.snapshot() is None
    assert producer.errors == 1 and producer.sequence == 0
    assert producer.last_error == 'Dynamixel 1 response COMM_RX_TIMEOUT'
    assert producer.last_read_s == pytest.approx(.005)
    assert producer.last_io_s == pytest.approx(.045)
    feedback.invalidate.assert_called_once()
    producer.reader.disable_current_damping.assert_called_once()
    actual, command, valid = producer.current_snapshot()
    assert not valid and not np.any(actual) and not np.any(command)
    assert producer.io_stage == 'failed' and producer.io_started_monotonic_s is None


def test_startup_requires_two_completed_damping_cycles(clock):
    pending_sequences = []
    def write(_current):
        pending_sequences.append(producer.sequence)
        assert producer.snapshot().sequence == producer.sequence
        with pytest.raises(TimeoutError, match='unavailable/stale'):
            pipeline.wait_for_leader_samples()
        if len(pending_sequences) == 2:
            producer.stop.set()
    producer = make_producer(clock, write)
    pipeline = startup_pipeline(producer)
    producer._run()
    assert pending_sequences == [0, 1]
    assert producer.sequence == 2
    pipeline.wait_for_leader_samples()


@pytest.mark.parametrize('duration, skipped', [(.101, 1), (.201, 2), (.35, 3)])
def test_io_overrun_starts_one_fresh_transaction_without_extra_grid_idle(clock, duration, skipped):
    starts = []
    def read():
        starts.append(clock.now)
        clock.advance(.03)
        return np.zeros(2)
    def write(_current):
        clock.advance(duration-.03 if len(starts) == 1 else .001)
        if len(starts) == 2:
            producer.stop.set()
    producer = make_producer(clock, write, read=read)
    producer._run()
    assert starts == pytest.approx([1., 1.+duration])
    assert producer.stop.waits[0] == 0.
    assert producer.skipped_ticks == skipped
    assert producer.sequence == 2


def test_control_still_stops_at_150_ms_sample_expiry(clock, monkeypatch):
    producer = make_producer(clock, lambda _current: None)
    producer.latest = collect.LeaderSample(np.zeros(2), np.zeros(2), 0, 0,
                                          clock.now-.151, 0, True)
    pipeline = startup_pipeline(producer)
    pipeline.sample_rate_hz = 100.
    pipeline.control = {}
    pipeline.control_deadline_policy = 'skip'
    pipeline.follow_stop = ClockStop(clock)
    pipeline.state = 'following'
    pipeline.control_tick_count = pipeline.control_skipped_tick_count = pipeline.stale_count = 0
    pipeline.control_max_lateness_s = pipeline.control_late_tick_count = 0
    pipeline._last_control_t = clock.now
    monkeypatch.setattr(collect, 'FixedRateTicker', lambda *_args, **_kwargs:
                        SimpleNamespace(wait=lambda _context: (0, 0.), last_skipped_ticks=0,
                                        maximum_lateness_s=.03))
    pipeline._control_loop()
    assert pipeline.stop_event.is_set()
    assert pipeline.stale_count == 1
    assert isinstance(pipeline.control_error, RuntimeError)
    assert 'expired/disconnected' in str(pipeline.control_error)

