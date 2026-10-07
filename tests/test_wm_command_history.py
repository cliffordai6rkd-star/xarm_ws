from collections import deque
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from inference.dp_wm.command_history import HostCommandHistory, RecordingArm
from inference.dp_wm.live import LiveSource, _cadence_samples
from inference.dp_wm.output import ActiveOutput
from inference.pi0_wm.wm import CausalChain


def test_history_uses_last_successful_send_at_acquisition_and_holds_without_age_limit():
    history = HostCommandHistory()
    with pytest.raises(ValueError, match='No successful host q_cmd'):
        history.at_or_before(1_000_000)
    initial = np.ones(7)
    history.record(initial, 1_000_000)
    initial[:] = 99
    history.record(np.full(7, 2), 1_010_000)
    q, stamp = history.at_or_before(1_009_999)
    np.testing.assert_array_equal(q, np.ones(7))
    assert stamp == 1_000_000
    q[:] = 88
    np.testing.assert_array_equal(history.at_or_before(1_000_000)[0], np.ones(7))
    np.testing.assert_array_equal(history.at_or_before(9_000_000)[0], np.full(7, 2))
    with pytest.raises(ValueError, match='No successful host q_cmd'):
        history.at_or_before(999_999)


class Arm:
    def __init__(self):
        self.calls = []
        self.fail = False

    def enable(self):
        self.calls.append('enable')

    def stop_motion(self):
        self.calls.append('stop')

    def command_joint_positions(self, q):
        if self.fail:
            raise RuntimeError('SDK rejected target')
        self.calls.append(q.copy())

    def move_to_reset(self, q, **kwargs):
        if self.fail:
            raise RuntimeError('SDK rejected reset')
        self.calls.append((q.copy(), kwargs))


def test_recording_arm_records_only_successful_servo_and_reset_completions(monkeypatch):
    import inference.dp_wm.command_history as module
    clock = SimpleNamespace(us=1_000_000)
    monkeypatch.setattr(module, 'now_us', lambda: clock.us)
    history, arm = HostCommandHistory(), Arm()
    recorded = RecordingArm(arm, history)
    assert history.latest is None
    recorded.command_joint_positions(np.ones(7))
    assert history.latest.timestamp_us == 1_000_000
    arm.fail = True
    clock.us += 10_000
    with pytest.raises(RuntimeError, match='rejected'):
        recorded.command_joint_positions(np.full(7, 2))
    assert len(history.commands) == 1
    with pytest.raises(RuntimeError, match='rejected'):
        recorded.move_to_reset(np.full(7, 3), speed=.2, acceleration=.5)
    assert len(history.commands) == 1
    arm.fail = False
    recorded.move_to_reset(np.full(7, 3), speed=.2, acceleration=.5)
    assert history.latest.kind == 'reset'
    assert history.latest.timestamp_us == 1_010_000
    np.testing.assert_array_equal(history.latest.q, np.full(7, 3))


def test_recording_arm_serializes_state_acquisition_after_inflight_send():
    sending, finish_send, acquired = threading.Event(), threading.Event(), threading.Event()
    history = HostCommandHistory()

    class BlockingArm(Arm):
        def command_joint_positions(self, q):
            sending.set()
            assert finish_send.wait(1.)
            super().command_joint_positions(q)

        def read_joint_state(self):
            assert history.latest is not None
            acquired.set()
            return 'state'

    arm = RecordingArm(BlockingArm(), history)
    sender = threading.Thread(target=arm.command_joint_positions, args=(np.ones(7),))
    reader = threading.Thread(target=arm.read_joint_state)
    sender.start()
    assert sending.wait(1.)
    reader.start()
    assert not acquired.wait(.02)
    finish_send.set()
    sender.join(1.)
    reader.join(1.)
    assert acquired.is_set() and not sender.is_alive() and not reader.is_alive()


def bare_source(history):
    source = LiveSource.__new__(LiveSource)
    source.command_history = history
    source.stop_event = threading.Event()
    source.lock = threading.Lock()
    source.states = deque(maxlen=100)
    source.rows = deque(maxlen=50)
    source.options = {'command_source': 'host_command', 'maximum_pair_skew_ms': 1.}
    source.operations = {}
    source.filters = {key: CausalChain([]) for key in ('q', 'dq', 'tau', 'delta_q')}
    source.history_interpolated_rows = 0
    source.history_rebuilds = 0
    source.last_history_gap_ms = None
    source.status = 'waiting'
    source.error = None
    return source


def test_live_pairs_before_acquisition_not_later_report_timestamp():
    history = HostCommandHistory()
    history.record(np.ones(7), 990_000)
    history.record(np.full(7, 9), 1_004_000)
    source = bare_source(history)

    class StateArm:
        def read_joint_state(self):
            source.stop_event.set()
            return SimpleNamespace(q_timestamp_us=1_005_000, timestamp_us=1_005_000,
                q_acquired_timestamp_us=1_000_000, q_valid=True, dq_valid=True, torque_valid=True,
                q=np.full(7, .5), dq=np.zeros(7), torque=np.zeros(7))

    source.arm = StateArm()
    source._read()
    assert source.error is None
    assert source.history_rebuilds == 0
    assert source.rows[0][0] == 1_005_000
    np.testing.assert_allclose(source.rows[0][1]['delta_q'], .5)


def test_host_command_live_history_rebuilds_on_missing_tick_without_fabricating_state_rows():
    history = HostCommandHistory()
    history.record(np.ones(7), 990_000)
    history.record(np.full(7, 4), 1_005_000)
    history.record(np.full(7, 8), 1_018_000)
    source = bare_source(history)

    class StateArm:
        stamps = iter((1_002_000, 1_022_000))

        def read_joint_state(self):
            stamp = next(self.stamps)
            if stamp == 1_022_000:
                source.stop_event.set()
            return SimpleNamespace(q_timestamp_us=stamp, timestamp_us=stamp,
                q_acquired_timestamp_us=stamp-2_000, q_valid=True, dq_valid=True, torque_valid=True,
                q=np.full(7, (stamp-1_002_000)/10_000), dq=np.zeros(7), torque=np.zeros(7))

    source.arm = StateArm()
    source._read()
    assert source.error is None and source.history_rebuilds == 1
    assert not source.rows
    assert source.history_interpolated_rows == 0
    assert 'strict measured 100 Hz cadence' in source.status


def test_optional_legacy_bridge_selects_command_at_midpoint_without_future_leakage():
    history = HostCommandHistory()
    history.record(np.ones(7), 990_000)
    history.record(np.full(7, 4), 1_005_000)
    history.record(np.full(7, 8), 1_018_000)
    before = {'q': np.zeros(7), 'q_cmd': np.ones(7), 'delta_q': np.ones(7)}
    after = {'q': np.full(7, 2), 'q_cmd': np.full(7, 8), 'delta_q': np.full(7, 6)}
    samples = _cadence_samples(1_002_000, before, 1_022_000, after,
                              command_lookup=lambda stamp: history.at_or_before(stamp-2_000))
    np.testing.assert_array_equal(samples[0][1]['delta_q'], np.full(7, 3))
    with pytest.raises(ValueError, match='strict measured 100 Hz cadence'):
        _cadence_samples(1_002_000, before, 1_022_000, after, interpolate=False)


def test_bridge_without_send_history_preserves_previous_command_instead_of_interpolating():
    before = {'q': np.zeros(7), 'delta_q': np.ones(7)}
    after = {'q': np.full(7, 2), 'delta_q': np.full(7, 6)}
    samples = _cadence_samples(1_000_000, before, 1_020_000, after)
    np.testing.assert_array_equal(samples[0][1]['delta_q'], np.zeros(7))


def test_startup_hold_is_actual_successful_send_and_keeps_fixed_captured_target(monkeypatch):
    import inference.dp_wm.command_history as commands
    import inference.dp_wm.output as output
    clock = SimpleNamespace(us=1_000_000)
    monkeypatch.setattr(commands, 'now_us', lambda: clock.us)
    monkeypatch.setattr(output, 'now_us', lambda: clock.us)
    history, raw_arm = HostCommandHistory(), Arm()
    sender = ActiveOutput(RecordingArm(raw_arm, history), maximum_state_age_ms=10, timeout_s=10)
    state = SimpleNamespace(timestamp_us=clock.us, q=np.full(7, .2))
    try:
        assert history.latest is None
        sender.start()
        assert sender.start_hold(None) is None and history.latest is None
        packet = sender.start_hold(state)
        assert packet['sent'] and packet['sent_at_us'] == history.latest.timestamp_us
        assert packet['mode'] == 'startup_hold'
        np.testing.assert_array_equal(history.latest.q, state.q)
        assert sender.hold(state) is None  # No duplicate-state sends.
        clock.us += 10_000
        next_state = SimpleNamespace(timestamp_us=clock.us, q=np.full(7, .3))
        assert sender.hold(next_state)['sent']
        np.testing.assert_array_equal(history.latest.q, np.full(7, .2))
        clock.us += 10_000
        next_state.timestamp_us = clock.us
        assert sender.start_hold(next_state)['sent']
        np.testing.assert_array_equal(history.latest.q, np.full(7, .2))
        sender.end_hold()
        assert sender.hold(next_state) is None
        sender.stop()
        assert sender.hold_q is None
    finally:
        sender.close()


def test_failed_bootstrap_send_does_not_create_host_command_history(monkeypatch):
    import inference.dp_wm.output as module
    monkeypatch.setattr(module, 'now_us', lambda: 1_000_000)
    history, arm = HostCommandHistory(), Arm()
    arm.fail = True
    sender = ActiveOutput(RecordingArm(arm, history), maximum_state_age_ms=10)
    try:
        sender.start()
        with pytest.raises(RuntimeError, match='SDK rejected target'):
            sender.start_hold(SimpleNamespace(timestamp_us=1_000_000, q=np.zeros(7)))
        assert history.latest is None
    finally:
        sender.close()
