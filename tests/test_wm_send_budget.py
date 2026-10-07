"""Feedback phase drift must wait for fresh data, without weakening send expiry."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference.controllers.wm_two_step import ControlState
from inference.core.wm_control import WmControlLayer
from inference.dp_wm.output import ActiveOutput

EPOCH = 1_000_000
URDF = Path(__file__).resolve().parents[1]/'gello_teleop/models/xarm7_dynamics_g2.urdf'


def settings():
    return dict(mode='q_only', execution='active', dt=.01, command_hz=50,
        horizon_steps=2, sample_id=0, maximum_state_age_ms=10., minimum_state_remaining_ms=5.,
        maximum_forecast_age_ms=240., maximum_compute_ms=4., torque_reference='raw',
        tau_filter_cutoff_hz=20., dynamics=dict(b_mode='zero'), weights={},
        forecast_transition=dict(enabled=True, duration_ms=20.),
        limits=dict(q_only_smoothing='jerk_tracking', max_velocity_rad_s=1.,
            max_acceleration_rad_s2=4., max_jerk_rad_s3=5., max_step_rad=.1, max_tracking_error_rad=.1))


class Arm:
    def __init__(self):
        self.commands = []
        self.stops = 0

    def enable(self):
        pass

    def command_joint_positions(self, q):
        self.commands.append(q.copy())

    def stop_motion(self):
        self.stops += 1


def state(stamp, q=None, dq=None):
    return ControlState(stamp, np.zeros(7) if q is None else q,
                        np.zeros(7) if dq is None else dq)


def sender(monkeypatch):
    import inference.dp_wm.output as module
    clock = SimpleNamespace(us=EPOCH+1000)
    monkeypatch.setattr(module, 'now_us', lambda: clock.us)
    monkeypatch.setattr(module.time, 'monotonic', lambda: (clock.us-EPOCH)/1e6)
    arm = Arm()
    output = ActiveOutput(arm, maximum_state_age_ms=10., command_hz=50,
                          minimum_state_remaining_ms=5.)
    output.start()
    return output, arm, clock


def test_logged_9_715_ms_state_defers_without_consuming_slot_or_sent_history(monkeypatch):
    layer = WmControlLayer(settings(), URDF, allow_active=True)
    layer.update_forecast(dict(anchor_us=EPOCH,
        future_timestamps_us=EPOCH+np.arange(1,33)*10_000, q=np.full((1,32,7), .03)))
    output, arm, clock = sender(monkeypatch)
    try:
        first = layer.poll(clock.us, state(EPOCH))
        assert first['valid'] and output.send(first)
        layer.mark_sent(first)
        previous = tuple(value.copy() for value in (
            layer.limits.previous_q, layer.limits.previous_v, layer.limits.previous_acceleration))
        slot, stamp = layer.next_us, layer.last_state_us
        clock.us = EPOCH+21_715
        # Reproduce the expired frame phase from the user's log. No solver
        # work or virtual command history should advance while awaiting a row.
        assert layer.poll(clock.us, state(EPOCH+12_000)) is None
        assert layer.next_us == slot and layer.last_state_us == stamp
        for actual, expected in zip((layer.limits.previous_q, layer.limits.previous_v,
                                     layer.limits.previous_acceleration), previous):
            np.testing.assert_array_equal(actual, expected)
        assert layer.counts['state_send_budget_skips'] == 1
        assert output.enabled and len(arm.commands) == 1 and arm.stops == 0
        clock.us = EPOCH+23_000
        fresh = layer.poll(clock.us, state(EPOCH+22_000, previous[0], previous[1]))
        assert fresh['valid'] and fresh['command_index'] == 1
        clock.us += 500  # A 0.5 ms calculation/dispatch cost, as in the log.
        assert output.send(fresh)
        layer.mark_sent(fresh)
        assert fresh['send_state_age_ms'] == 1.5
        assert output.enabled and arm.stops == 0 and len(arm.commands) == 2
        # The same command still expires after the ORIGINAL 10 ms state age.
        late = dict(fresh, sent=False)
        clock.us = EPOCH+33_000
        assert not output.send(late) and not late['valid']
        assert late['send_state_age_ms'] == 11.
        assert late['remaining_state_budget_ms'] == -1.
    finally:
        output.close()


def test_startup_hold_waits_through_9_953_ms_frame_without_latching_stop(monkeypatch):
    output, arm, clock = sender(monkeypatch)
    try:
        # Do not capture an almost-expired position as the startup target.
        clock.us = EPOCH+9_953
        assert output.start_hold(state(EPOCH)) is None
        assert output.hold_q is None and not arm.commands and output.enabled
        clock.us = EPOCH+11_000
        first = output.start_hold(state(EPOCH+10_000))
        assert first['sent']
        next_slot, source_stamp = output.hold_next_us, output.hold_state_timestamp_us
        clock.us = EPOCH+31_953
        assert output.hold(state(EPOCH+22_000)) is None
        assert output.hold_next_us == next_slot and output.hold_state_timestamp_us == source_stamp
        assert output.enabled and arm.stops == 0 and len(arm.commands) == 1
        clock.us = EPOCH+33_000
        fresh = output.hold(state(EPOCH+32_000, np.full(7,.02)))
        assert fresh['sent'] and output.enabled and arm.stops == 0
        np.testing.assert_array_equal(arm.commands[-1], np.zeros(7))
    finally:
        output.close()


@pytest.mark.parametrize('reserve', [10., 11., -1., True, float('nan')])
def test_invalid_compute_send_reserve_is_rejected_before_runtime(reserve):
    with pytest.raises(ValueError, match='minimum_state_remaining_ms'):
        WmControlLayer({**settings(), 'minimum_state_remaining_ms': reserve}, URDF, allow_active=True)
