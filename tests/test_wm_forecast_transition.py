"""Physical-time forecast handover and sent-command continuity regressions."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from inference.controllers.wm_two_step import ControlState, TwoStepReference
from inference.core.forecast_transition import ForecastTransition
from inference.core.wm_control import WmControlLayer


URDF = Path(__file__).resolve().parents[1] / 'gello_teleop/models/xarm7_dynamics_g2.urdf'
EPOCH = 1_000_000
INITIAL_Q = np.array([0., 0., 0., np.pi / 2, 0., np.pi / 2, -np.pi / 2])


def forecast(anchor, *, offset=0., velocity=0., plan_version=1):
    times = anchor + np.arange(1, 33, dtype=np.int64) * 10_000
    q = (np.broadcast_to(np.asarray(offset), (7,))[None]
         + (times - EPOCH)[:, None] / 1e6 * np.broadcast_to(np.asarray(velocity), (7,))[None])
    return dict(anchor_us=anchor, plan_version=plan_version,
                future_timestamps_us=times, q=q[None], tau=(q + 1.)[None])


def reference(raw, targets):
    targets = np.asarray(targets, dtype=np.int64)
    times = raw['future_timestamps_us']
    q = np.stack([np.interp(targets-times[0], times-times[0], raw['q'][0, :, j])
                  for j in range(7)], axis=-1)
    return TwoStepReference(targets, q, np.ones((len(targets), 7)),
                            anchor_us=raw['anchor_us'], indices=(3, 4))


def effective(transition, raw, targets):
    return transition.apply(reference(raw, targets))[0].q


def settings(*, enabled=None, execution='shadow', maximum_age_ms=240.):
    config = dict(mode='q_only', execution=execution, dt=.01, horizon_steps=2, sample_id=0,
                  maximum_state_age_ms=10., maximum_forecast_age_ms=maximum_age_ms,
                  maximum_compute_ms=100., torque_reference='raw', tau_filter_cutoff_hz=20.,
                  dynamics=dict(b_mode='zero'), weights={},
                  limits=dict(q_only_smoothing='jerk_tracking', max_velocity_rad_s=.5,
                              max_acceleration_rad_s2=1., max_jerk_rad_s3=5.,
                              max_step_rad=.01, max_tracking_error_rad=.1))
    if enabled is not None:
        config['forecast_transition'] = dict(enabled=enabled, duration_ms=50.)
    return config


def test_repeated_swaps_preserve_effective_position_and_velocity_at_same_time():
    transition = ForecastTransition(duration_ms=50., sample_id=0, maximum_age_ms=240.)
    current = forecast(EPOCH, offset=np.arange(7) * .01, velocity=np.arange(7) * .1)
    transition.update(current)
    effective(transition, current, [EPOCH + 10_000, EPOCH + 20_000])
    for index in range(1, 9):
        start = EPOCH + 10_000 + index * 10_000
        # Evaluate the already-smoothed curve, including an unfinished handover.
        old_position = effective(transition, current, [start, start + 1])[0]
        around = effective(transition, current, [start - 1, start + 1])
        old_velocity = (around[1] - around[0]) / 2e-6
        current = forecast(EPOCH + index * 10_000,
                           offset=(-1.) ** index * (np.arange(7) + 1) * .005,
                           velocity=(-1.) ** index * np.arange(7) * .04,
                           plan_version=index + 1)
        transition.update(current)
        after, details = transition.apply(reference(current, [start, start + 1]))
        np.testing.assert_allclose(after.q[0], old_position, atol=1e-12)
        np.testing.assert_allclose((after.q[1] - after.q[0]) / 1e-6,
                                   old_velocity, atol=2e-6)
        assert details['forecast_transition_active']
        assert details['forecast_transition_reason'] == 'overlap_handover'
    assert transition.switch_count == 8


def test_same_time_linear_trajectory_is_unchanged_despite_different_anchor_and_grid():
    transition = ForecastTransition(duration_ms=50., sample_id=0, maximum_age_ms=240.)
    offset, velocity = np.arange(7) * .01, np.arange(7) * .02
    first = forecast(EPOCH, offset=offset, velocity=velocity)
    transition.update(first)
    effective(transition, first, [EPOCH + 15_000, EPOCH + 25_000])
    second = forecast(EPOCH + 27_000, offset=offset, velocity=velocity, plan_version=2)
    transition.update(second)
    raw = reference(second, [EPOCH + 45_000, EPOCH + 55_000])
    result, details = transition.apply(raw)
    np.testing.assert_allclose(result.q, raw.q, atol=1e-15)
    assert not details['forecast_transition_active']
    assert details['forecast_transition_reason'] == 'same_time_trajectory_unchanged'


@pytest.mark.parametrize('maximum_age_ms,start,reason', [
    (60., EPOCH + 90_000, 'old_forecast_unavailable'),
    (50., EPOCH + 45_000, 'insufficient_valid_overlap'),
    (500., EPOCH + 325_000, 'old_forecast_unavailable'),
])
def test_expired_or_insufficient_old_overlap_does_not_modify_latest_reference(maximum_age_ms, start, reason):
    transition = ForecastTransition(duration_ms=50., sample_id=0, maximum_age_ms=maximum_age_ms)
    old = forecast(EPOCH, offset=.1, velocity=.03)
    transition.update(old)
    effective(transition, old, [EPOCH + 10_000, EPOCH + 20_000])
    new = forecast(start - 10_000, offset=-.1, velocity=-.03)
    transition.update(new)
    raw = reference(new, [start, start + 10_000])
    result, details = transition.apply(raw)
    np.testing.assert_array_equal(result.q, raw.q)
    assert not details['forecast_transition_active']
    assert details['forecast_transition_reason'] == reason


def test_handover_ends_on_latest_raw_trajectory_without_retiming_or_changing_torque():
    transition = ForecastTransition(duration_ms=50., sample_id=0, maximum_age_ms=60.)
    old = forecast(EPOCH, offset=.06, velocity=.2)
    transition.update(old)
    effective(transition, old, [EPOCH + 10_000, EPOCH + 20_000])
    new = forecast(EPOCH + 20_000, offset=-.01, velocity=-.1, plan_version=2)
    transition.update(new)
    raw = reference(new, [EPOCH + 30_000, EPOCH + 40_000])
    result, details = transition.apply(raw)
    assert details['forecast_transition_active']
    # The old forecast's age limit truncates the requested 50 ms transition.
    assert details['forecast_transition_duration_ms'] == 30.
    np.testing.assert_array_equal(result.timestamps_us, raw.timestamps_us)
    assert result.anchor_us == new['anchor_us'] and result.indices == raw.indices
    np.testing.assert_array_equal(result.tau, raw.tau)
    end = EPOCH + 60_000
    final_raw = reference(new, [end, end + 1])
    final, details = transition.apply(final_raw)
    np.testing.assert_array_equal(final.q, final_raw.q)
    np.testing.assert_allclose((final.q[1] - final.q[0]) / 1e-6, -.1, atol=1e-10)
    left = effective(transition, new, [end - 1, end])
    np.testing.assert_allclose((left[1] - left[0]) / 1e-6, -.1, atol=2e-6)
    assert not details['forecast_transition_active']


def test_disabled_transition_preserves_existing_control_layer_behavior():
    layers = [WmControlLayer(settings(), URDF),
              WmControlLayer(settings(enabled=False), URDF)]
    state = ControlState(EPOCH, INITIAL_Q.copy(), np.zeros(7))
    for index in range(25):
        target = INITIAL_Q + (.04 if (index // 5) % 2 else -.04)
        raw = forecast(state.timestamp_us, offset=target)
        events = []
        for layer in layers:
            layer.update_forecast(deepcopy(raw))
            event = layer.poll(state.timestamp_us + 1_000, state)
            assert event['valid']
            assert 'q_reference_correction' not in event
            events.append(event)
        for key in ('q_reference', 'q_raw', 'first_q_command', 'command_v_plan', 'command_a_plan'):
            np.testing.assert_array_equal(events[0][key], events[1][key])
        state = ControlState(state.timestamp_us + 10_000,
                             events[0]['first_q_command'], events[0]['command_v_plan'][0])


def test_transition_preserves_limits_and_only_acknowledged_active_commands_advance_history():
    layer = WmControlLayer(settings(enabled=True, execution='active'), URDF, allow_active=True)
    state = ControlState(EPOCH, INITIAL_Q.copy(), np.zeros(7))
    previous_q, previous_v, previous_a = state.q.copy(), state.dq.copy(), np.zeros(7)
    active_transitions = 0
    for index in range(100):
        if index % 5 == 0:
            target = INITIAL_Q + (.08 if (index // 5) % 2 else -.08)
            layer.update_forecast(forecast(state.timestamp_us, offset=target, plan_version=index + 1))
        event = layer.poll(state.timestamp_us + 1_000, state)
        assert event['valid'], event.get('reason')
        assert not event['sent'] and event['history_kind'] == 'pending_send'
        active_transitions += int(event['forecast_transition_active'])
        if index == 0:
            assert layer.limits.previous_q is None
            assert layer.limits.previous_v is None
        else:
            np.testing.assert_array_equal(layer.limits.previous_q, previous_q)
            np.testing.assert_array_equal(layer.limits.previous_v, previous_v)
            np.testing.assert_array_equal(layer.limits.previous_acceleration, previous_a)
        assert np.max(np.abs(event['command_v_plan'])) <= .5 + 1e-10
        assert np.max(np.abs(event['command_a_plan'])) <= 1. + 1e-10
        assert np.max(np.abs(event['command_jerk_plan'])) <= 5. + 1e-10
        assert np.max(np.abs(event['first_q_command'] - previous_q)) <= .01 + 1e-10
        assert np.max(np.abs(event['first_q_command'] - state.q)) <= .1 + 1e-10
        assert np.max(np.abs(event['command_a_plan'][0] - previous_a)) <= .05 + 1e-10
        event['sent'] = True
        layer.mark_sent(event)
        assert event['history_kind'] == 'sent_reference'
        previous_q = event['first_q_command'].copy()
        previous_v = event['command_v_plan'][0].copy()
        previous_a = event['command_a_plan'][0].copy()
        state = ControlState(state.timestamp_us + 10_000, previous_q.copy(), previous_v.copy())
    assert active_transitions > 0
    assert layer.counts['sent'] == 100


def test_transition_does_not_extend_latest_forecast_freshness_and_reset_discards_old_curve():
    layer = WmControlLayer(settings(enabled=True, maximum_age_ms=60.), URDF)
    layer.update_forecast(forecast(EPOCH, offset=INITIAL_Q + .03))
    first = layer.poll(EPOCH + 1_000, ControlState(EPOCH, INITIAL_Q.copy(), np.zeros(7)))
    assert first['valid']
    layer.update_forecast(forecast(EPOCH + 20_000, offset=INITIAL_Q - .03))
    second = layer.poll(EPOCH + 21_000, ControlState(EPOCH + 20_000, INITIAL_Q.copy(), np.zeros(7)))
    assert second['valid'] and second['forecast_transition_active']
    stale = layer.poll(EPOCH + 91_000, ControlState(EPOCH + 90_000, INITIAL_Q.copy(), np.zeros(7)))
    assert not stale['valid'] and stale['reason'] == 'WM reference exceeded the configured age'
    assert layer.transition.curve is None
    layer.reset()
    layer.update_forecast(forecast(EPOCH + 100_000, offset=INITIAL_Q + .02))
    reset = layer.poll(EPOCH + 101_000, ControlState(EPOCH + 100_000, INITIAL_Q.copy(), np.zeros(7)))
    assert reset['valid'] and reset['forecast_transition_reason'] == 'first_forecast'
    np.testing.assert_array_equal(reset['q_reference'], reset['q_reference_raw'])


def test_transition_missing_q_reference_remains_a_controlled_invalid_result():
    # A zero objective scale already produces an invalid q-only controller
    # result. Optional handover must retain that failure path without crashing.
    layer = WmControlLayer({**settings(enabled=True), 'position_scale': 0.}, URDF)
    layer.update_forecast(forecast(EPOCH, offset=INITIAL_Q))
    event = layer.poll(EPOCH + 1_000, ControlState(EPOCH, INITIAL_Q.copy(), np.zeros(7)))
    assert not event['valid']
    assert layer.counts['invalid'] == 1
    assert layer.transition.curve is None


@pytest.mark.parametrize('change, message', [
    ({'forecast_transition': {'enabled': True, 'duration_ms': 5.}}, 'duration_ms'),
    ({'forecast_transition': {'enabled': True, 'duration_ms': float('nan')}}, 'duration_ms'),
    ({'forecast_transition': {'enabled': 'true', 'duration_ms': 50.}}, 'enabled'),
    ({'mode': 'tau_only'}, 'only for q_only'),
])
def test_transition_configuration_rejects_invalid_duration_type_and_unsupported_mode(change, message):
    with pytest.raises(ValueError, match=message):
        WmControlLayer({**settings(enabled=True), **change}, URDF)
