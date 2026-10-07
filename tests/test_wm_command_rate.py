"""Command-rate tests using numerical forecasts and an in-memory arm only."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference.controllers.wm_two_step import ControlState, TwoStepReference
from inference.core.wm_control import WmControlLayer
from inference.dp_wm.output import ActiveOutput


URDF = Path(__file__).resolve().parents[1] / 'gello_teleop/models/xarm7_dynamics_g2.urdf'
EPOCH = 1_000_000


def settings(*, command_hz=None, execution='shadow', smoothing='legacy'):
    config = dict(
        mode='q_only', execution=execution, dt=.01, horizon_steps=2, sample_id=0,
        maximum_state_age_ms=10., maximum_forecast_age_ms=240., maximum_compute_ms=100.,
        torque_reference='raw', tau_filter_cutoff_hz=20., dynamics={'b_mode': 'zero'},
        weights={'eps_q': .01, 'eps_tau': 1., 'rho': 1.},
        limits=dict(max_velocity_rad_s=1., max_acceleration_rad_s2=4.,
                    max_step_rad=.1, max_tracking_error_rad=.1,
                    q_only_smoothing=smoothing),
    )
    if command_hz is not None:
        config['command_hz'] = command_hz
    if smoothing == 'jerk_tracking':
        config['limits']['max_jerk_rad_s3'] = 5.
    return config


def layer(**options):
    config = settings(**options)
    runtime = WmControlLayer(config, URDF, allow_active=config['execution'] == 'active')
    runtime.update_forecast(dict(
        anchor_us=EPOCH, plan_version=1,
        future_timestamps_us=EPOCH + np.arange(1, 33, dtype=np.int64)*10_000,
        q=np.full((1, 32, 7), .03),
    ))
    return runtime


def state(stamp=EPOCH, q=None, dq=None):
    return ControlState(stamp, np.zeros(7) if q is None else q,
                        np.zeros(7) if dq is None else dq)


@pytest.mark.parametrize('command_hz, expected_index, expected_period_us', [
    (None, 0, 10_000), (100, 0, 10_000), (50, 1, 20_000),
])
def test_command_clock_retains_two_100_hz_prediction_steps(
        command_hz, expected_index, expected_period_us):
    runtime = layer(command_hz=command_hz)
    observed_times = []
    for offset in range(0, 100_000, 1_000):
        feedback_stamp = EPOCH + offset//10_000*10_000
        event = runtime.poll(EPOCH + offset, state(feedback_stamp))
        if event is None:
            continue
        assert event['valid'], event.get('reason')
        observed_times.append(event['computed_at_us'])
        assert runtime.dt == .01
        assert event['command_hz'] == (100 if command_hz is None else command_hz)
        assert event['command_index'] == expected_index
        np.testing.assert_array_equal(
            event['target_timestamps_us'], feedback_stamp + np.array([10_000, 20_000]))
        assert event['command_target_timestamp_us'] == feedback_stamp + expected_period_us
        np.testing.assert_array_equal(event['first_q_command'], event['q_command'][expected_index])
    np.testing.assert_array_equal(observed_times, np.arange(EPOCH, EPOCH + 100_000, expected_period_us))
    assert runtime.report()['target_hz'] == 1_000_000/expected_period_us


@pytest.mark.parametrize('execution', ['offline', 'shadow'])
@pytest.mark.parametrize('smoothing', ['legacy', 'jerk_tracking'])
def test_50_hz_virtual_history_advances_the_second_physical_step(execution, smoothing):
    runtime = layer(command_hz=50, execution=execution, smoothing=smoothing)
    first = runtime.poll(EPOCH + 1_000, state())
    assert first['valid']
    # Independent constant-target integration: the legacy tracker has 4 rad/s²
    # acceleration, and the jerk tracker ramps from .05 to .10 rad/s².
    if smoothing == 'legacy':
        np.testing.assert_allclose(first['first_q_command'], .0012, atol=1e-12)
        np.testing.assert_allclose(runtime.limits.previous_v, .08, atol=1e-12)
        np.testing.assert_allclose(runtime.limits.previous_acceleration, 4., atol=1e-12)
    else:
        np.testing.assert_allclose(first['first_q_command'], .0000125, atol=1e-12)
        np.testing.assert_allclose(runtime.limits.previous_v, .0015, atol=1e-12)
        np.testing.assert_allclose(runtime.limits.previous_acceleration, .1, atol=1e-12)
    np.testing.assert_array_equal(runtime.limits.previous_q, first['q_command'][1])
    before = runtime.limits.previous_q.copy()
    assert runtime.poll(EPOCH + 11_000, state(EPOCH + 10_000)) is None
    np.testing.assert_array_equal(runtime.limits.previous_q, before)
    following = runtime.poll(EPOCH + 21_000, state(
        EPOCH + 20_000, q=before.copy(), dq=runtime.limits.previous_v.copy()))
    assert following['valid'], following.get('reason')
    assert np.all(following['first_q_command'] > before)
    np.testing.assert_array_equal(runtime.limits.previous_q, following['q_command'][1])
    np.testing.assert_array_equal(runtime.limits.previous_v, following['command_v_plan'][1])
    np.testing.assert_array_equal(runtime.limits.previous_acceleration, following['command_a_plan'][1])


def test_50_hz_active_history_requires_acknowledgement_and_records_second_step():
    runtime = layer(command_hz=50, execution='active', smoothing='jerk_tracking')
    event = runtime.poll(EPOCH + 1_000, state())
    assert event['valid'] and not event['sent']
    assert runtime.limits.previous_q is None
    assert runtime.limits.previous_v is None
    np.testing.assert_array_equal(runtime.limits.previous_acceleration, np.zeros(7))
    with pytest.raises(ValueError):
        runtime.mark_sent(event)
    event['sent'] = True
    runtime.mark_sent(event)
    np.testing.assert_array_equal(runtime.limits.previous_q, event['q_command'][1])
    np.testing.assert_array_equal(runtime.limits.previous_v, event['command_v_plan'][1])
    np.testing.assert_array_equal(runtime.limits.previous_acceleration, event['command_a_plan'][1])
    assert event['history_kind'] == 'sent_reference'


def test_50_hz_forecast_handover_starts_at_selected_endpoint_and_preserves_velocity():
    config = settings(command_hz=50, execution='active', smoothing='jerk_tracking')
    config['forecast_transition'] = {'enabled': True, 'duration_ms': 20.}
    runtime = WmControlLayer(config, URDF, allow_active=True)

    def positions(times, *, anchor, offset, velocity):
        values = offset + (np.asarray(times) - anchor)/1e6*velocity
        return np.repeat(values[:, None], 7, axis=1)

    old_times = EPOCH + np.arange(1, 33)*10_000
    runtime.update_forecast(dict(
        anchor_us=EPOCH, future_timestamps_us=old_times,
        q=positions(old_times, anchor=EPOCH, offset=.01, velocity=.1)[None],
    ))
    first = runtime.poll(EPOCH + 1_000, state(q=np.full(7, .01), dq=np.full(7, .1)))
    assert first['valid']
    first['sent'] = True
    runtime.mark_sent(first)
    previous_q = runtime.limits.previous_q.copy()
    previous_v = runtime.limits.previous_v.copy()

    new_anchor = EPOCH + 20_000
    new_times = new_anchor + np.arange(1, 33)*10_000
    runtime.update_forecast(dict(
        anchor_us=new_anchor, future_timestamps_us=new_times,
        q=positions(new_times, anchor=new_anchor, offset=.03, velocity=-.05)[None],
    ))
    switched = runtime.poll(new_anchor + 1_000, state(new_anchor, q=previous_q, dq=previous_v))
    assert switched['valid'] and switched['forecast_transition_active']
    np.testing.assert_allclose(switched['q_reference'], positions(
        switched['target_timestamps_us'], anchor=EPOCH, offset=.01, velocity=.1), atol=1e-12)
    assert switched['command_index'] == 1
    assert switched['forecast_transition_age_ms'] == 0.
    assert switched['forecast_transition_duration_ms'] == 20.
    assert np.max(np.abs(switched['q_reference'][1] - switched['q_reference_raw'][1])) > .01
    np.testing.assert_array_equal(runtime.limits.previous_q, previous_q)
    switched['sent'] = True
    runtime.mark_sent(switched)
    np.testing.assert_array_equal(runtime.limits.previous_q, switched['q_command'][1])
    np.testing.assert_array_equal(runtime.limits.previous_v, switched['command_v_plan'][1])
    np.testing.assert_array_equal(runtime.limits.previous_acceleration, switched['command_a_plan'][1])

    # Compare the correction's right slope with the previous linear forecast,
    # and its left slope at completion with the incoming linear forecast.
    start = switched['command_target_timestamp_us']
    end = start + 20_000
    for targets, expected_velocity in (([start, start + 1], .1), ([end - 1, end], -.05)):
        reference = TwoStepReference(np.asarray(targets, dtype=np.int64), positions(
            targets, anchor=new_anchor, offset=.03, velocity=-.05), None)
        corrected, _ = runtime.transition.apply(reference, command_index=1)
        np.testing.assert_allclose((corrected.q[1] - corrected.q[0])*1e6,
                                   expected_velocity, atol=1e-6)

    completed = runtime.poll(EPOCH + 41_000, state(
        EPOCH + 40_000, q=runtime.limits.previous_q.copy(), dq=runtime.limits.previous_v.copy()))
    assert completed['valid'] and not completed['forecast_transition_active']
    assert completed['command_target_timestamp_us'] == end
    np.testing.assert_allclose(completed['q_reference'][1], completed['q_reference_raw'][1], atol=1e-12)


def test_50_hz_gap_hold_retains_physical_targets_and_acknowledges_second_step():
    config = settings(command_hz=50, execution='active')
    config.update(maximum_forecast_age_ms=45.,
                  gap_hold={'enabled': True, 'maximum_ms': 100.})
    runtime = WmControlLayer(config, URDF, allow_active=True)
    runtime.update_forecast(dict(
        anchor_us=EPOCH, future_timestamps_us=EPOCH + np.arange(1, 33)*10_000,
        q=np.full((1, 32, 7), .03),
    ))
    first = runtime.poll(EPOCH + 1_000, state())
    first['sent'] = True
    runtime.mark_sent(first)
    saved_q = runtime.limits.previous_q.copy()
    saved_v = runtime.limits.previous_v.copy()
    held = runtime.poll(EPOCH + 61_000, state(EPOCH + 60_000, q=saved_q, dq=saved_v))
    assert held['valid'] and held['actual_mode'] == 'position_hold'
    assert held['command_index'] == 1
    assert held['command_target_timestamp_us'] == EPOCH + 80_000
    np.testing.assert_array_equal(held['target_timestamps_us'], EPOCH + np.array([70_000, 80_000]))
    np.testing.assert_array_equal(runtime.limits.previous_q, saved_q)
    held['sent'] = True
    runtime.mark_sent(held)
    np.testing.assert_array_equal(runtime.limits.previous_q, held['q_command'][1])
    np.testing.assert_array_equal(runtime.limits.previous_v, held['command_v_plan'][1])
    np.testing.assert_array_equal(runtime.limits.previous_acceleration, held['command_a_plan'][1])


@pytest.mark.parametrize('rate', [0, 25, 75, True, '50', float('nan')])
def test_unsupported_command_rate_rejected_before_hardware(rate):
    with pytest.raises(ValueError, match='command_hz'):
        layer(command_hz=rate)


class Arm:
    def __init__(self):
        self.commands = []
        self.stops = 0

    def enable(self):
        pass

    def stop_motion(self):
        self.stops += 1

    def command_joint_positions(self, q):
        self.commands.append(q.copy())


@pytest.fixture
def output_factory(monkeypatch):
    import inference.dp_wm.output as module
    clock = SimpleNamespace(us=EPOCH, seconds=0.)
    monkeypatch.setattr(module, 'now_us', lambda: clock.us)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.seconds)
    outputs = []

    def create(*, maximum_state_age_ms=10., command_hz=50):
        arm = Arm()
        sender = ActiveOutput(arm, maximum_state_age_ms=maximum_state_age_ms,
                              command_hz=command_hz)
        outputs.append(sender)
        sender.start()
        return sender, arm, clock

    yield create
    for sender in outputs:
        sender.close()


def command_event(*, command_index=1):
    return dict(valid=True, sent=False, first_q_command=np.full(7, .01),
                state_timestamp_us=EPOCH, command_index=command_index,
                command_hz=50, command_target_timestamp_us=EPOCH + 20_000,
                target_timestamps_us=EPOCH + np.array([10_000, 20_000]))


def test_send_uses_selected_20_ms_deadline_even_after_first_candidate_elapsed(output_factory):
    sender, arm, clock = output_factory(maximum_state_age_ms=30.)
    event = command_event()
    clock.us = EPOCH + 15_000
    assert sender.send(event) and event['sent']
    np.testing.assert_array_equal(arm.commands[-1], event['first_q_command'])
    expired = command_event()
    clock.us = EPOCH + 20_000
    assert not sender.send(expired)
    assert not expired['sent'] and not expired['valid']
    assert len(arm.commands) == 1 and not sender.enabled


def test_50_hz_deadline_does_not_relax_measured_state_age(output_factory):
    sender, arm, clock = output_factory(maximum_state_age_ms=10.)
    clock.us = EPOCH + 11_000
    event = command_event()
    assert not sender.send(event)
    assert not event['sent'] and not event['valid']
    assert not arm.commands


@pytest.mark.parametrize('acquired_at_us, should_send', [
    (EPOCH + 15_000, True), (EPOCH + 20_000, False),
])
def test_selected_deadline_is_rechecked_after_feedback_lock(
        output_factory, acquired_at_us, should_send):
    sender, arm, clock = output_factory(maximum_state_age_ms=30.)

    class FeedbackLock:
        def __init__(self):
            self.timeout = None
            self.releases = 0

        def acquire(self, *, timeout):
            self.timeout = timeout
            clock.us = acquired_at_us
            return True

        def release(self):
            self.releases += 1

    lock = FeedbackLock()
    arm.command_history = SimpleNamespace(lock=lock, latest=None)
    packet = command_event()
    assert sender.send(packet) is should_send
    assert packet['sent'] is should_send
    assert len(arm.commands) == int(should_send)
    assert lock.timeout == pytest.approx(.02)
    assert lock.releases == 1


@pytest.mark.parametrize('bad_index', [-1, 2, 1.0, True, '1', None])
def test_malformed_selected_index_never_sends(output_factory, bad_index):
    sender, arm, _ = output_factory()
    event = command_event(command_index=bad_index)
    assert not sender.send(event)
    assert not event['sent'] and not event['valid']
    assert not arm.commands


def test_malformed_ack_index_cannot_change_active_history():
    runtime = layer(command_hz=50, execution='active')
    event = runtime.poll(EPOCH + 1_000, state())
    event.update(sent=True, command_index=2)
    with pytest.raises(ValueError):
        runtime.mark_sent(event)
    assert runtime.limits.previous_q is None
    assert runtime.limits.previous_v is None


def test_legacy_output_without_command_index_retains_first_target_expiry(output_factory):
    sender, arm, clock = output_factory(maximum_state_age_ms=30., command_hz=100)
    packet = command_event()
    for key in ('command_index', 'command_hz', 'command_target_timestamp_us'):
        packet.pop(key)
    clock.us = EPOCH + 10_000
    assert not sender.send(packet)
    assert not packet['valid'] and not arm.commands


def test_50_hz_startup_hold_uses_second_target_deadline(output_factory):
    sender, arm, clock = output_factory()
    first = sender.start_hold(state())
    assert first is not None and first['sent']
    assert first['command_hz'] == 50 and first['command_index'] == 1
    assert first['command_target_timestamp_us'] == EPOCH + 20_000
    clock.us += 10_000
    clock.seconds += .01
    assert sender.hold(state(clock.us)) is None
    assert len(arm.commands) == 1
    clock.us += 10_000
    clock.seconds += .01
    following = sender.hold(state(clock.us, q=np.full(7, .02)))
    assert following is not None and following['sent']
    np.testing.assert_array_equal(arm.commands[-1], np.zeros(7))
    assert len(arm.commands) == 2 and sender.enabled


def test_50_hz_handover_starts_at_actual_sent_endpoint_and_preserves_earlier_candidate():
    config = settings(command_hz=50, execution='offline', smoothing='jerk_tracking')
    config['forecast_transition'] = dict(enabled=True, duration_ms=20.)
    runtime = WmControlLayer(config, URDF)
    def install(stamp, value):
        runtime.update_forecast(dict(anchor_us=stamp, plan_version=stamp,
            future_timestamps_us=stamp+np.arange(1,33,dtype=np.int64)*10_000,
            q=np.full((1,32,7), value)))
    install(EPOCH, .03)
    first = runtime.poll(EPOCH+1000, state())
    install(EPOCH+20_000, -.03)
    current = state(EPOCH+20_000, first['first_q_command'], first['command_v_plan'][1])
    switched = runtime.poll(EPOCH+21_000, current)
    assert switched['valid'] and switched['forecast_transition_active']
    assert switched['command_target_timestamp_us'] == EPOCH+40_000
    assert switched['forecast_transition_age_ms'] == 0.
    np.testing.assert_allclose(switched['q_reference'], .03)
    np.testing.assert_allclose(switched['q_reference_raw'], -.03)
    current = state(EPOCH+40_000, switched['first_q_command'], switched['command_v_plan'][1])
    settled = runtime.poll(EPOCH+41_000, current)
    assert settled['valid'] and not settled['forecast_transition_active']
    np.testing.assert_allclose(settled['q_reference'][1], -.03)
