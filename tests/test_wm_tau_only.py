"""Torque-only conversion shares MIT dynamics and acknowledged position output."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inference.controllers.wm_two_step import (
    ControlState, MitPositionController, TauOnlyController, TwoStepReference,
)
from inference.core.wm_control import WmControlLayer, validate_settings
from inference.dp_wm.output import ActiveOutput

URDF = Path(__file__).resolve().parents[1] / 'gello_teleop/models/xarm7_dynamics_g2.urdf'
EPOCH = 1_000_000


class Dynamics:
    """Coupled dynamics that change at each integrated pose and velocity."""
    def __init__(self):
        self.states = []

    def snapshot(self, q, dq):
        self.states.append((q.copy(), dq.copy()))
        coupling = .1 + .02*np.sin(q)
        mass = np.diag(.8 + np.cos(q)**2) + np.outer(coupling, coupling)
        nonlinear = .3*np.sin(q) + .05*dq**2
        return mass, nonlinear, np.full(7, .1)


class UnusedReference:
    def __array__(self, *args, **kwargs):
        raise AssertionError('tau_only must not read WM positions or their derivatives')


def mit_options():
    return dict(reference_source='wm_q', velocity_source='zero', kp=0., kd=0.,
                feedforward_source='wm_tau', nominal_acceleration_source='zero',
                integration_base='previous_command', feedback_state='command')


@pytest.mark.parametrize('base', ['measured', 'previous_command'])
@pytest.mark.parametrize('feedback_state', ['measured', 'command'])
@pytest.mark.parametrize('feedforward', [
    {'feedforward_source': 'wm_tau'},
    {'feedforward_source': 'nominal_wm_blend', 'wm_tau_weight': .1,
     'wm_residual_acceleration_limit_rad_s2': .5},
    {'feedforward_source': 'nominal_wm_blend', 'wm_tau_weight': [0., .1, .2, .3, .4, .5, 1.]},
])
def test_tau_only_matches_full_wm_mit_zero_tracking_with_state_dependent_coupled_dynamics(
        base, feedback_state, feedforward, monkeypatch):
    def reject_lstsq(*args, **kwargs):
        raise AssertionError('tau_only must use the MIT forward-dynamics path')

    monkeypatch.setattr(np.linalg, 'lstsq', reject_lstsq)
    state = ControlState(EPOCH, np.arange(7)*.1, np.arange(7)*.01)
    previous = replace(state, q=state.q+.02, dq=state.dq+.05)
    reference = TwoStepReference(np.array([EPOCH+10_000, EPOCH+20_000]),
        np.full((2, 7), .2), np.stack([np.arange(7)*.1, -np.arange(7)*.2]))
    options = {**mit_options(), 'integration_base': base, 'feedback_state': feedback_state,
               **feedforward}
    model = Dynamics()
    tau = TauOnlyController(dynamics=model, velocity_limit=1., acceleration_limit=.5,
                           jerk_limit=5., mit={**options, 'kp': 8., 'kd': 6.,
                           'velocity_source': 'filtered_wm_difference',
                           'nominal_acceleration_source': 'wm_difference'},
                           rho=0., smooth_weight=1e8, previous_weight=1e8)
    mit = MitPositionController(dynamics=Dynamics(), velocity_limit=1.,
                                acceleration_limit=.5, jerk_limit=5.,
                                mit={**options, 'feedforward_source': 'wm_tau', 'wm_tau_weight': 1.})
    result = tau.solve(state, replace(reference, q=UnusedReference(), dq=UnusedReference(),
                                     ddq=UnusedReference()), integration_state=previous,
                       previous_acceleration=np.full(7, .1))
    expected = mit.solve(state, reference, integration_state=previous,
                         previous_acceleration=np.full(7, .1))
    for key in ('q_raw', 'v_plan', 'a_plan', 'mass_matrix', 'nonlinear_effects',
                'tau_nominal', 'tau_wm_effective', 'tau_feedforward', 'tau_mit',
                'acceleration_raw', 'integration_q0', 'integration_v0'):
        np.testing.assert_allclose(getattr(result, key), getattr(expected, key), atol=1e-12)
    assert tau.information_usage() == (False, True)
    assert tau.feedforward_source == 'wm_tau'
    np.testing.assert_array_equal(result.wm_tau_weight, np.ones(7))
    assert result.actual_mode == 'tau_only' and result.solver_status == 'mit_forward_dynamics'
    np.testing.assert_array_equal(result.tau_feedback, np.zeros((2, 7)))
    np.testing.assert_allclose(result.dynamics_residual, 0., atol=1e-12)
    assert len(model.states) == 2
    np.testing.assert_array_equal(model.states[1][0], result.q_raw[0])
    np.testing.assert_array_equal(model.states[1][1], result.v_plan[0])
    assert not np.array_equal(result.mass_matrix[0], result.mass_matrix[1])


def config(execution='active', command_hz=50, **mit_changes):
    return dict(mode='tau_only', execution=execution, dt=.01, horizon_steps=2, sample_id=0,
        command_hz=command_hz, maximum_state_age_ms=10., maximum_forecast_age_ms=100.,
        maximum_compute_ms=100., torque_reference='raw', tau_filter_cutoff_hz=15.,
        dynamics={'b_mode': 'zero'}, weights={},
        mit={**mit_options(), 'kp': 8., 'kd': 6., 'velocity_source': 'filtered_wm_difference',
             'nominal_acceleration_source': 'wm_difference', **mit_changes},
        limits=dict(max_velocity_rad_s=.5, max_acceleration_rad_s2=1., max_jerk_rad_s3=5.,
                    max_step_rad=.01, max_tracking_error_rad=.1))


def layer(monkeypatch, **changes):
    monkeypatch.setattr('inference.core.wm_control.LocalDynamicsModel', lambda *args, **kw: Dynamics())
    settings = config(**changes)
    return WmControlLayer(settings, URDF, allow_active=settings['execution'] == 'active')


def forecast(runtime, state, torque=1000.):
    runtime.update_forecast(dict(anchor_us=state.timestamp_us,
        future_timestamps_us=state.timestamp_us+np.arange(1, 33)*10_000,
        # No q forecast: torque-only must not request q derivatives for MIT's settings.
        tau=np.full((1, 32, 7), torque)))


@pytest.mark.parametrize('command_hz', [50, 100])
@pytest.mark.parametrize('execution', ['active', 'shadow', 'offline'])
def test_tau_only_jerk_limits_and_history_follow_selected_sent_endpoint(
        monkeypatch, execution, command_hz):
    runtime = layer(monkeypatch, execution=execution, command_hz=command_hz)
    state = ControlState(EPOCH, np.zeros(7), np.zeros(7))
    selected = 1 if command_hz == 50 else 0
    previous_q = previous_v = None
    previous_a = np.zeros(7)
    for i in range(30):
        forecast(runtime, state, 1000. if i % 2 == 0 else -1000.)
        event = runtime.poll(state.timestamp_us+1000, state)
        assert event['valid'], event['reason']
        assert not event['wm_tracking_enabled']
        assert event['q_reference'] is None
        assert event['velocity_source'] == event['nominal_acceleration_source'] == 'zero'
        np.testing.assert_array_equal(event['first_q_command'], event['q_command'][selected])
        np.testing.assert_array_equal(event['integration_q0'], state.q if i == 0 else previous_q)
        np.testing.assert_array_equal(event['integration_v0'], state.dq if i == 0 else previous_v)
        np.testing.assert_array_equal(event['tau_feedback'], np.zeros((2, 7)))
        assert np.max(np.abs(event['command_jerk_plan'])) <= 5.+1e-9
        assert np.max(np.abs(event['command_a_plan'][selected]-previous_a)) <= 5./command_hz+1e-9
        assert np.max(np.abs(event['command_a_plan'])) <= 1.+1e-9
        assert np.max(np.abs(event['command_v_plan'])) <= .5+1e-9
        assert 'command_dynamics_residual' in event
        if execution == 'active':
            if i == 0:
                assert runtime.limits.previous_q is None
            else:
                np.testing.assert_array_equal(runtime.limits.previous_q, previous_q)
                np.testing.assert_array_equal(runtime.limits.previous_v, previous_v)
            event['sent'] = True
            runtime.mark_sent(event)
        previous_q = event['first_q_command']
        previous_v = event['command_v_plan'][selected]
        previous_a = event['command_a_plan'][selected]
        np.testing.assert_array_equal(runtime.limits.previous_q, previous_q)
        np.testing.assert_array_equal(runtime.limits.previous_v, previous_v)
        np.testing.assert_array_equal(runtime.limits.previous_acceleration, previous_a)
        # A lagging measured servo must not replace the acknowledged command base.
        state = replace(state, timestamp_us=state.timestamp_us+round(1e6/command_hz))


def test_tau_only_without_mit_config_preserves_raw_torque_and_measured_base(monkeypatch):
    settings = config(execution='shadow')
    settings.pop('mit')
    monkeypatch.setattr('inference.core.wm_control.LocalDynamicsModel', lambda *args, **kw: Dynamics())
    runtime = WmControlLayer(settings, URDF)
    state = ControlState(EPOCH, np.zeros(7), np.zeros(7))
    forecast(runtime, state, .1)
    event = runtime.poll(EPOCH+1000, state)
    assert event['valid'], event['reason']
    assert event['feedforward_source'] == 'wm_tau'
    assert event['integration_base'] == 'measured'
    np.testing.assert_array_equal(event['tau_mit'], event['tau_reference'])
    np.testing.assert_allclose(event['q_command'], 0., atol=1e-12)


@pytest.mark.parametrize('weight', [0., .01, .1, 1., [0., .1, .2, .3, .4, .5, 1.], float('nan')])
def test_tau_only_ignores_mit_weight_but_preserves_residual_limit(monkeypatch, weight):
    runtime = layer(monkeypatch, execution='shadow', feedforward_source='nominal_wm_blend',
                    wm_tau_weight=weight, wm_residual_acceleration_limit_rad_s2=.5)
    state = ControlState(EPOCH, np.zeros(7), np.zeros(7))
    forecast(runtime, state)
    event = runtime.poll(EPOCH+1000, state)
    assert event['valid'], event['reason']
    assert event['feedforward_source'] == 'wm_tau'
    np.testing.assert_array_equal(event['wm_tau_weight'], np.ones(7))
    np.testing.assert_allclose(event['acceleration_raw'], .5, atol=1e-12)
    np.testing.assert_array_equal(event['tau_feedforward'], event['tau_wm_effective'])
    np.testing.assert_array_equal(event['tau_feedback'], np.zeros((2, 7)))


@pytest.mark.parametrize('weight', [0., .01, .1, 1., [0., .1, .2, .3, .4, .5, 1.]])
def test_weight_changes_only_mit_torque_feedforward(weight):
    state = ControlState(EPOCH, np.zeros(7), np.zeros(7))
    reference = TwoStepReference(np.array([EPOCH+10_000, EPOCH+20_000]),
                                np.zeros((2, 7)), np.full((2, 7), 2.))
    options = {**mit_options(), 'feedforward_source': 'nominal_wm_blend',
               'wm_tau_weight': weight}
    tau = TauOnlyController(dynamics=Dynamics(), mit=options)
    mit = MitPositionController(dynamics=Dynamics(), mit=options, velocity_limit=1.)
    actual = tau.solve(state, reference)
    weighted = mit.solve(state, reference)
    np.testing.assert_array_equal(actual.tau_feedforward, reference.tau)
    np.testing.assert_allclose(weighted.tau_feedforward,
        (1-np.asarray(weight))*weighted.tau_nominal+np.asarray(weight)*reference.tau)
    np.testing.assert_array_equal(weighted.wm_tau_weight, np.broadcast_to(weight, (7,)))


@pytest.mark.parametrize('tau', [None, np.zeros((1, 7)), np.full((2, 7), np.nan)])
def test_tau_only_rejects_missing_malformed_or_nonfinite_torque(tau):
    state = ControlState(EPOCH, np.zeros(7), np.zeros(7))
    ref = TwoStepReference(np.array([EPOCH+10_000, EPOCH+20_000]), None, tau)
    with pytest.raises(ValueError, match='tau reference'):
        TauOnlyController(dynamics=Dynamics()).solve(state, ref)


@pytest.mark.parametrize('changes', [
    {'integration_base': 'unknown'}, {'feedback_state': 'unknown'},
    {'wm_residual_acceleration_limit_rad_s2': -1.},
])
def test_tau_only_validates_inherited_mit_torque_options(changes):
    with pytest.raises(ValueError):
        validate_settings(config(**changes), allow_active=True)


def test_tau_only_rejects_legacy_objective_torque_scale():
    settings = config()
    settings['torque_scale'] = 0.
    with pytest.raises(ValueError, match='weight to 1'):
        validate_settings(settings, allow_active=True)


def test_tau_only_uses_shared_position_sender_and_acknowledgement(monkeypatch):
    class Arm:
        def __init__(self):
            self.commands = []

        def enable(self):
            pass

        def stop_motion(self):
            pass

        def command_joint_positions(self, q):
            self.commands.append(q.copy())

    monkeypatch.setattr('inference.dp_wm.output.now_us', lambda: EPOCH+1000)
    runtime = layer(monkeypatch)
    state = ControlState(EPOCH, np.zeros(7), np.zeros(7))
    forecast(runtime, state)
    event = runtime.poll(EPOCH+1000, state)
    arm = Arm()
    sender = ActiveOutput(arm, command_hz=50, maximum_state_age_ms=10.)
    try:
        sender.start()
        assert sender.send(event)
        runtime.mark_sent(event)
        np.testing.assert_array_equal(arm.commands[0], event['q_command'][1])
        np.testing.assert_array_equal(runtime.limits.previous_q, arm.commands[0])
        np.testing.assert_array_equal(runtime.limits.previous_v, event['command_v_plan'][1])
    finally:
        sender.close()
