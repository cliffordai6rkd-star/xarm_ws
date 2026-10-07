"""Exercise optional URDF feedback paths without any robot or serial I/O."""

from unittest.mock import Mock

import numpy as np
import pytest

from gello_teleop.tau_free_feedback import TauFreeFeedbackWorker, URDFTorqueEstimator
from nero_collection.arms.base import ArmState
from xarm_stack.momentum import PinocchioMomentumModel
from xarm_stack.torque_visualization import DEFAULT_URDF


class FakeDynamics:
    def __init__(self, path=None, gravity=None):
        self.path, self.gravity = path, gravity
        self.gravity_tau = np.arange(1., 8.)
        self.calls = []
        self.term_calls = []
        self.payload = None

    def rnea(self, q, dq, ddq):
        self.calls.append((q.copy(), dq.copy(), ddq.copy()))
        return self.gravity_tau + q + .1 * dq + 2. * ddq

    def terms(self, q, dq):
        self.term_calls.append((q.copy(), dq.copy()))
        return dq.copy(), -self.gravity_tau.copy()

    def set_payload(self, payload):
        changed = payload != self.payload
        self.payload = payload
        return changed


def sample(timestamp_us, *, q=0., dq=0., ddq=0., tau=0.):
    vector = lambda value: np.broadcast_to(np.asarray(value, float), (7,)).copy()
    return ArmState(vector(q), vector(dq), vector(ddq), np.eye(4), vector(tau),
                    np.zeros(7), timestamp_us,
                    acquired_timestamp_us=timestamp_us - 1_000)


def raw_state_filters():
    return {key: {'enabled': False} for key in ('q', 'dq', 'ddq')}


@pytest.mark.parametrize('key', ['q', 'dq', 'ddq'])
@pytest.mark.parametrize('enabled', [False, True])
def test_rnea_each_state_filter_can_be_bypassed_or_use_per_joint_cutoffs(key, enabled):
    model = FakeDynamics()
    cutoffs = np.arange(1., 8.)
    filters = raw_state_filters()
    filters[key] = {'enabled': enabled, 'cutoff_hz': cutoffs.tolist()}
    estimator = URDFTorqueEstimator(model, state_filters=filters,
                                   measured_torque_filter_enabled=False)
    first = sample(1_000_000)
    second = sample(1_010_000, q=2., dq=3., ddq=4., tau=20.)
    estimator.update(first.timestamp_us, first)
    result = estimator.update(second.timestamp_us, second)
    inputs = dict(zip(('q', 'dq', 'ddq'), model.calls[-1]))
    for name in ('q', 'dq', 'ddq'):
        expected = getattr(second, name)
        if name == key and enabled:
            expected = expected * -np.expm1(-2. * np.pi * cutoffs * .01)
        np.testing.assert_allclose(inputs[name], expected)
    np.testing.assert_array_equal(result.tau_measured, second.torque)
    np.testing.assert_allclose(result.tau_pred,
                               model.gravity_tau + inputs['q'] + .1 * inputs['dq']
                               + 2. * inputs['ddq'])


@pytest.mark.parametrize('filter_enabled', [False, True])
def test_measured_torque_filter_has_a_real_bypass(filter_enabled):
    estimator = URDFTorqueEstimator(FakeDynamics(), state_filters=raw_state_filters(),
                                   measured_torque_filter_enabled=filter_enabled,
                                   measured_torque_cutoff_hz=5.)
    estimator.update(1_000_000, sample(1_000_000, tau=3.))
    result = estimator.update(1_010_000, sample(1_010_000, tau=13.))
    expected = 3. + 10. * -np.expm1(-2. * np.pi * 5. * .01) if filter_enabled else 13.
    np.testing.assert_allclose(result.tau_measured, expected)
    np.testing.assert_allclose(result.tau_ext, expected - np.arange(1., 8.))


def test_raw_rnea_inputs_and_effort_preserve_a_changing_free_space_torque():
    model = FakeDynamics()
    estimator = URDFTorqueEstimator(model, dynamics_method='rnea',
                                   state_filters=raw_state_filters(),
                                   measured_torque_filter_enabled=False)
    for index, acceleration in enumerate((0., 8., -5.)):
        state = sample(1_000_000 + index * 10_000, q=.2 * index,
                       dq=.1 * index, ddq=acceleration)
        state.torque = model.gravity_tau + state.q + .1 * state.dq + 2. * state.ddq
        result = estimator.update(state.timestamp_us, state)
        assert result.valid
        np.testing.assert_allclose(result.tau_ext, 0., atol=1e-12)


def test_legacy_rnea_defaults_keep_raw_q_dq_and_filtered_acceleration():
    model = FakeDynamics()
    estimator = URDFTorqueEstimator(model, acceleration_cutoff_hz=4.)
    estimator.update(1_000_000, sample(1_000_000))
    estimator.update(1_010_000, sample(1_010_000, q=2., dq=3., ddq=4.))
    q, dq, ddq = model.calls[-1]
    np.testing.assert_array_equal(q, np.full(7, 2.))
    np.testing.assert_array_equal(dq, np.full(7, 3.))
    np.testing.assert_allclose(ddq, 4. * -np.expm1(-2. * np.pi * 4. * .01))


def test_momentum_ignores_unavailable_acceleration_and_counts_only_ready_new_reports():
    model = FakeDynamics()
    options = dict(dynamics_method='momentum', state_filters=raw_state_filters(),
                   measured_torque_filter_enabled=False, tau_ext_mean_window=2)
    estimator = URDFTorqueEstimator(model, **options)
    reference = URDFTorqueEstimator(FakeDynamics(), **options)
    first = sample(1_000_000, dq=.2, ddq=np.nan, tau=model.gravity_tau + 2.)
    first.ddq_valid = False
    result = estimator.update(1_001_000, first)
    reference.update(1_001_000, sample(1_000_000, dq=.2,
                                     tau=model.gravity_tau + 2.))
    assert not result.valid
    repeated = sample(first.timestamp_us, dq=100., ddq=np.inf, tau=500.)
    repeated.ddq_valid = False
    assert estimator.update(1_006_000, repeated) is result
    assert len(model.term_calls) == 1
    assert len(estimator.residuals) == 0
    for index in range(1, 60):
        state = sample(1_000_000 + index * 10_000, dq=.2, ddq=np.nan,
                       tau=model.gravity_tau + 2.)
        state.ddq_valid = False
        result = estimator.update(state.timestamp_us + 1_000, state)
        expected = reference.update(state.timestamp_us + 1_000,
                                    sample(state.timestamp_us, dq=.2,
                                           tau=model.gravity_tau + 2.))
        assert result.valid == (index >= 2)
        assert result.valid == expected.valid
        np.testing.assert_allclose(result.tau_pred, expected.tau_pred, equal_nan=True)
        np.testing.assert_allclose(result.tau_ext, expected.tau_ext, equal_nan=True)
    assert not model.calls
    np.testing.assert_allclose(result.tau_pred, model.gravity_tau, atol=1e-7)
    np.testing.assert_allclose(result.tau_ext, 2., atol=1e-7)
    assert result.source_timestamp_us == 1_589_000


def test_configured_friction_removes_signed_free_space_torque_and_is_zero_at_rest():
    model = FakeDynamics()
    friction = dict(enabled=True, source='configured', viscous_nm_s_per_rad=.5,
                    coulomb_nm=2., velocity_smoothing_rad_s=.02)
    estimator = URDFTorqueEstimator(model, friction=friction,
                                   state_filters=raw_state_filters(),
                                   measured_torque_filter_enabled=False)
    for index, velocity in enumerate((.2, 0., -.2)):
        state = sample(1_000_000 + index * 10_000, dq=velocity)
        friction_tau = .5 * velocity + 2. * np.tanh(velocity / .02)
        state.torque = model.gravity_tau + .1 * state.dq + friction_tau
        result = estimator.update(state.timestamp_us, state)
        assert result.valid
        np.testing.assert_allclose(result.tau_pred, state.torque, atol=1e-12)
        np.testing.assert_allclose(result.tau_ext, 0., atol=1e-12)
    np.testing.assert_allclose(model.calls[1][1], 0.)


def test_disabled_friction_preserves_measured_residual():
    estimator = URDFTorqueEstimator(FakeDynamics(),
                                   friction=dict(enabled=False, source='configured',
                                                 viscous_nm_s_per_rad=.5, coulomb_nm=2.),
                                   state_filters=raw_state_filters(),
                                   measured_torque_filter_enabled=False)
    state = sample(1_000_000, dq=.2, tau=np.arange(1., 8.) + .02 + 2.)
    np.testing.assert_allclose(estimator.update(state.timestamp_us, state).tau_ext,
                               2., atol=1e-12)


@pytest.mark.parametrize('method', ['rnea', 'momentum'])
def test_real_pinocchio_moving_arm_separates_urdf_friction_and_signed_contact(method):
    pin = pytest.importorskip('pinocchio')
    model = PinocchioMomentumModel(DEFAULT_URDF)
    model.set_payload({'mass_kg': .79, 'com_m': [.04, .01, .06]})
    reference_data = model.model.createData()
    model.rnea = Mock(wraps=model.rnea)
    estimator = URDFTorqueEstimator(model, dynamics_method=method,
                                   measured_torque_filter_enabled=False,
                                   state_filters=raw_state_filters(),
                                   friction={'enabled': True, 'source': 'urdf'})
    np.testing.assert_array_equal(estimator.friction.viscous, [10., 10., 5., 5., 5., 2., 2.])
    np.testing.assert_array_equal(estimator.friction.coulomb, np.ones(7))
    contact = np.array([-2., -1., -.5, 0., .5, 1., 2.])
    offset = np.linspace(-.3, .3, 7)
    amplitude = np.linspace(.05, .15, 7)
    frequency = np.linspace(1., 2., 7)
    for index in range(601):
        time_s = index * .002
        q = offset + amplitude * np.sin(frequency * time_s)
        dq = amplitude * frequency * np.cos(frequency * time_s)
        ddq = -amplitude * frequency**2 * np.sin(frequency * time_s)
        rigid_body = pin.rnea(model.model, reference_data, q, dq, ddq).copy()
        tau = rigid_body + estimator.friction.torque(dq) + contact
        state = sample(1_000_000 + index * 2_000, q=q, dq=dq,
                       ddq=ddq if method == 'rnea' else np.nan, tau=tau)
        state.ddq_valid = method == 'rnea'
        result = estimator.update(state.timestamp_us, state)
        assert result.valid == (method == 'rnea' or index > 0)
    assert result.valid
    np.testing.assert_allclose(result.tau_ext, contact,
                               atol=1e-4 if method == 'momentum' else 1e-12)
    np.testing.assert_allclose(result.tau_measured - result.tau_pred,
                               result.tau_ext, atol=1e-12)
    if method == 'momentum':
        model.rnea.assert_not_called()


def test_momentum_contact_step_has_finite_bandwidth_and_preserves_residual_sign():
    model = FakeDynamics()
    estimator = URDFTorqueEstimator(model, dynamics_method='momentum',
                                   momentum_observer_bandwidth_hz=10.,
                                   measured_torque_filter_enabled=False,
                                   state_filters=raw_state_filters())
    estimator.update(1_000_000, sample(1_000_000, tau=model.gravity_tau))
    contact = np.linspace(-2., 2., 7)
    first_contact = estimator.update(1_010_000,
                                     sample(1_010_000, tau=model.gravity_tau + contact))
    assert first_contact.valid
    assert np.all(np.abs(first_contact.tau_ext[[0, 6]]) > 0.)
    assert np.all(np.abs(first_contact.tau_ext[[0, 6]]) < 2.)
    for index in range(2, 40):
        result = estimator.update(1_000_000 + index * 10_000,
                                   sample(1_000_000 + index * 10_000,
                                          tau=model.gravity_tau + contact))
    np.testing.assert_allclose(result.tau_ext, contact, atol=1e-7)


def test_momentum_passes_filtered_q_dq_to_the_dynamics_terms():
    model = FakeDynamics()
    estimator = URDFTorqueEstimator(model, dynamics_method='momentum',
                                   measured_torque_filter_enabled=False,
                                   state_filters={'q': {'enabled': True, 'cutoff_hz': 2.},
                                                  'dq': {'enabled': True, 'cutoff_hz': 4.},
                                                  'ddq': {'enabled': True}})
    estimator.update(1_000_000, sample(1_000_000))
    state = sample(1_010_000, q=2., dq=3., ddq=np.nan)
    state.ddq_valid = False
    assert estimator.update(state.timestamp_us, state).valid
    q, dq = model.term_calls[-1]
    np.testing.assert_allclose(q, 2. * -np.expm1(-2. * np.pi * 2. * .01))
    np.testing.assert_allclose(dq, 3. * -np.expm1(-2. * np.pi * 4. * .01))
    assert not model.calls


@pytest.mark.parametrize('bad_field', ['q', 'dq', 'torque', 'q_valid', 'dq_valid', 'torque_valid'])
def test_momentum_still_releases_invalid_position_velocity_or_effort(bad_field):
    model = FakeDynamics()
    estimator = URDFTorqueEstimator(model, dynamics_method='momentum',
                                   measured_torque_filter_enabled=False)
    estimator.update(1_000_000, sample(1_000_000, tau=10.))
    assert estimator.update(1_010_000, sample(1_010_000, tau=10.)).valid
    invalid = sample(1_020_000, tau=10.)
    if bad_field.endswith('_valid'):
        setattr(invalid, bad_field, False)
    else:
        getattr(invalid, bad_field)[0] = np.nan
    assert not estimator.update(invalid.timestamp_us, invalid).valid
    assert not estimator.residuals
    assert not estimator.update(1_030_000, sample(1_030_000, tau=10.)).valid
    assert estimator.update(1_040_000, sample(1_040_000, tau=10.)).valid


@pytest.mark.parametrize('method', ['rnea', 'momentum'])
def test_reset_clears_filter_history_and_momentum_history(method):
    options = dict(dynamics_method=method, measured_torque_filter_enabled=True,
                   state_filters={name: {'enabled': True, 'cutoff_hz': 3.}
                                  for name in ('q', 'dq', 'ddq')},
                   tau_ext_mean_window=2)
    estimator = URDFTorqueEstimator(FakeDynamics(), **options)
    for index in range(4):
        state = sample(1_000_000 + index * 10_000, q=index, dq=index,
                       ddq=index * 3., tau=20. + index)
        estimator.update(state.timestamp_us, state)
    estimator.reset()
    fresh = URDFTorqueEstimator(FakeDynamics(), **options)
    for index in range(4):
        state = sample(2_000_000 + index * 10_000, q=.2, dq=.3,
                       ddq=.4, tau=5.)
        actual = estimator.update(state.timestamp_us, state)
        expected = fresh.update(state.timestamp_us, state)
        assert actual.valid == expected.valid
        for field in ('tau_pred', 'tau_measured', 'tau_ext'):
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field),
                                       equal_nan=True, atol=1e-12)


@pytest.mark.parametrize('new_timestamp', [1_100_000, 990_000])
def test_momentum_time_discontinuity_requires_observer_warmup_again(new_timestamp):
    estimator = URDFTorqueEstimator(FakeDynamics(), dynamics_method='momentum',
                                   measured_torque_filter_enabled=False)
    estimator.update(1_000_000, sample(1_000_000, tau=10.))
    assert estimator.update(1_010_000, sample(1_010_000, tau=10.)).valid
    assert not estimator.update(new_timestamp, sample(new_timestamp, tau=10.)).valid
    assert not estimator.update(new_timestamp + 1_000,
                                sample(new_timestamp, tau=10.)).valid
    assert not estimator.update(new_timestamp + 10_000,
                                sample(new_timestamp + 10_000, tau=10.)).valid
    assert estimator.update(new_timestamp + 20_000,
                            sample(new_timestamp + 20_000, tau=10.)).valid


def test_bad_momentum_dynamics_clear_residual_window_before_recovery():
    class FailingDynamics(FakeDynamics):
        fail = False

        def terms(self, q, dq):
            if self.fail:
                return np.full(7, np.nan), np.zeros(7)
            return super().terms(q, dq)

    model = FailingDynamics()
    estimator = URDFTorqueEstimator(model, dynamics_method='momentum',
        tau_ext_mean_window=2, measured_torque_filter_enabled=False)
    for index in range(3):
        timestamp = 1_000_000 + index * 10_000
        result = estimator.update(timestamp, sample(timestamp, tau=20.))
    assert result.valid
    model.fail = True
    assert not estimator.update(1_030_000, sample(1_030_000, tau=20.)).valid
    assert not estimator.residuals
    model.fail = False
    for index in range(3):
        timestamp = 1_040_000 + index * 10_000
        result = estimator.update(timestamp, sample(timestamp, tau=model.gravity_tau))
        assert result.valid == (index == 2)
    np.testing.assert_allclose(result.tau_ext, 0., atol=1e-12)


def test_rnea_rejects_scalar_model_output_before_friction_can_broadcast_it():
    model = FakeDynamics()
    model.rnea = lambda *args: 1.
    estimator = URDFTorqueEstimator(model,
        friction={'enabled': True, 'coulomb_nm': 1.})
    with pytest.raises(ValueError, match='finite seven-joint'):
        estimator.update(1_000_000, sample(1_000_000))


@pytest.mark.parametrize('setting,value', [
    ('dynamics_method', 'unknown'),
    ('measured_torque_filter_enabled', 1),
    ('momentum_observer_bandwidth_hz', 0.),
    ('momentum_observer_bandwidth_hz', [1., 2.]),
    ('state_filters', {'unknown': {'enabled': False}}),
    ('state_filters', {'q': {'enabled': 'false'}}),
    ('state_filters', {'dq': {'enabled': True, 'cutoff_hz': 0.}}),
    ('state_filters', {'ddq': {'enabled': True, 'cutoff_hz': [3., 3.]}}),
    ('friction', {'enabled': 'false'}),
    ('friction', {'source': 'unknown'}),
    ('friction', {'enabled': True, 'coulomb_nm': -1.}),
    ('friction', {'enabled': True, 'viscous_nm_s_per_rad': [1., 2.]}),
    ('friction', {'enabled': True, 'velocity_smoothing_rad_s': 0.}),
])
def test_invalid_options_are_rejected_before_any_model_is_created(setting, value, tmp_path):
    factory = Mock(side_effect=AssertionError('must validate before creating a model'))
    block = {'enabled': True, 'source': 'urdf', setting: value}
    with pytest.raises(ValueError):
        worker = TauFreeFeedbackWorker(block, tmp_path / 'config.yaml', ('right',), 100.,
                                       model_factory=factory)
        worker.prepare()
    factory.assert_not_called()


def test_worker_deep_merges_per_side_options_and_reports_effective_filter_flags(tmp_path):
    factory = Mock(side_effect=FakeDynamics)
    worker = TauFreeFeedbackWorker(
        dict(enabled=True, source='urdf', dynamics_method='momentum',
             measured_torque_filter_enabled=False, tau_ext_mean_window=1,
             momentum_observer_bandwidth_hz=[8.] * 7,
             state_filters={'q': {'enabled': True, 'cutoff_hz': 2.},
                            'dq': {'enabled': True, 'cutoff_hz': 4.},
                            'ddq': {'enabled': True, 'cutoff_hz': 6.}},
             friction={'enabled': True, 'source': 'configured',
                       'viscous_nm_s_per_rad': .5, 'coulomb_nm': 2.,
                       'velocity_smoothing_rad_s': .04},
             sides={'right': {'dynamics_method': 'rnea',
                              'measured_torque_filter_enabled': True,
                              'tau_ext_mean_window': 3,
                              'state_filters': {'q': {'enabled': False},
                                                'dq': {'cutoff_hz': 9.}},
                              'friction': {'coulomb_nm': 3.}}}),
        tmp_path / 'config.yaml', ('left', 'right'), 100., model_factory=factory)
    factory.assert_not_called()
    worker.prepare()
    assert factory.call_count == 2
    metadata = worker.metadata()
    assert metadata['dynamics_methods'] == {'left': 'momentum', 'right': 'rnea'}
    assert set(metadata['urdf_prediction_definitions']) == {'left', 'right'}
    assert metadata['tau_ext_mean_window'] == {'left': 1, 'right': 3}
    left, right = metadata['input_filters']
    assert left['tau']['enabled'] is False
    assert left['q']['enabled'] is True
    assert left['dq']['enabled'] is True
    assert left['ddq']['enabled'] is False
    assert left['ddq']['used'] is False
    assert right['tau']['enabled'] is True
    assert right['q']['enabled'] is False
    assert right['dq']['enabled'] is True
    assert right['ddq']['enabled'] is True
    np.testing.assert_allclose(right['q']['cutoff_hz'], 2.)
    np.testing.assert_allclose(right['dq']['cutoff_hz'], 9.)
    np.testing.assert_allclose(right['ddq']['cutoff_hz'], 6.)
    for name, coulomb in [('left', 2.), ('right', 3.)]:
        friction = metadata['settings'][name]['friction']
        assert friction['enabled'] is True
        assert friction['source'] == 'configured'
        np.testing.assert_allclose(friction['coulomb_nm'], coulomb)
        np.testing.assert_allclose(friction['viscous_nm_s_per_rad'], .5)
        np.testing.assert_allclose(friction['velocity_smoothing_rad_s'], .04)


def test_invalid_side_override_cannot_start_an_earlier_valid_side(tmp_path):
    factory = Mock(side_effect=AssertionError('must validate every side first'))
    with pytest.raises(ValueError):
        worker = TauFreeFeedbackWorker(
            dict(enabled=True, source='urdf', dynamics_method='momentum',
                 sides={'right': {'state_filters': {'dq': {'enabled': 1}}}}),
            tmp_path / 'config.yaml', ('left', 'right'), 100., model_factory=factory)
        worker.prepare()
    factory.assert_not_called()


def test_worker_resolves_urdf_friction_before_prepare_and_applies_it_to_the_residual(tmp_path):
    urdf = tmp_path / 'with_friction.urdf'
    urdf.write_text('<robot name="fake">' + ''.join(
        f'<joint name="joint{i}"><dynamics damping="{i}" friction="{i / 10.}"/></joint>'
        for i in range(1, 8)) + '</robot>')
    factory = Mock(side_effect=FakeDynamics)
    worker = TauFreeFeedbackWorker(
        dict(enabled=True, source='urdf', urdf_path=urdf.name,
             dynamics_method='rnea', measured_torque_filter_enabled=False,
             state_filters=raw_state_filters(),
             friction={'enabled': True, 'source': 'urdf',
                       'velocity_smoothing_rad_s': .02},
             sides={'right': {'friction': {'coulomb_nm': 0.}}}),
        tmp_path / 'config.yaml', ('left', 'right'), 100., model_factory=factory)
    factory.assert_not_called()
    worker.prepare()
    metadata = worker.metadata()
    for index, name in enumerate(('left', 'right')):
        friction = metadata['settings'][name]['friction']
        assert friction['source'] == 'urdf'
        assert friction['enabled'] is True
        assert friction['urdf_path'] == str(urdf.resolve())
        viscous = np.arange(1., 8.)
        coulomb = np.arange(1., 8.) / 10. if name == 'left' else np.zeros(7)
        np.testing.assert_allclose(friction['viscous_nm_s_per_rad'], viscous)
        np.testing.assert_allclose(friction['coulomb_nm'], coulomb)
        estimator = worker.estimators[index]
        state = sample(1_000_000, dq=.2)
        state.torque = (estimator.model.gravity_tau + .1 * state.dq
                        + viscous * state.dq + coulomb * np.tanh(state.dq / .02))
        result = estimator.update(state.timestamp_us, state)
        assert result.valid
        np.testing.assert_allclose(result.tau_ext, 0., atol=1e-12)
