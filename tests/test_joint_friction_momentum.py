from pathlib import Path

import numpy as np
import pytest

from xarm_stack.momentum import JointFrictionModel, ExternalTorqueMomentumObserver


URDF = Path(__file__).resolve().parents[1] / 'gello_teleop/models/xarm7_dynamics.urdf'


def write_urdf(tmp_path, *, damping='2', friction='.5', missing_joint=None,
               duplicate_joint=False, duplicate_dynamics=False):
    joints = []
    for index in range(1, 8):
        if index == missing_joint:
            continue
        dynamics = f'<dynamics damping="{damping}" friction="{friction}"/>'
        if duplicate_dynamics and index == 1:
            dynamics *= 2
        joints.append(f'<joint name="joint{index}">{dynamics}</joint>')
    if duplicate_joint:
        joints.append(joints[0])
    path = tmp_path / 'friction.urdf'
    path.write_text('<robot name="test">' + ''.join(joints) + '</robot>')
    return path


def test_default_friction_does_not_enable_urdf_coefficients():
    friction = JointFrictionModel.from_config(urdf_path=URDF)
    np.testing.assert_array_equal(friction.torque(np.ones(7)), np.zeros(7))
    assert friction.metadata() == {
        'enabled': False, 'source': 'configured', 'viscous_nm_s_per_rad': [0.] * 7,
        'coulomb_nm': [0.] * 7, 'velocity_smoothing_rad_s': [.02] * 7,
    }


def test_configured_friction_is_odd_continuous_and_has_correct_units():
    viscous = np.arange(1., 8.)
    friction = JointFrictionModel.from_config(dict(enabled=True,
        viscous_nm_s_per_rad=viscous, coulomb_nm=2., velocity_smoothing_rad_s=.1))
    dq = np.full(7, .1)
    expected = viscous*.1 + 2.*np.tanh(1.)
    np.testing.assert_allclose(friction.torque(dq), expected)
    np.testing.assert_allclose(friction.torque(-dq), -expected)
    np.testing.assert_array_equal(friction.torque(np.zeros(7)), np.zeros(7))
    assert np.max(np.abs(friction.torque(np.full(7, 1e-9)))) < 1e-6
    assert friction.metadata()['viscous_nm_s_per_rad'] == viscous.tolist()


def test_real_urdf_coefficients_are_explicit_opt_in_and_overridable():
    friction = JointFrictionModel.from_config(dict(enabled=True, source='urdf'), URDF)
    assert friction.metadata()['viscous_nm_s_per_rad'] == [10., 10., 5., 5., 5., 2., 2.]
    assert friction.metadata()['coulomb_nm'] == [1.] * 7
    assert friction.metadata()['urdf_path'] == str(URDF.resolve())
    overridden = JointFrictionModel.from_config(dict(enabled=True, source='urdf',
        viscous_nm_s_per_rad=.25, coulomb_nm=[0., 1., 2., 3., 4., 5., 6.]), URDF)
    assert overridden.metadata()['viscous_nm_s_per_rad'] == [.25] * 7
    assert overridden.metadata()['coulomb_nm'] == [0., 1., 2., 3., 4., 5., 6.]
    disabled = JointFrictionModel.from_config(dict(source='urdf'), URDF)
    np.testing.assert_array_equal(disabled.torque(np.ones(7)), np.zeros(7))


@pytest.mark.parametrize('config', [
    False, [], {'extra': 1}, {'enabled': 1}, {'enabled': 'true'}, {'source': 'automatic'},
    {'source': []}, {'viscous_nm_s_per_rad': -1}, {'coulomb_nm': -1},
    {'viscous_nm_s_per_rad': float('nan')}, {'coulomb_nm': float('inf')},
    {'viscous_nm_s_per_rad': [1.] * 6}, {'coulomb_nm': [[1.] * 7]},
    {'velocity_smoothing_rad_s': 0}, {'velocity_smoothing_rad_s': -1},
    {'velocity_smoothing_rad_s': [1.] * 6}, {'coulomb_nm': True},
    {'viscous_nm_s_per_rad': [1., 1., True, 1., 1., 1., 1.]},
])
def test_invalid_friction_config_is_rejected(config):
    with pytest.raises(ValueError):
        JointFrictionModel.from_config(config, URDF)


@pytest.mark.parametrize('options', [
    {'damping': '-1'}, {'friction': '-1'}, {'damping': 'nan'}, {'friction': 'inf'},
    {'friction': 'not-a-number'}, {'missing_joint': 7}, {'duplicate_joint': True},
    {'duplicate_dynamics': True},
])
def test_invalid_urdf_coefficients_and_joint_contract_are_rejected(tmp_path, options):
    path = write_urdf(tmp_path, **options)
    with pytest.raises(ValueError):
        JointFrictionModel.from_config(dict(source='urdf', coulomb_nm=0.,
                                           viscous_nm_s_per_rad=0.), path)


def test_urdf_requires_readable_well_formed_file_and_missing_dynamics_default_to_zero(tmp_path):
    with pytest.raises(ValueError):
        JointFrictionModel.from_config(dict(source='urdf'))
    with pytest.raises(ValueError):
        JointFrictionModel.from_config(dict(source='urdf'), tmp_path / 'missing.urdf')
    path = tmp_path / 'broken.urdf'
    path.write_text('<robot>')
    with pytest.raises(ValueError):
        JointFrictionModel.from_config(dict(source='urdf'), path)
    path.write_text('<robot>' + ''.join(f'<joint name="joint{i}"/>' for i in range(1, 8)) + '</robot>')
    friction = JointFrictionModel.from_config(dict(enabled=True, source='urdf'), path)
    np.testing.assert_array_equal(friction.torque(np.ones(7)), np.zeros(7))


@pytest.mark.parametrize('velocity', [np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)])
def test_friction_rejects_invalid_velocity(velocity):
    with pytest.raises(ValueError):
        JointFrictionModel.from_config().torque(velocity)


class ConstantMassModel:
    def __init__(self):
        self.mass = np.linspace(1., 3., 7)
        self.gravity = np.linspace(-3., 3., 7)
        self.calls = 0

    def terms(self, q, dq):
        self.calls += 1
        return self.mass*dq, -self.gravity.copy()

    def rnea(self, *args):
        raise AssertionError('External momentum observer must never call RNEA or require ddq')


def test_static_contact_converges_with_measured_minus_rnea_direction_and_finite_bandwidth():
    model = ConstantMassModel()
    observer = ExternalTorqueMomentumObserver(model)
    contact = np.array([-4., -2., -1., 0., 1., 2., 4.])
    measured = model.gravity + contact
    assert np.isnan(observer.update(0., np.zeros(7), np.zeros(7), measured)).all()
    first = observer.update(.01, np.zeros(7), np.zeros(7), measured)
    assert np.all(np.abs(first[contact != 0]) < np.abs(contact[contact != 0]))
    assert np.all(np.sign(first) == np.sign(contact))
    for index in range(2, 101):
        residual = observer.update(index*.01, np.zeros(7), np.zeros(7), measured)
    np.testing.assert_allclose(residual, contact, atol=1e-10)
    np.testing.assert_allclose(observer.update(1.01, np.zeros(7), np.zeros(7),
                                             model.gravity), contact, rtol=.6)


def test_free_motion_uniform_acceleration_is_not_mistaken_for_contact():
    model = ConstantMassModel()
    observer = ExternalTorqueMomentumObserver(model)
    acceleration = np.linspace(-2., 2., 7)
    initial_velocity = np.linspace(.2, -.2, 7)
    measured = model.gravity + model.mass*acceleration
    for index in range(101):
        t = index*.01
        dq = initial_velocity + acceleration*t
        q = initial_velocity*t + .5*acceleration*t*t
        residual = observer.update(t, q, dq, measured)
        if index:
            np.testing.assert_allclose(residual, np.zeros(7), atol=1e-12)
        else:
            assert np.isnan(residual).all()


class ConfigurationDependentMassModel:
    """Independent scalar Lagrangian joints: M=2+q, C=.5*dq, g=3."""
    def terms(self, q, dq):
        return (2.+q)*dq, .5*dq*dq-3.


def test_configuration_dependent_mass_and_coriolis_do_not_create_contact():
    observer = ExternalTorqueMomentumObserver(ConfigurationDependentMassModel())
    dq = np.linspace(-.2, .2, 7)
    q0 = np.linspace(-.4, .4, 7)
    # Constant velocity: tau=.5*dq²+g; p_dot=dq². The two independent
    # dynamics terms must cancel without a supplied acceleration.
    measured = .5*dq*dq+3.
    for index in range(101):
        residual = observer.update(index*.01, q0+dq*index*.01, dq, measured)
        if index:
            np.testing.assert_allclose(residual, np.zeros(7), atol=1e-11)


def test_friction_compensation_excludes_friction_but_preserves_contact():
    model = ConstantMassModel()
    friction = JointFrictionModel.from_config(dict(enabled=True, viscous_nm_s_per_rad=2.,
                                                 coulomb_nm=.8))
    compensated = ExternalTorqueMomentumObserver(model)
    uncompensated = ExternalTorqueMomentumObserver(model)
    dq = np.linspace(-.3, .3, 7)
    friction_torque = friction.torque(dq)
    contact = np.linspace(-2., 2., 7)
    measured = model.gravity + friction_torque + contact
    for index in range(101):
        t = index*.01
        clean = compensated.update(t, dq*t, dq, measured, friction_torque)
        raw = uncompensated.update(t, dq*t, dq, measured)
    np.testing.assert_allclose(clean, contact, atol=1e-10)
    np.testing.assert_allclose(raw, contact+friction_torque, atol=1e-10)


def test_duplicate_timestamp_preserves_result_and_does_not_call_model():
    model = ConstantMassModel()
    observer = ExternalTorqueMomentumObserver(model)
    measured = model.gravity+2.
    initial = observer.update(0., np.zeros(7), np.zeros(7), measured)
    assert np.isnan(initial).all()
    assert np.isnan(observer.update(0., np.ones(7), np.ones(7), measured+20.)).all()
    first = observer.update(.01, np.zeros(7), np.zeros(7), measured)
    repeated = observer.update(.01, np.ones(7), np.ones(7), measured+20.)
    np.testing.assert_array_equal(repeated, first)
    assert model.calls == 2
    repeated[:] = 999.
    np.testing.assert_array_equal(observer.latest, first)


@pytest.mark.parametrize('restart', [.10, -.01])
def test_gap_and_backward_timestamp_require_a_new_seed(restart):
    model = ConstantMassModel()
    observer = ExternalTorqueMomentumObserver(model, max_gap_s=.03)
    for index in range(11):
        observer.update(index*.01, np.zeros(7), np.zeros(7), model.gravity+4.)
    timestamp = .1 + restart
    result = observer.update(timestamp, np.zeros(7), np.zeros(7), model.gravity)
    assert np.isnan(result).all()
    assert np.isnan(observer.update(timestamp, np.zeros(7), np.zeros(7), model.gravity)).all()
    np.testing.assert_allclose(observer.update(timestamp+.01, np.zeros(7), np.zeros(7),
                                             model.gravity), np.zeros(7), atol=1e-12)


@pytest.mark.parametrize('invalid', ['flag', 'timestamp', 'q', 'dq', 'measured', 'friction', 'shape'])
def test_invalid_inputs_release_and_reset_contact_history(invalid):
    model = ConstantMassModel()
    observer = ExternalTorqueMomentumObserver(model)
    observer.update(0., np.zeros(7), np.zeros(7), model.gravity+4.)
    observer.update(.01, np.zeros(7), np.zeros(7), model.gravity+4.)
    args = dict(timestamp_s=.02, q=np.zeros(7), dq=np.zeros(7), measured_torque=model.gravity)
    if invalid == 'flag':
        args['valid'] = False
    elif invalid == 'timestamp':
        args['timestamp_s'] = np.nan
    elif invalid == 'shape':
        args['measured_torque'] = np.zeros(6)
    else:
        key = {'measured': 'measured_torque', 'friction': 'friction_torque'}.get(invalid, invalid)
        args[key] = np.full(7, np.nan)
    assert np.isnan(observer.update(**args)).all()
    assert observer.timestamp_s is None
    assert np.isnan(observer.update(.03, np.zeros(7), np.zeros(7), model.gravity)).all()
    np.testing.assert_allclose(observer.update(.04, np.zeros(7), np.zeros(7), model.gravity), 0., atol=1e-12)


@pytest.mark.parametrize('bandwidth', [0., -1., np.nan, np.inf, [10.] * 6, True])
def test_observer_rejects_invalid_bandwidth(bandwidth):
    with pytest.raises(ValueError):
        ExternalTorqueMomentumObserver(ConstantMassModel(), bandwidth)


@pytest.mark.parametrize('gap', [0., -1., np.nan, np.inf])
def test_observer_rejects_invalid_sample_gap(gap):
    with pytest.raises(ValueError):
        ExternalTorqueMomentumObserver(ConstantMassModel(), max_gap_s=gap)


def test_observer_bandwidth_controls_response_without_extra_filters():
    model = ConstantMassModel()
    slow = ExternalTorqueMomentumObserver(model, bandwidth_hz=1.)
    fast = ExternalTorqueMomentumObserver(model, bandwidth_hz=[10.] * 7)
    for index in range(21):
        low = slow.update(index*.01, np.zeros(7), np.zeros(7), model.gravity+2.)
        high = fast.update(index*.01, np.zeros(7), np.zeros(7), model.gravity+2.)
    assert np.all(high > low)
    np.testing.assert_allclose(high, np.full(7, 2.), atol=1e-4)
    assert np.all((low > 1.) & (low < 1.6))
