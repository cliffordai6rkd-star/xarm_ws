from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gello_teleop.tau_free_feedback import (
    TauFreeFeedbackWorker, URDFTorqueEstimator,
)
from nero_collection.arms.base import ArmState
from xarm_stack.momentum import PinocchioMomentumModel
from xarm_stack.torque_visualization import DEFAULT_URDF, TorqueHistory


class Dynamics:
    def __init__(self, path=None, gravity=None):
        self.path, self.gravity = path, gravity
        self.calls = []
        self.payload = None

    def rnea(self, q, dq, ddq):
        self.calls.append((q.copy(), dq.copy(), ddq.copy()))
        return 3. + 2.*ddq + q + .1*dq

    def gravity_torque(self, q):
        return 3. + q

    def set_payload(self, payload):
        changed = payload != self.payload
        self.payload = payload
        return changed


def state(timestamp_us, *, acquired_us=0, torque=8., ddq=0.):
    return ArmState(np.zeros(7), np.zeros(7), np.full(7, ddq), np.eye(4),
                    np.full(7, torque), np.zeros(7), timestamp_us,
                    acquired_timestamp_us=acquired_us)


def test_urdf_is_immediately_ready_without_temporal_network_warmup():
    model = Dynamics()
    estimator = URDFTorqueEstimator(model)
    result = estimator.update(1_005_000, state(1_000_000, acquired_us=995_000))
    assert result.valid
    assert result.source_timestamp_us == 995_000
    assert result.sample_timestamp_us == 1_005_000
    np.testing.assert_array_equal(result.tau_pred, np.full(7, 3.))
    np.testing.assert_array_equal(result.tau_ext, np.full(7, 5.))


def test_both_filters_use_report_intervals_and_match_existing_urdf_plot():
    model = Dynamics()
    estimator = URDFTorqueEstimator(model)
    history = TorqueHistory(Dynamics(), .15, 30.)
    for index, (torque, ddq) in enumerate([(8., 0.), (18., 10.), (2., -4.)]):
        report_us = 1_000_000 + index*10_000
        # Receive/acquisition times differ: filters must still use report time.
        sample = state(report_us, acquired_us=report_us-index*3_000, torque=torque, ddq=ddq)
        result = estimator.update(report_us+5_000, sample)
        history.ingest(dict(timestamp_s=report_us/1e6, q=sample.q, dq=sample.dq,
                            ddq=sample.ddq, torque=sample.torque, q_valid=True,
                            dq_valid=True, ddq_valid=True, torque_valid=True), report_us/1e6)
        np.testing.assert_allclose(result.tau_pred, history.rows[-1][1])
        np.testing.assert_allclose(result.tau_measured, history.rows[-1][2])
        np.testing.assert_allclose(result.tau_ext_unaveraged,
                                   history.rows[-1][2]-history.rows[-1][1])
    assert np.all(model.calls[1][2] < 10.)
    np.testing.assert_allclose(result.tau_ext_raw, sample.torque-result.tau_pred)


def test_window_waits_for_distinct_reports_and_never_refreshes_old_source_time():
    model = Dynamics()
    estimator = URDFTorqueEstimator(model, tau_ext_mean_window=3)
    first = state(1_000_000, acquired_us=998_000)
    result = estimator.update(1_001_000, first)
    assert not result.valid
    repeated = state(1_000_000, acquired_us=1_005_000, torque=500., ddq=100.)
    assert estimator.update(1_006_000, repeated) is result
    assert result.source_timestamp_us == 998_000
    assert len(model.calls) == 1 and len(estimator.residuals) == 1
    result = estimator.update(1_010_000, state(1_010_000))
    assert not result.valid
    result = estimator.update(1_020_000, state(1_020_000))
    assert result.valid
    np.testing.assert_array_equal(result.tau_ext, np.full(7, 5.))


@pytest.mark.parametrize('invalid', ['dq_valid', 'ddq_valid', 'torque_valid', 'nonfinite_ddq'])
def test_invalid_motion_effort_releases_and_restarts_full_mean_window(invalid):
    estimator = URDFTorqueEstimator(Dynamics(), tau_ext_mean_window=2)
    estimator.update(1_000_000, state(1_000_000))
    assert estimator.update(1_010_000, state(1_010_000)).valid
    bad = state(1_020_000)
    if invalid == 'nonfinite_ddq':
        bad.ddq[0] = np.nan
    else:
        setattr(bad, invalid, False)
    assert not estimator.update(1_020_000, bad).valid
    assert not estimator.residuals
    assert not estimator.update(1_030_000, state(1_030_000)).valid
    assert estimator.update(1_040_000, state(1_040_000)).valid


@pytest.mark.parametrize('new_us', [1_100_000, 990_000])
def test_gap_or_backward_time_releases_then_restores_on_next_new_report(new_us):
    model = Dynamics()
    estimator = URDFTorqueEstimator(model)
    assert estimator.update(1_000_000, state(1_000_000)).valid
    assert not estimator.update(new_us, state(new_us)).valid
    assert not estimator.update(new_us+1_000, state(new_us)).valid
    assert len(model.calls) == 1
    result = estimator.update(new_us+10_000, state(new_us+10_000, torque=25.))
    assert result.valid
    np.testing.assert_array_equal(result.tau_measured, np.full(7, 25.))


def test_prepare_urdf_never_uses_checkpoint_factory_and_honors_side_overrides(tmp_path):
    urdf = tmp_path/'robot.urdf'
    urdf.write_text('<robot name="fake"/>')
    def checkpoint_forbidden(*args, **kwargs):
        pytest.fail('URDF feedback must not instantiate any checkpoint predictor')
    worker = TauFreeFeedbackWorker(dict(enabled=True, source='urdf', urdf_path='robot.urdf',
        tau_ext_mean_window=5, payload={'mass_kg': 0., 'com_m': [0., 0., 0.]},
        sides={'right': {'gravity_m_s2': [0., -9.81, 0.], 'acceleration_cutoff_hz': [2.]*7,
                         'tau_ext_mean_window': 3}}),
        tmp_path/'config.yaml', ('left', 'right'), 100.,
        predictor_factory=checkpoint_forbidden, model_factory=Dynamics)
    worker.prepare()
    assert [estimator.mean_window for estimator in worker.estimators] == [5, 3]
    assert all(estimator.model.path == str(urdf.resolve()) for estimator in worker.estimators)
    assert worker.estimators[1].model.gravity == [0., -9.81, 0.]
    assert worker.estimators[0].model.payload['mass_kg'] == 0.
    metadata = worker.metadata()
    assert metadata['source'] == 'urdf'
    assert metadata['horizon'] == [1, 1]
    assert metadata['tau_ext_mean_window'] == {'left': 5, 'right': 3}


def test_controller_payload_change_restarts_mean_without_cross_model_average(tmp_path):
    worker = TauFreeFeedbackWorker(dict(enabled=True, source='urdf', tau_ext_mean_window=2,
        payload={'source': 'controller'}), tmp_path/'config.yaml', ('right',), 100.,
        model_factory=Dynamics)
    worker.prepare()
    estimator = worker.estimators[0]
    assert not estimator.update(1_000_000, state(1_000_000)).valid
    arm = SimpleNamespace(reported_tcp_payload={'mass_kg': 1., 'com_m': [0., 0., .1]})
    worker.bind_payload_source('right', arm)
    # The first available controller payload replaces the unknown model, so
    # this report releases feedback before starting the new averaging window.
    assert not estimator.update(1_010_000, state(1_010_000)).valid
    assert not estimator.update(1_020_000, state(1_020_000)).valid
    assert estimator.update(1_030_000, state(1_030_000)).valid
    arm.reported_tcp_payload = {'mass_kg': 2., 'com_m': [0., 0., .1]}
    assert not estimator.update(1_040_000, state(1_040_000)).valid
    assert not estimator.residuals
    assert not estimator.update(1_050_000, state(1_050_000)).valid
    assert estimator.update(1_060_000, state(1_060_000)).valid


@pytest.mark.parametrize('setting,value', [
    ('source', 'unknown'), ('tau_ext_mean_window', 0), ('tau_ext_mean_window', True),
    ('tau_ext_mean_window', 1.5), ('gravity_m_s2', [0., 1.]),
    ('measured_torque_cutoff_hz', 0.), ('acceleration_cutoff_hz', [3., 3.]),
    ('urdf_path', 'missing.urdf'), ('payload', {'mass_kg': -1., 'com_m': [0.]*3}),
    ('payload', {'source': 'static'}), ('payload', {'source': 'controller', 'mass_kg': 1.}),
])
def test_invalid_settings_fail_before_model_or_hardware_start(setting, value, tmp_path):
    block = dict(enabled=True, source='urdf')
    block[setting] = value
    with pytest.raises(ValueError):
        TauFreeFeedbackWorker(block, tmp_path/'config.yaml', ('right',), 100., model_factory=Dynamics)


def test_source_cannot_be_overridden_for_one_side(tmp_path):
    with pytest.raises(ValueError, match='override'):
        TauFreeFeedbackWorker(dict(enabled=True, source='urdf',
            sides={'right': {'source': 'checkpoint'}}), tmp_path/'config.yaml',
            ('right',), 100., model_factory=Dynamics)


def test_actual_pinocchio_rnea_matches_independent_inverse_dynamics_with_payload():
    pin = pytest.importorskip('pinocchio')
    model = PinocchioMomentumModel(DEFAULT_URDF)
    model.set_payload({'mass_kg': .79, 'com_m': [.04, .01, .06]})
    estimator = URDFTorqueEstimator(model)
    sample = state(1_000_000, torque=0., ddq=.8)
    sample.q[:] = np.linspace(-.3, .3, 7)
    sample.dq[:] = np.linspace(-.2, .2, 7)
    expected = pin.rnea(model.model, model.model.createData(), sample.q, sample.dq, sample.ddq).copy()
    sample.torque[:] = expected + 2.
    result = estimator.update(sample.timestamp_us, sample)
    assert result.valid
    np.testing.assert_allclose(result.tau_pred, expected, atol=1e-12)
    np.testing.assert_allclose(result.tau_ext, 2., atol=1e-12)
