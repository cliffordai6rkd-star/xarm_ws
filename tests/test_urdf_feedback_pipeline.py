"""URDF force feedback regression tests with synthetic robots only."""

import builtins
import json
from pathlib import Path
from types import SimpleNamespace
import time

import h5py
import numpy as np
import pytest
import yaml

from gello_teleop.dual_gello_collect import DualGelloPipeline
from gello_teleop.gello_hardware import GelloReader
from gello_teleop.tau_free_feedback import TauFreeFeedbackWorker, URDFTorqueEstimator
from nero_collection.arms.base import ArmState
from scripts.smoke_dual_gello_pipeline import write_simulation_config
from test_dual_gello_pipeline import CONFIG, FakeArm, FakeReader
from xarm_stack.torque_visualization import DEFAULT_URDF


class FakeURDFModel:
    def __init__(self, urdf_path, gravity=(0., 0., -9.81)):
        self.urdf_path = Path(urdf_path)
        self.gravity = tuple(gravity)
        self.payload = None
        self.calls = []
        self.fail = False

    def set_payload(self, payload):
        changed = payload != self.payload
        self.payload = None if payload is None else dict(payload)
        return changed

    def rnea(self, q, dq, ddq):
        if self.fail:
            raise RuntimeError('synthetic RNEA failure')
        self.calls.append((q.copy(), dq.copy(), ddq.copy()))
        mass = 0. if self.payload is None else self.payload['mass_kg']
        return np.full(7, 2. + mass)


def state(timestamp_us, torque=5., ddq=0., *, acquired_us=None):
    result = ArmState(np.zeros(7), np.zeros(7), np.full(7, ddq), np.eye(4),
                      np.full(7, torque), np.zeros(7), timestamp_us,
                      acquired_timestamp_us=timestamp_us if acquired_us is None else acquired_us)
    result.ddq_valid = True
    return result


def wait_until(condition, *, timeout=2.):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, 'asynchronous feedback did not reach the expected state'
        time.sleep(.005)


def sample(torque=5.):
    timestamp = time.time_ns() // 1000
    return SimpleNamespace(timestamp_us=timestamp,
        follower_states=(state(timestamp, torque),), q_cmd=np.zeros(7), q_cmd_ok=(True,))


def test_urdf_worker_prepares_without_checkpoint_or_torch(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('URDF feedback must not import PyTorch')
        return original_import(name, *args, **kwargs)

    def no_checkpoint(*args, **kwargs):
        raise AssertionError('URDF feedback must not construct a checkpoint predictor')

    monkeypatch.setattr(builtins, '__import__', no_torch)
    worker = TauFreeFeedbackWorker(dict(enabled=True, source='urdf',
        urdf_path=str(DEFAULT_URDF), gravity_m_s2=[0., 0., -9.7]),
        tmp_path / 'config.yaml', ('right',), 100.,
        model_factory=FakeURDFModel, predictor_factory=no_checkpoint)
    try:
        worker.prepare()
        estimator, = worker.estimators
        assert estimator.model.urdf_path == DEFAULT_URDF
        assert estimator.model.gravity == (0., 0., -9.7)
        result = estimator.update(1_000_000, state(1_000_000), np.zeros(7))
        assert result.valid
        np.testing.assert_array_equal(result.tau_pred, np.full(7, 2.))
        np.testing.assert_array_equal(result.tau_ext, np.full(7, 3.))
        assert worker.settings['right']['source'] == 'urdf'
    finally:
        worker.close()


def test_urdf_filters_measured_torque_and_acceleration_before_subtraction():
    class AccelerationModel(FakeURDFModel):
        def rnea(self, q, dq, ddq):
            return super().rnea(q, dq, ddq) + ddq

    model = AccelerationModel(DEFAULT_URDF)
    estimator = URDFTorqueEstimator(model, measured_torque_cutoff_hz=3.,
        acceleration_cutoff_hz=3., tau_ext_mean_window=1)
    first = estimator.update(1_001_000, state(1_000_000, 5., 1., acquired_us=1_000_500), np.zeros(7))
    assert first.valid
    np.testing.assert_array_equal(first.tau_ext, np.full(7, 2.))
    result = estimator.update(1_011_000, state(1_010_000, 12., 10., acquired_us=1_010_500), np.zeros(7))
    alpha = 1. - np.exp(-2. * np.pi * 3. * .01)
    expected_tau = 5. + alpha * 7.
    expected_ddq = 1. + alpha * 9.
    assert result.source_timestamp_us == 1_010_500
    assert result.sample_timestamp_us == 1_011_000
    np.testing.assert_allclose(model.calls[-1][2], np.full(7, expected_ddq))
    np.testing.assert_allclose(result.tau_pred, np.full(7, 2. + expected_ddq))
    np.testing.assert_allclose(result.tau_measured, np.full(7, expected_tau))
    np.testing.assert_allclose(result.tau_ext_raw, np.full(7, 12. - 2. - expected_ddq))
    np.testing.assert_allclose(result.tau_ext, np.full(7, expected_tau - 2. - expected_ddq))


def test_urdf_mean_requires_distinct_acquisitions_and_invalid_input_rewarms():
    model = FakeURDFModel(DEFAULT_URDF)
    estimator = URDFTorqueEstimator(model, measured_torque_cutoff_hz=1e6,
        tau_ext_mean_window=3)
    first_state = state(1_000_000, 5.)
    first = estimator.update(1_000_000, first_state, np.zeros(7))
    assert not first.valid
    # Polling the same SDK report must not fill the mean window.
    assert not estimator.update(1_005_000, first_state, np.zeros(7)).valid
    assert len(model.calls) == 1
    assert not estimator.update(1_010_000, state(1_010_000, 8.), np.zeros(7)).valid
    result = estimator.update(1_020_000, state(1_020_000, 11.), np.zeros(7))
    assert result.valid
    np.testing.assert_array_equal(result.tau_ext, np.full(7, 6.))
    np.testing.assert_array_equal(result.tau_ext_unaveraged, np.full(7, 9.))
    np.testing.assert_array_equal(result.tau_ext_raw, np.full(7, 9.))
    invalid = state(1_030_000, 100.)
    invalid.torque_valid = False
    assert not estimator.update(1_030_000, invalid, np.zeros(7)).valid
    for index, torque in enumerate((14., 17., 20.)):
        timestamp = 1_040_000 + index * 10_000
        result = estimator.update(timestamp, state(timestamp, torque), np.zeros(7))
        assert result.valid == (index == 2)
    np.testing.assert_array_equal(result.tau_ext, np.full(7, 15.))


def test_urdf_acquisition_gap_restarts_the_mean_window():
    estimator = URDFTorqueEstimator(FakeURDFModel(DEFAULT_URDF), tau_ext_mean_window=3,
                                   max_gap_s=.03)
    for index in range(3):
        timestamp = 1_000_000 + index * 10_000
        result = estimator.update(timestamp, state(timestamp), np.zeros(7))
    assert result.valid
    assert not estimator.update(1_100_000, state(1_100_000), np.zeros(7)).valid
    for index in range(3):
        timestamp = 1_110_000 + index * 10_000
        result = estimator.update(timestamp, state(timestamp), np.zeros(7))
        assert result.valid == (index == 2)
    np.testing.assert_array_equal(result.tau_ext, np.full(7, 3.))


def test_urdf_worker_failure_stale_and_missing_payload_release_current(tmp_path):
    worker = TauFreeFeedbackWorker(dict(enabled=True, source='urdf',
        payload={'source': 'controller'}, tau_ext_mean_window=1,
        ramp_s=0, rate_limit_raw_s=1e6), tmp_path / 'config.yaml', ('right',),
        100., model_factory=FakeURDFModel)
    arm = SimpleNamespace(reported_tcp_payload={'mass_kg': .79, 'com_m': [0., 0., .04]})
    worker.bind_payload_source('right', arm)
    try:
        worker.start()
        worker.submit(sample(), True)
        worker.submit(sample(), True)
        wait_until(lambda: worker.snapshot(0).valid)
        result = worker.snapshot(0)
        controller = worker.controllers[0]
        assert controller.update(result.tau_ext, .01, valid=result.valid).any()
        stale = worker.snapshot(0, result.source_timestamp_us + 200_000)
        assert not stale.valid
        np.testing.assert_array_equal(controller.update(stale.tau_ext, .01, valid=stale.valid), np.zeros(7))

        model = worker.estimators[0].model
        model.fail = True
        worker.submit(sample(), True)
        wait_until(lambda: worker.errors == 1 and not worker.snapshot(0).valid)
        result = worker.snapshot(0)
        assert not result.valid
        np.testing.assert_array_equal(controller.update(result.tau_ext, .01, valid=result.valid), np.zeros(7))

        model.fail = False
        worker.submit(sample(), True)
        wait_until(lambda: worker.snapshot(0).valid)
        arm.reported_tcp_payload = None
        worker.submit(sample(), True)
        wait_until(lambda: not worker.snapshot(0).valid)
        assert worker.errors == 1
        result = worker.snapshot(0)
        np.testing.assert_array_equal(controller.update(result.tau_ext, .01, valid=result.valid), np.zeros(7))
    finally:
        worker.close()


@pytest.mark.parametrize('active_arms', [('right',), ('left', 'right')])
def test_urdf_pipeline_without_plot_records_source_and_releases_on_hold(tmp_path, active_arms):
    config = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw.update(active_arms=list(active_arms), cameras=[], require_cameras=False)
    raw['gripper'] = {'enabled': False, 'command_enabled': False}
    raw['gello_damping'] = dict(enabled=True, damping_mode='current', sample_rate_hz=30,
        damping_gain=[0] * 7, damping_brake_gain=[0] * 7, damping_current_limit=[15] * 7,
        damping_watchdog_ms=500, weak_hold_enabled=False)
    raw['force_feedback'] = dict(enabled=True, source='urdf', urdf_path=str(DEFAULT_URDF),
        payload={'source': 'controller'}, tau_ext_mean_window=3, threshold_nm=.1,
        gain_raw_per_nm=5, ramp_s=0, rate_limit_raw_s=1e6)
    config.write_text(yaml.safe_dump(raw))

    class Arm(FakeArm):
        reported_tcp_payload = {'mass_kg': .79, 'com_m': [0., 0., .04]}

        def read_state(self):
            result = super().read_state()
            result.torque[:] = 5.
            return result

        def get_tcp_load(self):
            pytest.fail('Payload must come from the cached rich report')

    class Reader(FakeReader):
        compute_damping_current = staticmethod(GelloReader.compute_damping_current)

        def enable_current_damping(self, config):
            self.current = np.zeros(7)
            return {'enabled': True}

        def disable_current_damping(self):
            self.current = np.zeros(7)

        def write_current_damping(self, current):
            self.current = current.copy()

    pipeline = DualGelloPipeline(config, arm_factory=Arm, reader_factory=Reader,
        torque_plot_enabled=False, feedback_model_factory=FakeURDFModel)
    try:
        assert pipeline.torque_visualizer is None
        pipeline.connect()
        pipeline.reset()
        pipeline.confirm_alignment()
        pipeline.takeover()
        pipeline.start_episode()

        def recorded_feedback():
            pipeline.poll()
            return (all(p.current_snapshot()[2] for p in pipeline.producers)
                and any(np.asarray(v).all() for v in pipeline.buffer.teleop_data.get('tau_free_valid', ())))

        wait_until(recorded_feedback)
        for reader in pipeline.readers:
            np.testing.assert_array_equal(reader.current, np.asarray(reader.config.joint_signs) * 11.)
        for estimator in pipeline.force_feedback.estimators:
            for key, value in Arm.reported_tcp_payload.items():
                assert estimator.model.payload[key] == value
        pipeline.freeze_following()
        wait_until(lambda: all(not p.current_snapshot()[2] for p in pipeline.producers))
        assert all(not reader.current.any() for reader in pipeline.readers)
        buffer = pipeline.buffer
        pipeline.stop_episode(save=False)
        output = buffer.save(tmp_path / 'urdf_feedback.h5')

        with h5py.File(output) as h5:
            metadata = json.loads(h5['metadata/episode_json'][()])['force_feedback']
            assert all(metadata['settings'][arm]['source'] == 'urdf' for arm in active_arms)
            assert all(metadata['settings'][arm]['tau_ext_mean_window'] == 3 for arm in active_arms)
            for arm in active_arms:
                group = h5['teleop']
                valid = group[f'{arm}_tau_free_valid_xarm'][:, 0].astype(bool)
                assert valid.any()
                model = group[f'{arm}_tau_urdf_xarm']
                np.testing.assert_allclose(model[:][valid], 2.79)
                np.testing.assert_array_equal(model[:], group[f'{arm}_tau_free_pred_xarm'][:])
                np.testing.assert_allclose(group[f'{arm}_tau_feedback_measured_xarm'][:][valid], 5.)
                for name in ('tau_ext_cal', 'tau_ext', 'tau_ext_raw'):
                    assert f'{arm}_{name}_xarm' not in group
                np.testing.assert_allclose(group[f'{arm}_tau_ext_l1_xarm'][:][valid], 7 * 2.21)
                assert model.attrs['source'] == 'asynchronous_urdf_inverse_dynamics'
                residual = group[f'{arm}_tau_ext_l1_xarm']
                assert residual.shape[1] == 1
                assert residual.attrs['norm_order'] == 1
                assert residual.attrs['moving_average_window'] == 3
                assert residual.attrs['source_timestamp_path'] == f'teleop/{arm}_tau_free_source_timestamp_us_xarm'
                assert 'mean' in residual.attrs['definition']
        assert pipeline.force_feedback.errors == 0
        assert pipeline.force_feedback.dropped_samples == 0
    finally:
        pipeline.close()
