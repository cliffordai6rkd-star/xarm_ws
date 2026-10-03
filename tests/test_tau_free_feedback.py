from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import time

import h5py
import numpy as np
import pytest
import yaml

from gello_teleop.tau_free_feedback import (
    ResidualCurrentFeedback, TauFreeEstimator, TauFreeFeedbackWorker, TauFreeResult,
)
from nero_collection.arms.base import ArmState


CHECKPOINT = Path('model/dp/pretrained_model-20260901T082955Z-1-001/bg/epoch_003_val_tau_mse_nm2_1.386437.pt')


class FakePredictor:
    def __init__(self, config=None, *, name='', horizon=3, filters=None):
        self.metadata = SimpleNamespace(input_keys=('q', 'dq', 'delta_q'), output_key='tau',
            input_dims={'q': 7, 'dq': 7, 'delta_q': 7}, output_dim=7, sample_rate_hz=100.,
            horizon=horizon, dataloader_filters=filters or {})
        self.reset()

    def reset(self):
        self.frames = []

    def warm_up(self):
        self.reset()

    def append_and_predict(self, features):
        self.frames.append({k: v.copy() for k, v in features.items()})
        return None if len(self.frames) < self.metadata.horizon else np.full(7, 2.)


def state(timestamp_us, torque=5.):
    return ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.eye(4),
                    np.full(7, torque), np.zeros(7), timestamp_us)


def test_hard_threshold_gain_sign_limits_and_immediate_release():
    feedback = ResidualCurrentFeedback(dict(threshold_nm=1, gain_raw_per_nm=5,
        current_limit_raw=10, ramp_s=0, rate_limit_raw_s=1e6))
    residual = np.array([1, -1, 1.1, -1.1, 0, 5, -5])
    np.testing.assert_allclose(feedback.update(residual, .01), [0, 0, 5.5, -5.5, 0, 10, -10])
    np.testing.assert_array_equal(feedback.update(np.zeros(7), .01), np.zeros(7))
    feedback.update(np.ones(7)*3, .01)
    np.testing.assert_array_equal(feedback.update(residual, .01, valid=False), np.zeros(7))


def test_startup_ramp_slew_and_signed_feedback():
    feedback = ResidualCurrentFeedback(dict(threshold_nm=1, gain_raw_per_nm=10,
        current_limit_raw=20, rate_limit_raw_s=5, ramp_s=1, sign=-1))
    np.testing.assert_allclose(feedback.update(np.ones(7)*3, .1), np.full(7, -.5))
    np.testing.assert_allclose(feedback.update(np.ones(7)*3, .1), np.full(7, -1.))
    # A sign reversal releases the old direction before ramping in the new one.
    assert np.all(feedback.update(np.ones(7)*-3, .1) > 0)


def test_percentage_limits_preserve_defaults_and_bound_signed_feedback():
    feedback = ResidualCurrentFeedback(dict(
        current_limit_percent=[1.14, 1.14, .86, .86, .86, .86, .86],
        ramp_s=0, gain_raw_per_nm=100, rate_limit_raw_s=1e6))
    np.testing.assert_array_equal(feedback.limit, [20, 20, 15, 15, 15, 15, 15])
    residual = np.array([10, -10, 10, -10, 10, -10, 10])
    np.testing.assert_array_equal(feedback.update(residual, .1), [20, -20, 15, -15, 15, -15, 15])


@pytest.mark.parametrize('percent,expected', [(0, 0), (1, 18), (10, 175), (100, 1750)])
def test_scalar_percentage_uses_xl330_maximum(percent, expected):
    feedback = ResidualCurrentFeedback(dict(current_limit_percent=percent))
    np.testing.assert_array_equal(feedback.limit, np.full(7, expected))


@pytest.mark.parametrize('value', [-1, 100.01, float('nan'), float('inf'), [1, 2]])
def test_invalid_percentage_limits_are_rejected(value):
    with pytest.raises(ValueError, match='current_limit_percent'):
        ResidualCurrentFeedback(dict(current_limit_percent=value))


def test_raw_and_percentage_limits_cannot_be_combined():
    with pytest.raises(ValueError, match='not both'):
        ResidualCurrentFeedback(dict(current_limit_percent=10, current_limit_raw=20))


def test_side_percentage_override_and_metadata_show_resolved_current(tmp_path):
    worker = TauFreeFeedbackWorker(dict(enabled=True, checkpoint_path='fake.pt',
        current_limit_percent=.86, sides={'right': {'current_limit_percent': 2}}),
        tmp_path/'config.yaml', ('left', 'right'), 100., predictor_factory=FakePredictor)
    np.testing.assert_array_equal(worker.controllers[0].limit, np.full(7, 15))
    np.testing.assert_array_equal(worker.controllers[1].limit, np.full(7, 35))
    assert worker.metadata()['resolved_current_limit_raw'] == {
        'left': [15] * 7, 'right': [35] * 7}
    # A side can migrate an inherited raw limit to a percentage.
    legacy = TauFreeFeedbackWorker(dict(enabled=True, checkpoint_path='fake.pt',
        current_limit_raw=20, sides={'right': {'current_limit_percent': 2}}),
        tmp_path/'config.yaml', ('left', 'right'), 100., predictor_factory=FakePredictor)
    np.testing.assert_array_equal(legacy.controllers[0].limit, np.full(7, 20))
    np.testing.assert_array_equal(legacy.controllers[1].limit, np.full(7, 35))


@pytest.mark.parametrize('key,value', [('threshold_nm', -1), ('gain_raw_per_nm', float('nan')),
                                     ('current_limit_raw', 0), ('sign', 0), ('ramp_s', -1)])
def test_invalid_feedback_parameters_are_rejected(key, value):
    with pytest.raises(ValueError):
        ResidualCurrentFeedback({key: value})


def test_estimator_preserves_deltaq_filters_warmup_and_gap_reset():
    filters = {'dq': {'enabled': True, 'operations': [{'type': 'lowpass', 'cutoff_hz': 10.}]},
               'tau': {'enabled': True, 'operations': [{'type': 'lowpass', 'cutoff_hz': 10.}]}}
    predictor = FakePredictor(filters=filters)
    estimator = TauFreeEstimator(predictor, 100.)
    for index in range(3):
        s = state(1_000_000 + index*10_000)
        s.dq[:] = index
        result = estimator.update(s.timestamp_us, s, np.full(7, .2))
        assert result.valid == (index == 2)
    np.testing.assert_array_equal(predictor.frames[-1]['delta_q'], np.full(7, .2))
    assert np.all(predictor.frames[-1]['dq'] < 2.)
    np.testing.assert_array_equal(result.tau_ext, np.full(7, 3.))
    np.testing.assert_array_equal(result.tau_ext_raw, result.tau_ext)
    assert not estimator.update(1_100_000, state(1_100_000), np.zeros(7)).valid
    s = state(1_110_000)
    s.torque_valid = False
    assert not estimator.update(s.timestamp_us, s, np.zeros(7)).valid
    assert predictor.frames == []


def test_checkpoint_tau_filter_is_optional_and_keeps_raw_residual():
    predictor = FakePredictor(horizon=1, filters={
        'tau': {'enabled': True, 'operations': [{'type': 'lowpass', 'cutoff_hz': 10.}]}})
    estimator = TauFreeEstimator(predictor, 100., measured_tau_filter='checkpoint')
    estimator.update(1_000_000, state(1_000_000, 2.), np.zeros(7))
    result = estimator.update(1_010_000, state(1_010_000, 12.), np.zeros(7))
    assert np.all(result.tau_ext < result.tau_ext_raw)
    np.testing.assert_array_equal(result.tau_ext_raw, np.full(7, 10.))


def test_serial_worker_combines_feedback_with_damping_signs_and_total_limit():
    from gello_teleop.dual_gello_collect import _LeaderProducer
    stop = threading.Event()
    written = []
    cfg = SimpleNamespace(port='fake', gripper_id=-1, weak_hold_enabled=False, watchdog_timeout=1.,
        damping_velocity_filter_alpha=1., damping_current_limit=[20]*7, joint_signs=[-1, 1, 1, 1, 1, 1, 1])
    mapper = SimpleNamespace(config=cfg, n=7, target=lambda raw: (raw, raw))
    def write(current):
        written.append(current.copy())
        stop.set()
    reader = SimpleNamespace(read=lambda: np.zeros(7),
        compute_damping_current=lambda *args: np.full(7, -12.), write_current_damping=write)
    result = TauFreeResult(0, 0, np.zeros(7), np.zeros(7), np.ones(7)*3, np.ones(7)*3, True)
    feedback = SimpleNamespace(snapshot=lambda index: result,
        controllers=[ResidualCurrentFeedback(dict(ramp_s=0, rate_limit_raw_s=1e6, current_limit_raw=20))])
    producer = _LeaderProducer(reader, mapper, .01, stop, damping_config=cfg,
        feedback_worker=feedback, feedback_active=lambda: True)
    producer._run()
    np.testing.assert_array_equal(written[0], [-20, 3, 3, 3, 3, 3, 3])
    current, total, valid = producer.current_snapshot()
    assert valid
    np.testing.assert_array_equal(total, written[0])
    np.testing.assert_array_equal(current, [-8, 15, 15, 15, 15, 15, 15])


def test_worker_stale_prediction_and_pause_invalidation(tmp_path):
    worker = TauFreeFeedbackWorker(dict(enabled=True, checkpoint_path='fake.pt'),
        tmp_path/'config.yaml', ('right',), 100., predictor_factory=FakePredictor)
    timestamp = time.time_ns()//1000
    worker.latest = (TauFreeResult(timestamp, timestamp, np.zeros(7), np.ones(7),
                                  np.ones(7), np.ones(7), True),)
    assert worker.snapshot(0, timestamp+1000).valid
    assert not worker.snapshot(0, timestamp+200_000).valid
    worker.invalidate()
    assert not worker.snapshot(0, timestamp).valid


def test_real_checkpoint_matches_training_model_and_normalizer():
    torch = pytest.importorskip('torch')
    if not CHECKPOINT.is_file() or not Path('../PINN_LCP/model/tau_other_sequence.py').is_file():
        pytest.skip('local trained weight/source not present')
    from inference.checkpoints import _prepare_pinn_source
    from nero_collection.config import SequenceCheckpointConfig
    from nero_collection.tau_ext_inference import SequenceTorquePredictor
    _prepare_pinn_source()
    from model.tau_other_sequence import build_tau_other_sequence_model
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    reference = build_tau_other_sequence_model(checkpoint['config']).eval()
    reference.load_state_dict(checkpoint['model'])
    predictor = SequenceTorquePredictor(SequenceCheckpointConfig(checkpoint_path=CHECKPOINT), name='tau_free')
    rng = np.random.default_rng(42)
    features = {key: rng.normal(size=(50, 7))*.01 for key in ('q', 'dq', 'delta_q')}
    stats = checkpoint['normalizer']['stats']
    eps = checkpoint['normalizer']['eps']
    batch = {key: ((torch.tensor(value, dtype=torch.float32)-stats[key]['mean']) /
                   (stats[key]['std']+eps)).unsqueeze(0) for key, value in features.items()}
    with torch.inference_mode():
        expected = reference(batch)['tau_other_pred'][0]*(stats['tau']['std']+eps)+stats['tau']['mean']
    for index in range(50):
        actual = predictor.append_and_predict({key: value[index] for key, value in features.items()})
        assert (actual is not None) == (index == 49)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('real_checkpoint', [False, True])
def test_simulated_dual_feedback_records_residual_norm_and_releases_on_hold(tmp_path, real_checkpoint):
    if real_checkpoint:
        pytest.importorskip('torch')
        if not CHECKPOINT.is_file():
            pytest.skip('local trained weight not present')
    from scripts.smoke_dual_gello_pipeline import write_simulation_config
    from test_dual_gello_pipeline import FakeArm, FakeReader, CONFIG
    from gello_teleop.dual_gello_collect import DualGelloPipeline
    from gello_teleop.gello_hardware import GelloReader
    path = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(active_arms=['left', 'right'], cameras=[], require_cameras=False)
    raw['gripper'] = {'enabled': False, 'command_enabled': False}
    raw['gello_damping'] = dict(enabled=True, damping_mode='current', sample_rate_hz=30,
        damping_gain=[0]*7, damping_brake_gain=[0]*7, damping_current_limit=[15]*7,
        damping_watchdog_ms=500, weak_hold_enabled=False)
    raw['force_feedback'] = dict(enabled=True,
        checkpoint_path=str(CHECKPOINT.resolve()) if real_checkpoint else 'fake.pt', threshold_nm=1,
        gain_raw_per_nm=5, ramp_s=0, rate_limit_raw_s=10000)
    path.write_text(yaml.safe_dump(raw))
    class Arm(FakeArm):
        def read_state(self):
            result = super().read_state()
            result.torque[:] = 5.
            return result
    class Reader(FakeReader):
        compute_damping_current = staticmethod(GelloReader.compute_damping_current)
        def enable_current_damping(self, config):
            return {'enabled': True}
        def disable_current_damping(self):
            pass
        def write_current_damping(self, current):
            self.current = current.copy()
    pipeline = DualGelloPipeline(path, arm_factory=Arm, reader_factory=Reader,
                                 feedback_predictor_factory=None if real_checkpoint else FakePredictor)
    plot = pipeline.torque_visualizer = Mock()
    try:
        pipeline.connect(); pipeline.reset(); pipeline.confirm_alignment(); pipeline.takeover()
        pipeline.start_episode()
        deadline = time.monotonic()+2.
        while not all(p.current_snapshot()[2] for p in pipeline.producers):
            pipeline.poll()
            assert time.monotonic() < deadline
            time.sleep(.01)
        for reader in pipeline.readers:
            assert np.all(np.abs(reader.current) <= 15)
            if not real_checkpoint:
                np.testing.assert_array_equal(reader.current, np.asarray(reader.config.joint_signs)*15.)
        # Recording observes asynchronous predictions on the next sampling tick.
        while not any(np.asarray(v).all() for v in pipeline.buffer.teleop_data.get('tau_free_valid', ())):
            pipeline.poll()
            assert time.monotonic() < deadline
            time.sleep(.01)
        packets = [call.kwargs.get('tau_free_results') for call in plot.publish.call_args_list]
        assert any(results is not None and all(r.valid for r in results) for results in packets)
        pipeline.freeze_following()
        deadline = time.monotonic()+1.
        while any(p.current_snapshot()[2] for p in pipeline.producers):
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert all(not reader.current.any() for reader in pipeline.readers)
        buffer = pipeline.buffer
        pipeline.stop_episode(save=False)
        output = buffer.save(tmp_path/'feedback.h5')
        with h5py.File(output) as h5:
            for arm in ('left', 'right'):
                group = h5['teleop']
                valid = group[f'{arm}_tau_free_valid_xarm'][:, 0].astype(bool)
                assert valid.any()
                prediction = group[f'{arm}_tau_free_pred_xarm'][:][valid]
                measured = group[f'{arm}_tau_feedback_measured_xarm'][:][valid]
                residual_norm = group[f'{arm}_tau_ext_l1_xarm']
                norm = residual_norm[:][valid]
                assert residual_norm.shape == (len(valid), 1)
                for vector in ('tau_ext', 'tau_ext_raw', 'tau_ext_cal'):
                    assert f'{arm}_{vector}_xarm' not in group
                assert np.isfinite(prediction).all()
                np.testing.assert_allclose(measured, 5.)
                np.testing.assert_allclose(norm, np.abs(measured - prediction).sum(axis=1, keepdims=True))
                assert np.isnan(residual_norm[:][~valid]).all()
                if not real_checkpoint:
                    np.testing.assert_array_equal(prediction, np.full((valid.sum(), 7), 2.))
                    np.testing.assert_array_equal(norm, np.full((valid.sum(), 1), 21.))
                assert group[f'{arm}_current_cmd_gello'].shape[1] == 7
                assert residual_norm.attrs['source_timestamp_path'] == f'teleop/{arm}_tau_free_source_timestamp_us_xarm'
        assert pipeline.force_feedback.errors == 0
        assert pipeline.force_feedback.dropped_samples == 0
    finally:
        pipeline.close()
