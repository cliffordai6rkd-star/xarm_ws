"""Seven-axis residual norms stay aligned with feedback and recorded samples."""

from types import SimpleNamespace
import threading

import h5py
import numpy as np
import pytest
import yaml

from gello_teleop.dual_gello_collect import DualGelloPipeline
from gello_teleop.gello_hardware import GelloReader
from gello_teleop.tau_free_feedback import TauFreeResult
from scripts.smoke_dual_gello_pipeline import write_simulation_config
from test_dual_gello_pipeline import CONFIG, FakeArm, FakeReader
from test_urdf_feedback_pipeline import FakeURDFModel, wait_until
from xarm_stack.torque_visualization import DEFAULT_URDF


def feedback_values(active_arms, residuals, *, valid=None):
    """Exercise the row encoder without creating any robot connections."""
    pipeline = DualGelloPipeline.__new__(DualGelloPipeline)
    pipeline.ARM_NAMES = active_arms
    pipeline.force_feedback = SimpleNamespace(source='urdf')
    pipeline.producers = [SimpleNamespace(current_snapshot=lambda: (
        np.zeros(7), np.ones(7), True)) for _ in active_arms]
    pipeline.state = 'recording'
    pipeline.follow_stop = threading.Event()
    pipeline.stop_event = threading.Event()
    valid = [True] * len(active_arms) if valid is None else valid
    results = [TauFreeResult(1_000_000, 1_001_000, np.full(7, 9.),
        np.full(7, 10.), np.full(7, 1.), np.asarray(residual, float), is_valid,
        tau_ext_unaveraged=np.full(7, 2.))
        for residual, is_valid in zip(residuals, valid)]
    return pipeline._force_feedback_values(1_002_000, results=results)


@pytest.mark.parametrize('active_arms,residuals', [
    (('right',), [[-3., .25, -1.5, 0., 2., -.75, 4.]]),
    (('left', 'right'), [[-3., .25, -1.5, 0., 2., -.75, 4.],
                        [.125, -5., 2.5, -1., 0., 3., -.375]]),
])
def test_tau_ext_l1_uses_averaged_seven_axis_residual_per_arm(active_arms, residuals):
    values = feedback_values(active_arms, residuals)
    state_name, norms = values['tau_ext_l1']
    assert state_name == 'torque_norm'
    assert norms.shape == (len(active_arms),)
    expected = np.abs(residuals).sum(axis=1)
    np.testing.assert_allclose(norms, expected)
    # The pre-mean residual, signed sum, Euclidean norm, and limited GELLO
    # command all differ, so only the actual averaged residual gives this norm.
    assert np.all(norms != 14.)
    assert np.all(norms != np.sum(residuals, axis=1))
    assert np.all(norms != np.linalg.norm(residuals, axis=1))
    assert np.all(norms != 7.)
    assert not {'tau_ext', 'tau_ext_raw', 'tau_ext_cal'} & values.keys()


@pytest.mark.parametrize('residual,is_valid,expected', [
    (np.zeros(7), True, 0.),
    (np.zeros(7), False, np.nan),
    (np.ones(7), False, np.nan),
    ([1., -2., np.nan, 0., 1., 2., 3.], True, np.nan),
    ([1., -2., np.inf, 0., 1., 2., 3.], True, np.nan),
])
def test_tau_ext_l1_distinguishes_valid_zero_from_missing_feedback(
        residual, is_valid, expected):
    values = feedback_values(('right',), [residual], valid=[is_valid])
    actual = values['tau_ext_l1'][1][0]
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected


def test_tau_ext_l1_invalid_arm_does_not_hide_the_other_arm():
    values = feedback_values(('left', 'right'),
        [np.zeros(7), [-3., .25, -1.5, 0., 2., -.75, 4.]], valid=[False, True])
    norms = values['tau_ext_l1'][1]
    assert np.isnan(norms[0])
    assert norms[1] == 11.5


@pytest.mark.parametrize('active_arms', [('left',), ('right',), ('left', 'right')])
def test_urdf_pipeline_h5_norm_matches_same_row_mean_and_invalidates_on_hold(
        tmp_path, active_arms):
    config = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw.update(active_arms=list(active_arms), cameras=[], require_cameras=False)
    raw['gripper'] = dict(enabled=False, command_enabled=False)
    raw['gello_damping'] = dict(enabled=True, damping_mode='current', sample_rate_hz=30,
        damping_gain=[0] * 7, damping_brake_gain=[0] * 7, damping_current_limit=[15] * 7,
        damping_watchdog_ms=500, weak_hold_enabled=False)
    raw['force_feedback'] = dict(enabled=True, source='urdf', urdf_path=str(DEFAULT_URDF),
        payload={'source': 'controller'}, tau_ext_mean_window=3, threshold_nm=4.,
        gain_raw_per_nm=40, ramp_s=0, rate_limit_raw_s=1e6)
    config.write_text(yaml.safe_dump(raw))
    residuals = {'left': np.array([-3., .25, -1.5, 0., 2., -.75, 4.]),
                 'right': np.array([.125, -5., 2.5, -1., 0., 3., -.375])}

    class Arm(FakeArm):
        reported_tcp_payload = {'mass_kg': .79, 'com_m': [0., 0., .04]}

        def read_state(self):
            result = super().read_state()
            result.torque[:] = residuals[self.name] + 2.79
            return result

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
    captured_residuals = {}
    encode_values = pipeline._force_feedback_values

    def capture_values(timestamp_us, *, results=None):
        if results is None:
            results = tuple(pipeline.force_feedback.snapshot(i, timestamp_us)
                for i in range(len(active_arms)))
        captured_residuals[timestamp_us] = tuple((result.tau_ext.copy(), result.valid)
            for result in results)
        return encode_values(timestamp_us, results=results)

    pipeline._force_feedback_values = capture_values
    try:
        pipeline.connect()
        pipeline.reset()
        pipeline.confirm_alignment()
        pipeline.takeover()
        pipeline.start_episode()

        def valid_norm_recorded():
            pipeline.poll()
            norms = pipeline.buffer.teleop_data.get('tau_ext_l1', ())
            return any(np.isfinite(value).all() for value in norms)

        wait_until(valid_norm_recorded)
        count_before_hold = len(pipeline.buffer.teleop_data['tau_ext_l1'])
        pipeline.freeze_following()

        def held_norm_recorded():
            pipeline.poll()
            rows = pipeline.buffer.teleop_data['tau_ext_l1']
            validity = pipeline.buffer.teleop_data['tau_free_valid']
            return (len(rows) >= count_before_hold + 4
                and all(np.isnan(row).all() for row in rows[-3:])
                and all(not np.asarray(row).any() for row in validity[-3:]))

        wait_until(held_norm_recorded)
        buffer = pipeline.buffer
        pipeline.stop_episode(save=False)
        output = buffer.save(tmp_path / 'urdf_tau_ext_l1.h5')

        with h5py.File(output) as h5:
            group = h5['teleop']
            timestamps = group['timestamp_us'][:]
            assert 'tau_ext_l1' not in group
            for arm_index, side in enumerate(active_arms):
                norms = group[f'{side}_tau_ext_l1_xarm'][:]
                valid = group[f'{side}_tau_free_valid_xarm'][:, 0].astype(bool)
                assert norms.shape == (len(timestamps), 1)
                assert valid.any() and (~valid).any()
                # Only the scalar is persisted. Match it against the actual
                # averaged worker result captured at the exact same row time.
                runtime = [captured_residuals[timestamp][arm_index] for timestamp in timestamps]
                np.testing.assert_array_equal(valid, [is_valid for _, is_valid in runtime])
                expected = np.array([np.abs(residual).sum() if is_valid else np.nan
                    for residual, is_valid in runtime])
                np.testing.assert_allclose(norms[:, 0], expected, equal_nan=True)
                np.testing.assert_allclose(norms[valid, 0], np.abs(residuals[side]).sum())
                assert np.isnan(norms[~valid]).all()
                assert np.isnan(norms[-3:]).all()
                for vector in ('tau_ext', 'tau_ext_raw', 'tau_ext_cal'):
                    assert f'{side}_{vector}_xarm' not in group
        assert pipeline.force_feedback.errors == 0
    finally:
        pipeline.close()
