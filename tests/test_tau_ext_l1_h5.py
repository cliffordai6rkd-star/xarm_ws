"""Persist only the plotted seven-joint residual norm and its per-arm provenance."""
import h5py
import numpy as np
import pytest

from nero_collection.config import (
    ArmEndpointConfig, ArmPairConfig, CollectionConfig, OutputConfig, TeleopConfig,
)
from nero_collection.h5_writer import EpisodeBuffer


def feedback_buffer(tmp_path, arm_names, source='urdf'):
    pairs = tuple(ArmPairConfig(arm,
        ArmEndpointConfig(f'{arm}_leader', rest_q=(0.,) * 7),
        ArmEndpointConfig(f'{arm}_follower', rest_q=(0.,) * 7)) for arm in arm_names)
    config = CollectionConfig(TeleopConfig(backend='xarm', master_slave=pairs), OutputConfig(tmp_path))
    buffer = EpisodeBuffer(config, arm_names, enable_online_tau_ext=False)
    buffer.episode_metadata['force_feedback'] = dict(source=source, settings={
        arm: dict(source=source, tau_ext_mean_window=index + 3)
        for index, arm in enumerate(arm_names)
    })
    return buffer


def append_residuals(buffer):
    base = (
        np.array([1., -2., 3., -4., 5., -6., 7.]),
        np.array([0., -.5, 1., -1.5, 2., -2.5, 3.]),
    )
    for row in range(3):
        residuals = np.stack([base[index] * (row + 1) for index in range(len(buffer.arm_names))])
        valid = np.array([row != (index + 1) for index in range(len(buffer.arm_names))], dtype=np.uint8)
        norms = np.sum(np.abs(residuals), axis=1)
        norms[valid == 0] = np.nan
        source_us = 1_000_000 + row * 10_000 + np.arange(len(buffer.arm_names)) * 100
        buffer.append_teleop(1_005_000 + row * 10_000, {
            'tau_ext_l1': ('torque_norm', norms),
            'tau_feedback_measured': ('torque', (residuals + 2).reshape(-1)),
            'tau_urdf': ('torque', np.full(residuals.size, 2.)),
            'tau_free_pred': ('torque', np.full(residuals.size, 2.)),
            'tau_free_valid': ('validity', valid),
            'tau_free_source_timestamp_us': ('timestamp', source_us),
            'tau_free_sample_timestamp_us': ('timestamp', source_us + 3_000),
        })


@pytest.mark.parametrize('arm_names', [('left',), ('right',), ('left', 'right'), ('right', 'left')])
def test_tau_ext_l1_preserves_scalar_layout_values_and_metadata(tmp_path, arm_names):
    buffer = feedback_buffer(tmp_path, arm_names)
    append_residuals(buffer)
    path = buffer.save(tmp_path / 'feedback.h5')
    with h5py.File(path) as h5:
        teleop = h5['teleop']
        assert 'tau_ext_l1' not in teleop
        assert {name for name in teleop if name.endswith('_tau_ext_l1_xarm')} == {
            'left_tau_ext_l1_xarm', 'right_tau_ext_l1_xarm'}
        for index, arm in enumerate(arm_names):
            norm = teleop[f'{arm}_tau_ext_l1_xarm']
            valid = teleop[f'{arm}_tau_free_valid_xarm'][:, 0].astype(bool)
            assert norm.shape == (3, 1)
            for residual_name in ('tau_ext', 'tau_ext_raw', 'tau_ext_cal'):
                assert f'{arm}_{residual_name}_xarm' not in teleop
            expected = (28. if index == 0 else 10.5) * np.arange(1, 4)
            np.testing.assert_allclose(norm[:, 0][valid], expected[valid])
            assert np.isnan(norm[:, 0][~valid]).all()
            # Mixed signs must not cancel, and arms must not contribute to one another.
            assert norm[0, 0] == (28. if index == 0 else 10.5)
            assert norm.attrs['state_name'] == 'torque_norm'
            assert norm.attrs['unit'] == 'Nm'
            assert norm.attrs['norm_order'] == 1
            assert norm.attrs['joint_count'] == 7
            assert norm.attrs['source'] == 'asynchronous_urdf_inverse_dynamics'
            assert norm.attrs['invalid_value'] == 'NaN'
            assert 'before feedback thresholds, gain, and current clipping' in norm.attrs['definition']
            assert 'sum(abs(moving_mean(tau_feedback_measured - tau_urdf)))' in norm.attrs['definition']
            assert 'per-joint moving mean precedes L1 reduction' in norm.attrs['definition']
            assert norm.attrs['input_signal'] == 'tau_ext'
            assert 'input_path' not in norm.attrs
            assert norm.attrs['validity_path'] == f'teleop/{arm}_tau_free_valid_xarm'
            assert norm.attrs['source_timestamp_path'] == f'teleop/{arm}_tau_free_source_timestamp_us_xarm'
            assert norm.attrs['model_sample_timestamp_path'] == f'teleop/{arm}_tau_free_sample_timestamp_us_xarm'
            assert norm.attrs['timestamp_path'] == 'teleop/timestamp_us'
            for key in ('validity_path', 'source_timestamp_path',
                        'model_sample_timestamp_path', 'timestamp_path'):
                assert norm.attrs[key] in h5
            assert norm.attrs['moving_average_window'] == index + 3
            assert norm.attrs['moving_average_requires_full_window']
            assert norm.attrs['arm_name'] == arm


def test_tau_ext_l1_checkpoint_source_does_not_claim_urdf_mean(tmp_path):
    buffer = feedback_buffer(tmp_path, ('right',), source='checkpoint')
    append_residuals(buffer)
    path = buffer.save(tmp_path / 'checkpoint_feedback.h5')
    with h5py.File(path) as h5:
        norm = h5['teleop/right_tau_ext_l1_xarm']
        assert norm.attrs['source'] == 'asynchronous_tau_free_sequence_model'
        assert norm.attrs['input_signal'] == 'tau_ext'
        assert 'input_path' not in norm.attrs
        assert 'moving_average_window' not in norm.attrs
        assert np.isnan(norm[1, 0])
        assert norm.attrs['arm_name'] == 'right'
        assert norm.attrs['validity_path'] == 'teleop/right_tau_free_valid_xarm'
