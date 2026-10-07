"""Momentum feedback keeps the public H5 channels with explicit provenance."""

import json

import h5py
import numpy as np
import pytest

from nero_collection.config import (
    ArmEndpointConfig, ArmPairConfig, CollectionConfig, OutputConfig, TeleopConfig,
)
from nero_collection.h5_writer import EpisodeBuffer


MOMENTUM_DEFINITION = (
    'tau_feedback_measured - momentum_observer_residual before moving mean '
    '(reconstructed baseline; not instantaneous RNEA)')
RNEA_DEFINITION = 'RNEA(configured q, dq, ddq) from URDF + configured friction torque'


def make_buffer(tmp_path, arm_names, metadata):
    pairs = tuple(ArmPairConfig(side,
        ArmEndpointConfig(f'{side}_leader', rest_q=(0.,) * 7),
        ArmEndpointConfig(f'{side}_follower', rest_q=(0.,) * 7))
        for side in arm_names)
    config = CollectionConfig(TeleopConfig(backend='xarm', master_slave=pairs),
                              OutputConfig(tmp_path))
    buffer = EpisodeBuffer(config, arm_names, enable_online_tau_ext=False)
    buffer.episode_metadata['force_feedback'] = metadata
    return buffer


@pytest.mark.parametrize('arm_names', [('left', 'right'), ('right', 'left')])
@pytest.mark.parametrize('method_metadata', ['top_level', 'settings'])
def test_mixed_methods_keep_per_arm_provenance_and_reconstruct_feedback(
        tmp_path, arm_names, method_metadata):
    methods = {'left': 'momentum', 'right': 'rnea'}
    windows = {'left': 2, 'right': 3}
    metadata = {
        'source': 'urdf',
        'sources': {side: 'urdf' for side in arm_names},
        'settings': {side: {
            'source': 'urdf', 'dynamics_method': methods[side],
            'tau_ext_mean_window': windows[side],
            'friction': {'enabled': True, 'source': 'urdf',
                         'velocity_smoothing_rad_s': .02},
        } for side in arm_names},
        'urdf_prediction_definitions': {
            'left': MOMENTUM_DEFINITION, 'right': RNEA_DEFINITION},
    }
    if method_metadata == 'top_level':
        metadata['dynamics_methods'] = methods
    buffer = make_buffer(tmp_path, arm_names, metadata)
    residuals = {side: [] for side in arm_names}
    norms = {side: [] for side in arm_names}
    for row in range(4):
        measured, predicted, valid = [], [], []
        for side in arm_names:
            side_offset = 100. if side == 'left' else 200.
            tau = side_offset + row + np.arange(7, dtype=float)
            residual = ((-1.) ** np.arange(7)) * (1. + row * .5)
            if side == 'right':
                residual *= 2.
            residuals[side].append(residual)
            measured.append(tau)
            predicted.append(tau - residual)
            ready = row + 1 >= windows[side]
            valid.append(ready)
            norms[side].append(np.sum(np.abs(np.mean(
                residuals[side][-windows[side]:], axis=0))) if ready else np.nan)
        prediction = np.concatenate(predicted)
        buffer.append_teleop(1_000_000 + row * 10_000, {
            'tau_urdf': ('torque', prediction),
            'tau_free_pred': ('torque', prediction.copy()),
            'tau_feedback_measured': ('torque', np.concatenate(measured)),
            'tau_ext_l1': ('torque_norm', np.array([norms[side][-1] for side in arm_names])),
            'tau_free_valid': ('validity', np.array(valid, dtype=np.uint8)),
            'tau_free_source_timestamp_us': ('timestamp', np.full(len(arm_names),
                999_000 + row * 10_000, dtype=np.int64)),
            'tau_free_sample_timestamp_us': ('timestamp', np.full(len(arm_names),
                1_000_000 + row * 10_000, dtype=np.int64)),
        })
    path = buffer.save(tmp_path / 'mixed_methods.h5')
    with h5py.File(path) as h5:
        saved_metadata = json.loads(h5['metadata/episode_json'][()])
        assert saved_metadata['force_feedback'] == metadata
        for side in arm_names:
            expected_source = ('asynchronous_urdf_momentum_observer' if side == 'left'
                               else 'asynchronous_urdf_inverse_dynamics')
            expected_definition = MOMENTUM_DEFINITION if side == 'left' else RNEA_DEFINITION
            datasets = {name: h5[f'teleop/{side}_{name}_xarm'] for name in (
                'tau_urdf', 'tau_free_pred', 'tau_feedback_measured', 'tau_ext_l1')}
            for name, dataset in datasets.items():
                assert dataset.attrs['arm_name'] == side
                assert dataset.attrs['dynamics_method'] == methods[side]
                assert dataset.attrs['source'] == expected_source
                assert dataset.attrs['validity_path'] == f'teleop/{side}_tau_free_valid_xarm'
                for attr in ('validity_path', 'source_timestamp_path', 'model_sample_timestamp_path'):
                    assert dataset.attrs[attr] in h5
            assert datasets['tau_urdf'].attrs['definition'] == expected_definition
            assert datasets['tau_free_pred'].attrs['definition'] == expected_definition
            assert datasets['tau_free_pred'].attrs['compatibility_alias_of'] == (
                f'teleop/{side}_tau_urdf_xarm')
            np.testing.assert_array_equal(datasets['tau_free_pred'][:], datasets['tau_urdf'][:])
            residual = datasets['tau_feedback_measured'][:] - datasets['tau_urdf'][:]
            np.testing.assert_allclose(residual, residuals[side])
            reconstructed = np.full((4, 1), np.nan)
            for row in range(windows[side] - 1, 4):
                reconstructed[row, 0] = np.sum(np.abs(np.mean(
                    residual[row - windows[side] + 1:row + 1], axis=0)))
            np.testing.assert_allclose(datasets['tau_ext_l1'][:], reconstructed, equal_nan=True)
            assert datasets['tau_ext_l1'].attrs['moving_average_window'] == windows[side]
            assert datasets['tau_ext_l1'].attrs['moving_average_requires_full_window']
            assert 'moving_mean(tau_feedback_measured - tau_urdf)' in (
                datasets['tau_ext_l1'].attrs['definition'])
            assert f'teleop/{side}_tau_ext_xarm' not in h5


def test_legacy_urdf_metadata_keeps_existing_definitions(tmp_path):
    metadata = {'source': 'urdf', 'settings': {'right': {'tau_ext_mean_window': 20}}}
    buffer = make_buffer(tmp_path, ('right',), metadata)
    buffer.append_teleop(1_000_000, {
        'tau_urdf': ('torque', np.full(7, 2.)),
        'tau_free_pred': ('torque', np.full(7, 2.)),
        'tau_feedback_measured': ('torque', np.full(7, 3.)),
        'tau_ext_l1': ('torque_norm', np.array([7.])),
    })
    path = buffer.save(tmp_path / 'legacy.h5')
    with h5py.File(path) as h5:
        for name in ('tau_urdf', 'tau_free_pred', 'tau_feedback_measured', 'tau_ext_l1'):
            dataset = h5[f'teleop/right_{name}_xarm']
            assert dataset.attrs['source'] == 'asynchronous_urdf_inverse_dynamics'
            assert 'dynamics_method' not in dataset.attrs
        assert h5['teleop/right_tau_urdf_xarm'].attrs['definition'] == (
            'RNEA(q, dq, filtered ddq) from URDF')
        assert h5['teleop/right_tau_ext_l1_xarm'].attrs['moving_average_window'] == 20
