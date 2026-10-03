"""Public GELLO/xArm names retain each arm's values, clocks, and provenance."""
import h5py
import numpy as np
import pytest

from nero_collection.config import (
    ArmEndpointConfig, ArmPairConfig, CollectionConfig, OutputConfig, TeleopConfig,
)
from nero_collection.h5_schema import DEVICE_SCHEMA_VERSION
from nero_collection.h5_writer import EpisodeBuffer, FOLLOWER_TELEOP_DATASETS


# Explicit expected names make suffix placement and raw/mapped distinctions
# independent of the production naming helper.
SIGNALS = {
    'q_follower': ('q_xarm', 'joint'),
    'q_leader': ('q_gello', 'joint'),
    'q_leader_mapped': ('q_gello', 'joint'),
    'q_leader_raw': ('q_raw_gello', 'joint'),
    'q_cmd': ('q_cmd_xarm', 'joint'),
    'delta_q': ('delta_q_xarm', 'joint'),
    'dq_cmd': ('dq_cmd_xarm', 'joint'),
    'dq_follower': ('dq_xarm', 'joint'),
    'dq_leader': ('dq_gello', 'joint'),
    'dq_valid_follower': ('dq_valid_xarm', 'flag'),
    'dq_valid_leader': ('dq_valid_gello', 'flag'),
    'ee_pose_follower': ('ee_pose_xarm', 'pose'),
    'tau_follower': ('tau_xarm', 'joint'),
    'tau_leader': ('tau_gello', 'joint'),
    'current_follower': ('current_xarm', 'joint'),
    'current_leader': ('current_gello', 'joint'),
    'torque_valid_leader': ('torque_valid_gello', 'flag'),
    'torque_valid_follower': ('torque_valid_xarm', 'flag'),
    'current_valid_leader': ('current_valid_gello', 'flag'),
    'current_valid_follower': ('current_valid_xarm', 'flag'),
    'ddq_follower': ('ddq_xarm', 'joint'),
    'ddq_follower_raw': ('ddq_raw_xarm', 'joint'),
    'q_leader_valid': ('q_valid_gello', 'flag'),
    'q_leader_timestamp_us': ('q_timestamp_us_gello', 'clock'),
    'q_leader_acquired_timestamp_us': ('q_acquired_timestamp_us_gello', 'clock'),
    'q_leader_sequence': ('q_sequence_gello', 'clock'),
    'q_leader_age_us': ('q_age_us_gello', 'clock'),
    'q_follower_timestamp_us': ('q_timestamp_us_xarm', 'clock'),
    'q_follower_acquired_timestamp_us': ('q_acquired_timestamp_us_xarm', 'clock'),
    'q_follower_sequence': ('q_sequence_xarm', 'clock'),
    'q_follower_age_us': ('q_age_us_xarm', 'clock'),
    'q_follower_valid': ('q_valid_xarm', 'flag'),
    'q_follower_repeated': ('q_repeated_xarm', 'flag'),
    'q_cmd_timestamp_us': ('q_cmd_timestamp_us_xarm', 'clock'),
    'q_cmd_send_ok': ('q_cmd_send_ok_xarm', 'flag'),
    'q_cmd_sequence': ('q_cmd_sequence_xarm', 'clock'),
    'ddq_valid_follower': ('ddq_valid_xarm', 'flag'),
    'gripper_follower_valid': ('gripper_valid_xarm', 'flag'),
    'gripper_cmd_valid': ('gripper_cmd_valid_xarm', 'flag'),
    'gripper_cmd_timestamp_us': ('gripper_cmd_timestamp_us_xarm', 'clock'),
    'gripper_follower_timestamp_us': ('gripper_timestamp_us_xarm', 'clock'),
    'gripper_leader_fraction': ('gripper_fraction_gello', 'scalar'),
    'gripper_follower': ('gripper_xarm', 'scalar'),
    'gripper_cmd': ('gripper_cmd_xarm', 'scalar'),
    'tau_g': ('tau_g_xarm', 'joint'),
    'tau_id': ('tau_id_xarm', 'joint'),
    'tau_id_filtered': ('tau_id_filtered_xarm', 'joint'),
    'tau_other_pred': ('tau_other_pred_xarm', 'joint'),
    'tau_next_pred': ('tau_next_pred_xarm', 'joint'),
    'tau_ext_cal_raw': ('tau_ext_cal_raw_xarm', 'joint'),
    'tau_ext_pred_raw': ('tau_ext_pred_raw_xarm', 'joint'),
    'tau_ext_cal': ('tau_ext_cal_xarm', 'joint'),
    'tau_ext_pred': ('tau_ext_pred_xarm', 'joint'),
    'tau_free_pred': ('tau_free_pred_xarm', 'joint'),
    'tau_urdf': ('tau_urdf_xarm', 'joint'),
    'tau_feedback_measured': ('tau_feedback_measured_xarm', 'joint'),
    'tau_ext_raw': ('tau_ext_raw_xarm', 'joint'),
    'tau_ext': ('tau_ext_xarm', 'joint'),
    'tau_ext_l1': ('tau_ext_l1_xarm', 'scalar'),
    'tau_free_valid': ('tau_free_valid_xarm', 'flag'),
    'tau_free_source_timestamp_us': ('tau_free_source_timestamp_us_xarm', 'clock'),
    'tau_free_sample_timestamp_us': ('tau_free_sample_timestamp_us_xarm', 'clock'),
    'tau_free_age_us': ('tau_free_age_us_xarm', 'clock'),
    'gello_feedback_current': ('current_feedback_gello', 'joint'),
    'gello_current_cmd': ('current_cmd_gello', 'joint'),
    'gello_feedback_valid': ('feedback_valid_gello', 'flag'),
}
SHARED = {
    'sample_lateness_us': ('duration', np.int64),
    'model_observation_updated': ('validity', np.uint8),
    'model_observation_timestamp_us': ('timestamp', np.int64),
    'model_prediction_age_us': ('duration', np.int64),
}
POSE_DATASETS = {'left': 'left_q_eepose_xarm', 'right': 'right_eepose_xarm'}


def make_buffer(tmp_path, arm_names, *, source='urdf', backend='xarm'):
    pairs = tuple(ArmPairConfig(side,
        ArmEndpointConfig(f'{side}_leader', rest_q=(0.,) * 7),
        ArmEndpointConfig(f'{side}_follower', rest_q=(0.,) * 7))
        for side in arm_names)
    config = CollectionConfig(TeleopConfig(backend=backend, master_slave=pairs), OutputConfig(tmp_path))
    buffer = EpisodeBuffer(config, arm_names, enable_online_tau_ext=False)
    buffer.episode_metadata['force_feedback'] = dict(source=source, settings={
        side: {'tau_ext_mean_window': 20} for side in arm_names})
    return buffer


def append_all_signals(buffer):
    arms = len(buffer.arm_names)
    for row in range(3):
        fields = {}
        for signal_index, (name, (_, kind)) in enumerate(SIGNALS.items()):
            base = signal_index * 1000 + row * 100
            if kind == 'joint':
                value = (np.arange(arms * 7, dtype=np.float32) + base).astype(np.float32)
                if name == 'current_follower' and row == 1:
                    value[0] = np.nan
                state = 'torque' if name.startswith('tau') else 'q'
            elif kind == 'pose':
                value = np.repeat(np.eye(4)[None], arms, axis=0)
                value[:, 0, 3] = base + np.arange(arms)
                if arms == 1:
                    value = value[0]
                state = 'ee_pose'
            elif kind == 'flag':
                value = ((row + np.arange(arms)) % 2).astype(np.uint8)
                state = 'validity'
            elif kind == 'clock':
                value = (1_000_000 + base + np.arange(arms) * 7).astype(np.int64)
                state = 'timestamp' if name.endswith('timestamp_us') else 'duration'
            else:
                value = (base + np.arange(arms) * .125).astype(np.float64)
                if name == 'tau_ext_l1' and row == 1:
                    value[-1] = np.nan
                state = 'torque_norm' if name == 'tau_ext_l1' else 'gripper'
            fields[name] = (state, value)
        # q_leader and q_leader_mapped are the same calibrated signal; raw
        # encoders remain separate and have deliberately different values.
        fields['q_leader'] = ('q', fields['q_leader_mapped'][1].copy())
        fields.update({name: (state, np.array([row % 2 if dtype == np.uint8 else
            2_000_000 + row * 100], dtype=dtype)) for name, (state, dtype) in SHARED.items()})
        buffer.append_teleop(1_000_000 + row * 10_000, fields)


@pytest.mark.parametrize('arm_names', [('left',), ('right',), ('left', 'right'), ('right', 'left')])
@pytest.mark.parametrize('source', ['urdf', 'checkpoint'])
def test_all_device_channels_preserve_values_dtypes_shapes_and_references(tmp_path, arm_names, source):
    assert set(SIGNALS) | set(SHARED) == FOLLOWER_TELEOP_DATASETS
    buffer = make_buffer(tmp_path, arm_names, source=source)
    append_all_signals(buffer)
    output = buffer.save(tmp_path / 'all_signals.h5')
    with h5py.File(output) as h5:
        teleop = h5['teleop']
        expected_names = {'timestamp_us', *SHARED}
        expected_names.update(f'{side}_{quantity}' for side in arm_names
            for quantity, kind in SIGNALS.values() if kind != 'pose')
        expected_names.update(POSE_DATASETS.values())
        expected_names.update(f'{side}_eepose_valid_xarm' for side in ('left', 'right'))
        for side in {'left', 'right'} - set(arm_names):
            expected_names.update(f'{side}_{quantity}_xarm' for quantity in
                ('q', 'dq', 'tau', 'q_cmd', 'tau_ext_l1', 'q_valid',
                 'dq_valid', 'torque_valid', 'q_cmd_send_ok', 'tau_ext_l1_valid'))
        assert set(teleop) == expected_names
        assert teleop.attrs['dataset_naming_schema'] == DEVICE_SCHEMA_VERSION
        assert teleop.attrs['dataset_layout'] == 'per_arm'
        assert list(teleop.attrs['arm_names']) == list(arm_names)
        assert set(teleop.attrs['command_datasets']) == {
            'left_q_cmd_xarm', 'right_q_cmd_xarm',
            *(f'{side}_{quantity}_xarm' for side in arm_names for quantity in ('dq_cmd', 'gripper_cmd'))}
        np.testing.assert_array_equal(teleop['timestamp_us'][:], buffer.teleop_timestamps_us)
        for name in SHARED:
            expected = np.stack(buffer.teleop_data[name])
            np.testing.assert_array_equal(teleop[name][:], expected)
            assert teleop[name].dtype == expected.dtype
        for name, (quantity, kind) in SIGNALS.items():
            original = np.stack(buffer.teleop_data[name])
            for index, side in enumerate(arm_names):
                dataset = teleop[POSE_DATASETS[side] if kind == 'pose' else f'{side}_{quantity}']
                if kind == 'joint':
                    expected = original[:, index * 7:(index + 1) * 7]
                elif kind == 'pose':
                    expected = original[:, index] if len(arm_names) > 1 else original
                else:
                    expected = original[:, index:index + 1]
                assert dataset.shape == expected.shape
                assert dataset.dtype == expected.dtype
                np.testing.assert_array_equal(dataset[:], expected)
                assert dataset.attrs['arm_name'] == side
                for value in dataset.attrs.values():
                    if isinstance(value, str) and value.startswith('teleop/'):
                        assert value in h5, (dataset.name, value)
                if source == 'urdf' and name == 'tau_free_pred':
                    assert dataset.attrs['compatibility_alias_of'] == f'teleop/{side}_tau_urdf_xarm'
                if name in {'q_follower', 'dq_follower', 'tau_follower'}:
                    assert dataset.attrs['source_timestamp_path'] == f'teleop/{side}_q_timestamp_us_xarm'
                if name == 'q_cmd':
                    assert dataset.attrs['source_timestamp_path'] == f'teleop/{side}_q_cmd_timestamp_us_xarm'
                if kind == 'pose':
                    assert dataset.attrs['source_timestamp_path'] == f'teleop/{side}_q_acquired_timestamp_us_xarm'
                    assert dataset.attrs['validity_path'] == f'teleop/{side}_eepose_valid_xarm'
                    expected_valid = np.stack(buffer.teleop_data['q_follower_valid'])[:, index:index + 1]
                    np.testing.assert_array_equal(h5[dataset.attrs['validity_path']][:], expected_valid)
        for side in arm_names:
            assert not np.array_equal(teleop[f'{side}_q_gello'][:], teleop[f'{side}_q_raw_gello'][:])
            assert teleop[f'{side}_delta_q_xarm'].attrs['definition'] == (
                f'{side}_q_cmd_xarm - {side}_q_xarm at the same state sample')


def test_disagreeing_calibrated_gello_aliases_fail_before_creating_h5(tmp_path):
    buffer = make_buffer(tmp_path, ('right',))
    buffer.append_teleop(1_000_000, {
        'q_leader': ('q', np.zeros(7)), 'q_leader_mapped': ('q', np.ones(7))})
    output = tmp_path / 'conflicting.h5'
    with pytest.raises(ValueError, match='q_leader and q_leader_mapped disagree'):
        buffer.save(output)
    assert not output.exists()
    assert not output.with_suffix('.h5.tmp').exists()


@pytest.mark.parametrize('arm_names', [('right',), ('right', 'left')])
@pytest.mark.parametrize('leader_fields', [('q_leader',), ('q_leader_mapped',),
                                        ('q_leader', 'q_leader_mapped')])
def test_calibrated_gello_alias_references_resolve_to_the_saved_channel(
        tmp_path, monkeypatch, arm_names, leader_fields):
    buffer = make_buffer(tmp_path, arm_names)
    values = np.arange(len(arm_names) * 7, dtype=float)
    fields = {'q_cmd': ('q', values + 1)}
    fields.update({name: ('q', values.copy()) for name in leader_fields})
    buffer.append_teleop(1_000_000, fields)
    finalize = buffer._finalize_teleop_data

    def with_source_references():
        data, states, attrs = finalize()
        attrs['q_cmd'].update(leader_source_path='teleop/q_leader',
                             mapped_source_path='teleop/q_leader_mapped')
        return data, states, attrs

    monkeypatch.setattr(buffer, '_finalize_teleop_data', with_source_references)
    output = buffer.save(tmp_path / 'alias_references.h5')
    with h5py.File(output) as h5:
        for side in arm_names:
            command = h5[f'teleop/{side}_q_cmd_xarm']
            for key in ('leader_source_path', 'mapped_source_path'):
                assert command.attrs[key] == f'teleop/{side}_q_gello'
                assert command.attrs[key] in h5


@pytest.mark.parametrize('arm_names', [('right',), ('left', 'right')])
def test_generic_nero_collections_keep_existing_public_names(tmp_path, arm_names):
    buffer = make_buffer(tmp_path, arm_names, backend='pyagxarm')
    buffer.append_teleop(1_000_000, {'q_follower': ('q', np.arange(len(arm_names) * 7))})
    output = buffer.save(tmp_path / 'nero.h5')
    with h5py.File(output) as h5:
        expected = {'q_follower'} if len(arm_names) == 1 else {f'{arm}_q_follower' for arm in arm_names}
        assert set(h5['teleop']) == {'timestamp_us', *expected}
        assert 'dataset_naming_schema' not in h5['teleop'].attrs


REQUIRED_SIGNALS = {
    'q_follower': ('q', 'q_valid'),
    'dq_follower': ('dq', 'dq_valid'),
    'tau_follower': ('tau', 'torque_valid'),
    'q_cmd': ('q_cmd', 'q_cmd_send_ok'),
    'tau_ext_l1': ('tau_ext_l1', 'tau_ext_l1_valid'),
}


def append_required_signals(buffer, *, missing=(), rows=3):
    arms = len(buffer.arm_names)
    for row in range(rows):
        values = {}
        for signal_index, (signal, (quantity, _)) in enumerate(REQUIRED_SIGNALS.items()):
            if signal in missing:
                continue
            width = 1 if signal == 'tau_ext_l1' else 7
            value = np.arange(arms * width, dtype=np.float32) + row * 100 + signal_index * 1000
            if signal == 'tau_ext_l1':
                value[0] = 0. if row == 0 else value[0]
            values[signal] = ('torque_norm' if width == 1 else quantity, value)
        if 'tau_ext_l1' not in missing:
            values.update(tau_free_valid=('validity', np.ones(arms, np.uint8)),
                          tau_free_source_timestamp_us=('timestamp', np.full(arms, 999_000 + row * 10_000, np.int64)),
                          tau_free_sample_timestamp_us=('timestamp', np.full(arms, 999_500 + row * 10_000, np.int64)))
        buffer.append_teleop(1_000_000 + row * 10_000, values)


@pytest.mark.parametrize('arm_names', [('left',), ('right',), ('left', 'right'), ('right', 'left')])
def test_every_xarm_episode_has_both_arms_required_channels_without_inventing_measurements(
        tmp_path, arm_names):
    import json

    buffer = make_buffer(tmp_path, arm_names)
    append_required_signals(buffer)
    output = buffer.save(tmp_path / 'required.h5')
    with h5py.File(output) as h5:
        teleop = h5['teleop']
        mandatory = {f'{side}_{quantity}_xarm' for side in ('left', 'right')
            for quantity, _ in REQUIRED_SIGNALS.values()}
        assert set(teleop.attrs['required_xarm_datasets']) == mandatory
        assert mandatory <= set(teleop)
        assert list(teleop.attrs['arm_names']) == list(arm_names)
        assert list(teleop.attrs['recorded_arm_names']) == list(arm_names)
        assert list(teleop.attrs['schema_arm_names']) == ['left', 'right']
        assert json.loads(h5['metadata/arm_names_json'][()]) == list(arm_names)
        for side in ('left', 'right'):
            for signal, (quantity, invalid_flag) in REQUIRED_SIGNALS.items():
                width = 1 if signal == 'tau_ext_l1' else 7
                dataset = teleop[f'{side}_{quantity}_xarm']
                assert dataset.shape == (3, width)
                assert dataset.attrs['arm_name'] == side
                assert dataset.attrs['device_name'] == 'xarm'
                assert dataset.attrs['timestamp_path'] == 'teleop/timestamp_us'
                if side in arm_names:
                    index = arm_names.index(side)
                    expected = np.stack(buffer.teleop_data[signal])[:, index * width:(index + 1) * width]
                    np.testing.assert_array_equal(dataset[:], expected)
                    assert dataset.dtype == expected.dtype
                    assert not dataset.attrs.get('placeholder', False)
                    if signal == 'tau_ext_l1':
                        assert dataset.attrs['moving_average_window'] == 20
                        assert dataset.attrs['moving_average_window_recorded']
                        assert dataset.attrs['validity_path'] == f'teleop/{side}_tau_free_valid_xarm'
                else:
                    assert np.isnan(dataset[:]).all()
                    assert dataset.attrs['placeholder']
                    assert dataset.attrs['source'] == 'not_recorded'
                    assert dataset.attrs['missing_reason'] == 'arm_not_recorded'
                    assert dataset.attrs['missing_context']
                    assert dataset.attrs['validity_path'] == f'teleop/{side}_{invalid_flag}_xarm'
                    validity = h5[dataset.attrs['validity_path']]
                    assert validity.shape == (3, 1)
                    assert validity.dtype == np.uint8
                    assert not validity[:].any()
                    if signal == 'tau_ext_l1':
                        assert not dataset.attrs['moving_average_window_recorded']
                        assert 'moving_average_window' not in dataset.attrs
                for value in dataset.attrs.values():
                    if isinstance(value, str) and value.startswith('teleop/'):
                        assert value in h5, (dataset.name, value)


@pytest.mark.parametrize('missing_signal', list(REQUIRED_SIGNALS))
def test_missing_active_signals_are_explicit_nan_placeholders(tmp_path, missing_signal):
    buffer = make_buffer(tmp_path, ('right',))
    if missing_signal == 'tau_ext_l1':
        buffer.episode_metadata['force_feedback']['enabled'] = False
    append_required_signals(buffer, missing=(missing_signal,))
    output = buffer.save(tmp_path / 'missing_signal.h5')
    with h5py.File(output) as h5:
        quantity, invalid_flag = REQUIRED_SIGNALS[missing_signal]
        dataset = h5[f'teleop/right_{quantity}_xarm']
        assert np.isnan(dataset[:]).all()
        assert dataset.attrs['placeholder']
        assert dataset.attrs['source'] == 'not_recorded'
        assert dataset.attrs['missing_reason'] == f'{missing_signal}_not_recorded'
        assert dataset.attrs['validity_path'] == f'teleop/right_{invalid_flag}_xarm'
        assert not h5[dataset.attrs['validity_path']][:].any()
        if missing_signal == 'tau_ext_l1':
            assert not dataset.attrs['moving_average_window_recorded']
            assert 'moving_average_window' not in dataset.attrs
        for signal, (saved_quantity, _) in REQUIRED_SIGNALS.items():
            if signal == missing_signal:
                continue
            expected = np.stack(buffer.teleop_data[signal])
            np.testing.assert_array_equal(h5[f'teleop/right_{saved_quantity}_xarm'][:], expected)


def test_missing_norm_does_not_overwrite_other_models_validity(tmp_path):
    buffer = make_buffer(tmp_path, ('right',))
    append_required_signals(buffer, missing=('tau_ext_l1',))
    buffer.teleop_data['tau_free_valid'] = [np.ones(1, np.uint8) for _ in range(3)]
    buffer.teleop_state_names['tau_free_valid'] = 'validity'
    output = buffer.save(tmp_path / 'missing_norm_other_model.h5')
    with h5py.File(output) as h5:
        norm = h5['teleop/right_tau_ext_l1_xarm']
        assert np.isnan(norm[:]).all()
        assert norm.attrs['validity_path'] == 'teleop/right_tau_ext_l1_valid_xarm'
        assert not h5[norm.attrs['validity_path']][:].any()
        assert h5['teleop/right_tau_free_valid_xarm'][:].all()


def test_missing_joint_signal_does_not_overwrite_existing_validity(tmp_path):
    buffer = make_buffer(tmp_path, ('right',))
    append_required_signals(buffer, missing=('q_follower',))
    buffer.teleop_data['q_follower_valid'] = [np.ones(1, np.uint8) for _ in range(3)]
    buffer.teleop_state_names['q_follower_valid'] = 'validity'
    output = buffer.save(tmp_path / 'missing_q_existing_validity.h5')
    with h5py.File(output) as h5:
        missing = h5['teleop/right_q_xarm']
        assert np.isnan(missing[:]).all()
        assert missing.attrs['validity_path'] == 'teleop/right_q_available_xarm'
        assert not h5[missing.attrs['validity_path']][:].any()
        assert h5['teleop/right_q_valid_xarm'][:].all()


@pytest.mark.parametrize('arm_names', [('right',), ('left', 'right')])
def test_generic_nero_norm_names_are_unchanged(tmp_path, arm_names):
    buffer = make_buffer(tmp_path, arm_names, backend='pyagxarm')
    for row in range(3):
        buffer.append_teleop(1_000_000 + row * 10_000,
            {'tau_ext_l1': ('torque_norm', row + np.arange(len(arm_names), dtype=float))})
    output = buffer.save(tmp_path / 'nero_norm.h5')
    with h5py.File(output) as h5:
        teleop = h5['teleop']
        assert set(teleop) == {'timestamp_us', *(f'{side}_tau_ext_l1' for side in arm_names)}
        for index, side in enumerate(arm_names):
            np.testing.assert_array_equal(teleop[f'{side}_tau_ext_l1'][:, 0], index + np.arange(3))
            assert f'{side}_tau_ext_l1_xarm' not in teleop


def test_empty_xarm_episode_still_has_required_zero_row_layout(tmp_path):
    buffer = make_buffer(tmp_path, ('left',))
    output = buffer.save(tmp_path / 'empty.h5')
    with h5py.File(output) as h5:
        for side in ('left', 'right'):
            for signal, (quantity, _) in REQUIRED_SIGNALS.items():
                width = 1 if signal == 'tau_ext_l1' else 7
                dataset = h5[f'teleop/{side}_{quantity}_xarm']
                assert dataset.shape == (0, width)
                assert dataset.attrs['placeholder']
                assert h5[dataset.attrs['validity_path']].shape == (0, 1)
        for side, name in POSE_DATASETS.items():
            assert h5[f'teleop/{name}'].shape == (0, 4, 4)
            assert h5[f'teleop/{side}_eepose_valid_xarm'].shape == (0, 1)


@pytest.mark.parametrize('arm_names', [('left',), ('right',), ('left', 'right')])
def test_missing_and_inactive_eepose_has_nan_placeholder_and_false_validity(tmp_path, arm_names):
    buffer = make_buffer(tmp_path, arm_names)
    append_required_signals(buffer)
    output = buffer.save(tmp_path / 'missing_eepose.h5')
    with h5py.File(output) as h5:
        teleop = h5['teleop']
        assert set(teleop.attrs['required_eepose_datasets']) == set(POSE_DATASETS.values())
        assert len(teleop.attrs['required_xarm_datasets']) == 10
        assert list(teleop.attrs['arm_names']) == list(arm_names)
        for side, name in POSE_DATASETS.items():
            dataset = teleop[name]
            assert dataset.shape == (3, 4, 4)
            assert np.isnan(dataset[:]).all()
            assert dataset.attrs['placeholder']
            assert dataset.attrs['source'] == 'not_recorded'
            assert dataset.attrs['validity_path'] == f'teleop/{side}_eepose_valid_xarm'
            validity = h5[dataset.attrs['validity_path']]
            assert validity.shape == (3, 1)
            assert validity.dtype == np.uint8
            assert not validity[:].any()


def test_eepose_validity_combines_finite_pose_and_feedback_validity(tmp_path):
    buffer = make_buffer(tmp_path, ('right',))
    poses = [np.eye(4), np.eye(4), np.eye(4)]
    poses[1][0, 3] = np.nan
    for row, pose in enumerate(poses):
        buffer.append_teleop(1_000_000 + row * 10_000, {
            'ee_pose_follower': ('ee_pose', pose),
            'q_follower_valid': ('validity', np.array([row != 2], np.uint8)),
            'q_follower_timestamp_us': ('timestamp', np.array([900_000 + row * 10_000], np.int64)),
        })
    output = buffer.save(tmp_path / 'eepose_validity.h5')
    with h5py.File(output) as h5:
        dataset = h5['teleop/right_eepose_xarm']
        np.testing.assert_array_equal(dataset[:], np.stack(poses))
        np.testing.assert_array_equal(h5['teleop/right_eepose_valid_xarm'][:, 0], [1, 0, 0])
        assert dataset.attrs['source_timestamp_path'] == 'teleop/right_q_timestamp_us_xarm'


def test_pose_only_records_use_finite_pose_validity_without_a_q_report(tmp_path):
    buffer = make_buffer(tmp_path, ('right',))
    poses = [np.eye(4), np.eye(4)]
    poses[1][2, 3] = np.nan
    for row, pose in enumerate(poses):
        buffer.append_teleop(1_000_000 + row * 10_000, {'ee_pose_follower': ('ee_pose', pose)})
    output = buffer.save(tmp_path / 'pose_only.h5')
    with h5py.File(output) as h5:
        dataset = h5['teleop/right_eepose_xarm']
        np.testing.assert_array_equal(dataset[:], np.stack(poses))
        np.testing.assert_array_equal(h5['teleop/right_eepose_valid_xarm'][:, 0], [1, 0])
        assert not h5['teleop/right_q_valid_xarm'][:].any()
        assert 'source_timestamp_path' not in dataset.attrs
