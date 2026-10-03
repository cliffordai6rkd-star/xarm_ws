"""Metadata-only migration preserves payloads and can restore every attribute."""
from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest

from nero_collection.h5_schema import (
    DEVICE_SCHEMA_VERSION, ensure_required_xarm_datasets, public_dataset_name,
)
from scripts import migrate_h5_device_schema as migration


def create_episode(path, arms=('right',), *, metadata=True):
    values = {}
    with h5py.File(path, 'w') as episode:
        episode.attrs['format'] = 'ufactory_multimodal_episode/v1'
        episode.attrs['original_numbers'] = np.array([2, 7], np.int16)
        teleop = episode.create_group('teleop')
        teleop.attrs['arm_names'] = arms
        teleop.attrs['dataset_layout'] = 'concatenated' if len(arms) == 1 else 'per_arm'
        teleop.attrs['command_datasets'] = np.array(
            ['q_cmd' if len(arms) == 1 else f'{arm}_q_cmd' for arm in arms], dtype='S24')
        teleop.create_dataset('timestamp_us', data=np.arange(3, dtype=np.int64) * 10_000 + 1_000_000)
        teleop.create_dataset('sample_lateness_us', data=np.zeros((3, 1), dtype=np.int64))
        for index, arm in enumerate(arms):
            prefix = '' if len(arms) == 1 else f'{arm}_'
            q = np.array([[-1., 2., -3., 4., -5., 6., -7.]], dtype=np.float32)
            q = np.repeat(q + 10 * index, 3, axis=0)
            for name, data in {
                'q_follower': q,
                'q_leader_mapped': q + 2,
                'q_cmd': q + .5,
                'delta_q': np.full_like(q, .5),
                'dq_valid_follower': np.ones((3, 1), dtype=np.uint8),
                'q_follower_timestamp_us': np.arange(3, dtype=np.int64).reshape(3, 1),
                'ee_pose_follower': np.repeat(np.eye(4)[None], 3, axis=0),
                'gello_current_cmd': np.arange(21, dtype=np.int16).reshape(3, 7),
                'tau_ext': q - 3,
            }.items():
                key = prefix + name
                dataset = teleop.create_dataset(key, data=data, compression='gzip', compression_opts=3)
                dataset.attrs['timestamp_path'] = 'teleop/timestamp_us'
                values[key] = data
            q_dataset = teleop[prefix + 'q_follower']
            q_dataset.attrs['source_timestamp_path'] = f'teleop/{prefix}q_follower_timestamp_us'
            q_dataset.attrs['validity_path'] = f'teleop/{prefix}dq_valid_follower'
            q_dataset.attrs['provenance_json'] = json.dumps({'input': f'/teleop/{prefix}q_follower'})
            norm = teleop.create_dataset(f'{arm}_tau_ext_l1', data=np.sum(np.abs(q - 3), axis=1, keepdims=True))
            norm.attrs['input_path'] = f'teleop/{prefix}tau_ext'
        camera = episode.create_dataset('cameras/wrist/frames', data=np.arange(54, dtype=np.uint8).reshape(2, 3, 3, 3),
                                        compression='gzip')
        camera.attrs['reference_path'] = ('teleop/q_follower' if len(arms) == 1
                                         else f'teleop/{arms[0]}_q_follower')
        if metadata:
            episode.create_dataset('metadata/episode_json', data='{"original":true}',
                                   dtype=h5py.string_dtype('utf-8'))
    return values


def object_info(episode):
    result = {}
    def record(name, obj):
        if isinstance(obj, h5py.Dataset):
            result[name] = (h5py.h5o.get_info(obj.id).addr, obj.dtype.str, obj.shape,
                            obj.compression, obj.compression_opts, hashlib.sha256(obj[:].tobytes()).hexdigest()
                            if obj.ndim else obj[()])
    episode.visititems(record)
    return result


@pytest.mark.parametrize('arms', [('right',), ('left',), ('left', 'right'), ('right', 'left')])
def test_migration_preserves_objects_payloads_and_all_references(tmp_path, arms):
    path = tmp_path / 'episode.h5'
    create_episode(path, arms)
    digest = hashlib.sha256(path.read_bytes()).digest()
    with h5py.File(path, 'r') as episode:
        before = object_info(episode)
        old_attrs = migration._snapshot(episode)
    plan = migration.migrate_file(path)
    assert plan.changed
    assert hashlib.sha256(path.read_bytes()).digest() == digest
    migration.migrate_file(path, apply=True)
    renames = dict(plan.renames)
    with h5py.File(path, 'r') as episode:
        after = object_info(episode)
        for name, info in before.items():
            old = name.removeprefix('teleop/')
            target = 'teleop/' + renames.get(old, old) if name.startswith('teleop/') else name
            assert after[target] == info
        teleop = episode['teleop']
        assert teleop.attrs['dataset_naming_schema'] == DEVICE_SCHEMA_VERSION
        assert teleop.attrs['dataset_layout'] == 'per_arm'
        assert list(teleop.attrs['command_datasets']) == [f'{arm}_q_cmd_xarm' for arm in arms]
        for arm in arms:
            q = teleop[f'{arm}_q_xarm']
            assert q.attrs['arm_name'] == arm
            assert q.attrs['source_signal_name'] == 'q_follower'
            assert q.attrs['device_name'] == 'xarm'
            assert q.attrs['source_timestamp_path'] == f'teleop/{arm}_q_timestamp_us_xarm'
            assert q.attrs['validity_path'] == f'teleop/{arm}_dq_valid_xarm'
            assert json.loads(q.attrs['provenance_json'])['input'] == f'/teleop/{arm}_q_xarm'
            assert teleop[f'{arm}_tau_ext_l1_xarm'].attrs['input_path'] == f'teleop/{arm}_tau_ext_xarm'
            assert teleop[f'{arm}_tau_ext_l1_xarm'].attrs['source_signal_name'] == 'tau_ext_l1'
            assert teleop[f'{arm}_tau_ext_l1_xarm'].attrs['device_name'] == 'xarm'
            assert teleop[f'{arm}_delta_q_xarm'].attrs['definition'] == (
                f'{arm}_q_cmd_xarm - {arm}_q_xarm at the same state sample')
            assert f'{arm}_q_gello' in teleop and f'{arm}_current_cmd_gello' in teleop
        assert episode['cameras/wrist/frames'].attrs['reference_path'] == f'teleop/{arms[0]}_q_xarm'
        journal = migration._load_journal(episode)
        assert journal['status'] == 'complete'
    second = migration.migrate_file(path, apply=True)
    assert not second.changed
    assert migration.undo_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert object_info(episode) == before
        assert migration._snapshot(episode) == old_attrs
        assert migration.JOURNAL_PATH not in episode
    assert not migration.undo_file(path, apply=True)


@pytest.mark.parametrize('collision', ['existing_target', 'calibrated_alias'])
def test_collision_is_rejected_before_any_file_changes(tmp_path, collision):
    path = tmp_path / 'episode.h5'
    create_episode(path)
    with h5py.File(path, 'r+') as episode:
        key = 'right_q_xarm' if collision == 'existing_target' else 'q_leader'
        episode['teleop'].create_dataset(key, data=np.full((3, 7), -200.))
    digest = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(ValueError, match='collision'):
        migration.migrate_file(path, apply=True)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


def test_concatenated_dual_and_bad_scalar_shapes_fail_before_writes(tmp_path):
    path = tmp_path / 'episode.h5'
    create_episode(path, ('left', 'right'))
    with h5py.File(path, 'r+') as episode:
        episode['teleop'].create_dataset('q_follower', data=np.zeros((3, 14)))
    digest = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(ValueError, match='concatenated dual-arm'):
        migration.migrate_file(path, apply=True)
    assert hashlib.sha256(path.read_bytes()).digest() == digest
    with h5py.File(path, 'r+') as episode:
        del episode['teleop/q_follower']
        del episode['teleop/left_dq_valid_follower']
        episode['teleop'].create_dataset('left_dq_valid_follower', data=np.zeros((3, 2)))
    digest = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(ValueError, match='scalar per arm'):
        migration.migrate_file(path, apply=True)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


@pytest.mark.parametrize('metadata', [True, False])
def test_mutation_failure_rolls_back_names_attributes_and_journal(tmp_path, monkeypatch, metadata):
    path = tmp_path / 'episode.h5'
    create_episode(path, metadata=metadata)
    with h5py.File(path, 'r') as episode:
        before = object_info(episode)
        attrs = migration._snapshot(episode)
    original = migration._rewrite_attributes
    def fail(episode, plan):
        original(episode, plan)
        raise RuntimeError('injected attribute failure')
    monkeypatch.setattr(migration, '_rewrite_attributes', fail)
    with pytest.raises(RuntimeError, match='injected'):
        migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert object_info(episode) == before
        assert migration._snapshot(episode) == attrs
        assert migration.JOURNAL_PATH not in episode
        assert ('metadata' in episode) == metadata


def test_no_camera_payloads_are_read_during_migration(tmp_path, monkeypatch):
    path = tmp_path / 'episode.h5'
    create_episode(path)
    original = h5py.Dataset.__getitem__
    def guarded(dataset, key):
        if dataset.name.startswith('/cameras/'):
            raise AssertionError('migration attempted to read camera payload')
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded)
    migration.migrate_file(path, apply=True)
    migration.undo_file(path, apply=True)


def test_incomplete_journal_can_recover_partial_link_moves(tmp_path):
    path = tmp_path / 'episode.h5'
    create_episode(path)
    with h5py.File(path, 'r') as episode:
        before = object_info(episode)
    migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r+') as episode:
        journal = migration._load_journal(episode)
        journal['status'] = 'pending'
        first_old, first_new = journal['renames'][0]
        episode['teleop'].move(first_new, first_old)
        migration._write_journal(episode, journal)
    with pytest.raises(ValueError, match='incomplete migration journal'):
        migration.migrate_file(path)
    migration.undo_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert object_info(episode) == before


def test_undo_collision_does_not_overwrite_a_later_dataset(tmp_path):
    path = tmp_path / 'episode.h5'
    create_episode(path)
    migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r+') as episode:
        episode['teleop'].create_dataset('q_follower', data=np.full((3, 7), 999.))
    digest = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(ValueError, match='both'):
        migration.undo_file(path, apply=True)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


def test_directory_preflight_skips_active_temporary_files_and_batches_collisions(tmp_path):
    first = tmp_path / '0001.h5'
    second = tmp_path / '0002.h5'
    create_episode(first)
    create_episode(second)
    (tmp_path / '0003.h5.tmp').write_text('active temporary file')
    with h5py.File(second, 'r+') as episode:
        episode['teleop'].create_dataset('right_q_xarm', data=np.zeros((3, 7)))
    digest = hashlib.sha256(first.read_bytes()).digest()
    assert migration.episode_paths([tmp_path]) == (first, second)
    with pytest.raises(SystemExit):
        migration.main([str(tmp_path), '--in-place'])
    assert hashlib.sha256(first.read_bytes()).digest() == digest


def test_unknown_signals_are_not_silently_left_in_old_format(tmp_path):
    path = tmp_path / 'episode.h5'
    create_episode(path)
    with h5py.File(path, 'r+') as episode:
        episode['teleop'].create_dataset('mystery_measurement', data=np.zeros((3, 7)))
    digest = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(ValueError, match='Unknown teleop signal'):
        migration.migrate_file(path, apply=True)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


def test_fixed_utf8_and_enum_attribute_types_roundtrip_without_other_attr_changes(tmp_path):
    path = tmp_path / 'episode.h5'
    create_episode(path)
    with h5py.File(path, 'r+') as episode:
        episode.attrs.create('fixed_utf8', np.array(['fixed', 'text'], dtype='S12'),
                             dtype=h5py.string_dtype('utf-8', 12))
        episode.attrs.create('enum', np.array([1, 2], dtype=np.uint8),
                             dtype=h5py.enum_dtype({'one': 1, 'two': 2}, basetype=np.uint8))
        camera = episode['cameras/wrist/frames']
        camera.attrs.create('unchanged_strings', np.array(['a', 'b'], dtype='S4'))
    migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert episode['cameras/wrist/frames'].attrs.get_id('unchanged_strings').dtype == np.dtype('S4')
        assert h5py.check_enum_dtype(episode.attrs.get_id('enum').dtype) == {'one': 1, 'two': 2}
    migration.undo_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        string = h5py.check_string_dtype(episode.attrs.get_id('fixed_utf8').dtype)
        assert string.encoding == 'utf-8' and string.length == 12
        assert h5py.check_enum_dtype(episode.attrs.get_id('enum').dtype) == {'one': 1, 'two': 2}
        np.testing.assert_array_equal(episode.attrs['enum'], [1, 2])


@pytest.mark.parametrize('problem', ['missing_object', 'invalid_attr'])
def test_undo_preflight_is_read_only_and_rejects_invalid_journal_targets(tmp_path, problem):
    path = tmp_path / 'episode.h5'
    create_episode(path)
    migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r+') as episode:
        if problem == 'missing_object':
            del episode['cameras/wrist/frames']
        else:
            journal = migration._load_journal(episode)
            journal['original_attributes']['/']['bad_attr'] = {'dtype': 'invalid'}
            migration._write_journal(episode, journal)
    digest = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(ValueError, match='Cannot restore|Cannot decode'):
        migration.undo_file(path)
    with pytest.raises(ValueError, match='Cannot restore|Cannot decode'):
        migration.undo_file(path, apply=True)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


def test_unsupported_later_file_attrs_are_rejected_by_whole_batch_preflight(tmp_path):
    first = tmp_path / '0001.h5'
    second = tmp_path / '0002.h5'
    create_episode(first)
    create_episode(second)
    with h5py.File(second, 'r+') as episode:
        episode.attrs['object_reference'] = episode['cameras/wrist/frames'].ref
    digest = hashlib.sha256(first.read_bytes()).digest()
    with pytest.raises(ValueError, match='unsupported dtype metadata'):
        migration.migrate_file(second)
    with pytest.raises(SystemExit):
        migration.main([str(tmp_path), '--in-place'])
    assert hashlib.sha256(first.read_bytes()).digest() == digest


def test_initial_journal_failure_rolls_back_created_metadata(tmp_path, monkeypatch):
    path = tmp_path / 'episode.h5'
    create_episode(path, metadata=False)
    original = migration._write_journal
    def fail(episode, journal):
        original(episode, journal)
        raise RuntimeError('injected journal flush failure')
    monkeypatch.setattr(migration, '_write_journal', fail)
    with pytest.raises(RuntimeError, match='journal flush failure'):
        migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert 'metadata' not in episode
        assert 'q_follower' in episode['teleop']
        assert 'right_q_xarm' not in episode['teleop']


@pytest.mark.parametrize('canonical_only', [False, True])
def test_already_canonical_signals_get_valid_metadata_and_preserve_source_names(tmp_path, canonical_only):
    path = tmp_path / 'episode.h5'
    create_episode(path)
    original_plan = migration.migrate_file(path)
    with h5py.File(path, 'r+') as episode:
        teleop = episode['teleop']
        for old, new in original_plan.renames:
            if canonical_only or old in {'q_follower', 'q_leader_mapped'}:
                teleop.move(old, new)
        teleop['right_q_gello'].attrs['source_signal_name'] = 'q_leader'
        before = object_info(episode)
        attrs = migration._snapshot(episode)
    plan = migration.migrate_file(path, apply=True)
    assert plan.changed and plan.needs_metadata
    assert bool(plan.renames) != canonical_only
    with h5py.File(path, 'r') as episode:
        teleop = episode['teleop']
        assert teleop['right_q_xarm'].attrs['source_signal_name'] == 'q_follower'
        assert teleop['right_q_gello'].attrs['source_signal_name'] == 'q_leader'
        assert teleop['right_dq_valid_xarm'].attrs['source_signal_name'] == 'dq_valid_follower'
        assert teleop['right_q_xarm'].attrs['source_timestamp_path'] == 'teleop/right_q_timestamp_us_xarm'
        assert episode['cameras/wrist/frames'].attrs['reference_path'] == 'teleop/right_q_xarm'
        assert list(teleop.attrs['command_datasets']) == ['right_q_cmd_xarm']
    migration.undo_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert object_info(episode) == before
        assert migration._snapshot(episode) == attrs


def create_v1_migrated_episode(path, monkeypatch):
    """Exercise upgrades against the actual previous naming/journal contract."""
    create_episode(path)
    with h5py.File(path, 'r') as episode:
        original_objects = object_info(episode)
        original_attrs = migration._snapshot(episode)
    current_name = public_dataset_name
    def v1_name(arm, signal):
        if signal == 'ee_pose_follower':
            return f'{arm}_ee_pose_xarm'
        return f'{arm}_tau_ext_l1' if signal == 'tau_ext_l1' else current_name(arm, signal)
    with monkeypatch.context() as patch:
        patch.setattr(migration, 'DEVICE_SCHEMA_VERSION', 'gello_xarm/side_quantity_device/v1')
        patch.setattr(migration, 'JOURNAL_VERSION', 1)
        patch.setattr(migration, 'public_dataset_name', v1_name)
        migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r+') as episode:
        journal = migration._load_journal(episode)
        journal.pop('previous_journal_path', None)
        journal.pop('previous_journal_address', None)
        migration._write_journal(episode, journal)
        del episode['teleop'].attrs['schema_arm_names']
        episode['teleop'].attrs['dataset_name_exception'] = (
            '{arm}_tau_ext_l1; shared timing channels are unprefixed')
    return original_objects, original_attrs


def test_v1_right_only_upgrade_preserves_payloads_and_undo_chain(tmp_path, monkeypatch):
    path = tmp_path / 'right_only.h5'
    original_objects, original_attrs = create_v1_migrated_episode(path, monkeypatch)
    with h5py.File(path, 'r') as episode:
        v1_objects = object_info(episode)
        v1_attrs = migration._snapshot(episode)
        previous_journal = episode[migration.JOURNAL_PATH][()]
    file_hash = hashlib.sha256(path.read_bytes()).digest()
    plan = migration.migrate_file(path)
    expected_renames = {'right_tau_ext_l1': 'right_tau_ext_l1_xarm'}
    current_pose = public_dataset_name('right', 'ee_pose_follower')
    if current_pose != 'right_ee_pose_xarm':
        expected_renames['right_ee_pose_xarm'] = current_pose
    assert dict(plan.renames) == expected_renames
    assert plan.previous_journal_path is not None
    assert hashlib.sha256(path.read_bytes()).digest() == file_hash
    migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert not any(name.startswith('left_') for name in episode['teleop'])
        assert tuple(episode['teleop'].attrs['arm_names']) == ('right',)
        assert tuple(episode['teleop'].attrs['schema_arm_names']) == ('right',)
        assert 'required_xarm_datasets' not in episode['teleop'].attrs
        assert episode[plan.previous_journal_path][()] == previous_journal
        after = object_info(episode)
        for name, info in v1_objects.items():
            target = name
            if name == migration.JOURNAL_PATH:
                target = plan.previous_journal_path
            elif name.startswith('teleop/'):
                old = name.removeprefix('teleop/')
                target = 'teleop/' + dict(plan.renames).get(old, old)
            assert after[target] == info
        assert episode['teleop/right_tau_ext_l1_xarm'].attrs['input_path'] == 'teleop/right_tau_ext_xarm'
        assert migration._load_journal(episode)['version'] == 2
    assert not migration.migrate_file(path, apply=True).changed
    assert migration.undo_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert object_info(episode) == v1_objects
        assert migration._snapshot(episode) == v1_attrs
        assert migration._load_journal(episode)['version'] == 1
    assert migration.undo_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert object_info(episode) == original_objects
        assert migration._snapshot(episode) == original_attrs
    assert not migration.undo_file(path, apply=True)


@pytest.mark.parametrize('failure_stage', ['journal', 'attributes'])
def test_failed_v1_upgrade_restores_original_journal_object(tmp_path, monkeypatch, failure_stage):
    path = tmp_path / 'right_only.h5'
    create_v1_migrated_episode(path, monkeypatch)
    with h5py.File(path, 'r') as episode:
        objects = object_info(episode)
        attrs = migration._snapshot(episode)
    target = '_write_journal' if failure_stage == 'journal' else '_rewrite_attributes'
    original = getattr(migration, target)
    def fail(*args):
        original(*args)
        raise RuntimeError('injected v1 upgrade failure')
    monkeypatch.setattr(migration, target, fail)
    with pytest.raises(RuntimeError, match='injected v1 upgrade'):
        migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert object_info(episode) == objects
        assert migration._snapshot(episode) == attrs
        assert migration._load_journal(episode)['version'] == 1


def test_missing_previous_journal_is_rejected_before_undo_changes(tmp_path, monkeypatch):
    path = tmp_path / 'right_only.h5'
    create_v1_migrated_episode(path, monkeypatch)
    plan = migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r+') as episode:
        del episode[plan.previous_journal_path]
    digest = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(ValueError, match='missing previous migration journal'):
        migration.undo_file(path, apply=True)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


def test_v1_norm_collision_leaves_previous_journal_and_file_untouched(tmp_path, monkeypatch):
    path = tmp_path / 'right_only.h5'
    create_v1_migrated_episode(path, monkeypatch)
    with h5py.File(path, 'r+') as episode:
        episode['teleop'].create_dataset('right_tau_ext_l1_xarm', data=np.ones((3, 1)))
    digest = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(ValueError, match='collision'):
        migration.migrate_file(path, apply=True)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


def test_future_required_layout_flags_are_accepted_without_historical_padding(tmp_path):
    path = tmp_path / 'future_writer.h5'
    with h5py.File(path, 'w') as episode:
        teleop = episode.create_group('teleop')
        teleop.attrs['arm_names'] = ('right',)
        teleop.attrs['dataset_layout'] = 'per_arm'
        teleop.attrs['dataset_naming_schema'] = DEVICE_SCHEMA_VERSION
        teleop.create_dataset('timestamp_us', data=np.arange(3, dtype=np.int64))
        for signal in ('q_follower_valid', 'dq_valid_follower', 'torque_valid_follower', 'q_cmd_send_ok'):
            teleop.create_dataset(public_dataset_name('right', signal), data=np.ones((3, 1), np.uint8))
        ensure_required_xarm_datasets(teleop, ('right',))
        for name in ('right_q_available_xarm', 'right_dq_available_xarm',
                     'right_tau_available_xarm', 'right_q_cmd_available_xarm',
                     'right_tau_ext_l1_valid_xarm'):
            assert name in teleop
    digest = hashlib.sha256(path.read_bytes()).digest()
    assert not migration.migrate_file(path).changed
    assert not migration.migrate_file(path, apply=True).changed
    assert hashlib.sha256(path.read_bytes()).digest() == digest


def test_existing_inactive_arm_reference_uses_its_side_without_changing_recorded_arms(tmp_path):
    path = tmp_path / 'existing_inactive_side.h5'
    with h5py.File(path, 'w') as episode:
        teleop = episode.create_group('teleop')
        teleop.attrs['arm_names'] = ('right',)
        teleop.attrs['dataset_layout'] = 'per_arm'
        teleop.attrs['dataset_naming_schema'] = 'gello_xarm/side_quantity_device/v1'
        teleop.create_dataset('timestamp_us', data=np.arange(3, dtype=np.int64))
        right = teleop.create_dataset('right_q_xarm', data=np.ones((3, 7)))
        left = teleop.create_dataset('left_q_xarm', data=np.full((3, 7), np.nan))
        left.attrs['placeholder'] = True
        left.attrs['coordinate_reference_path'] = 'teleop/q_follower'
        right.attrs['other_side_reference_path'] = 'teleop/left_q_follower'
    migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        teleop = episode['teleop']
        assert tuple(teleop.attrs['arm_names']) == ('right',)
        assert teleop['left_q_xarm'].attrs['coordinate_reference_path'] == 'teleop/left_q_xarm'
        assert teleop['right_q_xarm'].attrs['other_side_reference_path'] == 'teleop/left_q_xarm'
        assert set(teleop) == {'timestamp_us', 'left_q_xarm', 'right_q_xarm'}
        assert np.isnan(teleop['left_q_xarm'][:]).all()


@pytest.mark.parametrize('arms', [('left',), ('right',), ('left', 'right')])
@pytest.mark.parametrize('pose_suffix', ['ee_pose_xarm', 'q_eepose_xarm', 'eepose_xarm'])
def test_pose_aliases_keep_matrix_objects_and_rename_references_by_side(tmp_path, arms, pose_suffix):
    path = tmp_path / 'pose_aliases.h5'
    with h5py.File(path, 'w') as episode:
        teleop = episode.create_group('teleop')
        teleop.attrs['arm_names'] = arms
        teleop.create_dataset('timestamp_us', data=np.arange(3, dtype=np.int64))
        for index, arm in enumerate(arms):
            poses = np.repeat(np.eye(4)[None], 3, axis=0)
            poses[:, :3, 3] = np.arange(9).reshape(3, 3) + 20 * index
            pose = teleop.create_dataset(f'{arm}_{pose_suffix}', data=poses, compression='gzip')
            teleop.create_dataset(f'{arm}_eepose_valid_xarm', data=np.array([[1], [0], [1]], np.uint8))
            pose.attrs['validity_path'] = f'teleop/{arm}_eepose_valid_xarm'
            pose.attrs['legacy_reference_path'] = f'teleop/{arm}_ee_pose_xarm'
            pose.attrs['local_reference_path'] = 'teleop/ee_pose_xarm'
            pose.attrs['timestamp_path'] = 'teleop/timestamp_us'
        before = object_info(episode)
        original_attrs = migration._snapshot(episode)
    digest = hashlib.sha256(path.read_bytes()).digest()
    plan = migration.migrate_file(path)
    assert hashlib.sha256(path.read_bytes()).digest() == digest
    migration.migrate_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        after = object_info(episode)
        for name, info in before.items():
            old = name.removeprefix('teleop/')
            assert after['teleop/' + dict(plan.renames).get(old, old)] == info
        for arm in arms:
            target = public_dataset_name(arm, 'ee_pose_follower')
            pose = episode[f'teleop/{target}']
            assert pose.shape == (3, 4, 4)
            assert pose.attrs['source_signal_name'] == 'ee_pose_follower'
            assert pose.attrs['legacy_reference_path'] == f'teleop/{target}'
            assert pose.attrs['local_reference_path'] == f'teleop/{target}'
            assert pose.attrs['validity_path'] == f'teleop/{arm}_eepose_valid_xarm'
            assert pose.attrs['timestamp_path'] == 'teleop/timestamp_us'
        assert set(episode['teleop']) == {
            'timestamp_us', *(public_dataset_name(arm, 'ee_pose_follower') for arm in arms),
            *(f'{arm}_eepose_valid_xarm' for arm in arms),
        }
    assert migration.undo_file(path, apply=True)
    with h5py.File(path, 'r') as episode:
        assert object_info(episode) == before
        assert migration._snapshot(episode) == original_attrs


@pytest.mark.parametrize('problem', ['pose_shape', 'validity_shape', 'collision'])
def test_pose_preflight_rejects_bad_shapes_and_alias_collisions_without_writes(tmp_path, problem):
    path = tmp_path / 'bad_pose.h5'
    with h5py.File(path, 'w') as episode:
        teleop = episode.create_group('teleop')
        teleop.attrs['arm_names'] = ('right',)
        teleop.create_dataset('timestamp_us', data=np.arange(3, dtype=np.int64))
        teleop.create_dataset('right_eepose_xarm', data=np.zeros((3, 7) if problem == 'pose_shape' else (3, 4, 4)))
        teleop.create_dataset('right_eepose_valid_xarm', data=np.zeros((3, 2) if problem == 'validity_shape' else (3, 1)))
        if problem == 'collision':
            teleop.create_dataset('right_q_eepose_xarm', data=np.ones((3, 4, 4)))
    digest = hashlib.sha256(path.read_bytes()).digest()
    expected = {'pose_shape': 'arm pose', 'validity_shape': 'scalar per arm', 'collision': 'collision'}[problem]
    with pytest.raises(ValueError, match=expected):
        migration.migrate_file(path, apply=True)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


def test_future_asymmetric_pose_names_and_false_inactive_validity_are_noop(tmp_path):
    path = tmp_path / 'future_pose.h5'
    with h5py.File(path, 'w') as episode:
        teleop = episode.create_group('teleop')
        teleop.attrs['arm_names'] = ('right',)
        teleop.attrs['schema_arm_names'] = ('left', 'right')
        teleop.attrs['dataset_layout'] = 'per_arm'
        teleop.attrs['dataset_naming_schema'] = DEVICE_SCHEMA_VERSION
        teleop.create_dataset('timestamp_us', data=np.arange(3, dtype=np.int64))
        teleop.create_dataset('left_q_eepose_xarm', data=np.full((3, 4, 4), np.nan))
        teleop.create_dataset('left_eepose_valid_xarm', data=np.zeros((3, 1), np.uint8))
        teleop.create_dataset('right_eepose_xarm', data=np.repeat(np.eye(4)[None], 3, axis=0))
        teleop.create_dataset('right_eepose_valid_xarm', data=np.ones((3, 1), np.uint8))
    digest = hashlib.sha256(path.read_bytes()).digest()
    assert not migration.migrate_file(path).changed
    assert not migration.migrate_file(path, apply=True).changed
    assert hashlib.sha256(path.read_bytes()).digest() == digest
