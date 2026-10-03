#!/usr/bin/env python3
"""Rename completed GELLO/xArm episode channels without copying payloads.

The default is a read-only dry run. Use --in-place to apply the reviewed plan;
use --in-place --undo to restore the names and attributes from its journal.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nero_collection.h5_schema import DEVICE_SCHEMA_VERSION, SHARED_DATASETS, public_dataset_name
from nero_collection.h5_writer import FOLLOWER_TELEOP_DATASETS, _PER_ARM_DATASETS


JOURNAL_PATH = 'metadata/migration_json'
JOURNAL_VERSION = 2
_SUPPORTED_JOURNAL_VERSIONS = frozenset({1, JOURNAL_VERSION})
_PREVIOUS_JOURNAL_PREFIX = 'metadata/migration_previous_'
KNOWN_SIGNALS = FOLLOWER_TELEOP_DATASETS | frozenset({'wrench_ext', 'wrench_cal', 'wrench_pred'})
POSE_CANONICAL_ALIASES = ('ee_pose_xarm', 'q_eepose_xarm', 'eepose_xarm')
CANONICAL_SIGNALS = {
    public_dataset_name('', signal).removeprefix('_'): signal
    for signal in sorted(KNOWN_SIGNALS - SHARED_DATASETS)
}
# Pose basenames depend on the side, so an empty-arm name alone cannot define
# the complete set. All spellings retain the same matrix signal semantics.
CANONICAL_SIGNALS.update({name: 'ee_pose_follower' for name in POSE_CANONICAL_ALIASES})


@dataclass(frozen=True)
class MigrationPlan:
    path: Path
    arm_names: tuple[str, ...]
    renames: tuple[tuple[str, str], ...]
    needs_metadata: bool
    previous_journal_path: str | None = None

    @property
    def changed(self):
        return bool(self.renames or self.needs_metadata)

    def as_dict(self):
        return {'path': str(self.path), 'arm_names': list(self.arm_names),
                'renames': dict(self.renames), 'needs_metadata': self.needs_metadata,
                'previous_journal_path': self.previous_journal_path,
                'changed': self.changed}


def _text(value):
    return value.decode('utf-8') if isinstance(value, (bytes, np.bytes_)) else str(value)


def _arm_names(episode):
    values = episode['teleop'].attrs.get('arm_names', ())
    if isinstance(values, (str, bytes, np.str_, np.bytes_)):
        values = (values,)
    if not len(values) and 'metadata/arm_names_json' in episode:
        values = json.loads(_text(episode['metadata/arm_names_json'][()]))
        if isinstance(values, str):
            values = (values,)
    if not len(values):
        values = [arm for arm in ('left', 'right')
                  if any(name.startswith(f'{arm}_') for name in episode['teleop'])]
    names = tuple(_text(value) for value in values)
    if not names or len(set(names)) != len(names) or any(arm not in {'left', 'right'} for arm in names):
        raise ValueError(f'Cannot determine unique left/right arm_names: {names}')
    return names


def _signal_name(name, arms):
    if name in SHARED_DATASETS:
        return None, name
    # A modern single-arm episode can contain explicit inactive-arm fields;
    # recorded arm_names still describes only the hardware that was recorded.
    for arm in dict.fromkeys((*arms, 'left', 'right')):
        prefix = f'{arm}_'
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix in KNOWN_SIGNALS:
                return arm, suffix
            if suffix in CANONICAL_SIGNALS:
                return arm, CANONICAL_SIGNALS[suffix]
            if suffix in {'tau_ext_l1_valid_xarm', 'q_available_xarm',
                          'dq_available_xarm', 'tau_available_xarm',
                          'q_cmd_available_xarm', 'eepose_valid_xarm'}:
                return arm, suffix
            raise ValueError(f'Unknown arm signal {name!r}; no automatic renaming is safe')
    if name in KNOWN_SIGNALS:
        if len(arms) != 1:
            raise ValueError(f'Legacy concatenated dual-arm channel {name!r} requires splitting; '
                             'this metadata-only migration supports already split channels')
        return arms[0], name
    raise ValueError(f'Unknown teleop signal {name!r}; no automatic renaming is safe')


def _validate_shape(dataset, signal, count, arm):
    if dataset.ndim < 1 or dataset.shape[0] != count:
        raise ValueError(f'{dataset.name} shape {dataset.shape} does not share the {count}-sample timeline')
    if arm is None or signal is None:
        return
    if signal in _PER_ARM_DATASETS or signal.endswith(('_available_xarm', '_valid_xarm')):
        if dataset.shape not in {(count,), (count, 1)}:
            raise ValueError(f'{dataset.name} must be scalar per arm, got {dataset.shape}')
    elif signal == 'ee_pose_follower':
        if dataset.shape != (count, 4, 4):
            raise ValueError(f'{dataset.name} must be an arm pose (N,4,4), got {dataset.shape}')
    elif signal.startswith('wrench_'):
        if dataset.shape != (count, 6):
            raise ValueError(f'{dataset.name} must be a six-component wrench, got {dataset.shape}')
    elif (dataset.ndim != 2 or dataset.shape[1] not in {5, 6, 7, 8}
          or (dataset.shape[1] == 8 and signal != 'q_leader_raw')):
        raise ValueError(f'{dataset.name} must be one arm joint vector, got {dataset.shape}')


def _plan(episode, path):
    if 'teleop' not in episode or 'timestamp_us' not in episode['teleop']:
        raise ValueError(f'{path.name} has no teleop/timestamp_us')
    teleop = episode['teleop']
    arms = _arm_names(episode)
    timeline = teleop['timestamp_us']
    if not isinstance(timeline, h5py.Dataset) or timeline.ndim != 1:
        raise ValueError('teleop/timestamp_us must be a one-dimensional timeline')
    count = timeline.shape[0]
    targets = {}
    renames = []
    for name, dataset in teleop.items():
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f'Unexpected teleop subgroup {name!r}')
        arm, signal = _signal_name(name, arms)
        _validate_shape(dataset, signal, count, arm)
        target = name if arm is None or signal is None else public_dataset_name(arm, signal)
        if target in targets:
            raise ValueError(f'Target collision: {targets[target]!r} and {name!r} both map to {target!r}')
        targets[target] = name
        if target != name:
            renames.append((name, target))
    # Includes an existing new-name target, even if the old alias sorts first.
    for source, target in renames:
        if target in teleop and target != source:
            raise ValueError(f'Target collision: {target!r} already exists; refusing to overwrite {source!r}')
    needs_metadata = (teleop.attrs.get('dataset_naming_schema') != DEVICE_SCHEMA_VERSION
                      or teleop.attrs.get('dataset_layout') != 'per_arm')
    previous_journal_path = None
    if JOURNAL_PATH in episode:
        journal = _load_journal(episode)
        if journal.get('status') != 'complete':
            raise ValueError('An incomplete migration journal exists; run --in-place --undo first')
        if renames or needs_metadata:
            # Keep the earlier journal as the same HDF5 object. Each undo
            # restores one revision, including its original journal.
            index = 1
            while f'{_PREVIOUS_JOURNAL_PREFIX}{index:04d}_json' in episode:
                index += 1
            previous_journal_path = f'{_PREVIOUS_JOURNAL_PREFIX}{index:04d}_json'
    return MigrationPlan(path, arms, tuple(renames), needs_metadata, previous_journal_path)


def _pack_atom(value):
    if isinstance(value, (bytes, np.bytes_)):
        return {'bytes': base64.b64encode(bytes(value)).decode('ascii')}
    if isinstance(value, (str, np.str_)):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (complex, np.complexfloating)):
        return {'complex': [float(value.real), float(value.imag)]}
    raise ValueError(f'Unsupported attribute value type {type(value).__name__}; no changes made')


def _unpack_atom(value):
    if isinstance(value, dict) and 'bytes' in value:
        return base64.b64decode(value['bytes'])
    if isinstance(value, dict) and 'complex' in value:
        return complex(*value['complex'])
    return value


def _pack_attribute(obj, key):
    dtype = obj.attrs.get_id(key).dtype
    if dtype.fields or dtype.subdtype:
        raise ValueError(f'{obj.name} attribute {key!r} has unsupported structured dtype')
    value = np.asarray(obj.attrs[key])
    string = h5py.check_string_dtype(dtype)
    enum = h5py.check_enum_dtype(dtype)
    unsupported_metadata = set(dtype.metadata or ()) - {'vlen', 'h5py_encoding', 'enum'}
    if unsupported_metadata:
        raise ValueError(f'{obj.name} attribute {key!r} has unsupported dtype metadata')
    return {'dtype': dtype.str, 'shape': list(value.shape),
            'encoding': string.encoding if string else None,
            'string_length': string.length if string else None,
            'enum': enum,
            'items': [_pack_atom(item) for item in value.reshape(-1)]}


def _unpack_attribute(packed):
    dtype = (h5py.string_dtype(packed['encoding'], packed.get('string_length'))
             if packed['encoding'] else np.dtype(packed['dtype']))
    if packed.get('enum') is not None:
        dtype = h5py.enum_dtype(packed['enum'], basetype=dtype)
    values = np.asarray([_unpack_atom(item) for item in packed['items']], dtype=dtype)
    return values.reshape(tuple(packed['shape'])), dtype


def _objects(episode):
    objects = {'/': episode}
    episode.visititems(lambda name, obj: objects.__setitem__(f'/{name}', obj))
    return objects


def _snapshot(episode):
    return {path: {key: _pack_attribute(obj, key) for key in obj.attrs}
            for path, obj in _objects(episode).items()}


def _replace_string(value, references, names, attribute):
    was_bytes = isinstance(value, (bytes, np.bytes_))
    if not isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return value
    text = _text(value)
    if references:
        pattern = '|'.join(re.escape(path) for path in sorted(references, key=len, reverse=True))
        text = re.sub(r'(?<![A-Za-z0-9_])(' + pattern + r')(?![A-Za-z0-9_])',
                      lambda match: references[match.group(1)], text)
    if attribute == 'command_datasets':
        text = names.get(text, text)
    return text.encode('utf-8') if was_bytes else text


def _rewrite_attributes(episode, plan):
    names = dict(plan.renames)
    references = {f'teleop/{old}': f'teleop/{new}' for old, new in plan.renames}
    # Canonical links may already exist while their metadata still uses role
    # names (for example after an earlier manual rename).
    schema_arms = tuple(dict.fromkeys((*plan.arm_names, 'left', 'right')))
    for arm in schema_arms:
        for signal in KNOWN_SIGNALS - SHARED_DATASETS:
            output = public_dataset_name(arm, signal)
            if output not in episode['teleop']:
                continue
            signal_aliases = ((signal, *POSE_CANONICAL_ALIASES)
                              if signal == 'ee_pose_follower' else (signal,))
            aliases = [f'{arm}_{alias}' for alias in signal_aliases]
            if len(plan.arm_names) == 1 and arm == plan.arm_names[0]:
                aliases.extend(signal_aliases)
            for alias in aliases:
                names[alias] = output
                references[f'teleop/{alias}'] = f'teleop/{output}'
    objects = _objects(episode)
    for path, obj in objects.items():
        if path == f'/{JOURNAL_PATH}' or path.startswith(f'/{_PREVIOUS_JOURNAL_PREFIX}'):
            continue
        arm = next((name for name in schema_arms
                    if path.startswith(f'/teleop/{name}_')), None)
        local_references = dict(references)
        if arm is not None:
            for signal in KNOWN_SIGNALS - SHARED_DATASETS:
                output = public_dataset_name(arm, signal)
                if output in episode['teleop']:
                    local_references[f'teleop/{signal}'] = f'teleop/{output}'
                    if signal == 'ee_pose_follower':
                        for alias in POSE_CANONICAL_ALIASES:
                            local_references[f'teleop/{alias}'] = f'teleop/{output}'
        for key in list(obj.attrs):
            value = obj.attrs[key]
            if isinstance(value, np.ndarray):
                changed = [_replace_string(item, local_references, names, key)
                           for item in value.reshape(-1)]
                if all(new == old for new, old in zip(changed, value.reshape(-1))):
                    continue
                # H5 fixed-length strings need widening after longer renames.
                if any(isinstance(item, (str, bytes, np.str_, np.bytes_)) for item in changed):
                    changed = np.asarray([_text(item) for item in changed], dtype=object).reshape(value.shape)
                else:
                    continue
            else:
                changed = _replace_string(value, local_references, names, key)
                if changed == value:
                    continue
            del obj.attrs[key]
            if isinstance(changed, np.ndarray) and changed.dtype.kind == 'O':
                obj.attrs.create(key, changed, dtype=h5py.string_dtype('utf-8'))
            else:
                obj.attrs[key] = changed


def _write_journal(episode, journal):
    episode.require_group('metadata')
    if JOURNAL_PATH in episode:
        del episode[JOURNAL_PATH]
    episode.create_dataset(JOURNAL_PATH, data=json.dumps(journal), dtype=h5py.string_dtype('utf-8'))
    episode.flush()


def _load_journal(episode):
    journal = json.loads(_text(episode[JOURNAL_PATH][()]))
    if journal.get('version') not in _SUPPORTED_JOURNAL_VERSIONS:
        raise ValueError('Unsupported migration journal version')
    return journal


def _validate_restore(episode, journal):
    """Resolve original objects and decode all attrs before changing any links."""
    teleop = episode['teleop']
    moved_paths = {}
    previous_journal_path = journal.get('previous_journal_path')
    if previous_journal_path is not None:
        if previous_journal_path in episode:
            if h5py.h5o.get_info(episode[previous_journal_path].id).addr != journal.get('previous_journal_address'):
                raise ValueError('Cannot restore replaced previous migration journal')
            moved_paths[f'/{JOURNAL_PATH}'] = f'/{previous_journal_path}'
        elif (JOURNAL_PATH not in episode
              or h5py.h5o.get_info(episode[JOURNAL_PATH].id).addr
              != journal.get('previous_journal_address')):
            raise ValueError('Cannot restore missing previous migration journal')
    for old, new in journal['renames']:
        if old in teleop and new in teleop:
            raise ValueError(f'Cannot undo: both {old!r} and {new!r} exist')
        if old not in teleop and new not in teleop:
            raise ValueError(f'Cannot undo: neither {old!r} nor {new!r} exists')
        if new in teleop:
            moved_paths[f'/teleop/{old}'] = f'/teleop/{new}'
    decoded = {}
    for path, attrs in journal['original_attributes'].items():
        current_path = moved_paths.get(path, path)
        if current_path not in episode:
            raise ValueError(f'Cannot restore missing H5 object {path}')
        try:
            decoded[path] = {key: _unpack_attribute(packed) for key, packed in attrs.items()}
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f'Cannot decode original attributes for {path}: {exc}') from exc
    return decoded


def _restore(episode, journal):
    decoded = _validate_restore(episode, journal)
    teleop = episode['teleop']
    for old, new in reversed(journal['renames']):
        if new in teleop:
            teleop.move(new, old)
    previous_journal_path = journal.get('previous_journal_path')
    if previous_journal_path is not None and previous_journal_path in episode:
        if JOURNAL_PATH in episode:
            del episode[JOURNAL_PATH]
        episode.move(previous_journal_path, JOURNAL_PATH)
    for path, attrs in decoded.items():
        obj = episode[path]
        for key in list(obj.attrs):
            del obj.attrs[key]
        for key, (value, dtype) in attrs.items():
            obj.attrs.create(key, value, dtype=dtype)
    if previous_journal_path is None and JOURNAL_PATH in episode:
        del episode[JOURNAL_PATH]
    if journal['created_metadata'] and 'metadata' in episode and not len(episode['metadata']):
        del episode['metadata']
    episode.flush()


def migrate_file(path, *, apply=False):
    """Preflight one file; optionally rename its links and journal every change."""
    path = Path(path).expanduser().resolve()
    if path.suffix != '.h5' or path.name.endswith('.tmp'):
        raise ValueError(f'Only completed .h5 episodes can be migrated: {path}')
    with h5py.File(path, 'r+' if apply else 'r') as episode:
        plan = _plan(episode, path)
        if not plan.changed:
            return plan
        # Dry-run preflight also validates every attr, so a batch cannot reject
        # a later file's unsupported metadata after changing earlier files.
        original_attributes = _snapshot(episode)
        if not apply:
            return plan
        journal = {'version': JOURNAL_VERSION, 'status': 'pending',
                   'schema': DEVICE_SCHEMA_VERSION, 'renames': list(plan.renames),
                   'arm_names': list(plan.arm_names), 'created_metadata': 'metadata' not in episode,
                   'original_attributes': original_attributes,
                   'previous_journal_path': plan.previous_journal_path,
                   'previous_journal_address': (h5py.h5o.get_info(episode[JOURNAL_PATH].id).addr
                                                if plan.previous_journal_path is not None else None)}
        try:
            if plan.previous_journal_path is not None:
                episode.move(JOURNAL_PATH, plan.previous_journal_path)
            _write_journal(episode, journal)
            teleop = episode['teleop']
            for old, new in plan.renames:
                teleop.move(old, new)
            _rewrite_attributes(episode, plan)
            original_names = {new: old for old, new in plan.renames}
            for new, dataset in teleop.items():
                arm, signal = _signal_name(original_names.get(new, new), plan.arm_names)
                if arm is None:
                    continue
                if new not in original_names and 'source_signal_name' in dataset.attrs:
                    original_signal = _text(dataset.attrs['source_signal_name'])
                    if (original_signal in KNOWN_SIGNALS
                            and public_dataset_name(arm, original_signal) == new):
                        signal = original_signal
                dataset.attrs['arm_name'] = arm
                dataset.attrs['source_signal_name'] = signal
                dataset.attrs['device_name'] = 'gello' if new.endswith('_gello') else 'xarm'
                dataset.attrs['joint_layout'] = ('scalar L1 reduction over J1..J7 per arm'
                                                  if signal == 'tau_ext_l1' else 'J1..Jn per arm')
                if signal == 'delta_q':
                    dataset.attrs['definition'] = (
                        f'{public_dataset_name(arm, "q_cmd")} - '
                        f'{public_dataset_name(arm, "q_follower")} at the same state sample')
            teleop.attrs['dataset_layout'] = 'per_arm'
            teleop.attrs['dataset_naming_schema'] = DEVICE_SCHEMA_VERSION
            teleop.attrs['dataset_name_pattern'] = '{arm}_{quantity}_{device}'
            teleop.attrs['dataset_name_exception'] = 'shared timing channels are unprefixed'
            schema_arms = tuple(arm for arm in ('left', 'right')
                                if any(name.startswith(f'{arm}_') for name in teleop))
            teleop.attrs['schema_arm_names'] = np.asarray(schema_arms, dtype=h5py.string_dtype('utf-8'))
            teleop.attrs['joint_layout'] = 'separate arm-prefixed datasets; J1..Jn per arm'
            teleop.attrs['pose_layout'] = 'each arm pose: (N,4,4)'
            journal['status'] = 'complete'
            _write_journal(episode, journal)
        except BaseException:
            _restore(episode, journal)
            raise
    return plan


def undo_file(path, *, apply=False):
    """Restore journaled names/attrs; defaults to reporting that an undo exists."""
    path = Path(path).expanduser().resolve()
    with h5py.File(path, 'r+' if apply else 'r') as episode:
        if JOURNAL_PATH not in episode:
            return False
        journal = _load_journal(episode)
        _validate_restore(episode, journal)
        if apply:
            _restore(episode, journal)
        return True


def episode_paths(inputs):
    paths = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            paths.update(item for item in path.glob('*.h5') if item.is_file())
        elif path.suffix == '.h5' and path.is_file():
            paths.add(path)
        else:
            raise ValueError(f'Not a completed H5 episode or directory: {path}')
    return tuple(sorted(paths))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--in-place', action='store_true', help='apply the metadata-only migration')
    parser.add_argument('--undo', action='store_true', help='restore journaled names and attributes')
    args = parser.parse_args(argv)
    try:
        paths = episode_paths(args.paths)
        if args.undo:
            for path in paths:
                found = undo_file(path, apply=False)
                print(json.dumps({'path': str(path), 'undo_available': found, 'apply': args.in_place}))
            if args.in_place:
                for path in paths:
                    undo_file(path, apply=True)
        else:
            # Preflight the complete batch before any file changes.
            for path in paths:
                print(json.dumps(migrate_file(path).as_dict(), ensure_ascii=False))
            if args.in_place:
                for path in paths:
                    migrate_file(path, apply=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
