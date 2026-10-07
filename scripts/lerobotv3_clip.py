"""Transactional LeRobot v3 index clipping; shared video payloads stay unchanged."""
from __future__ import annotations

from collections import Counter
import copy
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HISTORY = 'meta/clip_history.json'
QUANTILES = (.01, .1, .5, .9, .99)


def output_dataset(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output must be separate from the source (not equal, inside, or a parent)')
    if output.exists():
        marker = output/HISTORY
        if not marker.is_file() or json.loads(marker.read_text()).get('source_root') != str(source):
            raise ValueError(f'Refusing to replace unrelated output: {output}')
        return output
    return source


def _numeric_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if not np.isfinite(values).all():
        raise ValueError('Cannot recompute statistics for nonfinite retained data')
    result = dict(min=values.min(0), max=values.max(0), mean=values.mean(0), std=values.std(0),
                  count=np.array([len(values)]))
    result.update({f'q{int(q*100):02d}': np.quantile(values, q, axis=0) for q in QUANTILES})
    return {key: value.tolist() for key, value in result.items()}


def _image_stats(dataset, episode, key, times):
    from scripts.rerun_lerobotv3 import VideoReader

    # Same frame sampling scale as LeRobot; accumulate bounded RGB histograms.
    count = max(min(100, len(times)), min(int(len(times)**.75), 10000))
    indices = np.round(np.linspace(0, len(times)-1, count)).astype(int)
    histogram = np.zeros((3, 256), dtype=np.int64)
    path, offset = dataset.video_location(episode, key)
    reader = VideoReader(path, dataset.fps)
    try:
        for index in indices:
            image = reader.read(offset+float(times[index]))
            for channel in range(3):
                histogram[channel] += np.bincount(image[..., channel].ravel(), minlength=256)
    finally:
        reader.close()
    pixels = histogram.sum(1)
    bins = np.arange(256)/255.
    mean = (histogram*bins).sum(1)/pixels
    variance = (histogram*(bins[None, :]-mean[:, None])**2).sum(1)/pixels
    cdf = np.cumsum(histogram, axis=1)
    result = dict(min=np.argmax(histogram > 0, axis=1)/255.,
                  max=(255-np.argmax(histogram[:, ::-1] > 0, axis=1))/255.,
                  mean=mean, std=np.sqrt(variance))
    for q in QUANTILES:
        result[f'q{int(q*100):02d}'] = np.array([np.searchsorted(c, q*n) for c, n in zip(cdf, pixels)])/255.
    return {**{key: value.reshape(3, 1, 1).tolist() for key, value in result.items()}, 'count': [count]}


def _aggregate(rows):
    features = {}
    for row in rows:
        episode = {}
        for key, value in row.items():
            if key.startswith('stats/') and value is not None:
                name, stat = key[6:].rsplit('/', 1)
                episode.setdefault(name, {})[stat] = np.asarray(value, dtype=float)
        for name, stats in episode.items():
            features.setdefault(name, []).append(stats)
    result = {}
    for name, stats in features.items():
        counts = np.array([s['count'][0] for s in stats])
        total = counts.sum()
        weights = counts.reshape((-1,)+(1,)*stats[0]['mean'].ndim)
        means = np.stack([s['mean'] for s in stats])
        mean = (means*weights).sum(0)/total
        variance = ((np.stack([s['std']**2 for s in stats])+(means-mean)**2)*weights).sum(0)/total
        values = dict(min=np.min([s['min'] for s in stats], axis=0),
                      max=np.max([s['max'] for s in stats], axis=0), mean=mean,
                      std=np.sqrt(variance), count=np.array([int(total)]))
        # LeRobot aggregates per-episode quantiles by sample count as well.
        for key in stats[0]:
            if key.startswith('q') and key[1:].isdigit() and all(key in s for s in stats):
                values[key] = (np.stack([s[key] for s in stats])*weights).sum(0)/total
        result[name] = {key: value.tolist() for key, value in values.items()}
    return result


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _json(path, value):
    # Staged files may be hardlinked to the source; always replace, never mutate.
    path.unlink(missing_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def _publish(stage, output):
    if not output.exists():
        os.rename(stage, output)
        return
    # Linux atomically swaps nonempty directories: a failed or interrupted
    # publication cannot expose a mixture of old data and new metadata.
    libc = ctypes.CDLL(None, use_errno=True)
    exchange = getattr(libc, 'renameat2', None)
    if exchange is None:
        raise RuntimeError('Atomic update of an existing output requires Linux renameat2')
    exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    exchange.restype = ctypes.c_int
    if exchange(-100, os.fsencode(stage), -100, os.fsencode(output), 2) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def save_clip(dataset, episode_index, keep, output, *, source_root, output_fn=print):
    """Materialize a complete cleaned dataset, retaining unreviewed episodes too."""
    output, source_root = Path(output).resolve(), Path(source_root).resolve()
    working = output_dataset(source_root, output)
    if working != dataset.root:
        raise RuntimeError('Dataset selection is stale; reopen the browser')
    episode = dataset.episodes[episode_index]
    if type(keep) is not int or not 1 <= keep < episode['length']:
        raise ValueError('Cut must retain at least one frame and remove at least one frame')
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent/('.'+output.name+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another clip save is in progress') from exc
        _save_locked(dataset, episode_index, keep, output, source_root, output_fn)


def _save_locked(dataset, index, keep, output, source_root, output_fn):
    before = dataset.revision()
    if before != dataset.loaded_revision:
        raise RuntimeError('Dataset changed since opening; reopen the browser')
    table = dataset.read_episode(index).slice(0, keep)
    rows_by_path = copy.deepcopy(dataset.metadata_rows)
    episodes = {int(row['episode_index']): row for rows in rows_by_path.values() for row in rows}
    starts, cursor = {}, 0
    for i in sorted(episodes):
        row = episodes[i]
        old_start = int(row['dataset_from_index'])
        row['length'] = keep if i == index else row['length']
        starts[i] = cursor
        row['dataset_from_index'], row['dataset_to_index'] = cursor, cursor+row['length']
        shift = cursor-old_start
        for key, value in list(row.items()):
            if key.startswith('stats/index/') and key.rsplit('/', 1)[1] not in ('std', 'count') and value is not None:
                row[key] = (np.asarray(value)+shift).tolist()
        cursor += row['length']
    selected = episodes[index]
    times = np.asarray(table['timestamp'], dtype=float)
    for key, spec in dataset.info['features'].items():
        if spec['dtype'] == 'video':
            output_fn(f'  Recomputing retained image statistics: {key}')
            stats = _image_stats(dataset, dataset.episodes[index], key, times)
            prefix = f'videos/{key}/'
            selected[prefix+'to_timestamp'] = min(float(selected[prefix+'to_timestamp']),
                                                  float(selected[prefix+'from_timestamp'])+float(times[-1])+1/dataset.fps)
        elif spec['dtype'] not in ('image', 'string') and key in table.column_names:
            values = np.arange(starts[index], starts[index]+keep) if key == 'index' else table[key].to_pylist()
            stats = _numeric_stats(values)
        elif spec['dtype'] == 'image':
            raise ValueError('This clipper supports v3 video features; embedded images are not supported')
        else:
            continue
        selected.update({f'stats/{key}/{stat}': value for stat, value in stats.items()})
    stage = Path(tempfile.mkdtemp(prefix='.'+output.name+'.staging-', dir=output.parent))
    try:
        shutil.copytree(dataset.root, stage, dirs_exist_ok=True, copy_function=_link_or_copy,
                        ignore=shutil.ignore_patterns('.cache', '.git', '*.lock', '*.tmp'))
        seen = Counter()
        for path in sorted((dataset.root/'data').rglob('*.parquet')):
            file = pq.ParquetFile(path)
            target = stage/path.relative_to(dataset.root)
            target.unlink()
            with pq.ParquetWriter(target, file.schema_arrow) as writer:
                for batch in file.iter_batches(batch_size=4096):
                    data = pa.Table.from_batches([batch])
                    ids = np.asarray(data['episode_index'])
                    frame = np.asarray(data['frame_index'])
                    mask = (ids != index) | (frame < keep)
                    data = data.filter(mask)
                    ids, frame = ids[mask], frame[mask]
                    if any(int(i) not in starts for i in ids):
                        raise ValueError('Data contains an episode missing from metadata')
                    global_index = np.array([starts[int(i)]+int(f) for i, f in zip(ids, frame)], dtype=np.int64)
                    field = data.schema.field('index')
                    data = data.set_column(data.schema.get_field_index('index'), field, pa.array(global_index, type=field.type))
                    writer.write_table(data)
                    seen.update(map(int, ids))
        if dict(seen) != {i: row['length'] for i, row in episodes.items()}:
            raise ValueError('Clipped rows do not match episode metadata lengths')
        for relative, rows in rows_by_path.items():
            schema = dataset.metadata_schemas[relative]
            result = pa.Table.from_pylist(rows, schema=schema)
            for extra in sorted(set().union(*(set(row) for row in rows))-set(schema.names)):
                result = result.append_column(extra, pa.array([row.get(extra) for row in rows]))
            target = stage/relative
            target.unlink()
            pq.write_table(result, target)
        info = copy.deepcopy(dataset.info)
        info['total_frames'] = cursor
        _json(stage/'meta/info.json', info)
        _json(stage/'meta/stats.json', _aggregate(list(episodes.values())))
        previous = dataset.root/HISTORY
        history = json.loads(previous.read_text()) if previous.exists() else dict(source_root=str(source_root), clips=[])
        history['source_root'] = str(source_root)
        history['clips'].append(dict(episode_index=index, before=dataset.episodes[index]['length'], after=keep,
                                    last_frame_index=keep-1, last_timestamp=float(times[-1]),
                                    saved_at=datetime.now(timezone.utc).isoformat()))
        history['video_storage'] = 'Original encoded MP4s; removed tails are unreferenced by data rows/video ranges.'
        _json(stage/HISTORY, history)
        # Flush rewritten metadata/data before publication. Unmodified videos
        # are hardlinked (or copied across filesystems), never decoded wholesale.
        for parent in ('data', 'meta'):
            for path in (stage/parent).rglob('*'):
                if path.is_file():
                    with path.open('rb') as handle:
                        os.fsync(handle.fileno())
        if dataset.revision() != before:
            raise RuntimeError('Source changed during save; reopen the browser')
        _publish(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
