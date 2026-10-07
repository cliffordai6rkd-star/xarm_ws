#!/usr/bin/env python3
"""Stream one LeRobot v3 episode to Rerun; enter a timeline time to clip its tail."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class V3Dataset:
    """Only episode metadata is loaded upfront, never the dataset's RGB videos."""
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.info = json.loads((self.root/'meta/info.json').read_text())
        if self.info.get('codebase_version') != 'v3.0':
            raise ValueError('Expected a LeRobot v3.0 dataset root containing meta/info.json')
        self.fps = float(self.info['fps'])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Dataset fps must be positive')
        self.video_keys = [k for k, v in self.info['features'].items() if v['dtype'] == 'video']
        self.metadata_files = sorted((self.root/'meta/episodes').rglob('*.parquet'))
        self.episodes, self.metadata_rows, self.metadata_schemas = {}, {}, {}
        for path in self.metadata_files:
            table = pq.read_table(path)
            rows = table.to_pylist()
            self.metadata_rows[path.relative_to(self.root)] = rows
            self.metadata_schemas[path.relative_to(self.root)] = table.schema
            for row in rows:
                index = int(row['episode_index'])
                if index in self.episodes:
                    raise ValueError(f'Duplicate episode {index}')
                if row['length'] < 1 or row['dataset_to_index']-row['dataset_from_index'] != row['length']:
                    raise ValueError(f'Invalid episode length/index for {index}')
                self.episodes[index] = row
        if not self.episodes:
            raise ValueError('No v3 episode metadata found')
        self.loaded_revision = self.revision()

    def revision(self):
        paths = [self.root/'meta/info.json', *self.metadata_files,
                 *sorted((self.root/'data').rglob('*.parquet'))]
        paths += [p for p in (self.root/'meta/stats.json', self.root/'meta/clip_history.json') if p.exists()]
        return [(str(p), p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]

    def location(self, template, **kwargs):
        path = (self.root/template.format(**kwargs)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Dataset path escapes its root')
        return path

    def data_path(self, episode):
        return self.location(self.info['data_path'], chunk_index=episode['data/chunk_index'],
                             file_index=episode['data/file_index'])

    def video_location(self, episode, key):
        prefix = f'videos/{key}/'
        return (self.location(self.info['video_path'], video_key=key,
                              chunk_index=episode[prefix+'chunk_index'], file_index=episode[prefix+'file_index']),
                float(episode[prefix+'from_timestamp']))

    def read_episode(self, index):
        episode = self.episodes[index]
        table = pq.read_table(self.data_path(episode), filters=[('episode_index', '=', index)])
        if len(table) != episode['length']:
            raise ValueError(f'Episode {index}: metadata/data length mismatch')
        indices = np.asarray(table['frame_index'])
        times = np.asarray(table['timestamp'], dtype=float)
        if not np.array_equal(indices, np.arange(len(table))):
            raise ValueError('Episode frame_index must be contiguous and start at zero')
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Episode timestamps must increase')
        # These datasets store all three images and low-dimensional fields on
        # one frame clock. Do not independently retime individual sensors.
        return table


class VideoReader:
    """Sequential PyAV decoder with at most two decoded frames, plus keyframe seek."""
    def __init__(self, path, fps):
        try:
            import av
        except ImportError as exc:
            raise RuntimeError('PyAV is required; use the local pinn environment (conda activate pinn)') from exc
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = 2
        self.tolerance = .51/float(fps)+1e-4
        self.iterator = None
        self.current = self.previous = None

    @staticmethod
    def time(frame):
        return float(frame.pts*frame.time_base)

    def read(self, timestamp):
        if self.current is None or timestamp < self.time(self.current)-self.tolerance or timestamp-self.time(self.current) > 1.:
            pts = max(0, int(timestamp/float(self.stream.time_base)))
            self.container.seek(pts, stream=self.stream, backward=True)
            self.iterator = self.container.decode(self.stream)
            self.current = self.previous = None
        while self.current is None or self.time(self.current) < timestamp-1e-5:
            try:
                frame = next(self.iterator)
            except StopIteration:
                break
            if frame.pts is None:
                continue
            self.previous, self.current = self.current, frame
        choices = [f for f in (self.previous, self.current) if f is not None]
        if not choices:
            raise ValueError(f'No decoded image at video time {timestamp:.6f}')
        chosen = min(choices, key=lambda f: abs(self.time(f)-timestamp))
        if abs(self.time(chosen)-timestamp) > self.tolerance:
            raise ValueError(f'Video/frame time mismatch at {timestamp:.6f}: nearest={self.time(chosen):.6f}')
        return chosen.to_ndarray(format='rgb24')

    def close(self):
        self.container.close()


class EpisodePlayer:
    def __init__(self, dataset, index, camera=None):
        self.dataset, self.index = dataset, index
        self.table = dataset.read_episode(index)
        self.length = len(self.table)
        self.times = np.asarray(self.table['timestamp'], dtype=float)
        aliases = {key.removeprefix('observation.images.'): key for key in dataset.video_keys}
        self.camera_keys = dataset.video_keys if not camera else [aliases.get(k, k) for k in camera.split('/')]
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys) or set(self.camera_keys)-set(dataset.video_keys):
            raise ValueError(f'Invalid camera selection; available={list(aliases)}')
        self.readers = {}
        try:
            for key in self.camera_keys:
                path, offset = dataset.video_location(dataset.episodes[index], key)
                self.readers[key] = (VideoReader(path, dataset.fps), offset)
        except BaseException:
            self.close()
            raise

    def frame(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        timestamp = float(self.times[index])
        row = self.table.slice(index, 1).to_pylist()[0]
        images = {key: reader.read(offset+timestamp) for key, (reader, offset) in self.readers.items()}
        return dict(index=index, timestamp=timestamp, row=row, images=images)

    def close(self):
        for reader, _ in self.readers.values():
            reader.close()


class RerunView:
    def __init__(self, player, *, save=None):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr, self.player = rr, player
        images = rrb.Vertical(*(rrb.Spatial2DView(origin=f'/camera/{key}', name=key.removeprefix('observation.images.'))
                                for key in player.camera_keys))
        self.signals = [key for key in ('observation.joint', 'action.joint', 'observation.torque', 'observation.velocity')
                        if key in player.table.column_names]
        plots = rrb.Vertical(rrb.TextDocumentView(origin='/clip', name='Episode / time-based clipping'),
                             *(rrb.TimeSeriesView(origin=f'/signals/{key}', name=key) for key in self.signals))
        blueprint = rrb.Blueprint(rrb.Horizontal(images, plots),
                                  rrb.TimePanel(timeline='episode_time', state=rrb.PanelState.Expanded,
                                                duration_cursor=float(player.times[0])),
                                  collapse_panels=True)
        viewer = Path(sys.executable).parent/'rerun'
        if viewer.is_file():
            os.environ['PATH'] = str(viewer.parent)+os.pathsep+os.environ.get('PATH', '')
        rr.init('lerobot_v3_clip', recording_id=str(uuid.uuid4()), spawn=save is None,
                default_blueprint=blueprint, strict=True)
        if save is not None:
            rr.save(str(save))
        # default_blueprint does not replace an existing per-application layout.
        # The old clip viewer used /current/...; explicitly activate this layout
        # so restored views cannot keep pointing at image entities we no longer log.
        rr.send_blueprint(blueprint, make_active=True, make_default=True)
        self.recording = rr.get_global_data_recording()

    def show(self, frame):
        rr = self.rr
        rr.set_time('episode_time', duration=frame['timestamp'])
        # Images and curves share the episode timeline, so scrubbing restores
        # the corresponding image instead of always showing the latest frame.
        for key, image in frame['images'].items():
            rr.log(f'camera/{key}/image', rr.Image(image, color_model='RGB'))
        for key in self.signals:
            for i, value in enumerate(np.asarray(frame['row'][key]).reshape(-1)):
                rr.log(f'signals/{key}/{i+1}', rr.Scalars(float(value)))
        rr.log('clip', rr.TextDocument(
            f'{self.player.dataset.root.name}\nEpisode {self.player.index}\n'
            f'Loaded frame {frame["index"]}/{self.player.length-1}, t={frame["timestamp"]:.6f}s\n'
            'Inspect images using the episode_time timeline.\n'
            'After loading, enter a cut time in terminal (seconds or MM:SS), then y + Enter.\n'
            'Frames at or before that time are kept. c=episode menu; q=quit.'), static=True)
        self.recording.flush(timeout_sec=.1)

    def close(self):
        self.recording.disconnect()


def stream_episode(player, view, *, clock=time.monotonic, sleep=time.sleep):
    """Publish at most 60 original frames/s, decoding one frame at a time."""
    deadline = clock()
    for index in range(player.length):
        sleep(max(0., deadline-clock()))
        started = clock()
        view.show(player.frame(index))
        deadline = started+1/60


def parse_cut_time(value):
    """Episode seconds or MM:SS[.fraction] / HH:MM:SS[.fraction]."""
    parts = value.strip().split(':')
    if not 1 <= len(parts) <= 3:
        raise ValueError('Enter seconds (12.34) or MM:SS (00:12.34)')
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError('Enter seconds (12.34) or MM:SS (00:12.34)') from exc
    if any(not np.isfinite(n) or n < 0 for n in numbers):
        raise ValueError('Cut time must be finite and nonnegative')
    if len(numbers) > 1 and (any(n >= 60 for n in numbers[1:]) or any(n != int(n) for n in numbers[:-1])):
        raise ValueError('Use integer hours/minutes and seconds below 60')
    return sum(n*60**i for i, n in enumerate(reversed(numbers)))


def select_cut(times, *, clip=False, input_fn=input, output_fn=print):
    """User-entered time is authoritative; y cannot save an implicit EOF selection."""
    times = np.asarray(times, dtype=float)
    pending = None
    output_fn(f'Loaded. Drag the Rerun episode_time timeline to inspect this episode '
              f'({times[0]:.6f}..{times[-1]:.6f} seconds).')
    if not clip:
        output_fn('Read-only: c=episode menu, q=quit. Use --clip to save cuts.')
    while True:
        try:
            value = input_fn('cut time (seconds / MM:SS), y=save, c=next, q=quit> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 'quit', None
        if value in ('q', 'quit', 'exit'):
            return 'quit', None
        if value == 'c':
            return 'skip', None
        if not clip:
            output_fn('Read-only: enter c or q; --clip is required to save.')
            continue
        if value == 'y':
            if pending is not None:
                return 'save', pending
            output_fn('Enter a valid cut time first, then y + Enter.')
            continue
        # An invalid new time must not leave an earlier selection armed.
        pending = None
        try:
            seconds = parse_cut_time(value)
            if seconds > times[-1]+1e-4:
                raise ValueError(f'Time exceeds this episode ({times[-1]:.6f}s)')
            # v3 timestamps are often float32: include an exact printed frame
            # boundary despite its floating-point representation error.
            tolerance = max(1e-9, float(abs(np.spacing(np.float32(seconds)))))
            keep = int(np.searchsorted(times, seconds+tolerance, side='right'))
            if keep == 0:
                raise ValueError(f'Time precedes the first frame ({times[0]:.6f}s)')
            if keep == len(times):
                raise ValueError('No frames after this time; choose an earlier time to clip')
        except ValueError as exc:
            output_fn(str(exc))
            continue
        pending = keep
        output_fn(f'Cut time {seconds:.6f}s -> keep frames 0..{keep-1} (last t={times[keep-1]:.6f}s), '
                  f'drop {len(times)-keep} frames. Enter y to save, or enter another time.')


def browse(root, *, clip=False, output=None, camera=None, input_fn=input, output_fn=print):
    from scripts.lerobotv3_clip import output_dataset, save_clip

    source = Path(root).expanduser().resolve()
    destination = Path(output).expanduser().resolve() if output else source.with_name(source.name+'_clipped')
    dataset = V3Dataset(output_dataset(source, destination) if clip else source)
    output_fn(f'LeRobot v3: {dataset.root}; episodes: {", ".join(map(str, sorted(dataset.episodes)))}')
    output_fn('Stream 60 original frames/s; inspect the Rerun timeline, enter a cut time, then y + Enter.')
    if clip:
        output_fn(f'Cleaned dataset: {destination}; source stays unchanged.')
    while True:
        try:
            value = input_fn('episode> ').strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if value.lower() in {'q', 'quit', 'exit'}:
            return 0
        try:
            index = int(value)
            if index not in dataset.episodes:
                raise ValueError(f'Episode {index} does not exist')
        except ValueError as exc:
            output_fn(str(exc)); continue
        player = view = None
        try:
            player = EpisodePlayer(dataset, index, camera)
            view = RerunView(player)
            stream_episode(player, view)
            action, keep = select_cut(player.times, clip=clip, input_fn=input_fn, output_fn=output_fn)
        except KeyboardInterrupt:
            return 0
        except (RuntimeError, ValueError, OSError, ImportError) as exc:
            output_fn(f'Playback failed: {exc}'); continue
        finally:
            if player is not None:
                player.close()
            if view is not None:
                view.close()
        if action == 'quit':
            return 0
        if action == 'save':
            if keep == dataset.episodes[index]['length']:
                output_fn('Selected the final frame; nothing to remove.'); continue
            output_fn(f'Saving episode {index}: keeping frames 0..{keep-1}; updating v3 indexes and statistics...')
            try:
                save_clip(dataset, index, keep, destination, source_root=source, output_fn=output_fn)
                dataset = V3Dataset(destination)
                output_fn(f'Saved: {destination}; select the next episode.')
            except (RuntimeError, ValueError, OSError) as exc:
                output_fn(f'Save failed; previous dataset remains intact: {exc}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, help='v3 dataset folder; prompted when omitted')
    parser.add_argument('--clip', action='store_true', help='enable y to save an inclusive tail cut')
    parser.add_argument('--output', type=Path, help='cleaned dataset; defaults to <root>_clipped')
    parser.add_argument('--camera', help='camera name(s), e.g. front or front/left_wrist/right_wrist')
    args = parser.parse_args(argv)
    if args.output and not args.clip:
        parser.error('--output requires --clip')
    try:
        root = args.root or Path(input('LeRobot v3 folder> ').strip())
        return browse(root, clip=args.clip, output=args.output, camera=args.camera)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
