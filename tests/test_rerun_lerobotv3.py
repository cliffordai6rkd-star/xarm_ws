import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import rerun_lerobotv3 as review
from scripts import lerobotv3_clip as clipper


def digests(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def dataset(tmp_path):
    av = pytest.importorskip('av')
    root = tmp_path/'dataset'
    (root/'meta/episodes/chunk-000').mkdir(parents=True)
    (root/'data/chunk-000').mkdir(parents=True)
    keys = ['observation.images.front', 'observation.images.right_wrist']
    features = {key: dict(dtype='video', shape=[16, 16, 3]) for key in keys}
    features.update({key: dict(dtype='int64', shape=[1]) for key in ('episode_index', 'frame_index', 'index', 'task_index')})
    features.update(timestamp=dict(dtype='float32', shape=[1]),
                    **{'observation.joint': dict(dtype='float32', shape=[7]),
                       'action.joint': dict(dtype='float32', shape=[7])})
    info = dict(codebase_version='v3.0', fps=25, total_episodes=2, total_frames=8, total_tasks=1,
                chunks_size=1000, splits={'train': '0:2'}, features=features,
                data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    (root/'meta/info.json').write_text(json.dumps(info))
    (root/'meta/stats.json').write_text('{}')
    pq.write_table(pa.table({'task_index': [0], '__index_level_0__': ['test']}), root/'meta/tasks.parquet')
    rows, episodes = [], []
    for episode_index in range(2):
        for i in range(4):
            rows.append(dict(episode_index=episode_index, frame_index=i, index=episode_index*4+i,
                             task_index=0, timestamp=i/25,
                             **{'observation.joint': [episode_index*10+i]*7, 'action.joint': [episode_index*10+i+1]*7}))
        ep = dict(episode_index=episode_index, length=4, tasks=['test'],
                  dataset_from_index=4*episode_index, dataset_to_index=4*(episode_index+1),
                  **{'data/chunk_index': 0, 'data/file_index': 0,
                     'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0})
        for key in keys:
            ep.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': 0,
                       f'videos/{key}/from_timestamp': episode_index*4/25,
                       f'videos/{key}/to_timestamp': (episode_index+1)*4/25})
        for key, spec in features.items():
            if spec['dtype'] != 'video':
                stats = clipper._numeric_stats([r[key] for r in rows[-4:]])
                ep.update({f'stats/{key}/{k}': v for k, v in stats.items()})
        episodes.append(ep)
    # Shared row group and shared videos: clipping ep0 must not corrupt ep1.
    pq.write_table(pa.Table.from_pylist(rows), root/'data/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist(episodes), root/'meta/episodes/chunk-000/file-000.parquet')
    for key in keys:
        path = root/info['video_path'].format(video_key=key, chunk_index=0, file_index=0)
        path.parent.mkdir(parents=True)
        with av.open(str(path), 'w') as output:
            stream = output.add_stream('mpeg4', rate=25)
            stream.width = stream.height = 16
            stream.pix_fmt = 'yuv420p'
            for i in range(8):
                image = np.full((16, 16, 3), 20+i*25, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    return root


def test_reader_decodes_only_requested_episode_frames_with_offsets_and_backstep(dataset, monkeypatch):
    calls = []
    original = review.VideoReader.read
    monkeypatch.setattr(review.VideoReader, 'read', lambda self, t: calls.append(t) or original(self, t))
    source = review.V3Dataset(dataset)
    player = review.EpisodePlayer(source, 1)
    assert calls == []  # Opening an episode does not decode its complete video.
    try:
        first = player.frame(0)
        last = player.frame(3)
        back = player.frame(1)
        assert len(calls) == 6
        for key in player.camera_keys:
            assert abs(first['images'][key].mean()-120) < 5
            assert abs(last['images'][key].mean()-195) < 5
            assert abs(back['images'][key].mean()-145) < 5
        assert calls[:2] == [.16, .16]
    finally:
        player.close()


def test_clip_updates_all_rows_indexes_ranges_stats_and_resumes_without_touching_source(dataset):
    before = digests(dataset)
    output = dataset.with_name('cleaned')
    original = review.V3Dataset(dataset)
    clipper.save_clip(original, 0, 2, output, source_root=dataset, output_fn=lambda _: None)
    result = review.V3Dataset(output)
    assert result.info['total_frames'] == 6
    assert result.episodes[0]['length'] == 2
    assert (result.episodes[1]['dataset_from_index'], result.episodes[1]['dataset_to_index']) == (2, 6)
    assert result.episodes[0]['videos/observation.images.front/to_timestamp'] == .08
    assert result.episodes[1]['videos/observation.images.front/from_timestamp'] == .16
    table = pq.read_table(output/'data/chunk-000/file-000.parquet').to_pydict()
    assert table['index'] == list(range(6))
    assert table['frame_index'] == [0, 1, 0, 1, 2, 3]
    assert table['action.joint'][1] == [2]*7
    assert table['observation.joint'][2] == [10]*7
    assert result.episodes[0]['stats/action.joint/count'] == [2]
    stats = json.loads((output/'meta/stats.json').read_text())
    np.testing.assert_allclose(stats['action.joint']['mean'], np.mean(table['action.joint'], axis=0))
    np.testing.assert_allclose(stats['action.joint']['std'], np.std(table['action.joint'], axis=0))
    assert stats['index']['count'] == [6]
    assert clipper.output_dataset(dataset, output) == output
    clipper.save_clip(result, 1, 1, output, source_root=dataset, output_fn=lambda _: None)
    result = review.V3Dataset(output)
    assert result.info['total_frames'] == 3
    assert result.read_episode(1)['index'].to_pylist() == [2]
    assert len(json.loads((output/clipper.HISTORY).read_text())['clips']) == 2
    assert digests(dataset) == before
    assert not list(output.parent.glob('.cleaned.staging-*'))
    player = review.EpisodePlayer(result, 1)
    try:
        assert abs(player.frame(0)['images']['observation.images.front'].mean()-120) < 5
    finally:
        player.close()


def test_failed_publish_keeps_existing_output_and_source(dataset, monkeypatch):
    output = dataset.with_name('cleaned')
    clipper.save_clip(review.V3Dataset(dataset), 0, 3, output, source_root=dataset, output_fn=lambda _: None)
    old_source, old_output = digests(dataset), digests(output)
    def fail(*args):
        raise OSError('simulated publish failure')
    monkeypatch.setattr(clipper, '_publish', fail)
    with pytest.raises(OSError, match='simulated'):
        clipper.save_clip(review.V3Dataset(output), 1, 2, output, source_root=dataset, output_fn=lambda _: None)
    assert digests(output) == old_output
    assert digests(dataset) == old_source
    assert not list(output.parent.glob('.cleaned.staging-*'))


def test_rejects_unrelated_output_empty_cut_and_changed_metadata(dataset):
    source = review.V3Dataset(dataset)
    output = dataset.with_name('unrelated')
    output.mkdir()
    with pytest.raises(ValueError, match='unrelated'):
        clipper.output_dataset(dataset, output)
    with pytest.raises(ValueError, match='separate'):
        clipper.output_dataset(dataset, dataset/'child')
    output = dataset.with_name('cleaned')
    with pytest.raises(ValueError, match='at least one'):
        clipper.save_clip(source, 0, 0, output, source_root=dataset)
    (dataset/'meta/stats.json').write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match='changed'):
        clipper.save_clip(source, 0, 2, output, source_root=dataset)
    assert not output.exists()


def test_stream_is_incremental_and_targets_60fps():
    loaded, shown, waits = [], [], []
    now = [0.]
    def frame(index):
        loaded.append(index)
        return {'index': index, 'timestamp': index/25}
    def show(frame):
        assert loaded == list(range(frame['index']+1))  # no decode-ahead
        shown.append(frame['index'])
    def sleep(duration):
        waits.append(duration); now[0] += duration
    player = SimpleNamespace(length=4, frame=frame)
    review.stream_episode(player, SimpleNamespace(show=show), clock=lambda: now[0], sleep=sleep)
    assert loaded == shown == list(range(4))
    np.testing.assert_allclose(waits, [0, 1/60, 1/60, 1/60])


@pytest.mark.parametrize('clip,keys,expected', [(True, ['y', 'c'], 'skip'),
                                              (False, ['0.04', 'y', 'q'], 'quit')])
def test_y_requires_clip_mode_and_explicit_time(clip, keys, expected):
    keys = iter(keys)
    action, keep = review.select_cut(np.arange(4)/25, input_fn=lambda _: next(keys), clip=clip,
                                    output_fn=lambda _: None)
    assert action == expected and keep is None


@pytest.mark.parametrize('value,keep', [('0', 1), ('0.08', 3), ('0.10', 3),
                                       ('0.20', 6), ('00:00.20', 6), ('0:00:00.20', 6)])
def test_entered_time_retains_frame_at_boundary_or_previous_frame(value, keep):
    times = (np.arange(10)/25).astype(np.float32)
    answers = iter([value, 'y'])
    assert review.select_cut(times, clip=True, input_fn=lambda _: next(answers), output_fn=lambda _: None) == ('save', keep)


@pytest.mark.parametrize('invalid', ['nan', 'inf', '-1', '100', '0.36', '00:61', 'nonsense', ''])
def test_invalid_or_final_time_clears_prior_selection(invalid):
    answers = iter(['0.08', invalid, 'y', 'c'])
    assert review.select_cut(np.arange(10)/25, clip=True, input_fn=lambda _: next(answers),
                             output_fn=lambda _: None) == ('skip', None)


def test_time_can_be_revised_before_confirmation():
    answers = iter(['0.20', '0.04', 'y'])
    assert review.select_cut(np.arange(10)/25, clip=True, input_fn=lambda _: next(answers),
                             output_fn=lambda _: None) == ('save', 2)


def test_camera_images_are_temporal_for_rerun_scrubbing():
    calls = []
    rr = SimpleNamespace(set_time=lambda *a, **kw: calls.append(('time', kw)),
                         Image=lambda *a, **kw: 'image', Scalars=lambda x: x, TextDocument=lambda x: x,
                         log=lambda path, value, **kw: calls.append((path, kw)))
    view = review.RerunView.__new__(review.RerunView)
    view.rr, view.signals = rr, []
    view.player = SimpleNamespace(dataset=SimpleNamespace(root=Path('example')), index=0, length=3)
    view.recording = SimpleNamespace(flush=lambda **kw: None)
    view.show(dict(timestamp=.04, index=1, images={'front': np.zeros((4, 4, 3), np.uint8)}))
    assert calls[0] == ('time', {'duration': .04})
    assert ('camera/front/image', {}) in calls  # must not use static=True


def test_rerun_recording_activates_image_layout_and_episode_cursor_without_gui(dataset, tmp_path, monkeypatch):
    rr = pytest.importorskip('rerun')
    import rerun.blueprint as rrb
    sent, panels, views = [], [], []
    send, panel, spatial = rr.send_blueprint, rrb.TimePanel, rrb.Spatial2DView
    def capture_send(blueprint, **kwargs):
        sent.append(kwargs)
        return send(blueprint, **kwargs)
    def capture_panel(**kwargs):
        panels.append(kwargs)
        return panel(**kwargs)
    def capture_view(**kwargs):
        views.append(kwargs)
        return spatial(**kwargs)
    monkeypatch.setattr(rr, 'send_blueprint', capture_send)
    monkeypatch.setattr(rrb, 'TimePanel', capture_panel)
    monkeypatch.setattr(rrb, 'Spatial2DView', capture_view)
    player = review.EpisodePlayer(review.V3Dataset(dataset), 0)
    view = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            view = review.RerunView(player, save=tmp_path/'clip.rrd')
            view.show(player.frame(0))
            view.show(player.frame(1))
    finally:
        if view:
            view.close()
        player.close()
    assert (tmp_path/'clip.rrd').stat().st_size > 1000
    # A cached old /current view is bypassed by explicitly activating /camera
    # views, and a cursor inherited from another episode is reset into range.
    assert any(s.get('make_active') is True for s in sent)
    assert panels[-1]['timeline'] == 'episode_time'
    assert panels[-1]['duration_cursor'] == 0.
    assert {view['origin'] for view in views} == {f'/camera/{key}' for key in player.camera_keys}


def test_browser_saves_then_prompts_for_next_episode_in_cleaned_dataset(dataset, monkeypatch):
    prompts = iter(['0', '0', 'y', '1', 'c', 'q'])
    opened = []
    class View:
        def __init__(self, player):
            opened.append((player.dataset.root, player.index))
        def show(self, *args, **kwargs):
            pass
        def close(self):
            pass
    monkeypatch.setattr(review, 'RerunView', View)
    assert review.browse(dataset, clip=True, input_fn=lambda _: next(prompts), output_fn=lambda _: None) == 0
    output = dataset.with_name(dataset.name+'_clipped')
    assert opened == [(dataset, 0), (output, 1)]
    assert review.V3Dataset(output).episodes[0]['length'] == 1
