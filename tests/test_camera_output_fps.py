"""Timestamp-based frame selection and recorded H5 rate without real devices."""
from dataclasses import replace
import queue
import threading

import h5py
import numpy as np
import pytest

from nero_collection import cameras
from nero_collection.config import CameraConfig, _parse_camera
from nero_collection.h5_writer import EpisodeBuffer


def native_frames(count=300, *, jitter=False):
    result = []
    for i in range(count):
        stamp = 1_700_000_000_000_000 + round(i * 1e6 / 30)
        if jitter and i:
            stamp += (-1 if i % 2 else 1) * 2000
        result.append(cameras.CameraFrame(
            'wrist', stamp, np.full((4, 5, 3), i % 256, np.uint8),
            np.full((8, 10, 3), i % 256, np.uint8), np.full((4, 5), i, np.uint16)))
    return result


class Source:
    name = 'wrist'

    def __init__(self, frames, stop=None):
        self.frames, self.stop_event = frames, stop
        self.started = self.stopped = False

    def start(self):
        self.started = True
        self.cursor = 0

    def stop(self):
        self.stopped = True

    def poll(self):
        if self.cursor >= len(self.frames):
            if self.stop_event is not None:
                self.stop_event.set()
            return None
        frame = self.frames[self.cursor]
        self.cursor += 1
        return frame


def selected_frames(frames, fps=25):
    source = Source(frames)
    limited = cameras.RateLimitedCamera(source, fps)
    limited.start()
    result = [frame for _ in frames if (frame := limited.poll()) is not None]
    limited.stop()
    assert source.started and source.stopped
    return result


@pytest.mark.parametrize('jitter', [False, True])
def test_30_to_25_retains_five_of_six_frames_without_clock_drift(jitter):
    raw = native_frames(3000, jitter=jitter)
    kept = selected_frames(raw)
    assert len(kept) == 2500  # 100 seconds; a naive 40 ms cooldown would give 1500.
    intervals = np.diff([frame.timestamp_us for frame in kept])
    assert np.all(intervals > 0)
    assert abs((len(kept)-1)*1e6/(kept[-1].timestamp_us-kept[0].timestamp_us)-25) < .02
    # Selection retains the exact acquisition frame, including RGB/depth/preview.
    original = {frame.timestamp_us: frame for frame in raw}
    assert all(frame is original[frame.timestamp_us] for frame in kept)
    assert len({id(frame) for frame in kept}) == len(kept)
    if not jitter:
        assert set(intervals) <= {33333, 33334, 66666, 66667}


def test_gaps_duplicates_and_old_timestamps_never_create_extra_frames():
    raw = native_frames(6)
    frames = [raw[0], raw[1], raw[1], raw[2], raw[1],
              replace(raw[3], timestamp_us=raw[0].timestamp_us+1_000_000),
              replace(raw[4], timestamp_us=raw[0].timestamp_us+1_033_333),
              replace(raw[5], timestamp_us=raw[0].timestamp_us+1_066_667)]
    kept = selected_frames(frames)
    assert [f.timestamp_us-raw[0].timestamp_us for f in kept] == [0, 66667, 1_000_000, 1_066_667]


def test_restarting_camera_resets_output_time_bins():
    frames = native_frames(60)
    source = Source(frames)
    limited = cameras.RateLimitedCamera(source, 25)
    for _ in range(2):
        limited.start()
        assert len([frame for _ in frames if (frame := limited.poll()) is not None]) == 50
        limited.stop()


@pytest.mark.parametrize('value', [0, -1, 31, float('nan'), float('inf'), True])
def test_invalid_output_fps_rejected_for_both_config_entry_points(value):
    with pytest.raises(ValueError, match='output_fps'):
        CameraConfig('wrist', fps=30, output_fps=value)
    with pytest.raises(ValueError, match='output_fps'):
        _parse_camera({'name': 'wrist', 'backend': 'mock', 'fps': 30, 'output_fps': value})


def test_camera_parsers_retain_native_and_output_frequencies():
    from gello_teleop.dual_gello_collect import _camera_config

    for parse in (_parse_camera, _camera_config):
        config = parse({'name': 'wrist', 'backend': 'mock', 'fps': 30, 'output_fps': 25})
        assert config.fps == 30 and config.output_fps == 25 and config.effective_fps == 25
    assert CameraConfig('wrist', fps=30).effective_fps == 30


@pytest.mark.parametrize('direct_preview', [False, True])
def test_acquisition_worker_selects_frames_before_recording_ipc(monkeypatch, direct_preview):
    raw = native_frames()
    stop = threading.Event()
    source = Source(raw, stop)
    seen_config = []

    def build(config):
        seen_config.append(config)
        return source

    # Keep the real factory and worker; only the device is replaced.
    monkeypatch.setattr(cameras, 'RealSenseCamera', build)
    monkeypatch.setattr(cameras.os, 'nice', lambda _: None)
    config = CameraConfig('wrist', backend='realsense', serial_number='offline',
                          fps=30, output_fps=25, depth=True, visualize=True)
    output, status, fault = queue.Queue(), queue.Queue(), queue.Queue()
    preview = queue.Queue() if direct_preview else None
    cameras._camera_acquisition_worker(config, output, status, fault, stop, preview)
    assert seen_config[0].fps == 30  # SDK still receives a supported native mode.
    assert status.get_nowait() == (True, None, None) and fault.empty()
    assert source.stopped
    kept = []
    while not output.empty():
        kept.append(output.get_nowait())
    assert len(kept) == 250
    assert all(frame.depth is not None for frame in kept)
    assert all((frame.preview_frame is None) == direct_preview for frame in kept)
    assert {frame.timestamp_us for frame in kept} <= {frame.timestamp_us for frame in raw}
    if direct_preview:
        assert preview.qsize() == 250


@pytest.mark.parametrize('output_fps', [None, 30])
def test_default_full_rate_camera_path_is_unchanged(monkeypatch, output_fps):
    source = Source(native_frames(30))
    monkeypatch.setattr(cameras, 'MockCamera', lambda _: source)
    assert cameras._build_camera(CameraConfig('wrist', backend='mock', fps=30,
                                              output_fps=output_fps)) is source


def test_recorded_h5_contains_25hz_images_and_100hz_joint_state(tmp_path):
    from test_dual_gello_pipeline import make_pipeline

    pipeline = make_pipeline()  # Synthetic arm/leader factories; no hardware connection.
    config = replace(pipeline.collection_config,
                     cameras=(CameraConfig('wrist', fps=30, output_fps=25, depth=True),))
    buffer = EpisodeBuffer(config, ('right',), enable_online_tau_ext=False)
    try:
        for i in range(1000):
            buffer.append_teleop(1_700_000_000_000_000+i*10000,
                                {'q_follower': ('q', np.full(7, i/1000))})
        kept = selected_frames(native_frames())
        for frame in kept:
            buffer.append_camera(frame.camera_name, frame.timestamp_us, frame.frame, frame.depth)
        path = buffer.save(tmp_path/'episode.h5')
        with h5py.File(path) as episode:
            assert episode['teleop/right_q_xarm'].shape == (1000, 7)
            assert episode['teleop'].attrs['sample_rate_hz'] == 100
            group = episode['cameras/wrist']
            assert group['frames'].shape == (250, 4, 5, 3)
            assert group['depth'].shape == (250, 4, 5)
            assert group.attrs['fps'] == 25 and group.attrs['capture_fps'] == 30
            assert group.attrs['measured_fps'] == pytest.approx(25, abs=.05)
            np.testing.assert_array_equal(group['timestamp_us'][:], [f.timestamp_us for f in kept])
            np.testing.assert_array_equal(group['frames'][:], np.stack([f.frame for f in kept]))
            np.testing.assert_array_equal(group['depth'][:], np.stack([f.depth for f in kept]))
    finally:
        pipeline.close()
