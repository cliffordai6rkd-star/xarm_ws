"""Verify RealSense startup clock calibration without opening a device."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nero_collection import cameras
from nero_collection.config import CameraConfig


class FakeStreamConfig:
    def enable_device(self, serial):
        self.serial = serial

    def enable_stream(self, *args):
        pass


class FakeFrames:
    def __init__(self, timestamp_ms, timestamp_error=None):
        self.timestamp_ms = timestamp_ms
        self.timestamp_error = timestamp_error

    def get_timestamp(self):
        if self.timestamp_error is not None:
            raise self.timestamp_error
        return self.timestamp_ms

    def get_color_frame(self):
        return SimpleNamespace(get_data=lambda: np.zeros((2, 3, 3), dtype=np.uint8))


class FakePipeline:
    def __init__(self, first_frames, *, wait_error=None, stop_error=None):
        self.first_frames = first_frames
        self.next_frames = FakeFrames(1267.5)
        self.wait_error = wait_error
        self.stop_error = stop_error
        self.stop_calls = 0

    def start(self, config):
        return SimpleNamespace(get_device=lambda: SimpleNamespace(
            first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001)))

    def wait_for_frames(self, *, timeout_ms):
        if self.wait_error is not None:
            raise self.wait_error
        return self.first_frames

    def poll_for_frames(self):
        return self.next_frames

    def stop(self):
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error


def _camera(monkeypatch, pipeline, *, depth=False):
    fake_rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=FakeStreamConfig,
        stream=SimpleNamespace(color=0, depth=1),
        format=SimpleNamespace(rgb8=0, z16=1),
        align=lambda stream: SimpleNamespace(process=lambda frames: frames),
    )
    monkeypatch.setitem(sys.modules, 'pyrealsense2', fake_rs)
    return cameras.RealSenseCamera(CameraConfig(
        'left_wrist', backend='realsense', serial_number='offline',
        width=3, height=2, depth=depth))


@pytest.mark.parametrize('failure', ['wait', 'timestamp', 'nan', 'infinite'])
def test_startup_timestamp_failure_stops_and_clears_pipeline(monkeypatch, failure):
    first_frames = FakeFrames(1234.5,
                             RuntimeError('timestamp unavailable') if failure == 'timestamp' else None)
    if failure in {'nan', 'infinite'}:
        first_frames.timestamp_ms = float('nan' if failure == 'nan' else 'inf')
    pipeline = FakePipeline(first_frames,
                            wait_error=TimeoutError('no first frame') if failure == 'wait' else None)
    camera = _camera(monkeypatch, pipeline, depth=True)

    with pytest.raises(cameras.CameraUnavailable, match='first frame timestamp') as error:
        camera.start()

    assert error.value.__cause__ is not None
    assert pipeline.stop_calls == 1
    assert camera._pipeline is None
    assert camera._align is None
    assert camera._timestamp_offset_us == 0
    with pytest.raises(cameras.CameraUnavailable, match='not started'):
        camera.poll()


def test_device_clock_is_calibrated_to_host_time_before_poll(monkeypatch):
    pipeline = FakePipeline(FakeFrames(1234.5))
    camera = _camera(monkeypatch, pipeline)
    clock = {'now': 1_700_000_000_000_000}
    monkeypatch.setattr(cameras, 'now_us', lambda: clock['now'])

    camera.start()
    calibrated_host_time = clock['now']
    clock['now'] += 35_000
    frame = camera.poll()

    assert frame.timestamp_us == calibrated_host_time + 33_000
    assert clock['now'] - frame.timestamp_us == 2_000
    assert frame.frame.shape == (2, 3, 3)
    assert camera.poll() is None  # A duplicate SDK frame is not a new observation.
    assert pipeline.stop_calls == 0
    camera.stop()
    assert pipeline.stop_calls == 1


def test_timestamp_error_remains_explicit_when_cleanup_stop_fails(monkeypatch):
    failure = TimeoutError('first frame timeout')
    pipeline = FakePipeline(FakeFrames(1234.5), wait_error=failure,
                            stop_error=RuntimeError('stop failed'))
    camera = _camera(monkeypatch, pipeline, depth=True)

    with pytest.raises(cameras.CameraUnavailable, match='first frame timeout') as error:
        camera.start()

    assert error.value.__cause__ is failure
    assert pipeline.stop_calls == 1
    assert camera._pipeline is None
    assert camera._align is None


def test_startup_queue_bias_is_corrected_without_refreshing_duplicates(monkeypatch):
    pipeline = FakePipeline(FakeFrames(1000.))
    camera = _camera(monkeypatch, pipeline)
    clock = {'now': 1_700_000_000_000_000}
    monkeypatch.setattr(cameras, 'now_us', lambda: clock['now'])
    camera.start()
    clock['now'] += 38_000
    pipeline.next_frames = FakeFrames(1300.)  # Old mapping would lead by 262 ms.
    first = camera.poll()
    assert first.timestamp_us == clock['now']
    clock['now'] += 50_000
    assert camera.poll() is None
    pipeline.next_frames = FakeFrames(1333.)
    second = camera.poll()
    assert second.timestamp_us - first.timestamp_us == 33_000
    assert clock['now'] - second.timestamp_us == 17_000


def test_delayed_frame_remains_old(monkeypatch):
    pipeline = FakePipeline(FakeFrames(1000.))
    camera = _camera(monkeypatch, pipeline)
    clock = {'now': 1_700_000_000_000_000}
    monkeypatch.setattr(cameras, 'now_us', lambda: clock['now'])
    camera.start()
    clock['now'] += 500_000
    pipeline.next_frames = FakeFrames(1033.)
    frame = camera.poll()
    assert clock['now'] - frame.timestamp_us == 467_000
