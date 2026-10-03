"""Independent policy and preview images keep their own IPC destinations."""
import pickle
import queue
import threading

import numpy as np
import pytest

from nero_collection import cameras
from nero_collection.config import CameraConfig


def _acquire_one_frame(monkeypatch, frame, preview_queue):
    stop = threading.Event()
    collector_queue, status_queue, fault_queue = queue.Queue(), queue.Queue(), queue.Queue()

    class Source:
        started = stopped = False

        def start(self):
            self.started = True

        def poll(self):
            stop.set()
            return frame

        def stop(self):
            self.stopped = True

    source = Source()
    monkeypatch.setattr(cameras, '_build_camera', lambda config: source)
    monkeypatch.setattr(cameras.os, 'nice', lambda priority: None)
    config = CameraConfig(frame.camera_name, output_size=(224, 224), visualize=True, depth=True)
    cameras._camera_acquisition_worker(config, collector_queue, status_queue, fault_queue, stop, preview_queue)
    assert source.started and source.stopped
    assert status_queue.get_nowait() == (True, None, None)
    assert fault_queue.empty()
    return collector_queue.get_nowait()


def test_direct_preview_omits_full_size_image_from_collector_ipc(monkeypatch):
    policy = np.full((224, 224, 3), [10, 20, 30], np.uint8)
    preview = np.full((480, 640, 3), [40, 50, 60], np.uint8)
    depth = np.full((224, 224), 1234, np.uint16)
    original = cameras.CameraFrame('right_wrist', 1_234_567, policy, preview, depth)
    preview_queue = queue.Queue()

    collected = _acquire_one_frame(monkeypatch, original, preview_queue)
    displayed = preview_queue.get_nowait()

    assert collected.preview_frame is None
    assert collected.camera_name == original.camera_name
    assert collected.timestamp_us == original.timestamp_us
    assert collected.frame is policy and collected.depth is depth
    assert collected.frame.shape == (224, 224, 3)
    assert displayed is original
    assert displayed.preview_frame is preview
    assert displayed.preview_frame.shape == (480, 640, 3)
    assert len(pickle.dumps(collected)) < len(pickle.dumps(original)) / 4


@pytest.mark.parametrize('has_preview', [False, True])
def test_without_direct_preview_keeps_original_frame_for_recording_and_dispatch(monkeypatch, has_preview):
    policy = np.full((224, 224, 3), 20, np.uint8)
    preview = np.full((480, 640, 3), 50, np.uint8) if has_preview else None
    depth = np.full((224, 224), 1234, np.uint16)
    original = cameras.CameraFrame('right_wrist', 1_234_567, policy, preview, depth)

    collected = _acquire_one_frame(monkeypatch, original, None)

    assert collected is original
    assert collected.frame is policy and collected.depth is depth
    assert collected.preview_frame is preview
