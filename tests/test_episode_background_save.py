"""Saving must not block takeover, mutate a frozen episode or weaken motion deadlines."""
from dataclasses import replace
import threading
import time
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from nero_collection.fixed_rate import FixedRateTicker
from nero_collection.h5_writer import EpisodeBuffer, _stack
from test_dual_gello_pipeline import make_pipeline


class Clock:
    def __init__(self):
        self.now = 0.
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def test_recorder_resumes_after_438ms_lateness_without_bursting_missed_samples():
    clock = Clock()
    ticker = FixedRateTicker(100, .03, strict=False, monotonic=clock.monotonic, sleep=clock.sleep)
    assert ticker.wait('recorder')[0] == 0
    clock.now = .0538  # Tick 1 was due at .010: the reported 43.8 ms delay.
    _, late = ticker.wait('recorder')
    assert late == pytest.approx(.0438)
    assert ticker.last_skipped_ticks == 4
    index, late = ticker.wait('recorder')
    assert index == 6 and late == 0
    assert clock.now == pytest.approx(.06)


def test_control_deadline_remains_fatal_for_the_same_delay():
    clock = Clock()
    ticker = FixedRateTicker(100, .03, monotonic=clock.monotonic, sleep=clock.sleep)
    ticker.wait('control')
    clock.now = .0538
    with pytest.raises(RuntimeError, match='missed deadline.*control'):
        ticker.wait('control')


def episode_pipeline(tmp_path):
    pipeline = make_pipeline()
    pipeline.collection_config = replace(pipeline.collection_config,
        output=replace(pipeline.collection_config.output, directory=tmp_path))
    pipeline.state = 'recording'
    pipeline.recording = True
    pipeline.recording_started_t = time.monotonic()
    pipeline.buffer = EpisodeBuffer(pipeline.collection_config, pipeline.ARM_NAMES, enable_online_tau_ext=False)
    for index in range(2):
        pipeline.buffer.append_teleop(1000000+index*10000,
            {'q_follower': ('q', np.full(14, index, float))})
    pipeline.buffer.append_camera('camera', 1000000, np.full((8, 8, 3), 7, np.uint8))
    return pipeline


def test_background_save_freezes_episode_and_close_waits_after_device_shutdown(tmp_path, monkeypatch, caplog):
    pipeline = episode_pipeline(tmp_path)
    entered, release = threading.Event(), threading.Event()
    frozen = pipeline.buffer
    original_save = frozen.save
    def blocked_save(target):
        entered.set()
        assert release.wait(3.)
        return original_save(target)
    monkeypatch.setattr(frozen, 'save', blocked_save)
    started = time.monotonic()
    target = pipeline.stop_episode(True, background=True)
    assert time.monotonic()-started < .5
    assert entered.wait(1.) and not target.exists()
    assert pipeline.buffer is None and not pipeline.recording
    assert pipeline.state == 'following'
    pipeline.poll()  # Continues handling state and camera queues while saving.
    assert not pipeline.stop_event.is_set()
    assert frozen.sample_count == 2
    stopped = threading.Event()
    monkeypatch.setattr(pipeline.camera_manager, 'stop', stopped.set)
    finished = threading.Event()
    def close():
        pipeline.close()
        finished.set()
    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert stopped.wait(1.) and not finished.is_set()
        release.set()
        assert finished.wait(3.)
        with h5py.File(target) as episode:
            assert len(episode['teleop/timestamp_us']) == 2
            np.testing.assert_array_equal(episode['teleop/q_follower'][:], np.array([np.zeros(14), np.ones(14)]))
            assert episode['cameras/camera/frames'].shape == (1, 8, 8, 3)
        assert pipeline.episode_save_executor is None and not pipeline.pending_episode_saves
    finally:
        release.set()
        closer.join(timeout=3.)


def test_two_queued_episodes_reserve_distinct_paths_and_never_mix_rows(tmp_path, monkeypatch):
    pipeline = episode_pipeline(tmp_path)
    release = threading.Event()
    monkeypatch.setattr(pipeline, '_consume_samples', lambda: None)
    first = pipeline.buffer
    save = first.save
    def blocked(target):
        assert release.wait(3.)
        return save(target)
    monkeypatch.setattr(first, 'save', blocked)
    try:
        first_path = pipeline.stop_episode(True, background=True)
        pipeline.recording = True
        pipeline.recording_started_t = time.monotonic()
        pipeline.buffer = EpisodeBuffer(pipeline.collection_config, pipeline.ARM_NAMES, enable_online_tau_ext=False)
        pipeline.buffer.append_teleop(2000000, {'q_follower': ('q', np.full(14, 9.))})
        second_path = pipeline.stop_episode(True, background=True)
        assert first_path != second_path
        assert '_0000_' in first_path.name and '_0001_' in second_path.name
        release.set()
        pipeline.close()
        with h5py.File(first_path) as first_h5, h5py.File(second_path) as second_h5:
            assert len(first_h5['teleop/timestamp_us']) == 2
            assert len(second_h5['teleop/timestamp_us']) == 1
            np.testing.assert_array_equal(second_h5['teleop/q_follower'][0], np.full(14, 9.))
    finally:
        release.set()
        pipeline.close()


def test_failed_background_write_is_reported_and_retains_frozen_data(tmp_path, monkeypatch, caplog):
    pipeline = episode_pipeline(tmp_path)
    frozen = pipeline.buffer
    monkeypatch.setattr(frozen, 'save', Mock(side_effect=OSError('disk full')))
    pipeline.stop_episode(True, background=True)
    future, target = pipeline.pending_episode_saves[0]
    with pytest.raises(OSError, match='disk full'):
        future.result(timeout=2.)
    pipeline.poll()
    assert 'episode 保存失败' in caplog.text and str(target) in caplog.text
    assert frozen.sample_count == 2 and frozen.camera_frames['camera']
    assert pipeline.failed_episode_saves == [(future, target)]
    assert not pipeline.stop_event.is_set() and pipeline.control_error is None
    pipeline.close()


def test_sampling_deadline_warning_does_not_stop_motion(tmp_path, monkeypatch, caplog):
    import gello_teleop.dual_gello_collect as collect
    pipeline = make_pipeline()
    class DelayedTicker:
        def __init__(self, rate, limit, *, strict):
            assert strict is False
            self.maximum_lateness_s = limit
            self.last_skipped_ticks = 4
        def wait(self, context):
            pipeline.sampling_stop.set()  # Capture one delayed scheduler step.
            return 1, .0438
    monkeypatch.setattr(collect, 'FixedRateTicker', DelayedTicker)
    pipeline._sampling_loop()
    assert pipeline.sampling_late_tick_count == 1 and pipeline.sampling_skipped_tick_count == 4
    assert '录制采样延迟 43.8 ms' in caplog.text
    assert pipeline.control_error is None and not pipeline.stop_event.is_set()
    pipeline.close()


def test_long_episode_stacks_in_bounded_batches_and_preserves_dtype_promotion(monkeypatch):
    import nero_collection.h5_writer as writer
    monkeypatch.setattr(writer, '_TELEOP_STACK_BATCH_SIZE', 16)
    values = [np.full(7, index, dtype=np.uint8) for index in range(31)]
    values += [np.full(7, 31.5, dtype=np.float64)]
    original = np.stack
    expected = original(values)
    def bounded(items, *args, **kwargs):
        assert len(items) <= 16
        return original(items, *args, **kwargs)
    monkeypatch.setattr(writer.np, 'stack', bounded)
    np.testing.assert_array_equal(_stack(values), expected)
