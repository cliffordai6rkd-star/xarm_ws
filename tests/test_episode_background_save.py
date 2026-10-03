"""Saving preserves takeover, frozen episodes and the configured motion deadline policy."""
from dataclasses import replace
import json
import os
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


def test_strict_control_deadline_remains_fatal_for_the_same_delay():
    clock = Clock()
    ticker = FixedRateTicker(100, .03, monotonic=clock.monotonic, sleep=clock.sleep)
    ticker.wait('control')
    clock.now = .0538
    with pytest.raises(RuntimeError, match='missed deadline.*control'):
        ticker.wait('control')


@pytest.mark.parametrize('policy', [None, 'strict', 'skip', 'unknown', {}])
def test_control_deadline_policy_configuration(tmp_path, policy):
    import yaml
    from gello_teleop.dual_gello_collect import DualGelloPipeline
    from scripts.smoke_dual_gello_pipeline import ROOT, write_simulation_config
    path = write_simulation_config(ROOT/'gello_teleop/config/xarm7_gello_dual_dataset.yaml', tmp_path)
    data = yaml.safe_load(path.read_text())
    if policy is None:
        data['control'].pop('deadline_policy', None)
    else:
        data['control']['deadline_policy'] = policy
    path.write_text(yaml.safe_dump(data))
    if policy is not None and policy not in ('strict', 'skip'):
        with pytest.raises(ValueError, match='control.deadline_policy'):
            DualGelloPipeline(path)
    else:
        pipeline = DualGelloPipeline(path)
        assert pipeline.control_deadline_policy == (policy or 'strict')


@pytest.mark.parametrize('policy, stall_s, expected_ticks, error_text', [
    ('skip', .0669, 3, None),
    ('strict', .0669, 1, 'missed deadline'),
    ('skip', .22, 1, 'GELLO target expired'),
])
def test_control_skips_delayed_commands_but_stops_for_expired_targets(
        monkeypatch, policy, stall_s, expected_ticks, error_text):
    from types import SimpleNamespace
    import gello_teleop.dual_gello_collect as collect
    pipeline = make_pipeline()
    pipeline.control_deadline_policy = policy
    pipeline.state = 'following'
    clock = Clock()
    clock.now = 10.
    pipeline._last_control_t = clock.now
    leader = SimpleNamespace(acquired_monotonic_s=clock.now, mapped=np.zeros(7))
    state = SimpleNamespace(acquired_timestamp_us=int(clock.now*1e6), q=np.zeros(7))
    pipeline.producers = [SimpleNamespace(snapshot=lambda: leader) for _ in pipeline.ARM_NAMES]
    pipeline.state_producers = [SimpleNamespace(snapshot=lambda: state) for _ in pipeline.ARM_NAMES]
    pipeline.arms = [Mock() for _ in pipeline.ARM_NAMES]
    pipeline.followers = [Mock(update=Mock(return_value=np.zeros(7))) for _ in pipeline.ARM_NAMES]
    monkeypatch.setattr(pipeline, 'leader_sample_status', lambda: {'age_s': clock.now-10.})
    monkeypatch.setattr(collect.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(collect, 'now_us', lambda: int(clock.now*1e6))

    class DelayedTicker(FixedRateTicker):
        def __init__(self, rate, limit, *, strict):
            super().__init__(rate, limit, strict=strict,
                             monotonic=clock.monotonic, sleep=clock.sleep)
            self.calls = 0

        def wait(self, context):
            if self.calls == 1:
                clock.now += stall_s
            if self.calls == 3:
                pipeline.follow_stop.set()
            self.calls += 1
            return super().wait(context)

    monkeypatch.setattr(collect, 'FixedRateTicker', DelayedTicker)
    pipeline._control_loop()
    assert pipeline.control_tick_count == expected_ticks + int(error_text == 'GELLO target expired')
    for arm in pipeline.arms:
        assert arm.command_joint_positions.call_count == expected_ticks
    if error_text is not None:
        assert error_text in str(pipeline.control_error)
        assert pipeline.stop_event.is_set()
    else:
        assert pipeline.control_error is None and not pipeline.stop_event.is_set()
        assert pipeline.control_late_tick_count == 1
        assert pipeline.control_skipped_tick_count == 5
        assert pipeline.control_max_lateness_s == pytest.approx(.0569)
        for follower in pipeline.followers:
            assert [call.args[2] for call in follower.update.call_args_list] == pytest.approx([.0001, .0669, .0031])
        for history in pipeline.command_history:
            assert [item[0] for item in history] == pytest.approx([10_000_000, 10_066_900, 10_070_000], abs=1)


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
    pipeline.buffer.append_camera('camera', 1000000, np.full((8, 8, 3), 7, np.uint8),
                                  depth=np.full((8, 8), 1234, np.uint16))
    return pipeline


def test_background_save_freezes_episode_and_close_waits_after_device_shutdown(tmp_path, monkeypatch, caplog):
    pipeline = episode_pipeline(tmp_path)
    entered, release = threading.Event(), threading.Event()
    frozen = pipeline.buffer
    original_save = pipeline.episode_save_process.save
    def blocked_save(buffer, target):
        entered.set()
        assert release.wait(3.)
        return original_save(buffer, target)
    monkeypatch.setattr(pipeline.episode_save_process, 'save', blocked_save)
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
        assert finished.wait(10.)
        with h5py.File(target) as episode:
            assert len(episode['teleop/timestamp_us']) == 2
            for arm in ('left', 'right'):
                np.testing.assert_array_equal(episode[f'teleop/{arm}_q_xarm'][:],
                                              np.array([np.zeros(7), np.ones(7)]))
            assert episode['cameras/camera/frames'].shape == (1, 8, 8, 3)
            np.testing.assert_array_equal(episode['cameras/camera/depth'][0], np.full((8, 8), 1234, np.uint16))
            metadata = json.loads(episode['metadata/episode_json'][()])
            assert metadata['writer_process_id'] != os.getpid()
            assert metadata['collector_process_id'] == os.getpid()
        assert pipeline.episode_save_executor is None and not pipeline.pending_episode_saves
        assert pipeline.episode_save_process.process is None
    finally:
        release.set()
        closer.join(timeout=10.)


def test_two_queued_episodes_reserve_distinct_paths_and_never_mix_rows(tmp_path, monkeypatch):
    pipeline = episode_pipeline(tmp_path)
    release = threading.Event()
    monkeypatch.setattr(pipeline, '_consume_samples', lambda: None)
    first = pipeline.buffer
    save = pipeline.episode_save_process.save
    def blocked(buffer, target):
        if buffer is first:
            assert release.wait(3.)
        return save(buffer, target)
    monkeypatch.setattr(pipeline.episode_save_process, 'save', blocked)
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
            for arm in ('left', 'right'):
                np.testing.assert_array_equal(second_h5[f'teleop/{arm}_q_xarm'][0], np.full(7, 9.))
            first_meta = json.loads(first_h5['metadata/episode_json'][()])
            second_meta = json.loads(second_h5['metadata/episode_json'][()])
            assert first_meta['writer_process_id'] == second_meta['writer_process_id'] != os.getpid()
    finally:
        release.set()
        pipeline.close()


def test_failed_background_write_is_reported_and_retains_frozen_data(tmp_path, monkeypatch, caplog):
    pipeline = episode_pipeline(tmp_path)
    frozen = pipeline.buffer
    monkeypatch.setattr(pipeline.episode_save_process, 'save', Mock(side_effect=OSError('disk full')))
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


def test_transfer_packets_are_bounded_even_when_dtype_changes():
    from nero_collection.episode_saver import EpisodeSaveProcess, _TRANSFER_BYTES, _TRANSFER_ROWS
    writer = EpisodeSaveProcess()
    writer.connection = Mock()
    values = [np.zeros(100, np.uint8)] + [np.full(100, index, np.float64) for index in range(1, 1500)]
    writer._send_values('rows', 'q_follower', values)
    packets = [call.args[0] for call in writer.connection.send.call_args_list]
    assert all(packet[2].nbytes <= _TRANSFER_BYTES and len(packet[2]) <= _TRANSFER_ROWS for packet in packets)
    np.testing.assert_array_equal(np.concatenate([packet[2] for packet in packets]), np.stack(values))


def test_worker_reports_disk_error_and_can_save_next_episode(tmp_path):
    pipeline = episode_pipeline(tmp_path)
    writer = pipeline.episode_save_process
    blocked_directory = tmp_path/'file_not_directory'
    blocked_directory.write_text('existing file')
    try:
        writer.start()
        pid = writer.pid
        with pytest.raises(RuntimeError, match='独立 H5 保存失败'):
            writer.save(pipeline.buffer, blocked_directory/'failed.h5')
        assert writer.process.is_alive() and writer.pid == pid
        assert pipeline.buffer.sample_count == 2
        target = writer.save(pipeline.buffer, tmp_path/'recovered.h5')
        with h5py.File(target) as episode:
            assert len(episode['teleop/timestamp_us']) == 2
            metadata = json.loads(episode['metadata/episode_json'][()])
            assert metadata['writer_process_id'] == pid != os.getpid()
    finally:
        pipeline.close()


def test_worker_preserves_independent_camera_and_lowdim_timestamps(tmp_path):
    pipeline = episode_pipeline(tmp_path)
    buffer = pipeline.buffer
    # 100 Hz lowdim; camera samples have their own ~30 Hz timestamps.
    for index in range(2, 11):
        buffer.append_teleop(1000000+index*10000, {'q_follower': ('q', np.full(14, index, float))})
    for index in range(1, 4):
        buffer.append_camera('camera', 1000000+index*33333,
            np.full((8, 8, 3), index, np.uint8), depth=np.full((8, 8), 1234+index, np.uint16))
    try:
        target = pipeline.stop_episode(True, background=True)
        pipeline.close()
        with h5py.File(target) as episode:
            np.testing.assert_array_equal(episode['teleop/timestamp_us'][:], 1000000+np.arange(11)*10000)
            camera = episode['cameras/camera']
            np.testing.assert_array_equal(camera['timestamp_us'][:], 1000000+np.arange(4)*33333)
            assert camera['frames'].shape == (4, 8, 8, 3) and camera['frames'].dtype == np.uint8
            assert camera['depth'].dtype == np.uint16 and camera['frames'].compression is None
    finally:
        pipeline.close()
