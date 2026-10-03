"""Episode confirmation keeps robot control active and preserves the stop boundary."""
from contextlib import nullcontext
from types import SimpleNamespace
import threading
import time
from unittest.mock import Mock

import pytest

import gello_teleop.dual_gello_collect as collect
from gello_teleop.dual_gello_collect import DualGelloPipeline
from test_dual_gello_pipeline import make_pipeline
from test_episode_background_save import episode_pipeline


def keyboard_pipeline(monkeypatch, sequence):
    pipeline = DualGelloPipeline.__new__(DualGelloPipeline)
    pipeline.state = 'following'
    pipeline.recording = False
    pipeline.buffer = None
    pipeline.episode_confirmation_pending = False
    pipeline.episode_confirmation_buffer = None
    for name in ('reset', 'wait_for_alignment', 'takeover', 'poll',
                 '_save_frozen_episode', '_discard_frozen_episode'):
        setattr(pipeline, name, Mock())

    def start():
        pipeline.recording = True
        pipeline.state = 'recording'
        pipeline.buffer = object()

    def stop(save, index):
        assert save is False and pipeline.recording
        pipeline.recording = False
        pipeline.state = 'following'

    pipeline.start_episode = Mock(side_effect=start)
    pipeline.stop_episode = Mock(side_effect=stop)
    keys = Mock(is_tty=True)
    keys.read_key.side_effect = sequence
    monkeypatch.setattr(collect, 'TerminalKeys', lambda: nullcontext(keys))
    monkeypatch.setattr('builtins.input', Mock(side_effect=AssertionError('prompt must not block')))
    return pipeline, keys


@pytest.mark.parametrize('stop_key', [' ', '\r', '\n'])
@pytest.mark.parametrize('choice', ['y', 'Y', 'n', 'N'])
def test_confirmation_stops_at_enter_and_polls_until_explicit_choice(monkeypatch, stop_key, choice):
    sequence = ['r', stop_key, None, 'R', stop_key, '?', choice, 'q']
    pipeline, keys = keyboard_pipeline(monkeypatch, sequence)
    observed = []

    def poll():
        observed.append((pipeline.recording, pipeline.episode_confirmation_pending,
                         pipeline.episode_confirmation_buffer))

    pipeline.poll.side_effect = poll
    assert pipeline.interactive() == 0
    pipeline.start_episode.assert_called_once()
    pipeline.stop_episode.assert_called_once_with(False, None)
    assert pipeline.poll.call_count == len(sequence)
    pending = [row for row in observed if row[1]]
    assert len(pending) == 5 and all(not row[0] for row in pending)
    frozen = pending[0][2]
    assert frozen is not None and all(row[2] is frozen for row in pending)
    if choice.lower() == 'y':
        pipeline._save_frozen_episode.assert_called_once_with(frozen, background=True)
        pipeline._discard_frozen_episode.assert_not_called()
    else:
        pipeline._save_frozen_episode.assert_not_called()
        pipeline._discard_frozen_episode.assert_called_once_with(frozen)
    assert not pipeline.episode_confirmation_pending
    assert pipeline.episode_confirmation_buffer is None


def test_pending_confirmation_keeps_hold_reset_and_takeover_keys_available(monkeypatch):
    pipeline, keys = keyboard_pipeline(monkeypatch, ['r', ' ', 'F', 'o', 't', 'Y', 'q'])

    def hold(*args):
        assert pipeline.episode_confirmation_pending and not pipeline.recording
        pipeline.state = 'holding'

    def takeover(*args):
        assert pipeline.episode_confirmation_pending and not pipeline.recording
        pipeline.state = 'following'

    pipeline.freeze_following = Mock(side_effect=hold)
    pipeline.reset_and_hold = Mock(side_effect=hold)
    pipeline.realign_and_takeover = Mock(side_effect=takeover)
    assert pipeline.interactive() == 0
    pipeline.freeze_following.assert_called_once()
    pipeline.reset_and_hold.assert_called_once_with(keys)
    pipeline.realign_and_takeover.assert_called_once_with(keys)
    pipeline._save_frozen_episode.assert_called_once()
    pipeline.stop_episode.assert_called_once_with(False, None)


@pytest.mark.parametrize('exit_key', ['q', 'Q', '\x03'])
def test_exit_cancels_unconfirmed_episode_without_saving(monkeypatch, exit_key):
    pipeline, _ = keyboard_pipeline(monkeypatch, ['r', '\r', exit_key])
    assert pipeline.interactive() == 0
    pipeline._save_frozen_episode.assert_not_called()
    discarded, = pipeline._discard_frozen_episode.call_args.args
    assert discarded is not None
    assert not pipeline.recording and not pipeline.episode_confirmation_pending
    assert pipeline.episode_confirmation_buffer is None


def test_device_error_still_propagates_while_waiting_for_confirmation(monkeypatch):
    pipeline, _ = keyboard_pipeline(monkeypatch, ['r', ' '])

    def poll():
        if pipeline.episode_confirmation_pending:
            raise RuntimeError('synthetic disconnected robot')

    pipeline.poll.side_effect = poll
    with pytest.raises(RuntimeError, match='synthetic disconnected robot'):
        pipeline.interactive()
    pipeline._save_frozen_episode.assert_not_called()
    assert pipeline._discard_frozen_episode.call_args.args[0] is not None
    assert not pipeline.episode_confirmation_pending
    assert pipeline.episode_confirmation_buffer is None


@pytest.mark.parametrize('choice', ['y', 'n', 'q', '\x03'])
def test_transition_key_reads_confirmation_and_preserves_quit(monkeypatch, choice):
    pipeline, _ = keyboard_pipeline(monkeypatch, [])
    pipeline._recording_key('r')
    pipeline._recording_key(' ')
    keys = SimpleNamespace(read_key=Mock(return_value=choice))
    assert pipeline._transition_key(keys, .01) == (choice if choice in {'q', '\x03'} else None)
    if choice == 'y':
        pipeline._save_frozen_episode.assert_called_once()
    elif choice == 'n':
        assert pipeline._discard_frozen_episode.call_args.args[0] is not None


def test_confirmation_freezes_real_episode_while_simulated_robot_keeps_servoing():
    pipeline = make_pipeline()
    try:
        pipeline.connect()
        pipeline.reset()
        pipeline.confirm_alignment()
        pipeline.takeover()
        pipeline.start_episode()
        time.sleep(.04)
        pipeline.poll()
        assert pipeline._recording_key(' ')
        frozen = pipeline.episode_confirmation_buffer
        assert frozen.sample_count > 0
        count = frozen.sample_count
        commands = [len(arm.commands) for arm in pipeline.arms]
        deadline = time.monotonic() + 1.
        while not all(len(arm.commands) >= before + 3 for arm, before in zip(pipeline.arms, commands)):
            assert time.monotonic() < deadline
            time.sleep(.01)
            pipeline.poll()
        assert frozen.sample_count == count
        assert pipeline.buffer is None and not pipeline.recording
        assert pipeline.state == 'following' and not pipeline.follow_stop.is_set()
        assert pipeline.episode_confirmation_pending
        assert pipeline._recording_key('N')
        for future in pipeline.pending_episode_discards:
            future.result(timeout=2.)
        assert not pipeline.episode_confirmation_pending
        assert pipeline.state == 'following'
    finally:
        pipeline.close()


def test_discard_detaches_data_without_h5_index_or_blocking_poll(tmp_path, monkeypatch):
    pipeline = episode_pipeline(tmp_path)
    frozen = pipeline.buffer
    entered, release = threading.Event(), threading.Event()
    original_release = collect._release_episode_buffer

    def blocked_release(buffer):
        entered.set()
        assert release.wait(3.)
        original_release(buffer)

    monkeypatch.setattr(collect, '_release_episode_buffer', blocked_release)
    monkeypatch.setattr(collect, 'next_episode_index', Mock(side_effect=AssertionError('discard must not allocate a path')))
    monkeypatch.setattr(collect, 'episode_path', Mock(side_effect=AssertionError('discard must not allocate a path')))
    pipeline.episode_save_process.save = Mock(side_effect=AssertionError('discard must not write H5'))
    try:
        pipeline._recording_key('\n')
        assert pipeline.buffer is None and pipeline.episode_confirmation_buffer is frozen
        pipeline._recording_key('n')
        assert entered.wait(1.)
        assert not pipeline.recording and pipeline.buffer is None
        assert pipeline.episode_confirmation_buffer is None
        assert pipeline._next_episode_save_index == 0 and not pipeline.pending_episode_saves
        pipeline.poll()
        assert frozen.sample_count == 2  # release remains blocked in its worker
        assert not pipeline.stop_event.is_set() and pipeline.state == 'following'
        pipeline.episode_save_process.save.assert_not_called()
        assert not list(tmp_path.glob('*.h5'))
        release.set()
        pipeline.pending_episode_discards[0].result(timeout=2.)
        assert frozen.sample_count == 0 and not frozen.teleop_data and not frozen.camera_frames
        pipeline.poll()
        assert not pipeline.pending_episode_discards
    finally:
        release.set()
        pipeline.close()


def test_programmatic_stop_without_save_keeps_buffer_available(tmp_path):
    pipeline = episode_pipeline(tmp_path)
    frozen = pipeline.buffer
    try:
        pipeline.stop_episode(False)
        assert pipeline.buffer is frozen and frozen.sample_count == 2
        assert not pipeline.pending_episode_discards
        assert not pipeline.episode_confirmation_pending
    finally:
        pipeline.close()
