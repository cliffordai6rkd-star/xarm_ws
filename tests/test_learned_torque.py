from pathlib import Path
from unittest.mock import Mock
import queue

import numpy as np
import pytest

from xarm_stack.learned_torque import (
    DEFAULT_CHECKPOINT, DelayedTorqueStream, LearnedTorqueModel, learned_config, predict_episode,
)
from xarm_stack.torque_visualization import TorqueVisualizer, TorqueWindow, plot_config


def trajectory(duration=7.):
    t = np.arange(0., duration, .01)
    phase = t[:, None]*1.7+np.arange(7)[None, :]*.4
    q = .1*np.sin(phase)+np.array([0., -.4, 0., .6, 0., .4, -1.5])
    dq = .17*np.cos(phase)
    return dict(t=t, q=q, dq=dq, q_cmd=q+.005*np.cos(phase*.7),
                tau=np.sin(phase)+.3*np.sin(t[:, None]*2*np.pi*30), valid=np.ones(len(t), bool))


def packet(e, i):
    return dict(timestamp_s=e['t'][i], lowdim_timestamp_s=e['t'][i],
                q=e['q'][i], dq=e['dq'][i], q_cmd=e['q_cmd'][i], torque=e['tau'][i],
                q_valid=True, dq_valid=True, q_cmd_valid=True, torque_valid=True)


@pytest.fixture(scope='module')
def network():
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip('Export the local model artifact to run trained-model regressions')
    pytest.importorskip('torch')
    pytest.importorskip('pinocchio')
    return LearnedTorqueModel()


def settings():
    return dict(delay_s=1.5, buffer_s=4., interval_s=.1)


class FakePredictor:
    def predict(self, times, q, dq, q_cmd, tau, *, output_times=None, **kwargs):
        t = times if output_times is None else output_times
        return dict(timestamp_s=t, tau_free=np.zeros((len(t), 7)), tau_measured=np.ones((len(t), 7)),
                    tau_ext=np.tile(np.array([-1., 2., -3., 4., -5., 6., -7.]), (len(t), 1)),
                    valid_context=(t >= times[0]+1.) & (t <= times[-1]-1.))


def test_measurement_only_changes_residual_and_commands_are_network_inputs(network):
    e = trajectory(4.)
    r = predict_episode(network, e)
    changed = predict_episode(network, {**e, 'tau': e['tau']+3.})
    np.testing.assert_array_equal(changed['tau_free'], r['tau_free'])
    np.testing.assert_allclose(changed['tau_ext']-r['tau_ext'], 3., atol=2e-6)
    command = predict_episode(network, {**e, 'q_cmd': e['q_cmd']+.02})
    assert np.max(np.abs(command['tau_free']-r['tau_free'])) > .01


def test_buffered_prediction_matches_whole_record_with_source_timestamps(network):
    e = trajectory()
    # Exercise epoch timestamps, irregular recorder periods, and rolling crops.
    e['t'] = 1_700_000_000.+e['t']+.0004*np.sin(e['t']*13)
    full = predict_episode(network, e)
    stream = DelayedTorqueStream(network, settings())
    for i, t in enumerate(e['t']):
        stream.append(packet(e, i), t)
        if i % 10 == 0 or i == len(e['t'])-1:
            stream.update(t)
    relative_t, ext = stream.arrays(e['t'][-1])
    source_t = relative_t+e['t'][-1]
    ix = np.searchsorted(e['t'], source_t)
    np.testing.assert_array_equal(source_t, e['t'][ix])
    assert np.all(np.diff(source_t) > 0)
    assert source_t[0] >= e['t'][0]+1.
    assert -relative_t[-1] >= 1.5
    np.testing.assert_allclose(ext, full['tau_ext'][ix], atol=1e-4)
    assert len(stream.inputs) < 410


def test_invalid_command_gap_stale_and_clock_reset_require_new_context():
    e = trajectory(8.)
    stream = DelayedTorqueStream(FakePredictor(), settings())
    for i in range(301):
        stream.append(packet(e, i), e['t'][i])
    stream.update(3.)
    assert np.isfinite(stream.arrays(3.)[1][-1]).all()
    sample = packet(e, 301)
    sample['q_cmd_valid'] = False
    stream.append(sample, 3.01)
    assert np.isnan(stream.arrays(3.01)[1][-1]).all()
    for i in range(302, 552):
        stream.append(packet(e, i), e['t'][i])
    stream.update(5.51)
    assert stream.emitted is None
    stream.append(packet(e, 552), e['t'][552])
    stream.update(e['t'][552])
    assert stream.emitted is not None
    stream.update(5.8)
    assert not stream.inputs and np.isnan(stream.arrays(5.8)[1][-1]).all()
    stream.append(packet(e, 580), 5.8)
    stream.append(packet(e, 590), 5.9)
    assert len(stream.inputs) == 1 and stream.origin == 5.9
    stream.append(packet(e, 100), 1.)
    assert len(stream.rows) == 0 and stream.origin == 1.


def test_reset_during_inference_discards_old_result():
    e = trajectory(3.)
    stream = DelayedTorqueStream(FakePredictor(), settings())
    for i, t in enumerate(e['t']):
        stream.append(packet(e, i), t)
    original = stream.model.predict

    def predict(*args, **kwargs):
        sample = packet(e, -1)
        sample['torque_valid'] = False
        stream.append(sample, e['t'][-1])
        return original(*args, **kwargs)

    stream.model.predict = predict
    stream.update(e['t'][-1])
    assert not stream.rows and stream.emitted is None


def test_repeated_robot_report_uses_distinct_recorder_rows():
    e = trajectory(1.)
    stream = DelayedTorqueStream(FakePredictor(), settings())
    first, second = packet(e, 0), packet(e, 1)
    second['timestamp_s'] = first['timestamp_s']
    stream.append(first, 0.)
    stream.append(second, .01)
    stream.append(second, .01)
    assert [row[0] for row in stream.inputs] == [0., .01]


def test_slow_plot_inference_marks_discarded_buffer_context_as_gap():
    e = trajectory(9.)
    stream = DelayedTorqueStream(FakePredictor(), settings())
    for i, t in enumerate(e['t']):
        stream.append(packet(e, i), t)
        if i == 300 or i == len(e['t'])-1:
            stream.update(t)
    times, residual = stream.arrays(e['t'][-1])
    gap = np.flatnonzero(np.isnan(residual).all(axis=1))
    assert len(gap) == 1 and 0 < gap[0] < len(times)-1


def test_publisher_copies_command_and_keeps_recorder_time_separate():
    from nero_collection.arms.base import ArmState
    visualizer = TorqueVisualizer({})
    visualizer.process = Mock()
    visualizer.process.is_alive.return_value = True
    visualizer.samples = queue.Queue()
    state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.eye(4), np.zeros(7), np.zeros(7), 1_000_000)
    command = np.arange(7, dtype=float)
    visualizer.publish(['right'], [state], q_cmd=command, q_cmd_valid=[True], lowdim_timestamp_us=1_004_000)
    command[:] = -99
    sample = visualizer.samples.get_nowait()['right']
    np.testing.assert_array_equal(sample['q_cmd'], np.arange(7))
    assert sample['lowdim_timestamp_s'] == 1.004 and sample['timestamp_s'] == 1.
    assert sample['q_cmd_valid']


@pytest.mark.parametrize('source', [None, 'urdf'])
def test_learned_plot_preserves_urdf_and_shows_correct_norms_and_delayed_time(monkeypatch, tmp_path, source):
    import xarm_stack.learned_torque as module
    monkeypatch.setattr(module, 'LearnedTorqueModel', lambda *args: FakePredictor())
    dummy = tmp_path/'model.pt'
    dummy.touch()
    cfg = plot_config(dict(learned_torque=dict(enabled=True, checkpoint=str(dummy))), __file__, ['right'])
    if source:
        cfg['force_feedback_source'] = source
    physics = Mock()
    physics.rnea.return_value = np.ones(7)
    physics.gravity_torque.return_value = np.ones(7)
    window = TorqueWindow(cfg, {'right': physics}, headless=True)
    e = trajectory(3.)
    try:
        for i, t in enumerate(e['t']):
            sample = packet(e, i)
            sample.update(ddq=np.zeros(7), ddq_valid=True,
                          urdf_feedback=dict(timestamp_s=t, tau_ext=np.ones(7), valid=True))
            window.ingest({'right': sample}, t)
        window.update_learned(e['t'][-1])
        window.draw(e['t'][-1])
        assert window.axes.shape == (8, 2)
        np.testing.assert_array_equal(window.learned_lines['right', 'l1'].get_ydata(), 28.)
        np.testing.assert_allclose(window.learned_lines['right', 'l2'].get_ydata(), np.sqrt(140))
        assert window.learned_lines['right', 'l1'].get_xdata()[-1] <= -1.5
        if source == 'urdf':
            assert window.lines[0, 0].get_ydata()[-1] == 7.
            physics.rnea.assert_not_called()
        else:
            assert window.lines[0, 1].get_ydata().size == len(e['t'])
        window.update_learned(e['t'][-1]+.2)
        window.draw(e['t'][-1]+.2)
        assert np.isnan(window.learned_lines['right', 'l1'].get_ydata()[-1])
    finally:
        window.close()


def test_offline_segmentation_does_not_filter_across_invalid_rows():
    e = trajectory(8.)
    e['valid'][350] = False
    e['q'][350] = np.nan
    e['t'][600:] += .1
    model = Mock(wraps=FakePredictor())
    result = predict_episode(model, e)
    assert model.predict.call_count == 2  # Last segment is shorter than 2 seconds.
    assert not result['valid_context'][350:451].any()
    assert not result['valid_context'][600:].any()
    assert np.isnan(result['tau_ext'][350]).all()
    for call in model.predict.call_args_list:
        assert np.diff(call.args[0]).max() < .03
        assert all(np.isfinite(a).all() for a in call.args)


@pytest.mark.parametrize('override', [{'delay_s': .5}, {'buffer_s': 2.}, {'interval_s': 0.},
                                    {'thresholds_l1': [10., 5.]}, {'threads': 0}])
def test_learned_config_rejects_invalid_context_and_thresholds(tmp_path, override):
    path = tmp_path/'model.pt'
    path.touch()
    with pytest.raises(ValueError):
        learned_config(dict(enabled=True, checkpoint=str(path), **override), __file__, ['right'])
