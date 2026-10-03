from types import SimpleNamespace
import queue
from unittest.mock import Mock

import numpy as np
import pytest

from nero_collection.arms.base import ArmState
from xarm_stack.torque_visualization import TorqueHistory, TorqueVisualizer, TorqueWindow


class ConstantTorqueModel:
    def gravity_torque(self, q):
        return np.full(7, 3.)

    def rnea(self, q, dq, ddq):
        return 2*ddq+3.


def sample(timestamp=10., **overrides):
    return dict(timestamp_s=timestamp, q=np.zeros(7), dq=np.zeros(7), ddq=np.zeros(7),
                q_valid=True, dq_valid=True, ddq_valid=True,
                torque=np.full(7, 9.), torque_valid=True) | overrides


def result(timestamp=10., torque=2., valid=True):
    return dict(timestamp_s=timestamp, tau_ext=np.full(7, torque), valid=valid)


def plot_config(source=None):
    cfg = dict(active_sides=('right',), stale_s=.15, window_s=30., update_hz=10.,
               sides={'right': dict(measured_torque_cutoff_hz=3., acceleration_cutoff_hz=3.)})
    if source is not None:
        cfg.update(force_feedback_source=source, force_feedback_mean_windows={'right': 5})
    return cfg


def test_publish_urdf_result_keeps_source_time_and_copies_mean_without_network_packet():
    visualizer = TorqueVisualizer(plot_config('urdf'))
    visualizer.process = Mock()
    visualizer.process.is_alive.return_value = True
    visualizer.samples = queue.Queue()
    state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.eye(4),
                     np.ones(7), np.zeros(7), 10_100_000)
    mean = np.arange(7, dtype=float)
    calculated = SimpleNamespace(source_timestamp_us=10_050_000, valid=True, tau_ext=mean)
    predicted = SimpleNamespace(source_timestamp_us=10_020_000, valid=True,
                                tau_measured=np.ones(7), tau_pred=np.zeros(7))
    visualizer.publish(('right',), (state,), urdf_results=(calculated,), tau_free_results=(predicted,))
    mean[:] = 100.
    packet = visualizer.samples.get_nowait()['right']
    assert packet['timestamp_s'] == 10.1
    assert packet['urdf_feedback']['timestamp_s'] == 10.05
    assert packet['urdf_feedback']['valid']
    np.testing.assert_array_equal(packet['urdf_feedback']['tau_ext'], np.arange(7))
    assert 'tau_free' not in packet


def test_urdf_history_waits_for_complete_mean_and_uses_result_on_repeated_robot_report():
    history = TorqueHistory(ConstantTorqueModel(), .15, 30., force_feedback_source='urdf')
    history.ingest(sample(urdf_feedback=result(valid=False)), 10.)
    assert history.urdf_feedback_arrays(10.)[1].shape == (0, 7)
    # Mean arrives later while this robot state is still repeated. It is not
    # recomputed from 9-3=6, nor smoothed a second time by the plot.
    history.ingest(sample(urdf_feedback=result(torque=2.5),
                          tau_free=dict(timestamp_s=10., valid=True,
                                        tau_measured=np.ones(7), tau_pred=np.zeros(7))), 10.01)
    times, residual = history.urdf_feedback_arrays(10.01)
    np.testing.assert_allclose(times, [-.01])
    np.testing.assert_array_equal(residual, np.full((1, 7), 2.5))
    assert history.prediction_arrays(10.01)[1].shape == (0, 7)
    history.ingest(sample(urdf_feedback=result(torque=2.5)), 10.02)
    assert len(history.urdf_feedback_rows) == 1


@pytest.mark.parametrize('invalid', [None, result(valid=False), result(torque=float('nan')),
                                   result(timestamp=9.), result(timestamp=11.),
                                   dict(timestamp_s=10., tau_ext=[1., 2.], valid=True)])
def test_invalid_urdf_result_creates_gap_and_recovery_keeps_actual_source_timestamp(invalid):
    history = TorqueHistory(ConstantTorqueModel(), .15, 30., force_feedback_source='urdf')
    history.ingest(sample(urdf_feedback=result()), 10.)
    history.ingest(sample(10.01, urdf_feedback=invalid), 10.01)
    assert np.isnan(history.urdf_feedback_arrays(10.01)[1][-1]).all()
    assert np.isnan(history.urdf_feedback_l1_arrays(10.01)[1][-1])
    history.ingest(sample(10.02, urdf_feedback=result(10.015, torque=-4.)), 10.02)
    times, residual = history.urdf_feedback_arrays(10.02)
    np.testing.assert_allclose(times[-1], -.005)
    np.testing.assert_array_equal(residual[-1], np.full(7, -4.))


def test_urdf_source_age_cannot_be_refreshed_by_repeated_publication():
    history = TorqueHistory(ConstantTorqueModel(), .15, 30., force_feedback_source='urdf')
    history.ingest(sample(urdf_feedback=result()), 10.)
    history.ingest(sample(10.14, urdf_feedback=result()), 10.14)
    assert len(history.urdf_feedback_rows) == 1
    assert np.isnan(history.urdf_feedback_arrays(10.151)[1][-1]).all()
    history.ingest(sample(10.2, urdf_feedback=result()), 10.2)
    assert len(history.urdf_feedback_rows) == 2


def test_repeated_urdf_result_survives_a_display_window_shorter_than_its_age():
    history = TorqueHistory(ConstantTorqueModel(), .15, .01, force_feedback_source='urdf')
    history.ingest(sample(urdf_feedback=result()), 10.)
    assert history.urdf_feedback_arrays(10.02)[0].size == 0
    history.ingest(sample(10.02, urdf_feedback=result()), 10.02)
    assert history.urdf_feedback_arrays(10.02)[0].size == 0


def test_urdf_history_ignores_older_result_and_restarts_after_clock_reversal():
    history = TorqueHistory(ConstantTorqueModel(), .15, 1., force_feedback_source='urdf')
    history.ingest(sample(10.1, urdf_feedback=result(10.1)), 10.1)
    history.ingest(sample(10.11, urdf_feedback=result(10.05, torque=100.)), 10.11)
    np.testing.assert_array_equal(history.urdf_feedback_arrays(10.11)[1][-1], np.full(7, 2.))
    history.ingest(sample(9., urdf_feedback=result(9., torque=4.)), 9.)
    times, residual = history.urdf_feedback_arrays(9.)
    np.testing.assert_array_equal(times, [0.])
    np.testing.assert_array_equal(residual, np.full((1, 7), 4.))
    times, residual = history.urdf_feedback_arrays(10.2)
    assert len(times) == 1 and np.isnan(residual).all()


def test_urdf_window_draws_mean_l1_and_seven_joints_without_recomputing_dynamics():
    model = Mock(spec=ConstantTorqueModel)
    window = TorqueWindow(plot_config('urdf'), {'right': model}, headless=True)
    try:
        assert window.axes.shape == (8, 1)
        assert set(window.lines) == {(row, 0) for row in range(8)}
        window.ingest({'right': sample(urdf_feedback=result(valid=False))}, 10.)
        window.draw(10.)
        assert all(window.labels[row, 0].get_text() == '-- Nm' for row in range(8))
        residual = np.array([-1., 2., -3., 4., -5., 6., -7.])
        window.ingest({'right': sample(10.01, urdf_feedback=result(10.005, torque=residual))}, 10.01)
        window.draw(10.01)
        assert window.labels[0, 0].get_text() == '28.000 Nm'
        assert '||tau_ext||₁ (URDF, mean 5 samples)' in window.axes[0, 0].get_title()
        assert 'tau_ext_l1' in window.axes[0, 0].get_title()
        assert window.axes[0, 0].get_ylim()[0] == 0.
        np.testing.assert_allclose(window.lines[0, 0].get_xdata(), [-.005])
        np.testing.assert_array_equal(window.lines[0, 0].get_ydata(), [28.])
        for joint, torque in enumerate(residual):
            row = joint + 1
            np.testing.assert_allclose(window.lines[row, 0].get_xdata(), [-.005])
            np.testing.assert_array_equal(window.lines[row, 0].get_ydata(), [torque])
            assert window.labels[row, 0].get_text() == f'{torque:+.3f} Nm'
            assert f'J{row} tau_ext' in window.axes[row, 0].get_title()
            low, high = window.axes[row, 0].get_ylim()
            assert low < 0 < high
        model.rnea.assert_not_called()
        model.gravity_torque.assert_not_called()
        window.draw(10.2)
        for row in range(8):
            assert window.labels[row, 0].get_text() == '-- Nm'
            assert np.isnan(window.lines[row, 0].get_ydata()[-1])
    finally:
        window.close()


@pytest.mark.parametrize('sides', [('left',), ('right',), ('left', 'right'), ('right', 'left')])
def test_urdf_layout_keeps_active_arm_norms_and_joint_traces_separate(tmp_path, sides):
    cfg = plot_config('urdf')
    cfg['active_sides'] = sides
    cfg['sides'] = {side: cfg['sides']['right'].copy() for side in sides}
    cfg['force_feedback_mean_windows'] = {side: 5 + 4 * index for index, side in enumerate(sides)}
    models = {side: Mock(spec=ConstantTorqueModel) for side in sides}
    window = TorqueWindow(cfg, models, headless=True)
    residuals = {'left': np.array([-1., 2., -3., 4., -5., 6., -7.]),
                 'right': np.array([.5, -.25, 0., 1.5, -2.25, 3.75, -4.5])}
    try:
        window.ingest({side: sample(urdf_feedback=result(torque=residuals[side]))
                       for side in ('left', 'right')}, 10.)
        window.draw(10.)
        assert window.axes.shape == (8, len(sides))
        assert len(window.lines) == 8 * len(sides)
        assert set(window.history) == set(sides)
        for index, side in enumerate(sides):
            assert side.title() in window.axes[0, index].get_title()
            assert f'mean {5 + 4 * index} samples' in window.axes[0, index].get_title()
            np.testing.assert_array_equal(window.lines[0, index].get_ydata(), [np.abs(residuals[side]).sum()])
            for joint in range(7):
                np.testing.assert_array_equal(window.lines[joint + 1, index].get_ydata(), [residuals[side][joint]])
                assert side.title() in window.axes[joint + 1, index].get_title()
            models[side].rnea.assert_not_called()
            models[side].gravity_torque.assert_not_called()
        for inactive in {'left', 'right'} - set(sides):
            assert not any(inactive.title() in axis.get_title() for axis in window.axes.flat)
        window.save(tmp_path/'norm_and_joints.png')
        assert (tmp_path/'norm_and_joints.png').stat().st_size > 10000
    finally:
        window.close()


@pytest.mark.parametrize('source', [None, 'checkpoint'])
def test_default_and_checkpoint_windows_keep_independent_calculated_and_network_columns(source):
    window = TorqueWindow(plot_config(source), {'right': ConstantTorqueModel()}, headless=True)
    try:
        window.ingest({'right': sample(tau_free=dict(timestamp_s=10., valid=True,
                                                     tau_measured=np.full(7, 5.),
                                                     tau_pred=np.full(7, 1.)),
                                       urdf_feedback=result(torque=-100.))}, 10.)
        window.draw(10.)
        assert window.axes.shape == (7, 2)
        assert window.labels[0, 0].get_text() == '+4.000 Nm'
        assert window.labels[0, 1].get_text() == '+6.000 Nm'
        assert 'mean' not in window.axes[0, 1].get_title()
    finally:
        window.close()


def test_urdf_window_reuses_exact_moving_mean_rows_for_norm_and_joints():
    model = Mock(spec=ConstantTorqueModel)
    window = TorqueWindow(plot_config('urdf'), {'right': model}, headless=True)
    first = np.array([-1., 2., -3., 4., -5., 6., -7.])
    second = np.array([14., -12., 10., -8., 6., -4., 2.])
    try:
        window.ingest({'right': sample(urdf_feedback=result(torque=first))}, 10.)
        # The worker's moving mean may update while the robot packet repeats.
        # Every trace must use these exact worker results without another filter.
        window.ingest({'right': sample(urdf_feedback=result(10.009, torque=second))}, 10.01)
        window.draw(10.01)
        expected = np.stack([first, second])
        np.testing.assert_array_equal(window.lines[0, 0].get_ydata(), np.abs(expected).sum(axis=1))
        for joint in range(7):
            np.testing.assert_array_equal(window.lines[joint + 1, 0].get_ydata(), expected[:, joint])
            np.testing.assert_allclose(window.lines[joint + 1, 0].get_xdata(), [-.01, -.001])
            np.testing.assert_array_equal(window.lines[joint + 1, 0].get_xdata(),
                                          window.lines[0, 0].get_xdata())
        model.rnea.assert_not_called()
        model.gravity_torque.assert_not_called()
    finally:
        window.close()


@pytest.mark.parametrize('invalid', [None, result(valid=False), result(torque=float('nan')),
                                   result(timestamp=9.), result(timestamp=11.),
                                   dict(timestamp_s=10., tau_ext=[1., 2.], valid=True)])
def test_invalid_arm_feedback_gaps_all_eight_traces_without_hiding_the_other_arm(invalid):
    cfg = plot_config('urdf')
    cfg['active_sides'] = ('left', 'right')
    cfg['sides']['left'] = cfg['sides']['right'].copy()
    cfg['force_feedback_mean_windows']['left'] = 20
    models = {side: Mock(spec=ConstantTorqueModel) for side in cfg['active_sides']}
    window = TorqueWindow(cfg, models, headless=True)
    try:
        window.ingest({side: sample(urdf_feedback=result(torque=2.))
                       for side in cfg['active_sides']}, 10.)
        window.ingest({'left': sample(10.01, urdf_feedback=result(10.005, torque=-3.)),
                       'right': sample(10.01, urdf_feedback=invalid)}, 10.01)
        window.draw(10.01)
        for row in range(8):
            left = window.lines[row, 0].get_ydata()
            right = window.lines[row, 1].get_ydata()
            assert np.isfinite(left[-1])
            assert left[-1] == (21. if row == 0 else -3.)
            assert np.isnan(right[-1])
            assert window.labels[row, 1].get_text() == '-- Nm'
        window.draw(10.2)
        for row in range(8):
            for column in range(2):
                assert np.isnan(window.lines[row, column].get_ydata()[-1])
        for model in models.values():
            model.rnea.assert_not_called()
            model.gravity_torque.assert_not_called()
    finally:
        window.close()
