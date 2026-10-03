from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from xarm_stack.momentum import FreeSpaceMomentumObserver, PinocchioMomentumModel
from xarm_stack.torque_visualization import (DEFAULT_URDF, FirstOrderLowPass, TorqueHistory,
                                           TorqueVisualizer, TorqueWindow, plot_config)


class ConstantMassModel:
    def terms(self, q, dq):
        return 2*dq, np.full(7, -3.)

    def gravity_torque(self, q):
        return np.full(7, 3.)

    def rnea(self, q, dq, ddq):
        return 2*ddq+3.


def test_observer_static_gravity_and_constant_acceleration():
    observer = FreeSpaceMomentumObserver(ConstantMassModel(), 3.)
    q, dq = np.zeros(7), np.zeros(7)
    assert np.isnan(observer.update(0., q, dq)).all()
    np.testing.assert_allclose(observer.update(.01, q, dq), 3.)
    # Constant dq ramp gives 2*ddq+g=4 Nm, without an acceleration input.
    for timestamp in np.arange(.02, 1.01, .01):
        torque = observer.update(timestamp, q, np.full(7, .5*(timestamp-.01)))
    np.testing.assert_allclose(torque, 4., atol=1e-6)


def test_observer_handles_duplicate_invalid_gap_and_clock_reversal():
    observer = FreeSpaceMomentumObserver(ConstantMassModel())
    q = dq = np.zeros(7)
    observer.update(1., q, dq)
    original = observer.update(1.01, q, dq)
    np.testing.assert_equal(observer.update(1.01, q, np.ones(7)*100), original)
    assert np.isnan(observer.update(1.02, q, dq, valid=False)).all()
    assert np.isnan(observer.update(1.03, q, dq)).all()
    np.testing.assert_allclose(observer.update(1.04, q, dq), 3.)
    assert np.isnan(observer.update(2., q, dq)).all()
    assert np.isnan(observer.update(1.5, q, dq)).all()


def test_model_observer_matches_filtered_rnea_on_moving_xarm():
    pin = pytest.importorskip('pinocchio')
    model = PinocchioMomentumModel(DEFAULT_URDF)
    observer = FreeSpaceMomentumObserver(model, bandwidth_hz=3.)
    expected = None
    previous = None
    gain = 2*np.pi*3.
    # Independently compare to low-pass RNEA with known analytic acceleration,
    # exercising changing inertia and C.T dq rather than only static gravity.
    for timestamp in np.arange(0., 2., .002):
        phase = np.arange(7)*.4
        q = .2*np.sin(2*timestamp+phase)
        dq = .4*np.cos(2*timestamp+phase)
        ddq = -.8*np.sin(2*timestamp+phase)
        exact = pin.rnea(model.model, model.data, q, dq, ddq).copy()
        actual = observer.update(timestamp, q, dq)
        if expected is None:
            expected = -model.terms(q, dq)[1]
        else:
            half = gain*.002/2
            expected = ((1-half)*expected+half*(previous+exact))/(1+half)
            np.testing.assert_allclose(actual, expected, atol=2e-5)
        previous = exact


def sample(timestamp=10., **overrides):
    return dict(timestamp_s=timestamp, q=np.zeros(7), dq=np.zeros(7),
                ddq=np.zeros(7), q_valid=True, dq_valid=True, ddq_valid=True) | overrides


def test_history_rnea_needs_acceleration_and_no_measured_torque():
    history = TorqueHistory(ConstantMassModel(), .15, 30.)
    history.ingest(sample(), 10.)
    history.ingest(sample(10.01), 10.01)
    _, model, measured, gravity = history.arrays(10.01)
    np.testing.assert_allclose(model[-1], 3.)
    assert np.isnan(measured).all()
    np.testing.assert_allclose(gravity[-1], 3.)
    history.ingest(sample(10.02, dq_valid=False), 10.02)
    _, model, _, _ = history.arrays(10.02)
    assert np.isnan(model[-1]).all()
    assert 'velocity unavailable' in history.status
    history.ingest(sample(10.025, ddq_valid=False), 10.025)
    assert np.isnan(history.rows[-1][1]).all()
    assert 'acceleration unavailable' in history.status
    history.ingest(sample(10.03, q_valid=False), 10.03)
    assert np.isnan(history.rows[-1][1]).all()
    history.arrays(10.3)
    assert history.measured_filter.timestamp_s is None
    history.ingest(sample(10.31), 10.31)
    np.testing.assert_allclose(history.rows[-1][1], 3.)
    history.ingest(sample(10.32), 10.32)
    np.testing.assert_allclose(history.rows[-1][1], 3.)


def test_history_deduplicates_and_marks_dropped_intervals():
    history = TorqueHistory(ConstantMassModel(), .15, 30.)
    history.ingest(sample(), 10.)
    history.ingest(sample(), 10.01)
    assert len(history.rows) == 1
    history.ingest(sample(10.3), 10.3)
    assert len(history.rows) == 3
    assert np.isnan(history.rows[1][1]).all()
    np.testing.assert_allclose(history.rows[-1][1], 3.)
    history.arrays(41.)
    assert len(history.rows) == 1


def test_rnea_uses_filtered_acceleration_and_independently_filtered_measured_torque():
    history = TorqueHistory(ConstantMassModel(), .15, 30., measured_torque_cutoff_hz=3.,
                            acceleration_cutoff_hz=10.)
    history.ingest(sample(torque=np.zeros(7), torque_valid=True), 10.)
    history.ingest(sample(10.01, ddq=np.ones(7)*5., torque=np.ones(7)*13., torque_valid=True), 10.01)
    _, model, measured, _ = history.arrays(10.01)
    # No low-pass on RNEA output: use 2*filtered_ddq+g directly.
    np.testing.assert_allclose(model[:, 0], [3., 3.+10.*(1-np.exp(-2*np.pi*10*.01))])
    np.testing.assert_allclose(measured[-1], 13.*(1-np.exp(-2*np.pi*3*.01)))
    assert history.status == 'RNEA'


def test_gravity_changes_are_not_filtered_as_a_model_torque_output():
    class PositionDependentGravity(ConstantMassModel):
        def gravity_torque(self, q):
            return 3.+q

        def rnea(self, q, dq, ddq):
            return 2*ddq+self.gravity_torque(q)
    history = TorqueHistory(PositionDependentGravity(), .15, 30.)
    history.ingest(sample(), 10.)
    history.ingest(sample(10.01, q=np.ones(7)), 10.01)
    np.testing.assert_allclose(history.rows[-1][1], 4.)


def test_invalid_acceleration_resets_its_filter_without_affecting_measured_torque():
    history = TorqueHistory(ConstantMassModel(), .15, 30.)
    history.ingest(sample(torque=np.ones(7)*8., torque_valid=True), 10.)
    history.ingest(sample(10.01, ddq_valid=False, torque=np.ones(7)*8., torque_valid=True), 10.01)
    assert np.isnan(history.rows[-1][1]).all()
    np.testing.assert_allclose(history.rows[-1][2], 8.)
    history.ingest(sample(10.02, ddq=np.ones(7)*5., torque=np.ones(7)*8., torque_valid=True), 10.02)
    np.testing.assert_allclose(history.rows[-1][1], 13.)


def test_moving_xarm_history_matches_dynamics_with_filtered_acceleration():
    pin = pytest.importorskip('pinocchio')
    model = PinocchioMomentumModel(DEFAULT_URDF)
    history = TorqueHistory(model, .15, 30., acceleration_cutoff_hz=5.)
    filtered_ddq = None
    for t in np.arange(0., .5, .005):
        phase = np.arange(7)*.4
        q, dq, ddq = .2*np.sin(2*t+phase), .4*np.cos(2*t+phase), -.8*np.sin(2*t+phase)
        filtered_ddq = (ddq.copy() if filtered_ddq is None else
                        np.exp(-2*np.pi*5*.005)*filtered_ddq+(1-np.exp(-2*np.pi*5*.005))*ddq)
        history.ingest(sample(10.+t, q=q, dq=dq, ddq=ddq), 10.+t)
        mass = np.asarray(pin.crba(model.model, model.data, q)).copy()
        coriolis = np.asarray(pin.computeCoriolisMatrix(model.model, model.data, q, dq)).copy()
        expected = mass@filtered_ddq+coriolis@dq+model.gravity_torque(q)
        np.testing.assert_allclose(history.rows[-1][1], expected, atol=1e-10)


def test_low_pass_step_response_uses_actual_intervals_and_per_joint_cutoffs():
    cutoff = np.arange(1., 8.)
    low_pass = FirstOrderLowPass(cutoff)
    low_pass.update(0., np.zeros(7))
    for timestamp in [.003, .011, .025, .06, .12]:
        actual = low_pass.update(timestamp, np.ones(7))
        np.testing.assert_allclose(actual, 1.-np.exp(-2*np.pi*cutoff*timestamp), atol=1e-14)


def test_low_pass_missing_samples_reset_without_a_zero_startup_transient():
    low_pass = FirstOrderLowPass(3.)
    np.testing.assert_allclose(low_pass.update(1., np.full(7, 8.)), 8.)
    np.testing.assert_allclose(low_pass.update(1., np.full(7, 99.)), 8.)
    assert np.isnan(low_pass.update(1.01, np.ones(7), valid=False)).all()
    np.testing.assert_allclose(low_pass.update(1.02, np.full(7, 6.)), 6.)
    np.testing.assert_allclose(low_pass.update(2., np.full(7, 4.)), 4.)
    assert np.isnan(low_pass.update(2.01, np.full(7, np.nan))).all()


def test_missing_effort_does_not_hold_filtered_torque_or_stop_rnea():
    history = TorqueHistory(ConstantMassModel(), .15, 30.)
    history.ingest(sample(torque=np.ones(7)*8., torque_valid=True), 10.)
    history.ingest(sample(10.01), 10.01)
    assert np.isnan(history.rows[-1][2]).all()
    np.testing.assert_allclose(history.rows[-1][1], 3.)
    history.ingest(sample(10.02, torque=np.ones(7)*5., torque_valid=True), 10.02)
    np.testing.assert_allclose(history.rows[-1][2], 5.)


@pytest.mark.parametrize('setting', ['measured_torque_cutoff_hz', 'acceleration_cutoff_hz'])
@pytest.mark.parametrize('cutoff', [0., -3., np.nan, [1., 2.], [1., 1., 1., -1., 1., 1., 1.]])
def test_invalid_cutoff_is_rejected_before_startup(setting, cutoff):
    with pytest.raises(ValueError, match=setting):
        plot_config({setting: cutoff}, __file__, ['left'])


def test_cutoffs_can_be_overridden_per_arm():
    cfg = plot_config({'measured_torque_cutoff_hz': 3., 'acceleration_cutoff_hz': 4.,
                       'sides': {'right': {'measured_torque_cutoff_hz': 5., 'acceleration_cutoff_hz': 8.}}},
                      __file__, ['left', 'right'])
    assert cfg['sides']['left']['measured_torque_cutoff_hz'] == 3.
    assert cfg['sides']['right']['measured_torque_cutoff_hz'] == 5.
    assert cfg['sides']['left']['acceleration_cutoff_hz'] == 4.
    assert cfg['sides']['right']['acceleration_cutoff_hz'] == 8.


@pytest.mark.parametrize('sides', [('left',), ('right',), ('left', 'right')])
def test_residual_window_and_single_arm_layout(tmp_path, sides):
    pytest.importorskip('matplotlib')
    cfg = plot_config({}, __file__, sides)
    window = TorqueWindow(cfg, {side: ConstantMassModel() for side in sides}, headless=True)
    try:
        for timestamp in [10., 10.01]:
            prediction = dict(timestamp_s=timestamp, valid=True,
                              tau_measured=np.full(7, 5.), tau_pred=np.full(7, 2.))
            window.ingest({side: sample(timestamp, torque=np.full(7, 5.), torque_valid=True,
                                       tau_free=prediction) for side in sides}, timestamp)
        window.draw(10.01)
        assert window.axes.shape == (7, 2*len(sides))
        for side_index, side in enumerate(sides):
            column = 2*side_index+1
            assert side.title() in window.axes[0, column].get_title()
            assert 'tau_ext_cal' in window.axes[0, column].get_title()
            assert 'tau_ext_pred' in window.axes[0, column-1].get_title()
            assert len(window.lines[0, column].get_ydata()) == 2
            np.testing.assert_allclose(window.lines[0, column].get_ydata()[-1], 2.)
            np.testing.assert_allclose(window.lines[0, column-1].get_ydata()[-1], 3.)
            assert window.axes[0, column].get_ylim() == window.axes[0, column-1].get_ylim()
        window.save(tmp_path/'torque.png')
        assert (tmp_path/'torque.png').stat().st_size > 10000
    finally:
        window.close()


def test_async_prediction_uses_paired_measured_torque_and_source_time():
    history = TorqueHistory(ConstantMassModel(), .15, 30.)
    history.ingest(sample(torque=np.full(7, 5.), torque_valid=True), 10.)
    prediction = dict(timestamp_s=9.98, valid=True,
                      tau_measured=np.full(7, 9.), tau_pred=np.full(7, 4.))
    # A new inference result may arrive without a new hardware state.
    history.ingest(sample(torque=np.full(7, 99.), torque_valid=True,
                          tau_free=prediction), 10.01)
    assert len(history.rows) == 1
    t, residual = history.prediction_arrays(10.01)
    np.testing.assert_allclose(t, [-.03])
    np.testing.assert_allclose(residual[-1], 5.)
    _, model, measured, _ = history.arrays(10.01)
    np.testing.assert_allclose(measured-model, 2.)


def test_invalid_and_stale_prediction_breaks_curve_and_releases_latest_label():
    history = TorqueHistory(ConstantMassModel(), .15, 30.)
    def prediction(timestamp, **overrides):
        return dict(timestamp_s=timestamp, valid=True,
                    tau_measured=np.full(7, 5.), tau_pred=np.full(7, 2.)) | overrides
    history.ingest(sample(tau_free=prediction(10.)), 10.)
    history.ingest(sample(10.01, tau_free=prediction(10.)), 10.01)
    assert len(history.prediction_rows) == 1
    history.ingest(sample(10.02, tau_free=prediction(10.02, valid=False)), 10.02)
    assert np.isnan(history.prediction_arrays(10.02)[1][-1]).all()
    history.ingest(sample(10.03, tau_free=prediction(10.03)), 10.03)
    np.testing.assert_allclose(history.prediction_arrays(10.03)[1][-1], 3.)
    assert np.isnan(history.prediction_arrays(10.2)[1][-1]).all()
    history.ingest(sample(10.21, tau_free=prediction(10.03)), 10.21)
    assert np.isnan(history.prediction_arrays(10.21)[1][-1]).all()


def test_prediction_unavailable_keeps_urdf_residual_visible():
    pytest.importorskip('matplotlib')
    window = TorqueWindow(plot_config({}, __file__, ['right']),
                          {'right': ConstantMassModel()}, headless=True)
    try:
        window.ingest({'right': sample(torque=np.full(7, 5.), torque_valid=True)}, 10.)
        window.draw(10.)
        assert len(window.lines[0, 0].get_ydata()) == 0
        assert window.labels[0, 0].get_text() == '-- Nm'
        np.testing.assert_allclose(window.lines[0, 1].get_ydata(), [2.])
    finally:
        window.close()


def test_prediction_ignores_late_result_and_recovers_after_host_clock_reversal():
    history = TorqueHistory(ConstantMassModel(), .15, 30.)
    def predicted_sample(timestamp, measured):
        return sample(timestamp, tau_free=dict(timestamp_s=timestamp, valid=True,
                      tau_measured=np.full(7, measured), tau_pred=np.full(7, 2.)))
    history.ingest(predicted_sample(10., 5.), 10.)
    history.ingest(predicted_sample(9.99, 99.), 10.01)
    np.testing.assert_allclose(history.prediction_arrays(10.01)[1][-1], 3.)
    history.ingest(predicted_sample(9., 6.), 9.)
    t, residual = history.prediction_arrays(9.)
    np.testing.assert_allclose(t, [0.])
    np.testing.assert_allclose(residual[-1], 4.)


def test_repeated_prediction_survives_a_display_window_shorter_than_its_age():
    history = TorqueHistory(ConstantMassModel(), .15, .01)
    prediction = dict(timestamp_s=10., valid=True,
                      tau_measured=np.full(7, 5.), tau_pred=np.full(7, 2.))
    history.ingest(sample(tau_free=prediction), 10.)
    assert history.prediction_arrays(10.02)[0].size == 0
    history.ingest(sample(10.02, tau_free=prediction), 10.02)
    assert history.prediction_arrays(10.02)[0].size == 0


def test_publisher_keeps_network_measurement_and_prediction_paired():
    import queue
    from gello_teleop.tau_free_feedback import TauFreeResult
    from nero_collection.arms.base import ArmState
    visualizer = TorqueVisualizer({})
    visualizer.process = Mock()
    visualizer.process.is_alive.return_value = True
    visualizer.samples = queue.Queue()
    state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.eye(4),
                     np.full(7, 99.), np.zeros(7), 1_020_000)
    result = TauFreeResult(1_000_000, 1_010_000, np.full(7, 2.), np.full(7, 5.),
                           np.full(7, 3.), np.full(7, 3.), True)
    visualizer.publish(['right'], [state], tau_free_results=[result])
    packet = visualizer.samples.get_nowait()['right']
    assert packet['timestamp_s'] == 1.02
    assert packet['tau_free']['timestamp_s'] == 1.
    np.testing.assert_allclose(packet['tau_free']['tau_measured'], 5.)
    result.tau_measured[:] = 0.
    np.testing.assert_allclose(packet['tau_free']['tau_measured'], 5.)


def test_publisher_queue_full_does_not_block():
    import queue
    from nero_collection.arms.base import ArmState
    visualizer = TorqueVisualizer({})
    visualizer.process = Mock()
    visualizer.process.is_alive.return_value = True
    visualizer.samples = queue.Queue(maxsize=1)
    state = ArmState(np.zeros(7), np.zeros(7), np.full(7, np.nan), np.eye(4),
                     np.ones(7), np.full(7, np.nan), 1_000_000, torque_valid=True)
    visualizer.publish(['left'], [state])
    visualizer.publish(['left'], [state])
    assert visualizer.dropped_samples == 1
    packet = visualizer.samples.get_nowait()['left']
    assert packet['ddq_valid'] and packet['torque_valid'] and packet['timestamp_s'] == 1.
    assert np.isnan(packet['ddq']).all()
    np.testing.assert_equal(packet['torque'], np.ones(7))
    np.testing.assert_equal(packet['q'], np.zeros(7))


def test_stationary_xarm_reset_pose_has_nonzero_theoretical_gravity_without_effort():
    pytest.importorskip('pinocchio')
    model = PinocchioMomentumModel(DEFAULT_URDF)
    history = TorqueHistory(model, .15, 30.)
    q = np.deg2rad([0, 0, 0, 90, 0, 90, 0])
    history.ingest(sample(q=q), 10.)
    history.ingest(sample(10.01, q=q), 10.01)
    _, torque, _, _ = history.arrays(10.01)
    np.testing.assert_allclose(torque[-1], model.gravity_torque(q), atol=1e-10)
    assert torque[-1, 1] < -12. and torque[-1, 3] > 9.


def test_official_inertial_parameters_and_inverse_dynamics():
    import yaml
    pin = pytest.importorskip('pinocchio')
    model = PinocchioMomentumModel(DEFAULT_URDF)
    parameters = yaml.safe_load(DEFAULT_URDF.with_name('xarm7_type7_HT_BR2.yaml').read_text())
    for index in range(1, 8):
        expected = parameters[f'link{index}']
        actual = model.model.inertias[index]
        assert actual.mass == expected['mass']
        np.testing.assert_allclose(actual.lever, [expected['origin'][axis] for axis in 'xyz'])
        assert np.linalg.eigvalsh(actual.inertia).min() > 0
    assert sum(inertia.mass for inertia in model.model.inertias[1:]) == pytest.approx(10.4315)
    rng = np.random.default_rng(7)
    for _ in range(10):
        q, dq, ddq = rng.uniform(-1., 1., (3, 7))
        mass = np.asarray(pin.crba(model.model, model.data, q)).copy()
        coriolis = np.asarray(pin.computeCoriolisMatrix(model.model, model.data, q, dq)).copy()
        gravity = model.gravity_torque(q)
        assert np.linalg.eigvalsh(mass).min() > 0
        np.testing.assert_allclose(mass@ddq+coriolis@dq+gravity,
                                   model.rnea(q, dq, ddq), atol=1e-10)
        step = 1e-6
        gradient = []
        for axis in np.eye(7):
            high = pin.computePotentialEnergy(model.model, model.data, q+step*axis)
            low = pin.computePotentialEnergy(model.model, model.data, q-step*axis)
            gradient.append((high-low)/(2*step))
        np.testing.assert_allclose(gravity, gradient, atol=1e-7)


def test_payload_gravity_matches_flange_jacobian_and_does_not_accumulate():
    pin = pytest.importorskip('pinocchio')
    model = PinocchioMomentumModel(DEFAULT_URDF)
    q = np.deg2rad([15, -25, 10, 90, 5, 70, -10])
    # Synthetic CoM exercises all three axes; not a claimed G2 CoM.
    payload = {'mass_kg': .79, 'com_m': [.04, .01, .06]}
    bare = model.gravity_torque(q)
    original_mass = model.model.inertias[7].mass
    assert model.set_payload(payload)
    assert not model.set_payload(payload)
    assert model.model.inertias[7].mass == pytest.approx(original_mass+.79)
    loaded = model.gravity_torque(q)
    frame = model.model.getFrameId('link_eef')
    jacobian = pin.computeFrameJacobian(model.model, model.data, q, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    rotation = model.data.oMf[frame].rotation
    com_world = rotation@np.asarray(payload['com_m'])
    force = -.79*np.asarray(model.model.gravity.linear)
    np.testing.assert_allclose(loaded-bare, jacobian[:3].T@force+jacobian[3:].T@np.cross(com_world, force), atol=1e-10)
    assert model.set_payload(None)
    np.testing.assert_allclose(model.gravity_torque(q), bare, atol=1e-12)


def test_payload_report_is_sent_without_new_hardware_commands():
    import queue
    from types import SimpleNamespace
    from nero_collection.arms.base import ArmState
    visualizer = TorqueVisualizer({'sides': {'left': {'payload': {'source': 'controller'}}}})
    visualizer.process = Mock()
    visualizer.process.is_alive.return_value = True
    visualizer.samples = queue.Queue()
    payload = {'mass_kg': .79, 'com_m': [.02, 0., .04]}
    visualizer.bind_payload_source('left', SimpleNamespace(reported_tcp_payload=payload))
    state = ArmState(np.zeros(7), np.zeros(7), np.zeros(7), np.eye(4), np.ones(7), np.zeros(7), 1_000_000)
    visualizer.publish(['left'], [state])
    assert visualizer.samples.get_nowait()['left']['payload'] == payload


def test_process_startup_failure_is_reported_and_cleaned_up(tmp_path):
    # Spawn the actual plot process, but fail model construction before a GUI
    # is opened. This also checks Pipe readiness and bounded shutdown.
    invalid = tmp_path/'invalid.urdf'
    invalid.write_text('<robot name="invalid"><link name="base"/></robot>')
    cfg = plot_config({'urdf_path': str(invalid)}, __file__, ['left'])
    visualizer = TorqueVisualizer(cfg)
    with pytest.raises(RuntimeError, match='joint1..joint7'):
        visualizer.start()
    assert visualizer.process is None and visualizer.samples is None
    visualizer.close()


@pytest.mark.parametrize('path', [
    'gello_teleop/config/xarm7_gello_dual_dataset.yaml',
    'gello_teleop/config/xarm7_gello_full_pipeline.yaml',
])
def test_collection_configs_resolve_the_xarm_urdf(path):
    import yaml
    cfg = plot_config(yaml.safe_load(Path(path).read_text())['torque_visualization'], path, ['left', 'right'])
    assert all(Path(side['urdf_path']) == DEFAULT_URDF for side in cfg['sides'].values())
    assert all(side['payload']['source'] == 'controller' for side in cfg['sides'].values())


def test_collector_starts_stops_plot_and_publishes_while_not_recording():
    import time
    from test_dual_gello_pipeline import make_pipeline
    pipeline = make_pipeline()
    visualizer = Mock()
    pipeline.torque_visualizer = visualizer
    try:
        pipeline.connect()
        visualizer.start.assert_called_once()
        pipeline.reset()
        pipeline.confirm_alignment()
        pipeline.takeover()
        deadline = time.monotonic()+1.
        while not visualizer.publish.called and time.monotonic() < deadline:
            time.sleep(.01)
        assert visualizer.publish.called
        assert not pipeline.recording
        sides, states = visualizer.publish.call_args.args
        assert sides == ('left', 'right') and len(states) == 2
    finally:
        pipeline.close()
    visualizer.close.assert_called_once()
