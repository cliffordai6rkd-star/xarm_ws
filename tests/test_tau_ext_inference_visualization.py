"""External torque is a measured-state residual, independent of WM/control output."""
from pathlib import Path
from types import SimpleNamespace
import queue
import time

import numpy as np
import pytest
import yaml

from inference.mujoco_visualization import MujocoKinematicVisualizer
from inference.wm_visualization import with_tau_ext_overlay
from nero_collection.cameras import CameraPreviewOverlay

ROOT = Path(__file__).resolve().parents[1]


def result(stamp=100_000_000, *, valid=True):
    return SimpleNamespace(source_timestamp_us=stamp, valid=valid,
                           tau_ext=np.array([-1., 2., -3., 4., -5., 6., -7.]))


def test_external_torque_overlay_stays_visible_when_wm_forecast_expires():
    wm = CameraPreviewOverlay(('old WM',), (2,)*32, 99.)
    overlay = with_tau_ext_overlay(wm, result(), 'URDF momentum', .15, 100.05)
    assert 'L1=28.00' in overlay.lines[0]
    assert '-1.00' in overlay.lines[1]
    assert '-7.00' in overlay.lines[2]
    assert overlay.lines[-1] == 'WM: forecast expired'
    assert overlay.phases == ()
    assert overlay.expires_at_s == pytest.approx(100.15)
    assert 'tau_ext' in overlay.expired_text


@pytest.mark.parametrize('value,now', [(result(valid=False), 100.05), (result(), 100.2), (result(), 99.)])
def test_invalid_stale_or_future_torque_is_not_shown_as_zero(value, now):
    overlay = with_tau_ext_overlay(CameraPreviewOverlay(), value, 'URDF momentum', .15, now)
    assert 'invalid / stale' in overlay.lines[0]
    assert len(overlay.lines) == 1


def test_mujoco_external_torque_is_independent_of_wm_total_torque_and_episode_reset():
    config = SimpleNamespace(enabled=True, robot_joint_names=[f'joint{i}' for i in range(1,8)],
                             prediction_visualization_hz=1000., sample_id=0)
    renderer = MujocoKinematicVisualizer(config)
    renderer._queue.close()
    renderer._queue = queue.Queue(maxsize=1)
    try:
        q = np.zeros(7)
        renderer.publish_observed(10., q, tau_ext=-np.ones(7), tau_ext_timestamp=9.99)
        renderer.publish_prediction(10., q, np.zeros((1,32,7)),
                                    predicted_tau=np.full((1,32,7), 20.), prediction_id=1)
        packet = renderer._queue.get_nowait()
        np.testing.assert_allclose(packet.predicted_tau, 20.)
        np.testing.assert_allclose(packet.observed_tau_ext, -1.)
        np.testing.assert_allclose(packet.initial_tau_ext, -1.)
        renderer.publish_observed(10.01, q, tau_ext=None)
        packet = renderer._queue.get_nowait()
        assert packet.observed_tau_ext is None
        np.testing.assert_allclose(packet.predicted_tau, 20.)
        renderer.reset()
        renderer.publish_observed(11., q, tau_ext=2*np.ones(7), tau_ext_timestamp=10.99)
        packet = renderer._queue.get_nowait()
        np.testing.assert_allclose(packet.initial_tau_ext, 2.)
        assert packet.predicted_tau is None
    finally:
        renderer.enabled = False


def test_actual_momentum_worker_estimates_added_torque_without_hardware_commands():
    pytest.importorskip('pinocchio')
    from inference.tau_ext_visualization import TauExtPreview
    path = ROOT/'gello_teleop/config/xarm7_gello_dual_dataset.yaml'
    hardware = yaml.safe_load(path.read_text())
    hardware['force_feedback']['tau_ext_mean_window'] = 1
    # The arm exposes only its cached payload, no robot command/read methods.
    arm = SimpleNamespace(reported_tcp_payload=dict(mass_kg=0., com_m=[0.,0.,0.]))
    preview = TauExtPreview(dict(plot=False), hardware, path, 'right', arm)
    q = np.array([0., -.4, 0., 1.4, 0., 1.2, -1.5])
    preview.start()
    try:
        model = preview.worker.estimators[0].model
        base = model.rnea(q, np.zeros(7), np.zeros(7))
        extra = np.array([0., 2., 0., 0., 0., 0., 0.])
        for _ in range(24):
            stamp = time.time_ns()//1000
            state = SimpleNamespace(timestamp_us=stamp, acquired_timestamp_us=stamp,
                q=q, dq=np.zeros(7), torque=base+extra, q_valid=True, dq_valid=True, torque_valid=True)
            preview.submit(state)
            time.sleep(.01)
        value = preview.snapshot()
        assert value.valid
        assert preview.worker.errors == 0
        np.testing.assert_allclose(value.tau_ext, extra, atol=.03)
        preview.reset()
        assert not preview.snapshot().valid
    finally:
        preview.close()


def test_active_single_config_enables_external_torque_without_enabling_force_feedback():
    from inference.dp_wm.config import load_config
    config = load_config(ROOT/'inference/configs/pi0_wm_xarm_single.yaml')
    hardware = yaml.safe_load(config['observation']['hardware_config'].read_text())
    assert config['visualization']['tau_ext'] == dict(enabled=True, plot=True)
    assert config['visualization']['mujoco']['enabled']
    assert hardware['force_feedback']['enabled'] is False
