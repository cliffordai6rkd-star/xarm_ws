"""WM previews preserve physical units, timestamps, episode baselines and input pixels."""
from contextlib import nullcontext
from pathlib import Path
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from inference.mujoco_visualization import MujocoKinematicFK, MujocoKinematicVisualizer, TorqueBars, VisualizationPacket
from inference.wm_visualization import contact_overlay
from nero_collection.cameras import _draw_camera_overlay

ROOT = Path(__file__).resolve().parents[1]


def settings():
    return SimpleNamespace(enabled=True, headless=True, prediction_visualization_hz=1000.,
        robot_joint_names=[f'joint{i}' for i in range(1, 8)], sample_id=0,
        mujoco_model_path=ROOT/'gello_teleop/models/xarm7_dynamics_g2.urdf',
        ee_site_name=None, ee_body_name='link_eef',
        torque_base_length_m=.08, torque_gain=2., torque_baseline_floor_nm=.1)


def test_torque_units_signed_values_and_zero_initial_baseline():
    bars = TorqueBars(settings())
    initial = np.array([0., .01, -2., 4., 5., 6., 7.])
    bars.lengths(initial)
    doubled = 2*initial
    doubled[:2] = [.2, -.2]
    np.testing.assert_allclose(bars.lengths(doubled), .32)
    np.testing.assert_allclose(bars.baseline, [.1, .1, 2, 4, 5, 6, 7])
    np.testing.assert_allclose(bars.lengths(np.zeros(7)), 0.)


def test_baseline_survives_lost_first_packet_and_restarts_after_reset():
    visualizer = MujocoKinematicVisualizer(settings())
    visualizer._queue.close()
    visualizer._queue = queue.Queue(maxsize=1)
    try:
        q = np.zeros((1, 32, 7))
        visualizer.publish_prediction(10., np.zeros(7), q, predicted_tau=np.ones_like(q), prediction_id=1)
        visualizer._last_prediction_publish_s = -np.inf
        visualizer.publish_prediction(11., np.zeros(7), q, predicted_tau=4*np.ones_like(q), prediction_id=2)
        visualizer.publish_observed(12., np.arange(7))
        packet = visualizer._queue.get_nowait()
        np.testing.assert_allclose(packet.initial_tau, 1.)
        np.testing.assert_allclose(packet.predicted_tau, 4.)
        np.testing.assert_allclose(packet.observed_q, np.arange(7))
        visualizer.reset()
        visualizer.publish_prediction(13., np.zeros(7), q, predicted_tau=2*np.ones_like(q), prediction_id=3)
        packet = visualizer._queue.get_nowait()
        np.testing.assert_allclose(packet.initial_tau, 2.)
        assert packet.episode_id == 1
    finally:
        # Replaced with an in-process test queue.
        visualizer.enabled = False


def test_contact_uses_trained_three_phase_labels_and_release_time():
    anchor = 1_000_000
    states = np.r_[np.zeros(3, dtype=int), np.ones(2, dtype=int),
                   np.full(15, 2), np.zeros(12, dtype=int)]
    forecast = dict(anchor_us=anchor, future_timestamps_us=anchor+np.arange(1,33)*10_000,
                    contact_probability=np.eye(3)[states][None])
    overlay = contact_overlay(forecast, 0, 1.07)
    assert 'contact' in overlay.lines[0]
    assert '+70 ms' in overlay.lines[0]
    assert 'contact=100.0%' in overlay.lines[1]
    assert '+210 ms' in overlay.lines[2]
    assert overlay.phases == tuple(states)
    assert overlay.expires_at_s == 1.32


def test_preview_pixels_unchanged_and_expired_forecasts_hidden():
    cv2 = pytest.importorskip('cv2')
    original = np.full((224, 672, 3), 111, np.uint8)
    forecast = dict(anchor_us=1_000_000, future_timestamps_us=1_000_000+np.arange(1,33)*10_000,
                    contact_probability=np.tile([0.,0.,1.], (1,32,1)))
    overlay = contact_overlay(forecast, 0, 1.1)
    image = _draw_camera_overlay(original, overlay, cv2, 1.1)
    np.testing.assert_array_equal(original, image[:224])
    np.testing.assert_array_equal(original, 111)
    np.testing.assert_array_equal(image[-1, 0], [70, 70, 240])
    expired = _draw_camera_overlay(original, overlay, cv2, 1.4)
    assert expired.shape[0] < image.shape[0]
    np.testing.assert_array_equal(expired[:224], original)


def test_urdf_joint_anchors_match_independent_pinocchio_fk_and_draw_bars():
    mj = pytest.importorskip('mujoco')
    pin = pytest.importorskip('pinocchio')
    config = settings()
    fk = MujocoKinematicFK(config)
    q = np.array([.2, -.3, .1, 1.4, .2, 1.2, -1.5])
    observed = fk.set_observed_q(q)
    model = pin.buildModelFromUrdf(str(config.mujoco_model_path))
    data = model.createData()
    pin.framesForwardKinematics(model, data, q)
    np.testing.assert_allclose(observed, data.oMf[model.getFrameId('link_eef')].translation, atol=1e-7)
    assert fk.model.nq == 7
    for name in config.robot_joint_names:
        joint = mj.mj_name2id(fk.model, mj.mjtObj.mjOBJ_JOINT, name)
        pose = data.oMi[model.getJointId(name)]
        np.testing.assert_allclose(fk.display_data.xanchor[joint], pose.translation, atol=1e-7)
        np.testing.assert_allclose(fk.display_data.xaxis[joint], pose.rotation[:,2], atol=1e-7)
    scene = mj.MjvScene(fk.model, maxgeom=100)
    viewer = SimpleNamespace(user_scn=scene, lock=lambda: nullcontext())
    bars = TorqueBars(config)
    tau = np.array([1., -2., 0., 4., -5., 6., 7.])
    bars.draw(viewer, mj, fk.model, fk.display_data, tau)
    assert scene.ngeom == 14
    np.testing.assert_allclose(scene.geoms[0].size[2], .08)
    np.testing.assert_allclose(scene.geoms[4].size[2], 0.)
    assert 'Nm' in scene.geoms[2].label
    assert scene.geoms[0].rgba[0] > scene.geoms[0].rgba[2]
    assert scene.geoms[2].rgba[2] > scene.geoms[2].rgba[0]


def test_packet_rejects_mismatched_torque_horizon():
    with pytest.raises(ValueError, match='predicted_tau'):
        VisualizationPacket(1., np.zeros(7), np.zeros((1,32,7)), predicted_tau=np.zeros((1,8,7)))


def test_alignment_report_does_not_call_matching_shapes_matching_semantics():
    from inference.dp_wm.models import alignment_report
    config = dict(wm=dict(flow_steps=32, precision='float16'),
        observation=dict(command_source='controller_reference'),
        controller=dict(enabled=True, torque_reference='causal_inverse', tau_filter_cutoff_hz=20.))
    training = dict(action_contract=dict(semantic='va_measured_joint_snapshot_fk'),
        dataloader=dict(filters=dict(tau=dict(enabled=True, operations=[dict(type='lowpass', cutoff_hz=15., order=1)]))),
        model=dict(flow_inference_steps=16), train=dict(amp=dict(enabled=True, dtype='bfloat16')))
    report = alignment_report(config, training)
    assert len(report['issues']) == 5


def test_common_preview_converts_unix_times_to_monotonic_without_changing_measured_pose(monkeypatch):
    from inference import wm_visualization as module
    observed, predicted, overlays = [], [], []
    renderer = SimpleNamespace(publish_observed=lambda *args: observed.append(args),
        publish_prediction=lambda *args, **kwargs: predicted.append((args, kwargs)))
    monkeypatch.setattr(module, 'MujocoKinematicVisualizer', lambda _: renderer)
    monkeypatch.setattr(module.time, 'time', lambda: 100.1)
    monkeypatch.setattr(module.time, 'monotonic', lambda: 20.1)
    camera = SimpleNamespace(set_overlay=lambda value: overlays.append(value))
    source = SimpleNamespace(cameras=SimpleNamespace(visualizer=camera))
    visual = module.WmVisualization(dict(visualization=dict(camera_contact=True, mujoco=dict(sample_id=0))), source)
    state = SimpleNamespace(timestamp_us=100_050_000, q=np.arange(7))
    times = 100_000_000+np.arange(1,33)*10_000
    event = dict(kind='wm', anchor_us=100_000_000, future_timestamps_us=times,
        q=np.full((1,32,7), 1.5), tau=np.full((1,32,7), 4.),
        contact_probability=np.tile([1.,0.,0.], (1,32,1)))
    visual.update(state, [event])
    assert observed[0][0] == pytest.approx(20.05)
    np.testing.assert_array_equal(observed[0][1], state.q)
    assert predicted[0][0][0] == pytest.approx(20.)
    np.testing.assert_allclose(predicted[0][1]['future_timestamps'], times/1e6-80.)
    np.testing.assert_allclose(predicted[0][1]['predicted_tau'], 4.)
    assert overlays[0].expires_at_s == pytest.approx(100.32)
