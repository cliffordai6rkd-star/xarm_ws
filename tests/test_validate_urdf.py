import numpy as np
import pytest

from xarm_stack.validate_urdf import (DEFAULT_URDF, DynamicsComparison, analyze,
                                      integrate_torque, managed_viewer, metrics, mit_torque,
                                      preprocess, read_episode)


def write_episode(path, *, timestamps=None, invalid=None):
    import h5py
    t = np.arange(201)*.01
    phase = np.arange(7)*.1
    q = .2*np.sin(t[:, None]+phase)
    dq = .2*np.cos(t[:, None]+phase)
    with h5py.File(path, 'w') as f:
        group = f.create_group('teleop')
        group.attrs['arm_names'] = ['right']
        group['timestamp_us'] = 1_000_000+(t*1e6).astype(np.int64)
        clock = group.create_dataset('right_q_timestamp_us_xarm', data=(
            1_000_000+(t*1e6).astype(np.int64) if timestamps is None else timestamps))
        for key, values, flag, unit in (
                ('q', q, 'q_valid', 'rad'), ('dq', dq, 'dq_valid', 'rad/s'),
                ('tau', q*0, 'torque_valid', 'Nm')):
            d = group.create_dataset(f'right_{key}_xarm', data=values)
            d.attrs['unit'] = unit
            d.attrs['source_timestamp_path'] = clock.name
            flags = np.ones((len(t), 1), np.uint8)
            if invalid is not None:
                flags[invalid] = 0
            group[f'right_{flag}_xarm'] = flags
    return path


def test_h5_source_clock_deduplicates_last_report_and_honors_invalidity(tmp_path):
    ts = 1_000_000+np.arange(201)*10_000
    ts[11] = ts[10]
    path = write_episode(tmp_path/'episode.h5', timestamps=ts, invalid=11)
    data = read_episode(path)
    assert data['input_rows'] == 201 and data['repeated_rows'] == 1
    assert not data['valid'][10]
    assert data['t'][11] == pytest.approx(.12)
    assert data['sources']['tau']['clock'] == '/teleop/right_q_timestamp_us_xarm'


def test_h5_rejects_other_arm_current_units_and_different_clocks(tmp_path):
    import h5py
    path = write_episode(tmp_path/'episode.h5')
    with pytest.raises(ValueError, match='was not recorded'):
        read_episode(path, 'left')
    with h5py.File(path, 'r+') as f:
        f['teleop/right_tau_xarm'].attrs['unit'] = 'A'
    with pytest.raises(ValueError, match='expected Nm'):
        read_episode(path)
    with h5py.File(path, 'r+') as f:
        d = f['teleop/right_tau_xarm']
        d.attrs['unit'] = 'Nm'
        f['teleop/torque_clock'] = f['teleop/right_q_timestamp_us_xarm'][:]+100
        d.attrs['source_timestamp_path'] = '/teleop/torque_clock'
    with pytest.raises(ValueError, match='different source clocks'):
        read_episode(path)


def test_legacy_dual_arm_columns_and_each_arms_feedback_clock(tmp_path):
    import h5py
    path = tmp_path/'dual.h5'
    with h5py.File(path, 'w') as f:
        g = f.create_group('teleop')
        g.attrs['arm_names'] = ['left', 'right']
        g['timestamp_us'] = 1_000_000+np.arange(20)*10_000
        clocks = np.column_stack((g['timestamp_us'][:], g['timestamp_us'][:]+100))
        clocks[5, 0] = clocks[4, 0]
        g['q_follower_timestamp_us'] = clocks
        for signal in ('q_follower', 'dq_follower', 'tau_follower'):
            g[signal] = np.tile(np.r_[np.ones(7), np.full(7, 2.)], (20, 1))
        for flag in ('q_follower_valid', 'dq_valid_follower', 'torque_valid_follower'):
            g[flag] = np.ones((20, 1), dtype=np.uint8)
    left, right = read_episode(path, 'left'), read_episode(path, 'right')
    assert left['repeated_rows'] == 1 and right['repeated_rows'] == 0
    np.testing.assert_array_equal(left['tau'], 1.)
    np.testing.assert_array_equal(right['tau'], 2.)
    assert right['timestamp_us'][0]-left['timestamp_us'][0] == 100


def test_polynomial_velocity_has_known_acceleration():
    t = np.arange(101)*.01
    dq = np.broadcast_to((3*t**2+2*t+1)[:, None], (len(t), 7)).copy()
    data = dict(t=t, valid=np.ones(len(t), bool), q=dq.copy(), dq=dq, tau=np.zeros_like(dq))
    processed = preprocess(data)
    valid = processed['valid']
    np.testing.assert_allclose(processed['ddq'][valid],
        np.broadcast_to((6*t+2)[valid, None], (valid.sum(), 7)), atol=1e-10)
    assert not valid[0] and not valid[-1]


def test_independent_position_and_velocity_derivatives_match_known_motion():
    t = np.arange(101)*.01
    q = np.broadcast_to((t**3+t**2+t)[:, None], (len(t), 7)).copy()
    dq = np.broadcast_to((3*t**2+2*t+1)[:, None], (len(t), 7)).copy()
    processed = preprocess(dict(t=t, valid=np.ones(len(t), bool), q=q, dq=dq, tau=q*0))
    mask = processed['valid']
    np.testing.assert_allclose(processed['ddq_from_q'][mask], processed['ddq'][mask], atol=1e-9)


def test_differentiation_never_crosses_invalid_sample_or_gap():
    t = np.r_[np.arange(50)*.01, 1+np.arange(50)*.01]
    dq = np.zeros((100, 7))
    dq[50:] = 100
    valid = np.ones(len(t), bool)
    valid[25] = False
    processed = preprocess(dict(t=t, valid=valid, q=dq, dq=dq, tau=dq))
    mask = processed['valid']
    assert not mask[25] and not mask[49] and not mask[50]
    np.testing.assert_allclose(processed['ddq'][mask], 0, atol=1e-10)


def test_metrics_zero_scored_rows_and_joint_order():
    error = np.tile(np.arange(1, 8), (4, 1))
    assert metrics(error, np.zeros(4, bool)) == {'samples': 0}
    result = metrics(error, np.ones(4, bool))
    assert result['rmse'] == list(range(1, 8))
    assert result['bias'] == list(range(1, 8))


def test_mit_feedback_and_feedforward_are_added_once():
    q, dq = np.zeros(7), np.ones(7)
    result = mit_torque(q, dq, np.ones(7), np.full(7, 3.), np.full(7, 5.), kp=2., kd=4.)
    np.testing.assert_array_equal(result, 15.)
    feedforward = np.arange(7, dtype=float)
    np.testing.assert_array_equal(mit_torque(q, dq, q, dq, feedforward), feedforward)
    with pytest.raises(ValueError, match='kp'):
        mit_torque(q, dq, q, dq, feedforward, kp=-1)


def test_model_tau_roundtrip_is_independent_of_reported_torque():
    pytest.importorskip('pinocchio')
    pytest.importorskip('mujoco')
    c = DynamicsComparison(DEFAULT_URDF)
    rng = np.random.default_rng(11)
    q = rng.uniform(-.3, .3, (5, 7))
    dq = rng.uniform(-.1, .1, (5, 7))
    ddq = rng.uniform(-1., 1., (5, 7))
    state = dict(q=q, dq=dq, ddq=ddq, tau=np.zeros_like(q), valid=np.ones(5, bool))
    a, _ = c.evaluate(state, roundtrip=True)
    state['tau'][:] = 10.
    b, _ = c.evaluate(state, roundtrip=True)
    for result in (a, b):
        np.testing.assert_allclose(result['ddq_model_roundtrip'], ddq, atol=1e-9)
        np.testing.assert_allclose(result['ddq_mujoco_model_roundtrip'], ddq, atol=1e-8)
    assert np.max(np.abs(a['ddq_pin']-b['ddq_pin'])) > 1


def test_doubled_masses_also_pass_roundtrip_without_proving_physical_accuracy(tmp_path):
    pytest.importorskip('pinocchio')
    pytest.importorskip('mujoco')
    import xml.etree.ElementTree as ET

    root = ET.parse(DEFAULT_URDF).getroot()
    for mesh in root.iter('mesh'):
        mesh.set('filename', str((DEFAULT_URDF.parent/mesh.get('filename')).resolve()))
    for inertial in root.iter('inertial'):
        mass = inertial.find('mass')
        mass.set('value', str(2*float(mass.get('value'))))
        for key, value in list(inertial.find('inertia').attrib.items()):
            inertial.find('inertia').set(key, str(2*float(value)))
    path = tmp_path/'doubled.urdf'
    ET.ElementTree(root).write(path)
    original, doubled = DynamicsComparison(DEFAULT_URDF), DynamicsComparison(path)
    state = dict(q=np.full((3, 7), .2), dq=np.full((3, 7), .1),
                 ddq=np.full((3, 7), .8), tau=np.zeros((3, 7)), valid=np.ones(3, bool))
    a, _ = original.evaluate(state, roundtrip=True)
    b, _ = doubled.evaluate(state, roundtrip=True)
    np.testing.assert_allclose(b['tau_model'], 2*a['tau_model'], atol=1e-9)
    np.testing.assert_allclose(a['ddq_model_roundtrip'], state['ddq'], atol=1e-9)
    np.testing.assert_allclose(b['ddq_model_roundtrip'], state['ddq'], atol=1e-9)
    np.testing.assert_allclose(b['ddq_mujoco_model_roundtrip'], state['ddq'], atol=1e-8)


@pytest.mark.parametrize('friction', [False, True])
def test_actual_engines_match_for_known_acceleration_and_torque_perturbation(friction):
    pytest.importorskip('pinocchio')
    pytest.importorskip('mujoco')
    comparison = DynamicsComparison(DEFAULT_URDF, dict(enabled=friction, source='urdf'))
    rng = np.random.default_rng(7)
    q = rng.uniform(-.3, .3, (8, 7))
    dq = rng.uniform(-.1, .1, (8, 7))
    ddq = rng.uniform(-1, 1, (8, 7))
    tau = np.array([comparison.pin.rnea(qi, vi, ai)+comparison.friction.torque(vi)
                    for qi, vi, ai in zip(q, dq, ddq)])
    data = dict(q=q, dq=dq, ddq=ddq, tau=tau, valid=np.ones(8, bool))
    result, engine = comparison.evaluate(data)
    np.testing.assert_allclose(result['ddq_pin'], ddq, atol=1e-10)
    np.testing.assert_allclose(result['ddq_mujoco'], ddq, atol=1e-7)
    assert engine['max_mass_difference'] < 1e-8
    assert engine['max_bias_difference'] < 1e-7
    perturbation = np.array([.1, .2, -.1, .3, -.2, .1, .05])
    data['tau'] = tau+perturbation
    perturbed, _ = comparison.evaluate(data)
    np.testing.assert_allclose(perturbed['tau_model'], tau, atol=1e-10)
    assert np.max(np.abs(perturbed['ddq_pin']-ddq)) > .1


def test_end_to_end_keeps_contact_unknown_and_user_free_space_separate(tmp_path):
    pytest.importorskip('pinocchio')
    pytest.importorskip('mujoco')
    path = write_episode(tmp_path/'episode.h5')
    report, traces = analyze(path, DynamicsComparison(DEFAULT_URDF), free_space=(.5, 1.))
    assert report['contact_status'] == 'unknown'
    assert report['user_declared_free_space']['torque_error']['samples'] > 0
    assert report['user_declared_free_space']['torque_error']['samples'] < report['scored_rows']
    assert np.isnan(traces['ddq_pin'][~traces['valid']]).all()
    assert report['engine_acceleration_difference']['rmse'][0] < 1e-6
    confirmed, _ = analyze(path, DynamicsComparison(DEFAULT_URDF), no_contact=True)
    assert confirmed['contact_status'] == 'user_confirmed_no_contact'
    assert confirmed['torque_error'] == report['torque_error']


def test_torque_rollout_integrates_state_without_resetting_position():
    mj = pytest.importorskip('mujoco')
    pytest.importorskip('pinocchio')
    comparison = DynamicsComparison(DEFAULT_URDF)
    data = mj.MjData(comparison.model)
    q = np.array([.1, -.1, .2, 1.4, -.2, 1.2, -.1])
    dq, ddq = np.full(7, .02), np.arange(7)*.1
    data.qpos[comparison.qadr], data.qvel[comparison.vadr] = q, dq
    tau = comparison.pin.rnea(q, dq, ddq)
    integrate_torque(comparison, data, tau, 1e-5)
    np.testing.assert_allclose((data.qvel[comparison.vadr]-dq)/1e-5, ddq, atol=1e-7)
    assert np.linalg.norm(data.qpos[comparison.qadr]-q) > 0
    with pytest.raises(ValueError, match='duration'):
        integrate_torque(comparison, data, tau, 0)


def test_viewer_waits_for_graphics_cleanup_even_after_exception(monkeypatch):
    pytest.importorskip('mujoco')
    import mujoco.viewer
    import threading
    from types import SimpleNamespace

    shutdown, finished = threading.Event(), threading.Event()

    def launch(*args):
        def render():
            shutdown.wait()
            finished.set()
        worker = threading.Thread(target=render, daemon=True)
        worker.start()
        return SimpleNamespace(close=shutdown.set)

    monkeypatch.setattr(mujoco.viewer, 'launch_passive', launch)
    with pytest.raises(RuntimeError, match='replay failed'):
        with managed_viewer(None, None):
            raise RuntimeError('replay failed')
    assert finished.is_set()
