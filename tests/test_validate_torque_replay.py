import numpy as np
import pytest

from xarm_stack.validate_torque_replay import filter_torque, one_step, windows
from xarm_stack.validate_urdf import DEFAULT_URDF, DynamicsComparison


def model():
    pytest.importorskip('pinocchio')
    pytest.importorskip('mujoco')
    return DynamicsComparison(DEFAULT_URDF)


def test_increment_includes_initial_velocity_and_half_acceleration():
    c = model()
    dt = 1e-4
    q0 = np.array([.1, -.2, .1, 1.2, -.1, 1.3, -.2])
    v0, a0 = np.full(7, .3), np.arange(7)*.2
    delta = v0*dt+.5*a0*dt**2
    q = np.array([q0, q0+delta])
    dq = np.array([v0, v0+a0*dt])
    tau = np.array([c.pin.rnea(q0, v0, a0)]*2)
    report, traces = one_step(c, dict(t=np.array([0., dt]), q=q, dq=dq, tau=tau,
                                    valid=np.ones(2, bool)), physics_dt_s=dt/10)
    np.testing.assert_allclose(traces['delta_q_taylor'][0], delta, atol=1e-12)
    np.testing.assert_allclose(traces['q_taylor'][0], q[1], atol=1e-12)
    assert max(report['taylor_q_error']['rad']['rmse']) < 1e-12
    assert max(report['mujoco_q_error']['rad']['rmse']) < 1e-8
    assert max(report['velocity_only_baseline_error']['rad']['rmse']) > 1e-9


def test_gaps_and_invalid_rows_are_excluded_from_step_and_rollout():
    c = model()
    t = np.array([0., .01, .02, .2, .21, .22])
    q = np.tile([0., -.2, 0., 1.2, 0., 1.3, 0.], (len(t), 1))
    tau = np.array([c.pin.gravity_torque(row) for row in q])
    episode = dict(t=t, q=q, dq=np.zeros_like(q), tau=tau,
                   valid=np.array([True, True, False, True, True, True]))
    _, traces = one_step(c, episode)
    np.testing.assert_array_equal(traces['valid'], [True, False, False, True, True])
    report, _ = windows(c, episode, horizon_s=.02, stride_s=.01)
    assert report['windows'] == 1
    assert report['start_times_s'][0] == pytest.approx(.2)


def test_continuous_replay_does_not_overwrite_predicted_position():
    c = model()
    t = np.arange(21)*.005
    q = np.tile([0., -.2, 0., 1.2, 0., 1.3, 0.], (len(t), 1))
    tau = np.tile(c.pin.gravity_torque(q[0]), (len(t), 1))
    tau[:, 6] += .02
    episode = dict(t=t, q=q, dq=np.zeros_like(q), tau=tau, valid=np.ones(len(t), bool))
    report, previews = windows(c, episode, horizon_s=.05, stride_s=1.)
    assert report['windows'] == 1
    preview = previews[0]
    np.testing.assert_array_equal(preview['q_predicted'][0], q[0])
    assert np.linalg.norm(preview['q_predicted'][-1]-preview['q_recorded'][-1]) > .001
    assert np.max(np.abs(preview['q_predicted'][2:]-preview['q_predicted'][1:-1])) > 0


def test_noise_filter_changes_only_torque_and_discards_edges():
    t = np.arange(701)*.01
    q = np.tile(np.arange(7), (len(t), 1)).astype(float)
    tau = np.tile((2+np.sin(2*np.pi*30*t))[:, None], (1, 7))
    episode = dict(t=t, q=q, dq=q*.1, tau=tau, valid=np.ones(len(t), bool))
    raw = filter_torque(episode)
    np.testing.assert_array_equal(raw['tau'], tau)
    filtered = filter_torque(episode, 5.)
    np.testing.assert_array_equal(filtered['q'], q)
    np.testing.assert_array_equal(filtered['dq'], episode['dq'])
    assert not filtered['valid'][0] and not filtered['valid'][-1]
    assert np.std(filtered['tau'][filtered['valid'], 0]) < .02
    with pytest.raises(ValueError, match='cutoff'):
        filter_torque(episode, -1)
