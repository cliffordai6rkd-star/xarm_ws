import hashlib
import shutil
import warnings

import h5py
import numpy as np
import pytest
import yaml

from nero_collection.h5_schema import public_dataset_name
from scripts import visualize_h5_rerun as viewer


def fixture_episode(path, *, recorded_tau=True, method='rnea'):
    with h5py.File(path, 'w') as h5:
        group = h5.create_group('teleop')
        group.attrs['recorded_arm_names'] = ['right']
        group.create_dataset('timestamp_us', data=[1_000_000, 1_010_000, 1_020_000])
        for key, values in [('q_follower', np.ones((3, 7))), ('q_cmd', np.ones((3, 7)) * 2),
                            ('dq_follower', np.ones((3, 7)) * .2), ('ddq_follower', np.ones((3, 7)) * .3),
                            ('tau_follower', np.arange(21).reshape(3, 7) + 10.)]:
            group.create_dataset(public_dataset_name('right', key), data=values)
        pose = np.tile(np.eye(4), (3, 1, 1))
        pose[:, 0, 3] = [0., .1, .2]
        group.create_dataset('right_eepose_xarm', data=pose)
        if recorded_tau:
            tau = group.create_dataset('right_tau_ext_cal_xarm', data=np.arange(21).reshape(3, 7))
            group.create_dataset('cal_valid', data=[True, False, True])
            tau.attrs['validity_path'] = 'teleop/cal_valid'
        else:
            # A norm is not a seven-axis substitute.
            group.create_dataset('right_tau_ext_l1_xarm', data=np.ones((3, 1)) * 999.)
        camera = h5.create_group('cameras/right_wrist')
        camera.create_dataset('timestamp_us', data=[995_000, 1_035_000])
        camera.create_dataset('frames', data=np.zeros((2, 4, 4, 3), np.uint8))
        settings = {'force_feedback': {'dynamics_method': method, 'payload': {'source': 'controller'},
                    'state_filters': {k: {'enabled': False} for k in ('q', 'dq', 'ddq')},
                    'measured_torque_filter_enabled': False,
                    'friction': {'enabled': True, 'viscous_nm_s_per_rad': .5, 'coulomb_nm': 0.}}}
        h5.create_dataset('config_yaml', data=yaml.safe_dump(settings))


class FakeModel:
    def __init__(self, path, gravity):
        self.payload = None

    def rnea(self, q, dq, ddq):
        return np.arange(1., 8.) + q + .1 * dq + 2. * ddq

    def terms(self, q, dq):
        return dq.copy(), -np.arange(1., 8.)

    def set_payload(self, payload):
        self.payload = payload


def test_recorded_cal_is_preferred_validity_masked_and_camera_times_independent(tmp_path, monkeypatch):
    path = tmp_path / 'episode.h5'
    fixture_episode(path)
    digest = hashlib.sha256(path.read_bytes()).digest()
    monkeypatch.setattr(viewer, 'estimate_external_torque', lambda *a, **k: pytest.fail('must prefer recorded'))
    data = viewer.load_episode(path)
    assert data.sources['tau_ext_cal'] == 'teleop/right_tau_ext_cal_xarm'
    np.testing.assert_array_equal(data.signals['tau_ext_cal'][0], np.arange(7))
    assert np.isnan(data.signals['tau_ext_cal'][1]).all()
    np.testing.assert_allclose(data.time_s, [0., .01, .02])
    np.testing.assert_allclose(data.cameras['right_wrist'][0], [-.005, .035])
    assert data.ee_pose.shape == (3, 4, 4)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


def test_legacy_arm_selection_does_not_relabel_or_merge_other_arm_validity(tmp_path):
    with h5py.File(tmp_path / 'legacy.h5', 'w') as h5:
        g = h5.create_group('teleop')
        g.attrs['arm_names'] = ['right', 'left']
        g.create_dataset('timestamp_us', data=[1, 2])
        values = np.tile(np.r_[np.arange(7), np.arange(7) + 100.], (2, 1))
        ds = g.create_dataset('q_follower', data=values)
        g.create_dataset('valid', data=[[1, 0], [1, 1]])
        ds.attrs['validity_path'] = 'teleop/valid'
        right, _ = viewer.read_signal(g, 'q_follower', 'right')
        left, _ = viewer.read_signal(g, 'q_follower', 'left')
        np.testing.assert_array_equal(right, values[:, :7])
        assert np.isnan(left[0]).all()
        np.testing.assert_array_equal(left[1], values[1, 7:])
        del g['q_follower']
        g.attrs['arm_names'] = ['right']
        g.create_dataset('q_follower', data=values[:, :7])
        assert viewer.read_signal(g, 'q_follower', 'left') == (None, None)


@pytest.mark.parametrize('method', ['rnea', 'momentum'])
def test_offline_estimator_preserves_seven_axes_sign_friction_and_h5(tmp_path, monkeypatch, method):
    from xarm_stack import momentum
    monkeypatch.setattr(momentum, 'PinocchioMomentumModel', FakeModel)
    path = tmp_path / 'episode.h5'
    fixture_episode(path, recorded_tau=False, method=method)
    digest = hashlib.sha256(path.read_bytes()).digest()
    data = viewer.load_episode(path, no_images=True)
    tau = data.signals['tau_ext_cal']
    assert tau.shape == (3, 7)
    assert data.sources['tau_ext_cal'].startswith('OFFLINE')
    if method == 'rnea':
        baseline = np.arange(1., 8.) + 1. + .02 + .6 + .1
        np.testing.assert_allclose(tau, np.arange(21).reshape(3, 7) + 10. - baseline)
    else:
        assert np.isnan(tau[0]).all()
        assert np.isfinite(tau[1:]).all()
    assert any('payload was not recorded' in note for note in data.notes)
    assert any('SHA256' in note for note in data.notes)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


def test_no_estimate_does_not_plot_norm_as_seven_axis_external_torque(tmp_path):
    path = tmp_path / 'episode.h5'
    fixture_episode(path, recorded_tau=False)
    data = viewer.load_episode(path, estimate_tau_ext=False, no_images=True)
    assert 'tau_ext_cal' not in data.signals
    assert not data.cameras
    assert any('not recorded' in note for note in data.notes)


def test_momentum_replay_invalid_reports_gaps_and_duplicates(tmp_path, monkeypatch):
    from xarm_stack import momentum
    monkeypatch.setattr(momentum, 'PinocchioMomentumModel', FakeModel)
    path = tmp_path / 'episode.h5'
    fixture_episode(path, recorded_tau=False, method='momentum')
    with h5py.File(path, 'r+') as h5:
        g = h5['teleop']
        # First two collector rows represent the same robot report: no new integration step.
        g.create_dataset('right_q_timestamp_us_xarm', data=[[1_000_000], [1_000_000], [1_010_000]])
    data = viewer.load_episode(path, no_images=True)
    assert np.isnan(data.signals['tau_ext_cal'][:2]).all()
    assert np.isfinite(data.signals['tau_ext_cal'][2]).all()
    with h5py.File(path, 'r+') as h5:
        h5['teleop/right_q_timestamp_us_xarm'][:] = [[1_000_000], [1_040_000], [1_050_000]]
    data = viewer.load_episode(path, no_images=True)
    assert np.isnan(data.signals['tau_ext_cal']).all()  # gap reset followed by warmup
    with h5py.File(path, 'r+') as h5:
        g = h5['teleop']
        g['right_q_timestamp_us_xarm'][:] = [[1_000_000], [1_010_000], [1_020_000]]
        g.create_dataset('valid', data=[True, False, True])
        g['right_tau_xarm'].attrs['validity_path'] = 'teleop/valid'
    data = viewer.load_episode(path, no_images=True)
    assert np.isnan(data.signals['tau_ext_cal']).all()  # invalid sample resets integration


@pytest.mark.parametrize('estimated', [False, True])
def test_real_rerun_export_has_no_column_shape_warnings(tmp_path, estimated):
    pytest.importorskip('rerun')
    path = tmp_path / 'episode.h5'
    fixture_episode(path)
    data = viewer.load_episode(path)
    if estimated:
        data.sources['tau_ext_cal'] = 'OFFLINE URDF ESTIMATE (not recorded)'
    # Reject silently dropped columns (Rerun otherwise reports them as warnings).
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        viewer.log_episode(data, tmp_path / 'review.rrd')
    assert (tmp_path / 'review.rrd').stat().st_size > 1000


def test_viewer_spawn_finds_current_python_cli_when_shell_uses_other_env(tmp_path, monkeypatch):
    rr = pytest.importorskip('rerun')
    path = tmp_path / 'episode.h5'
    fixture_episode(path)
    data = viewer.load_episode(path)
    python_bin = tmp_path / 'pinn' / 'bin'
    python_bin.mkdir(parents=True)
    executable = python_bin / 'rerun'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o755)
    monkeypatch.setattr(viewer.sys, 'executable', str(python_bin / 'python'))
    monkeypatch.setenv('PATH', str(tmp_path / 'gello' / 'bin'))
    original_init = rr.init
    calls = []

    def fake_init(*args, **kwargs):
        calls.append((kwargs['spawn'], shutil.which('rerun')))
        # Verify the spawn environment without opening a window during tests.
        kwargs['spawn'] = False
        original_init(*args, **kwargs)
        rr.save(tmp_path / 'spawn_test.rrd')

    monkeypatch.setattr(rr, 'init', fake_init)
    viewer.log_episode(data)
    assert calls == [(True, str(executable))]
