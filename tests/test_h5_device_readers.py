"""Read current device names and historical role names without changing files."""
from __future__ import annotations

import hashlib

import h5py
import numpy as np
import pytest

from inference.h5_observation_stream import H5ObservationEpisode
from nero_collection.h5_schema import public_dataset_name
from scripts.analyze_gello_feedback import read_arm
from scripts.rerun_episode import episode_channels


CHANNELS = {
    'q_follower': ('q', 7),
    'dq_follower': ('dq', 7),
    'ddq_follower': ('ddq', 7),
    'tau_follower': ('tau', 7),
    'wrench_ext': ('wrench_ext', 6),
}


def write_episode(path, arm_names, layout, *, metadata=True, alias_wrench=False):
    expected = {}
    with h5py.File(path, 'w') as episode:
        group = episode.create_group('teleop')
        if metadata:
            group.attrs['arm_names'] = arm_names
        group.create_dataset('timestamp_us', data=[1_000_000, 1_010_000, 1_020_000])
        for channel_index, (legacy, (_, width)) in enumerate(CHANNELS.items()):
            matrices = []
            for arm_index, arm in enumerate(arm_names):
                values = np.arange(3 * width, dtype=float).reshape(3, width)
                values += 1000 * arm_index + 100 * channel_index
                expected[arm, legacy] = values
                matrices.append(values)
                key = 'wrench_cal' if legacy == 'wrench_ext' and alias_wrench else legacy
                if layout == 'device':
                    group.create_dataset(public_dataset_name(arm, key), data=values)
                elif layout == 'split':
                    group.create_dataset(f'{arm}_{key}', data=values)
            if layout in {'concatenated', 'single'}:
                key = 'wrench_cal' if legacy == 'wrench_ext' and alias_wrench else legacy
                group.create_dataset(key, data=np.concatenate(matrices, axis=1))
        camera = episode.create_group('cameras/camera')
        camera.create_dataset('timestamp_us', data=[1_000_000])
        camera.create_dataset('frames', data=np.zeros((1, 2, 2, 3), np.uint8))
    return expected


@pytest.mark.parametrize('layout', ['device', 'split', 'concatenated'])
@pytest.mark.parametrize('arm_names', [('left', 'right'), ('right', 'left')])
def test_inference_reads_each_arm_and_original_order(tmp_path, layout, arm_names):
    path = tmp_path / 'episode.h5'
    expected = write_episode(path, arm_names, layout)
    digest = hashlib.sha256(path.read_bytes()).digest()
    for index, arm in enumerate(arm_names):
        by_name = H5ObservationEpisode.from_h5(path, camera_name='camera', arm_name=arm)
        by_index = H5ObservationEpisode.from_h5(path, camera_name='camera', arm_index=index)
        for legacy, (attribute, _) in CHANNELS.items():
            np.testing.assert_array_equal(getattr(by_name, attribute), expected[arm, legacy])
            np.testing.assert_array_equal(getattr(by_index, attribute), expected[arm, legacy])
        assert by_index.arm_name == arm
    assert hashlib.sha256(path.read_bytes()).digest() == digest


@pytest.mark.parametrize('layout', ['device', 'single'])
def test_inference_reads_right_only_without_relabeling(tmp_path, layout):
    path = tmp_path / 'episode.h5'
    expected = write_episode(path, ('right',), layout)
    view = H5ObservationEpisode.from_h5(path, camera_name='camera', arm_name='right')
    np.testing.assert_array_equal(view.q, expected['right', 'q_follower'])
    assert view.arm_name == 'right'


def test_inference_prefers_device_names_and_can_infer_single_side(tmp_path):
    path = tmp_path / 'episode.h5'
    expected = write_episode(path, ('right',), 'device', metadata=False)
    with h5py.File(path, 'r+') as episode:
        episode['teleop'].create_dataset('q_follower', data=np.full((3, 7), -999.))
    view = H5ObservationEpisode.from_h5(path, camera_name='camera')
    assert view.arm_name == 'right'
    np.testing.assert_array_equal(view.q, expected['right', 'q_follower'])


@pytest.mark.parametrize('layout', ['device', 'split', 'concatenated'])
def test_inference_wrench_aliases_follow_selected_arm(tmp_path, layout):
    path = tmp_path / 'episode.h5'
    expected = write_episode(path, ('right', 'left'), layout, alias_wrench=True)
    view = H5ObservationEpisode.from_h5(path, camera_name='camera', arm_name='left')
    np.testing.assert_array_equal(view.wrench_ext, expected['left', 'wrench_ext'])
    with pytest.raises(ValueError, match='missing inference datasets'):
        H5ObservationEpisode.from_h5(path, camera_name='camera', arm_name='left',
                                     allow_wrench_aliases=False)


def test_explicit_inference_dataset_override_stays_explicit(tmp_path):
    path = tmp_path / 'episode.h5'
    expected = write_episode(path, ('right',), 'device')
    with h5py.File(path, 'r+') as episode:
        episode['teleop'].create_dataset('custom_q', data=expected['right', 'q_follower'] + 42)
    view = H5ObservationEpisode.from_h5(path, camera_name='camera',
                                      datasets={'q': 'teleop/custom_q'})
    np.testing.assert_array_equal(view.q, expected['right', 'q_follower'] + 42)
    with pytest.raises(ValueError, match='missing inference datasets'):
        H5ObservationEpisode.from_h5(path, camera_name='camera',
                                     datasets={'q': 'teleop/missing'})


@pytest.mark.parametrize('layout', ['device', 'split', 'concatenated'])
def test_analysis_reads_per_arm_values_and_validity(tmp_path, layout):
    path = tmp_path / 'episode.h5'
    arms = ('right', 'left')
    expected = write_episode(path, arms, layout)
    with h5py.File(path, 'r+') as episode:
        group = episode['teleop']
        if layout == 'concatenated':
            group.create_dataset('dq_valid_follower', data=[[1, 0], [1, 0], [1, 0]])
        else:
            for index, arm in enumerate(arms):
                key = (public_dataset_name(arm, 'dq_valid_follower')
                       if layout == 'device' else f'{arm}_dq_valid_follower')
                group.create_dataset(key, data=np.full((3, 1), 1 - index))
    with h5py.File(path, 'r') as episode:
        for index, arm in enumerate(arms):
            np.testing.assert_array_equal(read_arm(episode['teleop'], 'q_follower', arm, arms),
                                          expected[arm, 'q_follower'])
            np.testing.assert_array_equal(read_arm(episode['teleop'], 'dq_valid_follower', arm, arms),
                                          np.full((3, 1), 1 - index))


@pytest.mark.parametrize('norm_name', ['right_tau_ext_l1_xarm', 'right_tau_ext_l1'])
def test_rerun_recognizes_physical_device_and_norm_channels(tmp_path, norm_name):
    path = tmp_path / 'episode.h5'
    write_episode(path, ('right',), 'device')
    with h5py.File(path, 'r+') as episode:
        group = episode['teleop']
        for name, width in [('right_q_gello', 7), ('right_current_cmd_gello', 7),
                            (norm_name, 1), ('sample_lateness_us', 1)]:
            group.create_dataset(name, data=np.zeros((3, width)))
    with h5py.File(path, 'r') as episode:
        names = episode_channels(episode['teleop'])
    assert {'right_q_xarm', 'right_q_gello', norm_name,
            'right_current_cmd_gello'} <= set(names)
    assert 'sample_lateness_us' not in names


def test_analysis_prefers_device_norm_and_accepts_old_norm_name(tmp_path):
    path = tmp_path / 'norm.h5'
    with h5py.File(path, 'w') as episode:
        group = episode.create_group('teleop')
        group.create_dataset('right_tau_ext_l1', data=np.full((3, 1), 9.))
    with h5py.File(path, 'r') as episode:
        np.testing.assert_array_equal(read_arm(episode['teleop'], 'tau_ext_l1', 'right', ('right',)), 9.)
    with h5py.File(path, 'r+') as episode:
        episode['teleop'].create_dataset('right_tau_ext_l1_xarm', data=np.full((3, 1), 3.))
    with h5py.File(path, 'r') as episode:
        np.testing.assert_array_equal(read_arm(episode['teleop'], 'tau_ext_l1', 'right', ('right',)), 3.)
