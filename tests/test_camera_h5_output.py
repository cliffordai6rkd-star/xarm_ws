"""Raw image persistence and independent camera/robot acquisition timelines."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import pytest
import yaml

from nero_collection.config import CameraConfig, OutputConfig, _parse_output
from nero_collection.h5_schema import public_dataset_name
from nero_collection.h5_writer import EpisodeBuffer


POSE_DATASETS = {'left': 'left_q_eepose_xarm', 'right': 'right_eepose_xarm'}


@pytest.mark.parametrize('arm_names', [('left', 'right'), ('right', 'left')])
def test_dual_arm_h5_splits_values_and_metadata_without_mixing_arms(tmp_path, arm_names):
    from test_dual_gello_pipeline import make_pipeline
    from inference.h5_observation_stream import H5ObservationEpisode
    pipeline = make_pipeline()
    try:
        buffer = EpisodeBuffer(pipeline.collection_config, arm_names, enable_online_tau_ext=False)
        poses = np.stack([np.eye(4), np.eye(4)])
        poses[1, 0, 3] = .5
        for index in range(3):
            q = np.arange(14, dtype=float) + index
            buffer.append_teleop(1_000_000 + index * 10_000, {
                'q_follower': ('q', q), 'dq_follower': ('velocity', q + 10),
                'ddq_follower': ('acceleration', q + 20),
                'tau_follower': ('torque', q + 30), 'q_cmd': ('q', q + 1),
                'delta_q': ('q_error', np.ones(14)),
                'dq_valid_follower': ('validity', np.array([1, 0], np.uint8)),
                'ddq_valid_follower': ('validity', np.array([0, 1], np.uint8)),
                'q_follower_timestamp_us': ('timestamp', np.array([100 + index, 200 + index], np.int64)),
                'gripper_follower': ('gripper', np.array([.01, .07])),
                'ee_pose_follower': ('ee_pose', poses),
                'sample_lateness_us': ('duration', np.array([index], np.int64)),
            })
        buffer.append_camera('camera', 1_000_000, np.zeros((8, 8, 3), np.uint8))
        path = buffer.save(tmp_path / 'split.h5')
        with h5py.File(path, 'r+') as episode:
            rows = episode['teleop']
            assert 'q_follower' not in rows and 'q_cmd' not in rows
            for index, arm in enumerate(arm_names):
                expected_q = np.stack(buffer.teleop_data['q_follower'])[:, index * 7:(index + 1) * 7]
                np.testing.assert_array_equal(rows[f'{arm}_q_xarm'][:], expected_q)
                assert rows[f'{arm}_q_xarm'].shape == (3, 7)
                np.testing.assert_array_equal(rows[f'{arm}_q_timestamp_us_xarm'][:, 0],
                                              (100 if index == 0 else 200) + np.arange(3))
                assert rows[f'{arm}_dq_valid_xarm'].shape == (3, 1)
                assert np.all(rows[f'{arm}_dq_valid_xarm'][:] == 1 - index)
                np.testing.assert_array_equal(rows[f'{arm}_gripper_xarm'][:, 0],
                                              np.full(3, .01 if index == 0 else .07))
                np.testing.assert_array_equal(rows[POSE_DATASETS[arm]][:],
                                              np.repeat(poses[index:index + 1], 3, axis=0))
                assert rows[f'{arm}_dq_xarm'].attrs['validity_path'] == f'teleop/{arm}_dq_valid_xarm'
                assert rows[f'{arm}_ddq_xarm'].attrs['validity_path'] == f'teleop/{arm}_ddq_valid_xarm'
                assert rows[f'{arm}_dq_xarm'].attrs['timestamp_path'] == 'teleop/timestamp_us'
                assert rows[f'{arm}_q_cmd_xarm'].attrs['arm_name'] == arm
                assert f'{arm}_q_cmd_xarm' in rows.attrs['command_datasets']
                # Inference additionally requires a wrench, supplied by its own pipeline.
                rows.create_dataset(f'{arm}_wrench_ext_xarm', data=np.zeros((3, 6)))
            assert rows['sample_lateness_us'].shape == (3, 1)
        for index, arm in enumerate(arm_names):
            view = H5ObservationEpisode.from_h5(path, camera_name='camera', arm_name=arm)
            view_by_index = H5ObservationEpisode.from_h5(path, camera_name='camera', arm_index=index)
            expected_q = np.stack(buffer.teleop_data['q_follower'])[:, index * 7:(index + 1) * 7]
            np.testing.assert_array_equal(view.q, expected_q)
            np.testing.assert_array_equal(view_by_index.q, expected_q)
            np.testing.assert_array_equal(view.dq, expected_q + 10)
            np.testing.assert_array_equal(view.tau, expected_q + 30)
        # Legacy episodes remain readable with their concatenated channels.
        with h5py.File(path, 'r+') as episode:
            rows = episode['teleop']
            for name in ('q_follower', 'dq_follower', 'ddq_follower', 'tau_follower', 'wrench_ext'):
                rows.create_dataset(name, data=np.concatenate(
                    [rows[public_dataset_name(arm, name)][:] for arm in arm_names], axis=1))
                for arm in arm_names:
                    del rows[public_dataset_name(arm, name)]
        for index, arm in enumerate(arm_names):
            legacy = H5ObservationEpisode.from_h5(path, camera_name='camera', arm_name=arm)
            expected_q = np.stack(buffer.teleop_data['q_follower'])[:, index * 7:(index + 1) * 7]
            np.testing.assert_array_equal(legacy.q, expected_q)
            np.testing.assert_array_equal(legacy.ddq, expected_q + 20)
    finally:
        pipeline.close()


@pytest.mark.parametrize('compression', [None, 'gzip', 'lzf'])
def test_raw_h5_preserves_30hz_images_and_100hz_joint_rows_with_bounded_copy(
        tmp_path, monkeypatch, compression):
    import nero_collection.h5_writer as writer
    from test_dual_gello_pipeline import make_pipeline
    pipeline = make_pipeline()
    config = replace(pipeline.collection_config,
        output=replace(pipeline.collection_config.output, camera_compression=compression),
        cameras=(CameraConfig('right wrist', fps=30, output_size=(224, 224), depth=True),))
    buffer = EpisodeBuffer(config, ('right',), enable_online_tau_ext=False)
    for index in range(100):
        q = np.arange(7, dtype=float)+index/100.
        buffer.append_teleop(1_000_000+index*10_000, {
            'q_follower': ('q', q), 'dq_follower': ('dq', q+1),
            'ddq_follower_raw': ('ddq', q+2), 'tau_follower': ('tau', q+3),
        })
    for index in range(30):
        buffer.append_camera('right wrist', 1_000_000+round(index*1e6/30),
                             np.full((224, 224, 3), index, np.uint8),
                             np.full((224, 224), index*100, np.uint16))
    monkeypatch.setattr(writer, '_CAMERA_WRITE_BATCH_BYTES', 2*224*224*3)
    original_stack = np.stack
    def bounded_stack(values, *args, **kwargs):
        if values and np.asarray(values[0]).shape[:2] == (224, 224):
            assert sum(np.asarray(value).nbytes for value in values) <= writer._CAMERA_WRITE_BATCH_BYTES
        return original_stack(values, *args, **kwargs)
    monkeypatch.setattr(writer.np, 'stack', bounded_stack)
    path = buffer.save(tmp_path/'episode.h5')
    assert not path.with_suffix('.h5.tmp').exists()
    with h5py.File(path) as episode:
        rows = episode['teleop']
        assert rows['right_q_xarm'].shape == (100, 7)
        assert rows.attrs['sample_rate_hz'] == 100
        np.testing.assert_array_equal(rows['timestamp_us'][:], buffer.teleop_timestamps_us)
        for key in ('q_follower', 'dq_follower', 'ddq_follower_raw', 'tau_follower'):
            np.testing.assert_array_equal(rows[public_dataset_name('right', key)][:],
                                          original_stack(buffer.teleop_data[key]))
        camera = episode['cameras/right wrist']
        assert camera['frames'].shape == (30, 224, 224, 3)
        assert camera['depth'].shape == (30, 224, 224)
        assert camera['frames'].dtype == np.uint8 and camera['depth'].dtype == np.uint16
        assert camera['frames'].compression == camera['depth'].compression == compression
        assert camera.attrs['fps'] == 30
        np.testing.assert_array_equal(camera['timestamp_us'][:], buffer.camera_timestamps_us['right wrist'])
        for index in range(30):
            np.testing.assert_array_equal(camera['frames'][index], buffer.camera_frames['right wrist'][index])
            np.testing.assert_array_equal(camera['depth'][index], buffer.camera_depth_frames['right wrist'][index])
    pipeline.close()


def test_realsense_resizes_rgb_and_aligned_depth_before_recording_without_hardware(monkeypatch):
    from nero_collection.cameras import RealSenseCamera
    monkeypatch.setitem(sys.modules, 'pyrealsense2', SimpleNamespace())
    rgb = np.full((480, 640, 3), [10, 20, 30], dtype=np.uint8)
    depth = np.tile(np.arange(640, dtype=np.uint16), (480, 1))*100
    frames = SimpleNamespace(
        get_timestamp=lambda: 33.333,
        get_color_frame=lambda: SimpleNamespace(get_data=lambda: rgb),
        get_depth_frame=lambda: SimpleNamespace(get_data=lambda: depth))
    camera = RealSenseCamera(CameraConfig('right wrist', serial_number='offline',
        width=640, height=480, fps=30, depth=True, output_size=(224, 224), visualize=True))
    camera._pipeline = SimpleNamespace(poll_for_frames=lambda: frames)
    frame = camera.poll()
    assert frame.frame.shape == (224, 224, 3)
    assert frame.depth.shape == (224, 224)
    assert frame.frame.dtype == np.uint8 and frame.depth.dtype == np.uint16
    np.testing.assert_array_equal(frame.frame[0, 0], [10, 20, 30])
    assert np.isin(frame.depth, depth).all()  # Nearest-neighbor retains depth values.
    assert frame.preview_frame.shape == (480, 640, 3)
    assert camera.poll() is None  # A repeated SDK frame is not copied onto a 100 Hz timeline.


@pytest.mark.parametrize('path', [
    'gello_teleop/config/xarm7_gello_dual_dataset.yaml',
    'gello_teleop/config/xarm7_gello_full_pipeline.yaml',
])
def test_collection_examples_use_224_rgb_depth_30hz_and_uncompressed_h5(path):
    config = yaml.safe_load(Path(path).read_text())
    assert config['control']['sample_rate_hz'] == 100
    assert config['output']['camera_compression'] is None
    assert all(camera['fps'] == 30 and camera['output_size'] == [224, 224]
               for camera in config.get('cameras', []) if camera.get('enabled', True))


def test_output_parser_retains_explicit_raw_camera_storage(tmp_path):
    assert _parse_output({'camera_compression': None}, tmp_path).camera_compression is None
    assert _parse_output({}, tmp_path).camera_compression == 'gzip'
    with pytest.raises(ValueError, match='camera_compression'):
        OutputConfig(tmp_path, camera_compression='mp4')
