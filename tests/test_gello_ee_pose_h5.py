"""Synthetic collector persists the xArm SDK base-to-TCP matrices per arm."""
import time

import h5py
import numpy as np
import pytest
import yaml

from gello_teleop.dual_gello_collect import DualGelloPipeline
from scripts.smoke_dual_gello_pipeline import write_simulation_config
from test_dual_gello_pipeline import CONFIG, FakeArm, FakeReader


POSE_DATASETS = {'left': 'left_q_eepose_xarm', 'right': 'right_eepose_xarm'}


def sdk_pose(side):
    angle = .7 if side == 'left' else -1.1
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = [[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]]
    matrix[:3, 3] = [.123, -.456, .789] if side == 'left' else [-.654, .321, .987]
    return matrix


def make_pose_pipeline(tmp_path, arm_names, *, unavailable=None):
    config = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw.update(active_arms=list(arm_names), cameras=[], require_cameras=False)
    raw['gripper'] = {'enabled': False, 'command_enabled': False}
    config.write_text(yaml.safe_dump(raw))

    class PoseArm(FakeArm):
        def read_state(self):
            state = super().read_state()
            state.ee_pose = sdk_pose(self.name)
            if self.name == 'right' and unavailable == 'none':
                state.ee_pose = None
            elif self.name == 'right' and unavailable == 'wrong_shape':
                state.ee_pose = np.ones((3, 4))
            elif self.name == 'right' and unavailable == 'nonfinite':
                state.ee_pose[0, 3] = np.nan
            return state

    return DualGelloPipeline(config, arm_factory=PoseArm, reader_factory=FakeReader,
                             torque_plot_enabled=False)


def record_pose_episode(pipeline, output):
    pipeline.connect()
    pipeline.reset()
    pipeline.confirm_alignment()
    pipeline.takeover()
    pipeline.start_episode()
    deadline = time.monotonic() + 2.
    while pipeline.buffer.sample_count < 5:
        pipeline.poll()
        assert time.monotonic() < deadline
        time.sleep(.005)
    assert pipeline.control_error is None
    assert not pipeline.stop_event.is_set()
    assert pipeline.state == 'recording'
    buffer = pipeline.buffer
    pipeline.stop_episode(save=False)
    return buffer.save(output)


@pytest.mark.parametrize('arm_names', [('left',), ('right',), ('left', 'right'), ('right', 'left')])
def test_collector_h5_preserves_real_arm_state_pose_without_mixing_arms(tmp_path, arm_names):
    pipeline = make_pose_pipeline(tmp_path, arm_names)
    try:
        output = record_pose_episode(pipeline, tmp_path / 'pose_episode.h5')
        with h5py.File(output) as h5:
            teleop = h5['teleop']
            rows = len(teleop['timestamp_us'])
            assert set(teleop.attrs['required_eepose_datasets']) == set(POSE_DATASETS.values())
            assert len(teleop.attrs['required_xarm_datasets']) == 10
            assert list(teleop.attrs['arm_names']) == list(pipeline.ARM_NAMES)
            assert set(teleop.attrs['arm_names']) == set(arm_names)
            for side, name in POSE_DATASETS.items():
                dataset = teleop[name]
                assert dataset.shape == (rows, 4, 4)
                validity = teleop[f'{side}_eepose_valid_xarm']
                assert validity.shape == (rows, 1)
                assert dataset.attrs['validity_path'] == f'teleop/{side}_eepose_valid_xarm'
                assert dataset.attrs['arm_name'] == side
                assert dataset.attrs['device_name'] == 'xarm'
                assert dataset.attrs['reference_frame'] == 'base'
                assert dataset.attrs['frame_name'] == 'tcp'
                if side in arm_names:
                    np.testing.assert_array_equal(dataset[:], np.repeat(sdk_pose(side)[None], rows, axis=0))
                    assert validity[:].all()
                    assert dataset.attrs['source_timestamp_path'] == f'teleop/{side}_q_acquired_timestamp_us_xarm'
                else:
                    assert np.isnan(dataset[:]).all()
                    assert not validity[:].any()
                    assert dataset.attrs['placeholder']
                for value in dataset.attrs.values():
                    if isinstance(value, str) and value.startswith('teleop/'):
                        assert value in h5, (dataset.name, value)
    finally:
        pipeline.close()


@pytest.mark.parametrize('unavailable', ['none', 'wrong_shape', 'nonfinite'])
def test_unavailable_arm_pose_is_invalid_without_stopping_teleoperation(tmp_path, unavailable):
    pipeline = make_pose_pipeline(tmp_path, ('left', 'right'), unavailable=unavailable)
    try:
        output = record_pose_episode(pipeline, tmp_path / 'unavailable_pose.h5')
        with h5py.File(output) as h5:
            rows = len(h5['teleop/timestamp_us'])
            np.testing.assert_array_equal(h5['teleop/left_q_eepose_xarm'][:],
                np.repeat(sdk_pose('left')[None], rows, axis=0))
            assert h5['teleop/left_eepose_valid_xarm'][:].all()
            assert not h5['teleop/right_eepose_valid_xarm'][:].any()
            assert np.isnan(h5['teleop/right_eepose_xarm'][:]).all()
            assert h5['teleop/right_q_valid_xarm'][:].all()
    finally:
        pipeline.close()
