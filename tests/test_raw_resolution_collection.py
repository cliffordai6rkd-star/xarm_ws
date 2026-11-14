"""Native-resolution recording entry: config override and H5 camera shapes."""
import time

import h5py
import pytest
import yaml

from gello_teleop.dual_gello_collect_raw_obs import RawObservationPipeline
from scripts.smoke_dual_gello_pipeline import (
    SimulatedArm,
    SimulatedLeader,
    write_simulation_config,
)

CONFIG = "gello_teleop/config/xarm7_gello_dual_dataset.yaml"
RAW_CAMERAS = [
    dict(name='front', backend='mock', width=64, height=48, fps=30.,
         output_size=[224, 224], output_fps=25., depth=False, visualize=False),
    dict(name='wrist', backend='mock', width=80, height=60, fps=30.,
         output_size=[224, 224], output_fps=25., depth=False, visualize=False),
]


def make_pipeline(tmp_path, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = write_simulation_config(CONFIG, tmp_path)
    data = yaml.safe_load(cfg.read_text())
    data['cameras'] = [dict(camera) for camera in RAW_CAMERAS]
    data['require_cameras'] = True
    data['gripper'] = dict(enabled=False, command_enabled=False)
    cfg.write_text(yaml.safe_dump(data))
    return RawObservationPipeline(cfg, output_dir=tmp_path/'raw',
                                  arm_factory=SimulatedArm,
                                  reader_factory=SimulatedLeader, **kwargs)


def test_native_config_and_separate_output(tmp_path):
    pipeline = make_pipeline(tmp_path)
    try:
        cameras = {camera.name: camera for camera in pipeline.collection_config.cameras}
        assert cameras['front'].output_size is None
        assert cameras['wrist'].output_size is None
        assert cameras['front'].output_fps is None  # keep every acquired frame
        assert cameras['front'].effective_fps == 30.
        assert pipeline.collection_config.output.directory == (tmp_path/'raw').resolve()
        raw = yaml.safe_load(pipeline.collection_config.raw_yaml)
        assert 'output_size' not in raw['cameras'][0]
        assert raw['cameras'][0]['output_fps'] is None
        assert raw['output']['directory'] == str((tmp_path/'raw').resolve())
    finally:
        pipeline.close()


def test_raw_fps_subsamples_and_rejects_above_capture(tmp_path):
    pipeline = make_pipeline(tmp_path, output_fps=10.)
    try:
        assert pipeline.collection_config.cameras[0].effective_fps == 10.
    finally:
        pipeline.close()
    with pytest.raises(ValueError, match='--raw-fps'):
        make_pipeline(tmp_path/'too-fast', output_fps=60.)


def test_episode_stores_native_resolution_frames(tmp_path):
    pipeline = make_pipeline(tmp_path)
    try:
        pipeline.connect()
        pipeline.reset()
        pipeline.confirm_alignment()
        pipeline.takeover()
        pipeline.start_episode()
        deadline = time.monotonic()+0.8
        while time.monotonic() < deadline:
            pipeline.poll()
            time.sleep(0.005)
        pipeline.stop_episode(save=False)
        target = tmp_path/'raw_episode.h5'
        pipeline.buffer.save(target)
    finally:
        pipeline.close()
    with h5py.File(target) as h5:
        assert h5['cameras/front/frames'].shape[1:] == (48, 64, 3)
        assert h5['cameras/wrist/frames'].shape[1:] == (60, 80, 3)
        assert h5['cameras/front/frames'].shape[0] > 0
        # The H5 config metadata reflects the effective raw settings.
        raw = yaml.safe_load(h5['config_yaml'][()].decode())
        assert 'output_size' not in raw['cameras'][0]
        assert str((tmp_path/'raw').resolve()) == raw['output']['directory']
