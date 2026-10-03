#!/usr/bin/env python3
"""Exercise dual-arm collection with simulated robots/grippers; never connect an arm.

--real-cameras tests the configured camera streams with simulated arm inputs.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gello_teleop.dual_gello_collect import DualGelloPipeline
from nero_collection.arms.base import ArmState, GripperState
from nero_collection.h5_schema import dataset_candidates


class SimulatedArm:
    def __init__(self, side, robot):
        self.name, self.dof = side, len(robot.reset_q)
        self.q = np.asarray(robot.reset_q).copy()
        self.width = .085
        self.lock = threading.Lock()

    def connect(self): pass
    def disconnect(self): pass
    def enable(self): pass
    def init_gripper(self): pass

    def move_to_reset(self, q, **_):
        self.command_joint_positions(q)

    def command_joint_positions(self, q):
        with self.lock:
            self.q = np.asarray(q).copy()

    def read_state(self):
        ts = time.time_ns()//1000
        with self.lock:
            q = self.q.copy()
        return ArmState(q=q, dq=np.zeros(self.dof), ddq=np.zeros(self.dof), ee_pose=np.eye(4),
                        torque=np.full(self.dof, np.nan), current=np.full(self.dof, np.nan),
                        timestamp_us=ts, acquired_timestamp_us=ts, torque_valid=False,
                        current_valid=False, feedback_source='synthetic')

    def read_gripper_state(self):
        with self.lock:
            return GripperState(self.width, np.nan, time.time_ns()//1000, 'width')

    def command_gripper(self, value, **_):
        with self.lock:
            self.width = value


class SimulatedLeader:
    def __init__(self, config):
        self.config = config
        self.started = time.monotonic()

    def prepare_alignment(self): pass
    def close(self): pass
    def set_torque(self, *_args, **_kwargs): pass

    def read(self):
        t = time.monotonic()-self.started
        raw = np.asarray(self.config.leader_reference_q).copy()
        raw += .005*np.sin(t*2+np.arange(len(raw)))*np.asarray(self.config.joint_signs)
        if self.config.gripper_id >= 0:
            opening = 0. if self.config.gripper_open_deg is None else np.deg2rad(self.config.gripper_open_deg)
            closing = opening+np.deg2rad(self.config.gripper_travel_deg) if self.config.gripper_close_deg is None else np.deg2rad(self.config.gripper_close_deg)
            raw = np.r_[raw, opening+(closing-opening)*(.3+.25*np.sin(t*3))]
        return np.r_[raw, np.zeros(len(self.config.torque_joint_ids or ()))]


def inspect_episode(path):
    with h5py.File(path) as h5:
        g = h5['teleop']
        time_us = g['timestamp_us'][:]
        arm_names = [name.decode() if isinstance(name, bytes) else str(name)
                     for name in g.attrs['arm_names']]
        def channel(arm, name):
            for candidate in dataset_candidates(arm, name):
                if candidate in g:
                    values = g[candidate][:]
                    if candidate == name and len(arm_names) > 1:
                        width = values.shape[1] // len(arm_names)
                        offset = arm_names.index(arm) * width
                        return values[:, offset:offset + width]
                    return values
            raise KeyError(f'{arm} channel {name!r} is missing')
        if len(time_us) < 10 or not np.all(np.diff(time_us)>0):
            raise RuntimeError('Episode needs >=10 strictly increasing state timestamps')
        metadata = json.loads(h5['metadata/episode_json'][()])
        for arm in arm_names:
            for key in ['q_cmd', 'q_follower', 'q_leader_raw', 'q_leader_mapped']:
                data = channel(arm, key)
                if data.shape != (len(time_us), 7) or not np.isfinite(data[:]).all():
                    raise RuntimeError(f'Invalid {arm} {key}')
            np.testing.assert_allclose(channel(arm, 'delta_q')[:],
                                       channel(arm, 'q_cmd')[:]-channel(arm, 'q_follower')[:])
            causal = channel(arm, 'q_cmd_send_ok')[:].astype(bool)
            acquired = channel(arm, 'q_follower_acquired_timestamp_us')[:]
            if not causal.any() or np.any(channel(arm, 'q_cmd_timestamp_us')[:][causal] > acquired[causal]):
                raise RuntimeError(f'{arm} joint commands are not causal at follower acquisition')
            valid = channel(arm, 'gripper_cmd_valid')[:].astype(bool)
            if (not channel(arm, 'gripper_follower_valid')[:].any()
                    or (metadata.get('gripper_mode') != 'trigger' and not valid.any())):
                raise RuntimeError(f'{arm} gripper command and feedback channels must be present')
            if np.any(channel(arm, 'gripper_cmd_timestamp_us')[:][valid] > acquired[valid]):
                raise RuntimeError(f'{arm} gripper commands are not causal')
            widths = channel(arm, 'gripper_cmd')[:][valid]
            if np.any(widths < 0) or np.any(widths > .085):
                raise RuntimeError(f'{arm} gripper width outside G1 interval')
        cameras = {}
        for name, camera in h5.get('cameras', {}).items():
            timestamps = camera['timestamp_us'][:]
            if len(timestamps)<2 or not np.all(np.diff(timestamps)>0):
                raise RuntimeError(f'Invalid camera timeline {name}')
            cameras[name] = dict(frames=len(timestamps), rgb_shape=list(camera['frames'].shape),
                                 depth_shape=list(camera['depth'].shape) if 'depth' in camera else None)
        return dict(path=str(Path(path).resolve()), samples=len(time_us),
                    timeline_hz=float(1e6/np.median(np.diff(time_us))), cameras=cameras,
                    metadata=json.loads(h5['metadata/episode_json'][()]))


def write_simulation_config(config_path, directory, real_cameras=False):
    """Generate temporary SYNTHETIC mappings; never depend on real calibrations."""
    directory = Path(directory)
    raw = yaml.safe_load(Path(config_path).read_text())
    template = yaml.safe_load((ROOT/'gello_teleop/config/xarm7_gello_teleop_dual.yaml').read_text())
    for entry in template.values():
        leader = entry['TeleoperatorConfig']
        n = len(leader['joint_ids'])
        q = np.asarray(entry['RobotConfig']['reset_q'])
        ref = np.linspace(2.5, 3.5, n)
        leader.update(leader_reference_q=ref.tolist(), joint_offsets=(ref-q/leader['joint_signs']).tolist(),
                      leader_passive=True, damping_enabled=False)
        entry['CalibrationStatus'] = dict(reference_ready=True, direction_verified=[True]*n, synthetic=True)
    calibration = directory/'synthetic_calibration.yaml'
    calibration.write_text(yaml.safe_dump(template))
    raw['gello_config'] = str(calibration)
    # Synthetic readers never energize motors or claim to validate haptics.
    raw['gello_damping'] = {'enabled': False}
    raw['force_feedback'] = {'enabled': False}
    raw['torque_visualization'] = {'enabled': False}
    raw.setdefault('alignment', {}).pop('gello_hold_current_raw', None)
    for arm in raw['arms'].values():
        arm['execution_enabled'] = False
    if not real_cameras:
        for camera in raw.get('cameras', []):
            camera.update(backend='mock', width=64, height=48, depth=False, visualize=False)
    cfg = directory/'simulation.yaml'
    cfg.write_text(yaml.safe_dump(raw))
    return cfg


def run_smoke(config_path, output, duration=2., real_cameras=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(duration) or duration < .5:
        raise ValueError('duration must be at least 0.5 seconds')
    with tempfile.TemporaryDirectory() as directory:
        cfg = write_simulation_config(config_path, directory, real_cameras)
        pipeline = DualGelloPipeline(cfg, arm_factory=SimulatedArm, reader_factory=SimulatedLeader)
        try:
            pipeline.connect()
            pipeline.reset()
            pipeline.confirm_alignment()
            pipeline.takeover()
            pipeline.start_episode()
            pipeline.buffer.episode_metadata.update(synthetic_robots=True,
                                                   synthetic_cameras=not real_cameras,
                                                   validation_scope='software pipeline; no physical arm/gripper validation')
            deadline = time.monotonic()+duration
            while time.monotonic()<deadline:
                pipeline.poll()
                time.sleep(.005)
            pipeline.stop_episode(save=False)
            output.parent.mkdir(parents=True, exist_ok=True)
            pipeline.buffer.save(output)
        finally:
            pipeline.close()
    return inspect_episode(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'gello_teleop/config/xarm7_gello_dual_dataset.yaml'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=float, default=2.)
    parser.add_argument('--real-cameras', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config, args.output, args.duration, args.real_cameras), indent=2))


if __name__ == '__main__':
    main()
