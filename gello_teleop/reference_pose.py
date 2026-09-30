"""Share live joint feedback and atomically replace a dual-arm reference pose."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

import numpy as np
import yaml

from gello_teleop.uf_robot_gello_teleop import load_configs

SIDES = ('left', 'right')
log = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parent/'config/xarm7_gello_dual_dataset.yaml'


def live_pose_path(config_path):
    identity = hashlib.sha256(str(Path(config_path).expanduser().resolve()).encode()).hexdigest()[:20]
    return Path(tempfile.gettempdir())/f'gello-reference-{os.getuid()}'/f'{identity}.json'


class ReferencePosePublisher:
    """Publish cached feedback outside all device and control threads."""
    def __init__(self, config_path, provider):
        self.config_path = Path(config_path).expanduser().resolve()
        self.path = live_pose_path(self.config_path)
        self.provider = provider
        self.session = uuid.uuid4().hex
        self.stop_event = threading.Event()
        self.thread = None
        self.lock_file = None

    def start(self):
        if self.thread is not None:
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.path.parent.is_symlink() or self.path.parent.stat().st_uid != os.getuid():
            raise RuntimeError('参考姿态缓存目录不属于当前用户')
        self.path.parent.chmod(0o700)
        self.lock_file = self.path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            self.lock_file = None
            raise RuntimeError('同一配置已有遥操进程发布关节反馈') from None
        self.thread = threading.Thread(target=self._run, name='reference-pose-publisher', daemon=True)
        self.thread.start()

    def _run(self):
        temporary = self.path.with_suffix(f'.{self.session}.tmp')
        try:
            while not self.stop_event.is_set():
                try:
                    pose = self.provider()
                    pose.update(format='dual_gello_live_pose_v1', unit='rad', pid=os.getpid(),
                                session=self.session, config_path=str(self.config_path),
                                published_monotonic_s=time.monotonic())
                    temporary.write_text(json.dumps(pose, allow_nan=False), encoding='utf-8')
                    temporary.chmod(0o600)
                    temporary.replace(self.path)
                except Exception:
                    log.exception('关节反馈缓存发布失败；遥操和录制继续运行')
                self.stop_event.wait(.025)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
            if self.thread.is_alive():
                raise RuntimeError('Reference pose publisher did not stop')
            try:
                if json.loads(self.path.read_text()).get('session') == self.session:
                    self.path.unlink(missing_ok=True)
            except (FileNotFoundError, ValueError):
                pass
        if self.lock_file is not None:
            self.lock_file.close()
            self.lock_file = None


def _vector(value, n, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f'{label} 必须包含 {n} 个有效关节角（rad）')
    return result


def _pose_sides(pose):
    sides = pose.get('active_arms', list(SIDES))
    if (not isinstance(sides, list) or not sides
            or any(side not in SIDES for side in sides) or len(set(sides)) != len(sides)):
        raise ValueError('Invalid active_arms in live pose')
    if set(pose['arms']) != set(sides):
        raise ValueError('Live pose arms do not match active_arms')
    return tuple(side for side in SIDES if side in sides)


def capture_current_pose(snapshot):
    """Capture the latest feedback at a keypress, without waiting or hardware I/O."""
    if not snapshot.get('valid'):
        raise RuntimeError(f"关节采样失败：{snapshot.get('reason', '无效反馈')}")
    pose = deepcopy(snapshot)
    now_us = time.time_ns()//1000
    for side in _pose_sides(pose):
        values = pose['arms'][side]
        for device, age_limit in (('xarm', pose['state_max_age_s']), ('gello', pose['leader_max_age_s'])):
            values[device+'_q'] = _vector(values[device+'_q'], 7, f'{side} {device} q').tolist()
            age = (now_us-int(values[device+'_timestamp_us']))/1e6
            if not np.isfinite(age) or not 0 <= age <= float(age_limit):
                raise RuntimeError(f'{side} {device} 关节反馈过期；原标定文件未修改')
            values[device+'_motion_deg'] = None  # a single snapshot makes no stationary-motion claim
    pose.update(captured_at=datetime.now(timezone.utc).isoformat(),
                captured_timestamp_us=now_us, capture_method='live teleoperation keypress snapshot',
                sample_count=1, max_motion_deg=None)
    return pose


def capture_live_pose(config_path, *, samples=5, max_motion_deg=1., timeout_s=3.):
    """Read distinct fresh source samples; never open a robot or a serial port."""
    if type(samples) is not int or samples < 2:
        raise ValueError('samples 至少为 2')
    if not np.isfinite([max_motion_deg, timeout_s]).all() or min(max_motion_deg, timeout_s) <= 0:
        raise ValueError('静止容忍度和超时必须为正数')
    config_path = Path(config_path).expanduser().resolve()
    workflow = yaml.safe_load(config_path.read_text())
    calibration_path = (config_path.parent/workflow.get('gello_config', 'xarm7_gello_calibration.yaml')).resolve()
    path = live_pose_path(config_path)
    collected = []
    previous_sources = None
    session = None
    deadline = time.monotonic()+timeout_s
    while len(collected) < samples:
        if time.monotonic() >= deadline:
            raise TimeoutError('未收到足够的所选臂新采样；原标定文件未修改')
        try:
            pose = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            raise RuntimeError('未找到实时关节反馈。请先用同一配置启动新版 dual_gello_collect，'
                               '遥操到目标位置后保持运行，再执行本脚本') from None
        if (pose.get('format') != 'dual_gello_live_pose_v1' or pose.get('unit') != 'rad'
                or pose.get('config_path') != str(config_path)
                or pose.get('calibration_path') != str(calibration_path)):
            raise RuntimeError('实时关节反馈与当前配置不一致')
        now = time.monotonic()
        age = now-float(pose['published_monotonic_s'])
        if not np.isfinite(age) or not 0 <= age <= .25:
            raise RuntimeError('实时关节反馈已过期；原标定文件未修改')
        if not pose.get('valid'):
            raise RuntimeError(f"关节采样失败：{pose.get('reason', '无效反馈')}")
        if session is not None and pose['session'] != session:
            raise RuntimeError('采样期间遥操进程发生变化；原标定文件未修改')
        session = pose['session']
        if _pose_sides(pose) != tuple(side for side in SIDES if side in workflow.get('active_arms', SIDES)):
            raise RuntimeError('实时反馈的 active_arms 与当前配置不一致')
        sources = []
        for side in _pose_sides(pose):
            values = pose['arms'][side]
            _vector(values['xarm_q'], 7, f'{side} xArm q')
            _vector(values['gello_q'], 7, f'{side} GELLO q')
            for device, age_limit in (('xarm', pose['state_max_age_s']), ('gello', pose['leader_max_age_s'])):
                source_age = (time.time_ns()//1000-int(values[device+'_timestamp_us']))/1e6
                if not 0 <= source_age <= float(age_limit):
                    raise RuntimeError(f'{side} {device} 关节反馈过期；原标定文件未修改')
            sources.extend((int(values['gello_sequence']), int(values['xarm_timestamp_us'])))
        if previous_sources is None or all(new > old for new, old in zip(sources, previous_sources)):
            collected.append(pose)
            previous_sources = sources
        elif any(new < old for new, old in zip(sources, previous_sources)):
            raise RuntimeError('关节采样序号倒退；原标定文件未修改')
        if len(collected) < samples:
            time.sleep(.01)

    result = deepcopy(collected[-1])
    result['sample_count'] = samples
    result['max_motion_deg'] = max_motion_deg
    for side in _pose_sides(pose):
        for device in ('xarm', 'gello'):
            values = np.asarray([pose['arms'][side][device+'_q'] for pose in collected])
            if device == 'gello':
                values = np.unwrap(values, axis=0)
            motion = np.rad2deg(np.ptp(values, axis=0))
            if np.any(motion > max_motion_deg):
                details = ', '.join(f'J{i+1}={value:.2f}°' for i, value in enumerate(motion)
                                    if value > max_motion_deg)
                raise RuntimeError(f'{side} {device} 采样期间移动超过 {max_motion_deg:g}°：{details}；'
                                   '请保持静止后重试，原标定文件未修改')
            result['arms'][side][device+'_q'] = values.mean(axis=0).tolist()
            result['arms'][side][device+'_motion_deg'] = motion.tolist()
    result['captured_at'] = datetime.now(timezone.utc).isoformat()
    return result


def overwrite_reference_pose(pose, *, wait_for_lock=True):
    """Replace both references in ONE file, keeping verified signs and gripper endpoints."""
    destination = Path(pose['calibration_path']).resolve()
    temporary = None
    with destination.with_suffix(destination.suffix+'.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait_for_lock else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError('标定文件正在由其他进程写入；此次未覆盖') from None
        sides = _pose_sides(pose)
        configs = load_configs(destination, dual=True, require_calibrated=True,
                               side=sides[0] if len(sides) == 1 else None)
        data = yaml.safe_load(destination.read_text())
        for side, robot, leader in configs:
            captured = pose['arms'][side]
            expected = dict(robot_ip=robot.robot_ip, port=leader.port, baudrate=leader.baudrate,
                            joint_ids=list(leader.joint_ids), joint_signs=list(leader.joint_signs),
                            gripper_id=leader.gripper_id)
            if captured['identity'] != expected:
                raise RuntimeError(f'{side} 当前设备或关节方向与标定文件不一致；未覆盖')
            n = len(leader.joint_ids)
            q = _vector(captured['xarm_q'], n, f'{side} xArm q')
            raw = _vector(captured['gello_q'], n, f'{side} GELLO q')
            if leader.joint_limits is not None:
                limits = np.asarray(leader.joint_limits)
                if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
                    raise ValueError(f'{side} 新 reset_q 超出配置的关节范围')
            status = data[side].get('CalibrationStatus', {})
            if status.get('reference_ready') is not True or status.get('direction_verified') != [True]*n:
                raise ValueError(f'{side} 原有方向标定不完整；不能仅更新参考姿态')
            data[side]['RobotConfig']['reset_q'] = q.tolist()
            teleop = data[side]['TeleoperatorConfig']
            teleop.update(start_joints=q.tolist(), leader_reference_q=raw.tolist(),
                          joint_offsets=(raw-q/np.asarray(leader.joint_signs)).tolist(),
                          leader_passive=True)
            status['reference_pose_capture'] = dict(
                captured_at=pose['captured_at'], method=pose.get('capture_method', 'stationary live teleoperation feedback'), unit='rad',
                sample_count=pose['sample_count'], max_motion_deg=pose['max_motion_deg'],
                xarm_timestamp_us=captured['xarm_timestamp_us'], gello_timestamp_us=captured['gello_timestamp_us'],
                xarm_motion_deg=captured['xarm_motion_deg'], gello_motion_deg=captured['gello_motion_deg'])
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                             prefix='.'+destination.name+'.', suffix='.tmp', delete=False) as output:
                temporary = Path(output.name)
                yaml.safe_dump(data, output, sort_keys=False, allow_unicode=True)
                output.flush()
                os.fsync(output.fileno())
            temporary.chmod(destination.stat().st_mode & 0o777)
            load_configs(temporary, dual=True, require_calibrated=True,
                         side=sides[0] if len(sides) == 1 else None)
            temporary.replace(destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination
