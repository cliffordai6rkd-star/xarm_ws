#!/usr/bin/env python3
"""Single/dual GELLO: reset robots, manually align leaders, then follow."""
import argparse
import logging
import os
from pathlib import Path
import queue
import select
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ufactory_devices.robot import UFRobotConfig
from gello_teleop.gello_hardware import Arm, GelloReader
from gello_teleop.reset_display import ResetDisplay
from gello_teleop.mit_position import MitPositionController, PinocchioArmModel

logger = logging.getLogger('gello_teleop')


@dataclass
class GelloRobotConfig(UFRobotConfig):
    robot_mode: int = 6
    robot_speed: float = 20
    robot_acc: float = 100
    reset_q: Optional[Tuple[float, ...]] = None
    reset_speed: float = 10  # deg/s, independent of following speed
    reset_acc: float = 50  # deg/s^2
    reset_timeout: float = 60
    reset_tolerance: float = 0.03  # rad
    command_timeout: float = 0.5
    collision_sensitivity: Optional[int] = None  # preserve controller setting


@dataclass
class GelloTeleopConfig:
    fps: float = 30
    port: str = ''
    baudrate: int = 57600
    joint_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    joint_signs: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1)
    start_joints: Optional[Tuple[float, ...]] = None  # legacy, must match reset_q
    gripper_id: int = 8
    torque_joint_ids: Tuple[int, ...] = ()
    gripper_open_deg: Optional[float] = None  # absolute encoder endpoints, both or neither
    gripper_close_deg: Optional[float] = None
    gripper_travel_deg: float = -42  # from the manually opened alignment pose
    read_timeout: float = 0.2
    watchdog_timeout: float = 1.0
    alignment_tolerance: float = 0.05  # rad
    max_joint_step: float = 0.25  # raw sample jump rejection, rad
    max_joint_velocity: float = 0.5  # commanded target slew limit, rad/s
    joint_limits: Optional[Tuple[Tuple[float, float], ...]] = None  # optional stricter rad limits
    control_mode: str = 'position'  # position or mit_to_position
    dynamics_urdf: Optional[str] = None
    dynamics_joint_names: Optional[Tuple[str, ...]] = None  # exact SDK J1..Jn order
    dynamics_locked_joint_names: Tuple[str, ...] = ()
    mit_kp: Optional[Tuple[float, ...]] = None  # Nm/rad
    mit_kd: Optional[Tuple[float, ...]] = None  # Nm/(rad/s)
    mit_torque_limit_nm: Optional[Tuple[float, ...]] = None
    mit_acceleration_limit: float = 1.0  # rad/s^2
    mit_tracking_error_limit: float = 0.15  # max command minus measured q, rad
    # Filled by calibrate_joint_directions.py.  When present, startup uses
    # these values verbatim and does not infer an offset from a live pose.
    joint_offsets: Optional[Tuple[float, ...]] = None
    leader_reference_q: Optional[Tuple[float, ...]] = None
    leader_passive: bool = False  # Manual reference check; never move/hold the leader.
    passive_start_tolerance_deg: float = 10.0  # Bounded, slew-limited startup catch-up.
    leader_reset_speed_deg: float = 10.0
    leader_reset_timeout: float = 30.0
    leader_reset_max_travel_deg: float = 90.0
    # Optional holding settings accepted by existing dual YAML files.  The
    # position-mode driver keeps the leader at its reference pose.
    hold_pwm: Optional[int] = None
    hold_pwm_by_joint: Optional[Tuple[int, ...]] = None
    # Optional, model-gated GELLO damping.  Values are Dynamixel current
    # command units; they are deliberately not labelled Nm without a motor
    # calibration.  The reader refuses unknown servo models.
    damping_enabled: bool = False
    damping_mode: str = 'none'  # none or current
    damping_gain: Optional[Tuple[float, ...]] = None  # raw current/(rad/s)
    damping_brake_gain: Optional[Tuple[float, ...]] = None
    damping_current_limit: Optional[Tuple[int, ...]] = None
    damping_velocity_threshold: float = 1.0  # rad/s
    damping_velocity_filter_alpha: float = 0.35
    weak_hold_enabled: bool = False
    weak_hold_gain: Optional[Tuple[float, ...]] = None  # raw current/rad
    weak_hold_limit: Optional[Tuple[int, ...]] = None
    weak_hold_release_velocity: float = 0.08
    damping_watchdog_ms: int = 100


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def validate(robot, leader):
    n = len(leader.joint_ids)
    if n not in (5, 6, 7) or len(leader.joint_signs) != n:
        raise ValueError('joint_ids/joint_signs must match the 5/6/7 robot axes')
    if not all(sign in (-1, 1) for sign in leader.joint_signs):
        raise ValueError('joint_signs must be +1 or -1')
    ids = list(leader.joint_ids) + list(leader.torque_joint_ids or ())
    if leader.gripper_id != -1:
        ids.append(leader.gripper_id)
    if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i <= 252 for i in ids):
        raise ValueError('Dynamixel IDs must be unique integers in [0, 252]')
    if not leader.port or not robot.robot_ip:
        raise ValueError('GELLO port and robot_ip are required')
    if robot.robot_mode != 6:
        raise ValueError('GELLO requires robot_mode=6')
    if robot.start_tcp_pose is not None:
        raise ValueError('GELLO uses reset_q; start_tcp_pose is not supported')
    if robot.gripper_type not in (0, 1, 2, 3, 11):
        raise ValueError('This checked GELLO path supports xArm/Bio/Robotiq grippers or no gripper')
    if robot.gripper_type and leader.gripper_id < 0:
        raise ValueError('Robot gripper requires a GELLO gripper_id')
    robot.reset_q = tuple(robot.reset_q if robot.reset_q is not None else robot.start_joints)
    if len(robot.reset_q) != n or not np.isfinite(robot.reset_q).all():
        raise ValueError('reset_q must contain one finite radian value per robot joint')
    if leader.start_joints is not None and (
        len(leader.start_joints) != n or not np.allclose(leader.start_joints, robot.reset_q)
    ):
        raise ValueError('Legacy TeleoperatorConfig.start_joints must match RobotConfig.reset_q')
    for name, value in (('joint_offsets', leader.joint_offsets),
                        ('leader_reference_q', leader.leader_reference_q)):
        if value is not None and (len(value) != n or not np.isfinite(value).all()):
            raise ValueError(f'{name} must contain one finite value per mapped joint')
    if (leader.joint_offsets is None) != (leader.leader_reference_q is None):
        raise ValueError('joint_offsets and leader_reference_q must be saved together')
    if type(leader.leader_passive) is not bool:
        raise ValueError('leader_passive must be a boolean')
    if leader.leader_passive and (leader.joint_offsets is None or leader.torque_joint_ids):
        raise ValueError('leader_passive requires saved calibration and no torque_joint_ids')
    positive(leader.passive_start_tolerance_deg, 'passive_start_tolerance_deg')
    if leader.passive_start_tolerance_deg > 30:
        raise ValueError('passive_start_tolerance_deg must not exceed 30 degrees')
    for name in ('leader_reset_speed_deg', 'leader_reset_timeout', 'leader_reset_max_travel_deg'):
        positive(getattr(leader, name), name)
    if leader.leader_reset_speed_deg > 15 or leader.leader_reset_max_travel_deg > 180:
        raise ValueError('GELLO reset speed must be <=15 deg/s and travel <=180 deg')
    if leader.hold_pwm is not None and (type(leader.hold_pwm) is not int or leader.hold_pwm < 0):
        raise ValueError('hold_pwm must be a non-negative integer')
    if leader.hold_pwm_by_joint is not None and (
        len(leader.hold_pwm_by_joint) != n or any(type(x) is not int or x < 0 for x in leader.hold_pwm_by_joint)
    ):
        raise ValueError('hold_pwm_by_joint must contain non-negative integers per mapped joint')
    if leader.damping_mode not in ('none', 'current'):
        raise ValueError('damping_mode must be none or current')
    if leader.damping_enabled and leader.damping_mode == 'current':
        for name in ('damping_gain', 'damping_brake_gain', 'damping_current_limit'):
            value = getattr(leader, name)
            if value is None or len(value) != n or not np.isfinite(value).all():
                raise ValueError(f'{name} must contain {n} finite values when current damping is enabled')
        if np.any(np.asarray(leader.damping_gain) < 0) or np.any(np.asarray(leader.damping_brake_gain) < 0):
            raise ValueError('damping gains must be non-negative')
        if any(int(x) <= 0 for x in leader.damping_current_limit):
            raise ValueError('damping_current_limit must be positive')
        if leader.weak_hold_enabled:
            for name in ('weak_hold_gain', 'weak_hold_limit'):
                value = getattr(leader, name)
                if value is None or len(value) != n or not np.isfinite(value).all():
                    raise ValueError(f'{name} must contain {n} finite values when weak hold is enabled')
            if np.any(np.asarray(leader.weak_hold_gain) < 0):
                raise ValueError('weak_hold_gain must be non-negative')
            if np.any(np.asarray(leader.weak_hold_limit) <= 0):
                raise ValueError('weak_hold_limit must be positive')
        positive(leader.damping_velocity_threshold, 'damping_velocity_threshold')
        if not np.isfinite(leader.damping_velocity_filter_alpha) or not 0.0 < leader.damping_velocity_filter_alpha <= 1.0:
            raise ValueError('damping_velocity_filter_alpha must be in (0, 1]')
        positive(leader.weak_hold_release_velocity, 'weak_hold_release_velocity')
        if type(leader.damping_watchdog_ms) is not int or leader.damping_watchdog_ms <= 0 or leader.damping_watchdog_ms % 20:
            raise ValueError('damping_watchdog_ms must be a positive multiple of 20')
    for obj, names in ((robot, ('robot_speed', 'robot_acc', 'reset_speed', 'reset_acc',
                               'reset_timeout', 'reset_tolerance', 'command_timeout')),
                       (leader, ('fps', 'baudrate', 'read_timeout', 'watchdog_timeout',
                                 'alignment_tolerance', 'max_joint_step', 'max_joint_velocity'))):
        for name in names:
            positive(getattr(obj, name), name)
    if leader.watchdog_timeout <= max(1 / leader.fps, leader.read_timeout, robot.command_timeout):
        raise ValueError('watchdog_timeout must exceed the loop period and individual I/O timeouts')
    if robot.collision_sensitivity is not None and (
        type(robot.collision_sensitivity) is not int or not 1 <= robot.collision_sensitivity <= 5
    ):
        raise ValueError('collision_sensitivity must be null (preserve) or 1..5')
    if (leader.gripper_open_deg is None) != (leader.gripper_close_deg is None):
        raise ValueError('Specify both gripper_open_deg and gripper_close_deg, or neither')
    endpoints = (leader.gripper_open_deg, leader.gripper_close_deg)
    if endpoints[0] is not None and (not np.isfinite(endpoints).all() or endpoints[0] == endpoints[1]):
        raise ValueError('Gripper endpoints must be finite and different')
    if not np.isfinite(leader.gripper_travel_deg) or leader.gripper_travel_deg == 0:
        raise ValueError('gripper_travel_deg must be finite and nonzero')
    if leader.joint_limits is not None:
        limits = np.asarray(leader.joint_limits, dtype=float)
        if limits.shape != (n, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError('joint_limits must contain [lower, upper] radians per joint')
        if np.any(np.asarray(robot.reset_q) < limits[:, 0]) or np.any(np.asarray(robot.reset_q) > limits[:, 1]):
            raise ValueError('reset_q is outside configured joint_limits')
    if leader.control_mode not in ('position', 'mit_to_position'):
        raise ValueError('control_mode must be position or mit_to_position')
    if leader.control_mode == 'mit_to_position':
        if not leader.dynamics_urdf:
            raise ValueError('mit_to_position requires a matching xArm dynamics_urdf')
        names = leader.dynamics_joint_names
        if names is None or len(names) != n or len(set(names)) != n or not all(isinstance(x, str) and x for x in names):
            raise ValueError('dynamics_joint_names must list the robot joints in SDK order')
        if not all(isinstance(x, str) and x for x in leader.dynamics_locked_joint_names):
            raise ValueError('dynamics_locked_joint_names must contain URDF joint names')
        for name in ('mit_kp', 'mit_kd', 'mit_torque_limit_nm'):
            value = getattr(leader, name)
            if value is None or np.asarray(value).shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f'{name} must contain {n} finite values')
            if np.any(np.asarray(value) < (0 if name != 'mit_torque_limit_nm' else 1e-12)):
                raise ValueError(f'{name} must be non-negative (limits must be positive)')
        positive(leader.mit_acceleration_limit, 'mit_acceleration_limit')
        positive(leader.mit_tracking_error_limit, 'mit_tracking_error_limit')


def calibration_file_for(path):
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text())
    if all(data[side]['TeleoperatorConfig'].get('joint_offsets') is not None
           for side in ('left', 'right')):
        return None
    candidate = path.with_name(path.stem + '_calibrated.yaml')
    return candidate if candidate.is_file() else None


def apply_saved_calibration(data, saved):
    """Use saved mapping only when the physical arm and encoder identity still match."""
    for side in ('left', 'right'):
        for section, fields in (
            ('RobotConfig', ('robot_ip',)),
            ('TeleoperatorConfig', ('port', 'baudrate', 'joint_ids', 'gripper_id')),
        ):
            for field in fields:
                if data[side][section].get(field) != saved[side][section].get(field):
                    raise ValueError(
                        f'{side}: saved calibration {section}.{field} differs from the current config; '
                        'recalibrate before moving the robots'
                    )
        base_q = np.asarray(data[side]['RobotConfig']['reset_q'], dtype=float)
        saved_q = np.asarray(saved[side]['RobotConfig']['reset_q'], dtype=float)
        tolerance = float(data[side]['RobotConfig'].get('reset_tolerance', 0.03))
        if base_q.shape != saved_q.shape or np.max(np.abs(base_q - saved_q)) > tolerance:
            raise ValueError(f'{side}: saved reset_q differs from the current config; recalibrate')
        data[side]['RobotConfig']['reset_q'] = saved_q.tolist()
        source = saved[side]['TeleoperatorConfig']
        if source.get('joint_offsets') is None or source.get('leader_reference_q') is None:
            raise ValueError(f'{side}: saved calibration has no offsets/reference')
        for field in ('joint_signs', 'joint_offsets', 'leader_reference_q'):
            data[side]['TeleoperatorConfig'][field] = source[field]
        data[side]['TeleoperatorConfig']['leader_passive'] = source.get('leader_passive', False)
        data[side]['TeleoperatorConfig']['passive_start_tolerance_deg'] = source.get(
            'passive_start_tolerance_deg', 10.0)
        for field, default in (('leader_reset_speed_deg', 5.0), ('leader_reset_timeout', 30.0),
                               ('leader_reset_max_travel_deg', 90.0)):
            data[side]['TeleoperatorConfig'][field] = source.get(field, default)
    return data


def save_current_alignment(source_path, destination_path, configs, mappers):
    """Persist a confirmed manual alignment for automatic reuse on later starts."""
    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    if destination.exists() or source == destination:
        raise FileExistsError(f'Calibration output already exists: {destination}')
    data = yaml.safe_load(source.read_text())
    for (side, robot, leader), mapper in zip(configs, mappers):
        if mapper.offsets is None or mapper.leader_reference_q is None:
            raise ValueError(f'{side}: manual alignment is incomplete')
        section = data[side]['TeleoperatorConfig']
        section['joint_signs'] = list(leader.joint_signs)
        section['joint_offsets'] = mapper.offsets.tolist()
        section['leader_reference_q'] = mapper.leader_reference_q.tolist()
        section['start_joints'] = list(robot.reset_q)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    temporary.replace(destination)
    return destination


def load_configs(path, dual=False, require_calibrated=None, calibration_path=None):
    config_path = Path(path).expanduser().resolve()
    with open(config_path) as stream:
        data = yaml.safe_load(stream)
    if calibration_path is not None:
        with open(Path(calibration_path).expanduser()) as stream:
            data = apply_saved_calibration(data, yaml.safe_load(stream))
    entries = [(name, data[name]) for name in ('left', 'right')] if dual else [('arm', data)]
    if require_calibrated is None:
        require_calibrated = False
    result = []
    for name, entry in entries:
        robot = GelloRobotConfig(**entry['RobotConfig'])
        leader = GelloTeleopConfig(**entry['TeleoperatorConfig'])
        if leader.dynamics_urdf is not None:
            urdf = Path(leader.dynamics_urdf).expanduser()
            if not urdf.is_absolute():
                urdf = config_path.parent / urdf
            leader.dynamics_urdf = str(urdf.resolve())
        validate(robot, leader)
        if dual and 'reset_q' not in entry['RobotConfig']:
            raise ValueError(f'{name}: explicit RobotConfig.reset_q is required for dual GELLO')
        if require_calibrated and not leader.joint_offsets:
            raise ValueError(
                f'{name}: calibrated joint_offsets/leader_reference_q are required; '
                'run calibrate_joint_directions.py first')
        result.append((name, robot, leader))
    if dual:
        if result[0][2].leader_passive != result[1][2].leader_passive:
            raise ValueError('Both GELLO sides must use the same leader_passive mode')
        if any(mapper[2].joint_offsets is not None for mapper in result) and not all(
            mapper[2].joint_offsets is not None for mapper in result
        ):
            raise ValueError('Both GELLO sides must use saved calibration together')
        if result[0][1].robot_ip == result[1][1].robot_ip:
            raise ValueError('Left and right robot_ip must differ')
        if os.path.realpath(result[0][2].port) == os.path.realpath(result[1][2].port):
            raise ValueError('Left and right GELLO ports resolve to the same device')
    return result


class JointMapper:
    def __init__(self, robot, config):
        self.robot, self.config = robot, config
        self.n = len(config.joint_ids)
        self.offsets = None

    def _sample(self, raw):
        raw = np.asarray(raw, dtype=float)
        expected = self.n + int(self.config.gripper_id >= 0) + len(self.config.torque_joint_ids or ())
        if raw.shape != (expected,) or not np.isfinite(raw).all():
            raise ValueError('Invalid GELLO joint sample')
        return raw

    def align(self, raw):
        raw = self._sample(raw)
        if self.config.joint_offsets is not None:
            self.offsets = np.asarray(self.config.joint_offsets, dtype=float)
            if self.config.leader_passive:
                # Torque-off feedback can differ by full turns after power cycling.
                # Select the saved reference's equivalent encoder branch ONCE;
                # retain continuous feedback and jump detection during following.
                turns = np.rint((raw[:self.n] - self.config.leader_reference_q) / (2 * np.pi))
                self.offsets = self.offsets + turns * (2 * np.pi)
                q = (raw[:self.n] - self.offsets) * self.config.joint_signs
                if np.max(np.abs(q - self.robot.reset_q)) > np.deg2rad(self.config.passive_start_tolerance_deg):
                    error = np.rad2deg(q - self.robot.reset_q)
                    raise ValueError(f'主手未回到保存的参考姿态，各轴误差(度)={np.round(error, 2)}；'
                                     f'允许各轴偏差 {self.config.passive_start_tolerance_deg:g}°；'
                                     '请大致摆回参考姿态后重新启动；不会重算零点')
        else:
            self.offsets = raw[:self.n] - np.asarray(self.robot.reset_q) / self.config.joint_signs
        self.leader_reference_q = raw[:self.n].copy()
        self.last_raw = np.asarray(self.robot.reset_q, dtype=float)
        self.last_target = self.last_raw.copy()
        if self.config.leader_passive:
            self.last_raw = (raw[:self.n] - self.offsets) * self.config.joint_signs
        if self.config.gripper_id >= 0:
            self.gripper_open = raw[self.n] if self.config.gripper_open_deg is None else np.deg2rad(self.config.gripper_open_deg)
            self.gripper_close = (self.gripper_open + np.deg2rad(self.config.gripper_travel_deg)
                                  if self.config.gripper_close_deg is None else np.deg2rad(self.config.gripper_close_deg))

    @property
    def has_saved_calibration(self):
        return self.config.joint_offsets is not None and self.config.leader_reference_q is not None

    def target(self, raw):
        if self.offsets is None:
            raise RuntimeError('GELLO must be aligned after robot reset')
        raw = self._sample(raw)
        q = (raw[:self.n] - self.offsets) * self.config.joint_signs
        if self.config.joint_limits is not None:
            bounds = np.asarray(self.config.joint_limits)
            if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
                raise ValueError('GELLO target exceeds configured joint_limits')
        return q, raw

    def action(self, raw, dt):
        q, raw = self.target(raw)
        if np.max(np.abs(q - self.last_raw)) > self.config.max_joint_step:
            raise ValueError('GELLO target jumped beyond max_joint_step; realign before restarting')
        self.last_raw = q.copy()
        # A late cycle must not permit a proportionally large catch-up step.
        step = self.config.max_joint_velocity * min(dt, 1 / self.config.fps)
        self.last_target += np.clip(q - self.last_target, -step, step)
        action = self.last_target.copy()
        if self.config.gripper_id >= 0:
            grip = np.clip((raw[self.n] - self.gripper_open) / (self.gripper_close - self.gripper_open), 0, 1)
            action = np.append(action, grip)
        return action


class Session:
    def __init__(self, configs, arm_factory=Arm, reader_factory=GelloReader):
        self.configs = configs
        self.arm_factory, self.reader_factory = arm_factory, reader_factory
        self.arms, self.readers, self.mappers = [], [], []
        self.controllers = []
        self.stop_event = threading.Event()
        self.motion_started = False
        self.stage = 'disconnected'
        self.aligned_sides = []

    def connect(self):
        # Load and validate every dynamics model before opening either robot.
        for _, robot, leader in self.configs:
            if leader.control_mode == 'mit_to_position':
                model = PinocchioArmModel(leader.dynamics_urdf, leader.dynamics_joint_names,
                                          leader.dynamics_locked_joint_names)
                controller = MitPositionController(
                    model, leader.mit_kp, leader.mit_kd, leader.mit_torque_limit_nm,
                    leader.max_joint_velocity, leader.mit_acceleration_limit,
                    leader.mit_tracking_error_limit,
                )
                _, _, gravity = model.terms(np.asarray(robot.reset_q), np.zeros(model.dof))
                if np.any(np.abs(gravity) > controller.torque_limit):
                    raise ValueError('MIT torque limits cannot hold gravity at reset_q')
                self.controllers.append(controller)
            else:
                self.controllers.append(None)
        for name, robot, leader in self.configs:
            arm = self.arm_factory(robot)
            self.arms.append(arm)
            if arm.api.axis != len(leader.joint_ids):
                raise ValueError(f'{name}: robot axis count does not match joint_ids/reset_q')
            arm.health()
            arm.check_target(np.asarray(robot.reset_q))
            self.readers.append(self.reader_factory(leader))
            self.mappers.append(JointMapper(robot, leader))
        self.stage = 'connected'

    def _parallel(self, jobs, timeout):
        if self.stop_event.is_set():
            raise RuntimeError('Session cancelled')
        errors = queue.Queue()
        def worker(job):
            try:
                job()
            except BaseException as error:
                errors.put(error)
                self.stop_event.set()
        threads = [threading.Thread(target=worker, args=(job,), daemon=True) for job in jobs]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + timeout
        while any(t.is_alive() for t in threads):
            if self.stop_event.wait(0.02):
                if not errors.empty():
                    raise errors.get()
                raise RuntimeError('Session cancelled')
            if time.monotonic() > deadline:
                self.stop_event.set()
                raise TimeoutError('Parallel operation timed out')
        if not errors.empty():
            raise errors.get()
        if self.stop_event.is_set():
            raise RuntimeError('Session cancelled')

    def reset(self):
        if self.stage != 'connected':
            raise RuntimeError('Connect both sides before reset')
        self.motion_started = True
        barrier = threading.Barrier(len(self.arms))
        def reset_one(arm):
            try:
                arm.reset(self.stop_event, barrier)
            except BaseException:
                barrier.abort()
                raise
        self._parallel([lambda arm=arm: reset_one(arm) for arm in self.arms],
                       2 * max(c.reset_timeout for _, c, _ in self.configs))
        self.stage = 'reset'
        def reset_leader(side, reader, display):
            reference = getattr(reader.config, 'leader_reference_q', None)
            try:
                if reader.config.leader_passive:
                    reader.set_torque(False)
                elif reference is not None:
                    reader.return_to_reference(reference, stop=self.stop_event,
                                               progress=lambda sample: display.update(side, sample))
                else:
                    reader.prepare_alignment()
                display.finish(side, 'done' if reference is not None and not reader.config.leader_passive else 'manual')
            except BaseException:
                display.finish(side, 'failed')
                raise
        # Each leader owns a separate serial port; neither must wait while the
        # other returns and potentially drifts during the operator's wait.
        with ResetDisplay([side for side, _, _ in self.configs]) as display:
            self._parallel([lambda side=side, reader=reader: reset_leader(side, reader, display)
                            for (side, _, _), reader in zip(self.configs, self.readers)],
                           max(c.leader_reset_timeout for _, _, c in self.configs) + 5.0)
        if all(mapper.has_saved_calibration and not mapper.config.leader_passive
               for mapper in self.mappers):
            for mapper, reader in zip(self.mappers, self.readers):
                mapper.align(reader.read())
            self.aligned_sides = [name for name, _, _ in self.configs]
            self.stage = 'aligned'

    def align(self, side):
        if self.stage not in ('reset', 'aligning'):
            raise RuntimeError('Both robot resets must finish before GELLO alignment')
        if self.stop_event.is_set():
            raise RuntimeError('Session cancelled')
        config = self.readers[len(self.aligned_sides)].config
        if config.leader_reference_q is not None and not config.leader_passive:
            raise RuntimeError('Saved calibration is active; manual alignment is not required')
        index = len(self.aligned_sides)
        expected = self.configs[index][0]
        if side != expected:
            raise ValueError(f'Align {expected} before {side}')
        # Check both robots, but only sample the leader explicitly confirmed this time.
        for arm, (_, robot, _) in zip(self.arms, self.configs):
            arm.health(active=True)
            if np.max(np.abs(arm.joints() - robot.reset_q)) > robot.reset_tolerance:
                raise ValueError('Robot moved away from reset_q; reset again before alignment')
        reader = self.readers[index]
        if reader.config.leader_passive:
            self.mappers[index].align(reader.read())
            self.aligned_sides.append(side)
            self.stage = 'aligned' if len(self.aligned_sides) == len(self.configs) else 'aligning'
            return
        before_hold = reader.read()
        # Position-mode feedback can change by a whole turn when torque is
        # enabled.  Save the reference only after the motor enters hold.
        reader.hold()
        after_hold = reader.read()
        n = len(reader.config.joint_ids)
        physical_change = (after_hold[:n] - before_hold[:n] + np.pi) % (2 * np.pi) - np.pi
        if np.max(np.abs(physical_change)) > reader.config.alignment_tolerance:
            raise ValueError(f'{side}: GELLO moved while enabling position hold; realign before saving')
        self.mappers[index].align(after_hold)
        self.aligned_sides.append(side)
        self.stage = 'aligned' if len(self.aligned_sides) == len(self.configs) else 'aligning'

    def print_hold_status(self):
        """Report actual Dynamixel state after both leaders reach their held poses."""
        if self.stage != 'aligned':
            raise RuntimeError('Both GELLO leaders must be aligned before reading hold status')
        for (name, _, _), reader in zip(self.configs, self.readers):
            print(f'{name}: GELLO 保持状态', flush=True)
            for status in reader.read_hold_status():
                motor_id = status['id']
                if 'error' in status:
                    print(f"  ID {motor_id}: {status['error']}", flush=True)
                    continue
                print(
                    f"  ID {motor_id}: mode={status['mode']} torque={status['torque_on']} "
                    f"PWM configured={status['configured']} EEPROM limit={status['pwm_limit']} "
                    f"goal={status['goal_pwm']} present={status['present_pwm']} "
                    f"position_error_ticks={status['position_error_ticks']} "
                    f"hardware_error=0x{status['hardware_error']:02x}",
                    flush=True,
                )

    def start_follow(self):
        if self.stage != 'aligned':
            raise RuntimeError('Align both GELLO leaders before following')
        # This confirmation may have taken time: reject movement since calibration.
        for (side, _, _), arm, reader, mapper in zip(self.configs, self.arms, self.readers, self.mappers):
            arm.health(active=True)
            q, _ = mapper.target(reader.read())
            actual = arm.joints()
            tolerance = (np.deg2rad(mapper.config.passive_start_tolerance_deg)
                         if mapper.config.leader_passive else mapper.config.alignment_tolerance)
            if np.max(np.abs(q - actual)) > tolerance:
                i = int(np.argmax(np.abs(q - actual)))
                raise ValueError(
                    f'{side}: GELLO/robot alignment changed：J{i+1}/ID {mapper.config.joint_ids[i]}，'
                    f'主手映射目标={np.rad2deg(q[i]):+.2f}°，xArm实际={np.rad2deg(actual[i]):+.2f}°，'
                    f'偏差={np.rad2deg(q[i]-actual[i]):+.2f}°，允许={np.rad2deg(tolerance):.2f}°；'
                    f'各轴偏差(度)={np.round(np.rad2deg(q-actual), 2)}。'
                    '回位完成后到开始遥操之间姿态不一致；未重新计算零点，请检查保持/外力后重试'
                )
            if mapper.config.leader_passive:
                arm.check_target(q)
                mapper.last_raw = q.copy()
                mapper.last_target = actual.copy()
                print(f'xArm {arm.config.robot_ip}: 启动追赶误差(度)='
                      f'{np.round(np.rad2deg(q - actual), 2)}；按 max_joint_velocity 限速跟随。', flush=True)
        self._parallel([arm.prepare_follow for arm in self.arms],
                       max(c.reset_timeout for _, c, _ in self.configs))
        # Release alignment/reset holding before accepting operator motion.
        # Explicitly configured unmapped fixed joints remain energized.
        for reader in self.readers:
            # A configured current-damping leader must stay energized after
            # takeover.  The dedicated dual dataset pipeline updates its
            # current command from the independent GELLO reader thread.  The
            # legacy zero-force path keeps its historical torque-off behavior.
            if (bool(getattr(reader.config, 'damping_enabled', False)) and
                    str(getattr(reader.config, 'damping_mode', 'none')).lower() == 'current'):
                reader.enable_current_damping(reader.config)
                continue
            released_ids = list(reader.config.joint_ids)
            if reader.config.gripper_id >= 0:
                released_ids.append(reader.config.gripper_id)
            reader.set_torque(False, ids=released_ids)
        self.stage = 'following'

    def follow(self):
        if self.stage != 'following':
            raise RuntimeError('Following has not been confirmed')
        errors = queue.Queue()
        progress = [time.monotonic()] * len(self.arms)
        def worker(i):
            arm, reader, mapper = self.arms[i], self.readers[i], self.mappers[i]
            controller = self.controllers[i]
            damping_previous_q = None
            damping_velocity = None
            damping_reference_q = None
            period = 1 / mapper.config.fps
            previous = time.monotonic() - period
            deadline = time.monotonic()
            first = True
            last_warning = 0
            try:
                while not self.stop_event.is_set():
                    arm.health(active=True)
                    raw = reader.read()
                    q, _ = mapper.target(raw)
                    now = time.monotonic()
                    if (bool(getattr(reader.config, 'damping_enabled', False)) and
                            str(getattr(reader.config, 'damping_mode', 'none')).lower() == 'current'):
                        if damping_reference_q is None:
                            damping_reference_q = q.copy()
                        damping_current = reader.compute_damping_current(
                            q, damping_previous_q, max(now - previous, 1e-4), reader.config,
                            damping_reference_q, damping_velocity,
                        )
                        reader.write_current_damping(damping_current)
                        raw_dq = np.zeros_like(q) if damping_previous_q is None else (q - damping_previous_q) / max(now - previous, 1e-4)
                        alpha = float(getattr(reader.config, 'damping_velocity_filter_alpha', 1.0))
                        damping_velocity = raw_dq if damping_velocity is None else alpha * raw_dq + (1.0 - alpha) * damping_velocity
                        damping_previous_q = q.copy()
                    arm.check_target(q)  # check the raw target, before slew limiting
                    measured_q = arm.joints() if first or controller is not None else None
                    if first:
                        tolerance = (np.deg2rad(mapper.config.passive_start_tolerance_deg)
                                     if mapper.config.leader_passive else mapper.config.alignment_tolerance)
                        difference = q - measured_q
                        if np.max(np.abs(difference)) > tolerance:
                            joint = int(np.argmax(np.abs(difference)))
                            raise ValueError(
                                f'{self.configs[i][0]}: GELLO moved while preparing robot control：'
                                f'J{joint+1}/ID {mapper.config.joint_ids[joint]}，'
                                f'主手映射目标={np.rad2deg(q[joint]):+.2f}°，'
                                f'xArm实际={np.rad2deg(measured_q[joint]):+.2f}°，'
                                f'偏差={np.rad2deg(difference[joint]):+.2f}°，'
                                f'允许={np.rad2deg(tolerance):.2f}°；'
                                '夹爪初始化及释放主手力矩后的首帧检查未通过，请托稳主手；不会重算零点'
                            )
                    action = mapper.action(raw, now - previous)
                    previous = now
                    if controller is not None:
                        if first:
                            controller.reset(measured_q, now)
                            action[:mapper.n] = measured_q
                        else:
                            action[:mapper.n] = controller.step(
                                action[:mapper.n], measured_q, now, period
                            )
                        arm.check_target(action[:mapper.n])
                    first = False
                    if self.stop_event.is_set():
                        return
                    arm.send(action)
                    progress[i] = time.monotonic()
                    deadline += period
                    if progress[i] - deadline > period:
                        if progress[i] - last_warning >= 5:
                            logger.warning('%s: loop overrun %.3fs', self.configs[i][0], progress[i] - deadline)
                            last_warning = progress[i]
                        deadline = progress[i]
                    self.stop_event.wait(max(0, deadline - time.monotonic()))
            except BaseException as error:
                errors.put(error)
                self.stop_event.set()
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(self.arms))]
        for thread in threads:
            thread.start()
        while not self.stop_event.wait(0.02):
            for i, (_, _, leader) in enumerate(self.configs):
                if time.monotonic() - progress[i] > leader.watchdog_timeout:
                    self.stop_event.set()
                    raise TimeoutError(f'{self.configs[i][0]} control loop stopped responding')
        if not errors.empty():
            raise errors.get()

    def shutdown(self):
        self.stop_event.set()
        # Stop requests run independently so one unresponsive controller cannot delay the other.
        def cleanup(arm):
            try:
                if self.motion_started:
                    arm.stop()
            except Exception:
                logger.exception('Robot stop failed; use the hardware emergency stop')
            finally:
                try:
                    arm.close()
                except Exception:
                    logger.exception('Robot disconnect failed')
        threads = [threading.Thread(target=cleanup, args=(arm,), daemon=True) for arm in self.arms]
        for thread in threads:
            thread.start()
        for reader in self.readers:
            # Closing may wait for an in-flight serial transaction, so keep it off the main thread.
            thread = threading.Thread(target=reader.close, daemon=True)
            thread.start()
            threads.append(thread)
        deadline = time.monotonic() + 3
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            logger.error('Cleanup timed out; confirm both robots stopped using hardware emergency stops')
        self.stage = 'stopped'


def confirm(prompt, stop):
    print(prompt, flush=True)
    while not stop.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            if sys.stdin.readline() == '':
                raise EOFError('Interactive confirmation requires an open terminal')
            return
    raise KeyboardInterrupt


def main(dual=False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--check-config', action='store_true',
                        help='Show saved-calibration status without connecting to hardware')
    parser.add_argument('--hold-diagnostics', action='store_true',
                        help='Read and print each GELLO motor status after alignment, before following')
    args = parser.parse_args()
    saved_path = calibration_file_for(args.config) if dual else None
    if saved_path is not None:
        print(f'自动加载已保存的 GELLO 标定：{saved_path}', flush=True)
    configs = load_configs(args.config, dual=dual, calibration_path=saved_path)
    if args.check_config:
        for name, robot, leader in configs:
            state = '已保存标定；启动时复用' if leader.joint_offsets is not None else '尚未保存标定；启动时需人工对齐'
            print(f'{name}: {state}; xArm={robot.robot_ip}; GELLO={leader.port}')
        return 0
    session = Session(configs)
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda *_: session.stop_event.set())
    try:
        session.connect()
        confirm('连接检查完成。确认复位路径畅通，按 Enter 让机械臂同时低速回到各自 reset_q。', session.stop_event)
        session.reset()
        if session.stage == 'aligned':
            print('已加载保存的 GELLO 标定结果，两个主手已自动复位到记录位置。', flush=True)
        else:
            print('机械臂复位完成，开始逐侧对齐。被动零点模式仅核对保存的参考姿态。', flush=True)
            for side, robot, leader in session.configs:
                label = {'left': '左侧', 'right': '右侧'}.get(side, '当前')
                reminder = '已标定的主手请保持不动。' if session.aligned_sides else ''
                if leader.leader_passive:
                    print(f'{label} GELLO 保存参考角(度，按360°等价): '
                          f'{np.round(np.rad2deg(leader.leader_reference_q) % 360, 1)}', flush=True)
                    print(f'无需精确复现，各轴允许偏差 {leader.passive_start_tolerance_deg:g}°；'
                          '开始遥操后从臂会限速追赶主臂对应姿态。', flush=True)
                operation = '核对参考姿态' if leader.leader_passive else '标定'
                confirm(f'{reminder}将{label} GELLO 摆到对应 xArm（{robot.robot_ip}）的姿态，'
                        f'并打开该主手夹爪；托住主手，按 Enter {operation}。', session.stop_event)
                session.align(side)
                print(f'{label}标定完成。', flush=True)
        if args.hold_diagnostics:
            time.sleep(0.3)  # Let the position controller settle before sampling its output.
            session.print_hold_status()
        confirm('全部主手标定完成。托住主手并保持不动，按 Enter 初始化机器人夹爪、释放主手力矩并开始遥操。Ctrl+C 停止两侧。', session.stop_event)
        session.start_follow()
        if dual and saved_path is None and not all(mapper.has_saved_calibration for mapper in session.mappers):
            destination = Path(args.config).expanduser().resolve().with_name(
                Path(args.config).stem + '_calibrated.yaml'
            )
            path = save_current_alignment(args.config, destination, session.configs, session.mappers)
            print(f'已保存 GELLO 标定：{path}；下次使用同一个 --config 会自动加载。', flush=True)
        session.follow()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        logger.exception('GELLO session stopped')
        return 1
    finally:
        session.shutdown()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
