"""Single-side manual GELLO origin and encoder-direction calibration."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import replace
import fcntl
import logging
import math
from pathlib import Path
import signal
import sys
import threading
import time
import os

import numpy as np
import yaml

from gello_teleop.calibrate_joint_directions import _move_arm, calibration_values
from gello_teleop.calibrate_zero import stable_reference
from gello_teleop.gello_hardware import Arm, GelloReader
from gello_teleop.gello_damping import damping_config
from gello_teleop.uf_robot_gello_teleop import load_configs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT/'gello_teleop/config/xarm7_gello_dual_dataset.yaml'


class CalibrationDamping:
    """Pure encoder-coordinate damping, independent of unverified mappings."""
    def __init__(self, reader, config, on_failure):
        self.reader = reader
        self.config = replace(config, joint_signs=tuple([1]*len(config.joint_ids)),
                              weak_hold_enabled=False)
        self.on_failure = on_failure
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.capabilities = None

    def start(self, reference):
        try:
            self.capabilities = self.reader.enable_current_damping(self.config)
            raw = np.asarray(self.reader.read(), float)[:len(self.config.joint_ids)]
            if not np.isfinite(raw).all() or np.max(np.abs(raw-reference)) > np.deg2rad(1):
                raise ValueError('GELLO 电流模式切换后编码器原点变化，请重新标定')
            self.thread = threading.Thread(target=self._run, name='calibration-damping', daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def _run(self):
        previous = velocity = None
        previous_t = None
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                raw = np.asarray(self.reader.read(), float)[:len(self.config.joint_ids)]
                if not np.isfinite(raw).all():
                    raise ValueError('GELLO 阻尼采样含无效角度')
                now = time.monotonic()
                dt = max(now-(previous_t or now), 1e-4)
                current = self.reader.compute_damping_current(raw, previous, dt, self.config,
                                                              previous_velocity=velocity)
                dq = np.zeros_like(raw) if previous is None else (raw-previous)/dt
                alpha = self.config.damping_velocity_filter_alpha
                velocity = dq if velocity is None else alpha*dq+(1-alpha)*velocity
                self.reader.write_current_damping(current)
                previous, previous_t = raw.copy(), now
            except Exception as exc:
                self.error = exc
                try:
                    self.reader.disable_current_damping()
                finally:
                    if not self.stop.is_set():
                        self.on_failure()
                return
            self.stop.wait(max(0., 1/self.config.fps-(time.monotonic()-started)))

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
        self.reader.disable_current_damping()


def paths_for(config_path, output=None):
    config_path = Path(config_path).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text())
    if 'gello_config' in data:
        template = config_path.parent/data.get('gello_template', 'xarm7_gello_teleop_dual.yaml')
        destination = config_path.parent/data['gello_config']
    else:
        template = config_path
        destination = config_path.with_name('xarm7_gello_calibration.yaml')
    if output:
        destination = Path(output).expanduser().resolve()
    if template.resolve() == destination.resolve():
        raise ValueError('标定结果与配置模板必须是不同文件')
    return template.resolve(), destination.resolve()


def save_side(template, destination, side, entry):
    """Merge only this side and atomically OVERWRITE; preserve the other side."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(destination.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = destination if destination.exists() else Path(template)
        data = yaml.safe_load(source.read_text())
        data[side] = deepcopy(entry)
        data['calibration_format'] = 'manual_origin_selected_axes_v1'
        temporary = destination.with_suffix(destination.suffix+'.tmp')
        temporary.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        temporary.replace(destination)
    print(f'已覆盖保存 {side}：{destination}', flush=True)


def begin_side(template, destination, side):
    base = yaml.safe_load(Path(template).read_text())
    entry = deepcopy(base[side])
    if Path(destination).exists():
        previous = yaml.safe_load(Path(destination).read_text()).get(side)
        if previous:
            # Keep candidate signs only as an explicit unverified starting point.
            entry['TeleoperatorConfig']['joint_signs'] = previous['TeleoperatorConfig']['joint_signs']
    teleop = entry['TeleoperatorConfig']
    n = len(teleop['joint_ids'])
    teleop.update(joint_offsets=None, leader_reference_q=None, leader_passive=False,
                  damping_enabled=False, gripper_open_deg=None, gripper_close_deg=None)
    entry['CalibrationStatus'] = dict(reference_ready=False, direction_verified=[False]*n,
                                    checks={}, started_at=datetime.now(timezone.utc).isoformat())
    return entry


def record_origin(entry, robot_q, raw):
    teleop = entry['TeleoperatorConfig']
    n = len(teleop['joint_ids'])
    values = calibration_values(robot_q, raw[:n], teleop['joint_signs'])
    entry['RobotConfig']['reset_q'] = values['reset_q']
    for key in ('joint_signs', 'joint_offsets', 'leader_reference_q'):
        teleop[key] = values[key]
    teleop['start_joints'] = values['reset_q']
    teleop['leader_passive'] = True
    if teleop.get('gripper_id', -1) >= 0:
        teleop['gripper_open_deg'] = float(np.rad2deg(raw[n]))
        teleop['gripper_close_deg'] = teleop['gripper_open_deg']+teleop.get('gripper_travel_deg', -42)
    entry['CalibrationStatus'].update(reference_ready=True, direction_verified=[False]*n, checks={})


def infer_direction(before, after, robot_delta, joint, step_deg=15.):
    before, after = np.asarray(before, float), np.asarray(after, float)
    if before.shape != after.shape or before.ndim != 1 or not 0 <= joint < len(before) or not np.isfinite(np.r_[before, after, robot_delta]).all():
        raise ValueError('无效的关节采样')
    delta = after-before
    if abs(np.rad2deg(robot_delta)-step_deg) > .5:
        raise ValueError('xArm 实测运动与 15° 指令不符')
    if not step_deg-5 <= abs(np.rad2deg(delta[joint])) <= step_deg+5:
        raise ValueError(f'GELLO J{joint+1} 实测 {np.rad2deg(delta[joint]):+.2f}°，需要约 {step_deg:g}°（±5°）')
    other_joint_tolerance_deg = 5.0
    offenders = [f'J{i+1}={np.rad2deg(value):+.2f}°'
                 for i, value in enumerate(delta)
                 if i != joint and abs(value) > np.deg2rad(other_joint_tolerance_deg)]
    if offenders:
        raise ValueError(f'其他 GELLO 关节移动超过 {other_joint_tolerance_deg:g}°：'
                         + ', '.join(offenders) + '；支撑其他关节后重新测试')
    # Operator supplies the SAME physical direction. The encoder delta then
    # reveals the model-to-encoder sign; it cannot certify their observation.
    return int(np.sign(delta[joint]*robot_delta)), delta


def update_direction(entry, joint, sign, before, after, robot_delta):
    teleop = entry['TeleoperatorConfig']
    old_sign = teleop['joint_signs'][joint]
    teleop['joint_signs'][joint] = int(sign)
    q = np.asarray(entry['RobotConfig']['reset_q'])
    raw_ref = np.asarray(teleop['leader_reference_q'])
    # Always recompute offset when a sign changes, keeping the captured origin.
    teleop['joint_offsets'] = (raw_ref-q/np.asarray(teleop['joint_signs'])).tolist()
    status = entry['CalibrationStatus']
    status['direction_verified'][joint] = True
    status['checks'][str(joint+1)] = dict(old_sign=int(old_sign), new_sign=int(sign),
                                      before_raw_rad=np.asarray(before).tolist(),
                                      after_raw_rad=np.asarray(after).tolist(),
                                      raw_delta_deg=np.rad2deg(np.asarray(after)-before).tolist(),
                                      xarm_delta_deg=float(np.rad2deg(robot_delta)),
                                      method='Operator manually moved GELLO in the observed positive xArm direction')


def prompt(message):
    value = input(message).strip().lower()
    if value == 'q':
        raise KeyboardInterrupt
    return value


def calibrate_axes(arm, reader, entry, persist):
    n = len(entry['TeleoperatorConfig']['joint_ids'])
    q_ref = np.asarray(entry['RobotConfig']['reset_q'])
    raw_ref = np.asarray(entry['TeleoperatorConfig']['leader_reference_q'])
    while True:
        verified = entry['CalibrationStatus']['direction_verified']
        print('已完成轴：'+(', '.join(str(i+1) for i, done in enumerate(verified) if done) or '无'), flush=True)
        selection = prompt(f'输入要标定的轴号 1～{n}；q 保存当前结果并退出：')
        if not selection.isdigit() or not 1 <= int(selection) <= n:
            print('请输入有效轴号。', flush=True)
            continue
        joint = int(selection)-1
        prompt('将 GELLO 所有关节手动返回刚才的原点，支撑静止；按 Enter 读取起点，q 退出：')
        before_q, before_raw = stable_reference(arm, reader)
        origin_tolerance_deg = 5.0
        origin_delta = np.rad2deg(before_raw[:n]-raw_ref)
        offenders = [f'J{i+1}={value:+.2f}°' for i, value in enumerate(origin_delta)
                     if abs(value) > origin_tolerance_deg]
        if offenders:
            print(f'主手未回到记录原点（允许各轴 {origin_tolerance_deg:g}°）：'
                  + ', '.join(offenders) + '；请调整后重新选择轴。', flush=True)
            continue
        if np.max(np.abs(before_q-q_ref)) > np.deg2rad(.5):
            raise ValueError('xArm 已离开记录原点，请重新运行本侧标定')
        target = before_q.copy()
        target[joint] += np.deg2rad(15)
        _move_arm(arm, target)
        prompt(f'xArm J{joint+1} 已正向转动 15°。手动将 GELLO 同一物理关节按相同方向转动约 15°，'
               '其他关节保持不动，静止后按 Enter 采样；q 停止：')
        try:
            after_q, after_raw = stable_reference(arm, reader)
            if np.max(np.abs(after_q-target)) > np.deg2rad(.5):
                raise ValueError('采样时 xArm 不在试动目标位置')
            sign, delta = infer_direction(before_raw[:n], after_raw[:n], after_q[joint]-before_q[joint], joint)
        except ValueError as exc:
            _move_arm(arm, before_q)
            print(f'本轴未更新：{exc}；GELLO 请手动返回原点。', flush=True)
            continue
        update_direction(entry, joint, sign, before_raw[:n], after_raw[:n], after_q[joint]-before_q[joint])
        persist()
        print(f'J{joint+1}: GELLO 编码器变化 {np.rad2deg(delta[joint]):+.2f}°，'
              f'映射方向={sign:+d}，偏置已同步更新。', flush=True)
        _move_arm(arm, before_q)
        print('xArm 已返回原点；请将 GELLO 手动返回原点，再选择下一轴。', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='单侧手动原点＋选轴方向标定，结果实时覆盖保存')
    parser.add_argument('-c', '--config', default=str(DEFAULT_CONFIG), help='数采配置或双臂模板')
    sides = parser.add_mutually_exclusive_group()
    sides.add_argument('--left', action='store_true')
    sides.add_argument('--right', action='store_true')
    parser.add_argument('--output', help='默认使用数采配置 gello_config 指定的结果文件')
    parser.add_argument('--show-mapping', action='store_true')
    parser.add_argument('--no-damping', action='store_true', help='本次标定关闭主手阻尼，不改变采集配置')
    args = parser.parse_args(argv)
    template, output = paths_for(args.config, args.output)
    side = 'left' if args.left else 'right' if args.right else None
    if args.show_mapping:
        for name, robot, leader in load_configs(output, dual=True, side=side, require_calibrated=True):
            print(name, 'signs=', leader.joint_signs, 'offsets=', leader.joint_offsets)
        return 0
    if side is None:
        parser.error('必须选择 --left 或 --right')
    entry = begin_side(template, output, side)
    saved_this_run = False
    def persist():
        nonlocal saved_this_run
        save_side(template, output, side, entry)
        saved_this_run = True
    # Build hardware configuration from the template, without touching the
    # destination file. New origins/directions stay in memory until an axis passes.
    _, robot, leader = load_configs(template, dual=True, side=side, require_calibrated=False)[0]
    leader = replace(leader, joint_signs=tuple(entry['TeleoperatorConfig']['joint_signs']),
                     joint_offsets=None, leader_reference_q=None, leader_passive=False,
                     damping_enabled=False, gripper_open_deg=None, gripper_close_deg=None)
    workflow = yaml.safe_load(Path(args.config).read_text())
    template_settings = yaml.safe_load(template.read_text())[side]['TeleoperatorConfig']
    runtime_leader = damping_config(robot, replace(leader, damping_enabled=template_settings.get(
        'damping_enabled', False)), side, workflow.get('gello_damping'))
    if leader.torque_joint_ids:
        parser.error('手动方向标定要求 torque_joint_ids 为空')
    arm = reader = None
    damping = None
    motion_started = False
    old_handlers = {}
    def interrupt(*_):
        raise KeyboardInterrupt
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupt)
        arm = Arm(robot)
        arm.health()
        if arm.api.axis != len(leader.joint_ids):
            raise ValueError('xArm 轴数与 GELLO 映射不一致')
        reader = GelloReader(leader)
        print(f'仅连接 {side}：xArm {robot.robot_ip} / GELLO {leader.port}', flush=True)
        prompt('先用控制界面把 xArm 摆到参考姿态并停止。托住 GELLO，按 Enter 关闭主手力矩：')
        reader.prepare_alignment()
        prompt('先标定 GELLO 原点：手动摆到与 xArm 对应的初始姿态，并打开主手夹爪，静止后按 Enter：')
        robot_q, raw = stable_reference(arm, reader)
        arm.check_target(robot_q)
        record_origin(entry, robot_q, raw)
        if runtime_leader.damping_enabled and not args.no_damping:
            damping = CalibrationDamping(reader, runtime_leader,
                                         lambda: os.kill(os.getpid(), signal.SIGINT))
            damping.start(raw[:len(leader.joint_ids)])
            entry['CalibrationStatus']['damping'] = dict(
                enabled=True, coordinates='encoder', weak_hold_enabled=False,
                gain=runtime_leader.damping_gain, current_limit=runtime_leader.damping_current_limit,
                capabilities=damping.capabilities)
            print('主手纯阻尼已启用（不锁位置、不抗静态重力）；方向测试只驱动 xArm。', flush=True)
        else:
            print('GELLO 保持力矩关闭，方向测试只驱动 xArm。', flush=True)
        robot.reset_q = tuple(robot_q)
        motion_started = True
        arm.prepare_reset()
        if np.max(np.abs(arm.joints()-robot_q)) > np.deg2rad(.5):
            raise ValueError('xArm 使能时参考姿态发生变化，请重新标定')
        print('原点已记录，暂未写文件；每轴成功后覆盖保存，可以选择轴号开始方向测试。', flush=True)
        calibrate_axes(arm, reader, entry, persist)
    except KeyboardInterrupt:
        if damping is not None and damping.error is not None:
            print(f'主手阻尼故障，标定已停止：{damping.error}', flush=True)
            return 1
        print('已退出；已通过的轴及本次原点已保存，未完成轴保持未确认状态。'
              if saved_this_run else '已退出；本次尚无成功轴，原有标定文件未修改。', flush=True)
        return 0
    except Exception:
        logging.exception('标定停止；已保存结果不会丢失，未通过的数据不会写入')
        return 1
    finally:
        try:
            if damping is not None:
                damping.close()
        finally:
            try:
                if reader is not None:
                    reader.close()
            finally:
                try:
                    if arm is not None:
                        try:
                            if motion_started:
                                arm.stop()
                        finally:
                            arm.close()
                finally:
                    for sig, handler in old_handlers.items():
                        signal.signal(sig, handler)
