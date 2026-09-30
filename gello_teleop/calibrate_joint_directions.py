#!/usr/bin/env python3
"""单侧 GELLO 原点与选轴方向标定：--left/--right，输入轴号，结果实时覆盖。

xArm 选中轴正向运动 15°；人手动让 GELLO 同轴按相同物理方向运动约 15°。
记录原点后按数采配置启用纯阻尼；--no-damping 可关闭。结果供数采直接复用。
旧零点工具使用的纯映射辅助函数保留在本模块。
"""
import math
from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gello_teleop.uf_robot_gello_teleop import Session, load_configs

def checked_reference(robot_q, leader_q, n):
    robot_q = np.asarray(robot_q, dtype=float).reshape(-1)
    leader_q = np.asarray(leader_q, dtype=float).reshape(-1)
    if robot_q.shape != (n,) or leader_q.shape != (n,):
        raise ValueError("Invalid calibration reference dimension")
    if not np.isfinite(robot_q).all() or not np.isfinite(leader_q).all():
        raise ValueError("Invalid calibration reference")
    return robot_q, leader_q


def calibration_values(robot_q, leader_q, signs):
    signs = np.asarray(signs, dtype=int).reshape(-1)
    robot_q, leader_q = checked_reference(robot_q, leader_q, len(signs))
    if not np.isin(signs, (-1, 1)).all():
        raise ValueError("Every joint requires a y/n direction answer before saving")
    return {
        "reset_q": robot_q.tolist(),
        "joint_signs": signs.tolist(),
        "joint_offsets": (leader_q - robot_q / signs).tolist(),
        "leader_reference_q": leader_q.tolist(),
    }


def _move_arm(arm, target):
    target = np.asarray(target, dtype=float)
    arm.health(active=True)
    arm.check_target(target)
    before = np.asarray(arm.joints(), dtype=float)
    print(f'xArm {arm.config.robot_ip} 开始运动（角度单位：度）', flush=True)
    arm._command(
        arm.api.set_servo_angle,
        angle=target.tolist(), speed=math.radians(20), mvacc=math.radians(10),
        is_radian=True, wait=True,
    )
    arm.health(active=True)
    actual = np.asarray(arm.joints(), dtype=float)
    errors = actual - target
    tolerance = math.radians(0.25)
    for i in range(len(target)):
        if abs(target[i] - before[i]) > math.radians(0.01) or abs(errors[i]) > tolerance:
            print(
                f'  J{i + 1}: 起点={math.degrees(before[i]):+.3f}°, '
                f'目标={math.degrees(target[i]):+.3f}°, '
                f'实际={math.degrees(actual[i]):+.3f}°, '
                f'实测变化={math.degrees(actual[i] - before[i]):+.3f}°',
                flush=True,
            )
    if np.max(np.abs(errors)) > tolerance:
        index = int(np.argmax(np.abs(errors)))
        raise RuntimeError(
            f'xArm {arm.config.robot_ip} 指令返回后未到位：J{index + 1} '
            f'误差={math.degrees(errors[index]):+.3f}°，允许误差=0.25°；'
            '停止标定，不继续驱动 GELLO'
        )


def save_result(source, destination, results):
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if source == destination:
        raise ValueError("Calibration output must be a separate file")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; choose another --output")
    data = yaml.safe_load(source.read_text())
    for side in ("left", "right"):
        if side not in data or side not in results:
            raise ValueError("Both left and right sides must finish before saving")
        values = results[side]
        data[side]["RobotConfig"]["reset_q"] = values["reset_q"]
        teleop = data[side]["TeleoperatorConfig"]
        teleop["joint_signs"] = values["joint_signs"]
        teleop["joint_offsets"] = values["joint_offsets"]
        teleop["leader_reference_q"] = values["leader_reference_q"]
        teleop['leader_passive'] = values.get('leader_passive', False)
        if 'gripper_open_deg' in values:
            teleop['gripper_open_deg'] = values['gripper_open_deg']
            teleop['gripper_close_deg'] = values.get('gripper_close_deg', values['gripper_open_deg']+teleop.get('gripper_travel_deg', -42))
        for field in ('leader_reset_speed_deg', 'leader_reset_timeout', 'leader_reset_max_travel_deg'):
            if field in values:
                teleop[field] = values[field]
        # Keep legacy readers understandable when inspecting the generated file.
        teleop["start_joints"] = values["reset_q"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    temporary.replace(destination)
    return destination


def prepare_manual_reference(session, samples=10, interval=.05):
    """Capture both current xArm poses before enabling; never approach old reset_q."""
    captured = []
    for arm in session.arms:
        readings = []
        for _ in range(samples):
            if session.stop_event.is_set():
                raise KeyboardInterrupt
            if arm.health() == 1:
                raise ValueError('xArm 正在运动，请停止后重新记录参考姿态')
            readings.append(arm.joints())
            time.sleep(interval)
        readings = np.asarray(readings, dtype=float)
        if not np.isfinite(readings).all() or np.max(np.ptp(readings, axis=0)) > math.radians(1):
            raise ValueError('xArm 参考姿态不稳定；未开始试动')
        q = readings.mean(axis=0)
        arm.check_target(q)
        captured.append(q)
    # Validate both before changing either mode. There is no position command
    # here: prepare_reset only enables normal position mode at the current pose.
    for (_, robot, leader), q in zip(session.configs, captured):
        robot.reset_q = tuple(q)
        leader.start_joints = tuple(q)
        leader.joint_offsets = None
        leader.leader_reference_q = None
    session.motion_started = True
    for arm, q in zip(session.arms, captured):
        arm.prepare_reset()
        if np.max(np.abs(arm.joints()-q)) > math.radians(1):
            raise ValueError('xArm 使能时姿态改变；停止标定')
    session.stage = 'reset'


def main(argv=None):
    from gello_teleop.selected_joint_calibration import main as selected_main
    return selected_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
