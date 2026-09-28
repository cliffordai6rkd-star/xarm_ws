#!/usr/bin/env python3
"""手动参考姿态零点标定：不驱动 xArm，不使能 GELLO，不自动试转。"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gello_teleop.calibrate_joint_directions import calibration_values, save_result
from gello_teleop.gello_hardware import Arm, GelloReader
from gello_teleop.uf_robot_gello_teleop import load_configs, validate


def stable_reference(arm, reader, samples=10, interval=0.05, tolerance_deg=1.0):
    """Reject motion during capture; sample feedback only, never command a pose."""
    robot_samples, leader_samples = [], []
    for _ in range(samples):
        state = arm.health()
        if state == 1:
            raise ValueError('xArm 正在运动，请停止并稳定后重试')
        robot_samples.append(arm.joints())
        leader_samples.append(reader.read())
        time.sleep(interval)
    for name, values in (('xArm', robot_samples), ('GELLO', leader_samples)):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'{name} 返回无效角度')
        spread = np.ptp(values, axis=0)
        if np.max(spread) > np.deg2rad(tolerance_deg):
            raise ValueError(f'{name} 采样时发生移动，各轴变化(度)={np.round(np.rad2deg(spread), 2)}')
    return np.mean(robot_samples, axis=0), np.mean(leader_samples, axis=0)


def capture_side(side, robot, leader, manual_start=False):
    arm = reader = None
    try:
        arm = Arm(robot)
        if arm.api.axis != len(leader.joint_ids):
            raise ValueError('xArm 轴数与 joint_ids 不一致')
        arm.health()
        reader = GelloReader(leader)
        print(f'\n{side}: 已连接 xArm {robot.robot_ip} / GELLO {leader.port}', flush=True)
        print('请托住 GELLO；按 Enter 关闭主手力矩，输入 q 退出。', flush=True)
        if input().strip().lower() == 'q':
            raise KeyboardInterrupt
        reader.set_torque(False)
        while True:
            print('将 GELLO 手动摆成与从臂对应的参考姿态，打开主手夹爪。\n'
                  '无需把编码器数值摆成 0；托住并保持不动，按 Enter 采样；q 退出。', flush=True)
            if input().strip().lower() == 'q':
                raise KeyboardInterrupt
            try:
                robot_q, raw = stable_reference(arm, reader)
            except ValueError as exc:
                print(f'未保存：{exc}', flush=True)
                continue
            n = len(leader.joint_ids)
            arm.check_target(robot_q)
            # Position mode returns to single-turn feedback when torque turns on.
            # Store the corresponding branch and recompute offsets together.
            reference = raw[:n] if manual_start else (np.rint(raw[:n] * 2048 / np.pi) % 4096) * np.pi / 2048
            if not manual_start:
                for motor_id, angle in zip(leader.joint_ids, reference):
                    reader._checked_position_goal(motor_id, int(round(angle * 2048 / np.pi)))
            values = calibration_values(robot_q, reference, leader.joint_signs)
            values['leader_passive'] = manual_start
            for field in ('leader_reset_speed_deg', 'leader_reset_timeout', 'leader_reset_max_travel_deg'):
                values[field] = getattr(leader, field)
            # Check the captured reset pose against the actual config constraints.
            robot.reset_q = tuple(robot_q)
            leader.start_joints = tuple(robot_q)
            leader.joint_offsets = tuple(values['joint_offsets'])
            leader.leader_reference_q = tuple(values['leader_reference_q'])
            leader.leader_passive = manual_start
            validate(robot, leader)
            for i, motor_id in enumerate(leader.joint_ids):
                print(f'J{i+1} <- ID {motor_id}: GELLO参考={np.rad2deg(reference[i]):+.2f}°, '
                      f'xArm参考={np.rad2deg(robot_q[i]):+.2f}°, 方向={leader.joint_signs[i]:+d}')
            print('确认姿态和关节对应正确：输入 y 保存此侧，r 重新采样，其他输入退出。', flush=True)
            answer = input().strip().lower()
            if answer == 'y':
                return values
            if answer != 'r':
                raise KeyboardInterrupt
    finally:
        try:
            if reader is not None:
                reader.close()
        finally:
            if arm is not None:
                arm.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, help='双臂 YAML；使用其中的 joint_ids / joint_signs')
    parser.add_argument('--output', required=True, help='新的零点配置文件；不能覆盖已有文件')
    parser.add_argument('--manual-start', action='store_true', help='可选：以后手动回位；默认以后开机自动回位')
    args = parser.parse_args(argv)
    source = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    # Refuse existing output before any hardware connection.
    if output == source or output.exists():
        parser.error('输出文件已存在或与输入相同，请换一个 --output 文件名')
    configs = load_configs(source, dual=True)
    if any(leader.torque_joint_ids for _, _, leader in configs):
        parser.error('此零点标定要求 torque_joint_ids 为空')
    print('请先通过 xArm 控制界面把两侧从臂摆到安全、易对应的参考姿态并停止。\n'
          '本脚本只读取从臂角度；只向 GELLO 发送关闭力矩指令。\n'
          '方向沿用输入配置，不自动识别方向或关节顺序。', flush=True)
    try:
        results = {side: capture_side(side, robot, leader, args.manual_start) for side, robot, leader in configs}
        path = save_result(source, output, results)
        startup = ('GELLO 手动大致回位。' if args.manual_start else
                   'GELLO 将低速自动回到保存位置并保持；确认遥操作后释放力矩。')
        print(f'零点已保存：{path}\n下次遥操作使用 --config {path}\n'
              f'从臂将复位到记录姿态；{startup}')
        return 0
    except KeyboardInterrupt:
        print('已取消，未保存本次结果。')
        return 130
    except Exception as exc:
        print(f'零点标定失败，未保存本次结果：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
