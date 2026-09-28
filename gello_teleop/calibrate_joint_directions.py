#!/usr/bin/env python3
"""逐轴标定 GELLO 方向和零点，并把结果写入新的 YAML 配置。

标定过程会先把两台 xArm 复位，然后逐侧记录参考姿态。每个关节再依次
向正方向移动一个小角度，操作者观察后输入 y（同向）或 n（反向）。原始配置
不会被覆盖，输出文件可直接交给 uf_robot_gello_teleop_dual.py。
"""
import argparse
from copy import deepcopy
import logging
import math
from pathlib import Path
import select
import signal
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gello_teleop.uf_robot_gello_teleop import Session, load_configs

logger = logging.getLogger("gello_calibration")


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
        angle=target.tolist(), speed=math.radians(2), mvacc=math.radians(10),
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


def wait_reference(arm, leader, n):
    """Show both raw angles while the operator aligns the physical reference."""
    print("请按物理关节姿态对齐；GELLO 编码器零点尚未标定，不能直接比较角度数值。", flush=True)
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if ready:
            if sys.stdin.readline() == "":
                raise EOFError("Calibration requires an interactive terminal")
            return
        robot_q = np.asarray(arm.joints(), dtype=float)[:n]
        leader_q = np.asarray(leader.read(), dtype=float)[:n]
        values = " ".join(
            f"J{i + 1}: GELLO {math.degrees(leader_q[i]):+.1f}° / xArm {math.degrees(robot_q[i]):+.1f}°"
            for i in range(n)
        )
        print(values, flush=True)


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


def run_calibration(session, delta_deg=2.0):
    if not (0.5 <= delta_deg <= 5.0):
        raise ValueError("delta-deg must be between 0.5 and 5")
    results = {}
    delta = math.radians(delta_deg)
    for index, (side, robot_config, leader_config) in enumerate(session.configs):
        arm = session.arms[index]
        leader = session.readers[index]
        label = {"left": "左侧", "right": "右侧"}.get(side, side)
        print(f"\n{label}：请将 GELLO 摆到与 reset_q 对应的位置，打开夹爪后按 Enter。", flush=True)
        wait_reference(arm, leader, len(leader_config.joint_ids))
        # Capture the reference after torque-on: position-mode feedback can
        # discard whole-turn offsets when the motor starts holding its pose.
        leader.hold()
        robot_zero, leader_zero = checked_reference(
            arm.joints(), leader.read()[:len(leader_config.joint_ids)], len(leader_config.joint_ids))
        signs = []
        for joint in range(len(signs), len(leader_config.joint_ids)):
            while True:
                print(f"{label} J{joint + 1}: 按 Enter 让 xArm 和 GELLO 依次正向移动 {delta_deg:g}°；输入 q 退出", flush=True)
                ready = input().strip().lower()
                if ready == 'q':
                    raise KeyboardInterrupt
                if ready:
                    print('请直接按 Enter 开始，或输入 q 退出。', flush=True)
                    continue
                robot_target = robot_zero.copy(); robot_target[joint] += delta
                leader_target = leader_zero.copy(); leader_target[joint] += delta
                _move_arm(arm, robot_target)
                print(f'{label} xArm 已核实到位，现在驱动 GELLO J{joint + 1}。', flush=True)
                try:
                    leader.move(leader_target, duration=5.0, tolerance=math.radians(0.5))
                except TimeoutError as exc:
                    raise TimeoutError(f'{label} J{joint + 1} 正向试动失败：{exc}') from exc
                answer = input(
                    '先确认动的是同一物理关节；输入 y=同向 n=反向 x=关节不对应 r=重试 q=退出：'
                ).strip().lower()
                _move_arm(arm, robot_zero)
                try:
                    leader.move(leader_zero, duration=5.0, tolerance=math.radians(0.5))
                except TimeoutError as exc:
                    raise TimeoutError(f'{label} J{joint + 1} 试动后返回参考姿态失败：{exc}') from exc
                if answer == 'q':
                    raise KeyboardInterrupt
                if answer == 'x':
                    raise ValueError(
                        f'{label} xArm J{joint + 1} 与 GELLO ID {leader_config.joint_ids[joint]} '
                        '不是同一物理关节；调整 joint_ids 顺序后重新标定'
                    )
                if answer == 'r':
                    continue
                if answer not in ('y', 'n'):
                    print('请输入 y、n、x、r 或 q；该关节将重新测试。', flush=True)
                    continue
                signs.append(1 if answer == "y" else -1)
                print(f"{label} J{joint + 1}: joint_signs={signs[-1]:+d}", flush=True)
                break
        results[side] = calibration_values(robot_zero, leader_zero, signs)
        leader.hold()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="双臂原始 YAML")
    parser.add_argument("--output", help="输出标定 YAML，默认在源文件名后追加 _calibrated")
    parser.add_argument("--delta-deg", type=float, default=2.0)
    parser.add_argument("--show-mapping", action="store_true", help="只读显示已保存的逐轴映射，不连接硬件")
    args = parser.parse_args(argv)
    source = Path(args.config).expanduser().resolve()
    if args.show_mapping:
        for side, robot, leader in load_configs(source, dual=True, require_calibrated=True):
            print(f'{side}: {source}')
            for index, (motor_id, sign, offset) in enumerate(zip(
                leader.joint_ids, leader.joint_signs, leader.joint_offsets
            ), start=1):
                print(f'  xArm J{index} <- GELLO ID {motor_id}: '
                      f'q_robot = {sign:+d} * (q_gello - {offset:+.6f} rad)')
        return 0
    output = Path(args.output).expanduser().resolve() if args.output else source.with_name(source.stem + "_calibrated.yaml")
    configs = load_configs(source, dual=True, require_calibrated=False)
    session = Session(configs)
    previous = {sig: signal.signal(sig, lambda *_: session.stop_event.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        session.connect()
        print("确认复位路径畅通，按 Enter 让两台 xArm 回到 reset_q。", flush=True)
        input()
        session.reset()
        results = run_calibration(session, args.delta_deg)
        path = save_result(source, output, results)
        print(f"标定结果已保存：{path}\n下次遥操请使用 --config {path}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        logger.exception("Calibration stopped; incomplete results were not saved")
        return 1
    finally:
        session.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
