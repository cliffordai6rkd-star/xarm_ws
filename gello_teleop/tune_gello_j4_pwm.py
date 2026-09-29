#!/usr/bin/env python3
"""Interactively tune the XL330 Goal PWM ceiling used to hold GELLO J4.

This tool does not measure or command a physical torque in Nm.  It keeps J4 at
its current single-turn position, enables position mode, and lets the operator
raise the X-series Goal PWM ceiling one step at a time.  ``y`` saves the chosen
ceiling to a new teleoperation YAML; ``q`` aborts and restores the original
Goal Position/Goal PWM before disabling torque.

The tool intentionally operates on one Dynamixel only and never connects to an
xArm.  Support the leader mechanically before enabling it.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import List, Optional, Tuple

import yaml


CONTROL_TABLE_MAX_PWM = 885
SUPPORTED_XL330_MODELS = {1190, 1200}  # XL330-M077 / XL330-M288
J4_INDEX = 3


def signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def load_leader(data: dict, side: Optional[str]) -> Tuple[dict, Optional[str]]:
    """Return a TeleoperatorConfig section and its dual-config side name."""
    if 'TeleoperatorConfig' in data:
        if side is not None:
            raise ValueError('--side is only valid for a dual GELLO YAML')
        leader = data['TeleoperatorConfig']
        name = None
    else:
        if set(('left', 'right')).difference(data):
            raise ValueError('Expected a single TeleoperatorConfig or left/right dual YAML')
        if side not in ('left', 'right'):
            raise ValueError('Dual GELLO YAML requires --side left or --side right')
        leader = data[side].get('TeleoperatorConfig')
        if not isinstance(leader, dict):
            raise ValueError(f'{side}: missing TeleoperatorConfig')
        name = side
    if not isinstance(leader, dict):
        raise ValueError('TeleoperatorConfig must be a mapping')
    ids = leader.get('joint_ids')
    if not isinstance(ids, list) or len(ids) <= J4_INDEX:
        raise ValueError('joint_ids must contain at least four IDs so J4 can be selected')
    if any(type(motor_id) is not int or not 0 <= motor_id <= 252 for motor_id in ids):
        raise ValueError('joint_ids must contain Dynamixel IDs in [0, 252]')
    if not leader.get('port'):
        raise ValueError('TeleoperatorConfig.port is required')
    return leader, name


def update_j4_pwm(data: dict, side: Optional[str], pwm: int) -> Tuple[int, int]:
    """Update only J4's hold_pwm_by_joint entry and return (motor_id, index)."""
    leader, _ = load_leader(data, side)
    ids = leader['joint_ids']
    if type(pwm) is not int or not 1 <= pwm <= CONTROL_TABLE_MAX_PWM:
        raise ValueError(f'J4 Goal PWM must be an integer in [1, {CONTROL_TABLE_MAX_PWM}]')
    configured = leader.get('hold_pwm_by_joint')
    if configured is None:
        default = leader.get('hold_pwm')
        if default is not None and (type(default) is not int or default < 0):
            raise ValueError('hold_pwm must be a non-negative integer')
        configured = [0 if default is None else default] * len(ids)
    else:
        configured = list(configured)
        if len(configured) != len(ids) or any(type(value) is not int or value < 0
                                              for value in configured):
            raise ValueError('hold_pwm_by_joint must contain one non-negative integer per joint')
    configured[J4_INDEX] = pwm
    leader['hold_pwm_by_joint'] = configured
    return ids[J4_INDEX], J4_INDEX


def save_config(source: Path, destination: Path, data: dict) -> None:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        raise FileExistsError('Refusing to overwrite the input YAML; choose a new --output path')
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    try:
        temporary.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_register(packet, port, motor_id: int, address: int, size: int) -> int:
    method = {1: packet.read1ByteTxRx, 2: packet.read2ByteTxRx,
              4: packet.read4ByteTxRx}[size]
    value, communication, device_error = method(port, motor_id, address)
    if communication != 0 or device_error != 0:
        detail = f'communication={communication}, device={device_error}'
        raise RuntimeError(f'ID {motor_id} read address {address} failed ({detail})')
    return value


def write_register(packet, port, motor_id: int, address: int, size: int, value: int) -> None:
    method = {1: packet.write1ByteTxRx, 2: packet.write2ByteTxRx,
              4: packet.write4ByteTxRx}[size]
    communication, device_error = method(port, motor_id, address, int(value))
    if communication != 0 or device_error != 0:
        raise RuntimeError(
            f'ID {motor_id} write address {address} failed '
            f'(communication={communication}, device={device_error})'
        )


class J4PwmTester:
    """One-motor X-series session with explicit cleanup and no xArm access."""

    def __init__(self, leader: dict, *, motor_id: int, max_temperature_c: float,
                 max_current_a: Optional[float]):
        try:
            from dynamixel_sdk import PacketHandler, PortHandler
        except ImportError as exc:
            raise RuntimeError('This tool requires the dynamixel_sdk Python package') from exc

        self.leader = leader
        self.motor_id = motor_id
        self.max_temperature_c = max_temperature_c
        self.max_current_a = max_current_a
        self.port = PortHandler(leader['port'])
        self.packet = PacketHandler(2.0)
        self.original = None
        self.active = False
        opened = False
        try:
            opened = bool(self.port.openPort())
            if not opened or not self.port.setBaudRate(int(leader.get('baudrate', 57600))):
                raise ConnectionError(f'Cannot open/set baudrate on {leader["port"]}')
            self.port.ser.exclusive = True
            self.port.ser.write_timeout = float(leader.get('read_timeout', 0.2))
            self.original = self.snapshot()
        except BaseException:
            if opened:
                self.port.closePort()
            raise

    def snapshot(self) -> dict:
        motor_id = self.motor_id
        goal_position_raw = read_register(self.packet, self.port, motor_id, 116, 4)
        return {
            'model': read_register(self.packet, self.port, motor_id, 0, 2),
            'mode': read_register(self.packet, self.port, motor_id, 11, 1),
            'pwm_limit': read_register(self.packet, self.port, motor_id, 36, 2),
            'position_max': read_register(self.packet, self.port, motor_id, 48, 4),
            'position_min': read_register(self.packet, self.port, motor_id, 52, 4),
            'torque_on': read_register(self.packet, self.port, motor_id, 64, 1),
            'hardware_error': read_register(self.packet, self.port, motor_id, 70, 1),
            'goal_pwm': signed(read_register(self.packet, self.port, motor_id, 100, 2), 16),
            'goal_position_raw': goal_position_raw,
            'goal_position': signed(goal_position_raw, 32),
            'present_pwm': signed(read_register(self.packet, self.port, motor_id, 124, 2), 16),
            'present_current_ma': signed(read_register(self.packet, self.port, motor_id, 126, 2), 16),
            'present_position': signed(read_register(self.packet, self.port, motor_id, 132, 4), 32),
            'voltage_v': read_register(self.packet, self.port, motor_id, 144, 2) * 0.1,
            'temperature_c': read_register(self.packet, self.port, motor_id, 146, 1),
        }

    def write(self, address: int, size: int, value: int) -> None:
        write_register(self.packet, self.port, self.motor_id, address, size, value)

    def prepare(self, start_pwm: Optional[int], step: int) -> Tuple[int, int]:
        status = self.original
        if status['model'] not in SUPPORTED_XL330_MODELS:
            raise ValueError(
                f'ID {self.motor_id} 型号 {status["model"]} 不在 XL330 白名单中；'
                '拒绝使用可能不兼容的控制表'
            )
        if status['mode'] != 3:
            raise ValueError(f'ID {self.motor_id} 必须是位置模式 3，当前为 {status["mode"]}')
        if status['hardware_error']:
            raise RuntimeError(f'ID {self.motor_id} hardware error=0x{status["hardware_error"]:02x}')
        if status['position_min'] > status['position_max']:
            raise ValueError('Dynamixel position limits are invalid')
        maximum = min(int(status['pwm_limit']), CONTROL_TABLE_MAX_PWM)
        if maximum < 1:
            raise ValueError(f'ID {self.motor_id} has no usable PWM range (limit={maximum})')
        if step < 1 or step > maximum:
            raise ValueError(f'--step must be in [1, {maximum}]')
        if start_pwm is None:
            start_pwm = max(0, int(status['goal_pwm']))
        if not 0 <= start_pwm <= maximum:
            raise ValueError(f'start PWM {start_pwm} exceeds safe maximum {maximum}')

        # Position mode accepts one absolute encoder turn.  Set the goal to the
        # current branch before enabling torque, so this tool never asks J4 to
        # chase a stale goal left by another session.
        goal = int(status['present_position'] % 4096)
        if not status['position_min'] <= goal <= status['position_max']:
            raise ValueError(
                f'Current position {goal} is outside Goal Position limits '
                f'[{status["position_min"]}, {status["position_max"]}]'
            )
        self.active = True
        # ``close`` will disable torque even when arming fails halfway through
        # these three register writes.
        self.write(116, 4, goal)
        self.write(100, 2, start_pwm)
        self.write(64, 1, 1)
        time.sleep(0.15)
        return start_pwm, maximum

    def safety_check(self, status: dict) -> None:
        if status['hardware_error']:
            raise RuntimeError(f'硬件错误 0x{status["hardware_error"]:02x}，立即停止')
        if status['temperature_c'] >= self.max_temperature_c:
            raise RuntimeError(
                f'温度 {status["temperature_c"]}°C 达到上限 {self.max_temperature_c:g}°C，立即停止'
            )
        if self.max_current_a is not None and abs(status['present_current_ma']) >= self.max_current_a * 1000:
            raise RuntimeError(
                f'输入电流 {status["present_current_ma"]}mA 达到上限 '
                f'{self.max_current_a:g}A，立即停止'
            )

    def close(self) -> List[str]:
        failures = []
        if self.active:
            # Disable output before restoring the old goal.  Restoring a stale
            # goal while torque is still on could make the motor move suddenly.
            for address, size, value in (
                (64, 1, 0),
                (116, 4, self.original['goal_position_raw']),
                (100, 2, max(0, self.original['goal_pwm'])),
            ):
                try:
                    self.write(address, size, value)
                except Exception as exc:
                    failures.append(f'address {address}: {exc}')
            self.active = False
        try:
            self.port.closePort()
        except Exception as exc:
            failures.append(f'close port: {exc}')
        return failures


@contextmanager
def cbreak_stdin():
    if not sys.stdin.isatty():
        raise RuntimeError('交互调节需要在真实终端中运行')
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        print('', flush=True)


def print_status(status: dict, pwm: int, maximum: int) -> None:
    position_error = status['goal_position'] - status['present_position']
    print(
        f'\rJ4 PWM={pwm}/{maximum}  实际PWM={status["present_pwm"]:+d}  '
        f'电流={status["present_current_ma"]:+d}mA  温度={status["temperature_c"]}°C  '
        f'位置误差={position_error:+d} ticks  电压={status["voltage_v"]:.1f}V',
        end='', flush=True,
    )


def tune(tester: J4PwmTester, *, step: int, start_pwm: Optional[int],
         settle_s: float) -> int:
    current_pwm, maximum = tester.prepare(start_pwm, step)
    status = tester.snapshot()
    tester.safety_check(status)
    print(
        f'ID {tester.motor_id} 已进入位置保持：目标={status["goal_position"] % 4096} ticks，'
        f'当前 Goal PWM={current_pwm}，安全上限={maximum}。', flush=True,
    )
    print('按 d 增加一个步长，按 y 接受并保存，按 q 退出；请始终托住 J4。', flush=True)
    with cbreak_stdin() as fd:
        while True:
            status = tester.snapshot()
            tester.safety_check(status)
            print_status(status, current_pwm, maximum)
            ready, _, _ = select.select([fd], [], [], max(0.05, settle_s))
            if not ready:
                continue
            key = os.read(fd, 1).decode(errors='ignore').lower()
            if key == 'd':
                new_pwm = min(maximum, current_pwm + step)
                if new_pwm == current_pwm:
                    print('\n已经达到 PWM 安全上限。按 y 保存或 q 退出。', flush=True)
                else:
                    tester.write(100, 2, new_pwm)
                    current_pwm = new_pwm
                    print(f'\n已将 Goal PWM 提高到 {current_pwm}，等待 {settle_s:g}s。', flush=True)
                    time.sleep(settle_s)
            elif key == 'y':
                print(f'\n接受 J4 Goal PWM={current_pwm}。', flush=True)
                return current_pwm
            elif key in ('q', '\x03', '\x1b'):
                raise KeyboardInterrupt('用户取消调节')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, help='单臂或双臂 GELLO YAML')
    parser.add_argument('--side', choices=('left', 'right'), help='双臂 YAML 中的 GELLO 侧')
    parser.add_argument('-o', '--output', required=True, help='保存调节结果的新 YAML，禁止覆盖输入文件')
    parser.add_argument('--step', type=int, default=25, help='每次按 d 增加的 Goal PWM，默认 25')
    parser.add_argument('--start-pwm', type=int, help='起始 Goal PWM；默认读取电机当前值')
    parser.add_argument('--settle-s', type=float, default=0.5, help='每次提高 PWM 后等待时间，默认 0.5 秒')
    parser.add_argument('--max-temperature-c', type=float, default=50.0,
                        help='温度保护上限，默认 50°C')
    parser.add_argument('--max-current-a', type=float, default=0.6,
                        help='输入电流保护上限，默认 0.6A；设为 0 可关闭')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.step < 1 or args.settle_s <= 0 or not math.isfinite(args.settle_s):
        raise ValueError('--step 必须为正整数，--settle-s 必须为正数')
    if args.max_temperature_c <= 0 or not math.isfinite(args.max_temperature_c):
        raise ValueError('--max-temperature-c 必须为正数')
    max_current = None if args.max_current_a == 0 else args.max_current_a
    if max_current is not None and (max_current <= 0 or not math.isfinite(max_current)):
        raise ValueError('--max-current-a 必须为正数，或设为 0 关闭')

    source = Path(args.config).expanduser().resolve()
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict):
        raise ValueError('配置文件必须是 YAML mapping')
    leader, side = load_leader(data, args.side)
    motor_id = leader['joint_ids'][J4_INDEX]
    tester = None
    accepted = None
    cleanup_failures = []
    try:
        print(f'即将测试 {side or "当前"} GELLO J4 / Dynamixel ID {motor_id}。', flush=True)
        print('请停止其他 GELLO 程序，托住主手并确认 J4 附近没有夹点。', flush=True)
        input('准备好后按 Enter 开始；Ctrl+C 取消：')
        tester = J4PwmTester(leader, motor_id=motor_id,
                             max_temperature_c=args.max_temperature_c,
                             max_current_a=max_current)
        accepted = tune(tester, step=args.step, start_pwm=args.start_pwm,
                        settle_s=args.settle_s)
    except KeyboardInterrupt as exc:
        print(f'\n已取消：{exc}', flush=True)
        return 130
    finally:
        if tester is not None:
            cleanup_failures = tester.close()
            if cleanup_failures:
                print('清理未完全成功：' + '; '.join(cleanup_failures), flush=True)

    if accepted is None:
        return 130
    update_j4_pwm(data, side, accepted)
    save_config(source, Path(args.output), data)
    print(f'已保存：{Path(args.output).expanduser().resolve()}', flush=True)
    print(f'J4 / ID {motor_id} 的 hold_pwm_by_joint[3] = {accepted}。', flush=True)
    print('后续遥操请使用这个输出 YAML；它只影响回位/对齐阶段的位置保持。', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ConnectionError, OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(f'错误：{exc}')
