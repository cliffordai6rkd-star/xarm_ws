"""独立 xArm G1 夹爪测试：回车闭合/张开，持续打印真实反馈。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import yaml

from nero_collection.keyboard import TerminalKeys

DEFAULT_CONFIG = Path(__file__).resolve().parent/'config/xarm7_gello_dual_dataset.yaml'


@dataclass(frozen=True)
class Settings:
    side: str
    ip: str
    open_position: int
    close_position: int
    speed: int | None
    min_width_m: float
    max_width_m: float


def load_settings(config, side, ip=None):
    config = Path(config).expanduser().resolve()
    workflow = yaml.safe_load(config.read_text())
    calibration_path = Path(workflow.get('gello_config', 'xarm7_gello_calibration.yaml'))
    if not calibration_path.is_absolute():
        calibration_path = config.parent/calibration_path
    robot = yaml.safe_load(calibration_path.read_text())[side]['RobotConfig']
    if robot.get('gripper_type') != 1:
        raise ValueError('本脚本使用 xArm G1 协议；gripper_type 必须为 1')
    arm = workflow.get('arms', {}).get(side, {})
    opened = int(arm.get('gripper_open_position', 800))
    closed = int(arm.get('gripper_close_position', 0))
    minimum, maximum = float(arm.get('gripper_min_width_m', 0.)), float(arm.get('gripper_max_width_m', .085))
    speed = arm.get('gripper_speed')
    if speed is not None and (type(speed) is not int or (speed != -1 and speed <= 0)):
        raise ValueError('gripper_speed 必须为正整数或 -1')
    if opened == closed or not 0 <= minimum < maximum:
        raise ValueError('夹爪开闭位置/开口配置无效')
    return Settings(side, ip or robot['robot_ip'], opened, closed,
                    None if speed in (None, -1) else speed, minimum, maximum)


def _read_register(arm, address, count=1):
    """Read actual G1 registers using the installed SDK's Modbus transport."""
    backend = getattr(arm, '_arm', None)
    command = getattr(backend, 'arm_cmd', None)
    if command is None or not callable(getattr(command, 'gripper_modbus_r16s', None)):
        return 'SDK不支持寄存器读取'
    result = command.gripper_modbus_r16s(address, count)
    # SDK's response: [transport_code, host_id, slave_id, function, byte_count, data...].
    if not isinstance(result, (tuple, list)) or len(result) != 5+count*2 or result[0] != 0:
        return result
    value = int.from_bytes(bytes(result[5:]), 'big', signed=count == 2)
    return (result[0], value)


def _optional_call(arm, name, **kwargs):
    fn = getattr(arm, name, None)
    return fn(**kwargs) if callable(fn) else 'SDK不支持'


class GripperConsole:
    def __init__(self, arm, settings, emit=print):
        self.arm, self.settings, self.emit = arm, settings, emit
        self.initialized = False
        self.next_close = True
        self.last_command = None

    def _send(self, name, *args, **kwargs):
        result = getattr(self.arm, name)(*args, **kwargs)
        self.emit(f'{name}{args} 参数={kwargs} 返回={result!r}')
        code = result[0] if isinstance(result, (tuple, list)) else result
        if code != 0:
            raise RuntimeError(f'{name} 失败，返回={result!r}')

    def enter(self):
        if not self.initialized:
            # Configure position mode before enabling; never move or reset the arm.
            self._send('set_gripper_mode', 0)
            self._send('set_gripper_enable', True)
            if self.settings.speed is not None:
                self._send('set_gripper_speed', self.settings.speed)
            self.initialized = True
        closing = self.next_close
        target = self.settings.close_position if closing else self.settings.open_position
        self.emit(f">>> {'闭合' if closing else '张开'}：目标原始位置={target}")
        # wait=False alone still waits for arm motion in SDK 1.18.5.
        self._send('set_gripper_position', target, wait=False, wait_motion=False)
        self.last_command = target
        self.next_close = not closing
        self.status()

    def status(self):
        pos = self.arm.get_gripper_position(check_baud=False)
        error = self.arm.get_gripper_err_code(check_baud=False)
        backend = getattr(self.arm, '_arm', None)
        status = _optional_call(backend, 'get_gripper_status', check_baud=False)
        width = '不可用'
        if isinstance(pos, (tuple, list)) and len(pos) == 2 and pos[0] == 0 and pos[1] is not None:
            s = self.settings
            # Show the measured width without clipping away out-of-range feedback.
            width = f'{1000*(s.min_width_m+(pos[1]-s.close_position)/(s.open_position-s.close_position)*(s.max_width_m-s.min_width_m)):.1f}mm'
        meaning = {0: '停止', 1: '运动', 2: '抓取'}.get(status[1] & 3, '未知') if (
            isinstance(status, (tuple, list)) and len(status) == 2 and status[0] == 0) else '不可用'
        self.emit(
            f'{time.strftime("%H:%M:%S")} {self.settings.side} '
            f'已发={self.last_command} 位置返回={pos!r} 开口={width} '
            f'夹爪错误={error!r} 状态={status!r}({meaning}) '
            f'实际使能={_read_register(self.arm, 0x0100)!r} '
            f'实际模式={_read_register(self.arm, 0x0101)!r} '
            f'实际速度={_read_register(self.arm, 0x0303)!r} '
            f'目标寄存器={_read_register(self.arm, 0x0700, 2)!r} '
            f'机械臂状态={_optional_call(self.arm, "get_state")!r} '
            f'机械臂错误/警告={_optional_call(self.arm, "get_err_warn_code")!r}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--side', choices=('left', 'right'), default='left')
    parser.add_argument('--ip', help='覆盖所选侧的机器人 IP')
    parser.add_argument('--interval', type=float, default=.5, help='反馈打印间隔，秒，默认 0.5')
    parser.add_argument('--read-only', action='store_true', help='只读状态，回车不发送指令')
    parser.add_argument('--check-config', action='store_true', help='打印连接参数，不连接硬件')
    args = parser.parse_args(argv)
    if not .1 <= args.interval <= 10.:
        parser.error('--interval 必须在 0.1 到 10 秒之间')
    arm = None
    try:
        settings = load_settings(args.config, args.side, args.ip)
        print(json.dumps(asdict(settings), ensure_ascii=False, indent=2), flush=True)
        if args.check_config:
            return 0
        from xarm.wrapper import XArmAPI
        with TerminalKeys() as keys:
            if not keys.is_tty:
                raise RuntimeError('需要交互终端；请直接在终端运行')
            print('请先退出遥操脚本，避免同时控制同一夹爪。', flush=True)
            arm = XArmAPI(settings.ip)
            arm.set_timeout(1.)
            console = GripperConsole(arm, settings, emit=lambda message: print(message, flush=True))
            backend = getattr(arm, '_arm', None)
            print(f'夹爪固件={_optional_call(backend, "get_gripper_version", check_baud=False)!r} '
                  f'控制器仿真模式={getattr(backend, "is_simulation_robot", "未知")}', flush=True)
            print('启动仅读取状态；回车依次闭合、张开；q/Ctrl+C 断开连接退出。'
                  if not args.read_only else '只读模式：回车仅打印状态；q/Ctrl+C 退出。', flush=True)
            next_status = 0.
            while True:
                key = keys.read_key(max(0., min(.1, next_status-time.monotonic())))
                if key in {'q', 'Q', '\x03'}:
                    return 0
                if key in {'\r', '\n'}:
                    try:
                        console.status() if args.read_only else console.enter()
                    except RuntimeError as exc:
                        print(f'指令未成功：{exc}；下一次回车仍重试同一动作。', flush=True)
                    next_status = time.monotonic()+args.interval
                if time.monotonic() >= next_status:
                    console.status()
                    next_status = time.monotonic()+args.interval
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'夹爪测试失败：{exc}', flush=True)
        return 1
    finally:
        if arm is not None:
            arm.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
