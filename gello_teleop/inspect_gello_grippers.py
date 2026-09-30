#!/usr/bin/env python3
"""只读查看正在运行的数采进程的夹爪映射和反馈，不重复连接硬件。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gello_teleop.reference_pose import DEFAULT_CONFIG, live_pose_path


def read_gripper_status(config_path, sides=None):
    config_path = Path(config_path).expanduser().resolve()
    try:
        pose = json.loads(live_pose_path(config_path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RuntimeError('未找到反馈缓存；请先用同一配置启动新版 dual_gello_collect 并完成接管') from None
    if pose.get('format') != 'dual_gello_live_pose_v1' or pose.get('config_path') != str(config_path):
        raise RuntimeError('反馈缓存与配置不一致')
    age = time.monotonic()-float(pose['published_monotonic_s'])
    if not np.isfinite(age) or not 0 <= age <= .25:
        raise RuntimeError('反馈缓存已过期；请检查数采进程是否仍在运行')
    if not pose.get('valid'):
        raise RuntimeError(f"无法读取夹爪：{pose.get('reason', '无效反馈')}")
    if sides is None:
        sides = tuple(side for side in ('left', 'right') if side in pose['arms'])
    result = {}
    for side in sides:
        if side not in pose['arms']:
            raise RuntimeError(f'{side} 未启用；请检查 active_arms 配置')
        arm = pose['arms'][side]
        grip = arm.get('gripper')
        if grip is None:
            raise RuntimeError('当前数采进程尚未发布夹爪状态；请退出并重启新版数采脚本')
        source_age = (time.time_ns()//1000-int(arm['gello_timestamp_us']))/1e6
        if not 0 <= source_age <= float(pose['leader_max_age_s']):
            raise RuntimeError(f'{side} GELLO 夹爪采样已过期')
        result[side] = dict(grip, gello_motor_id=arm['identity']['gripper_id'])
    return result


def _number(value, scale=1., unit=''):
    return '不可用' if value is None else f'{value*scale:.1f}{unit}'


def format_status(status):
    parts = []
    for side, grip in status.items():
        parts.append(
            f"{side} ID{grip['gello_motor_id']} 原始={_number(grip['gello_angle_deg'], unit='°')} "
            f"闭合={_number(grip['closure_fraction'], 100., '%')} "
            f"映射开口={_number(grip['desired_width_m'], 1000., 'mm')} "
            f"已发={_number(grip['command_width_m'], 1000., 'mm')} "
            f"实际={_number(grip['actual_width_m'], 1000., 'mm')} "
            f"端点={_number(grip['open_deg'])}→{_number(grip['close_deg'])}° "
            f"{'跟随' if grip['following'] else '保持/未下发'} 模式={grip.get('mode', 'follow')}")
    return ' | '.join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default=str(DEFAULT_CONFIG), help='与数采进程使用同一配置')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--left', action='store_true', help='只看左侧')
    selection.add_argument('--right', action='store_true', help='只看右侧')
    parser.add_argument('--watch', action='store_true', help='每 0.2 秒刷新；Ctrl+C 退出查看')
    parser.add_argument('--json', action='store_true', help='输出完整夹爪状态 JSON')
    args = parser.parse_args(argv)
    sides = ('left',) if args.left else ('right',) if args.right else None
    try:
        while True:
            status = read_gripper_status(args.config, sides)
            output = json.dumps(status, ensure_ascii=False, indent=2) if args.json else format_status(status)
            if args.watch and sys.stdout.isatty() and not args.json:
                print('\r\033[2K'+output, end='', flush=True)
            else:
                print(output, flush=True)
            if not args.watch:
                return 0
            time.sleep(.2)
    except KeyboardInterrupt:
        print()
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'夹爪检查失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
