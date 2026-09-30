#!/usr/bin/env python3
"""独立读取正在遥操的所选主从臂的 q，覆盖数采下一次启动使用的参考姿态。

在线遥操按键保存请使用 dual_gello_collect --reset-q，然后按 s。
本工具额外支持多次静止采样，要求数采仍在运行；--dry-run 只打印，不覆盖。
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gello_teleop.reference_pose import DEFAULT_CONFIG, capture_live_pose, overwrite_reference_pose


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default=str(DEFAULT_CONFIG), help='与正在运行的数采脚本使用同一配置')
    parser.add_argument('--samples', type=int, default=5, help='每侧独立新采样的数量，默认 5')
    parser.add_argument('--max-motion-deg', type=float, default=1., help='采样期间每轴允许的最大变化，默认 1°')
    parser.add_argument('--timeout', type=float, default=3., help='等待新反馈的超时秒数，默认 3')
    parser.add_argument('--dry-run', action='store_true', help='只读取打印，不修改文件')
    parser.add_argument('--json', action='store_true', help='输出完整采样记录 JSON')
    args = parser.parse_args(argv)
    try:
        pose = capture_live_pose(args.config, samples=args.samples,
                                 max_motion_deg=args.max_motion_deg, timeout_s=args.timeout)
        destination = None if args.dry_run else overwrite_reference_pose(pose)
        if args.json:
            pose['saved_to'] = str(destination) if destination else None
            print(json.dumps(pose, ensure_ascii=False, indent=2))
        else:
            for side in pose['arms']:
                for device in ('xarm', 'gello'):
                    q = pose['arms'][side][device+'_q']
                    print(f'{side} {device} q_rad = {json.dumps(q)}')
                    print(f'{side} {device} q_deg = {np.round(np.rad2deg(q), 3).tolist()}')
            print(f'已覆盖保存所选臂参考：{destination}；重启数采后使用新姿态对齐。' if destination else
                  '只读检查完成；标定文件未修改。')
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'参考姿态未保存：{exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
