#!/usr/bin/env python3
"""Check GELLO joint directions against the viewer (read-only unless --execute).

Active checks hold the manually aligned pose and move one joint at a time.
Neither mode updates encoder calibration or enables general trajectory execution.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gello_teleop.build_gello_urdf import vector
from gello_teleop.gello_identification_hardware import IdentificationBus
from gello_teleop.measure_gello_ranges import stable_snapshot


def review_step(before, after, index, signs):
    before, after, signs = np.asarray(before), np.asarray(after), np.asarray(signs)
    if before.ndim != 1 or after.shape != before.shape or signs.shape != before.shape:
        raise ValueError('Invalid encoder arrays')
    if not np.isfinite(np.r_[before, after, signs]).all() or not np.all(np.abs(signs) == 1):
        raise ValueError('Need finite readings and encoder signs ±1')
    if not 0 <= index < len(before):
        raise ValueError('Invalid joint index')
    delta = after-before
    target = delta[index]
    if not np.deg2rad(.5) <= abs(target) <= np.deg2rad(15):
        raise ValueError('当前关节应小幅移动 0.5～15°；过小、过大或跨圈均不能确认方向。')
    other = np.delete(delta, index)
    if np.any(np.abs(other) > np.deg2rad(2)):
        raise ValueError('其他关节移动超过 2°；请支撑好并只转动当前关节，再重试。')
    inferred = 1 if target > 0 else -1
    return {'raw_delta_deg': np.rad2deg(delta).tolist(),
            'viewer_positive_encoder_sign': inferred,
            'configured_encoder_sign': int(signs[index]),
            'configured_sign_matches': bool(inferred == signs[index])}


def run(config, geometry, output):
    if Path(output).exists():
        raise FileExistsError('Use a new output path')
    n = len(config['joint_ids'])
    if n != 7:
        raise ValueError('This viewer check expects seven GELLO arm joints')
    signs = vector(config['encoder_signs'], 'encoder_signs', n)
    offsets = vector(config['encoder_offsets_rad'], 'encoder_offsets_rad', n)
    reference = vector(geometry['comparison_reference_q_deg'], 'model reference degrees', n)
    print('打开最新 gello_xarm7_viewer.html。主手力矩须已关闭；本工具不会更改电机。')
    print('点击网页“返回中位”，模型角度应为下列数值；将实物摆成同一姿态：')
    print(reference.tolist())
    print('这些是模型角度，不是旧标定的编码器映射角度；本次只记录对应关系。')
    print('每次只把网页一个关节增加约 3～5°，观察运动方向，再手动让实物同一关节同向小幅移动。')
    print('若实物和网页姿态明显不符，按 Ctrl+C 停止，不要猜测方向或强行摆动。')
    rows = []
    with IdentificationBus(config) as bus:
        status = bus.inspect()
        if any(m['torque_on'] or m['mode'] != 3 or m['homing_offset'] != 0 for m in status['motors']):
            raise ValueError('需要位置模式 3、零 Homing Offset、力矩关闭；请先退出其他控制程序。')
        input('确认实物与网页中位姿态一致、关节对应无误，支撑静止后按 Enter 记录参考姿态：')
        reference_raw = stable_snapshot(bus)
        for i, joint_id in enumerate(config['joint_ids']):
            while True:
                input(f'J{i+1} / ID {joint_id}：网页回到上列中位角度；实物支撑静止，按 Enter 记录起点：')
                before = stable_snapshot(bus)
                input(f'把网页 J{i+1} 增加 3～5°，实物同向转动同一关节，保持静止，按 Enter：')
                try:
                    after = stable_snapshot(bus)
                    row = review_step(before, after, i, signs)
                    row.update(joint_id=joint_id, joint_name=f'joint{i+1}',
                               before_raw_rad=before.tolist(), after_raw_rad=after.tolist())
                    break
                except ValueError as exc:
                    print(exc)
                    input('恢复起点、支撑好；按 Enter 重试，或 Ctrl+C 退出：')
            rows.append(row)
            print('现有符号与手动观察一致。' if row['configured_sign_matches'] else
                  f"发现符号不一致：现有 {row['configured_encoder_sign']:+d}，观察得到 {row['viewer_positive_encoder_sign']:+d}。未修改配置。")
            input('网页与实物都返回中位，按 Enter 继续：')
    report = {'read_only': True, 'motor_register_writes': 0,
              'geometry_verified': geometry.get('geometry_verified') is True,
              'joint_coordinates_verified': False, 'encoder_zero_verified': False,
              'method': 'Operator-matched positive viewer movement; relative direction check only',
              'reference_raw_rad': reference_raw.tolist(),
              'reference_q_in_existing_coordinates_rad': ((reference_raw-offsets)*signs).tolist(),
              'model_comparison_reference_q_deg': reference.tolist(), 'joint_checks': rows,
              'reference_model_q_rad': np.deg2rad(reference).tolist(),
              'reference_capture_method': 'Operator aligned physical arm to displayed model middle before sampling',
              'all_configured_signs_match_observation': all(r['configured_sign_matches'] for r in rows),
              'hardware_snapshot': status}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open('x') as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f'方向核对已保存：{output}。零位尚未标定，自动运动未启用。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--geometry', default='gello_teleop/models/gello_xarm7_geometry.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true', help='Enable holding torque and supervised single-joint motion')
    parser.add_argument('--step-deg', type=float, default=15., help='Active check step, at most 15 degrees')
    parser.add_argument('--move-seconds', type=float, default=8., help='Duration of each outward/return motion')
    parser.add_argument('--answer-timeout', type=float, default=120., help='Maximum hold while waiting for an answer')
    args = parser.parse_args()
    try:
        config = yaml.safe_load(Path(args.config).read_text())
        geometry = yaml.safe_load(Path(args.geometry).read_text())
        if args.execute:
            from gello_teleop.gello_direction_motion import run_active
            run_active(config, geometry, args.output, args.step_deg, args.move_seconds, args.answer_timeout)
        else:
            run(config, geometry, args.output)
    except (ValueError, OSError, RuntimeError, EOFError) as exc:
        parser.exit(1, f'{exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, '核对已取消；请检查主手支撑及力矩状态。\n' if args.execute else
                    '核对已取消；未写入电机。\n')
