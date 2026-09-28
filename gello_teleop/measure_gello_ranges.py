#!/usr/bin/env python3
"""Record manually chosen GELLO ranges without writing any motor register."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gello_teleop.build_gello_urdf import vector
from gello_teleop.gello_identification_hardware import IdentificationBus


def stable_snapshot(bus, count=10, sleep=time.sleep):
    samples = []
    for _ in range(count):
        sample = bus.sample()
        if np.any(sample['torque_on']):
            raise ValueError('检测到力矩开启。请退出其他控制程序，并通过其原有方式关闭力矩后重试。')
        if np.any(sample['hardware_error']):
            raise ValueError('电机硬件错误；停止测量')
        q = np.asarray(sample['raw_q_rad'], float)
        if not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 4095*np.pi/2048):
            raise ValueError('需要单圈 0..4095 编码器读数；不能将跨圈读数当作机械行程')
        samples.append(q)
        sleep(.125)
    samples = np.asarray(samples)
    if np.any(np.ptp(samples, axis=0) > np.deg2rad(1)):
        raise ValueError('采样期间关节移动超过 1°；支撑主手、保持静止后重试')
    return samples.mean(axis=0)


def measured_config(config, center_raw, endpoints_raw, margin_rad):
    """Keep the existing encoder convention explicitly unverified after measurement."""
    n = len(config['joint_ids'])
    signs = vector(config['encoder_signs'], 'encoder_signs', n)
    offsets = vector(config['encoder_offsets_rad'], 'encoder_offsets_rad', n)
    center_raw = vector(center_raw, 'center_raw', n)
    endpoints = np.asarray(endpoints_raw, float)
    if not np.all(np.abs(signs) == 1) or endpoints.shape != (n, 2) or not np.isfinite(endpoints).all():
        raise ValueError('Expected signs ±1 and two raw endpoints for each joint')
    if not np.isfinite(margin_rad) or margin_rad <= 0:
        raise ValueError('Margin must be finite and positive')
    raw_min, raw_max = endpoints.min(axis=1), endpoints.max(axis=1)
    if np.any(raw_min < 0) or np.any(raw_max > 4095*np.pi/2048):
        raise ValueError('Endpoint outside the single-turn encoder range')
    if np.any(raw_max-raw_min >= np.pi):
        raise ValueError('单轴跨度达到 180°：需要连续扫动日志排除跨圈/路径歧义，不能只凭两个端点确认。')
    mapped = (endpoints-offsets[:, None])*signs[:, None]
    lower, upper = mapped.min(axis=1)+margin_rad, mapped.max(axis=1)-margin_rad
    center = (center_raw-offsets)*signs
    clearance = np.minimum(center-lower, upper-center)
    if np.any(clearance <= np.deg2rad(.5)):
        raise ValueError('中位必须在两端留有至少 0.5° 余量；重新选择舒适中位和两端')
    result = dict(config)
    result.update(lower_rad=lower.tolist(), upper_rad=upper.tolist(), center_rad=center.tolist(),
                  amplitude_rad=np.minimum(np.deg2rad(3), .25*clearance).tolist(),
                  joint_coordinates_verified=False,
                  measured_range_status='manually selected local operating interval; no swept-volume/collision certification',
                  note='Read-only manual endpoints, expressed in the PREVIOUS encoder convention. '
                       'Confirm/recalibrate that convention against the reconstructed leader URDF before driving. '
                       'These local ranges do not prove mechanical hard limits or multi-joint clearance.')
    return result


def run(config, output, margin_rad, joints=None):
    report_path = Path(output).with_suffix('.ranges.json')
    if Path(output).exists() or report_path.exists():
        raise FileExistsError('Use new output files')
    n = len(config['joint_ids'])
    selected = list(range(n)) if joints is None else [j-1 for j in joints]
    if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= n for i in selected):
        raise ValueError(f'--joints 必须是不重复的关节编号 1..{n}')
    prior, prior_path = None, None
    endpoints, postures = [None]*n, [None]*n
    if len(selected) != n:
        if not config.get('range_measurement_file'):
            raise ValueError('局部重测需要已测配置及其 range_measurement_file 原始范围日志')
        prior_path = Path(config['range_measurement_file'])
        prior = json.loads(prior_path.read_text())
        if prior['joint_ids'] != config['joint_ids'] or prior.get('read_only') is not True:
            raise ValueError('复用范围日志的关节顺序/来源不匹配')
        old_endpoints = np.asarray(prior['endpoints_raw_rad'], float)
        if old_endpoints.shape != (n, 2) or not np.isfinite(old_endpoints).all():
            raise ValueError('复用范围日志端点格式不正确')
        if not np.isclose(margin_rad, prior['inset_margin_rad'], rtol=0, atol=1e-12):
            raise ValueError('局部重测必须保持原日志的 --margin-deg，避免改变未重测关节范围')
        # Check that the source log actually describes this config's limits.
        recovered = measured_config(config, prior['center_raw_rad'], old_endpoints, margin_rad)
        if not all(np.allclose(config[k], recovered[k], rtol=0, atol=1e-10) for k in ['lower_rad', 'upper_rad']):
            raise ValueError('复用范围日志与当前配置限位不一致')
        endpoints = old_endpoints.tolist()
        postures = list(prior['endpoint_full_postures_raw_rad'])
    with IdentificationBus(config) as bus:
        # Read-only: no torque-off writes, goals, profiles, modes or EEPROM writes.
        status = bus.inspect()
        if any(m['torque_on'] or m['mode'] != 3 or m['homing_offset'] != 0 for m in status['motors']):
            raise ValueError('需要位置模式 3、零 Homing Offset、力矩关闭；本工具不更改电机设置。')
        print('本工具只读取电机。请支撑主手，并保留线缆/橡皮筋余量。', flush=True)
        print('每次只摆动一个关节，选择中位附近的一小段舒适工作区间；不要寻找或顶住硬限位。', flush=True)
        if prior is not None:
            print('仅重测关节 '+', '.join(f'J{i+1}' for i in selected)+'；其他关节保留原始端点。', flush=True)
        print('若用于网页方向核对，请将全部关节摆成网页“返回中位”的姿态。', flush=True)
        input('把全部关节放在本次中位、保持静止，按 Enter 记录：')
        center = stable_snapshot(bus)
        for i in selected:
            joint_id = config['joint_ids'][i]
            row, poses = [], []
            for label in ['A', 'B']:
                input(f'关节 {i+1} / 电机 ID {joint_id}：手动摆至端点 {label}（中位两侧），保持其他关节和底座不动，按 Enter：')
                while True:
                    try:
                        pose = stable_snapshot(bus)
                        break
                    except ValueError as exc:
                        if '保持静止' not in str(exc):
                            raise
                        print(exc, flush=True)
                        input('保持静止后按 Enter 重试：')
                row.append(float(pose[i]))
                poses.append(pose.tolist())
                print(f'原始角度 {pose[i]:.6f} rad / {pose[i]*2048/np.pi:.1f} ticks', flush=True)
            endpoints[i] = row
            postures[i] = poses
            input(f'把关节 {i+1} 返回中位并支撑好，按 Enter 继续：')
    result = measured_config(config, center, endpoints, margin_rad)
    result['range_measurement_file'] = str(report_path.resolve())
    report = {'read_only': True, 'motor_register_writes': 0,
              'center_raw_rad': center.tolist(), 'endpoints_raw_rad': endpoints,
              'endpoint_full_postures_raw_rad': postures, 'inset_margin_rad': margin_rad,
              'joint_ids': config['joint_ids'], 'encoder_zero_verified': False,
              'method': 'Manual local endpoints, 10 stable samples each; no physical-stop search',
              'remeasured_joint_numbers': [i+1 for i in selected],
              'reused_joint_numbers': [i+1 for i in range(n) if i not in selected],
              'previous_range_measurement_file': str(prior_path.resolve()) if prior is not None else None,
              'hardware_snapshot': status}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x') as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    with Path(output).open('x') as file:
        yaml.safe_dump(result, file, sort_keys=False, allow_unicode=True)
    print(f'已保存：{output}\n范围已记录；URDF 零位/方向仍需核对，自动运动保持禁用。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--margin-deg', type=float, default=2., help='Inset from manually selected endpoints')
    parser.add_argument('--joints', type=int, nargs='+', help='Remeasure only these joint numbers; reuse other raw endpoints from the config range log')
    args = parser.parse_args()
    if not np.isfinite(args.margin_deg) or args.margin_deg <= 0:
        parser.error('--margin-deg must be finite and positive')
    try:
        run(yaml.safe_load(Path(args.config).read_text()), args.output, np.deg2rad(args.margin_deg), args.joints)
    except (ValueError, OSError, RuntimeError, EOFError) as exc:
        parser.exit(1, f'{exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, '测量已取消；未写入电机。\n')


if __name__ == '__main__':
    main()
