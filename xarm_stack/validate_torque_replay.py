"""Recorded xArm torque -> acceleration -> joint increments -> recorded q.

The reference is measured position. No experimental torque predictor, inverse
dynamics torque generator, or differentiated reference acceleration is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from xarm_stack.validate_urdf import (DEFAULT_URDF, ROOT, DynamicsComparison,
                                     integrate_torque, metrics, read_episode)


def acceleration(comparison, q, dq, tau):
    p = comparison.pin
    mass = np.asarray(p.pin.crba(p.model, p.data, q), dtype=float).copy()
    mass = np.triu(mass)+np.triu(mass, 1).T
    np.linalg.cholesky(mass)
    bias = np.asarray(p.pin.nonLinearEffects(p.model, p.data, q, dq), dtype=float).copy()
    return np.linalg.solve(mass, tau-bias-comparison.friction.torque(dq))


def filter_torque(episode, cutoff_hz=0., max_gap_s=.05):
    """Optional torque-only zero-phase filtering; q/dq reference is untouched."""
    if not np.isfinite(cutoff_hz) or cutoff_hz < 0:
        raise ValueError('Torque cutoff must be finite and nonnegative')
    output = {**episode, 'tau': episode['tau'].copy(), 'valid': episode['valid'].copy()}
    if cutoff_hz == 0:
        return output
    from scipy.signal import butter, sosfiltfilt
    t, valid = episode['t'], episode['valid']
    start = None
    for i in range(len(t)+1):
        boundary = i == len(t) or not valid[i] or (i > 0 and t[i]-t[i-1] > max_gap_s)
        if start is not None and boundary:
            ts = t[start:i]
            output['valid'][start:i] = False
            if len(ts) >= 32:
                dt = float(np.median(np.diff(ts)))
                if cutoff_hz >= .5/dt:
                    raise ValueError('Torque cutoff must be below Nyquist frequency')
                grid = np.arange(ts[0], ts[-1]+dt*.01, dt)
                if len(grid) >= 32:
                    uniform = np.column_stack([np.interp(grid, ts, episode['tau'][start:i, j])
                                               for j in range(7)])
                    filtered = sosfiltfilt(butter(4, cutoff_hz, fs=1/dt, output='sos'), uniform, axis=0)
                    output['tau'][start:i] = np.column_stack([
                        np.interp(ts, grid, filtered[:, j]) for j in range(7)])
                    margin = 5/cutoff_hz
                    output['valid'][start:i] = (ts >= ts[0]+margin) & (ts <= ts[-1]-margin)
            start = None
        if i < len(t) and valid[i] and start is None:
            start = i
    return output


def one_step(comparison, episode, *, max_gap_s=.05, physics_dt_s=.001):
    t, q, dq, tau = (episode[k] for k in ('t', 'q', 'dq', 'tau'))
    dt = np.diff(t)
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('Maximum gap must be finite and positive')
    valid = episode['valid'][:-1] & episode['valid'][1:] & (dt > 0) & (dt <= max_gap_s)
    valid &= np.isfinite(dt)
    for a in (q[:-1], q[1:], dq[:-1], tau[:-1]):
        valid &= np.isfinite(a).all(axis=1)
    if not valid.any():
        raise ValueError('No valid adjacent feedback reports')
    actual_delta = q[1:]-q[:-1]
    velocity_delta = dq[:-1]*dt[:, None]
    predicted_delta, simulated_delta, acc = [np.full_like(actual_delta, np.nan) for _ in range(3)]
    mj, model = comparison.mujoco, comparison.model
    data = mj.MjData(model)
    for i in np.flatnonzero(valid):
        acc[i] = acceleration(comparison, q[i], dq[i], tau[i])
        predicted_delta[i] = dq[i]*dt[i]+.5*acc[i]*dt[i]**2
        # One-step prediction: initialize at this recorded state. Within the
        # interval MuJoCo advances freely, recomputing dynamics every substep.
        mj.mj_resetData(model, data)
        data.qpos[comparison.qadr], data.qvel[comparison.vadr] = q[i], dq[i]
        integrate_torque(comparison, data, tau[i], dt[i], physics_dt_s=physics_dt_s)
        simulated_delta[i] = data.qpos[comparison.qadr]-q[i]
    def scored(error):
        return dict(rad=metrics(error, valid), deg=metrics(np.rad2deg(error), valid))
    actual_rms = np.sqrt(np.mean(actual_delta[valid]**2, axis=0))
    def relative(error):
        rms = np.sqrt(np.mean(error[valid]**2, axis=0))
        return [float(x/y) if y > 1e-12 else None for x, y in zip(rms, actual_rms)]
    report = dict(samples=int(valid.sum()),
        dt_ms=dict(median=float(np.median(dt[valid])*1000),
                   p95=float(np.percentile(dt[valid], 95)*1000),
                   max=float(np.max(dt[valid])*1000)),
        actual_increment_rms_rad=actual_rms.tolist(),
        actual_increment_rms_deg=np.rad2deg(actual_rms).tolist(),
        taylor_error_to_actual_increment_rms_ratio=relative(predicted_delta-actual_delta),
        mujoco_error_to_actual_increment_rms_ratio=relative(simulated_delta-actual_delta),
        taylor_q_error=scored(predicted_delta-actual_delta),
        mujoco_q_error=scored(simulated_delta-actual_delta),
        velocity_only_baseline_error=scored(velocity_delta-actual_delta),
        interpretation='Each single step starts from measured q/dq; its q error equals '
            'its delta-q error. Small absolute one-step errors alone do not demonstrate '
            'accurate free-running dynamics; compare with the velocity-only baseline and multi-step rollouts.')
    traces = dict(t=t[1:], valid=valid, dt=dt, acceleration=acc,
        delta_q_recorded=actual_delta, delta_q_taylor=predicted_delta,
        delta_q_mujoco=simulated_delta, delta_q_velocity_only=velocity_delta,
        q_recorded=q[1:], q_taylor=q[:-1]+predicted_delta,
        q_mujoco=q[:-1]+simulated_delta, tau_applied=tau[:-1])
    return report, traces


def windows(comparison, episode, *, horizon_s=.1, stride_s=5., physics_dt_s=.001,
            max_gap_s=.05):
    if any(not np.isfinite(v) or v <= 0 for v in (horizon_s, stride_s, max_gap_s)):
        raise ValueError('Window, stride and maximum gap must be finite and positive')
    t, q, dq, tau, valid = (episode[k] for k in ('t', 'q', 'dq', 'tau', 'valid'))
    model, mj = comparison.model, comparison.mujoco
    data = mj.MjData(model)
    last_start = -np.inf
    errors, durations, start_times, previews = [], [], [], []
    for start in np.flatnonzero(valid):
        if t[start]-last_start < stride_s:
            continue
        end = int(np.searchsorted(t, t[start]+horizon_s))
        if end >= len(t) or not valid[start:end+1].all():
            continue
        intervals = np.diff(t[start:end+1])
        if np.any(intervals <= 0) or np.max(intervals) > max_gap_s:
            continue
        mj.mj_resetData(model, data)
        data.qpos[comparison.qadr], data.qvel[comparison.vadr] = q[start], dq[start]
        prediction = [q[start].copy()]
        for i in range(start, end):
            integrate_torque(comparison, data, tau[i], t[i+1]-t[i], physics_dt_s=physics_dt_s)
            prediction.append(data.qpos[comparison.qadr].copy())
        errors.append(prediction[-1]-q[end])
        durations.append(float(t[end]-t[start]))
        start_times.append(float(t[start]))
        last_start = t[start]
        if len(previews) < 1 or (len(previews) == 1 and t[start] >= 100):
            previews.append(dict(t=t[start:end+1].copy(), q_recorded=q[start:end+1].copy(),
                                 q_predicted=np.asarray(prediction)))
    error = np.asarray(errors).reshape(-1, 7)
    mask = np.ones(len(errors), bool)
    return dict(windows=len(errors), target_horizon_s=horizon_s,
        actual_horizon_s=durations, start_times_s=start_times,
        q_error_rad=metrics(error, mask), q_error_deg=metrics(np.rad2deg(error), mask),
        interpretation='Initialize only once at the start of each window, then replay '
            'recorded torque through all subsequent intervals without overwriting simulated q/dq.'), previews


def save_plots(traces, previews, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager

    fonts = [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
             Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')]
    for font in fonts:
        if font.is_file():
            fontManager.addfont(str(font))
            plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False
    keep = traces['valid'] & (traces['t'] >= 100) & (traces['t'] <= 110)
    if not keep.any():
        keep = traces['valid']
    t = traces['t'][keep]
    fig, axes = plt.subplots(7, 2, figsize=(15, 17), sharex=True)
    for j in range(7):
        axes[j, 0].plot(t, np.rad2deg(traces['q_recorded'][keep, j]),
                        label='数据里的 q', color='#1764ab', lw=1.1)
        axes[j, 0].plot(t, np.rad2deg(traces['q_taylor'][keep, j]),
                        label='记录力矩经 URDF 反算、积分得到的 q', color='#e87518', lw=.9)
        difference = np.rad2deg(traces['q_taylor'][keep, j]-traces['q_recorded'][keep, j])
        axes[j, 1].plot(t, difference, color='#b63838', label='反推 q − 数据 q', lw=.9)
        axes[j, 1].axhline(0., color='#555555', ls='--', lw=.8, label='两组 q 完全相同')
        for ax in axes[j]:
            ax.set_ylabel(f'J{j+1}（度）')
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]: ax.set_xlabel('反馈时间（秒）')
    fig.suptitle('直接比较同一时刻的 q：左列是两组 q，右列是它们的差\n'
                 '蓝线：数据 q；橙线：URDF 积分 q。单步比较，每次从前一条记录 q、dq 初始化。', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output/'single_step_q_comparison.png', dpi=130); plt.close(fig)
    # A focused, directly readable example rather than seven compressed rows.
    focused = traces['valid'] & (traces['t'] >= 100) & (traces['t'] <= 101)
    if not focused.any():
        first_time = traces['t'][np.flatnonzero(traces['valid'])[0]]
        focused = traces['valid'] & (traces['t'] <= first_time+1.)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ts = traces['t'][focused]
    data_q = np.rad2deg(traces['q_recorded'][focused, 6])
    urdf_q = np.rad2deg(traces['q_taylor'][focused, 6])
    axes[0].plot(ts, data_q, label='数据里的 J7 q', color='#1764ab', lw=1.7)
    axes[0].plot(ts, urdf_q, label='记录力矩经 URDF 反算、积分得到的 J7 q', color='#e87518', lw=1.2)
    axes[0].set_ylabel('关节角 q（度）'); axes[0].legend(fontsize=9)
    axes[1].plot(ts, urdf_q-data_q, color='#b63838', label='差值：URDF q − 数据 q')
    axes[1].axhline(0., color='#555555', ls='--', label='差值为 0 = 两组 q 相同')
    axes[1].set_ylabel('两组 q 的差（度）'); axes[1].set_xlabel('同一反馈时刻（秒）')
    axes[1].legend(fontsize=9)
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle('J7：你要看的两组 q 直接对照\n每个点是单步积分结果，横轴严格对齐到记录 q 的时刻')
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(output/'j7_direct_q_comparison.png', dpi=140); plt.close(fig)
    valid_rows = np.flatnonzero(traces['valid'])
    header = ['时间_秒']
    for j in range(1, 8):
        header.extend([f'J{j}_数据q_度', f'J{j}_URDF反推q_度', f'J{j}_差值_度'])
    with (output/'q_direct_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f); writer.writerow(header)
        for i in valid_rows:
            measured = np.rad2deg(traces['q_recorded'][i])
            reconstructed = np.rad2deg(traces['q_taylor'][i])
            row = [f'{traces["t"][i]:.6f}']
            for j in range(7):
                row.extend(f'{value:.6f}' for value in
                           (measured[j], reconstructed[j], reconstructed[j]-measured[j]))
            writer.writerow(row)
    if previews:
        preview = previews[-1]
        fig, axes = plt.subplots(7, 1, figsize=(11, 13), sharex=True)
        t = preview['t']-preview['t'][0]
        for j, ax in enumerate(axes):
            ax.plot(t, np.rad2deg(preview['q_recorded'][:, j]), 'o-', label='记录 q', markersize=3)
            ax.plot(t, np.rad2deg(preview['q_predicted'][:, j]), 'o-', label='连续积分 q', markersize=3)
            ax.set_ylabel(f'J{j+1}（度）'); ax.grid(alpha=.25)
        axes[0].legend(); axes[-1].set_xlabel('窗口内时间（秒）')
        fig.suptitle(f'约 100 ms 连续力矩积分：只在起点初始化\n起始反馈时间 {preview["t"][0]:.3f} s')
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(output/'continuous_window_q_comparison.png', dpi=130); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--arm', choices=('left', 'right'), default='right')
    parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/urdf_validation/recorded_torque_q')
    parser.add_argument('--torque-cutoff-hz', type=float, default=0.)
    parser.add_argument('--physics-dt-s', type=float, default=.001)
    parser.add_argument('--max-gap-s', type=float, default=.05)
    parser.add_argument('--window-s', type=float, default=.1)
    parser.add_argument('--window-stride-s', type=float, default=5.)
    parser.add_argument('--friction', choices=('none', 'urdf'), default='none')
    args = parser.parse_args(argv)
    raw = read_episode(args.source, args.arm)
    episode = filter_torque(raw, args.torque_cutoff_hz, args.max_gap_s)
    c = DynamicsComparison(args.urdf, dict(enabled=args.friction == 'urdf',
        source='urdf' if args.friction == 'urdf' else 'configured'))
    print('Replaying recorded torque; comparing q and delta-q, without reference-acceleration differentiation', flush=True)
    single, traces = one_step(c, episode, max_gap_s=args.max_gap_s, physics_dt_s=args.physics_dt_s)
    continuous, previews = windows(c, episode, horizon_s=args.window_s,
        stride_s=args.window_stride_s, physics_dt_s=args.physics_dt_s, max_gap_s=args.max_gap_s)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'q_predictions.npz', **traces)
    save_plots(traces, previews, args.output)
    report = dict(source=str(args.source.resolve()), arm=args.arm,
        urdf=str(args.urdf.resolve()), urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        tau_source='Recorded H5 joint torque, not learned prediction or URDF inverse-dynamics torque',
        signal_sources=raw['sources'], input_rows=raw['input_rows'], repeated_rows_removed=raw['repeated_rows'],
        reference='Recorded q; recorded dq initializes velocity. No differentiated reference ddq.',
        formula='a=solve(M(q), tau_recorded-h(q,dq)-friction); delta_q=dq*dt+0.5*a*dt^2; q_next=q+delta_q',
        torque_cutoff_hz=args.torque_cutoff_hz,
        filtering='None' if args.torque_cutoff_hz == 0 else 'Torque only: 4th-order zero-phase Butterworth, locally resampled; q/dq untouched; exclude filter edges.',
        physics_dt_s=args.physics_dt_s, max_gap_s=args.max_gap_s,
        friction=c.friction.metadata(), single_step=single, continuous_windows=continuous,
        assumptions=['Reported torque is treated as net driving joint effort; its physical convention has not been independently calibrated.',
                     'URDF inertias are not repaired; contact/limit constraints, armature, and implicit damping are disabled.',
                     'Each reported torque is held until the next feedback timestamp.',
                     'The selected model contains a fixed G2; gripper inertia and actuator friction/drive inertia remain uncalibrated.'])
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for label, values in (('Taylor single-step q', single['taylor_q_error']['deg']['rmse']),
                          ('MuJoCo single-step q', single['mujoco_q_error']['deg']['rmse']),
                          ('Velocity-only baseline q', single['velocity_only_baseline_error']['deg']['rmse'])):
        print(label+' RMSE J1..J7 (deg):', np.round(values, 5).tolist())
    if continuous['windows']:
        print('Continuous-window q RMSE J1..J7 (deg):', np.round(continuous['q_error_deg']['rmse'], 5).tolist())
    print('Report:', args.output/'summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
