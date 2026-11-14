"""Offline xArm URDF comparison using recorded q, dq and joint effort.

No robot connection or command interface is imported. Contact is unknown unless
the user explicitly supplies a known free-space interval. Model disagreement
is a diagnostic, not a binary certification of a physical URDF.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

from nero_collection.h5_schema import dataset_candidates
from xarm_stack.momentum import JointFrictionModel, PinocchioMomentumModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / 'gello_teleop/models/xarm7_dynamics_g2.urdf'


def read_episode(path, arm='right'):
    """Read lowdim signals, honoring validity and the feedback source clock."""
    import h5py

    with h5py.File(path, 'r') as f:
        group = f['teleop']
        arm_names = tuple(v.decode() if isinstance(v, bytes) else str(v)
                          for v in group.attrs.get('arm_names', [arm]))
        if arm not in arm_names:
            raise ValueError(f'{path}: arm {arm!r} was not recorded')
        n = len(group['timestamp_us'])
        values, sources = {}, {}
        valid = np.ones(n, dtype=bool)
        clocks = []
        for key, signal, flag, unit in (
                ('q', 'q_follower', 'q_follower_valid', 'rad'),
                ('dq', 'dq_follower', 'dq_valid_follower', 'rad/s'),
                ('tau', 'tau_follower', 'torque_valid_follower', 'Nm')):
            name = next((p for p in dataset_candidates(arm, signal) if p in group), None)
            if name is None:
                raise ValueError(f'{path}: missing {signal}')
            dataset = group[name]
            if dataset.attrs.get('placeholder', False) or dataset.attrs.get('source') == 'not_recorded':
                raise ValueError(f'{path}: {name} is an unrecorded placeholder')
            if 'unit' in dataset.attrs and dataset.attrs['unit'] != unit:
                raise ValueError(f'{name}: expected {unit}, got {dataset.attrs["unit"]}')
            array = np.asarray(dataset[:], dtype=float)
            if array.ndim != 2 or array.shape[0] != n:
                raise ValueError(f'{name}: expected an N x joints array')
            index = arm_names.index(arm) if array.shape[1] != 7 else 0
            array = array[:, index*7:(index+1)*7]
            if array.shape != (n, 7):
                raise ValueError(f'{name}: expected seven joint columns for {arm}')
            values[key] = array
            valid &= np.isfinite(array).all(axis=1)
            validity_path = dataset.attrs.get('validity_path')
            if isinstance(validity_path, bytes):
                validity_path = validity_path.decode()
            validity = f.get(validity_path) if validity_path else None
            if validity is None:
                flag_name = next((p for p in dataset_candidates(arm, flag) if p in group), None)
                if flag_name is None:
                    raise ValueError(f'{name}: missing effort/state validity flag')
                validity = group[flag_name]
            flags = np.asarray(validity[:]).reshape(n, -1)
            flag_column = arm_names.index(arm) if flags.shape[1] > 1 else 0
            valid &= flags[:, flag_column].astype(bool)
            source_path = dataset.attrs.get('source_timestamp_path')
            if isinstance(source_path, bytes):
                source_path = source_path.decode()
            source_clock = f.get(source_path) if source_path else None
            if source_clock is None:
                clock_name = next((p for p in dataset_candidates(arm, 'q_follower_timestamp_us')
                                   if p in group), None)
                source_clock = group[clock_name] if clock_name else group['timestamp_us']
            clock_array = np.asarray(source_clock[:], dtype=np.int64).reshape(n, -1)
            column = arm_names.index(arm) if clock_array.shape[1] > 1 else 0
            clocks.append(clock_array[:, column])
            sources[key] = {'dataset': name, 'clock': source_clock.name,
                            'validity': validity.name, 'unit': unit}
        if not all(np.array_equal(clocks[0], clock) for clock in clocks[1:]):
            raise ValueError('q, dq and torque have different source clocks; align them before validation')
        timestamps = clocks[0]
        if np.any(timestamps <= 0) or np.any(np.diff(timestamps) < 0):
            raise ValueError('Feedback timestamps must be positive and nondecreasing')
        # Keep the final row of a repeated report, retaining its validity.
        keep = np.r_[np.diff(timestamps) > 0, True]
        retained = timestamps[keep]
        if len(retained) < 5:
            raise ValueError('Too few distinct feedback samples')
        t = (retained - timestamps[0]).astype(float)*1e-6
        return dict(t=t, timestamp_us=retained, valid=valid[keep],
                    sources=sources, input_rows=n, repeated_rows=int((~keep).sum()),
                    **{key: value[keep] for key, value in values.items()})


def preprocess(episode, window_s=.11, max_gap_s=.05):
    """Local uniform resampling + matched zero-phase polynomial smoothing.

    Invalid samples and acquisition gaps split segments. Filter edges are not
    scored. Acceleration is obtained from dq, independent of torque and URDF.
    """
    from scipy.signal import savgol_filter

    if not np.isfinite(window_s) or window_s <= 0 or not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('Filter window and maximum gap must be finite and positive')
    t, good = episode['t'], episode['valid']
    if np.any(np.diff(t) <= 0):
        raise ValueError('Deduplicated timestamps must strictly increase')
    output = {k: np.full((len(t), 7), np.nan) for k in ('q', 'dq', 'tau', 'ddq', 'ddq_from_q')}
    scored = np.zeros(len(t), dtype=bool)
    starts, segments = None, []
    for i in range(len(t)+1):
        boundary = i == len(t) or not good[i] or (i > 0 and t[i]-t[i-1] > max_gap_s)
        if starts is not None and boundary:
            segments.append((starts, i))
            starts = None
        if i < len(t) and good[i] and starts is None:
            starts = i
    for start, end in segments:
        ts = t[start:end]
        if len(ts) < 7:
            continue
        dt = float(np.median(np.diff(ts)))
        grid = np.arange(ts[0], ts[-1]+dt*.01, dt)
        window = max(5, int(round(window_s/dt)) | 1)
        if len(grid) < window:
            continue
        for key in ('q', 'dq', 'tau'):
            uniform = np.column_stack([np.interp(grid, ts, episode[key][start:end, j])
                                       for j in range(7)])
            smooth = savgol_filter(uniform, window, 3, axis=0)
            output[key][start:end] = np.column_stack([np.interp(ts, grid, smooth[:, j])
                                                     for j in range(7)])
            if key == 'dq':
                acceleration = savgol_filter(uniform, window, 3, deriv=1, delta=dt, axis=0)
                output['ddq'][start:end] = np.column_stack([np.interp(ts, grid, acceleration[:, j])
                                                          for j in range(7)])
            elif key == 'q':
                acceleration = savgol_filter(uniform, window, 3, deriv=2, delta=dt, axis=0)
                output['ddq_from_q'][start:end] = np.column_stack([
                    np.interp(ts, grid, acceleration[:, j]) for j in range(7)])
        margin = (window//2)*dt
        scored[start:end] = (ts >= grid[0]+margin) & (ts <= grid[-1]-margin)
    if not scored.any():
        raise ValueError('No valid segments remain after filtering and edge exclusion')
    return dict(t=t.copy(), valid=scored, **output)


def metrics(error, mask):
    values = np.asarray(error)[mask]
    if not len(values):
        return {'samples': 0}
    return dict(samples=len(values), rmse=np.sqrt(np.mean(values**2, axis=0)).tolist(),
                bias=np.mean(values, axis=0).tolist(),
                p95_absolute=np.percentile(np.abs(values), 95, axis=0).tolist(),
                max_absolute=np.max(np.abs(values), axis=0).tolist())


def mit_torque(q, dq, q_target, dq_target, tau_feedforward, *, kp=0., kd=0.):
    """Torque-space MIT law, without clipping or added gravity compensation."""
    vectors = [np.asarray(v, dtype=float) for v in
               (q, dq, q_target, dq_target, tau_feedforward)]
    if any(v.shape != (7,) or not np.isfinite(v).all() for v in vectors):
        raise ValueError('MIT states, targets and feedforward must be finite seven-vectors')
    gains = []
    for name, value in (('kp', kp), ('kd', kd)):
        gain = np.asarray(value, dtype=float)
        if gain.ndim == 0:
            gain = np.full(7, float(gain))
        if gain.shape != (7,) or not np.isfinite(gain).all() or np.any(gain < 0):
            raise ValueError(f'MIT {name} must be a finite nonnegative scalar or seven-vector')
        gains.append(gain)
    q, dq, q_target, dq_target, feedforward = vectors
    return gains[0]*(q_target-q)+gains[1]*(dq_target-dq)+feedforward


class DynamicsComparison:
    def __init__(self, urdf, friction_config=None):
        import mujoco
        from inference.mujoco_visualization import load_kinematic_model

        self.urdf = Path(urdf).resolve()
        self.pin = PinocchioMomentumModel(self.urdf)
        self.friction = JointFrictionModel.from_config(friction_config, self.urdf)
        self.mujoco = mujoco
        self.model = load_kinematic_model(mujoco, self.urdf, balance_inertia=False)
        if self.model.nq != 7 or self.model.nv != 7:
            raise ValueError('MuJoCo model must have exactly seven arm DOFs')
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}')
               for i in range(1, 8)]
        if any(i < 0 for i in ids):
            raise ValueError('MuJoCo model is missing an arm joint')
        self.qadr = self.model.jnt_qposadr[ids]
        self.vadr = self.model.jnt_dofadr[ids]
        self.model.opt.gravity[:] = self.pin.model.gravity.linear
        # Explicitly match the two engines. URDF friction is an optional,
        # uncalibrated experiment, rather than silently enabled in MuJoCo.
        self.model.dof_damping[:] = 0
        self.model.dof_frictionloss[:] = 0
        self.model.dof_armature[:] = 0
        self.model.jnt_stiffness[:] = 0
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        self.data = mujoco.MjData(self.model)

    def evaluate(self, processed, *, roundtrip=False):
        arrays = {k: np.full_like(processed['q'], np.nan)
                  for k in ('ddq_pin', 'ddq_mujoco', 'tau_model', 'gravity', 'friction')}
        if roundtrip:
            arrays.update({k: np.full_like(processed['q'], np.nan) for k in
                           ('tau_mit_model', 'ddq_model_roundtrip', 'ddq_mujoco_model_roundtrip')})
        matrix_difference, bias_difference = [], []
        for i in np.flatnonzero(processed['valid']):
            q, dq, tau, ddq = (processed[k][i] for k in ('q', 'dq', 'tau', 'ddq'))
            p = self.pin
            mass = np.asarray(p.pin.crba(p.model, p.data, q)).copy()
            mass = np.triu(mass) + np.triu(mass, 1).T
            np.linalg.cholesky(mass)
            bias = np.asarray(p.pin.nonLinearEffects(p.model, p.data, q, dq)).copy()
            friction = self.friction.torque(dq)
            arrays['ddq_pin'][i] = np.linalg.solve(mass, tau-bias-friction)
            arrays['tau_model'][i] = p.rnea(q, dq, ddq)+friction
            arrays['gravity'][i] = p.gravity_torque(q)
            arrays['friction'][i] = friction
            d, m, mj = self.data, self.model, self.mujoco
            mj.mj_resetData(m, d)
            d.qpos[self.qadr], d.qvel[self.vadr] = q, dq
            d.qfrc_applied[self.vadr] = tau-friction
            mj.mj_forward(m, d)
            arrays['ddq_mujoco'][i] = d.qacc[self.vadr]
            mj_mass = np.empty((7, 7))
            # MuJoCo 3.4 changed mj_fullM's Python binding to take MjData.
            try:
                mj.mj_fullM(m, d, mj_mass)
            except TypeError:
                mj.mj_fullM(m, mj_mass, d.qM)
            mj_mass = mj_mass[np.ix_(self.vadr, self.vadr)]
            matrix_difference.append(np.max(np.abs(mj_mass-mass)))
            bias_difference.append(np.max(np.abs(d.qfrc_bias[self.vadr]-bias)))
            if roundtrip:
                # These are distinct from the measured-effort test above.
                # RNEA(model, q, dq, ddq) -> MIT feedforward -> forward(model)
                # is an algebraic self-consistency check for any valid model.
                torque = mit_torque(q, dq, q, dq, arrays['tau_model'][i], kp=0., kd=0.)
                arrays['tau_mit_model'][i] = torque
                arrays['ddq_model_roundtrip'][i] = np.linalg.solve(mass, torque-bias-friction)
                mj.mj_resetData(m, d)
                d.qpos[self.qadr], d.qvel[self.vadr] = q, dq
                d.qfrc_applied[self.vadr] = torque-friction
                mj.mj_forward(m, d)
                arrays['ddq_mujoco_model_roundtrip'][i] = d.qacc[self.vadr]
        return arrays, dict(max_mass_difference=float(max(matrix_difference)),
                            max_bias_difference=float(max(bias_difference)))


def analyze(path, comparison, *, arm='right', window_s=.11, max_gap_s=.05,
            free_space=None, no_contact=False, roundtrip=False):
    episode = read_episode(path, arm)
    processed = preprocess(episode, window_s, max_gap_s)
    arrays, engines = comparison.evaluate(processed, roundtrip=roundtrip)
    mask, t = processed['valid'], processed['t']
    torque_error = processed['tau']-arrays['tau_model']
    acceleration_error = arrays['ddq_pin']-processed['ddq']
    # Low speed is NOT evidence of no contact (a grasp can be static).
    quasi_static = mask & (np.max(np.abs(processed['dq']), axis=1) < .01)
    stationary = quasi_static & (np.max(np.abs(processed['ddq']), axis=1) < .1)
    moving = mask & (np.max(np.abs(processed['dq']), axis=1) >= .05)
    q_derivative = np.gradient(processed['q'], t, axis=0)
    velocity_check = mask & np.isfinite(q_derivative).all(axis=1)
    report = dict(source=str(Path(path).resolve()), arm=arm, input_rows=episode['input_rows'],
                  repeated_rows_removed=episode['repeated_rows'],
                  scored_rows=int(mask.sum()), duration_s=float(t[-1]-t[0]),
                  signal_sources=episode['sources'],
                  contact_status='user_confirmed_no_contact' if no_contact else 'unknown',
                  acceleration_error=metrics(acceleration_error, mask),
                  torque_error=metrics(torque_error, mask),
                  low_speed_torque_error=metrics(torque_error, quasi_static),
                  stationary_torque_error=metrics(torque_error, stationary),
                  stationary_gravity_error=metrics(processed['tau']-arrays['gravity'], stationary),
                  moving_torque_error=metrics(torque_error, moving),
                  measured_acceleration_rms=np.sqrt(np.mean(processed['ddq'][mask]**2, axis=0)).tolist(),
                  position_derived_acceleration_rms=np.sqrt(
                      np.mean(processed['ddq_from_q'][mask]**2, axis=0)).tolist(),
                  acceleration_from_q_consistency=metrics(processed['ddq_from_q']-processed['ddq'], mask),
                  measured_velocity_rms=np.sqrt(np.mean(processed['dq'][mask]**2, axis=0)).tolist(),
                  velocity_consistency=metrics(q_derivative-processed['dq'], velocity_check),
                  engine_acceleration_difference=metrics(arrays['ddq_mujoco']-arrays['ddq_pin'], mask),
                  engine_terms=engines)
    if roundtrip:
        report['model_tau_roundtrip'] = dict(
            tau_source='RNEA of the same selected URDF using velocity-derived acceleration',
            mit=dict(kp=0., kd=0., feedforward='tau_model', clipping=False,
                     additional_gravity=False),
            pinocchio_acceleration_error=metrics(arrays['ddq_model_roundtrip']-processed['ddq'], mask),
            mujoco_acceleration_error=metrics(arrays['ddq_mujoco_model_roundtrip']-processed['ddq'], mask),
            interpretation='Tests mathematical and implementation consistency only. '
                'A physically incorrect URDF can pass this test because its inverse '
                'and forward dynamics cancel. No claim of physical accuracy is made.')
    if free_space is not None:
        start, end = free_space
        free = mask & (t >= start) & (t <= end)
        report['user_declared_free_space'] = dict(interval_s=[start, end],
            acceleration_error=metrics(acceleration_error, free),
            torque_error=metrics(torque_error, free))
    return report, dict(**processed, **arrays, q_raw=episode['q'], dq_raw=episode['dq'],
                        tau_raw=episode['tau'], timestamp_us=episode['timestamp_us'])


def save_plot(traces, path, title, *, no_contact=False, tau_source='recorded torque'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 2, figsize=(15, 17), sharex=True)
    t = traces['t']
    for j in range(7):
        axes[j, 0].plot(t, traces['ddq'][:, j], label='From recorded velocity', lw=1)
        axes[j, 0].plot(t, traces['ddq_pin'][:, j], label='URDF + '+tau_source, lw=.8)
        axes[j, 0].plot(t, traces['ddq_mujoco'][:, j], '--', label='MuJoCo + '+tau_source, lw=.8)
        axes[j, 0].set_ylabel(f'J{j+1} rad/s²')
        axes[j, 1].plot(t, traces['tau'][:, j], label=tau_source.capitalize(), lw=1)
        axes[j, 1].plot(t, traces['tau_model'][:, j], label='URDF inverse dynamics', lw=.8)
        axes[j, 1].set_ylabel(f'J{j+1} Nm')
        for ax in axes[j]:
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Feedback time (s)')
    note = ('User-confirmed no contact; residual includes friction and payload/model errors'
            if no_contact else 'Contact unknown; residual includes contact, friction and payload/model errors')
    fig.suptitle(title+'\n'+note, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roundtrip_plot(traces, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 2, figsize=(15, 17), sharex=True)
    t = traces['t']
    for j in range(7):
        axes[j, 0].plot(t, traces['ddq'][:, j], label='From recorded velocity', lw=1)
        axes[j, 0].plot(t, traces['ddq_model_roundtrip'][:, j], '--',
                       label='URDF tau -> MIT -> Pinocchio acceleration', lw=.8)
        axes[j, 0].plot(t, traces['ddq_mujoco_model_roundtrip'][:, j], ':',
                       label='URDF tau -> MIT -> MuJoCo acceleration', lw=.8)
        for key, label in (('ddq_model_roundtrip', 'Pinocchio residual'),
                           ('ddq_mujoco_model_roundtrip', 'MuJoCo residual')):
            axes[j, 1].plot(t, traces[key][:, j]-traces['ddq'][:, j], label=label, lw=.7)
        for ax in axes[j]:
            ax.set_ylabel(f'J{j+1} rad/s²')
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Feedback time (s)')
    fig.suptitle(title+'\nSame-model inverse -> MIT -> forward: self-consistency only, '
                 'not physical URDF accuracy', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def integrate_torque(comparison, data, torque, duration_s, *, physics_dt_s=.001):
    """Hold recorded joint effort over one interval; never overwrite q/dq."""
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError('Integration duration must be finite and positive')
    if not np.isfinite(physics_dt_s) or physics_dt_s <= 0:
        raise ValueError('Physics timestep must be finite and positive')
    mj, model = comparison.mujoco, comparison.model
    steps = max(1, int(np.ceil(duration_s/physics_dt_s)))
    model.opt.timestep = duration_s/steps
    for _ in range(steps):
        velocity = data.qvel[comparison.vadr]
        data.qfrc_applied[comparison.vadr] = torque-comparison.friction.torque(velocity)
        mj.mj_step(model, data)


def render_snapshot(comparison, traces, path, start_s=1., rollout_s=.1, *, no_contact=False):
    """Save actual pose and a short independent torque-driven prediction."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mj, model = comparison.mujoco, comparison.model
    t, valid = traces['t'], traces['valid']
    candidates = np.flatnonzero(valid & (t >= start_s))
    selected = None
    for start in candidates:
        end = int(np.searchsorted(t, t[start]+rollout_s))
        if end < len(t) and np.all(valid[start:end+1]) and np.max(np.diff(t[start:end+1])) <= .05:
            selected = start, end
            break
    if selected is None:
        raise ValueError('No valid continuous rollout at/after --render-time')
    start, end = selected
    predicted, observed = mj.MjData(model), mj.MjData(model)
    predicted.qpos[comparison.qadr], predicted.qvel[comparison.vadr] = traces['q'][start], traces['dq'][start]
    for i in range(start, end):
        integrate_torque(comparison, predicted, traces['tau'][i], t[i+1]-t[i])
    observed.qpos[comparison.qadr], observed.qvel[comparison.vadr] = traces['q'][end], traces['dq'][end]
    camera = mj.MjvCamera()
    camera.lookat[:] = [0., 0., .4]
    camera.distance, camera.azimuth, camera.elevation = 1.8, 130., -20.
    frames = []
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 700)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 600)
    with mj.Renderer(model, height=600, width=700) as renderer:
        for data in (observed, predicted):
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, frame, label in zip(axes, frames, ('Recorded pose', 'URDF + recorded torque prediction')):
        ax.imshow(frame)
        ax.set_title(label)
        ax.axis('off')
    note = ('User-confirmed no contact; exact payload/friction remain uncalibrated'
            if no_contact else 'Contact force and exact payload unknown; this is a diagnostic comparison')
    fig.suptitle(f'Time {t[end]:.3f} s | independent rollout {t[end]-t[start]:.3f} s\n'+note)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(start_s=float(t[start]), end_s=float(t[end]),
                q_observed=traces['q'][end].tolist(),
                q_predicted=predicted.qpos[comparison.qadr].tolist(),
                q_error_rad=(predicted.qpos[comparison.qadr]-traces['q'][end]).tolist())


@contextmanager
def managed_viewer(model, data):
    """Wait for this viewer's render threads before interpreter shutdown."""
    import mujoco.viewer

    previous_threads = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data)
    owned_threads = [thread for thread in threading.enumerate()
                     if thread not in previous_threads]
    try:
        yield viewer
    finally:
        # Handle.close() requests asynchronous shutdown. Leaving its daemon
        # render thread alive during interpreter teardown can crash GLFW.
        viewer.close()
        for thread in owned_threads:
            thread.join(timeout=5.)


def play_viewer(comparison, traces, rollout_s=.1):
    """Display measured replay and a short independent torque rollout ghost.

    The prediction is reset every rollout_s, never during its integration
    window. This limits divergence on recordings with unknown contact forces.
    """
    import mujoco.viewer

    mj, m = comparison.mujoco, comparison.model
    observed, predicted = mj.MjData(m), mj.MjData(m)
    scene = mj.MjvScene(m, maxgeom=2000)
    option = mj.MjvOption()
    valid = np.flatnonzero(traces['valid'])
    if not len(valid):
        raise ValueError('No samples to display')
    m.opt.timestep = .001
    print('MuJoCo: solid = measured replay; translucent blue = recorded-torque rollout (reset every %.3f s)' % rollout_s)
    with managed_viewer(m, observed) as viewer:
        anchor = None
        wall_start = time.monotonic()
        for i in valid:
            if not viewer.is_running():
                break
            t = traces['t'][i]
            if anchor is None or t-anchor >= rollout_s or t-previous_t > .05:
                mj.mj_resetData(m, predicted)
                predicted.qpos[comparison.qadr] = traces['q'][i]
                predicted.qvel[comparison.vadr] = traces['dq'][i]
                anchor = t
            else:
                integrate_torque(comparison, predicted, previous_tau, t-previous_t)
            previous_t, previous_tau = t, traces['tau'][i].copy()
            observed.qpos[comparison.qadr] = traces['q'][i]
            observed.qvel[comparison.vadr] = traces['dq'][i]
            mj.mj_forward(m, observed)
            mj.mj_forward(m, predicted)
            with viewer.lock():
                mj.mjv_updateScene(m, predicted, option, None, viewer.cam,
                                   mj.mjtCatBit.mjCAT_ALL, scene)
                viewer.user_scn.ngeom = 0
                for geom in scene.geoms[:scene.ngeom]:
                    if geom.objtype != mj.mjtObj.mjOBJ_GEOM:
                        continue
                    j = viewer.user_scn.ngeom
                    if j >= viewer.user_scn.maxgeom:
                        break
                    target = viewer.user_scn.geoms[j]
                    mj.mjv_initGeom(target, geom.type, geom.size, geom.pos, geom.mat.ravel(),
                                   np.array([.15, .55, 1., .35], dtype=np.float32))
                    target.dataid, target.objtype, target.objid = geom.dataid, geom.objtype, geom.objid
                    viewer.user_scn.ngeom += 1
            viewer.sync()
            delay = wall_start+(t-traces['t'][valid[0]])-time.monotonic()
            if delay > 0:
                time.sleep(min(delay, .05))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='H5 file or episode directory')
    parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    parser.add_argument('--arm', choices=('left', 'right'), default='right')
    parser.add_argument('--no-contact', action='store_true',
                        help='User confirms the entire input recording has no external contact')
    parser.add_argument('--tool-configuration', choices=('unknown', 'g2_only'), default='unknown',
                        help='Recorded user confirmation; does not change the selected URDF')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/urdf_validation')
    parser.add_argument('--limit', type=int, default=1, help='Directory: first N files; 0 means all')
    parser.add_argument('--window-s', type=float, default=.11)
    parser.add_argument('--max-gap-s', type=float, default=.05)
    parser.add_argument('--friction', choices=('none', 'urdf'), default='none')
    parser.add_argument('--roundtrip', action='store_true',
                        help='Also compare velocity-derived ddq with URDF tau -> MIT -> forward dynamics; '
                             'tests implementation consistency, not physical accuracy')
    parser.add_argument('--free-space', nargs=2, type=float, metavar=('START_S', 'END_S'),
                        help='User-identified no-contact interval; only for a single H5 file')
    parser.add_argument('--viewer', action='store_true', help='MuJoCo replay with short torque rollout ghost')
    parser.add_argument('--rollout-s', type=float, default=.1)
    parser.add_argument('--render', action='store_true', help='Save a MuJoCo pose comparison for the first episode')
    parser.add_argument('--render-time', type=float, default=1.)
    args = parser.parse_args(argv)
    if args.limit < 0 or not np.isfinite(args.rollout_s) or args.rollout_s <= 0:
        parser.error('limit must be nonnegative; rollout-s must be finite and positive')
    if not np.isfinite(args.render_time) or args.render_time < 0:
        parser.error('render-time must be finite and nonnegative')
    if args.source.is_dir():
        if args.free_space is not None:
            parser.error('--free-space needs a specific H5 file')
        files = sorted(args.source.glob('*.h5'))
        files = files[:args.limit] if args.limit else files
    else:
        files = [args.source]
    if not files or any(not p.is_file() for p in files):
        parser.error('No input H5 files found')
    if args.viewer and len(files) != 1:
        parser.error('--viewer needs one episode (--limit 1 or a specific file)')
    if args.free_space is not None:
        a, b = args.free_space
        if not np.isfinite([a, b]).all() or a < 0 or b <= a:
            parser.error('free-space requires finite 0 <= START_S < END_S')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'enabled': args.friction == 'urdf', 'source': 'urdf' if args.friction == 'urdf' else 'configured'}
    comparison = DynamicsComparison(args.urdf, config)
    reports, last_traces = [], None
    for index, path in enumerate(files):
        report, traces = analyze(path, comparison, window_s=args.window_s,
                                 arm=args.arm, max_gap_s=args.max_gap_s,
                                 free_space=args.free_space, no_contact=args.no_contact,
                                 roundtrip=args.roundtrip)
        reports.append(report)
        if index == 0:
            np.savez_compressed(args.output/'first_episode.npz', **traces)
            save_plot(traces, args.output/'first_episode.png', f'{args.arm}: {path.name}',
                      no_contact=args.no_contact)
            if args.roundtrip:
                save_roundtrip_plot(traces, args.output/'model_tau_roundtrip.png',
                                    f'{args.arm}: {path.name}')
            if args.render:
                report['pose_snapshot'] = render_snapshot(comparison, traces,
                    args.output/'first_episode_pose.png', args.render_time, args.rollout_s,
                    no_contact=args.no_contact)
        if (index+1) % 10 == 0:
            print(f'Analyzed {index+1}/{len(files)} episodes', flush=True)
        last_traces = traces
    scored = np.array([r['scored_rows'] for r in reports])
    aggregate = {}
    for key in ('acceleration_error', 'torque_error', 'engine_acceleration_difference'):
        rmses = np.array([r[key]['rmse'] for r in reports])
        biases = np.array([r[key]['bias'] for r in reports])
        aggregate[key] = dict(samples=int(scored.sum()),
            rmse=np.sqrt(np.average(rmses**2, weights=scored, axis=0)).tolist(),
            bias=np.average(biases, weights=scored, axis=0).tolist())
    summary = dict(arm=args.arm, tool_configuration=args.tool_configuration,
        urdf=str(comparison.urdf), urdf_sha256=hashlib.sha256(comparison.urdf.read_bytes()).hexdigest(),
        friction=comparison.friction.metadata(), preprocessing=dict(window_s=args.window_s, max_gap_s=args.max_gap_s),
        simulation=dict(gravity_m_s2=comparison.model.opt.gravity.tolist(), constraints=False,
                        inertia_balancing=False, mesh_derived_inertia=False),
        contact_status='user_confirmed_no_contact' if args.no_contact else 'unknown; no inferred contact mask used',
        limitations=['Reported joint torque is not independently calibrated.',
                     ('G2 only, no extra tool: user confirmed. Gripper inertia remains uncalibrated.'
                      if args.tool_configuration == 'g2_only' else 'Tool configuration was not specified.'),
                     ('No external contact; G2 CoM and inertia were not independently measured.'
                      if args.no_contact and args.tool_configuration == 'g2_only'
                      else 'User confirms no external contact; exact tool payload was not supplied.'
                      if args.no_contact else 'No independent contact force or exact tool payload was supplied.'),
                     'Engine agreement checks model conversion, not physical accuracy.',
                     'MIT host torque is not substituted for measured joint effort.'],
        aggregate=aggregate, episodes=reports)
    output = args.output/'summary.json'
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(f'Analyzed {len(reports)} episodes; {scored.sum()} scored samples; report: {output}')
    print('Torque RMSE J1..J7 (Nm):',np.round(aggregate['torque_error']['rmse'], 4).tolist())
    print('Acceleration RMSE J1..J7 (rad/s²):',np.round(aggregate['acceleration_error']['rmse'], 4).tolist())
    if args.roundtrip:
        for engine, key in (('Pinocchio', 'pinocchio_acceleration_error'),
                            ('MuJoCo', 'mujoco_acceleration_error')):
            rmses = np.array([r['model_tau_roundtrip'][key]['rmse'] for r in reports])
            values = np.sqrt(np.average(rmses**2, weights=scored, axis=0))
            print(f'Model-tau roundtrip {engine} ddq RMSE J1..J7 (rad/s²):',
                  '['+', '.join(f'{v:.3e}' for v in values)+']')
        print('Model-tau roundtrip checks self-consistency only; physical URDF accuracy is not established.')
    if args.viewer:
        play_viewer(comparison, last_traces, args.rollout_s)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
