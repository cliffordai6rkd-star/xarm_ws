#!/usr/bin/env python3
"""GELLO excitation planning, telemetry collection and identifiable-parameter fitting.

No hardware is accessed unless inspect/collect is explicitly selected. collect is
read-only unless --execute is supplied. Raw XL330 supply current is NOT torque.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gello_teleop.build_gello_urdf import vector


def initial_config(leader=None):
    leader = leader or {}
    n = len(leader.get('joint_ids', list(range(1, 8))))
    signs = leader.get('joint_signs', [1] * n)
    offsets = leader.get('joint_offsets')
    reference = leader.get('leader_reference_q')
    center = (np.asarray(signs) * (np.asarray(reference) - np.asarray(offsets))).tolist() if (
        offsets is not None and reference is not None) else None
    return {'port': leader.get('port'), 'baudrate': leader.get('baudrate', 57600),
            'joint_ids': leader.get('joint_ids', list(range(1, 8))),
            'encoder_signs': signs, 'encoder_offsets_rad': offsets,
            'joint_coordinates_verified': False,
            'note': 'Encoder offsets must define the SAME q=0 as the measured leader URDF. '
                    'Supply measured mechanical limits, including cable/elastic restrictions. '
                    'Profile supports velocity-based drive mode, position operating mode 3.',
            'center_rad': center, 'lower_rad': None, 'upper_rad': None,
            'amplitude_rad': [float(np.deg2rad(3))] * n,
            'duration_s': 60.0, 'sample_hz': 8.0, 'max_velocity_rad_s': 0.15,
            'max_acceleration_rad_s2': 0.5, 'max_tracking_error_rad': float(np.deg2rad(5)),
            'start_tolerance_rad': float(np.deg2rad(1)), 'max_temperature_c': 50,
            'min_voltage_v': 3.7, 'max_voltage_v': 6.0,
            'max_current_a': [0.6] * n,
            'pwm_ceiling': leader.get('hold_pwm_by_joint') or [300] * n,
            'watchdog_s': 0.3, 'max_sample_gap_s': 0.15,
            'elastic_elements': 'unknown', 'payload_description': 'unknown'}


class ExcitationPlan:
    """Windowed multisine, with analytic envelope/velocity/acceleration bounds."""

    def __init__(self, config):
        self.config = config
        self.ids = list(config['joint_ids'])
        self.n = len(self.ids)
        if not self.n or len(set(self.ids)) != self.n or any(
                type(i) is not int or not 0 <= i <= 252 for i in self.ids):
            raise ValueError('joint_ids must be unique Dynamixel IDs')
        self.center = vector(config['center_rad'], 'center_rad', self.n)
        self.lower = vector(config['lower_rad'], 'lower_rad', self.n)
        self.upper = vector(config['upper_rad'], 'upper_rad', self.n)
        self.amplitude = vector(config['amplitude_rad'], 'amplitude_rad', self.n)
        self.signs = vector(config['encoder_signs'], 'encoder_signs', self.n)
        self.offsets = vector(config['encoder_offsets_rad'], 'encoder_offsets_rad', self.n)
        if np.any(~np.isin(self.signs, [-1, 1])):
            raise ValueError('encoder_signs must be +1/-1')
        self.duration = float(config['duration_s'])
        self.hz = float(config['sample_hz'])
        self.max_velocity = float(config['max_velocity_rad_s'])
        self.max_acceleration = float(config['max_acceleration_rad_s2'])
        for name, value in [('duration_s', self.duration), ('sample_hz', self.hz),
                            ('max_velocity_rad_s', self.max_velocity),
                            ('max_acceleration_rad_s2', self.max_acceleration)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if np.any(self.lower >= self.upper) or np.any(self.amplitude < 0) or not np.any(self.amplitude > 0):
            raise ValueError('Need ordered mechanical limits and nonzero, nonnegative amplitudes')
        if np.any(self.center - self.amplitude < self.lower) or np.any(self.center + self.amplitude > self.upper):
            raise ValueError('The entire excitation envelope must lie within measured mechanical limits')
        # Different frequencies per joint excite coupling; they are integer
        # multiples of 1/duration. sin^4 window gives q/dq/ddq=rest at both ends.
        self.omega = 2 * np.pi / self.duration * (np.arange(self.n)[:, None] * 2 + [1, 2, 3])
        if self.omega.max() / (2 * np.pi) >= self.hz / 10:
            raise ValueError('Use at least ten samples per highest excitation cycle')
        self.weights = self.amplitude[:, None] * np.array([0.5, 0.3, 0.2])
        self.phase = np.arange(self.n)[:, None] * 0.37 + np.array([0, 0.7, 1.2])
        w = np.pi / self.duration
        first = np.sum(np.abs(self.weights * self.omega), axis=1)
        second = np.sum(np.abs(self.weights * self.omega**2), axis=1)
        self.velocity_bound = 4 * w * self.amplitude + first
        self.acceleration_bound = 16 * w*w * self.amplitude + 8*w * first + second
        if np.any(self.velocity_bound > self.max_velocity):
            raise ValueError('Excitation exceeds velocity bound; lengthen duration or reduce amplitude')
        if np.any(self.acceleration_bound > self.max_acceleration):
            raise ValueError('Excitation exceeds acceleration bound; lengthen duration or reduce amplitude')
        raw_endpoints = self.offsets + np.stack([self.center-self.amplitude,
                                               self.center+self.amplitude]) / self.signs
        ticks = raw_endpoints * 2048 / np.pi
        if np.any(ticks < 0) or np.any(ticks > 4095):
            raise ValueError('Trajectory crosses the single-turn Goal Position range; wrapping is forbidden')

    def evaluate(self, time_s):
        t = np.asarray(time_s, dtype=float)
        if not np.isfinite(t).all() or np.any(t < 0) or np.any(t > self.duration):
            raise ValueError('Excitation time is outside [0, duration]')
        s, c = np.sin(np.pi*t/self.duration), np.cos(np.pi*t/self.duration)
        w = np.pi / self.duration
        envelope, de, dde = s**4, 4*w*s**3*c, 4*w*w*(3*s*s*c*c - s**4)
        angles = t[..., None, None]*self.omega + self.phase
        wave = np.sum(self.weights*np.sin(angles), axis=-1)
        dw = np.sum(self.weights*self.omega*np.cos(angles), axis=-1)
        ddw = -np.sum(self.weights*self.omega**2*np.sin(angles), axis=-1)
        return (self.center + envelope[..., None]*wave,
                de[..., None]*wave + envelope[..., None]*dw,
                dde[..., None]*wave + 2*de[..., None]*dw + envelope[..., None]*ddw)

    def summary(self):
        return {'joint_ids': self.ids, 'duration_s': self.duration, 'sample_hz': self.hz,
                'envelope_lower_rad': (self.center-self.amplitude).tolist(),
                'envelope_upper_rad': (self.center+self.amplitude).tolist(),
                'velocity_bound_rad_s': self.velocity_bound.tolist(),
                'acceleration_bound_rad_s2': self.acceleration_bound.tolist(),
                'collision_checked': False,
                'note': 'Per-joint bounds do not prove self-collision or obstacle clearance.'}


def smooth_derivatives(time_s, q, window=11):
    """Local cubic fit using real timestamps, with endpoint samples discarded."""
    time_s = np.asarray(time_s, dtype=float)
    q = np.asarray(q, dtype=float)
    if time_s.ndim != 1 or q.ndim != 2 or len(time_s) != len(q) or not np.isfinite(q).all():
        raise ValueError('Expected finite time[N] and measured q[N,dof]')
    dt = np.diff(time_s)
    if not np.isfinite(time_s).all() or len(dt) == 0 or np.any(dt <= 0):
        raise ValueError('Measured timestamps must strictly increase')
    if np.max(dt) > 2.5*np.median(dt):
        raise ValueError('Telemetry contains large gaps; collect a continuous session')
    if window < 5 or window % 2 != 1 or len(q) < 3*window:
        raise ValueError('Use an odd smoothing window >=5 and at least three windows of data')
    half = window // 2
    output = []
    for i in range(half, len(q)-half):
        local_t = time_s[i-half:i+half+1] - time_s[i]
        scale = np.max(np.abs(local_t))
        design = np.vander(local_t/scale, 4, increasing=True)
        coefficients = np.linalg.lstsq(design, q[i-half:i+half+1], rcond=None)[0]
        output.append((coefficients[0], coefficients[1]/scale, 2*coefficients[2]/scale**2))
    values = np.asarray(output)
    return np.arange(half, len(q)-half), values[:, 0], values[:, 1], values[:, 2]


def fit_observable(design, torque, split, rtol=1e-6):
    """Fit only the observable SVD subspace; never interpret minimum-norm masses."""
    design, torque = np.asarray(design), np.asarray(torque)
    if design.ndim != 3 or torque.shape != design.shape[:2] or not (
            np.isfinite(design).all() and np.isfinite(torque).all()):
        raise ValueError('Expected finite regressor[N,dof,p] and torque[N,dof]')
    if not 1 < split < len(design)-1 or not 0 < rtol < 1:
        raise ValueError('Need disjoint training/validation blocks and 0<rtol<1')
    train = design[:split].reshape(-1, design.shape[-1])
    target = torque[:split].reshape(-1)
    scaling = np.linalg.norm(train, axis=0)
    scaling[scaling < 1e-14] = 1.0
    u, singular, vt = np.linalg.svd(train/scaling, full_matrices=False)
    if not len(singular) or singular[0] <= 0:
        raise ValueError('No excitation in the torque regressor')
    rank = int(np.sum(singular > singular[0]*rtol))
    basis = vt[:rank].T
    coefficients = (u[:, :rank].T@target)/singular[:rank]
    representative = (basis@coefficients)/scaling
    prediction = np.einsum('nij,j->ni', design, representative)
    def rms(values):
        return np.sqrt(np.mean(values**2, axis=0)).tolist()
    report = {'rank': rank, 'parameter_count': design.shape[-1],
              'nullity': design.shape[-1]-rank, 'svd_relative_cutoff': rtol,
              'retained_condition_number': float(singular[0]/singular[rank-1]),
              'training_samples': split, 'validation_samples': len(design)-split,
              'training_rmse_nm': rms(prediction[:split]-torque[:split]),
              'validation_rmse_nm': rms(prediction[split:]-torque[split:]),
              'validation_torque_rms_nm': rms(torque[split:]),
              'individual_link_parameters_confirmed': False,
              'note': 'Observable parameter combinations only. A small residual does not '
                      'uniquely determine link mass/COM/inertia or validate torque calibration.'}
    return report, {'observable_basis_scaled': basis, 'observable_coefficients': coefficients,
                    'column_scaling': scaling, 'singular_values': singular,
                    'minimum_norm_representative_NOT_link_parameters': representative,
                    'predicted_torque_nm': prediction}


def fit_dataset(urdf_path, data_path, torque_path, window=11, elastic_model=False):
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(Path(urdf_path).resolve()))
    if 'UNVERIFIED' in model.name:
        raise ValueError('Unverified geometry draft: confirm assembly and encoder coordinates before fitting dynamics')
    if model.nq != model.nv or model.nv == 0 or not model.name.startswith('gello_'):
        raise ValueError('Need a fixed-base revolute-joint model, not the static mesh preview')
    with np.load(data_path, allow_pickle=False) as file:
        t, measured_q = file['time_s'].copy(), file['q_rad'].copy()
        metadata = json.loads(str(file['metadata_json']))
        if t.ndim != 1 or measured_q.shape != (len(t), model.nv) or not (
                np.isfinite(t).all() and np.isfinite(measured_q).all()):
            raise ValueError('Expected finite time[N] and measured q[N,model.nv]')
        q_times = ((file['receive_ns']-int(metadata['session_start_ns']))*1e-9
                   if 'receive_ns' in file and 'session_start_ns' in metadata else
                   np.broadcast_to(t[:, None], measured_q.shape).copy())
    if metadata.get('session_status') != 'complete':
        raise ValueError('Session did not complete; do not fit an aborted trajectory as valid excitation')
    if metadata.get('joint_coordinates_verified') is not True:
        raise ValueError('Confirm telemetry q=0/sign/order matches the measured leader URDF')
    if tuple(metadata.get('joint_names', ())) != tuple(model.names[1:]):
        raise ValueError('Telemetry joint order/names do not match the URDF')
    with np.load(torque_path, allow_pickle=False) as file:
        torque = file['tau_nm'].copy()
        torque_meta = json.loads(str(file['metadata_json']))
        if not np.array_equal(file['time_s'], t):
            raise ValueError('Torque timestamps must be synchronized to the telemetry samples')
        torque_times = (file['sample_time_s'].copy() if 'sample_time_s' in file else
                        np.broadcast_to(t[:, None], torque.shape).copy())
    if torque.shape != measured_q.shape or torque_meta.get('calibrated_output_torque') is not True:
        raise ValueError('Supply calibrated, signed output torque in Nm; supply current/PWM are insufficient')
    if not torque_meta.get('source'):
        raise ValueError('Torque metadata must describe the sensor/calibration source')
    if torque_meta.get('joint_names') != list(model.names[1:]):
        raise ValueError('Torque order/sign convention must match the URDF joint coordinates')
    if metadata.get('elastic_elements', 'unknown') != 'none' and not elastic_model:
        raise ValueError('Elastic elements exist/are unknown: model their contribution with --fit-elastic '
                         'or collect with them removed and document elastic_elements: none')
    for times in [q_times, torque_times]:
        if times.shape != measured_q.shape or not np.isfinite(times).all() or np.any(np.diff(times, axis=0) <= 0):
            raise ValueError('Each joint needs finite, strictly increasing measurement timestamps')
    # Align the sequential servo responses to a common host timeline. Reception
    # time estimates acquisition time; keep this approximation in the report.
    start = max(float(q_times[0].max()), float(torque_times[0].max()))
    end = min(float(q_times[-1].min()), float(torque_times[-1].min()))
    shared_t = t[(t >= start) & (t <= end)]
    aligned_q = np.column_stack([np.interp(shared_t, q_times[:, j], measured_q[:, j])
                                for j in range(model.nv)])
    aligned_torque = np.column_stack([np.interp(shared_t, torque_times[:, j], torque[:, j])
                                     for j in range(model.nv)])
    t, measured_q, torque = shared_t, aligned_q, aligned_torque
    indices, q, dq, ddq = smooth_derivatives(t, measured_q, window)
    if q.shape[1] != model.nq:
        raise ValueError('Telemetry degrees of freedom do not match the URDF')
    if np.any(q < model.lowerPositionLimit) or np.any(q > model.upperPositionLimit):
        raise ValueError('Measured positions exceed the model limits')
    data = model.createData()
    regressors = []
    for qi, vi, ai in zip(q, dq, ddq):
        inertial = np.asarray(pin.computeJointTorqueRegressor(model, data, qi, vi, ai)).reshape(model.nv, -1).copy()
        # Motor friction, bias and reflected rotor inertia are not link inertias.
        nuisance = [np.diag(vi), np.diag(np.tanh(vi/0.02)), np.eye(model.nv), np.diag(ai)]
        if elastic_model:
            nuisance.extend([np.diag(qi), np.diag(np.sin(qi)), np.diag(np.cos(qi))])
        regressors.append(np.concatenate([inertial] + nuisance, axis=1))
    report, arrays = fit_observable(np.asarray(regressors), torque[indices], int(len(q)*0.7))
    report.update({'model_sha256': hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
                   'data_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                   'torque_sha256': hashlib.sha256(Path(torque_path).read_bytes()).hexdigest(),
                   'joint_names': list(model.names[1:]), 'derivative_window_samples': window,
                   'validation_method': 'held-out final 30 percent contiguous block of measured samples',
                   'torque_source': torque_meta['source'], 'elastic_terms_fitted': elastic_model,
                   'joint_timing_alignment': 'Interpolated at common host reception timestamps; '
                                             'servo acquisition/transport delay is not calibrated.',
                   'maximum_response_skew_s': metadata.get('maximum_response_skew_s'),
                   'position_span_rad': np.ptp(q, axis=0).tolist(),
                   'velocity_rms_rad_s': np.sqrt(np.mean(dq**2, axis=0)).tolist(),
                   'acceleration_rms_rad_s2': np.sqrt(np.mean(ddq**2, axis=0)).tolist(),
                   'inertial_parameter_columns': 10*model.nv,
                   'nuisance_groups': ['viscous', 'coulomb_tanh_0.02', 'bias', 'rotor_inertia'] +
                                      (['linear_spring', 'sin_spring', 'cos_spring'] if elastic_model else []),
                   'warning': 'Spring/gravity and rotor/link inertia may be inseparable. '
                              'Rubber-band hysteresis is not represented by this simple elastic basis. '
                              'Validate on a second trajectory and known loads before control use.'})
    arrays.update({'time_s': t[indices], 'smoothed_q_rad': q, 'dq_rad_s': dq,
                   'ddq_rad_s2': ddq, 'measured_torque_nm': torque[indices]})
    return report, arrays


def save_npz(path, **arrays):
    with Path(path).open('xb') as file:
        np.savez_compressed(file, **arrays)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init')
    init.add_argument('--leader-config')
    init.add_argument('--side', choices=['left', 'right'])
    init.add_argument('--output', required=True)
    for name in ['inspect', 'plan', 'collect']:
        sub = commands.add_parser(name)
        sub.add_argument('--config', required=True)
        sub.add_argument('--output', required=True)
        if name == 'collect':
            sub.add_argument('--execute', action='store_true', help='Drive the bounded excitation; otherwise only read')
            sub.add_argument('--duration', type=float, help='Read-only recording duration; cannot override an executed plan')
    fit = commands.add_parser('fit')
    fit.add_argument('--urdf', required=True)
    fit.add_argument('--data', required=True)
    fit.add_argument('--torque-data', required=True)
    fit.add_argument('--output', required=True, help='New result NPZ; companion JSON report is also written')
    fit.add_argument('--window', type=int, default=11)
    fit.add_argument('--fit-elastic', action='store_true')
    args = parser.parse_args(argv)
    try:
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        if args.command == 'init':
            leader = None
            if args.leader_config:
                content = yaml.safe_load(Path(args.leader_config).read_text())
                if args.side:
                    content = content[args.side]
                leader = content['TeleoperatorConfig']
            with Path(args.output).open('x') as file:
                yaml.safe_dump(initial_config(leader), file, sort_keys=False, allow_unicode=True)
        elif args.command == 'fit':
            report_path = Path(args.output).with_suffix('.json')
            if report_path.exists() or report_path == Path(args.output):
                raise FileExistsError('Fit output must use a new .npz path with a new .json companion')
            report, arrays = fit_dataset(args.urdf, args.data, args.torque_data, args.window, args.fit_elastic)
            save_npz(args.output, metadata_json=json.dumps(report), **arrays)
            with report_path.open('x') as file:
                json.dump(report, file, indent=2)
            print(json.dumps(report, indent=2))
        else:
            config = yaml.safe_load(Path(args.config).read_text())
            if args.command == 'plan':
                plan = ExcitationPlan(config)
                t = np.linspace(0, plan.duration, int(np.ceil(plan.duration*plan.hz))+1)
                q, dq, ddq = plan.evaluate(t)
                save_npz(args.output, time_s=t, command_q_rad=q, command_dq_rad_s=dq,
                         command_ddq_rad_s2=ddq, metadata_json=json.dumps(plan.summary()))
                print(json.dumps(plan.summary(), indent=2))
            else:
                from gello_teleop.gello_identification_hardware import IdentificationBus, collect
                if args.command == 'inspect':
                    with IdentificationBus(config) as bus:
                        status = bus.inspect()
                    with Path(args.output).open('x') as file:
                        json.dump(status, file, indent=2)
                    print(json.dumps(status, indent=2))
                else:
                    if args.execute and args.duration is not None:
                        raise ValueError('--duration is only for read-only collection')
                    collect(config, args.output, args.execute, args.duration)
        print(f'Saved: {args.output}')
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Collection interrupted; partial telemetry saved when available.\n')


if __name__ == '__main__':
    sys.exit(main())
