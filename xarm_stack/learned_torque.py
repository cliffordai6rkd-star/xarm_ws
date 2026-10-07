"""Offline identified dynamics + BiLSTM, with delayed, bounded plot-only replay.

The network sees q/dq/(q_cmd-q), never measured torque. Both prediction inputs
and measured torque use the training pipeline: 100 Hz -> zero-phase 5 Hz ->
50 Hz. Live results approximate whole-record filtering using overlapping
buffers; their timestamps remain the source times, not inference completion.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import threading

import numpy as np

DEFAULT_CHECKPOINT = (Path(__file__).resolve().parents[1] /
                      'model/tau_offline/residual_bilstm_zero5_50/model.pt')


def learned_config(raw, config_path, active_sides):
    cfg = dict(raw)
    if not cfg.get('enabled', False):
        return cfg
    cfg['sides'] = tuple(side for side in cfg.get('sides', ['right']) if side in active_sides)
    path = Path(cfg.get('checkpoint', DEFAULT_CHECKPOINT)).expanduser()
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    if not path.is_file():
        raise ValueError(f'Learned torque checkpoint does not exist: {path}')
    cfg['checkpoint'] = str(path.resolve())
    for key, default in [('delay_s', 1.5), ('buffer_s', 4.), ('interval_s', .1)]:
        cfg[key] = float(cfg.get(key, default))
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'learned_torque.{key} must be positive')
    if cfg['delay_s'] < 1. or cfg['buffer_s'] < cfg['delay_s'] + 1.5 + cfg['interval_s']:
        raise ValueError('learned_torque requires delay_s >= 1 and buffer_s >= delay_s + 1.5 + interval_s')
    if 'thresholds_l1' in cfg:
        a, b = cfg['thresholds_l1']
        if not np.isfinite([a, b]).all() or not 0 <= a < b:
            raise ValueError('learned_torque.thresholds_l1 requires 0 <= a < b')
    cfg.setdefault('device', 'cpu')
    cfg.setdefault('threads', 2)
    if not isinstance(cfg['threads'], int) or cfg['threads'] < 1:
        raise ValueError('learned_torque.threads must be a positive integer')
    return cfg


class LearnedTorqueModel:
    """Portable artifact: weights, normalization, 98 coefficients and URDF XML."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device='cpu', threads=2):
        import pinocchio as pin
        import torch
        from torch import nn
        from scipy.signal import butter

        torch.set_num_threads(threads)
        cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
        expected = dict(arch='bilstm', filter='zero5', rate=50, horizon=51)
        if cp.get('format_version') != 1 or any(cp['spec'].get(k) != v for k, v in expected.items()):
            raise ValueError('Expected the exported zero5/50 Hz/51-frame residual BiLSTM artifact')

        class Regressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(21, 128, 2, batch_first=True, dropout=.1, bidirectional=True)
                self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 7))

            def forward(self, x):
                x, _ = self.encoder(x)
                return self.head(x[:, x.shape[1]//2])

        self.device = torch.device(device)
        self.network = Regressor().to(self.device).eval()
        self.network.load_state_dict(cp['model'])
        self.norm = {k: v.numpy() for k, v in cp['normalization'].items()}
        for key, size in [('x_mean', 21), ('x_std', 21), ('y_mean', 7), ('y_std', 7)]:
            value = self.norm[key]
            if value.shape != (size,) or not np.isfinite(value).all() or ('std' in key and np.any(value <= 0)):
                raise ValueError(f'Invalid learned torque normalization: {key}')
        self.coefficients = cp['coefficients'].numpy()
        if self.coefficients.shape != (98,) or not np.isfinite(self.coefficients).all():
            raise ValueError('Expected 98 finite dynamics/friction coefficients')
        self.physics = pin.buildModelFromXML(cp['urdf_xml'])
        if self.physics.nq != 7 or self.physics.nv != 7:
            raise ValueError('Learned torque requires the training 7-DOF URDF')
        self.data = self.physics.createData()
        self.sos = butter(4, 5, fs=100, output='sos')

    def predict(self, times, q, dq, q_cmd, tau, *, output_times=None, grid_origin=None):
        """One uninterrupted segment; optional sparse outputs reduce live CPU cost."""
        import pinocchio as pin
        import torch
        from scipy.signal import resample_poly, sosfiltfilt

        times = np.asarray(times, dtype=float)
        columns = [np.asarray(a) for a in (q, dq, q_cmd, tau)]
        if (times.ndim != 1 or len(times) < 32 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .03 + 1e-6)
                or any(a.shape != (len(times), 7) or not np.isfinite(a).all() for a in columns)):
            raise ValueError('Learned torque needs >=32 finite, increasing, contiguous 100 Hz rows')
        q, dq, q_cmd, tau = columns
        origin = times[0] if grid_origin is None else grid_origin
        t = times-origin
        # Keep the 50 Hz lattice anchored to the beginning of the segment,
        # even when the live input buffer discards old rows.
        start = int(np.ceil(t[0]*50-1e-5))*2
        stop = int(np.floor(t[-1]*100+1e-5))+1
        grid = np.arange(start, stop)*.01
        if len(grid) < 32:
            raise ValueError('Insufficient uniform-grid context')
        processed = []
        for a in (q, dq, q_cmd-q, tau):
            uniform = np.stack([np.interp(grid, t, a[:, j]) for j in range(7)], axis=1)
            filtered = sosfiltfilt(self.sos, uniform, axis=0)
            processed.append(resample_poly(filtered, 1, 2, axis=0, padtype='line').astype(np.float32))
        pq, pdq, delta, measured = processed
        grid = grid[::2]
        target = times if output_times is None else np.asarray(output_times, dtype=float)
        if target.ndim != 1 or not np.isfinite(target).all() or np.any(np.diff(target) <= 0):
            raise ValueError('output_times must be finite and increasing')
        if len(target) and (target[0] < times[0] or target[-1] > times[-1]):
            raise ValueError('output_times must be within the input segment')
        # Only evaluate the 50 Hz points bracketing requested 100 Hz outputs.
        high = np.searchsorted(grid, target-origin).clip(0, len(grid)-1)
        indices = np.unique(np.concatenate((high, (high-1).clip(0))))
        if not len(indices):
            return dict(timestamp_s=target, tau_free=np.empty((0, 7)),
                        tau_measured=np.empty((0, 7)), tau_ext=np.empty((0, 7)), valid_context=np.empty(0, bool))
        features = (np.concatenate((pq, pdq, delta), axis=1)-self.norm['x_mean'])/self.norm['x_std']
        x = torch.as_tensor(features, device=self.device)
        predictions = []
        with torch.inference_mode():
            for first in range(0, len(indices), 512):
                ix = indices[first:first+512, None] + np.arange(-25, 26)[None, :]
                batch = x[torch.as_tensor(ix.clip(0, len(x)-1), device=self.device)]
                predictions.append(self.network(batch).cpu().numpy())
        pred = np.concatenate(predictions)*self.norm['y_std']+self.norm['y_mean']
        q64, dq64 = pq.astype(float), pdq.astype(float)
        ddq = np.gradient(dq64, .02, axis=0)
        prior = []
        for i in indices:
            reg = pin.computeJointTorqueRegressor(self.physics, self.data, q64[i], dq64[i], ddq[i])
            dynamics = reg@self.coefficients[:70] + pin.rnea(self.physics, self.data, q64[i], dq64[i], ddq[i])
            friction = np.stack((dq64[i], np.tanh(dq64[i]/.02), np.tanh(dq64[i]/.002), np.ones(7)), axis=1)
            prior.append(dynamics + (friction*self.coefficients[70:].reshape(7, 4)).sum(axis=1))
        pred += np.asarray(prior, dtype=np.float32)
        free100 = np.stack([np.interp(target-origin, grid[indices], pred[:, j]) for j in range(7)], axis=1)
        tau100 = np.stack([np.interp(target-origin, grid, measured[:, j]) for j in range(7)], axis=1)
        return dict(timestamp_s=target.copy(), tau_free=free100, tau_measured=tau100, tau_ext=tau100-free100,
                    valid_context=(target >= times[0]+1.) & (target <= times[-1]-1.))


class DelayedTorqueStream:
    """Sampling appends only; a separate plot thread performs buffered inference."""

    def __init__(self, model, cfg, window_s=30., stale_s=.15):
        self.model, self.cfg, self.window_s, self.stale_s = model, cfg, window_s, stale_s
        self.lock = threading.Lock()
        self.inputs = deque(maxlen=int(np.ceil(cfg['buffer_s']*200))+100)
        self.rows = deque(maxlen=max(20000, int(window_s*200)))
        self.origin = self.emitted = None
        self.generation = 0
        self.status = 'waiting for q_cmd and valid state'

    def _reset(self, reason):
        self.inputs.clear()
        self.origin = self.emitted = None
        self.generation += 1
        if self.rows and np.isfinite(self.rows[-1][1]).any():
            self.rows.append((self.rows[-1][0]+.01, np.full(7, np.nan)))
        self.status = reason

    def append(self, sample, now):
        stamp = float(sample.get('lowdim_timestamp_s', sample['timestamp_s']))
        state_stamp = float(sample['timestamp_s'])
        values = [np.asarray(sample.get(k, np.full(7, np.nan)), float) for k in ('q', 'dq', 'q_cmd', 'torque')]
        valid = all(sample.get(k, False) for k in ('q_valid', 'dq_valid', 'q_cmd_valid', 'torque_valid'))
        with self.lock:
            if (not valid or not np.isfinite([stamp, state_stamp]).all()
                    or not 0 <= now-stamp <= self.stale_s or not 0 <= now-state_stamp <= self.stale_s
                    or any(a.shape != (7,) or not np.isfinite(a).all() for a in values)):
                self._reset('invalid / stale state or command')
                return
            if self.rows and stamp < self.rows[-1][0]:
                self.rows.clear()
                self._reset('clock reset; buffering')
            if self.inputs:
                dt = stamp-self.inputs[-1][0]
                if dt == 0:
                    return
                if dt < 0:
                    self.rows.clear()
                    self._reset('clock reset; buffering')
                elif dt > .03+1e-6:
                    self._reset('sample gap; buffering')
            if self.origin is None:
                self.origin = stamp
            self.inputs.append((stamp, *[a.copy() for a in values]))
            # Retain one extra interpolation endpoint before the buffer cutoff.
            while len(self.inputs) > 2 and self.inputs[1][0] < stamp-self.cfg['buffer_s']:
                self.inputs.popleft()

    def update(self, now):
        with self.lock:
            if not self.inputs:
                return
            if not 0 <= now-self.inputs[-1][0] <= self.stale_s:
                self._reset('state unavailable / stale')
                return
            rows = list(self.inputs)
            t = np.array([row[0] for row in rows])
            origin, generation = self.origin, self.generation
            first = max(origin+1., t[0]+1.)
            selected = (t >= first) & (t <= t[-1]-self.cfg['delay_s'])
            if self.emitted is not None:
                selected &= t > self.emitted
            if not selected.any():
                if self.emitted is None:
                    self.status = f'buffering ({t[-1]-origin:.1f} / {1+self.cfg["delay_s"]:.1f} s)'
                return
        result = self.model.predict(t, *[np.stack([r[i] for r in rows]) for i in range(1, 5)],
                                    output_times=t[selected], grid_origin=origin)
        with self.lock:
            if generation != self.generation:
                return  # Input reset while inference was running.
            if (self.rows and np.isfinite(self.rows[-1][1]).any()
                    and result['timestamp_s'][0]-self.rows[-1][0] > .03+1e-6):
                self.rows.append((self.rows[-1][0]+.01, np.full(7, np.nan)))
            for stamp, residual in zip(result['timestamp_s'], result['tau_ext']):
                self.rows.append((stamp, residual))
            self.emitted = result['timestamp_s'][-1]
            self.status = f'delayed {t[-1]-self.emitted:.2f} s'

    def set_offline_result(self, result):
        with self.lock:
            self.rows.clear()
            for stamp, residual, valid in zip(result['timestamp_s'], result['tau_ext'], result['valid_context']):
                self.rows.append((stamp, residual if valid else np.full(7, np.nan)))
            self.status = 'whole-record offline; 1 s edges excluded'

    def arrays(self, now):
        with self.lock:
            while self.rows and self.rows[0][0] < now-self.window_s:
                self.rows.popleft()
            if not self.rows:
                return np.empty(0), np.empty((0, 7))
            return np.array([r[0] for r in self.rows])-now, np.stack([r[1] for r in self.rows])


def read_h5(path, side):
    """Read either dual-arm bg or per-side peeling lowdim schema, without cameras."""
    import h5py
    with h5py.File(path, 'r') as f:
        group = f['teleop']
        stamp = group['timestamp_us'][:]
        arm_index = 0 if side == 'left' else 1
        dual = 'q_follower' in group
        if dual:
            column = slice(7*arm_index, 7*(arm_index+1))
            values = {k: group[name][:, column] for k, name in
                      [('q', 'q_follower'), ('dq', 'dq_follower'), ('tau', 'tau_follower'), ('q_cmd', 'q_cmd')]}
            flags = ['q_follower_valid', 'dq_valid_follower', 'torque_valid_follower']
        else:
            values = {k: group[f'{side}_{k}_xarm'][:] for k in ('q', 'dq', 'tau', 'q_cmd')}
            flags = [f'{side}_{key}_xarm' for key in ('q_valid', 'dq_valid', 'torque_valid')]
        valid = np.ones(len(stamp), bool)
        for name in flags:
            flag = group[name][:].reshape(len(stamp), -1)
            valid &= flag[:, min(arm_index if dual else 0, flag.shape[1]-1)].astype(bool)
        for value in values.values():
            valid &= np.isfinite(value).all(axis=1)
    return dict(t=(stamp-stamp[0])*1e-6, timestamp_us=stamp, valid=valid, **values)


def predict_episode(model, episode):
    """Split on missing data/gaps, leaving short segments and filter edges invalid."""
    t = episode['t']
    output = dict(timestamp_s=t.copy(), valid_context=np.zeros(len(t), bool),
                  **{k: np.full((len(t), 7), np.nan) for k in ('tau_free', 'tau_measured', 'tau_ext')})
    breaks = np.r_[True, (np.diff(t) <= 0) | (np.diff(t) > .03+1e-6), True]
    valid = episode['valid']
    start = None
    for i in range(len(t)+1):
        if start is not None and (i == len(t) or breaks[i] or not valid[i]):
            if i-start >= 32 and t[i-1]-t[start] >= 2.:
                result = model.predict(t[start:i], *[episode[k][start:i] for k in ('q', 'dq', 'q_cmd', 'tau')])
                for key in output:
                    output[key][start:i] = result[key]
            start = None
        if i < len(t) and valid[i] and start is None:
            start = i
    return output
