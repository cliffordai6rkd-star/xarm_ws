"""Bounded, time-aligned handover of q-only WM references."""
from __future__ import annotations

from dataclasses import replace

import numpy as np


class _Trajectory:
    def __init__(self, forecast, sample_id):
        self.anchor_us = int(forecast['anchor_us'])
        self.times = np.asarray(forecast['future_timestamps_us'])
        self.q = np.asarray(forecast['q'])[sample_id]
        if (self.times.ndim != 1 or self.times.dtype.kind not in 'iu' or len(self.times) < 2
                or np.any(np.diff(self.times) <= 0) or self.q.shape != (len(self.times), 7)
                or not np.isfinite(self.q).all()):
            raise ValueError('Invalid q trajectory for WM handover')

    def sample(self, targets):
        targets = np.asarray(targets, dtype=np.int64)
        if targets.min() < self.times[0] or targets.max() > self.times[-1]:
            raise ValueError('WM handover cannot extrapolate a prediction')
        relative = self.times-self.times[0]
        q = np.stack([np.interp(targets-self.times[0], relative, self.q[:, j])
                      for j in range(7)], axis=-1)
        indices = np.searchsorted(self.times, targets, side='right').clip(1, len(self.times)-1)-1
        v = (self.q[indices+1]-self.q[indices]) / ((self.times[indices+1]-self.times[indices])[:, None]/1e6)
        return q, v


class ForecastTransition:
    """Quintic offset/slope correction, preserving the newest forecast clock.

    Only the latest raw trajectory and one polynomial are retained. Repeated
    swaps start from the effective trajectory, including its current correction.
    Old data may contribute only inside its valid overlapping time interval.
    """

    def __init__(self, *, duration_ms, sample_id, maximum_age_ms, dt=.01):
        self.duration_us = round(duration_ms*1000)
        self.maximum_age_us = round(maximum_age_ms*1000)
        self.step_us = round(dt*1e6)
        self.sample_id = sample_id
        self.reset()

    def reset(self):
        self.pending = self.curve = self.correction = None
        self.reason = 'first_forecast'
        self.last_jump = np.zeros(7)
        self.last_velocity_jump = np.zeros(7)
        self.switch_count = 0

    def update(self, forecast):
        # The owning control layer has already copied the immutable snapshot.
        # Defer handover to the first actual physical target after arrival.
        self.pending = forecast

    def _effective(self, targets):
        q, v = self.curve.sample(targets)
        if self.correction is not None:
            start, end, offset, slope, _ = self.correction
            duration = (end-start)/1e6
            s = np.clip((np.asarray(targets)-start)/(end-start), 0., 1.)[:, None]
            hq = 1-10*s**3+15*s**4-6*s**5
            hv = s-6*s**3+8*s**4-3*s**5
            dhq = -30*s**2+60*s**3-30*s**4
            dhv = 1-18*s**2+32*s**3-15*s**4
            q = q+hq*offset+duration*hv*slope
            v = v+dhq*offset/duration+dhv*slope
        return q, v

    def _install(self, start_us):
        new = _Trajectory(self.pending, self.sample_id)
        self.pending = None
        correction = None
        self.reason = 'first_forecast'
        self.last_jump = np.zeros(7)
        self.last_velocity_jump = np.zeros(7)
        if self.curve is not None:
            self.switch_count += 1
            old = self.curve
            old_until = min(int(old.times[-1]), old.anchor_us+self.maximum_age_us)
            if self.correction is not None and start_us < self.correction[1]:
                old_until = min(old_until, self.correction[4])
            new_until = min(int(new.times[-1]), new.anchor_us+self.maximum_age_us)
            end_us = min(start_us+self.duration_us, old_until, new_until)
            if start_us < old.times[0] or start_us > old_until:
                self.reason = 'old_forecast_unavailable'
            elif end_us-start_us < self.step_us:
                self.reason = 'insufficient_valid_overlap'
            else:
                old_q, old_v = self._effective(np.array([start_us], dtype=np.int64))
                new_q, new_v = new.sample(np.array([start_us], dtype=np.int64))
                offset, slope = old_q[0]-new_q[0], old_v[0]-new_v[0]
                self.last_jump = -offset
                self.last_velocity_jump = -slope
                if np.max(np.abs(np.r_[offset, slope])) > 1e-12:
                    correction = (start_us, end_us, offset, slope, old_until)
                    self.reason = 'overlap_handover'
                else:
                    self.reason = 'same_time_trajectory_unchanged'
        self.curve, self.correction = new, correction

    def apply(self, reference, *, command_index=0):
        if reference.q is None:
            raise ValueError('WM forecast handover requires q references')
        if type(command_index) is not int or not 0 <= command_index < len(reference.timestamps_us):
            raise ValueError('Invalid WM handover command_index')
        prefix = None
        if self.pending is not None:
            if command_index and self.curve is not None:
                try:
                    prefix = self._effective(reference.timestamps_us[:command_index])[0]
                except ValueError:
                    pass
            self._install(int(reference.timestamps_us[command_index]))
        if self.curve is None:
            return reference, {}
        q, _ = self._effective(reference.timestamps_us)
        # Earlier physical candidates stay on the previous trajectory during
        # installation. Only the selected endpoint is sent at 50 Hz.
        if prefix is not None and self.correction is not None:
            q[:command_index] = prefix
        selected_time = int(reference.timestamps_us[command_index])
        active = self.correction is not None and selected_time < self.correction[1]
        age_ms = (0. if self.correction is None else
                  max(0., (selected_time-self.correction[0])/1000))
        duration_ms = 0. if self.correction is None else (self.correction[1]-self.correction[0])/1000
        details = dict(q_reference_raw=reference.q.copy(), q_reference_correction=q-reference.q,
                       forecast_transition_active=active, forecast_transition_age_ms=age_ms,
                       forecast_transition_duration_ms=duration_ms,
                       forecast_transition_reason=self.reason,
                       forecast_switch_jump_rad=self.last_jump.copy(),
                       forecast_switch_velocity_jump_rad_s=self.last_velocity_jump.copy())
        return replace(reference, q=q), details
