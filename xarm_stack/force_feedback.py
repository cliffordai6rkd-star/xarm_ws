"""Bounded torque feedback from the follower state to the input device."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TorqueFeedback:
    """Apply calibration, filtering, limits, and a startup ramp to feedback."""

    gain: np.ndarray
    sign: np.ndarray
    bias: np.ndarray
    deadband: np.ndarray
    limit: np.ndarray
    rate_limit: np.ndarray
    lowpass_hz: float | None = 5.0
    ramp_s: float = 2.0
    enabled: bool = False
    dof: int = 7
    _filtered: np.ndarray | None = None
    _previous: np.ndarray | None = None
    _elapsed_s: float = 0.0

    @classmethod
    def from_config(cls, config: dict) -> "TorqueFeedback":
        dof = int(config.get("dof", 7))
        if dof not in {5, 6, 7}:
            raise ValueError("force_feedback.dof must be 5, 6, or 7")
        def vector(name: str, default: float) -> np.ndarray:
            value = config.get(name, default)
            if np.isscalar(value):
                return np.full(dof, float(value), dtype=np.float64)
            result = np.asarray(value, dtype=np.float64).reshape(-1)
            if result.size != dof:
                raise ValueError(f"force_feedback.{name} must have {dof} values")
            return result

        lowpass = config.get("lowpass_hz", 5.0)
        return cls(
            gain=vector("gain", 0.0),
            sign=vector("sign", 1.0),
            bias=vector("bias_nm", 0.0),
            deadband=vector("deadband_nm", 0.0),
            limit=vector("limit_nm", 0.0),
            rate_limit=vector("rate_limit_nm_s", 10.0),
            lowpass_hz=None if lowpass is None else float(lowpass),
            ramp_s=float(config.get("ramp_s", 2.0)),
            enabled=bool(config.get("enabled", False)),
            dof=dof,
        )

    def reset(self) -> None:
        self._filtered = None
        self._previous = np.zeros(self.dof, dtype=np.float64)
        self._elapsed_s = 0.0

    def update(self, measured_torque: np.ndarray, dt_s: float) -> np.ndarray:
        measured = np.asarray(measured_torque, dtype=np.float64).reshape(-1)
        if measured.size != self.dof or not np.isfinite(measured).all():
            raise ValueError(f"measured_torque must be a finite {self.dof}-axis vector")
        dt = max(float(dt_s), 1.0e-4)
        self._elapsed_s += dt
        if not self.enabled:
            return np.zeros(self.dof, dtype=np.float64)
        residual = measured - self.bias
        residual = np.sign(residual) * np.maximum(np.abs(residual) - self.deadband, 0.0)
        if self.lowpass_hz is not None and self.lowpass_hz > 0.0:
            alpha = 1.0 - np.exp(-2.0 * np.pi * self.lowpass_hz * dt)
            if self._filtered is None:
                self._filtered = residual.copy()
            else:
                self._filtered += alpha * (residual - self._filtered)
            residual = self._filtered
        command = np.clip(residual * self.gain * self.sign, -self.limit, self.limit)
        ramp = 1.0 if self.ramp_s <= 0.0 else min(1.0, self._elapsed_s / self.ramp_s)
        command *= ramp
        previous = np.zeros(self.dof, dtype=np.float64) if self._previous is None else self._previous
        maximum_delta = self.rate_limit * dt
        command = previous + np.clip(command - previous, -maximum_delta, maximum_delta)
        self._previous = command.copy()
        return command
