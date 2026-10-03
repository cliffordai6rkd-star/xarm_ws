"""Bounded joint reference shared by teleoperation and policy inference."""
from __future__ import annotations

import numpy as np


class SecondOrderPositionFollower:
    """Bounded independent reference state for xArm position servoing."""

    def __init__(self, dof: int, mode: str = "direct", *, kp=18.0, kd=3.0,
                 max_velocity=1.5, max_acceleration=8.0, max_step=0.08,
                 max_tracking_error=0.25, joint_limits=None):
        self.dof = int(dof)
        self.mode = str(mode).lower()
        if self.mode not in {"direct", "second_order"}:
            raise ValueError("control.position_mode must be direct or second_order")
        self.kp = _joint_vector(kp, self.dof, "kp")
        self.kd = _joint_vector(kd, self.dof, "kd")
        self.max_velocity = _joint_vector(max_velocity, self.dof, "max_velocity")
        self.max_acceleration = _joint_vector(max_acceleration, self.dof, "max_acceleration")
        self.max_step = _joint_vector(max_step, self.dof, "max_step")
        self.max_tracking_error = _joint_vector(max_tracking_error, self.dof, "max_tracking_error")
        self.joint_limits = None if joint_limits is None else np.asarray(joint_limits, dtype=float)
        if self.joint_limits is not None and self.joint_limits.shape != (self.dof, 2):
            raise ValueError("joint_limits must have shape (dof, 2)")
        self.q_ref: np.ndarray | None = None
        self.dq_ref = np.zeros(self.dof)

    def initialize(self, q_actual: np.ndarray) -> None:
        q = np.asarray(q_actual, dtype=float).reshape(self.dof)
        if not np.isfinite(q).all():
            raise ValueError("follower initialization is not finite")
        self.q_ref = q.copy()
        self.dq_ref.fill(0.0)

    def update(self, target: np.ndarray, q_actual: np.ndarray, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=float).reshape(self.dof)
        q_actual = np.asarray(q_actual, dtype=float).reshape(self.dof)
        if not np.isfinite(target).all() or not np.isfinite(q_actual).all():
            raise ValueError("position reference contains non-finite values")
        if self.q_ref is None:
            self.initialize(q_actual)
        dt = float(np.clip(dt, 1e-4, 0.1))
        if self.mode == "direct":
            step_limit = np.minimum(self.max_step, self.max_velocity*dt)
            q_next = self.q_ref + np.clip(target - self.q_ref, -step_limit, step_limit)
            q_next = np.minimum(q_next, q_actual + self.max_tracking_error)
            q_next = np.maximum(q_next, q_actual - self.max_tracking_error)
            self.dq_ref = np.clip((q_next - self.q_ref) / dt, -self.max_velocity, self.max_velocity)
            self.q_ref = q_next
        else:
            acceleration = np.clip(self.kp * (target - self.q_ref) - self.kd * self.dq_ref,
                                   -self.max_acceleration, self.max_acceleration)
            self.dq_ref = np.clip(self.dq_ref + acceleration * dt, -self.max_velocity, self.max_velocity)
            q_next = self.q_ref + self.dq_ref * dt
            step = np.clip(q_next - self.q_ref, -self.max_step, self.max_step)
            q_next = self.q_ref + step
            q_next = np.minimum(q_next, q_actual + self.max_tracking_error)
            q_next = np.maximum(q_next, q_actual - self.max_tracking_error)
            self.q_ref = q_next
        if self.joint_limits is not None:
            self.q_ref = np.clip(self.q_ref, self.joint_limits[:, 0], self.joint_limits[:, 1])
        return self.q_ref.copy()



def _joint_vector(value, dof, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(dof, float(array))
    array = array.reshape(-1)
    if array.shape != (dof,) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must be a non-negative scalar or {dof}-vector")
    return array

