"""Convert a virtual MIT joint torque into a bounded position target."""

from pathlib import Path

import numpy as np


def _vector(name, value, dof):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (dof,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite {dof}-joint vector')
    return result


class PinocchioArmModel:
    def __init__(self, urdf_path, joint_names, locked_joint_names=()):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError('MIT-to-position control requires pin>=3,<4') from exc

        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Arm dynamics URDF does not exist: {path}')
        full_model = pin.buildModelFromUrdf(str(path))
        missing = [name for name in locked_joint_names if not full_model.existJointName(name)]
        if missing:
            raise ValueError(f'URDF locked joints not found: {missing}')
        locked_ids = [full_model.getJointId(name) for name in locked_joint_names]
        self.pin = pin
        self.model = (pin.buildReducedModel(full_model, locked_ids, pin.neutral(full_model))
                      if locked_ids else full_model)
        self.dof = len(joint_names)
        actual_names = tuple(str(name) for name in self.model.names[1:])
        if self.model.nq != self.dof or self.model.nv != self.dof or actual_names != tuple(joint_names):
            raise ValueError(
                f'URDF must reduce to the configured xArm joint order {tuple(joint_names)}; '
                f'got nq={self.model.nq}, nv={self.model.nv}, joints={actual_names}'
            )
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()
        self.data = self.model.createData()

    def terms(self, q, dq):
        q = _vector('measured q', q, self.dof)
        dq = _vector('measured dq', dq, self.dof)
        mass = np.asarray(self.pin.crba(self.model, self.data, q), dtype=np.float64)
        # Pinocchio CRBA fills the upper triangle; mirror it without halving
        # the off-diagonal inertia terms.
        mass = np.triu(mass) + np.triu(mass, 1).T
        nonlinear = np.asarray(
            self.pin.nonLinearEffects(self.model, self.data, q, dq), dtype=np.float64
        ).copy()
        gravity = np.asarray(
            self.pin.computeGeneralizedGravity(self.model, self.data, q), dtype=np.float64
        ).copy()
        if (mass.shape != (self.dof, self.dof) or not np.isfinite(mass).all()
                or not np.isfinite(nonlinear).all() or not np.isfinite(gravity).all()):
            raise ValueError('Invalid URDF dynamics result')
        return mass, nonlinear, gravity


class MitPositionController:
    """Semi-implicit integration: ddq -> dq_cmd -> q_cmd, with live q/dq feedback."""

    def __init__(self, model, kp, kd, torque_limit, velocity_limit,
                 acceleration_limit, tracking_error_limit):
        self.model = model
        n = model.dof
        self.kp = _vector('mit_kp', kp, n)
        self.kd = _vector('mit_kd', kd, n)
        self.torque_limit = _vector('mit_torque_limit_nm', torque_limit, n)
        if np.any(self.kp < 0) or np.any(self.kd < 0) or np.any(self.torque_limit <= 0):
            raise ValueError('MIT gains must be non-negative and torque limits positive')
        for name, value in (('velocity_limit', velocity_limit),
                            ('acceleration_limit', acceleration_limit),
                            ('tracking_error_limit', tracking_error_limit)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        self.velocity_limit = float(velocity_limit)
        self.acceleration_limit = float(acceleration_limit)
        self.tracking_error_limit = float(tracking_error_limit)
        self.q_cmd = None
        self.dq_cmd = None
        self.last_measured_q = None
        self.last_measured_time = None

    def reset(self, q, timestamp):
        q = _vector('measured q', q, self.model.dof)
        if not np.isfinite(timestamp):
            raise ValueError('Invalid feedback timestamp')
        self._check_limits(q, 'Measured q')
        self.q_cmd = q.copy()
        self.dq_cmd = np.zeros(self.model.dof, dtype=np.float64)
        self.last_measured_q = q.copy()
        self.last_measured_time = float(timestamp)

    def step(self, desired_q, measured_q, timestamp, max_integration_dt):
        if self.q_cmd is None:
            raise RuntimeError('MIT-to-position controller must be reset before use')
        desired_q = _vector('desired q', desired_q, self.model.dof)
        measured_q = _vector('measured q', measured_q, self.model.dof)
        self._check_limits(desired_q, 'Desired q')
        self._check_limits(measured_q, 'Measured q')
        sample_dt = float(timestamp) - self.last_measured_time
        if not np.isfinite(sample_dt) or sample_dt <= 0:
            raise ValueError('Feedback timestamps must increase')
        if not np.isfinite(max_integration_dt) or max_integration_dt <= 0:
            raise ValueError('max_integration_dt must be finite and positive')
        measured_dq = (measured_q - self.last_measured_q) / sample_dt
        dt = min(sample_dt, max_integration_dt)

        mass, nonlinear, gravity = self.model.terms(measured_q, measured_dq)
        if np.any(np.abs(gravity) > self.torque_limit):
            raise ValueError('MIT torque limits cannot hold gravity at the current q')
        # Gravity is the MIT feed-forward term. The nonlinear term also includes
        # gravity, so these cancel at rest when the target equals the feedback.
        tau = self.kp * (desired_q - measured_q) - self.kd * measured_dq + gravity
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        try:
            np.linalg.cholesky(mass)
            ddq = np.linalg.solve(mass, tau - nonlinear)
        except np.linalg.LinAlgError as exc:
            raise ValueError('URDF mass matrix must be positive definite') from exc
        if not np.isfinite(ddq).all():
            raise ValueError('Invalid joint acceleration from dynamics')
        ddq = np.clip(ddq, -self.acceleration_limit, self.acceleration_limit)
        next_dq = np.clip(self.dq_cmd + ddq * dt, -self.velocity_limit, self.velocity_limit)
        next_q = self.q_cmd + next_dq * dt
        self._check_limits(next_q, 'Commanded q')
        if np.max(np.abs(next_q - measured_q)) > self.tracking_error_limit:
            raise ValueError('MIT-to-position command exceeds measured joint tracking limit')

        self.dq_cmd = next_dq
        self.q_cmd = next_q
        self.last_measured_q = measured_q.copy()
        self.last_measured_time = float(timestamp)
        return next_q.copy()

    def _check_limits(self, q, label):
        if np.any(q < self.model.lower) or np.any(q > self.model.upper):
            raise ValueError(f'{label} exceeds URDF joint limits')
