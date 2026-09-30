"""Seven-axis rigid-body dynamics and an acceleration-free momentum observer."""

from pathlib import Path

import numpy as np


class PinocchioMomentumModel:
    def __init__(self, urdf_path, gravity=(0., 0., -9.81)):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError('Torque visualization requires pin>=3,<4') from exc
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(Path(urdf_path).resolve()))
        names = tuple(str(name) for name in self.model.names[1:])
        if self.model.nq != 7 or self.model.nv != 7 or names != tuple(f'joint{i}' for i in range(1, 8)):
            raise ValueError('Torque URDF must contain only xArm joint1..joint7 in that order')
        gravity = np.asarray(gravity, dtype=float)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ValueError('gravity_m_s2 must be a finite three-vector in the base frame')
        self.model.gravity.linear[:] = gravity
        self.data = self.model.createData()
        self._flange_inertia = self.model.inertias[7].copy()
        self._payload_key = None

    def set_payload(self, payload):
        """Add a fixed tool in link_eef/flange coordinates (kg, metres, kg m²).

        Without a supplied CoM rotational tensor, model the tool as a point
        mass. This includes its gravity and translational inertia, but not its
        own rotational inertia. Never use the TCP position as the tool CoM.
        """
        if payload is None:
            key, tool = None, None
        else:
            mass = float(payload['mass_kg'])
            com = np.asarray(payload['com_m'], dtype=float)
            inertia = np.asarray(payload.get('inertia_kg_m2', np.zeros((3, 3))), dtype=float)
            if (not np.isfinite(mass) or mass < 0 or com.shape != (3,)
                    or not np.isfinite(com).all() or inertia.shape != (3, 3)
                    or not np.isfinite(inertia).all() or not np.allclose(inertia, inertia.T)
                    or np.linalg.eigvalsh(inertia).min() < -1e-12):
                raise ValueError('Invalid flange payload mass, CoM or inertia')
            key = (mass, *com, *inertia.ravel())
            tool = self.pin.Inertia(mass, com, inertia)
        if key == self._payload_key:
            return False
        flange = self.model.frames[self.model.getFrameId('link_eef')]
        self.model.inertias[7] = self._flange_inertia.copy()
        if tool is not None:
            self.model.inertias[7] += flange.placement.act(tool)
        self._payload_key = key
        return True

    def terms(self, q, dq):
        """Return p=M(q)dq and beta=C(q,dq).T dq-g(q), both in joint order."""
        pin, model, data = self.pin, self.model, self.data
        upper = np.triu(np.asarray(pin.crba(model, data, q), dtype=float))
        mass = upper + np.triu(upper, 1).T
        coriolis = np.asarray(pin.computeCoriolisMatrix(model, data, q, dq), dtype=float).copy()
        gravity = np.asarray(pin.computeGeneralizedGravity(model, data, q), dtype=float).copy()
        return mass @ dq, coriolis.T @ dq - gravity

    def gravity_torque(self, q):
        return np.asarray(self.pin.computeGeneralizedGravity(self.model, self.data, q), dtype=float).copy()

    def rnea(self, q, dq, ddq):
        """Instantaneous inverse dynamics in Nm; inputs are rad, rad/s, rad/s²."""
        return np.asarray(self.pin.rnea(self.model, self.data, q, dq, ddq), dtype=float).copy()


class FreeSpaceMomentumObserver:
    """Observe total rigid-body torque from q,dq alone, without measured effort.

    p_dot = tau_free + beta; p_hat_dot = beta + r; r = K(p-p_hat).
    Thus r_dot = K(tau_free-r). This is a filtered inverse-dynamics torque,
    not an external-torque residual. Integrate by the trapezoidal rule using
    actual state timestamps. No ddq or differentiated dq is consumed.
    """

    def __init__(self, model, bandwidth_hz=3., max_gap_s=.15):
        bandwidth = np.asarray(bandwidth_hz, dtype=float)
        if bandwidth.ndim == 0:
            bandwidth = np.full(7, float(bandwidth))
        if bandwidth.shape != (7,) or not np.isfinite(bandwidth).all() or np.any(bandwidth <= 0):
            raise ValueError('observer_bandwidth_hz must be a positive scalar or seven-vector')
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('max_gap_s must be positive')
        self.model, self.gain, self.max_gap_s = model, 2*np.pi*bandwidth, max_gap_s
        self.reset()

    def reset(self):
        self.timestamp_s = None
        self.p = self.beta = self.torque = None

    def update(self, timestamp_s, q, dq, valid=True):
        q, dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
        if (not valid or not np.isfinite(timestamp_s) or q.shape != (7,) or dq.shape != (7,)
                or not np.isfinite(q).all() or not np.isfinite(dq).all()):
            self.reset()
            return np.full(7, np.nan)
        if self.timestamp_s is not None and timestamp_s == self.timestamp_s:
            return self.torque.copy()  # Repeated acquisition is not a new integration step.
        p, beta = self.model.terms(q, dq)
        if not np.isfinite(p).all() or not np.isfinite(beta).all():
            self.reset()
            return np.full(7, np.nan)
        dt = None if self.timestamp_s is None else timestamp_s-self.timestamp_s
        if dt is None or dt <= 0 or dt > self.max_gap_s:
            self.timestamp_s, self.p, self.beta = timestamp_s, p.copy(), beta.copy()
            # Initialize to the constant-momentum torque, avoiding a gravity
            # step transient at startup. Acceleration at this first sample is unknown.
            self.torque = -beta.copy()
            return np.full(7, np.nan)
        half_step = self.gain*dt/2
        self.torque = ((1-half_step)*self.torque + self.gain*(p-self.p-dt/2*(self.beta+beta))) / (1+half_step)
        self.timestamp_s, self.p, self.beta = timestamp_s, p.copy(), beta.copy()
        return self.torque.copy()
