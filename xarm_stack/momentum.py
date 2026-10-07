"""Seven-axis rigid-body dynamics and an acceleration-free momentum observer."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _seven_joint_parameter(value, name, *, positive=False):
    """Normalize a finite numeric scalar or J1..J7 vector."""
    try:
        objects = np.asarray(value, dtype=object)
        if any(isinstance(item, (bool, np.bool_)) for item in objects.flat):
            raise ValueError('boolean values are not coefficients')
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a numeric scalar or seven-vector') from exc
    if array.ndim == 0:
        array = np.full(7, float(array))
    if (array.shape != (7,) or not np.isfinite(array).all()
            or np.any(array <= 0 if positive else array < 0)):
        qualifier = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be a finite {qualifier} scalar or seven-vector')
    return array.copy()


class JointFrictionModel:
    """Explicit viscous and smoothed Coulomb compensation, in Nm.

    URDF damping/friction values are optional initial coefficients, not a
    calibration of the physical robot. Merely loading a URDF never enables
    compensation.
    """

    def __init__(self, *, enabled=False, source='configured',
                 viscous_nm_s_per_rad=0., coulomb_nm=0.,
                 velocity_smoothing_rad_s=.02, urdf_path=None):
        if type(enabled) is not bool:
            raise ValueError('friction.enabled must be a boolean')
        if not isinstance(source, str) or source not in {'configured', 'urdf'}:
            raise ValueError('friction.source must be configured or urdf')
        self.enabled, self.source = enabled, source
        self.viscous = _seven_joint_parameter(viscous_nm_s_per_rad, 'friction.viscous_nm_s_per_rad')
        self.coulomb = _seven_joint_parameter(coulomb_nm, 'friction.coulomb_nm')
        self.smoothing = _seven_joint_parameter(velocity_smoothing_rad_s,
                                               'friction.velocity_smoothing_rad_s', positive=True)
        self.urdf_path = None if urdf_path is None else str(Path(urdf_path).expanduser().resolve())

    @classmethod
    def from_config(cls, config=None, urdf_path=None):
        config = {} if config is None else config
        allowed = {'enabled', 'source', 'viscous_nm_s_per_rad', 'coulomb_nm',
                   'velocity_smoothing_rad_s'}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError('friction must be a mapping with supported fields')
        source = config.get('source', 'configured')
        if not isinstance(source, str) or source not in {'configured', 'urdf'}:
            raise ValueError('friction.source must be configured or urdf')
        viscous, coulomb = np.zeros(7), np.zeros(7)
        if source == 'urdf':
            if urdf_path is None:
                raise ValueError('friction.source urdf requires urdf_path')
            try:
                root = ET.parse(Path(urdf_path).expanduser()).getroot()
            except (OSError, ET.ParseError, TypeError, ValueError) as exc:
                raise ValueError('Cannot read friction coefficients from URDF') from exc
            for index in range(7):
                name = f'joint{index + 1}'
                joints = [joint for joint in root.findall('joint') if joint.get('name') == name]
                if len(joints) != 1:
                    raise ValueError(f'Friction URDF must contain exactly one {name}')
                dynamics = joints[0].findall('dynamics')
                if len(dynamics) > 1:
                    raise ValueError(f'Friction URDF {name} has duplicate dynamics')
                if dynamics:
                    try:
                        viscous[index] = float(dynamics[0].get('damping', '0'))
                        coulomb[index] = float(dynamics[0].get('friction', '0'))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f'Invalid URDF friction coefficients for {name}') from exc
            # Validate the source itself even if explicit coefficients override it.
            _seven_joint_parameter(viscous, 'URDF damping')
            _seven_joint_parameter(coulomb, 'URDF friction')
        return cls(enabled=config.get('enabled', False), source=source,
                   viscous_nm_s_per_rad=config.get('viscous_nm_s_per_rad', viscous),
                   coulomb_nm=config.get('coulomb_nm', coulomb),
                   velocity_smoothing_rad_s=config.get('velocity_smoothing_rad_s', .02),
                   urdf_path=urdf_path if source == 'urdf' else None)

    def torque(self, dq):
        dq = np.asarray(dq, dtype=float)
        if dq.shape != (7,) or not np.isfinite(dq).all():
            raise ValueError('friction velocity must be a finite seven-vector')
        if not self.enabled:
            return np.zeros(7)
        return self.viscous*dq + self.coulomb*np.tanh(dq/self.smoothing)

    def metadata(self):
        metadata = {'enabled': self.enabled, 'source': self.source,
                    'viscous_nm_s_per_rad': self.viscous.tolist(),
                    'coulomb_nm': self.coulomb.tolist(),
                    'velocity_smoothing_rad_s': self.smoothing.tolist()}
        if self.urdf_path is not None:
            metadata['urdf_path'] = self.urdf_path
        return metadata


class PinocchioMomentumModel:
    def __init__(self, urdf_path, gravity=(0., 0., -9.81)):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError('Torque visualization requires pin>=3,<4') from exc
        self.pin = pin
        self.urdf_path = str(Path(urdf_path).resolve())
        self.model = pin.buildModelFromUrdf(self.urdf_path)
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


class ExternalTorqueMomentumObserver:
    """Observe measured-minus-rigid-body-minus-friction torque using q/dq.

    With p=M(q)dq and beta=C(q,dq).T dq-g(q), integrate
    p_hat_dot=measured_torque+beta-friction-r, r=K(p_hat-p).
    Thus r_dot=K(tau_residual-r): the observer itself has finite bandwidth,
    even when no additional signal filters are enabled. No ddq or RNEA call
    is needed. Trapezoidal integration uses actual sample intervals.
    """

    def __init__(self, model, bandwidth_hz=10., max_gap_s=.03):
        bandwidth = _seven_joint_parameter(bandwidth_hz, 'observer_bandwidth_hz', positive=True)
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('max_gap_s must be positive')
        self.model, self.gain, self.max_gap_s = model, 2*np.pi*bandwidth, float(max_gap_s)
        self.reset()

    def reset(self):
        self.timestamp_s = None
        self.p = self.s = None
        self.residual = np.zeros(7)
        self.latest = np.full(7, np.nan)

    def update(self, timestamp_s, q, dq, measured_torque, friction_torque=None, valid=True):
        q, dq, measured = (np.asarray(value, dtype=float) for value in (q, dq, measured_torque))
        friction = np.zeros(7) if friction_torque is None else np.asarray(friction_torque, dtype=float)
        if (not valid or not np.isfinite(timestamp_s)
                or any(value.shape != (7,) or not np.isfinite(value).all()
                       for value in (q, dq, measured, friction))):
            self.reset()
            return self.latest.copy()
        if self.timestamp_s is not None and timestamp_s == self.timestamp_s:
            return self.latest.copy()
        p, beta = (np.asarray(value, dtype=float) for value in self.model.terms(q, dq))
        if any(value.shape != (7,) or not np.isfinite(value).all() for value in (p, beta)):
            self.reset()
            return self.latest.copy()
        s = measured + beta - friction
        if not np.isfinite(s).all():
            self.reset()
            return self.latest.copy()
        dt = None if self.timestamp_s is None else timestamp_s-self.timestamp_s
        if dt is None or dt <= 0 or dt > self.max_gap_s:
            self.reset()
            self.timestamp_s, self.p, self.s = float(timestamp_s), p.copy(), s.copy()
            return self.latest.copy()
        half_step = self.gain*dt/2
        self.residual = ((1-half_step)*self.residual
                         + self.gain*(dt/2*(self.s+s) - (p-self.p))) / (1+half_step)
        self.timestamp_s, self.p, self.s = float(timestamp_s), p.copy(), s.copy()
        self.latest = self.residual.copy()
        return self.latest.copy()


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
