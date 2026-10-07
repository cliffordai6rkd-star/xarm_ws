"""Two-step q/tau reference shaping. No robot APIs, contact optimization or QP dependency."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class ControlState:
    timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray | None = None


@dataclass(frozen=True)
class TwoStepReference:
    timestamps_us: np.ndarray
    q: np.ndarray | None
    tau: np.ndarray | None
    sample_id: int = 0
    anchor_us: int = 0
    indices: tuple = ()
    tau_filtered: np.ndarray | None = None
    dq: np.ndarray | None = None
    ddq: np.ndarray | None = None


@dataclass
class ReferenceResult:
    q_raw: np.ndarray
    q_command: np.ndarray
    v_plan: np.ndarray
    a_plan: np.ndarray
    requested_mode: str
    actual_mode: str
    valid: bool
    reason: str
    solve_time_ms: float
    dynamics_residual: np.ndarray | None
    model_confidence: str
    solver_status: str = 'weighted_least_squares'
    mass_matrix: np.ndarray | None = None
    nonlinear_effects: np.ndarray | None = None
    bias: np.ndarray | None = None
    dq_reference: np.ndarray | None = None
    tau_feedback: np.ndarray | None = None
    tau_feedforward: np.ndarray | None = None
    tau_mit: np.ndarray | None = None
    delta_q_raw: np.ndarray | None = None
    integration_q0: np.ndarray | None = None
    integration_v0: np.ndarray | None = None
    acceleration_reference: np.ndarray | None = None
    acceleration_raw: np.ndarray | None = None
    tau_nominal: np.ndarray | None = None
    wm_tau_weight: np.ndarray | None = None
    tau_wm_effective: np.ndarray | None = None
    wm_acceleration_raw: np.ndarray | None = None

    @property
    def first_q_command(self):
        return self.q_command[0].copy()


def vector(value, size, name, *, nonnegative=False, positive=False):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array))
    if (array.shape != (size,) or not np.isfinite(array).all()
            or (nonnegative and np.any(array < 0)) or (positive and np.any(array <= 0))):
        raise ValueError(f'{name} must be a finite {size}-vector with the required sign')
    return array


class LocalDynamicsModel:
    """Thread-owned Pinocchio Data; b0 is explicit and independent of WM q."""
    def __init__(self, urdf, *, gravity=(0.,0.,-9.81), b0=0.):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        if self.model.nq != self.model.nv or self.model.nq != 7 or list(self.model.names[1:]) != [f'joint{i}' for i in range(1,8)]:
            raise ValueError('Dynamics requires joint1..joint7 and a fixed gripper')
        self.model.gravity.linear[:] = vector(gravity, 3, 'gravity')
        self.data = self.model.createData()
        self.bias = vector(b0, 7, 'b0')

    def snapshot(self, q, dq):
        raw = np.array(self.pin.crba(self.model, self.data, q), copy=True)
        mass = np.triu(raw) + np.triu(raw, 1).T
        if not np.isfinite(mass).all():
            raise ValueError('Nonfinite inertia matrix')
        np.linalg.cholesky(mass)
        nonlinear = np.array(self.pin.rnea(self.model, self.data, q, dq, np.zeros(7)), copy=True)
        return mass, nonlinear, self.bias.copy()


class TwoStepController:
    mode = 'joint_q_tau'

    def __init__(self, *, dt=.01, dynamics=None, eps_q=.01, eps_tau=1., rho=1.,
                 sigma_model=0., smooth_weight=0., previous_weight=0., eps_acceleration=4.):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('controller.dt must be positive')
        self.dt, self.dynamics = float(dt), dynamics
        self.q_weight = 1 / vector(eps_q,7,'eps_q',positive=True)**2
        self.tau_weight = vector(rho,7,'rho',nonnegative=True) / (
            vector(eps_tau,7,'eps_tau',positive=True)**2 + vector(sigma_model,7,'sigma_model',nonnegative=True)**2)
        scale = vector(eps_acceleration,7,'eps_acceleration',positive=True)**2
        self.smooth_weight = vector(smooth_weight,7,'smooth_weight',nonnegative=True)/scale
        self.previous_weight = vector(previous_weight,7,'previous_weight',nonnegative=True)/scale
        self.position_scale = self.torque_scale = 1.

    def information_usage(self):
        use_q = self.mode != 'tau_only' and self.position_scale > 0
        use_tau = self.mode != 'q_only' and self.torque_scale > 0
        if self.mode == 'joint_q_tau' and not np.any(self.tau_weight > 0):
            use_tau = False
        return use_q, use_tau

    def solve(self, state, reference, *, previous_acceleration=None):
        started = time.perf_counter()
        q0, v0 = vector(state.q,7,'q0'), vector(state.dq,7,'v0')
        h, eye = self.dt, np.eye(7)
        p = h*h*np.block([[.5*eye, np.zeros((7,7))], [1.5*eye,.5*eye]])
        offset = np.r_[q0+h*v0, q0+2*h*v0]
        rows, targets = [], []
        mass = nonlinear = bias = None
        use_q, use_tau = self.information_usage()
        if use_q:
            qref=np.asarray(reference.q,dtype=float)
            if qref.shape != (2,7) or not np.isfinite(qref).all():
                raise ValueError('q reference must be finite [2,7]')
            weight=np.tile(np.sqrt(self.q_weight*self.position_scale),2)
            rows.append(weight[:,None]*p); targets.append(weight*(qref.reshape(-1)-offset))
        if use_tau:
            if self.dynamics is None:
                raise ValueError('Torque shaping requires a dynamics model')
            if not np.any(self.tau_weight > 0):
                raise ValueError('Torque-only objective has no valid nonzero torque weight')
            tref=np.asarray(reference.tau,dtype=float)
            if tref.shape != (2,7) or not np.isfinite(tref).all():
                raise ValueError('Torque reference must be finite [2,7]')
            mass, nonlinear, bias = self.dynamics.snapshot(q0,v0)
            mass=np.asarray(mass,dtype=float)
            if mass.shape != (7,7) or not np.isfinite(mass).all() or not np.allclose(mass,mass.T,atol=1e-10):
                raise ValueError('Invalid inertia matrix')
            np.linalg.cholesky(mass)
            nonlinear, bias = vector(nonlinear,7,'n0'), vector(bias,7,'b0')
            t=np.block([[mass,np.zeros((7,7))],[np.zeros((7,7)),mass]])
            weight=np.tile(np.sqrt(self.tau_weight*self.torque_scale),2)
            rows.append(weight[:,None]*t)
            targets.append(weight*(tref.reshape(-1)-np.tile(nonlinear+bias,2)))
        if not rows:
            raise ValueError('Controller has no valid primary objective')
        if np.any(self.smooth_weight):
            rows.append(np.sqrt(self.smooth_weight)[:,None]*np.c_[-eye,eye]); targets.append(np.zeros(7))
        if np.any(self.previous_weight):
            previous = np.zeros(7) if previous_acceleration is None else vector(previous_acceleration,7,'a_prev_ref')
            rows.append(np.sqrt(self.previous_weight)[:,None]*np.c_[eye,np.zeros((7,7))])
            targets.append(np.sqrt(self.previous_weight)*previous)
        a, _, rank, _ = np.linalg.lstsq(np.vstack(rows),np.concatenate(targets),rcond=None)
        if rank != 14 or not np.isfinite(a).all():
            raise ValueError('Reference objective is rank deficient or nonfinite')
        accelerations=a.reshape(2,7)
        q=(offset+p@a).reshape(2,7)
        v=np.array([v0+h*accelerations[0],v0+h*(accelerations[0]+accelerations[1])])
        residual = None if not use_tau else accelerations@mass.T + nonlinear+bias-np.asarray(reference.tau)
        return ReferenceResult(q,q.copy(),v,accelerations,self.mode,self.mode,True,'ok',
            (time.perf_counter()-started)*1000,residual,'not_used' if not use_tau else 'uncalibrated_rigid_body',
            mass_matrix=mass,nonlinear_effects=nonlinear,bias=bias)


class QOnlyController(TwoStepController):
    """Pass time-aligned WM positions directly to the common command limiter."""
    mode = 'q_only'

    def solve(self, state, reference, *, previous_acceleration=None):
        started = time.perf_counter()
        if self.position_scale <= 0:
            raise ValueError('Direct WM q output requires a positive position scale')
        q0, v0 = vector(state.q,7,'q0'), vector(state.dq,7,'v0')
        q = np.asarray(reference.q,dtype=float)
        if q.shape != (2,7) or not np.isfinite(q).all():
            raise ValueError('q reference must be finite [2,7]')
        q = q.copy()
        # Finite differences are diagnostics only. No dynamics, torque,
        # least-squares objective or continuity penalty changes the WM target.
        velocities = np.diff(np.vstack([q0,q]),axis=0)/self.dt
        accelerations = np.diff(np.vstack([v0,velocities]),axis=0)/self.dt
        return ReferenceResult(q,q.copy(),velocities,accelerations,self.mode,self.mode,True,'ok',
            (time.perf_counter()-started)*1000,None,'not_used',solver_status='direct_wm_q')


class JointQTauController(TwoStepController):
    mode = 'joint_q_tau'


class MitPositionController(TwoStepController):
    """Host MIT torque -> forward dynamics -> integrated position references."""
    mode = 'mit_position'

    def __init__(self, *, mit, velocity_limit, acceleration_limit=None, jerk_limit=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(mit,dict):
            raise ValueError('controller.mit must be a mapping')
        for name,expected in (('reference_source','wm_q'),):
            if mit.get(name)!=expected:
                raise ValueError(f'controller.mit.{name} must be {expected}')
        self.velocity_source=mit.get('velocity_source')
        if self.velocity_source not in ('wm_difference','filtered_wm_difference','zero'):
            raise ValueError('controller.mit.velocity_source must be wm_difference, filtered_wm_difference or zero')
        self.feedback_space=mit.get('feedback_space','torque')
        if self.feedback_space not in ('torque','acceleration'):
            raise ValueError('controller.mit.feedback_space must be torque or acceleration')
        self.nominal_acceleration_source=mit.get('nominal_acceleration_source','wm_difference')
        if self.nominal_acceleration_source not in ('wm_difference','zero'):
            raise ValueError('controller.mit.nominal_acceleration_source must be wm_difference or zero')
        self.wm_residual_acceleration_limit=(None if mit.get('wm_residual_acceleration_limit_rad_s2') is None
            else vector(mit['wm_residual_acceleration_limit_rad_s2'],7,'WM residual acceleration limit',positive=True))
        self.feedforward_source=mit.get('feedforward_source')
        if self.feedforward_source not in ('wm_tau','nominal_wm_blend'):
            raise ValueError('controller.mit.feedforward_source must be wm_tau or nominal_wm_blend')
        if self.feedforward_source=='nominal_wm_blend' and 'wm_tau_weight' not in mit:
            raise ValueError('Nominal/WM fusion requires explicit mit.wm_tau_weight')
        self.wm_tau_weight=vector(mit.get('wm_tau_weight',1.),7,'mit.wm_tau_weight',nonnegative=True)
        if np.any(self.wm_tau_weight>1):
            raise ValueError('mit.wm_tau_weight must be in [0,1]')
        if self.feedforward_source=='wm_tau' and np.any(self.wm_tau_weight!=1):
            raise ValueError('wm_tau uses weight 1; select nominal_wm_blend to mix torques')
        self.integration_base=mit.get('integration_base')
        if self.integration_base not in ('measured','previous_command'):
            raise ValueError('controller.mit.integration_base must be measured or previous_command')
        self.feedback_state=mit.get('feedback_state','measured')
        if self.feedback_state not in ('measured','command'):
            raise ValueError('controller.mit.feedback_state must be measured or command')
        if 'kp' not in mit or 'kd' not in mit:
            raise ValueError('controller.mit requires explicit kp and kd gains')
        self.kp=vector(mit['kp'],7,'mit.kp',nonnegative=True)
        self.kd=vector(mit['kd'],7,'mit.kd',nonnegative=True)
        self.velocity_limit=vector(mit.get('reference_velocity_limit_rad_s',velocity_limit),7,
                                   'MIT desired velocity limit',positive=True)
        if np.any(self.velocity_limit > vector(velocity_limit,7,'command velocity limit',positive=True)):
            raise ValueError('MIT desired velocity limit cannot exceed command velocity limit')
        self.acceleration_limit=(None if acceleration_limit is None else
                                 vector(acceleration_limit,7,'MIT acceleration limit',positive=True))
        self.jerk_limit=(None if jerk_limit is None else
                        vector(jerk_limit,7,'MIT jerk limit',positive=True))
        if self.jerk_limit is not None and self.acceleration_limit is None:
            raise ValueError('MIT jerk limiting requires an acceleration limit')

    def information_usage(self):
        return True,True

    def solve(self,state,reference,*,previous_acceleration=None,integration_state=None):
        started=time.perf_counter()
        if self.dynamics is None:
            raise ValueError('MIT position conversion requires a dynamics model')
        q_measured=vector(state.q,7,'q0')
        v_measured=vector(state.dq,7,'v0')
        use_q,_=self.information_usage()
        arrays={'tau':np.asarray(reference.tau,dtype=float)}
        if use_q:
            arrays.update(q=np.asarray(reference.q,dtype=float),
                          dq=np.asarray(np.zeros((2,7)) if self.velocity_source=='zero'
                                        else reference.dq,dtype=float))
        for key,value in arrays.items():
            if value.shape!=(2,7) or not np.isfinite(value).all():
                raise ValueError(f'MIT {key} reference must be finite [2,7]')
        dq_reference=(np.zeros((2,7)) if not use_q or self.velocity_source=='zero' else
                      np.clip(arrays['dq'],-self.velocity_limit,self.velocity_limit))
        acceleration_reference=np.zeros((2,7))
        if use_q and self.feedforward_source=='nominal_wm_blend' and self.nominal_acceleration_source=='wm_difference':
            acceleration_reference=np.asarray(reference.ddq,dtype=float)
            if acceleration_reference.shape!=(2,7) or not np.isfinite(acceleration_reference).all():
                raise ValueError('MIT acceleration reference must be finite [2,7] for nominal/WM fusion')
            if self.acceleration_limit is not None:
                acceleration_reference=np.clip(acceleration_reference,-self.acceleration_limit,self.acceleration_limit)
        base=state if self.integration_base=='measured' or integration_state is None else integration_state
        q_base=vector(base.q,7,'integration q0').copy()
        v_base=vector(base.dq,7,'integration v0').copy()
        q_initial,v_initial=q_base.copy(),v_base.copy()
        h=self.dt
        positions=[];velocities=[];accelerations=[];increments=[]
        feedbacks=[];torques=[];masses=[];nonlinear=[];biases=[];residuals=[]
        raw_accelerations=[];nominal_torques=[];feedforward_torques=[]
        effective_wm_torques=[];wm_accelerations=[]
        previous_a=(np.zeros(7) if previous_acceleration is None else
                    vector(previous_acceleration,7,'previous MIT acceleration').copy())
        q_feedback,v_feedback=(q_measured,v_measured) if self.feedback_state=='measured' else (q_base,v_base)
        for i in range(2):
            feedback=(self.kp*(arrays['q'][i]-q_feedback)+self.kd*(dq_reference[i]-v_feedback)
                      if use_q else np.zeros(7))
            mass,n,b=self.dynamics.snapshot(q_feedback,v_feedback)
            mass=np.asarray(mass,dtype=float)
            if (mass.shape!=(7,7) or not np.isfinite(mass).all()
                    or not np.allclose(mass,mass.T,atol=1e-10)):
                raise ValueError('Invalid MIT inertia matrix')
            np.linalg.cholesky(mass)
            n,b=vector(n,7,'n'),vector(b,7,'b')
            if self.feedback_space=='acceleration':
                # Full matrix conversion gives explicit acceleration-space
                # tracking gains rather than amplifying wrist errors by 1/Mjj.
                feedback=mass@feedback
            nominal=mass@acceleration_reference[i]+n+b
            wm_acceleration=np.linalg.solve(mass,arrays['tau'][i]-nominal)
            if not np.isfinite(wm_acceleration).all():
                raise ValueError('WM torque conversion returned nonfinite acceleration')
            effective_wm=arrays['tau'][i]
            if self.wm_residual_acceleration_limit is not None:
                correction=np.clip(wm_acceleration,-self.wm_residual_acceleration_limit,
                                   self.wm_residual_acceleration_limit)
                effective_wm=nominal+mass@correction
            feedforward=(effective_wm if self.feedforward_source=='wm_tau' else
                         (1-self.wm_tau_weight)*nominal+self.wm_tau_weight*effective_wm)
            torque=feedback+feedforward
            raw_acceleration=np.linalg.solve(mass,torque-n-b)
            if not np.isfinite(raw_acceleration).all():
                raise ValueError('MIT forward dynamics returned nonfinite acceleration')
            acceleration=raw_acceleration.copy()
            if self.acceleration_limit is not None:
                low,high=-self.acceleration_limit,self.acceleration_limit
                if self.jerk_limit is not None:
                    low=np.maximum(low,previous_a-self.jerk_limit*h)
                    high=np.minimum(high,previous_a+self.jerk_limit*h)
                if np.any(low>high):
                    raise ValueError('MIT acceleration/jerk limits have no feasible intersection')
                acceleration=np.clip(acceleration,low,high)
            increment=h*v_base+.5*h*h*acceleration
            q_next=q_base+increment
            v_next=v_base+h*acceleration
            if not np.isfinite(np.r_[torque,acceleration,increment,q_next,v_next]).all():
                raise ValueError('MIT integration returned nonfinite values')
            positions.append(q_next);velocities.append(v_next);accelerations.append(acceleration)
            increments.append(increment);feedbacks.append(feedback);torques.append(torque)
            raw_accelerations.append(raw_acceleration);nominal_torques.append(nominal)
            feedforward_torques.append(feedforward)
            effective_wm_torques.append(effective_wm);wm_accelerations.append(wm_acceleration)
            masses.append(mass.copy());nonlinear.append(n.copy());biases.append(b.copy())
            # Residual refers to the unshaped dynamics solution. Command residuals
            # are logged separately; limiting changes the torque equivalence.
            residuals.append(mass@raw_acceleration+n+b-torque)
            previous_a=acceleration
            q_base,v_base=q_next,v_next
            # Only the second diagnostic candidate uses a simulated state;
            # next live cycle restarts its MIT feedback from measured q/dq.
            q_feedback,v_feedback=q_next,v_next
        q=np.asarray(positions)
        return ReferenceResult(q,q.copy(),np.asarray(velocities),np.asarray(accelerations),
            self.mode,self.mode,True,'ok',(time.perf_counter()-started)*1000,
            np.asarray(residuals),'uncalibrated_rigid_body',solver_status='mit_forward_dynamics',
            mass_matrix=np.asarray(masses),nonlinear_effects=np.asarray(nonlinear),bias=np.asarray(biases),
            dq_reference=dq_reference,tau_feedback=np.asarray(feedbacks),tau_feedforward=np.asarray(feedforward_torques),
            tau_mit=np.asarray(torques),delta_q_raw=np.asarray(increments),
            integration_q0=q_initial,integration_v0=v_initial,
            acceleration_reference=acceleration_reference,acceleration_raw=np.asarray(raw_accelerations),
            tau_nominal=np.asarray(nominal_torques),wm_tau_weight=self.wm_tau_weight.copy(),
            tau_wm_effective=np.asarray(effective_wm_torques),wm_acceleration_raw=np.asarray(wm_accelerations))


class TauOnlyController(MitPositionController):
    """MIT torque-to-position conversion with no WM q/dq/ddq tracking."""
    mode = 'tau_only'

    def __init__(self, *, mit=None, velocity_limit=1., **kwargs):
        if mit is not None and not isinstance(mit,dict):
            raise ValueError('controller.mit must be a mapping')
        options=dict(feedforward_source='wm_tau',integration_base='measured',feedback_state='measured')
        options.update({} if mit is None else mit)
        # Only the torque limits and integration settings carry over.
        # q/dq/ddq are neither required nor synthesized for torque-only output.
        options.update(reference_source='wm_q',velocity_source='zero',
                       nominal_acceleration_source='zero',kp=0.,kd=0.,
                       feedforward_source='wm_tau',wm_tau_weight=1.)
        options.pop('reference_velocity_limit_rad_s',None)
        super().__init__(mit=options,velocity_limit=velocity_limit,**kwargs)

    def information_usage(self):
        if self.torque_scale != 1.:
            raise ValueError('tau_only fixes WM torque weight to 1; torque_scale must remain 1')
        return False,True


class CommandLimitError(ValueError):
    """An infeasible command with the exact joint/step bounds for logging."""
    def __init__(self, message, details):
        super().__init__(message)
        self.details=details


class ReferenceLimits:
    """Common position-command projection; virtual history in shadow, never sent history."""
    def __init__(self, urdf, settings, dt):
        root=ET.parse(urdf).getroot()
        bounds=[]
        for name in (f'joint{i}' for i in range(1,8)):
            joint=next(j for j in root.findall('joint') if j.get('name')==name)
            limit=joint.find('limit'); bounds.append([float(limit.get('lower')),float(limit.get('upper'))])
        self.bounds=np.asarray(bounds)
        self.vmax=vector(settings['max_velocity_rad_s'],7,'velocity',positive=True)
        self.amax=vector(settings['max_acceleration_rad_s2'],7,'acceleration',positive=True)
        self.step=vector(settings['max_step_rad'],7,'step',positive=True)
        self.lead=vector(settings['max_tracking_error_rad'],7,'leading amount',positive=True)
        self.jmax=(None if settings.get('max_jerk_rad_s3') is None else
                   vector(settings['max_jerk_rad_s3'],7,'jerk',positive=True))
        self.q_only_smoothing=settings.get('q_only_smoothing','legacy')
        if self.q_only_smoothing not in ('legacy','jerk_tracking'):
            raise ValueError('q_only_smoothing must be legacy or jerk_tracking')
        self.q_only_kp=vector(settings.get('q_only_position_gain_s2',16.),7,'q_only position gain',positive=True)
        self.q_only_kd=vector(settings.get('q_only_velocity_gain_s',2*np.sqrt(self.q_only_kp)),
                              7,'q_only velocity gain',positive=True)
        self.dt=dt
        self.reset()

    def reset(self):
        self.previous_q=self.previous_v=None
        self.previous_acceleration=np.zeros(7)

    def project(self, result, state):
        if self.jmax is not None and (result.actual_mode in ('mit_position','tau_only') or
                (result.actual_mode=='q_only' and self.q_only_smoothing=='jerk_tracking')):
            return self._project_with_jerk(result,state)
        q=vector(state.q,7,'q0') if self.previous_q is None else self.previous_q.copy()
        v=vector(state.dq,7,'dq0') if self.previous_v is None else self.previous_v.copy()
        commands=[]; velocities=[]; accelerations=[]
        for target in result.q_raw:
            low=np.maximum.reduce([self.bounds[:,0], q-self.step, q-self.vmax*self.dt,
                q+self.dt*(v-self.amax*self.dt), np.asarray(state.q)-self.lead])
            high=np.minimum.reduce([self.bounds[:,1], q+self.step, q+self.vmax*self.dt,
                q+self.dt*(v+self.amax*self.dt), np.asarray(state.q)+self.lead])
            if np.any(low>high):
                raise ValueError('Common position-command limits have no feasible intersection')
            command=np.clip(target,low,high)
            velocity=(command-q)/self.dt
            acceleration=(velocity-v)/self.dt
            commands.append(command); velocities.append(velocity); accelerations.append(acceleration)
            q,v=command,velocity
        result.q_command=np.array(commands)
        # Only the first candidate advances shadow history. This is not hardware state.
        self.previous_q=commands[0].copy(); self.previous_v=velocities[0].copy()
        self.previous_acceleration=accelerations[0].copy()
        return {'limited':bool(np.max(np.abs(result.q_command-result.q_raw))>1e-12),
                'command_v_plan':np.array(velocities),'command_a_plan':np.array(accelerations)}

    def _project_with_jerk(self,result,state):
        """Project acceleration before constant-acceleration integration.

        Command velocity is the endpoint velocity, rather than delta_q/dt.
        Every constraint shares this convention, including next-cycle history.
        """
        q=vector(state.q,7,'q0') if self.previous_q is None else self.previous_q.copy()
        v=vector(state.dq,7,'dq0') if self.previous_v is None else self.previous_v.copy()
        a_previous=self.previous_acceleration.copy()
        h=self.dt
        commands=[];velocities=[];accelerations=[];jerks=[]
        tracking_targets=[]
        for step_index,target in enumerate(result.q_raw):
            position_low=np.maximum.reduce([self.bounds[:,0],q-self.step,np.asarray(state.q)-self.lead])
            position_high=np.minimum.reduce([self.bounds[:,1],q+self.step,np.asarray(state.q)+self.lead])
            # Follow the solver's POSITION target from the last command state.
            # Reusing a_plan here silently turned integration_base=measured into
            # an accumulating command-state integrator and discarded feedback
            # corrections embedded in q_raw.
            if result.actual_mode=='q_only':
                # A finite, critically damped reference tracker brakes before
                # a position target. The one-tick inversion would demand huge
                # acceleration until crossing the target, then jerk-limited
                # reversal can overshoot and sustain a bang-bang oscillation.
                # A far WM target must not keep accelerating through the
                # measured tracking envelope. Bound the tracker's target
                # before feedback, so damping can brake ahead of that edge.
                tracking_target=np.clip(target,position_low,position_high)
                tracking_targets.append(tracking_target.copy())
                desired_a=self.q_only_kp*(tracking_target-q)-self.q_only_kd*v
            else:
                desired_a=2*(target-q-h*v)/(h*h)
            # Reserve the velocity needed to ramp acceleration back to zero.
            # Clipping velocity only at this step can leave the next step with
            # no feasible intersection between velocity and jerk constraints.
            acceleration_to_vmax=self._braking_acceleration(self.vmax-v)
            acceleration_to_vmin=self._braking_acceleration(self.vmax+v)
            low_parts=np.asarray([-self.amax,-acceleration_to_vmin,a_previous-self.jmax*h,
                (-self.vmax-v)/h,2*(position_low-q-h*v)/(h*h)])
            high_parts=np.asarray([self.amax,acceleration_to_vmax,a_previous+self.jmax*h,
                (self.vmax-v)/h,2*(position_high-q-h*v)/(h*h)])
            low=np.max(low_parts,axis=0)
            high=np.min(high_parts,axis=0)
            if np.any(low-high>1e-10):
                failed=np.flatnonzero(low-high>1e-10)
                names=('acceleration','velocity_braking','jerk','velocity','position/tracking')
                lower_sources=[names[i] for i in np.argmax(low_parts,axis=0)]
                upper_sources=[names[i] for i in np.argmin(high_parts,axis=0)]
                reason='; '.join(f'J{j+1} step {step_index+1}: '
                    f'lower {low[j]:.6g} ({lower_sources[j]}) > '
                    f'upper {high[j]:.6g} ({upper_sources[j]})' for j in failed)
                raise CommandLimitError(
                    'Position/velocity/acceleration/jerk limits have no feasible intersection; '+reason,
                    dict(limit_failure_step=step_index+1,limit_failure_joints=(failed+1).tolist(),
                         limit_acceleration_low=low.copy(),limit_acceleration_high=high.copy(),
                         limit_lower_sources=lower_sources,limit_upper_sources=upper_sources,
                         limit_q_base=q.copy(),limit_v_base=v.copy(),limit_a_base=a_previous.copy(),
                         limit_position_low=position_low.copy(),limit_position_high=position_high.copy()))
            # Braking and jerk bounds can meet within floating-point roundoff.
            low=np.minimum(low,high)
            acceleration=np.clip(desired_a,low,high)
            command=q+h*v+.5*h*h*acceleration
            velocity=v+h*acceleration
            commands.append(command);velocities.append(velocity);accelerations.append(acceleration)
            jerks.append((acceleration-a_previous)/h)
            q,v,a_previous=command,velocity,acceleration
        result.q_command=np.asarray(commands)
        self.previous_q=commands[0].copy();self.previous_v=velocities[0].copy()
        self.previous_acceleration=accelerations[0].copy()
        details={'limited':bool(np.max(np.abs(result.q_command-result.q_raw))>1e-12
                    or (result.acceleration_raw is not None
                        and np.max(np.abs(result.a_plan-result.acceleration_raw))>1e-12)),
                'command_v_plan':np.asarray(velocities),'command_a_plan':np.asarray(accelerations),
                'command_jerk_plan':np.asarray(jerks)}
        if result.actual_mode=='q_only':
            details['q_tracking_target']=np.asarray(tracking_targets)
            details['q_tracking_target_limited']=bool(np.max(np.abs(result.q_raw-details['q_tracking_target']))>1e-12)
        if result.tau_mit is not None:
            details['command_dynamics_residual']=(np.einsum('ijk,ik->ij',result.mass_matrix,accelerations)
                +result.nonlinear_effects+result.bias-result.tau_mit)
        return details

    def _braking_acceleration(self,velocity_room):
        """Maximum positive acceleration that can settle under discrete jerk."""
        room=np.maximum(velocity_room,0.)/self.dt
        step=self.jmax*self.dt
        count=np.floor((np.sqrt(1+8*room/step)-1)*.5)
        return np.minimum(self.amax,room/(count+1)+step*count*.5)
