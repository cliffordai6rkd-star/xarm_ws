"""Common WM forecast -> two-step position reference layer, independent of the action policy."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace
import time

import numpy as np
from scipy.signal import savgol_filter

from inference.controllers.wm_two_step import (
    ControlState, TwoStepReference, QOnlyController, TauOnlyController,
    JointQTauController, MitPositionController, LocalDynamicsModel, ReferenceLimits, CommandLimitError,
)


class WmReferenceAligner:
    def __init__(self, *, dt=.01, sample_id=0, torque_reference='raw', cutoff_hz=20.):
        if torque_reference not in ('raw','causal_inverse'):
            raise ValueError('Unknown torque reference preprocessing')
        if type(sample_id) is not int or sample_id < 0 or dt <= 0 or cutoff_hz <= 0:
            raise ValueError('Invalid reference alignment parameters')
        self.dt, self.sample_id = dt, sample_id
        self.torque_reference, self.cutoff_hz = torque_reference, cutoff_hz

    @staticmethod
    def interpolate(times, values, targets):
        if (targets[0]<times[0] or targets[-1]>times[-1] or not np.isfinite(values).all()):
            raise ValueError('Forecast does not cover the requested times or contains invalid values')
        # Subtract integer timestamps first, preserving microsecond precision.
        relative=times-times[0]; requested=targets-times[0]
        return np.array([np.interp(requested,relative,values[:,j]) for j in range(7)]).T

    def align(self, forecast, state, *, use_q=True, use_tau=True, use_dq=False,
              velocity_method='raw'):
        times=np.asarray(forecast['future_timestamps_us'])
        if times.ndim != 1 or times.dtype.kind not in 'iu' or len(times)<2 or np.any(np.diff(times)<=0):
            raise ValueError('Forecast timestamps must be increasing integer microseconds')
        if np.any(np.abs(np.diff(times)-10_000)>4_000):
            raise ValueError('Forecast must use the 100 Hz physical prediction grid')
        targets=int(state.timestamp_us)+np.array([1,2],dtype=np.int64)*round(self.dt*1e6)
        q=tau=filtered=dq=ddq=None
        def values(key):
            value=np.asarray(forecast[key])
            if value.ndim != 3 or value.shape[1:] != (len(times),7) or self.sample_id>=value.shape[0]:
                raise ValueError(f'{key} must be [samples,time,7] with the selected sample available')
            return value[self.sample_id]
        if use_q:
            q=self.interpolate(times,values('q'),targets)
        if use_dq:
            relative_seconds=(times-times[0])/1e6
            if velocity_method == 'smoothed':
                if len(times) < 9 or np.any(np.abs(np.diff(times)-round(self.dt*1e6))>100):
                    raise ValueError('Smoothed WM velocity requires at least nine regular 100 Hz samples')
                derivative=savgol_filter(values('q'),9,1,deriv=1,delta=self.dt,axis=0,mode='interp')
            elif velocity_method == 'raw':
                derivative=np.gradient(values('q'),relative_seconds,axis=0)
            else:
                raise ValueError('Unknown WM velocity method')
            dq=self.interpolate(times,derivative,targets)
            ddq=self.interpolate(times,np.gradient(derivative,relative_seconds,axis=0),targets)
        if use_tau:
            data=values('tau')
            if self.torque_reference=='causal_inverse':
                extended=times
                if 'anchor_tau_filtered' in forecast:
                    anchor=int(forecast['anchor_us'])
                    if anchor >= times[0]:
                        raise ValueError('Invalid filtered torque anchor')
                    extended=np.r_[anchor,times]
                    data=np.vstack([np.asarray(forecast['anchor_tau_filtered']),data])
                points=self.interpolate(extended,data,np.r_[int(state.timestamp_us),targets])
                filtered=points[1:].copy()
                retention=np.exp(-2*np.pi*self.cutoff_hz*self.dt)
                tau=(points[1:]-retention*points[:-1])/(1-retention)
            else:
                tau=self.interpolate(times,data,targets)
                filtered=tau.copy()
        indices=tuple(np.searchsorted(times,targets,side='left').tolist())
        return TwoStepReference(targets,q,tau,self.sample_id,int(forecast['anchor_us']),indices,filtered,dq,ddq)


class WmControlLayer:
    """Reference computation without an arm handle; active history needs send acknowledgment."""
    def __init__(self, settings, urdf, *, allow_active=False):
        validate_settings(settings, allow_active=allow_active)
        self.settings=dict(settings)
        if settings['execution'] not in ('offline','shadow') and not allow_active:
            raise ValueError('WM control layer supports offline/shadow only; active sending is not connected')
        classes={'q_only':QOnlyController,'tau_only':TauOnlyController,'joint_q_tau':JointQTauController,
                 'mit_position':MitPositionController}
        if settings['mode'] not in classes or settings.get('horizon_steps',2)!=2:
            raise ValueError('Select a supported WM controller with horizon_steps=2')
        self.dt=float(settings['dt'])
        if self.dt != .01:
            raise ValueError('Current WM control bridge uses the checkpoint 0.01 s state timestep')
        dynamics=None
        d=settings['dynamics']
        if d.get('b_mode','zero') not in ('zero','configured'):
            raise ValueError('No calibrated/online contact bias provider is connected')
        if settings['mode']!='q_only':
            dynamics=LocalDynamicsModel(urdf,gravity=d.get('gravity_m_s2',[0,0,-9.81]),
                b0=0. if d.get('b_mode','zero')=='zero' else d['b0_nm'])
        self.torque_position=settings['mode'] in ('mit_position','tau_only')
        extra=({'mit':settings.get('mit'),'velocity_limit':settings['limits']['max_velocity_rad_s']}
               if self.torque_position else {})
        if self.torque_position and settings['limits'].get('max_jerk_rad_s3') is not None:
            extra.update(acceleration_limit=settings['limits']['max_acceleration_rad_s2'],
                         jerk_limit=settings['limits']['max_jerk_rad_s3'])
        self.controller=classes[settings['mode']](dt=self.dt,dynamics=dynamics,**settings['weights'],**extra)
        self.controller.position_scale=float(settings.get('position_scale',1.))
        self.controller.torque_scale=float(settings.get('torque_scale',1.))
        if (not np.isfinite([self.controller.position_scale,self.controller.torque_scale]).all()
                or min(self.controller.position_scale,self.controller.torque_scale)<0):
            raise ValueError('Primary objective scales must be finite and nonnegative')
        self.aligner=WmReferenceAligner(dt=self.dt,sample_id=settings.get('sample_id',0),
            torque_reference=settings['torque_reference'],cutoff_hz=settings.get('tau_filter_cutoff_hz',20.))
        self.limits=ReferenceLimits(urdf,settings['limits'],self.dt)
        self.command_hz = settings.get('command_hz', 100)
        self.step_us = round(self.dt*1e6)
        self.period_us = round(1e6/self.command_hz)
        self.command_index = round(self.period_us/self.step_us)-1
        self.counts=Counter()
        self.transition = None
        transition = settings.get('forecast_transition', {})
        if transition.get('enabled', False):
            from inference.core.forecast_transition import ForecastTransition
            self.transition = ForecastTransition(duration_ms=transition['duration_ms'],
                sample_id=self.aligner.sample_id, maximum_age_ms=settings['maximum_forecast_age_ms'], dt=self.dt)
        self.reset()

    def reset(self):
        self.forecast=None
        self.next_us=None
        self.last_state_us=-1
        self.last_stale_state_us=-1
        self.last_repeated_state_us=-1
        self.last_reason='waiting for a forecast and measured state'
        self.latest_result=None
        self.hold_started_us=None
        self.hold_target_q=None
        self.limits.reset()
        if self.transition is not None:
            self.transition.reset()

    def update_forecast(self, forecast):
        # Result arrays belong to this immutable snapshot, never reused by the worker.
        self.forecast={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in forecast.items()}
        if self.transition is not None:
            self.transition.update(self.forecast)

    def poll(self, now_us, state=None):
        if self.next_us is not None and now_us<self.next_us:
            return None
        if self.forecast is None or state is None:
            return None
        if state.timestamp_us==self.last_state_us:
            if state.timestamp_us!=self.last_repeated_state_us:
                self.counts['repeated_state_skips']+=1
                self.last_repeated_state_us=state.timestamp_us
            return None
        age=now_us-state.timestamp_us
        if (state.timestamp_us>self.last_state_us and age>=0
                and (age>self.settings['maximum_state_age_ms']*1000 or age>=self.step_us)):
            # Feedback acquisition and this loop have independent phases.
            # Do not consume a control slot or clear sent-command history
            # while waiting for the next fresh sample. ActiveOutput's
            # independent command-gap watchdog still enforces the 30 ms stop.
            if state.timestamp_us!=self.last_stale_state_us:
                self.counts['stale_state_skips']+=1
                self.last_stale_state_us=state.timestamp_us
            self.last_reason='waiting for fresh measured state'
            return None
        # Do not start work at the very end of a feedback sample's lifetime.
        # The command still has to pass computation and the shared read/send
        # lock. Waiting here preserves the pending control slot and sent q/v/a.
        remaining_us = min(self.settings['maximum_state_age_ms']*1000, self.step_us)-age
        reserve_us = self.settings.get('minimum_state_remaining_ms', 0.)*1000
        if self.settings['execution'] == 'active' and age >= 0 and remaining_us < reserve_us:
            if state.timestamp_us != self.last_stale_state_us:
                self.counts['state_send_budget_skips'] += 1
                self.last_stale_state_us = state.timestamp_us
            self.last_reason = 'waiting for feedback with enough compute/send budget'
            return None
        if self.next_us is None:
            self.next_us=now_us
        self.next_us+=((now_us-self.next_us)//self.period_us+1)*self.period_us
        time_reversed=state.timestamp_us<self.last_state_us
        self.last_state_us=state.timestamp_us
        started=time.perf_counter()
        event={'kind':'controller','execution':self.settings['execution'],
            'requested_mode':self.settings['mode'],'actual_mode':self.settings['mode'],
            'state_timestamp_us':int(state.timestamp_us),'computed_at_us':int(now_us),
            'anchor_us':int(self.forecast['anchor_us']),'sample_id':self.aligner.sample_id,
            'command_hz':self.command_hz,'command_index':self.command_index,
            'plan_version':self.forecast.get('plan_version'), 'sent':False,
            'plan_superseded':bool(self.forecast.get('plan_superseded',False))}
        result=None
        try:
            if time_reversed:
                raise ValueError('Measured state timestamp moved backwards')
            if state.timestamp_us < event['anchor_us']:
                raise ValueError('Measured state precedes the WM request anchor')
            event['state_age_ms']=age/1000
            event['forecast_age_ms']=(state.timestamp_us-self.forecast['anchor_us'])/1000
            event['maximum_forecast_age_ms']=self.settings['maximum_forecast_age_ms']
            if not 0<=age<=self.settings['maximum_state_age_ms']*1000 or age>=self.step_us:
                raise ValueError('Measured state is stale/future for the first physical prediction step')
            if event['forecast_age_ms']>self.settings['maximum_forecast_age_ms']:
                if self.settings.get('gap_hold', {}).get('enabled', False):
                    return self._hold_position(event, state, now_us, started)
                raise ValueError('WM reference exceeded the configured age')
            use_q,use_tau=self.controller.information_usage()
            use_dq=(self.settings['mode']=='mit_position' and
                (self.controller.velocity_source in ('wm_difference','filtered_wm_difference') or
                 (self.controller.feedforward_source=='nominal_wm_blend'
                  and self.controller.nominal_acceleration_source=='wm_difference')))
            reference=self.aligner.align(self.forecast,state,use_q=use_q,use_tau=use_tau,
                                        use_dq=use_dq,
                                        velocity_method=('smoothed' if self.settings['mode']=='mit_position'
                                            and self.controller.velocity_source=='filtered_wm_difference' else 'raw'))
            if self.transition is not None:
                previous_switches = self.transition.switch_count
                reference, transition_details = self.transition.apply(reference, command_index=self.command_index)
                event.update(transition_details)
                self.counts['forecast_handover_swaps'] += self.transition.switch_count-previous_switches
                self.counts['forecast_transition_ticks'] += int(transition_details.get('forecast_transition_active', False))
            solve_options={'previous_acceleration':self.limits.previous_acceleration}
            if self.torque_position and self.limits.previous_q is not None:
                solve_options['integration_state']=ControlState(int(state.timestamp_us),
                    self.limits.previous_q.copy(),self.limits.previous_v.copy())
            result=self.controller.solve(state,reference,**solve_options)
            elapsed=(time.perf_counter()-started)*1000
            if elapsed>self.settings['maximum_compute_ms']:
                raise ValueError('Controller computation missed its deadline')
            previous = (self.limits.previous_q, self.limits.previous_v, self.limits.previous_acceleration)
            projection=self.limits.project(result,state)
            if self.settings['execution'] == 'active':
                # Only a successful hardware send may advance active command history.
                self.limits.previous_q, self.limits.previous_v, self.limits.previous_acceleration = previous
            elif self.command_index:
                self.limits.previous_q = result.q_command[self.command_index].copy()
                self.limits.previous_v = projection['command_v_plan'][self.command_index].copy()
                self.limits.previous_acceleration = projection['command_a_plan'][self.command_index].copy()
            if (time.perf_counter()-started)*1000>self.settings['maximum_compute_ms']:
                raise ValueError('Controller/reference projection missed its deadline')
            event.update(asdict(result),**projection, target_timestamps_us=reference.timestamps_us,
                first_q_command=result.q_command[self.command_index].copy(),
                command_target_timestamp_us=int(reference.timestamps_us[self.command_index]),
                reference_indices=list(reference.indices), q_reference=reference.q, tau_reference=reference.tau,
                tau_reference_filtered=reference.tau_filtered, q0=np.array(state.q),v0=np.array(state.dq),
                tau_measured=None if state.tau is None else np.array(state.tau),
                history_kind=('pending_send' if self.settings['execution']=='active' else 'virtual_shadow_reference'),
                compute_ms=(time.perf_counter()-started)*1000)
            if use_tau:
                # The model snapshot already used by the solver is logged without another RNEA call.
                event['b_mode']=self.settings['dynamics'].get('b_mode','zero')
            if self.torque_position:
                event['feedforward_source']=self.controller.feedforward_source
                event['feedback_space']=self.controller.feedback_space
                event['velocity_source']=self.controller.velocity_source
                event['nominal_acceleration_source']=self.controller.nominal_acceleration_source
                event['integration_base']=self.controller.integration_base
                event['feedback_state']=self.controller.feedback_state
                event['wm_tracking_enabled']=use_q
                event['integration_history_kind']=('measured' if self.controller.integration_base=='measured'
                    or previous[0] is None else 'sent_reference' if self.settings['execution']=='active'
                    else 'virtual_shadow_reference')
                event['delta_q_command']=event['first_q_command']-result.integration_q0
            self.counts['valid']+=1
            self.last_reason='ok'
            self.hold_started_us=self.hold_target_q=None
        except (ValueError,np.linalg.LinAlgError,KeyError) as exc:
            event.update(q0=np.array(state.q),v0=np.array(state.dq))
            if result is not None:
                event['q_raw']=result.q_raw.copy()
            if isinstance(exc,CommandLimitError):
                event.update(exc.details)
            event.update(valid=False,reason=str(exc),compute_ms=(time.perf_counter()-started)*1000)
            self.last_reason=str(exc)
            self.counts['invalid']+=1
            self.hold_started_us=self.hold_target_q=None
            self.limits.reset()
            if self.transition is not None:
                self.transition.reset()
        self.latest_result=event
        return event

    def _hold_position(self, event, state, now_us, started):
        """Brake toward the last sent target during a bounded planned DP gap."""
        previous_q=self.limits.previous_q
        if previous_q is None:
            raise ValueError('No sent position available for DP gap hold')
        if self.hold_started_us is None:
            self.hold_started_us=int(now_us)
            self.hold_target_q=previous_q.copy()
        held_ms=(now_us-self.hold_started_us)/1000
        if held_ms>self.settings['gap_hold']['maximum_ms']:
            raise ValueError('DP gap hold exceeded its configured duration')
        if np.any(np.abs(previous_q-np.asarray(state.q))>self.limits.lead):
            raise ValueError('DP gap hold exceeded measured tracking limits')
        raw=np.repeat(self.hold_target_q[None],2,axis=0)
        result=SimpleNamespace(q_raw=raw,actual_mode=self.settings['mode'],
                               acceleration_raw=None,tau_mit=None)
        previous=(self.limits.previous_q,self.limits.previous_v,self.limits.previous_acceleration)
        projection=self.limits.project(result,state)
        if self.settings['execution']=='active':
            self.limits.previous_q,self.limits.previous_v,self.limits.previous_acceleration=previous
        elif self.command_index:
            self.limits.previous_q = result.q_command[self.command_index].copy()
            self.limits.previous_v = projection['command_v_plan'][self.command_index].copy()
            self.limits.previous_acceleration = projection['command_a_plan'][self.command_index].copy()
        elapsed=(time.perf_counter()-started)*1000
        if elapsed>self.settings['maximum_compute_ms']:
            raise ValueError('DP gap hold missed its computation deadline')
        targets=int(state.timestamp_us)+np.array([1,2],dtype=np.int64)*self.step_us
        event.update(valid=True,reason='bounded DP gap position hold',actual_mode='position_hold',
                     hold_age_ms=held_ms,hold_target_q=self.hold_target_q.copy(),
                     q_raw=raw,q_command=result.q_command.copy(),
                     first_q_command=result.q_command[self.command_index].copy(),
                     command_target_timestamp_us=int(targets[self.command_index]),
                     target_timestamps_us=targets,q_reference=raw.copy(),
                     q0=np.asarray(state.q).copy(),v0=np.asarray(state.dq).copy(),
                     tau_measured=None if state.tau is None else np.asarray(state.tau).copy(),
                     history_kind=('pending_send' if self.settings['execution']=='active'
                                   else 'virtual_shadow_reference'),compute_ms=elapsed,**projection)
        self.counts['valid']+=1
        self.counts['hold']+=1
        self.last_reason='bounded DP gap position hold'
        self.latest_result=event
        return event

    def report(self):
        return {'mode':self.settings['mode'],'execution':self.settings['execution'],
                'target_hz':self.command_hz,'prediction_step_s':self.dt,
                'command_index':self.command_index,'counts':dict(self.counts),'last_reason':self.last_reason,
                'minimum_state_remaining_ms':self.settings.get('minimum_state_remaining_ms', 0.),
                'maximum_forecast_age_ms':self.settings['maximum_forecast_age_ms'],
                'forecast_age_ms':None if self.latest_result is None else self.latest_result.get('forecast_age_ms'),
                'forecast_transition':dict(enabled=self.transition is not None,
                    duration_ms=None if self.transition is None else self.transition.duration_us/1000,
                    last_reason=None if self.transition is None else self.transition.reason)}

    def mark_sent(self, event):
        if self.settings['execution'] != 'active' or not event.get('sent') or not event.get('valid'):
            raise ValueError('Only successful active commands may advance command history')
        index = event.get('command_index', 0)
        if type(index) is not int or index not in (0, 1) or index != self.command_index:
            raise ValueError('Invalid acknowledged command_index for configured command rate')
        self.limits.previous_q = event['first_q_command'].copy()
        self.limits.previous_v = event['command_v_plan'][index].copy()
        self.limits.previous_acceleration = event['command_a_plan'][index].copy()
        event['history_kind'] = 'sent_reference'
        self.counts['sent'] += 1


def validate_settings(settings, *, allow_active=False):
    """Pure configuration validation; no models, hardware or solver construction."""
    if settings.get('execution') not in (('offline','shadow','active') if allow_active else ('offline','shadow')):
        raise ValueError('controller.execution must be offline or shadow; active is not connected')
    if settings.get('mode') not in ('q_only','tau_only','joint_q_tau','mit_position') or settings.get('horizon_steps')!=2:
        raise ValueError('Invalid WM controller mode/horizon')
    if settings.get('dt')!=.01:
        raise ValueError('controller.dt must be 0.01 seconds for this checkpoint')
    rate = settings.get('command_hz', 100)
    if type(rate) not in (int, float) or rate not in (50, 100):
        raise ValueError('controller.command_hz must be 50 or 100; WM physical dt remains 0.01')
    reserve = settings.get('minimum_state_remaining_ms', 0.)
    if (type(reserve) not in (int, float) or not np.isfinite(reserve)
            or not 0 <= reserve < min(float(settings['maximum_state_age_ms']), 10.)):
        raise ValueError('controller.minimum_state_remaining_ms must leave a positive fresh-state window')
    for key in ('maximum_state_age_ms','maximum_forecast_age_ms','maximum_compute_ms'):
        value=float(settings[key])
        if not np.isfinite(value) or value<=0:
            raise ValueError(f'Invalid controller.{key}')
    if type(settings.get('sample_id')) is not int or settings['sample_id']<0:
        raise ValueError('controller.sample_id must be a nonnegative integer')
    if settings['dynamics'].get('b_mode') not in ('zero','configured'):
        raise ValueError('Only explicit zero/configured bias is supported')
    gap_hold=settings.get('gap_hold',{})
    transition = settings.get('forecast_transition', {})
    if (not isinstance(transition, dict) or type(transition.get('enabled', False)) is not bool):
        raise ValueError('controller.forecast_transition.enabled must be a boolean')
    if transition.get('enabled', False):
        if settings['mode'] != 'q_only':
            raise ValueError('forecast_transition is currently supported only for q_only')
        duration = transition.get('duration_ms')
        if type(duration) not in (int, float) or not np.isfinite(duration) or not 10 <= duration <= 320:
            raise ValueError('controller.forecast_transition.duration_ms must be in [10,320]')
    if (not isinstance(gap_hold,dict) or type(gap_hold.get('enabled',False)) is not bool):
        raise ValueError('controller.gap_hold.enabled must be a boolean')
    if gap_hold.get('enabled'):
        duration=gap_hold.get('maximum_ms')
        if type(duration) not in (int,float) or not np.isfinite(duration) or not 0<duration<=320:
            raise ValueError('controller.gap_hold.maximum_ms must be in (0,320]')
    from inference.controllers.wm_two_step import vector
    for key in ('max_velocity_rad_s','max_acceleration_rad_s2','max_step_rad','max_tracking_error_rad'):
        vector(settings['limits'][key],7,key,positive=True)
    if settings['limits'].get('max_jerk_rad_s3') is not None:
        vector(settings['limits']['max_jerk_rad_s3'],7,'max_jerk_rad_s3',positive=True)
    for key in ('q_only_position_gain_s2','q_only_velocity_gain_s'):
        if key in settings['limits']:
            vector(settings['limits'][key],7,key,positive=True)
    if settings['limits'].get('q_only_smoothing','legacy') not in ('legacy','jerk_tracking'):
        raise ValueError('q_only_smoothing must be legacy or jerk_tracking')
    if settings['dynamics'].get('b_mode')=='configured':
        vector(settings['dynamics']['b0_nm'],7,'b0_nm')
    JointQTauController(dt=settings['dt'],**settings['weights'])
    if settings['mode']=='mit_position':
        if settings.get('position_scale',1.)!=1. or settings.get('torque_scale',1.)!=1.:
            raise ValueError('MIT mode uses mit.kp/kd directly; objective scales must remain 1')
        MitPositionController(dt=settings['dt'],mit=settings.get('mit'),
            velocity_limit=settings['limits']['max_velocity_rad_s'],**settings['weights'])
    elif settings['mode']=='tau_only':
        if settings.get('torque_scale',1.)!=1.:
            raise ValueError('tau_only fixes WM torque weight to 1; torque_scale must remain 1')
        TauOnlyController(dt=settings['dt'],mit=settings.get('mit'),
            velocity_limit=settings['limits']['max_velocity_rad_s'],**settings['weights'])
    WmReferenceAligner(dt=settings['dt'],sample_id=settings['sample_id'],
                      torque_reference=settings['torque_reference'],cutoff_hz=settings['tau_filter_cutoff_hz'])
