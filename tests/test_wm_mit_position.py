from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inference.controllers.wm_two_step import ControlState,TwoStepReference,MitPositionController
from inference.core.wm_control import WmControlLayer,WmReferenceAligner,validate_settings

URDF=Path(__file__).resolve().parents[1]/'gello_teleop/models/xarm7_dynamics_g2.urdf'


class Dynamics:
    def __init__(self):
        self.mass=np.eye(7)*2+.3*np.ones((7,7))
        self.n=np.arange(7)*.2
        self.b=np.ones(7)*.1
        self.states=[]
    def snapshot(self,q,dq):
        self.states.append((q.copy(),dq.copy()))
        return self.mass,self.n,self.b


def settings(base='measured'):
    return dict(reference_source='wm_q',velocity_source='wm_difference',feedforward_source='wm_tau',
                integration_base=base,kp=2.,kd=.3)


def inputs():
    state=ControlState(1_000_000,np.arange(7)*.1,np.arange(7)*.01)
    ref=TwoStepReference(np.array([1_010_000,1_020_000]),
        np.tile(state.q+.05,(2,1)),np.ones((2,7))*.8,dq=np.ones((2,7))*.2)
    return state,ref


def test_mit_feedback_coupled_dynamics_increment_and_two_step_integration():
    model=Dynamics();state,ref=inputs()
    controller=MitPositionController(mit=settings(),velocity_limit=2,dt=.01,dynamics=model)
    result=controller.solve(state,ref)
    torque=2*(ref.q[0]-state.q)+.3*(ref.dq[0]-state.dq)+ref.tau[0]
    # Independent check through the physical balance equation, not elementwise tau/Mjj.
    np.testing.assert_allclose(model.mass@result.a_plan[0]+model.n+model.b,torque)
    expected_delta=.01*state.dq+.5*.01**2*result.a_plan[0]
    np.testing.assert_allclose(result.delta_q_raw[0],expected_delta)
    np.testing.assert_allclose(result.q_raw[0],state.q+expected_delta)
    np.testing.assert_allclose(result.v_plan[0],state.dq+.01*result.a_plan[0])
    np.testing.assert_allclose(result.q_raw[1],result.q_raw[0]+.01*result.v_plan[0]+.5*.01**2*result.a_plan[1])
    second_tau=2*(ref.q[1]-result.q_raw[0])+.3*(ref.dq[1]-result.v_plan[0])+ref.tau[1]
    np.testing.assert_allclose(result.tau_mit[1],second_tau)
    np.testing.assert_allclose(model.states[1][0],result.q_raw[0])
    np.testing.assert_allclose(result.dynamics_residual,0,atol=1e-12)
    assert result.solver_status=='mit_forward_dynamics'


def test_previous_command_base_changes_integration_but_keeps_measured_feedback():
    model=Dynamics();state,ref=inputs()
    previous=replace(state,q=state.q+.02,dq=state.dq+.1)
    controller=MitPositionController(mit=settings('previous_command'),velocity_limit=2,dynamics=model)
    result=controller.solve(state,ref,integration_state=previous)
    np.testing.assert_allclose(result.q_raw[0],previous.q+.01*previous.dq+.5*.01**2*result.a_plan[0])
    np.testing.assert_allclose(result.tau_feedback[0],2*(ref.q[0]-state.q)+.3*(ref.dq[0]-state.dq))
    np.testing.assert_equal(model.states[0][0],state.q)
    first=controller.solve(state,ref)
    np.testing.assert_equal(first.integration_q0,state.q)


def test_command_state_feedback_prevents_virtual_acceleration_windup_on_stalled_servo():
    model=Dynamics()
    measured=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    ref=TwoStepReference(np.array([1_010_000,1_020_000]),np.ones((2,7))*.1,
                         np.zeros((2,7)))
    mit={**settings('previous_command'),'velocity_source':'zero',
         'feedback_state':'command','feedback_space':'acceleration',
         'feedforward_source':'nominal_wm_blend','nominal_acceleration_source':'zero',
         'wm_tau_weight':0.,'kp':8.,'kd':6.}
    controller=MitPositionController(mit=mit,velocity_limit=.5,acceleration_limit=1.,
                                     jerk_limit=5.,dynamics=model)
    command=None
    acceleration=np.zeros(7)
    for i in range(1000):
        result=controller.solve(measured,ref,previous_acceleration=acceleration,
                                integration_state=command)
        command=ControlState(measured.timestamp_us+i*10_000,result.q_raw[0],result.v_plan[0])
        acceleration=result.a_plan[0]
    np.testing.assert_allclose(command.q,.1,atol=.003)
    np.testing.assert_allclose(command.dq,0,atol=.005)


def test_measured_base_ignores_previous_command_and_static_feedforward_does_not_double_gravity():
    model=Dynamics();state,ref=inputs();state=replace(state,dq=np.zeros(7))
    ref=replace(ref,q=np.tile(state.q,(2,1)),dq=np.zeros((2,7)),tau=np.tile(model.n+model.b,(2,1)))
    controller=MitPositionController(mit=settings(),velocity_limit=2,dynamics=model)
    result=controller.solve(state,ref,integration_state=replace(state,q=state.q+.1))
    np.testing.assert_allclose(result.a_plan,0,atol=1e-12)
    np.testing.assert_allclose(result.q_raw,np.tile(state.q,(2,1)),atol=1e-12)


def test_mit_desired_velocity_uses_forecast_slope_and_common_limit():
    times=1_000_000+np.arange(1,33,dtype=np.int64)*10_000
    q=(times-times[0])[:,None]/1e6*np.ones((1,7))*3
    forecast=dict(anchor_us=1_000_000,future_timestamps_us=times,q=q[None],tau=np.zeros((1,32,7)))
    state=ControlState(1_025_000,np.zeros(7),np.zeros(7))
    ref=WmReferenceAligner().align(forecast,state,use_dq=True)
    np.testing.assert_allclose(ref.dq,3)
    # Desired speed derives from WM's trajectory, not from its position error.
    controller=MitPositionController(mit={**settings(),'kp':0,'kd':1},velocity_limit=.5,dynamics=Dynamics())
    result=controller.solve(state,ref)
    np.testing.assert_allclose(result.dq_reference,.5)
    np.testing.assert_allclose(result.tau_feedback[0],.5)
    np.testing.assert_allclose(result.tau_feedback[1],.5-result.v_plan[0])


def test_filtered_wm_velocity_recovers_ramp_from_noisy_forecast_and_caps_feedback():
    times=1_000_000+np.arange(1,33,dtype=np.int64)*10_000
    clean=(times-times[0])[:,None]/1e6*np.ones((1,7))*.08
    noisy=clean+np.arange(32)[:,None]%2*.004
    forecast=dict(anchor_us=1_000_000,future_timestamps_us=times,q=noisy[None],
                  tau=np.zeros((1,32,7)))
    state=ControlState(1_050_000,np.zeros(7),np.zeros(7))
    aligner=WmReferenceAligner()
    raw=aligner.align(forecast,state,use_tau=False,use_dq=True)
    smooth=aligner.align(forecast,state,use_tau=False,use_dq=True,velocity_method='smoothed')
    assert np.max(np.abs(smooth.dq-.08)) < np.max(np.abs(raw.dq-.08))
    np.testing.assert_allclose(smooth.dq,.08,atol=.02)
    controller=MitPositionController(mit={**settings(),'velocity_source':'filtered_wm_difference',
        'reference_velocity_limit_rad_s':.06,'kp':0.,'kd':1.},velocity_limit=.5,dynamics=Dynamics())
    smooth=replace(smooth,tau=np.zeros((2,7)))
    result=controller.solve(state,smooth)
    np.testing.assert_allclose(result.dq_reference,.06)
    with pytest.raises(ValueError,match='cannot exceed'):
        MitPositionController(mit={**settings(),'reference_velocity_limit_rad_s':.6},velocity_limit=.5)


@pytest.mark.parametrize('field',['q','dq','tau'])
def test_mit_rejects_invalid_references(field):
    state,ref=inputs()
    controller=MitPositionController(mit=settings(),velocity_limit=2,dynamics=Dynamics())
    with pytest.raises(ValueError,match='reference'):
        controller.solve(state,replace(ref,**{field:np.full((2,7),np.nan)}))


@pytest.mark.parametrize('change',[{'kp':-1},{'kd':float('nan')},{'kp':[1]*6},
    {'reference_source':'dp_q'},{'feedforward_source':'gravity'},{'integration_base':'unknown'}])
def test_mit_invalid_settings_rejected(change):
    with pytest.raises(ValueError):
        MitPositionController(mit={**settings(),**change},velocity_limit=2)


def test_mit_layer_limits_and_only_successfully_sent_commands_advance_history():
    config=dict(mode='mit_position',execution='active',dt=.01,horizon_steps=2,sample_id=0,
        maximum_state_age_ms=10,maximum_forecast_age_ms=100,maximum_compute_ms=100,
        torque_reference='raw',tau_filter_cutoff_hz=20,dynamics={'b_mode':'zero'},weights={},
        mit=settings('previous_command'),
        limits=dict(max_velocity_rad_s=2,max_acceleration_rad_s2=4,max_step_rad=.01,max_tracking_error_rad=.1))
    validate_settings(config,allow_active=True)
    layer=WmControlLayer(config,URDF,allow_active=True)
    layer.controller.dynamics=Dynamics()
    state=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    layer.update_forecast(dict(anchor_us=1_000_000,plan_version=1,
        future_timestamps_us=1_000_000+np.arange(1,33,dtype=np.int64)*10_000,
        q=np.ones((1,32,7))*2,tau=np.ones((1,32,7))*50))
    event=layer.poll(1_001_000,state)
    assert event['valid'] and event['limited'] and not event['sent']
    assert np.max(np.abs(event['command_a_plan']))<=4+1e-9
    np.testing.assert_allclose(event['first_q_command'],.0004)
    assert layer.limits.previous_q is None
    event['sent']=True
    layer.mark_sent(event)
    next_event=layer.poll(1_011_000,replace(state,timestamp_us=1_010_000))
    np.testing.assert_equal(next_event['integration_q0'],event['first_q_command'])
    np.testing.assert_equal(next_event['integration_v0'],event['command_v_plan'][0])


def test_mit_gap_hold_brakes_with_jerk_limits_without_reusing_stale_torque():
    from inference.dp_wm.config import load_config
    config=load_config(URDF.parents[2]/'inference/configs/dp_wm_xarm.yaml')['controller']
    config['gap_hold']={'enabled':True,'maximum_ms':200.}
    layer=WmControlLayer(config,URDF,allow_active=True)
    anchor=1_000_000
    layer.update_forecast({'anchor_us':anchor,'plan_version':1,
        'future_timestamps_us':anchor+np.arange(1,33,dtype=np.int64)*10_000,
        'q':np.zeros((1,32,7)),'tau':np.zeros((1,32,7))})
    layer.limits.previous_q=np.zeros(7)
    layer.limits.previous_v=np.full(7,.1)
    state=ControlState(anchor+110_000,np.zeros(7),np.zeros(7))
    event=layer.poll(state.timestamp_us+1_000,state)
    assert event['valid'] and event['actual_mode']=='position_hold'
    assert 'tau_reference' not in event
    assert np.max(np.abs(event['command_jerk_plan']))<=5.+1e-9


@pytest.mark.parametrize('beta',[0.,.05,1.,[0.,.1,.2,.3,.4,.5,1.]])
def test_nominal_wm_fusion_obeys_total_torque_balance(beta):
    model=Dynamics();state,ref=inputs()
    desired_a=np.tile(np.arange(7)*.1,(2,1))
    ref=replace(ref,ddq=desired_a)
    controller=MitPositionController(mit={**settings(),'feedforward_source':'nominal_wm_blend',
        'wm_tau_weight':beta},velocity_limit=2,dynamics=model)
    result=controller.solve(state,ref)
    nominal=desired_a@model.mass.T+model.n+model.b
    expected_ff=(1-np.asarray(beta))*nominal+np.asarray(beta)*ref.tau
    np.testing.assert_allclose(result.tau_nominal,nominal)
    np.testing.assert_allclose(result.tau_feedforward,expected_ff)
    np.testing.assert_allclose(result.tau_mit,result.tau_feedback+expected_ff)
    np.testing.assert_allclose(result.a_plan@model.mass.T+model.n+model.b,result.tau_mit)


def test_nominal_only_tracks_reference_acceleration_without_extra_gravity():
    model=Dynamics();state,ref=inputs()
    controller=MitPositionController(mit={**settings(),'kp':0.,'kd':0.,
        'feedforward_source':'nominal_wm_blend','wm_tau_weight':0.},velocity_limit=2,dynamics=model)
    desired=np.tile(np.arange(7)*.1,(2,1))
    result=controller.solve(state,replace(ref,ddq=desired,tau=np.ones((2,7))*1000))
    np.testing.assert_allclose(result.a_plan,desired,atol=1e-12)


@pytest.mark.parametrize('ddq',[None,np.ones((1,7)),np.full((2,7),np.nan)])
def test_nominal_fusion_rejects_invalid_acceleration_reference(ddq):
    state,ref=inputs()
    controller=MitPositionController(mit={**settings(),'feedforward_source':'nominal_wm_blend',
        'wm_tau_weight':.05},velocity_limit=2,dynamics=Dynamics())
    with pytest.raises(ValueError,match='acceleration reference'):
        controller.solve(state,replace(ref,ddq=ddq))


@pytest.mark.parametrize('beta',[-.1,1.1,float('nan'),[.05]*6])
def test_fusion_rejects_invalid_weight(beta):
    with pytest.raises(ValueError,match='wm_tau_weight'):
        MitPositionController(mit={**settings(),'feedforward_source':'nominal_wm_blend',
            'wm_tau_weight':beta},velocity_limit=2)


def test_aligner_acceleration_derives_from_timestamped_wm_trajectory():
    t=np.arange(1,33)*.01
    forecast=dict(anchor_us=1_000_000,future_timestamps_us=1_000_000+np.arange(1,33)*10_000,
                  q=np.broadcast_to((.5*.6*t*t)[None,:,None],(1,32,7)),tau=np.zeros((1,32,7)))
    ref=WmReferenceAligner().align(forecast,ControlState(1_100_000,np.zeros(7),np.zeros(7)),use_dq=True)
    np.testing.assert_allclose(ref.dq[:,0],.6*np.array([.11,.12]),atol=1e-12)
    np.testing.assert_allclose(ref.ddq,.6,atol=1e-12)


def smooth_layer(execution='active'):
    config=dict(mode='mit_position',execution=execution,dt=.01,horizon_steps=2,sample_id=0,
        maximum_state_age_ms=10,maximum_forecast_age_ms=100,maximum_compute_ms=100,
        torque_reference='raw',tau_filter_cutoff_hz=20,dynamics={'b_mode':'zero'},weights={},
        mit={**settings('previous_command'),'feedforward_source':'nominal_wm_blend','wm_tau_weight':.05},
        limits=dict(max_velocity_rad_s=.5,max_acceleration_rad_s2=1,max_jerk_rad_s3=5,
                    max_step_rad=.01,max_tracking_error_rad=.1))
    layer=WmControlLayer(config,URDF,allow_active=True)
    layer.controller.dynamics=Dynamics()
    return layer


def update_smooth_forecast(layer,state,torque):
    layer.update_forecast(dict(anchor_us=state.timestamp_us,plan_version=state.timestamp_us,
        future_timestamps_us=state.timestamp_us+np.arange(1,33)*10_000,
        q=np.tile(state.q,(1,32,1)),tau=np.full((1,32,7),torque)))


def test_alternating_large_torque_never_causes_acceleration_jumps_or_history_drift():
    layer=smooth_layer()
    state=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    previous_a=np.zeros(7)
    for i in range(80):
        update_smooth_forecast(layer,state,1000 if i%2==0 else -1000)
        event=layer.poll(state.timestamp_us+1000,state)
        assert event['valid'],event['reason']
        a=event['command_a_plan'][0];v=event['command_v_plan'][0]
        np.testing.assert_allclose(event['first_q_command'],state.q+.01*state.dq+.5*.01**2*a,atol=1e-12)
        np.testing.assert_allclose(v,state.dq+.01*a,atol=1e-12)
        assert np.max(np.abs(a-previous_a))<=.05+1e-12
        assert np.max(np.abs(event['command_jerk_plan']))<=5+1e-9
        assert np.max(np.abs(event['command_a_plan']))<=1+1e-12
        assert np.max(np.abs(event['command_v_plan']))<=.5+1e-12
        assert layer.limits.previous_q is None if i==0 else np.array_equal(layer.limits.previous_q,state.q)
        event['sent']=True;layer.mark_sent(event)
        state=ControlState(state.timestamp_us+10_000,event['first_q_command'],v)
        previous_a=a


def test_velocity_braking_reserves_room_to_obey_jerk_at_speed_limit():
    layer=smooth_layer()
    state=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    # Move bounds away to isolate velocity braking from robot joint limits.
    layer.limits.bounds[:]=[-100,100]
    last_a=np.zeros(7)
    for i in range(150):
        update_smooth_forecast(layer,state,1000)
        event=layer.poll(state.timestamp_us+1000,state)
        assert event['valid'],(i,event['reason'])
        a=event['command_a_plan'][0]
        assert np.max(np.abs(a-last_a))<=.05+1e-10
        assert np.max(np.abs(event['command_v_plan']))<=.5+1e-10
        event['sent']=True;layer.mark_sent(event)
        state=ControlState(state.timestamp_us+10_000,event['first_q_command'],event['command_v_plan'][0])
        last_a=a
    np.testing.assert_allclose(state.dq,.5,atol=1e-8)


def test_unacknowledged_preview_does_not_change_smoothing_history_and_reset_clears_it():
    layer=smooth_layer()
    state=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    update_smooth_forecast(layer,state,1000)
    first=layer.poll(1_001_000,state)
    assert first['valid']
    second=layer.poll(1_011_000,replace(state,timestamp_us=1_010_000))
    assert second['valid']
    np.testing.assert_equal(first['first_q_command'],second['first_q_command'])
    assert layer.limits.previous_q is None
    np.testing.assert_equal(layer.limits.previous_acceleration,0)
    second['sent']=True;layer.mark_sent(second)
    layer.reset()
    assert layer.limits.previous_q is None
    np.testing.assert_equal(layer.limits.previous_acceleration,0)


def test_infeasible_jerk_and_position_bounds_reject_command_instead_of_jumping():
    layer=smooth_layer()
    state=ControlState(1_000_000,layer.limits.bounds[:,1]-.00001,np.ones(7)*.2)
    update_smooth_forecast(layer,state,1000)
    event=layer.poll(state.timestamp_us+1000,state)
    assert not event['valid'] and not event['sent']
    assert 'feasible intersection' in event['reason']


@pytest.mark.parametrize('jerk',[0.,-1.,float('nan'),[5]*6])
def test_invalid_jerk_limits_fail_validation(jerk):
    layer=smooth_layer()
    config={**layer.settings,'limits':{**layer.settings['limits'],'max_jerk_rad_s3':jerk}}
    with pytest.raises(ValueError,match='max_jerk'):
        validate_settings(config,allow_active=True)


def test_acceleration_limit_does_not_hide_nonfinite_forward_dynamics():
    model=Dynamics();model.mass=np.eye(7)*1e-300
    state,ref=inputs()
    controller=MitPositionController(mit=settings(),velocity_limit=2,dynamics=model,
                                    acceleration_limit=1,jerk_limit=5)
    with pytest.raises(ValueError,match='nonfinite acceleration'):
        controller.solve(state,replace(ref,tau=np.full((2,7),1e100)))


def test_smooth_projection_preserves_measured_integration_target_instead_of_accumulating_acceleration():
    layer=smooth_layer()
    layer.controller.integration_base='measured'
    state=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    # A previous position command is ahead of feedback. Zero acceleration at
    # measured q must pull the target back; continuing from previous_q drifts.
    layer.limits.previous_q=np.full(7,.01)
    layer.limits.previous_v=np.zeros(7)
    layer.limits.previous_acceleration=np.zeros(7)
    model=layer.controller.dynamics
    layer.update_forecast(dict(anchor_us=state.timestamp_us,plan_version=1,
        future_timestamps_us=state.timestamp_us+np.arange(1,33)*10_000,
        q=np.zeros((1,32,7)),tau=np.tile(model.n+model.b,(1,32,1))))
    event=layer.poll(state.timestamp_us+1000,state)
    assert event['valid'],event['reason']
    np.testing.assert_allclose(event['a_plan'],0,atol=1e-12)
    np.testing.assert_allclose(event['q_raw'],0,atol=1e-12)
    assert np.all(event['first_q_command']<.01)
    np.testing.assert_allclose(event['command_a_plan'][0],-.05,atol=1e-12)
    assert np.max(np.abs(event['command_jerk_plan']))<=5+1e-10


def test_continuous_integration_and_measured_integration_keep_distinct_targets_under_jerk_projection():
    measured=smooth_layer('offline');continuous=smooth_layer('offline')
    measured.controller.integration_base='measured'
    state=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    for layer in (measured,continuous):
        layer.limits.previous_q=np.full(7,.01)
        model=layer.controller.dynamics
        layer.limits.previous_v=np.zeros(7)
        layer.update_forecast(dict(anchor_us=state.timestamp_us,plan_version=1,
            future_timestamps_us=state.timestamp_us+np.arange(1,33)*10_000,
            q=np.zeros((1,32,7)),tau=np.tile(model.n+model.b,(1,32,1))))
    a=measured.poll(state.timestamp_us+1000,state)
    b=continuous.poll(state.timestamp_us+1000,state)
    assert a['valid'] and b['valid']
    assert np.all(a['first_q_command']<b['first_q_command'])
    np.testing.assert_allclose(b['first_q_command'],.01,atol=1e-12)


def tracking_settings():
    return dict(reference_source='wm_q',velocity_source='zero',feedforward_source='nominal_wm_blend',
                wm_tau_weight=.05,integration_base='previous_command',feedback_space='acceleration',
                nominal_acceleration_source='zero',wm_residual_acceleration_limit_rad_s2=.5,kp=8.,kd=6.)


def test_mass_scaled_feedback_and_bounded_wm_torque_cannot_dominate_position_tracking():
    model=Dynamics();model.mass=np.diag([1.,1.,.5,.3,.02,.03,.0005])+.001*np.ones((7,7))
    state,ref=inputs()
    controller=MitPositionController(mit=tracking_settings(),velocity_limit=.5,dynamics=model)
    ref=replace(ref,tau=np.full((2,7),1000.),dq=None,ddq=np.full((2,7),1e6))
    result=controller.solve(state,ref)
    expected_feedback=8*(ref.q[0]-state.q)-6*state.dq
    np.testing.assert_allclose(result.tau_feedback[0],model.mass@expected_feedback)
    np.testing.assert_allclose(result.tau_nominal[0],model.n+model.b)
    np.testing.assert_equal(result.dq_reference,0.)
    correction=result.a_plan[0]-expected_feedback
    assert np.max(np.abs(correction))<=.025+1e-10
    assert np.max(np.abs(result.wm_acceleration_raw))>1e4


@pytest.mark.parametrize('servo_time_constant',[.02,.04,.08])
def test_closed_loop_position_tracking_with_servo_lag_and_large_wrist_torque_bias(servo_time_constant):
    seed=smooth_layer()
    layer=WmControlLayer({**seed.settings,'execution':'offline','mit':tracking_settings()},URDF)
    model=Dynamics();model.mass=np.diag([1.,1.,.5,.3,.02,.03,.0005])+.001*np.ones((7,7))
    layer.controller.dynamics=model
    initial=np.array([0.,0.,0.,np.pi/2,0.,np.pi/2,-np.pi/2])
    target=initial.copy();target[1]+=.05;target[3]-=.12
    state=ControlState(1_000_000,initial.copy(),np.zeros(7))
    alpha=1-np.exp(-.01/servo_time_constant)
    previous_a=np.zeros(7)
    max_error_after_settling=0.
    for i in range(800):
        # The WM torque is strongly biased and flips the tiny-inertia wrist.
        # q remains the task target; the command must converge to that target.
        torque=model.n+model.b+np.array([5.,-4.,6.,3.,2.,-2.,20. if i%8<4 else -20.])
        layer.update_forecast(dict(anchor_us=state.timestamp_us,plan_version=i,
            future_timestamps_us=state.timestamp_us+np.arange(1,33)*10_000,
            q=np.tile(target,(1,32,1)),tau=np.tile(torque,(1,32,1))))
        event=layer.poll(state.timestamp_us+1000,state)
        assert event['valid'],(i,event['reason'])
        command=event['first_q_command'];a=event['command_a_plan'][0]
        assert np.max(np.abs(a-previous_a))<=.05+1e-10
        assert np.max(np.abs(event['command_v_plan']))<=.5+1e-10
        next_q=state.q+alpha*(command-state.q)
        state=ControlState(state.timestamp_us+10_000,next_q,(next_q-state.q)/.01)
        previous_a=a
        if i>600:max_error_after_settling=max(max_error_after_settling,float(np.max(np.abs(state.q-target))))
    assert max_error_after_settling<.01
    assert state.q[3]<initial[3]-.1
    assert abs(state.q[6]-initial[6])<.01


@pytest.mark.parametrize('change',[{'feedback_space':'unknown'},{'nominal_acceleration_source':'unknown'},
    {'wm_residual_acceleration_limit_rad_s2':0},{'wm_residual_acceleration_limit_rad_s2':float('nan')}])
def test_invalid_position_tracking_settings_rejected(change):
    with pytest.raises(ValueError):
        MitPositionController(mit={**tracking_settings(),**change},velocity_limit=.5)


@pytest.mark.parametrize('budget,valid',[(2.,False),(4.,True)])
def test_observed_compute_jitter_is_accepted_only_within_configured_budget(monkeypatch,budget,valid):
    from types import SimpleNamespace
    import inference.core.wm_control as module
    layer=smooth_layer();layer.settings['maximum_compute_ms']=budget
    clock=SimpleNamespace(seconds=0.)
    original=layer.controller.solve
    def delayed(*args,**kwargs):
        result=original(*args,**kwargs)
        clock.seconds+=.002057587
        return result
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock.seconds)
    monkeypatch.setattr(layer.controller,'solve',delayed)
    state=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    update_smooth_forecast(layer,state,0.)
    event=layer.poll(1_001_000,state)
    assert event['valid'] is valid
    assert event['sent'] is False
    if not valid:assert 'deadline' in event['reason']
