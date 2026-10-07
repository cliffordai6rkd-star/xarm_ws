"""Pure numerical and offline model tests; no robot access or neural model loading."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inference.controllers.wm_two_step import (
    ControlState, TwoStepReference, QOnlyController, TauOnlyController, JointQTauController,
    LocalDynamicsModel, ReferenceLimits,
)
from inference.core.wm_control import WmReferenceAligner, WmControlLayer

ROOT=Path(__file__).resolve().parents[1]
URDF=ROOT/'gello_teleop/models/xarm7_dynamics_g2.urdf'


class Dynamics:
    def __init__(self):
        self.mass=np.eye(7)*2 + .3*np.ones((7,7))
        self.n=np.arange(7,dtype=float)*.2
        self.b=np.ones(7)*.5
        self.calls=0

    def snapshot(self,q,dq):
        self.calls+=1
        return self.mass,self.n,self.b


def consistent():
    model=Dynamics()
    state=ControlState(1_000_000,np.arange(7)*.1,np.arange(7)*.01)
    a=np.array([[1,2,3,4,5,6,7],[-2,-1,0,1,2,3,4]],dtype=float)
    h=.01
    q=np.array([state.q+h*state.dq+.5*h*h*a[0],
        state.q+2*h*state.dq+1.5*h*h*a[0]+.5*h*h*a[1]])
    tau=a@model.mass.T+model.n+model.b
    ref=TwoStepReference(np.array([1_010_000,1_020_000]),q,tau)
    return model,state,a,ref


def test_coupled_torque_solve_and_second_step_inherit_velocity():
    model,state,a,ref=consistent()
    result=TauOnlyController(dynamics=model).solve(state,ref)
    np.testing.assert_allclose(result.a_plan,a,atol=1e-12)
    np.testing.assert_allclose(result.q_command,ref.q,atol=1e-12)
    np.testing.assert_allclose(result.v_plan[1],state.dq+.01*(a[0]+a[1]),atol=1e-12)
    # A per-axis division loses the joint coupling in this example.
    diagonal=(ref.tau[0]-model.n-model.b)/np.diag(model.mass)
    assert np.linalg.norm(diagonal-a[0])>1


def test_all_three_modes_agree_on_dynamically_consistent_reference():
    model,state,a,ref=consistent()
    for cls in (QOnlyController,TauOnlyController,JointQTauController):
        result=cls(dynamics=model).solve(state,ref)
        np.testing.assert_allclose(result.q_command,ref.q,atol=1e-12)
        if cls is not QOnlyController:
            np.testing.assert_allclose(result.a_plan,a,atol=1e-10)


def test_information_purity_and_joint_limits_of_objective_weights():
    model,state,a,ref=consistent()
    qonly=QOnlyController(dynamics=model)
    first=qonly.solve(state,ref)
    second=qonly.solve(state,replace(ref,tau=np.full((2,7),np.nan)))
    np.testing.assert_equal(first.q_command,second.q_command)
    assert model.calls==0
    tauonly=TauOnlyController(dynamics=model)
    first=tauonly.solve(state,ref)
    second=tauonly.solve(state,replace(ref,q=None))
    np.testing.assert_equal(first.q_command,second.q_command)
    joint=JointQTauController(dynamics=model)
    joint.torque_scale=0
    qonly=QOnlyController()
    np.testing.assert_allclose(joint.solve(state,ref).q_command,qonly.solve(state,ref).q_command)
    joint.position_scale=0; joint.torque_scale=1
    tauonly=TauOnlyController(dynamics=model)
    np.testing.assert_allclose(joint.solve(state,ref).q_command,tauonly.solve(state,ref).q_command)
    joint=JointQTauController(dynamics=model,rho=0)
    np.testing.assert_allclose(joint.solve(state,ref).q_command,QOnlyController().solve(state,ref).q_command)
    # tau_only now follows MIT forward dynamics, independent of LS weights.
    np.testing.assert_allclose(TauOnlyController(dynamics=model,rho=0).solve(state,ref).q_command,
                               tauonly.solve(state,ref).q_command)


def test_q_only_direct_wm_targets_bypass_solver_dynamics_and_objective_weights(monkeypatch):
    model,state,_,ref=consistent()
    def fail(*args,**kwargs):
        raise AssertionError('Direct q path must not use a least-squares solver')
    monkeypatch.setattr(np.linalg,'lstsq',fail)
    controller=QOnlyController(dynamics=model,smooth_weight=1e6,previous_weight=1e6)
    result=controller.solve(state,replace(ref,tau=np.full((2,7),np.nan)),previous_acceleration=np.full(7,np.nan))
    np.testing.assert_equal(result.q_raw,ref.q)
    assert result.solver_status=='direct_wm_q' and model.calls==0
    assert result.mass_matrix is None and result.dynamics_residual is None
    result.q_raw[:]=0
    assert np.any(ref.q!=0)  # The output owns its target snapshot.


@pytest.mark.parametrize('q',[None,np.zeros((1,7)),np.full((2,7),np.nan),np.full((2,7),np.inf)])
def test_direct_wm_q_rejects_missing_malformed_or_nonfinite_targets(q):
    _,state,_,ref=consistent()
    with pytest.raises(ValueError,match='q reference'):
        QOnlyController().solve(state,replace(ref,q=q))


def test_q_only_layer_time_aligns_selected_sample_and_limits_direct_target(monkeypatch):
    config={**settings(),'sample_id':1}
    layer=WmControlLayer(config,URDF)
    def fail(*args,**kwargs):
        raise AssertionError('Direct q mode must not run the objective solver')
    monkeypatch.setattr(np.linalg,'lstsq',fail)
    times=1_000_000+np.arange(1,33,dtype=np.int64)*10_000
    q=np.stack([np.ones((32,7))*-2,np.arange(32)[:,None]*np.ones((1,7))*.01+1])
    # q_only has no dependency on the existence/quality of forecast torque.
    layer.update_forecast({'anchor_us':1_000_000,'plan_version':1,'future_timestamps_us':times,'q':q})
    state=ControlState(1_025_000,np.zeros(7),np.zeros(7))
    event=layer.poll(1_026_000,state)
    assert event['valid'] and event['limited'] and event['solver_status']=='direct_wm_q'
    np.testing.assert_allclose(event['q_reference'][:,0],[1.025,1.035])
    np.testing.assert_equal(event['q_raw'],event['q_reference'])
    np.testing.assert_equal(event['first_q_command'],event['q_command'][0])
    np.testing.assert_allclose(event['first_q_command'],np.ones(7)*.0004,atol=1e-12)
    assert np.max(np.abs(event['command_a_plan']))<=4+1e-9


def test_static_balance_needs_bias_and_is_not_contact_control():
    model,state,_,ref=consistent()
    state=replace(state,dq=np.zeros(7))
    ref=replace(ref,tau=np.tile(model.n+model.b,(2,1)))
    result=TauOnlyController(dynamics=model).solve(state,ref)
    np.testing.assert_allclose(result.q_command,np.tile(state.q,(2,1)),atol=1e-12)
    model.b=np.zeros(7)
    wrong=TauOnlyController(dynamics=model).solve(state,ref)
    assert np.max(np.abs(wrong.q_command-state.q))>1e-6
    assert wrong.model_confidence=='uncalibrated_rigid_body'


def test_alignment_selects_same_sample_and_skips_expired_prefix():
    times=1_000_000+np.arange(1,33,dtype=np.int64)*10_000
    q=np.stack([np.arange(32)[:,None]*np.ones((1,7)),100+np.arange(32)[:,None]*np.ones((1,7))])
    forecast={'anchor_us':1_000_000,'future_timestamps_us':times,'q':q,'tau':q+10}
    state=ControlState(1_025_000,np.zeros(7),np.zeros(7))
    ref=WmReferenceAligner(sample_id=1).align(forecast,state)
    np.testing.assert_allclose(ref.q[:,0],[102.5,103.5])
    np.testing.assert_allclose(ref.tau[:,0],[112.5,113.5])
    np.testing.assert_equal(ref.timestamps_us,[1_035_000,1_045_000])
    with pytest.raises(ValueError,match='cover'):
        WmReferenceAligner().align(forecast,replace(state,timestamp_us=1_315_000))
    with pytest.raises(ValueError,match='selected sample'):
        WmReferenceAligner(sample_id=2).align(forecast,state)


def test_causal_torque_inverse_reconstructs_same_sample_interval_reference():
    retention=np.exp(-2*np.pi*20*.01)
    raw=np.array([2.,-1.]); filtered=[]; previous=5.
    for value in raw:
        previous=retention*previous+(1-retention)*value
        filtered.append(previous)
    f={'anchor_us':1_000_000,'anchor_tau_filtered':np.ones(7)*5,
       'future_timestamps_us':np.array([1_010_000,1_020_000]),
       'q':np.zeros((1,2,7)),'tau':np.tile(np.array(filtered)[None,:,None],(1,1,7))}
    state=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    ref=WmReferenceAligner(torque_reference='causal_inverse').align(f,state)
    np.testing.assert_allclose(ref.tau[:,0],raw,atol=1e-12)


def test_common_limits_bound_commands_and_only_advance_first_shadow_step():
    model,state,_,ref=consistent(); state=replace(state,q=np.zeros(7),dq=np.zeros(7))
    ref=replace(ref,q=np.ones((2,7))*2)
    result=QOnlyController().solve(state,ref)
    limits=ReferenceLimits(URDF,dict(max_velocity_rad_s=2,max_acceleration_rad_s2=4,
                                  max_step_rad=.01,max_tracking_error_rad=.1),.01)
    details=limits.project(result,state)
    assert details['limited']
    assert np.max(np.abs(details['command_a_plan']))<=4+1e-9
    assert np.max(np.abs(details['command_v_plan']))<=2
    np.testing.assert_equal(limits.previous_q,result.q_command[0])
    limits.reset(); assert limits.previous_q is None


def test_q_only_noisy_targets_use_jerk_limits_without_dynamics_and_keep_raw_reference(monkeypatch):
    layer=WmControlLayer({**settings(),'limits':{**settings()['limits'],
        'max_acceleration_rad_s2':1.,'max_jerk_rad_s3':5.,'q_only_smoothing':'jerk_tracking'}},URDF)
    def fail(*args,**kwargs):
        raise AssertionError('q_only smoothing must not invoke torque dynamics or objective solver')
    monkeypatch.setattr(np.linalg,'lstsq',fail)
    state=ControlState(1_000_000,np.array([0.,0.,0.,np.pi/2,0.,np.pi/2,-np.pi/2]),np.zeros(7))
    target=state.q.copy();target[3]-=.1
    previous_a=np.zeros(7)
    for i in range(600):
        noisy=target+(.002 if i%2==0 else -.002)
        layer.update_forecast({'anchor_us':state.timestamp_us,'plan_version':i,
            'future_timestamps_us':state.timestamp_us+np.arange(1,33,dtype=np.int64)*10_000,
            'q':np.tile(noisy,(1,32,1))})
        event=layer.poll(state.timestamp_us+1000,state)
        assert event['valid'],(i,event['reason'])
        np.testing.assert_allclose(event['q_raw'],np.tile(noisy,(2,1)))
        assert event['tau_mit'] is None and event['mass_matrix'] is None
        assert 'command_dynamics_residual' not in event
        assert np.max(np.abs(event['command_jerk_plan']))<=5+1e-9
        acceleration=event['command_a_plan'][0]
        assert np.max(np.abs(acceleration-previous_a))<=.05+1e-10
        assert np.max(np.abs(acceleration))<=1+1e-10
        state=ControlState(state.timestamp_us+10_000,event['first_q_command'],event['command_v_plan'][0])
        previous_a=acceleration
    assert np.max(np.abs(state.q-target))<.005


@pytest.mark.parametrize('kp,kd,jmax',[(16.,8.,5.),(100.,20.,20.)])
def test_q_only_jerk_limited_step_brakes_before_target_with_delayed_position_servo(kp,kd,jmax):
    cfg={**settings(),'limits':{**settings()['limits'],'max_velocity_rad_s':.5,
         'max_acceleration_rad_s2':1.,'max_jerk_rad_s3':jmax,'q_only_smoothing':'jerk_tracking',
         'q_only_position_gain_s2':kp,'q_only_velocity_gain_s':kd}}
    layer=WmControlLayer(cfg,URDF)
    initial=np.array([0.,0.,0.,np.pi/2,0.,np.pi/2,-np.pi/2])
    state=ControlState(1_000_000,initial.copy(),np.zeros(7));target=initial.copy();target[3]-=.1
    positions=[]
    for i in range(600):
        layer.update_forecast({'anchor_us':state.timestamp_us,'plan_version':i,
            'future_timestamps_us':state.timestamp_us+np.arange(1,33,dtype=np.int64)*10_000,
            'q':np.tile(target,(1,32,1))})
        event=layer.poll(state.timestamp_us+1000,state)
        assert event['valid'],(i,event['reason'])
        assert np.max(np.abs(event['command_jerk_plan']))<=jmax+1e-9
        q_next=state.q+(1-np.exp(-.01/.04))*(event['first_q_command']-state.q)
        state=ControlState(state.timestamp_us+10_000,q_next,(q_next-state.q)/.01)
        positions.append(state.q[3])
    assert min(positions)>=target[3]-.005
    assert abs(state.q[3]-target[3])<.001


def test_q_only_default_preserves_original_limiter_even_when_mit_jerk_limit_is_configured():
    layers=[WmControlLayer(settings(),URDF),
            WmControlLayer({**settings(),'limits':{**settings()['limits'],'max_jerk_rad_s3':5.}},URDF)]
    state=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    forecast={'anchor_us':1_000_000,'plan_version':1,
        'future_timestamps_us':1_000_000+np.arange(1,33,dtype=np.int64)*10_000,
        'q':np.ones((1,32,7))*.2}
    events=[]
    for layer in layers:
        layer.update_forecast(forecast);events.append(layer.poll(1_001_000,state))
    assert all(e['valid'] for e in events)
    np.testing.assert_equal(events[0]['q_command'],events[1]['q_command'])
    np.testing.assert_equal(events[0]['command_v_plan'],events[1]['command_v_plan'])
    np.testing.assert_equal(events[0]['command_a_plan'],events[1]['command_a_plan'])
    assert 'command_jerk_plan' not in events[1]
    np.testing.assert_allclose(events[1]['first_q_command'],.0004)


def settings():
    return dict(mode='q_only',execution='shadow',dt=.01,horizon_steps=2,sample_id=0,
        maximum_state_age_ms=10,maximum_forecast_age_ms=100,maximum_compute_ms=100,
        torque_reference='raw',tau_filter_cutoff_hz=20,dynamics={'b_mode':'zero'},
        weights={'eps_q':.01,'eps_tau':1.,'rho':1.},
        limits=dict(max_velocity_rad_s=2,max_acceleration_rad_s2=4,max_step_rad=.01,max_tracking_error_rad=.1))


def test_public_layer_never_sends_and_handles_stale_missing_and_pause():
    layer=WmControlLayer(settings(),URDF)
    assert not hasattr(layer,'arm') and not hasattr(layer,'send')
    state=ControlState(1_000_000,np.zeros(7),np.zeros(7))
    forecast={'anchor_us':1_000_000,'plan_version':1,
        'future_timestamps_us':1_000_000+np.arange(1,33,dtype=np.int64)*10_000,
        'q':np.zeros((1,32,7)),'tau':np.zeros((1,32,7))}
    layer.update_forecast(forecast)
    event=layer.poll(1_001_000,state)
    assert event['valid'] and event['sent'] is False
    assert event['history_kind']=='virtual_shadow_reference'
    assert layer.poll(1_012_000,state) is None
    event=layer.poll(1_023_000,replace(state,timestamp_us=1_010_000))
    assert event is None and layer.counts['stale_state_skips']==1
    assert layer.last_reason=='waiting for fresh measured state'
    layer.reset(); assert layer.forecast is None and layer.limits.previous_q is None
    with pytest.raises(ValueError,match='active'):
        WmControlLayer({**settings(),'execution':'active'},URDF)


def test_g2_geometry_and_dynamics_preserve_seven_axis_training_fk():
    pin=pytest.importorskip('pinocchio')
    from inference.dp_wm.models import fk_chain_signature, JointToPose
    bare=URDF.with_name('xarm7_dynamics.urdf')
    assert fk_chain_signature(bare,'link_base','link_eef')==fk_chain_signature(URDF,'link_base','link_eef')
    q=np.array([0,0,0,1.57,0,1.57,-1.57])
    np.testing.assert_allclose(JointToPose(bare)([q]),JointToPose(URDF)([q]),atol=1e-7)
    model=pin.buildModelFromUrdf(str(URDF))
    assert model.nq==model.nv==7
    assert list(model.names[1:])==[f'joint{i}' for i in range(1,8)]
    geom=pin.buildGeomFromUrdf(model,str(URDF),pin.GeometryType.VISUAL,package_dirs=[str(URDF.parent)])
    assert len(geom.geometryObjects)==15
    dyn=LocalDynamicsModel(URDF)
    mass,n,b=dyn.snapshot(q,np.zeros(7))
    a=np.arange(7)*.1
    np.testing.assert_allclose(mass@a+n,pin.rnea(dyn.model,dyn.data,q,np.zeros(7),a),atol=1e-10)
    modified=URDF.read_text().replace('<axis xyz="0 0 1" />','<axis xyz="0 1 0" />',1)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'changed.urdf';p.write_text(modified)
        assert fk_chain_signature(bare,'link_base','link_eef')!=fk_chain_signature(p,'link_base','link_eef')


def test_active_projection_history_only_advances_after_successful_send():
    layer = WmControlLayer({**settings(), 'execution': 'active'}, URDF, allow_active=True)
    state = ControlState(1_000_000, np.zeros(7), np.zeros(7))
    layer.update_forecast({'anchor_us': 1_000_000, 'plan_version': 1,
        'future_timestamps_us': 1_000_000 + np.arange(1, 33, dtype=np.int64)*10_000,
        'q': np.ones((1, 32, 7))*.1, 'tau': np.zeros((1, 32, 7))})
    packet = layer.poll(1_001_000, state)
    assert packet['valid'] and not packet['sent']
    assert layer.limits.previous_q is None
    with pytest.raises(ValueError):
        layer.mark_sent(packet)
    packet['sent'] = True
    layer.mark_sent(packet)
    np.testing.assert_equal(layer.limits.previous_q, packet['first_q_command'])
    assert layer.counts['sent'] == 1


def test_active_dp_gap_hold_is_bounded_and_uses_last_sent_target():
    base=settings()
    config={**base,'execution':'active',
            'limits':{**base['limits'],'max_jerk_rad_s3':5.,'q_only_smoothing':'jerk_tracking'},
            'gap_hold':{'enabled':True,'maximum_ms':30.}}
    layer=WmControlLayer(config,URDF,allow_active=True)
    anchor=1_000_000
    layer.update_forecast({'anchor_us':anchor,'plan_version':1,
        'future_timestamps_us':anchor+np.arange(1,33,dtype=np.int64)*10_000,
        'q':np.zeros((1,32,7)),'tau':np.zeros((1,32,7))})
    first=layer.poll(anchor+1_000,ControlState(anchor,np.zeros(7),np.zeros(7)))
    assert first['valid']
    first['sent']=True
    layer.mark_sent(first)
    layer.limits.previous_v=np.full(7,.1)
    stale=layer.poll(anchor+111_000,ControlState(anchor+110_000,np.zeros(7),np.zeros(7)))
    assert stale['valid'] and stale['actual_mode']=='position_hold'
    assert stale['reason']=='bounded DP gap position hold'
    assert np.max(np.abs(stale['first_q_command']-first['first_q_command']))<config['limits']['max_step_rad']
    assert np.max(np.abs(stale['command_jerk_plan']))<=5.+1e-9
    assert layer.limits.previous_q is not None
    stale['sent']=True
    layer.mark_sent(stale)
    expired=layer.poll(anchor+151_000,ControlState(anchor+150_000,np.zeros(7),np.zeros(7)))
    assert not expired['valid'] and 'hold exceeded' in expired['reason']
    assert layer.limits.previous_q is None


def test_stale_feedback_retries_new_sample_without_consuming_slot_or_sent_history():
    layer=WmControlLayer({**settings(),'execution':'active'},URDF,allow_active=True)
    forecast={'anchor_us':1_000_000,'plan_version':1,
        'future_timestamps_us':1_000_000+np.arange(1,33,dtype=np.int64)*10_000,
        'q':np.zeros((1,32,7)),'tau':np.zeros((1,32,7))}
    layer.update_forecast(forecast)
    first=layer.poll(1_001_000,ControlState(1_000_000,np.zeros(7),np.zeros(7)))
    first['sent']=True
    layer.mark_sent(first)
    previous=layer.limits.previous_q.copy()
    # This distinct sample is 10.132 ms old and its first target has elapsed.
    assert layer.poll(1_020_132,ControlState(1_010_000,np.zeros(7),np.zeros(7))) is None
    assert layer.next_us==1_011_000 and layer.last_state_us==1_000_000
    np.testing.assert_equal(layer.limits.previous_q,previous)
    # The next acquisition is accepted immediately, without waiting another 10 ms.
    result=layer.poll(1_020_500,ControlState(1_020_000,np.zeros(7),np.zeros(7)))
    assert result['valid'] and result['state_age_ms']==.5
    assert layer.counts['stale_state_skips']==1 and layer.counts['invalid']==0


def test_future_and_backwards_feedback_still_produce_invalid_results():
    layer=WmControlLayer(settings(),URDF)
    layer.update_forecast({'anchor_us':1_000_000,'plan_version':1,
        'future_timestamps_us':1_000_000+np.arange(1,33,dtype=np.int64)*10_000,
        'q':np.zeros((1,32,7)),'tau':np.zeros((1,32,7))})
    result=layer.poll(1_001_000,ControlState(1_002_000,np.zeros(7),np.zeros(7)))
    assert not result['valid']
    result=layer.poll(1_012_000,ControlState(1_000_000,np.zeros(7),np.zeros(7)))
    assert not result['valid'] and 'backwards' in result['reason']


@pytest.mark.parametrize('maximum_age, valid', [(100., False), (180., True)])
def test_contact_update_period_and_inference_latency_count_from_history_anchor(maximum_age, valid):
    # Reproduce the observed 80 ms refresh + 57 ms inference delay.
    layer = WmControlLayer({**settings(), 'maximum_forecast_age_ms': maximum_age}, URDF)
    anchor = 1_000_000
    layer.update_forecast({'anchor_us': anchor, 'plan_version': 1,
        'future_timestamps_us': anchor + np.arange(1, 33, dtype=np.int64)*10_000,
        'q': np.zeros((1, 32, 7))})
    state = ControlState(anchor + 137_000, np.zeros(7), np.zeros(7))
    event = layer.poll(state.timestamp_us + 1_000, state)
    assert event['valid'] is valid
    assert event['forecast_age_ms'] == 137.
    assert layer.report()['forecast_age_ms'] == 137.
    if not valid:
        assert event['reason'] == 'WM reference exceeded the configured age'
    else:
        expired_state = replace(state, timestamp_us=anchor + 187_000)
        expired = layer.poll(expired_state.timestamp_us + 1_000, expired_state)
        assert not expired['valid']
        assert expired['reason'] == 'WM reference exceeded the configured age'


def test_increased_age_limit_still_rejects_reference_outside_prediction_horizon():
    layer = WmControlLayer({**settings(), 'maximum_forecast_age_ms': 500.}, URDF)
    anchor = 1_000_000
    layer.update_forecast({'anchor_us': anchor, 'plan_version': 1,
        'future_timestamps_us': anchor + np.arange(1, 33, dtype=np.int64)*10_000,
        'q': np.zeros((1, 32, 7))})
    state = ControlState(anchor + 310_000, np.zeros(7), np.zeros(7))
    event = layer.poll(state.timestamp_us + 1_000, state)
    assert not event['valid']
    assert 'does not cover' in event['reason']
