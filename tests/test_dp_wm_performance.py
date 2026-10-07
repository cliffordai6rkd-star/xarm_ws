from collections import deque
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference.dp_wm.event_writer import EventWriter


def packet():
    return dict(kind='controller', anchor_us=1, plan_version=2, computed_at_us=3,
                first_q_command=np.arange(7,dtype=float), sent=True, valid=True)


@pytest.mark.parametrize('asynchronous', [False, True])
def test_event_writer_drains_ordered_snapshots_and_keeps_metadata(tmp_path, asynchronous):
    writer = EventWriter(tmp_path, mock=True, command_source='controller_reference', asynchronous=asynchronous)
    event = packet()
    writer.write(event)
    event['first_q_command'][:] = -99
    writer.write({'kind':'status','active':True})
    writer.close()
    rows = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [r['kind'] for r in rows] == ['controller','status']
    assert rows[0]['sent']
    with np.load(tmp_path/rows[0]['file']) as data:
        np.testing.assert_equal(data['first_q_command'],np.arange(7))
        assert data['completed_us']==3


def test_writer_failure_propagates_after_cleanup(tmp_path, monkeypatch):
    def fail(*args,**kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(np,'savez',fail)
    writer = EventWriter(tmp_path,mock=True,command_source='controller_reference')
    writer.write(packet())
    with pytest.raises(RuntimeError,match='recording failed'):
        writer.close()
    assert writer.log.closed


@pytest.mark.parametrize('asynchronous',[False,True])
def test_policy_actions_and_stop_reason_survive_recording(tmp_path,asynchronous):
    writer=EventWriter(tmp_path,mock=True,command_source='controller_reference',asynchronous=asynchronous)
    actions=np.arange(105,dtype=float).reshape(15,7)
    times=1_000_000+np.arange(15,dtype=np.int64)*40_000
    writer.write(dict(kind='dp',version=3,anchor_us=1_000_000,completed_us=1_050_000,
                      policy_q=actions,policy_timestamps_us=times,policy_clock='observation',usable_now=True))
    actions[:]=-999;times[:]=0
    writer.write(dict(kind='output_stop',timestamp_us=1_060_000,
                      reason='invalid controller result: WM reference exceeded the configured age',
                      compute_ms=.004,forecast_age_ms=181.314))
    writer.close()
    rows=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert rows[0]['file']=='policy_000000.npz'
    with np.load(tmp_path/rows[0]['file'],allow_pickle=False) as f:
        np.testing.assert_equal(f['policy_q'],np.arange(105).reshape(15,7))
        np.testing.assert_equal(f['policy_timestamps_us'],1_000_000+np.arange(15)*40_000)
        assert f['plan_version']==3
    assert rows[1]['kind']=='output_stop' and rows[1]['forecast_age_ms']==181.314


def test_writer_backpressure_is_explicit_and_does_not_block_control(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = EventWriter._write
    def delayed(self,event):
        started.set()
        assert release.wait(2)
        original(self,event)
    monkeypatch.setattr(EventWriter,'_write',delayed)
    writer = EventWriter(tmp_path,mock=True,command_source='controller_reference',maximum_pending=1)
    try:
        writer.write(packet())
        assert started.wait(1)
        writer.write(packet())
        with pytest.raises(RuntimeError,match='queue is full'):
            writer.write(packet())
    finally:
        release.set()
        writer.close()


def test_source_history_cache_refreshes_and_does_not_keep_invalid_history():
    from inference.dp_wm.live import LiveSource
    source = LiveSource.__new__(LiveSource)
    source.error=None
    source.cameras=SimpleNamespace(poll=lambda: [])
    source.frames={}
    source.states=deque()
    source.rows=deque([(i*10_000,{key:np.ones(7)*i for key in ('q','dq','delta_q','tau')})
                       for i in range(1,51)],maxlen=50)
    source.observations=deque()
    source.lock=threading.Lock()
    source.history_cache=None
    source.observation_after_us=0
    source.single_observation=False
    first=source.poll()[1]
    assert source.poll()[1] is first
    source.rows.append((510_000,{key:np.ones(7)*51 for key in ('q','dq','delta_q','tau')}))
    second=source.poll()[1]
    assert second is not first
    np.testing.assert_equal(first.values['q'][-1],50)
    np.testing.assert_equal(second.values['q'][-1],51)
    source.rows.clear()
    assert source.poll()[1] is None and source.history_cache is None


def observation_source(monkeypatch):
    import inference.dp_wm.live as module
    source=module.LiveSource.__new__(module.LiveSource)
    source.error=None;source.frames={'front':deque(maxlen=20)};source.mapping={'front':'front'}
    source.states=deque(maxlen=100);source.rows=deque(maxlen=50);source.observations=deque(maxlen=20)
    source.lock=threading.Lock();source.history_cache=None;source.observation_after_us=0
    source.single_observation=False
    source.options=dict(maximum_age_ms=150,maximum_camera_skew_ms=60)
    clock=SimpleNamespace(us=1_000_000,frame_stamp=1_000_000)
    monkeypatch.setattr(module,'now_us',lambda:clock.us)
    source.cameras=SimpleNamespace(poll=lambda:[SimpleNamespace(camera_name='front',
        timestamp_us=clock.frame_stamp,frame=np.zeros((2,2,3),np.uint8))])
    return source,clock


@pytest.mark.parametrize('gaps',[[33_333]*20,[33_333,33_333,66_667,33_333,33_333]*4])
def test_dp_observation_pair_accepts_real_30hz_and_rate_limited_25hz_timestamps(monkeypatch,gaps):
    source,clock=observation_source(monkeypatch)
    source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)))
    assert source.poll()[0]==()
    previous_stamp=clock.frame_stamp
    for gap in gaps:
        clock.us+=gap;clock.frame_stamp=clock.us
        source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)))
        observations,_=source.poll()
        assert len(observations)==2
        assert observations[0].timestamp_us==previous_stamp
        assert observations[1].timestamp_us==clock.frame_stamp
        assert observations[0].timestamp_us<observations[1].timestamp_us
        previous_stamp=clock.frame_stamp


def test_dp_pair_still_rejects_missing_or_stale_camera_data(monkeypatch):
    source,clock=observation_source(monkeypatch)
    source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)));source.poll()
    clock.us+=125_000;clock.frame_stamp=clock.us
    source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)))
    assert source.poll()[0]==()
    clock.us+=33_333;clock.frame_stamp=clock.us
    source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)))
    assert len(source.poll()[0])==2
    source.cameras=SimpleNamespace(poll=lambda:[])
    clock.us+=151_000
    assert source.poll()[0]==()


def test_dp_pair_does_not_reuse_pre_reset_observation(monkeypatch):
    source,clock=observation_source(monkeypatch)
    source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)));source.poll()
    source.observation_after_us=clock.us+1
    clock.us+=33_333;clock.frame_stamp=clock.us
    source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)))
    assert source.poll()[0]==()
    clock.us+=33_333;clock.frame_stamp=clock.us
    source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)))
    assert len(source.poll()[0])==2


def test_pi0_single_observation_does_not_require_a_camera_pair(monkeypatch):
    source,clock=observation_source(monkeypatch)
    source.single_observation=True
    source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)))
    observations,_=source.poll()
    assert len(observations)==1
    assert source.observation_status=='ready; single policy observation'
    source.observation_after_us=clock.us
    assert source.poll()[0]==()
    assert source.observation_status=='waiting for single policy observation after reset/start'


def test_pi0_stale_camera_reports_camera_and_age_instead_of_pair_spacing(monkeypatch):
    source,clock=observation_source(monkeypatch)
    source.single_observation=True
    source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)))
    assert len(source.poll()[0])==1
    source.cameras=SimpleNamespace(poll=lambda:[])
    # Retaining a still-fresh observation is intentional; its timestamp stays real.
    clock.us+=150_000
    observations,_=source.poll()
    assert len(observations)==1 and observations[0].timestamp_us==clock.frame_stamp
    clock.us+=1_000
    assert source.poll()[0]==()
    assert source.observation_status=='camera frames stale: front age_ms=151.000; maximum_age_ms=150.000'
    assert 'pair' not in source.observation_status


@pytest.mark.parametrize('case,reason',[
    ('missing_camera','waiting for camera frames: wrist'),
    ('missing_state','waiting for causal measured state at camera anchor'),
    ('state_skew','camera/state skew_ms=151.000 exceeds maximum_age_ms=150.000'),
    ('unmatched_camera','waiting for camera frames at anchor: wrist'),
    ('camera_skew','camera frames exceed maximum_camera_skew_ms=60.000: wrist skew_ms=100.000'),
    ('future_camera','camera anchor is ahead of host time; ahead_ms=1.000'),
])
def test_live_observation_failure_identifies_actual_camera_or_state_condition(monkeypatch,case,reason):
    source,clock=observation_source(monkeypatch)
    source.single_observation=True
    source.cameras=SimpleNamespace(poll=lambda:[])
    def frame(stamp):
        return SimpleNamespace(timestamp_us=stamp,frame=np.zeros((2,2,3),np.uint8))
    source.frames['front'].append(frame(clock.us))
    source.states.append(SimpleNamespace(timestamp_us=clock.us,q=np.zeros(7)))
    if case in ('missing_camera','unmatched_camera','camera_skew'):
        source.frames['wrist']=deque(maxlen=20)
        source.mapping['wrist']='wrist'
    if case=='missing_state':
        source.states.clear()
        source.states.append(SimpleNamespace(timestamp_us=clock.us+1,q=np.zeros(7)))
    elif case=='state_skew':
        source.states.clear()
        source.states.append(SimpleNamespace(timestamp_us=clock.us-151_000,q=np.zeros(7)))
    elif case=='unmatched_camera':
        source.frames['wrist'].append(frame(clock.us+1_000))
    elif case=='camera_skew':
        source.frames['wrist'].extend([frame(clock.us-100_000),frame(clock.us+1_000)])
    elif case=='future_camera':
        source.frames['front'].clear()
        source.frames['front'].append(frame(clock.us+1_000))
    assert source.poll()[0]==()
    assert source.observation_status==reason


def test_vision_cache_reuses_only_identical_immutable_frames_and_refreshes_new_frames():
    from inference.dp_wm.vision_cache import ObservationFeatureCache
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batches=[]
        def forward(self,batch):
            self.batches.append(batch['front'].shape[0])
            return batch['front'].flatten(1).sum(1,keepdim=True)
    class Policy:
        n_obs_steps=2
        image_keys=['front']
        def __init__(self):
            self.model=SimpleNamespace(obs_encoder=Encoder().eval())
        def predict_observations(self,observations,**kwargs):
            data=torch.from_numpy(np.stack([obs.images['front'] for obs in observations]))
            with torch.inference_mode():
                return self.model.obs_encoder({'front':data})
    def obs(stamp,value):
        image=np.full((2,2,3),value,dtype=np.float32)
        image.setflags(write=False)
        return SimpleNamespace(timestamp_us=stamp,images={'front':image})
    policy=Policy()
    cache=ObservationFeatureCache(policy,maximum_entries=2)
    a,b,c=obs(0,1),obs(40_000,2),obs(80_000,3)
    torch.testing.assert_close(policy.predict_observations((a,b)),torch.tensor([[12.],[24.]]))
    torch.testing.assert_close(policy.predict_observations((b,c)),torch.tensor([[24.],[36.]]))
    assert policy.model.obs_encoder.batches == [2,1]
    assert cache.hits==1 and cache.misses==3 and len(cache.cache)==2
    # Equal timestamps alone are not sufficient: another object must be encoded.
    replacement=obs(80_000,4)
    torch.testing.assert_close(policy.predict_observations((c,replacement)),torch.tensor([[36.],[48.]]))
    replacement.images['front'].setflags(write=True)
    replacement.images['front'][:]=5
    torch.testing.assert_close(policy.predict_observations((c,replacement)),torch.tensor([[36.],[60.]]))
    assert policy.model.obs_encoder.batches[-1]==2
    replacement.images['front'].setflags(write=False)
    torch.testing.assert_close(policy.predict_observations((c,replacement)),torch.tensor([[36.],[60.]]))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_contact_graph_refreshes_conditioning_and_source_noise():
    from inference.dp_wm.cuda_sampler import CudaContactSampler
    class Model:
        training=False
        runtime_checks=False
        inputs=['q']
        future_horizon=2
        flow_dim=1
        def sample(self,batch,*,source_noise,**kwargs):
            return {'q_pred':batch['q'][:,None]+source_noise+batch['action'][:,None]}
    model=Model()
    sampler=CudaContactSampler(model,steps=2,solver='heun',num_samples=2)
    with torch.inference_mode():
        batch={'q':torch.ones((1,2,1),device='cuda'),'action':torch.ones((1,2,1),device='cuda')}
        noise=torch.ones((1,2,2,1),device='cuda')
        torch.testing.assert_close(sampler.sample(batch,source_noise=noise)['q_pred'],torch.full_like(noise,3))
        batch['q']+=4
        noise+=7
        torch.testing.assert_close(sampler.sample(batch,source_noise=noise)['q_pred'],torch.full_like(noise,14))
        with pytest.raises(ValueError,match='contract changed'):
            sampler.sample({**batch,'action':batch['action'].double()},source_noise=noise)


@pytest.mark.parametrize('mode', ['gaussian','limit','quantile'])
def test_cached_statistics_match_original_scaling(mode):
    from inference.dp_wm.models import add_pinn_source
    root=Path(__file__).resolve().parents[2]/'PINN_LCP'
    if not root.exists():
        pytest.skip('PINN source checkout required')
    add_pinn_source(root)
    from model.pinn_model.latent_pretrained import convert_scale
    from inference.dp_wm.scaling import CheckpointScaler
    stats={'mean':torch.arange(7.),'std':torch.ones(7)*2,'min':torch.zeros(7),'max':torch.ones(7)*3,
           'q01':torch.zeros(7),'q99':torch.ones(7)*3}
    envelope=dict(stats={'q':stats},normalize_mode=mode,normalize_lowdim_keys=['q'],eps=1e-6)
    scaler=CheckpointScaler(envelope,'cpu')
    values=torch.arange(21.).reshape(3,7)/5
    torch.testing.assert_close(scaler('q',values),convert_scale('q',values,envelope))
    if mode!='quantile':
        torch.testing.assert_close(scaler('q',values,inverse=True),convert_scale('q',values,envelope,inverse=True))
    else:
        with pytest.raises(ValueError,match='cannot be inverted'):
            scaler('q',values,inverse=True)
