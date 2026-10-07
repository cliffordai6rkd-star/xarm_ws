"""Real models, synthetic observations, no hardware: acceleration and pipeline benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def stats(values):
    return dict(p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))


def timed(function):
    start=time.perf_counter()
    function()
    return (time.perf_counter()-start)*1000


def main(argv=None):
    import torch
    from inference.dp_wm.config import load_config
    from inference.dp_wm.models import inspect_checkpoints,load_models,warmup_inputs
    from inference.dp_wm.core import AsyncDPWM,DP_PLAN_STEPS
    from inference.dp_wm.cuda_sampler import CudaContactSampler
    from inference.dp_wm.event_writer import EventWriter
    from inference.dp_wm.mock import MockSource
    from inference.core.wm_control import WmControlLayer
    from nero_collection.time_utils import now_us
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='inference/configs/dp_contact_wm_xarm.yaml')
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--duration',type=float,default=5.)
    parser.add_argument('--output',type=Path,default=Path('runs/dp_contact_wm_benchmark.json'))
    args=parser.parse_args(argv)
    if args.iterations<1 or not np.isfinite(args.duration) or args.duration<=0:
        parser.error('iterations and duration must be positive')
    config=load_config(args.config)
    if config['pipeline']!='dp_contact_wm' or not config['wm']['device'].startswith('cuda'):
        parser.error('requires dp_contact_wm on CUDA')
    baseline=copy.deepcopy(config)
    baseline['wm'].update(cuda_graph=False,cache_condition_kv=False,precision='float32')
    policy,wm=load_models(baseline,inspect_checkpoints(baseline))
    observations,request=warmup_inputs(policy._image_shapes)
    feature_cache=getattr(policy,'feature_cache',None)
    def next_observation(previous):
        images={key:value.copy() for key,value in previous.images.items()}
        for image in images.values():
            image.setflags(write=False)
        return SimpleNamespace(timestamp_us=previous.timestamp_us+40_000,q=previous.q.copy(),images=images,metadata={})
    dp_feature_errors=[]
    if feature_cache is not None:
        for window in (observations,(observations[-1],next_observation(observations[-1]))):
            feature_cache.enabled=False
            torch.manual_seed(123)
            expected=policy.predict_observations(window,plan_steps=DP_PLAN_STEPS).values
            feature_cache.enabled=True
            torch.manual_seed(123)
            actual=policy.predict_observations(window,plan_steps=DP_PLAN_STEPS).values
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
            dp_feature_errors.append(float(np.max(np.abs(actual-expected))))
    current_window=list(observations)
    def call_dp():
        previous=current_window[-1]
        current_window[:]=[previous,next_observation(previous)]
        return policy.predict_observations(tuple(current_window),plan_steps=DP_PLAN_STEPS)
    # Nonzero inputs ensure graph replay refreshes changing observations.
    rng=np.random.default_rng(42)
    request['history'].values['q'][:]=[0,0,0,1.57,0,1.57,-1.57]
    request['actions'][:]=[0,0,0,1.57,0,1.57,-1.57]
    variants={}
    samplers={}
    errors={}
    scaler=wm.scaler
    with torch.inference_mode(),torch.cuda.stream(wm.stream):
        raw={key:torch.as_tensor(value[None],device=wm.device) for key,value in request['history'].values.items()}
        raw['action']=torch.as_tensor(wm.fk(request['actions'])[None],device=wm.device)
        batch={key:scaler(key,value) for key,value in raw.items()}
        cases=[('eager_fp32','float32',False,False),('kv_fp32','float32',True,False),
               ('graph_kv_fp32','float32',True,True),('graph_kv_bfloat16','bfloat16',True,True),
               ('graph_kv_float16','float16',True,True)]
        for name,precision,cache,graph in cases:
            if graph:
                samplers[name]=CudaContactSampler(wm.model,steps=config['wm']['flow_steps'],
                    solver=config['wm']['solver'],num_samples=config['wm']['num_samples'],cache_condition_kv=cache)
            errors[name]=[]
        for _ in range(3):
            inputs={key:value.clone() for key,value in batch.items()}
            inputs['q']+=torch.as_tensor(rng.normal(0,.02,inputs['q'].shape),dtype=torch.float32,device=wm.device)
            inputs['action']+=torch.as_tensor(rng.normal(0,.01,inputs['action'].shape),dtype=torch.float32,device=wm.device)
            noise=torch.randn((1,config['wm']['num_samples'],wm.model.future_horizon,wm.model.flow_dim),device=wm.device)
            reference=wm.model.sample(inputs,source_noise=noise,steps=config['wm']['flow_steps'],
                solver=config['wm']['solver'],num_samples=config['wm']['num_samples'])
            for name,precision,cache,graph in cases:
                dtype=torch.float16 if precision=='float16' else torch.bfloat16
                with torch.autocast('cuda',dtype=dtype,enabled=precision!='float32',cache_enabled=False):
                    native=wm.model.sample(inputs,source_noise=noise,steps=config['wm']['flow_steps'],
                        solver=config['wm']['solver'],num_samples=config['wm']['num_samples'],cache_condition_kv=cache)
                    output=samplers[name].sample(inputs,source_noise=noise) if graph else native
                if graph:
                    for key in ('q_pred','tau_pred','contact_probability'):
                        torch.testing.assert_close(output[key],native[key],rtol=1e-5,atol=1e-6)
                row={}
                for key in ('q','tau'):
                    value=scaler(key,output[key+'_pred'].float(),inverse=True)
                    original=scaler(key,reference[key+'_pred'].float(),inverse=True)
                    if not torch.isfinite(value).all():
                        raise ValueError(f'Nonfinite {name} {key}')
                    row[key]=dict(max_abs=float((value-original).abs().max()),
                                  rms=float(((value-original)**2).mean().sqrt()))
                row['contact_max_abs']=float((output['contact_probability']-reference['contact_probability']).abs().max())
                errors[name].append(row)

    def select(name,precision,cache,graph):
        wm.config={**config['wm'],'precision':precision,'cache_condition_kv':cache}
        wm.sampler=samplers.get(name) if graph else None

    for case in cases:
        name=case[0]
        select(*case)
        if feature_cache is not None:
            feature_cache.enabled=name!='eager_fp32'
        call_wm=lambda:wm.infer(request)
        single=[timed(call_wm) for _ in range(args.iterations+5)][5:]
        pairs=[]
        with ThreadPoolExecutor(2) as pool:
            for _ in range(args.iterations+5):
                a,b=pool.submit(timed,call_wm),pool.submit(timed,call_dp)
                pairs.append([a.result(),b.result()])
        variants[name]=dict(wm_alone=stats(single),concurrent_wm=stats([p[0] for p in pairs[5:]]),
                            concurrent_dp=stats([p[1] for p in pairs[5:]]))
        print(name,variants[name],flush=True)

    # Include scheduling, feedback generation, shared controller and real NPZ/JSON
    # recording. No arm/camera SDK is imported or connected by this loop.
    class ImmutableMockSource(MockSource):
        def __init__(self):
            self.observation_cache={}
        def poll(self):
            obs,history=super().poll()
            selected=[]
            for value in obs:
                if value.timestamp_us not in self.observation_cache:
                    for image in value.images.values():
                        image.setflags(write=False)
                    self.observation_cache[value.timestamp_us]=value
                selected.append(self.observation_cache[value.timestamp_us])
            while len(self.observation_cache)>8:
                del self.observation_cache[min(self.observation_cache)]
            return tuple(selected),history
    loops={}
    for mode,case in [('baseline',cases[0]),('optimized',next(c for c in cases if c[1]==config['wm']['precision'] and c[3]))]:
        select(*case)
        if feature_cache is not None:
            feature_cache.enabled=mode=='optimized'
        settings=copy.deepcopy(config['controller'])
        settings['execution']='shadow'
        layer=WmControlLayer(settings,config['wm']['urdf_path'])
        pipeline=AsyncDPWM(policy,wm,execute_steps=config['wm']['execute_steps'],
            initial_latency_ms=config['wm']['initial_latency_ms'],margin_ms=config['wm']['margin_ms'],aligned_actions=True)
        pipeline.warmup(observations,request)
        source=ImmutableMockSource()
        with tempfile.TemporaryDirectory(prefix='dp-contact-benchmark-') as directory:
            writer=EventWriter(directory,mock=True,command_source='synthetic',asynchronous=mode=='optimized')
            start=time.monotonic()
            values=[]
            try:
                while time.monotonic()-start<args.duration:
                    t=time.perf_counter()
                    obs,history=source.poll()
                    events=pipeline.poll(now_us(),obs,history)
                    for event in events:
                        if event['kind']=='wm':
                            layer.update_forecast(event)
                    event=layer.poll(now_us(),source.control_state())
                    if event is not None:
                        events.append(event)
                    for event in events:
                        writer.write(event)
                    writer.check_error()
                    values.append((time.perf_counter()-t)*1000)
                    time.sleep(.001)
                elapsed=time.monotonic()-start
                loops[mode]=dict(loop=stats(values),elapsed_s=elapsed,wm_hz=pipeline.counts['wm_predictions']/elapsed,
                    dp_hz=pipeline.counts['dp_plans']/elapsed,pipeline=pipeline.report(),controller=layer.report(),logging=writer.report())
            finally:
                pipeline.close()
                writer.close()
        print('pipeline',mode,loops[mode],flush=True)
    report=dict(hardware_connected=False,synthetic_observations=True,gpu=torch.cuda.get_device_name(),
        wm_checkpoint=str(config['wm']['checkpoint']),flow_steps=config['wm']['flow_steps'],solver=config['wm']['solver'],
        iterations=args.iterations,variants=variants,fixed_noise_errors=errors,pipeline=loops,
        dp_cached_feature_max_abs_error_rad=dp_feature_errors)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Report: {args.output}')


if __name__=='__main__':
    main()
