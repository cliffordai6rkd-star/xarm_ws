"""Hardware-free, warmed end-to-end DP/WM timing and CUDA replay equivalence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from inference.dp_wm.config import load_config
    from inference.dp_wm.models import inspect_checkpoints, load_models
    from inference.dp_wm.mock import MockSource
    from inference.dp_wm.core import DP_PLAN_STEPS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='inference/configs/dp_wm_xarm.yaml')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--output', type=Path, default=Path('runs/dp_wm_benchmark.json'))
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    cfg = load_config(args.config)
    if cfg['pipeline']=='dp_contact_wm':
        from benchmark_dp_contact_wm import main as contact_benchmark
        return contact_benchmark(['--config',args.config,'--iterations',str(args.iterations),
                                  '--output',str(args.output)])
    cfg['wm']['cuda_graph'] = True
    policy, wm = load_models(cfg, inspect_checkpoints(cfg))
    if wm.sampler is None:
        parser.error('This comparison requires a CUDA WM device')
    observations, history = MockSource().poll()
    request = dict(history=history, actions=np.zeros((8, 7)),
        action_timestamps_us=history.anchor_us+np.arange(1, 9)*40_000,
        plan_version=1, deadline_us=history.anchor_us+40_000)
    sampler, stream = wm.sampler, wm.cuda_stream
    saved = {}
    original = sampler.sample
    def capture(batch, **kwargs):
        saved['batch'] = batch
        return original(batch, **kwargs)
    sampler.sample = capture
    wm.infer(request)
    sampler.sample = original
    errors = []
    with torch.inference_mode(), torch.cuda.stream(stream):
        for offset in (0., .123):
            batch = {key: value.clone() for key, value in saved['batch'].items()}
            batch['q'] += offset
            batch['action'] += offset
            noise = torch.randn((1, cfg['wm']['num_samples'], wm.model.future_horizon,
                                 wm.model.latent_dim), device=wm.model.latent_mean.device)
            with torch.autocast('cuda', dtype=torch.float16 if cfg['wm']['precision']=='float16' else torch.bfloat16,
                                enabled=cfg['wm']['precision']!='float32', cache_enabled=False):
                expected = wm.model.sample(batch, source_noise=noise, steps=cfg['wm']['flow_steps'],
                    solver=cfg['wm']['solver'], num_samples=cfg['wm']['num_samples'])
                actual = sampler.sample(batch, source_noise=noise)
            errors.append({key: float((expected[key]-actual[key]).abs().max())
                           for key in ('q_pred','tau_pred','contact_probability')})
            for key in errors[-1]:
                torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)

    def timed(function):
        start = time.perf_counter()
        function()  # Both adapters return CPU results, hence synchronize GPU output.
        return (time.perf_counter()-start)*1000

    def stats(values):
        return dict(p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values,95)),
                    samples_ms=values)

    result = dict(checkpoint=str(cfg['wm']['checkpoint']), gpu=torch.cuda.get_device_name(),
        precision=cfg['wm']['precision'], solver=cfg['wm']['solver'], flow_steps=cfg['wm']['flow_steps'],
        num_samples=cfg['wm']['num_samples'], iterations=args.iterations, warmup=5,
        synthetic_inputs=True, hardware_connected=False, equivalence_max_abs_errors=errors)
    for name, graph, cuda_stream in [('baseline', None, None), ('optimized', sampler, stream)]:
        wm.sampler, wm.cuda_stream = graph, cuda_stream
        call_wm = lambda: wm.infer(request)
        call_dp = lambda: policy.predict_observations(observations, plan_steps=DP_PLAN_STEPS)
        single = [timed(call_wm) for _ in range(args.iterations+5)][5:]
        pairs = []
        with ThreadPoolExecutor(2) as pool:
            for _ in range(args.iterations+5):
                a, b = pool.submit(timed, call_wm), pool.submit(timed, call_dp)
                pairs.append([a.result(),b.result()])
        result[name] = dict(wm_alone=stats(single),
            concurrent_wm=stats([p[0] for p in pairs[5:]]), concurrent_dp=stats([p[1] for p in pairs[5:]]))
        print(name, {key: {k:v for k,v in value.items() if k!='samples_ms'}
                     for key,value in result[name].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Equivalence passed; report: {args.output}')


if __name__ == '__main__':
    main()
