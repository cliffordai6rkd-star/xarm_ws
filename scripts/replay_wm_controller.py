#!/usr/bin/env python3
"""Offline reference shaping from a saved WM NPZ and recorded xArm H5 state."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inference.dp_wm.config import load_config
from inference.core import WmControlLayer
from inference.controllers.wm_two_step import ControlState


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='inference/configs/dp_wm_xarm.yaml')
    parser.add_argument('--forecast',type=Path,required=True,help='prediction_*.npz from the WM logger')
    parser.add_argument('--states',type=Path,required=True,help='recorded H5 with matching absolute timestamps')
    parser.add_argument('--mode',choices=('q_only','tau_only','joint_q_tau','mit_position'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=load_config(args.config)
    settings=dict(cfg['controller'],execution='offline')
    if args.mode: settings['mode']=args.mode
    with np.load(args.forecast,allow_pickle=False) as archive:
        forecast={key:archive[key].copy() for key in archive.files}
    forecast['anchor_us']=int(forecast['anchor_us'])
    start=forecast['anchor_us']; end=int(forecast['future_timestamps_us'][-1])-20_000
    side=cfg['observation']['arm']
    with h5py.File(args.states,'r') as file:
        rows=file['teleop']
        timestamps=np.asarray(rows['timestamp_us']).reshape(-1)
        selected=np.flatnonzero((timestamps>=start)&(timestamps<=end))
        if not len(selected):
            parser.error('Recorded state and WM forecast timestamps do not overlap; timestamps are never reanchored')
        values={key:np.asarray(rows[f'{side}_{name}_xarm'])[selected] for key,name in (
            ('q','q'),('dq','dq'),('tau','tau'))}
        valid=np.ones(len(selected),bool)
        for signal in ('q_valid','dq_valid','torque_valid'):
            key=f'{side}_{signal}_xarm'
            if key in rows:
                valid &= np.asarray(rows[key])[selected].reshape(-1).astype(bool)
        source_times=(np.asarray(rows[f'{side}_q_timestamp_us_xarm'])[selected].reshape(-1)
                      if f'{side}_q_timestamp_us_xarm' in rows else timestamps[selected])
        now_times=timestamps[selected]
    layer=WmControlLayer(settings,cfg['wm']['urdf_path'])
    layer.update_forecast(forecast)
    args.output.mkdir(parents=True,exist_ok=False)
    latencies=[]
    with (args.output/'events.jsonl').open('w') as log:
        for i in range(len(selected)):
            if not valid[i]: continue
            state=ControlState(int(source_times[i]),values['q'][i],values['dq'][i],values['tau'][i])
            before=time.perf_counter();event=layer.poll(int(now_times[i]),state)
            latencies.append((time.perf_counter()-before)*1000)
            if event is None: continue
            arrays={key:value for key,value in event.items() if isinstance(value,np.ndarray)}
            name=f'controller_{i:06d}.npz';np.savez(args.output/name,**arrays)
            packet={key:value for key,value in event.items() if not isinstance(value,np.ndarray)}
            packet['file']=name
            log.write(json.dumps(packet)+'\n')
    summary=layer.report()
    if latencies:
        summary.update(p50_ms=float(np.percentile(latencies,50)),p95_ms=float(np.percentile(latencies,95)),
                       p99_ms=float(np.percentile(latencies,99)),max_ms=float(max(latencies)))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
