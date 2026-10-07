#!/usr/bin/env python3
"""Audit saved WM training metadata and existing inference logs without hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit_run(directory, training=None):
    differences, displacement = [], []
    prediction_count = 0
    for path in sorted(directory.glob('prediction_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            prediction_count += 1
            if 'policy_condition_q' not in data:
                continue
            target_times = data['policy_condition_timestamps_us']
            wm_times = data['future_timestamps_us']
            valid = (target_times >= wm_times[0]) & (target_times <= wm_times[-1])
            if valid.any():
                # Relative timestamps avoid losing precision at UNIX microsecond scale.
                t = (wm_times-wm_times[0])/1e6
                target_t = (target_times[valid]-wm_times[0])/1e6
                q = np.stack([np.interp(target_t, t, data['q'][0, :, joint])
                              for joint in range(7)], axis=-1)
                differences.extend(np.abs(q-data['policy_condition_q'][valid]))
            if 'wm_history_q_anchor' in data:
                displacement.append(np.abs(data['q'][0, -1]-data['wm_history_q_anchor']))
    commands = sorted(directory.glob('controller_*.npz'))
    command_info = {}
    if commands:
        with np.load(commands[0]) as first, np.load(commands[-1]) as last:
            command_info = dict(first_q_command=first['first_q_command'].tolist(),
                                last_q_command=last['first_q_command'].tolist(),
                                sent_command_displacement_rad=(last['first_q_command']-first['first_q_command']).tolist())
    summary = directory/'summary.json'
    saved_config = directory/'config.json'
    historical_alignment = None
    if training is not None and saved_config.exists():
        from inference.dp_wm.models import alignment_report
        historical_alignment = alignment_report(json.loads(saved_config.read_text()), training)
    return dict(directory=str(directory), prediction_count=prediction_count,
                original_deployment_alignment=historical_alignment,
                wm_vs_policy_target_time_aligned_mae_rad=(np.mean(differences, axis=0).tolist() if differences else None),
                wm_320ms_displacement_from_history_mae_rad=(np.mean(displacement, axis=0).tolist() if displacement else None),
                interpretation='WM predicts measured response, not desired policy targets. Differences quantify changed references, not supervised WM error; these logs contain no future ground-truth labels.',
                commands=command_info, summary=json.loads(summary.read_text()) if summary.exists() else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    from inference.dp_wm.config import load_config
    from inference.dp_wm.models import inspect_checkpoints
    config = load_config(args.config)
    metadata = inspect_checkpoints(config)
    report = dict(hardware_connected=False, config=str(Path(args.config).resolve()),
                  checkpoint=str(config['wm']['checkpoint']), **metadata['alignment'])
    if args.run_dir is not None:
        report['existing_run'] = audit_run(args.run_dir, metadata['wm_training'])
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')
    print(rendered)
    return report


if __name__ == '__main__':
    main()
