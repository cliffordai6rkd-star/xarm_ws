#!/usr/bin/env python3
"""Replay contact-free H5 episodes with the deployed tau_free preprocessing.

This is an offline diagnostic: it never connects to a robot or changes its config.
Outputs physical residual statistics and per-episode CSV tables. Batched windows
use the deployed predictor's model and normalization, and are checked against
its streaming API before statistics are produced.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nero_collection.config import SequenceCheckpointConfig
from nero_collection.h5_schema import dataset_candidates
from nero_collection.filters import CausalFilterPipeline
from nero_collection.tau_ext_inference import SequenceTorquePredictor
from gello_teleop.tau_free_feedback import ResidualCurrentFeedback


def read_arm(group, key, arm, arm_names):
    for candidate in dataset_candidates(arm, key):
        if candidate in group:
            values = group[candidate][:]
            if candidate != key:
                return values
            break
    else:
        raise KeyError(f'{arm} channel {key!r} is missing')
    if values.ndim == 2 and values.shape[1] == 7 * len(arm_names):
        index = arm_names.index(arm)
        return values[:, index * 7:(index + 1) * 7]
    if values.ndim == 2 and values.shape[1] == len(arm_names):
        return values[:, [arm_names.index(arm)]]
    return values


def replay(predictor, timestamps, inputs, valid, max_gap_s, tau_filter):
    torch = predictor._torch
    horizon = predictor.metadata.horizon
    filters = {
        key: CausalFilterPipeline(spec['operations'])
        for key, spec in predictor.metadata.dataloader_filters.items()
        if spec.get('enabled') and (key in predictor.metadata.input_keys
                                   or (key == 'tau' and tau_filter == 'checkpoint'))
    }
    processed = {key: np.full_like(value, np.nan) for key, value in inputs.items()}
    endpoints = []
    count = 0
    for i, timestamp in enumerate(timestamps):
        gap = i > 0 and not 0 < timestamp - timestamps[i-1] <= max_gap_s * 1e6
        if gap or not valid[i]:
            for pipeline in filters.values():
                pipeline.reset()
            count = 0
        if not valid[i]:
            continue
        for key, values in inputs.items():
            processed[key][i] = (filters[key].apply(values[i], int(timestamp))
                                  if key in filters else values[i])
        count += 1
        if count >= horizon:
            endpoints.append(i)
    endpoints = np.asarray(endpoints, dtype=int)
    if not len(endpoints):
        raise ValueError('episode has no complete valid prediction window')
    prediction = np.full_like(inputs['tau'], np.nan)
    # Private predictor operations are intentionally confined to this diagnostic,
    # preserving exactly the same trained model, normalization, and window layout.
    tensors = [predictor._normalize(key, torch.as_tensor(
        processed[key], dtype=torch.float32, device=predictor._device))
        for key in predictor.metadata.input_keys]
    windows = torch.cat(tensors, dim=-1).unfold(0, horizon, 1).permute(0, 2, 1)
    with torch.inference_mode():
        for start in range(0, len(endpoints), 256):
            indices = endpoints[start:start+256]
            output = predictor._model(windows[indices - horizon + 1].contiguous())
            output = predictor._denormalize(predictor.metadata.output_key, output)
            prediction[indices] = output.cpu().numpy()
    # Check representative windows via the actual online API, without zero padding.
    for end in endpoints[np.linspace(0, len(endpoints)-1, min(3, len(endpoints))).astype(int)]:
        predictor.reset()
        for index in range(end-horizon+1, end+1):
            online = predictor.append_and_predict({k: processed[k][index]
                                                  for k in predictor.metadata.input_keys})
        np.testing.assert_allclose(prediction[end], online, atol=2e-5, rtol=2e-5)
    return prediction, processed['tau'] - prediction, endpoints


def statistics(residual, dq, controller, total_limit):
    magnitude = np.abs(residual)
    threshold = controller.threshold
    active = magnitude > threshold
    demand = np.where(active, magnitude * controller.gain, 0.)
    result = {
        'n_valid': len(residual),
        'bias_nm': np.mean(residual, axis=0).tolist(),
        'rmse_nm': np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        'abs_p50_nm': np.quantile(magnitude, .5, axis=0).tolist(),
        'abs_p95_nm': np.quantile(magnitude, .95, axis=0).tolist(),
        'abs_p99_nm': np.quantile(magnitude, .99, axis=0).tolist(),
        'abs_p995_nm': np.quantile(magnitude, .995, axis=0).tolist(),
        'abs_p999_nm': np.quantile(magnitude, .999, axis=0).tolist(),
        'abs_max_nm': np.max(magnitude, axis=0).tolist(),
        'trigger_fraction_current': active.mean(axis=0).tolist(),
        'any_joint_trigger_fraction_current': active.any(axis=1).mean().item(),
        'settled_feedback_demand_reaches_total_cap_fraction': (
            demand >= total_limit).mean(axis=0).tolist(),
    }
    for label, mask in [('slow', np.max(np.abs(dq), axis=1) < .05),
                        ('moving', np.max(np.abs(dq), axis=1) >= .05)]:
        result[f'{label}_n'] = int(mask.sum())
        if np.any(mask):
            result[f'{label}_abs_p99_nm'] = np.quantile(magnitude[mask], .99, axis=0).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', nargs='*', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'gello_teleop/config/xarm7_gello_dual_dataset.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/feedback_analysis')
    args = parser.parse_args()
    episodes = args.episodes or sorted((ROOT / 'runs/bg_data').glob('*.h5'))
    if not episodes:
        parser.error('no H5 episodes found')
    config = yaml.safe_load(args.config.read_text())
    feedback = config['force_feedback']
    checkpoint = (args.config.parent / feedback['checkpoint_path']).resolve()
    predictor = SequenceTorquePredictor(SequenceCheckpointConfig(
        checkpoint_path=checkpoint, device='cpu'), name='feedback_analysis')
    report = {'checkpoint': str(checkpoint), 'config_snapshot': config,
              'baseline_contact_free': True,
              'method': 'Every valid 100 Hz endpoint; full causal windows and checkpoint filters; no asynchronous drops simulated.',
              'streaming_batch_checks_passed': True,
              'episodes': {}, 'arms': {}}
    all_residuals, all_dq, all_times, all_episode_ids = {}, {}, {}, {}
    args.output.mkdir(parents=True, exist_ok=True)
    for episode_id, path in enumerate(episodes):
        with h5py.File(path, 'r') as f:
            group = f['teleop']
            names = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in group.attrs['arm_names']]
            if not np.isclose(group.attrs['sample_rate_hz'], predictor.metadata.sample_rate_hz):
                raise ValueError('episode and checkpoint sample rates differ')
            timestamps = group['timestamp_us'][:].reshape(-1)
            report['episodes'][path.name] = {}
            for arm in names:
                inputs = {key: np.asarray(read_arm(group, dataset, arm, names), float)
                          for key, dataset in [('q', 'q_follower'), ('dq', 'dq_follower'),
                                               ('delta_q', 'delta_q'), ('tau', 'tau_follower')]}
                valid = np.logical_and.reduce([np.isfinite(value).all(axis=1) for value in inputs.values()])
                for key in ['q_follower_valid', 'dq_valid_follower', 'torque_valid_follower']:
                    valid &= read_arm(group, key, arm, names).reshape(len(timestamps), -1).all(axis=1)
                valid &= read_arm(group, 'q_cmd_send_ok', arm, names).reshape(len(timestamps), -1).all(axis=1)
                age = read_arm(group, 'q_follower_age_us', arm, names).reshape(-1)
                valid &= (age >= 0) & (age <= feedback.get('maximum_age_s', .15) * 1e6)
                q_cmd = read_arm(group, 'q_cmd', arm, names)
                np.testing.assert_allclose(inputs['delta_q'], q_cmd-inputs['q'], atol=1e-12)
                prediction, residual, indices = replay(
                    predictor, timestamps, inputs, valid, feedback.get('maximum_sample_gap_s', .03),
                    feedback.get('measured_tau_filter', 'raw'))
                settings = dict(feedback, **feedback.get('sides', {}).get(arm, {}))
                controller = ResidualCurrentFeedback(settings)
                damping = dict(config['gello_damping'], **config['gello_damping'].get('sides', {}).get(arm, {}))
                total_limit = np.broadcast_to(damping['damping_current_limit'], (7,))
                stats = statistics(residual[indices], inputs['dq'][indices], controller, total_limit)
                stats.update(duration_s=float((timestamps[-1]-timestamps[0])/1e6),
                             n_input=len(timestamps), n_invalid=int((~valid).sum()))
                report['episodes'][path.name][arm] = stats
                all_residuals.setdefault(arm, []).append(residual[indices])
                all_dq.setdefault(arm, []).append(inputs['dq'][indices])
                all_times.setdefault(arm, []).append((timestamps[indices]-timestamps[0])/1e6)
                all_episode_ids.setdefault(arm, []).append(np.full(len(indices), episode_id))
                print(path.name, arm, 'n', len(indices), 'p99', np.round(stats['abs_p99_nm'], 2),
                      'trigger%', np.round(np.array(stats['trigger_fraction_current'])*100, 1), flush=True)
    arrays = {}
    for arm in all_residuals:
        residual = np.concatenate(all_residuals[arm])
        dq = np.concatenate(all_dq[arm])
        settings = dict(feedback, **feedback.get('sides', {}).get(arm, {}))
        controller = ResidualCurrentFeedback(settings)
        damping = dict(config['gello_damping'], **config['gello_damping'].get('sides', {}).get(arm, {}))
        total_limit = np.broadcast_to(damping['damping_current_limit'], (7,))
        stats = statistics(residual, dq, controller, total_limit)
        # The envelope of per-episode P99.5 avoids hiding a shorter bad episode.
        envelope = np.max([ep[arm]['abs_p995_nm'] for ep in report['episodes'].values() if arm in ep], axis=0)
        proposal = np.ceil((envelope + .2)*10)/10
        stats['provisional_threshold_nm'] = proposal.tolist()
        stats['proposal_method'] = 'max episode P99.5(abs residual) + 0.2 Nm; round up to 0.1 Nm'
        stats['trigger_fraction_provisional'] = (np.abs(residual) > proposal).mean(axis=0).tolist()
        stats['any_joint_trigger_fraction_provisional'] = (np.abs(residual) > proposal).any(axis=1).mean().item()
        report['arms'][arm] = stats
        arrays.update({f'{arm}_residual': residual, f'{arm}_dq': dq,
                       f'{arm}_time_s': np.concatenate(all_times[arm]),
                       f'{arm}_episode_id': np.concatenate(all_episode_ids[arm])})
    (args.output / 'statistics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    np.savez_compressed(args.output / 'replay_residuals.npz', **arrays)
    keys = ['bias_nm', 'rmse_nm', 'abs_p95_nm', 'abs_p99_nm', 'abs_p995_nm', 'abs_p999_nm',
            'trigger_fraction_current', 'settled_feedback_demand_reaches_total_cap_fraction',
            'provisional_threshold_nm', 'trigger_fraction_provisional']
    with (args.output / 'joint_statistics.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['arm', 'joint', *keys])
        for arm, stats in report['arms'].items():
            for joint in range(7):
                writer.writerow([arm, joint+1, *[stats[key][joint] for key in keys]])
    print('Results:', args.output, flush=True)


if __name__ == '__main__':
    main()
