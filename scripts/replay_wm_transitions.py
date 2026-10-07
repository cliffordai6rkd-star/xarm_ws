#!/usr/bin/env python3
"""Compare raw and transitioned WM references using recorded states, entirely offline.

Example:
  python scripts/replay_wm_transitions.py --run-dir runs/pi0_contact_wm_shadow/RUN \
      --output runs/wm_transition_replay/RUN.json

This replays the original journal order and absolute timestamps. It cannot
predict task success: both variants see the same recorded physical feedback.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.controllers.wm_two_step import ControlState
from inference.core.wm_control import WmControlLayer, WmReferenceAligner


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def distribution(values):
    values = np.abs(np.asarray(values, dtype=float))
    if not values.size:
        return None
    return dict(zip(('p50', 'p95', 'p99', 'max'),
                    map(float, np.percentile(values, (50, 95, 99, 100)))))


def selected_command_index(row):
    """Older journals sent candidate zero; newer rates record their endpoint."""
    return int(row.get('command_index', 0))


def read_settings(run_dir, config_path):
    path = config_path or run_dir/'config.json'
    if path.suffix == '.json':
        config = json.loads(path.read_text())
    else:
        import yaml
        config = yaml.safe_load(path.read_text())
    settings = deepcopy(config['controller'])
    if settings['mode'] != 'q_only':
        raise ValueError('This replay compares q_only controller references only')
    urdf = Path(config['wm']['urdf_path']).expanduser()
    if not urdf.is_absolute():
        urdf = (path.parent/urdf).resolve()
    return settings, urdf, path.resolve()


def previous_reference(layer, targets, *, effective):
    """Sample the trajectory preceding an installation at the exact new targets."""
    if effective:
        transition = layer.transition
        if transition is None or transition.curve is None:
            return None
        # Diagnostic sampling uses the production curve, including any earlier
        # unfinished correction. It never mutates or advances that curve.
        try:
            return transition._effective(targets)[0]
        except ValueError:
            return None
    if layer.forecast is None:
        return None
    try:
        return WmReferenceAligner.interpolate(
            layer.forecast['future_timestamps_us'],
            layer.forecast['q'][layer.aligner.sample_id], targets)
    except ValueError:
        return None


def command_summary(rows):
    if not rows:
        return {'valid_count': 0}
    commands = np.asarray([row['first_q_command'] for row in rows])
    targets = np.asarray([row['target_timestamps_us'][selected_command_index(row)] for row in rows])
    steps = []
    # Do not measure a command step across a reset/rearm or missing replay slot.
    for previous, current, delta in zip(rows[:-1], rows[1:], np.diff(commands, axis=0)):
        if current['_segment'] == previous['_segment']:
            steps.append(delta)
    return {
        'valid_count': len(rows),
        'command_indices': dict(Counter(selected_command_index(row) for row in rows)),
        'command_step_rad': distribution(steps),
        # Keep existing report keys; these describe the actual selected command.
        'first_step_velocity_rad_s': distribution([r['command_v_plan'][selected_command_index(r)] for r in rows]),
        'first_step_acceleration_rad_s2': distribution([r['command_a_plan'][selected_command_index(r)] for r in rows]),
        'first_step_jerk_rad_s3': distribution(
            [r['command_jerk_plan'][selected_command_index(r)] for r in rows if 'command_jerk_plan' in r]),
        'all_candidate_velocity_rad_s': distribution([r['command_v_plan'] for r in rows]),
        'all_candidate_acceleration_rad_s2': distribution([r['command_a_plan'] for r in rows]),
        'all_candidate_jerk_rad_s3': distribution(
            [r['command_jerk_plan'] for r in rows if 'command_jerk_plan' in r]),
        'reference_to_command_difference_rad': distribution(
            [r['q_reference'][selected_command_index(r)]-r['first_q_command'] for r in rows]),
        'command_to_recorded_measured_q_difference_rad': distribution(
            [r['first_q_command']-r['q0'] for r in rows]),
        'target_interval_ms': distribution(np.diff(targets)/1000),
    }


def replay(run_dir, config_path=None, duration_ms=20.):
    saved_settings, urdf, settings_source = read_settings(run_dir, config_path)
    settings = deepcopy(saved_settings)
    settings['execution'] = 'offline'
    # OS contention in offline analysis must not create different numerical
    # guard verdicts. Every state-age/forecast-age/coverage/limit guard remains.
    settings['maximum_compute_ms'] = max(float(settings['maximum_compute_ms']), 60_000.)
    raw_settings = deepcopy(settings)
    raw_settings['forecast_transition'] = {'enabled': False}
    transition_settings = deepcopy(settings)
    transition_settings['forecast_transition'] = {'enabled': True, 'duration_ms': duration_ms}
    layers = {
        'raw': WmControlLayer(raw_settings, urdf),
        'transitioned': WmControlLayer(transition_settings, urdf),
    }
    rows = {key: [] for key in layers}
    rejected = {key: Counter() for key in layers}
    skipped = Counter()
    pending = []
    switches = []
    mismatches = []
    baseline_reference_errors, baseline_command_errors = [], []
    transitioned_reference_errors, transitioned_command_errors = [], []
    correction_rows, installed_corrections = [], []
    source_controller_count = segment = arrivals = forecast_anchor_mismatches = 0
    last_delivered = last_applied = None
    last_raw_curve = None
    with (run_dir/'events.jsonl').open() as journal:
        for line in journal:
            if not line.strip():
                continue
            source_event = json.loads(line)
            kind = source_event.get('kind')
            if kind == 'output_stop' or (kind == 'interaction' and source_event.get('key') in ('s', 'p')):
                for layer in layers.values():
                    layer.reset()
                pending.clear()
                last_delivered = last_applied = last_raw_curve = None
                segment += 1
                continue
            if kind == 'wm':
                pending.append(source_event)
                continue
            if kind != 'controller':
                continue
            if source_event.get('actual_mode', source_event.get('mode')) == 'startup_hold':
                skipped['startup_hold'] += 1
                if source_event.get('sent') and source_event.get('file'):
                    hold = load_npz(run_dir/source_event['file'])
                    # Replay neither startup hold nor its feedback. Equal last
                    # successfully sent histories provide the initial boundary.
                    command_index = selected_command_index(source_event)
                    for layer in layers.values():
                        layer.limits.previous_q = hold['first_q_command'].copy()
                        layer.limits.previous_v = hold['command_v_plan'][command_index].copy()
                        layer.limits.previous_acceleration = hold['command_a_plan'][command_index].copy()
                continue
            if source_event.get('actual_mode') != 'q_only' or not source_event.get('sent'):
                skipped['other_or_unsent_controller'] += 1
                continue
            source_controller_count += 1
            now_us = int(source_event['computed_at_us'])
            state_us = int(source_event['state_timestamp_us'])
            targets = state_us+np.asarray((10_000, 20_000), dtype=np.int64)
            old_effective = previous_reference(layers['transitioned'], targets, effective=True)
            old_raw = None
            if last_raw_curve is not None:
                try:
                    old_raw = WmReferenceAligner.interpolate(
                        last_raw_curve['future_timestamps_us'],
                        last_raw_curve['q'][layers['raw'].aligner.sample_id], targets)
                except ValueError:
                    pass
            while pending and int(pending[0]['completed_us']) <= now_us:
                arrival = pending.pop(0)
                forecast = load_npz(run_dir/arrival['file'])
                forecast['anchor_us'] = int(forecast['anchor_us'])
                forecast['plan_version'] = int(forecast['plan_version'])
                forecast['plan_superseded'] = bool(arrival.get('plan_superseded', False))
                for layer in layers.values():
                    layer.update_forecast(forecast)
                last_delivered = forecast['anchor_us']
                arrivals += 1
            if layers['raw'].forecast is None:
                skipped['initial_no_prediction'] += 1
                continue
            if last_delivered != int(source_event['anchor_us']):
                forecast_anchor_mismatches += 1
            record = load_npz(run_dir/source_event['file'])
            state = ControlState(state_us, record['q0'], record['v0'], record.get('tau_measured'))
            produced = {key: layer.poll(now_us, state) for key, layer in layers.items()}
            verdicts = {key: 'poll_skipped' if event is None else
                        'valid' if event.get('valid') else event.get('reason', 'invalid')
                        for key, event in produced.items()}
            if len(set(verdicts.values())) > 1:
                mismatches.append({'source_file': source_event['file'], 'verdicts': verdicts})
            for key, event in produced.items():
                if verdicts[key] != 'valid':
                    rejected[key][verdicts[key]] += 1
                else:
                    event['_segment'] = segment
                    rows[key].append(event)
            raw, smoothed = produced['raw'], produced['transitioned']
            if raw is not None and raw.get('valid'):
                baseline_reference_errors.append(raw['q_reference']-record['q_reference'])
                baseline_command_errors.append(raw['first_q_command']-record['first_q_command'])
            if smoothed is not None and smoothed.get('valid'):
                transitioned_reference_errors.append(smoothed['q_reference']-record['q_reference'])
                transitioned_command_errors.append(smoothed['first_q_command']-record['first_q_command'])
                correction_rows.append({
                    'anchor_us': int(smoothed['anchor_us']),
                    'target_us': int(targets[0]),
                    'age_ms': smoothed['forecast_transition_age_ms'],
                    'duration_ms': smoothed['forecast_transition_duration_ms'],
                    'active': smoothed['forecast_transition_active'],
                    'first_correction_rad': smoothed['q_reference_correction'][0].copy(),
                    'all_correction_rad': smoothed['q_reference_correction'].copy(),
                    'reason': smoothed['forecast_transition_reason'],
                })
                if int(smoothed['anchor_us']) != last_applied:
                    if last_applied is not None and old_raw is not None and old_effective is not None and raw is not None and raw.get('valid'):
                        raw_jump = raw['q_reference'][0]-old_raw[0]
                        effective_jump = smoothed['q_reference'][0]-old_effective[0]
                        switches.append({
                            'source_file': source_event['file'], 'target_us': int(targets[0]),
                            'new_anchor_us': int(smoothed['anchor_us']), 'old_anchor_us': last_applied,
                            'raw_jump_rad': raw_jump.tolist(), 'effective_jump_rad': effective_jump.tolist(),
                            'raw_max_joint_rad': float(np.max(np.abs(raw_jump))),
                            'effective_max_joint_rad': float(np.max(np.abs(effective_jump))),
                            'handover_reason': smoothed['forecast_transition_reason'],
                            'handover_duration_ms': smoothed['forecast_transition_duration_ms'],
                        })
                    installed_corrections.append({
                        'anchor_us': int(smoothed['anchor_us']), 'start_us': int(targets[0]),
                        'duration_ms': smoothed['forecast_transition_duration_ms'],
                        'reason': smoothed['forecast_transition_reason'],
                    })
                    last_applied = int(smoothed['anchor_us'])
                last_raw_curve = layers['raw'].forecast

    converged = []
    zero_correction_target = {}
    for row in correction_rows:
        if not row['active'] and np.max(np.abs(row['first_correction_rad'])) <= 1e-10:
            zero_correction_target.setdefault(row['anchor_us'], row['target_us'])
    for installation in installed_corrections:
        if installation['duration_ms'] <= 0:
            continue
        completed_at = zero_correction_target.get(installation['anchor_us'])
        if completed_at is not None:
            converged.append((completed_at-installation['start_us'])/1000)
    jump_raw = [s['raw_jump_rad'] for s in switches]
    jump_effective = [s['effective_jump_rad'] for s in switches]
    raw_by_target = {(r['_segment'], int(r['target_timestamps_us'][0])): r for r in rows['raw']}
    paired = [(raw_by_target[(b['_segment'], int(b['target_timestamps_us'][0]))], b)
              for b in rows['transitioned']
              if (b['_segment'], int(b['target_timestamps_us'][0])) in raw_by_target]
    return {
        'run_dir': str(run_dir.resolve()), 'settings_source': str(settings_source),
        'method': {
            'journal': 'Original events.jsonl arrival order; forecasts delivered only after completed_us <= original computed_at_us. Absolute timestamps are never reanchored.',
            'feedback': 'Both offline variants use the same recorded q0, v0 and tau_measured.',
            'command_history': 'Startup records skipped; latest successful startup_hold seeds equal command histories. Offline limits then advance on valid polls without hardware or mark_sent.',
            'selected_command': 'command_index selects the sent endpoint and its q/v/a/jerk for command summaries and startup history; legacy records default to candidate zero. Physical targets remain +10/+20 ms.',
            'seam': 'Previous raw/effective trajectory evaluated at identical first target timestamp before installing a new forecast; compare with new raw/effective reference at that timestamp.',
            'offline_override': {'execution': 'offline', 'maximum_compute_ms': settings['maximum_compute_ms'],
                                 'original_maximum_compute_ms': saved_settings['maximum_compute_ms'],
                                 'reason': 'Isolate numerical comparison from OS scheduling; this is not a production deadline change.'},
        },
        'controller_settings': saved_settings, 'transition_duration_ms': duration_ms,
        'source_sent_q_only_count': source_controller_count, 'delivered_forecasts': arrivals,
        'skipped_source_records': dict(skipped), 'forecast_anchor_mismatches': forecast_anchor_mismatches,
        'raw': command_summary(rows['raw']), 'transitioned': command_summary(rows['transitioned']),
        'rejected_or_skipped_replay_polls': {key: dict(value) for key, value in rejected.items()},
        'guard_verdict_mismatches': mismatches,
        'baseline_vs_saved': {'q_reference_error_rad': distribution(baseline_reference_errors),
                              'first_command_error_rad': distribution(baseline_command_errors)},
        'transitioned_vs_saved': {'q_reference_error_rad': distribution(transitioned_reference_errors),
                                 'first_command_error_rad': distribution(transitioned_command_errors)},
        'same_target_seams': {
            'count': len(switches), 'raw_joint_jump_rad': distribution(jump_raw),
            'effective_joint_jump_rad': distribution(jump_effective),
            'raw_max_joint_per_switch_rad': distribution([s['raw_max_joint_rad'] for s in switches]),
            'effective_max_joint_per_switch_rad': distribution([s['effective_max_joint_rad'] for s in switches]),
            'handover_reasons': dict(Counter(s['handover_reason'] for s in switches)),
            'largest_raw_switches': sorted(switches, key=lambda s: s['raw_max_joint_rad'], reverse=True)[:10],
        },
        'correction': {
            'q_correction_rad': distribution([r['all_correction_rad'] for r in correction_rows]),
            'first_target_q_correction_rad': distribution([r['first_correction_rad'] for r in correction_rows]),
            'active_age_ms': distribution([r['age_ms'] for r in correction_rows if r['active']]),
            'installed_duration_ms': distribution([r['duration_ms'] for r in installed_corrections if r['duration_ms'] > 0]),
            'observed_uninterrupted_convergences': len(converged),
            'observed_time_to_zero_correction_ms': distribution(converged),
            'nonzero_corrections_without_observed_convergence': sum(r['duration_ms'] > 0 for r in installed_corrections)-len(converged),
            'paired_command_count': len(paired),
            'paired_first_command_change_rad': distribution([b['first_q_command']-a['first_q_command'] for a, b in paired]),
        },
        'limitations': [
            'This is a fixed-feedback offline replay, not a closed-loop robot simulation or task-success test.',
            'Only original successfully sent q_only records supply feedback; unsent/invalid records lack the necessary saved states.',
            'Command continuity was already bounded in these recordings; reduced reference seam jump can introduce additional lag.',
            'Time-to-zero correction describes handover settling only; it is not motion tracking delay or a ground-truth error.',
            'WM sampling schedule and pi0 policy chunks are replayed exactly as logged; this does not benchmark a new continuous scheduler.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, type=Path, help='Saved run containing config.json, events.jsonl and NPZ files')
    parser.add_argument('--config', type=Path, help='Optional YAML/JSON settings override; defaults to the saved run config')
    parser.add_argument('--duration-ms', type=float, default=20., help='Production forecast transition duration (default: 20 ms)')
    parser.add_argument('--output', type=Path, help='Optional JSON report path; otherwise print the report')
    args = parser.parse_args()
    try:
        report = replay(args.run_dir.resolve(), args.config, args.duration_ms)
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f'{args.output}: {report["source_sent_q_only_count"]} source commands, '
              f'{report["same_target_seams"]["count"]} compared switches, '
              f'{len(report["guard_verdict_mismatches"])} differing guard verdicts')
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
