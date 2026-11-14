"""Compare frozen experimental torque predictions with xArm URDF dynamics.

The predictor is separate from the URDF under test. It emits total joint
effort, so MIT's feedback gains are zero and the prediction is replayed as
feedforward. No hardware connection is made and no model is trained here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from xarm_stack.learned_torque import DEFAULT_CHECKPOINT, LearnedTorqueModel, read_h5
from xarm_stack.validate_urdf import (DEFAULT_URDF, ROOT, DynamicsComparison,
                                     metrics, mit_torque, play_viewer, render_snapshot, save_plot)


def native_predictions(model, episode):
    """Predict on each segment's native 50 Hz grid; never cross invalid gaps."""
    t, valid = episode['t'], episode['valid']
    segments, start = [], None
    for i in range(len(t)+1):
        boundary = i == len(t) or not valid[i] or (i > 0 and
            (t[i] <= t[i-1] or t[i]-t[i-1] > .03+1e-6))
        if start is not None and boundary:
            if i-start >= 32 and t[i-1]-t[start] >= 2.:
                segments.append((start, i))
            start = None
        if i < len(t) and valid[i] and start is None:
            start = i
    if not segments:
        raise ValueError('No valid segments with sufficient model context')
    results = []
    for first, last in segments:
        times = t[first:last]
        count = int(np.floor((times[-1]-times[0])*50+1e-5))+1
        targets = times[0]+np.arange(count)*.02
        targets = targets[targets <= times[-1]]
        result = model.predict(times, *[episode[k][first:last]
                               for k in ('q', 'dq', 'q_cmd', 'tau')],
                               output_times=targets, include_states=True)
        results.append(result)
    return {key: np.concatenate([r[key] for r in results], axis=0) for key in results[0]}


def compare_predictions(comparison, predictions):
    """Compare experimental total torque against the candidate URDF."""
    count = len(predictions['timestamp_s'])
    valid = np.asarray(predictions['valid_context'], dtype=bool).copy()
    for key in ('q_filtered', 'dq_filtered', 'ddq_filtered', 'tau_free', 'tau_measured'):
        array = np.asarray(predictions[key])
        if array.shape != (count, 7):
            raise ValueError(f'{key} must have shape (N, 7)')
        valid &= np.isfinite(array).all(axis=1)
    if not valid.any():
        raise ValueError('No valid experimental predictions')
    torque = np.full((count, 7), np.nan)
    for i in np.flatnonzero(valid):
        q, dq = predictions['q_filtered'][i], predictions['dq_filtered'][i]
        torque[i] = mit_torque(q, dq, q, dq, predictions['tau_free'][i], kp=0., kd=0.)
    processed = dict(t=predictions['timestamp_s'], valid=valid,
        q=predictions['q_filtered'], dq=predictions['dq_filtered'],
        ddq=predictions['ddq_filtered'], tau=torque)
    arrays, engines = comparison.evaluate(processed)
    model_metrics = dict(
        samples=int(valid.sum()),
        predicted_torque_vs_reported=metrics(predictions['tau_free']-predictions['tau_measured'], valid),
        predicted_torque_vs_urdf=metrics(predictions['tau_free']-arrays['tau_model'], valid),
        acceleration_error=metrics(arrays['ddq_pin']-processed['ddq'], valid),
        mujoco_acceleration_error=metrics(arrays['ddq_mujoco']-processed['ddq'], valid),
        engine_acceleration_difference=metrics(arrays['ddq_pin']-arrays['ddq_mujoco'], valid),
        measured_acceleration_rms=np.sqrt(np.mean(processed['ddq'][valid]**2, axis=0)).tolist(),
        engine_terms=engines)
    # Also replay filtered reported effort on exactly the same states/time
    # axis, to expose the predictor's contribution to acceleration error.
    recorded = dict(processed, tau=predictions['tau_measured'])
    baseline, _ = comparison.evaluate(recorded)
    model_metrics['reported_torque_acceleration_error'] = metrics(
        baseline['ddq_pin']-processed['ddq'], valid)
    model_metrics['prediction_contribution_to_acceleration'] = metrics(
        arrays['ddq_pin']-baseline['ddq_pin'], valid)
    traces = dict(**processed, **arrays, tau_predicted=predictions['tau_free'],
                  tau_reported_filtered=predictions['tau_measured'],
                  ddq_from_reported_torque=baseline['ddq_pin'])
    return model_metrics, traces


def training_role(source, arm, manifest):
    """Read this experiment's archived split if its provenance is available."""
    candidates = [Path(path).parent/'split.json' for path in
                  manifest.get('source_sha256', {}) if Path(path).parent.name == 'residual']
    for path in candidates:
        if path.is_file():
            split = json.loads(path.read_text())
            source_id = f'{Path(source).stem}_{arm}'
            if source_id in split.get('train_sources', []):
                role = 'training_replay'
            elif source_id == split.get('validation_source'):
                role = 'archived_validation'
            else:
                role = 'not_listed_in_archived_split'
            return dict(role=role, split_path=str(path), source_id=source_id,
                        train_sources=split.get('train_sources'),
                        validation_source=split.get('validation_source'))
    return dict(role='unknown', source_id=f'{Path(source).stem}_{arm}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/urdf_validation/learned_tau')
    parser.add_argument('--no-contact', action='store_true')
    parser.add_argument('--friction', choices=('none', 'urdf'), default='none')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render-time', type=float, default=100.)
    args = parser.parse_args(argv)
    if not np.isfinite(args.render_time) or args.render_time < 0:
        parser.error('render-time must be finite and nonnegative')
    # The deployed artifact's identified prior is calibrated for the right
    # arm; this tool does not silently apply it to left-arm data.
    arm = 'right'
    manifest_path = args.checkpoint.with_suffix('.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    model = LearnedTorqueModel(args.checkpoint, device=args.device)
    episode = read_h5(args.source, arm)
    print(f'Predicting frozen experimental torque for {args.source.name} ({len(episode["t"])} input rows)', flush=True)
    predictions = native_predictions(model, episode)
    comparison = DynamicsComparison(args.urdf, dict(
        enabled=args.friction == 'urdf', source='urdf' if args.friction == 'urdf' else 'configured'))
    result, traces = compare_predictions(comparison, predictions)
    traces['timestamp_us'] = episode['timestamp_us'][0]+np.rint(traces['t']*1e6).astype(np.int64)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'predictions_and_acceleration.npz', **traces)
    save_plot(traces, args.output/'acceleration_and_torque.png',
              'Experimental model total torque -> MIT feedforward -> candidate URDF',
              no_contact=args.no_contact, tau_source='experimental model torque')
    pose = None
    if args.render:
        pose = render_snapshot(comparison, traces, args.output/'pose_comparison.png',
                               args.render_time, no_contact=args.no_contact)
    provenance = training_role(args.source, arm, manifest)
    report = dict(source=str(args.source.resolve()), arm=arm,
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        candidate_urdf=str(args.urdf.resolve()),
        candidate_urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        tau_source='Frozen residual_bilstm_mixed_zero5_50 experiment output tau_free; total joint effort',
        tau_generated_by_candidate_urdf=False,
        model_provenance=manifest, evaluation_provenance=provenance,
        preprocessing='Exact experiment pipeline: 100 Hz -> fourth-order zero-phase 5 Hz Butterworth -> 50 Hz; '
                      'ddq from the filtered 50 Hz dq; exclude one-second model-context edges.',
        mit=dict(kp=0., kd=0., t_ff='predicted total joint torque', additional_gravity=False,
                 reason='The model predicts total reported effort, not a feedforward residual. '
                        'Adding PD changes the force being validated.'),
        contact_status='user_confirmed_no_contact' if args.no_contact else 'unknown',
        friction=comparison.friction.metadata(), metrics=result, pose_snapshot=pose,
        interpretation='Compares the frozen experimental torque estimator plus candidate dynamics with recorded motion. '
            'Predictor error and missing drive/friction physics affect this result. '
            'The predictor contains an identified URDF-based prior, and training replay is not independent validation.')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Evaluation role:', provenance['role'])
    print('Predicted tau vs reported tau RMSE (Nm):', np.round(result['predicted_torque_vs_reported']['rmse'], 4).tolist())
    print('Predicted tau -> URDF ddq RMSE (rad/s²):', np.round(result['acceleration_error']['rmse'], 4).tolist())
    print('Predicted tau -> MuJoCo ddq RMSE (rad/s²):', np.round(result['mujoco_acceleration_error']['rmse'], 4).tolist())
    print('Report:', args.output/'summary.json')
    if args.viewer:
        play_viewer(comparison, traces)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
