import json
from pathlib import Path

import numpy as np
import pytest

from xarm_stack.validate_learned_tau import (compare_predictions, native_predictions,
                                            training_role)
from xarm_stack.validate_urdf import DEFAULT_URDF, DynamicsComparison


def test_native_prediction_does_not_bridge_gaps_or_fabricate_context():
    t = np.r_[np.arange(301)*.01, 5+np.arange(301)*.01]
    episode = dict(t=t, valid=np.ones(len(t), bool))
    episode.update({key: np.zeros((len(t), 7)) for key in ('q', 'dq', 'q_cmd', 'tau')})

    class Predictor:
        def predict(self, times, *args, output_times, include_states):
            assert include_states and np.max(np.diff(times)) < .03
            zeros = np.zeros((len(output_times), 7))
            return dict(timestamp_s=output_times, tau_free=zeros.copy(),
                tau_measured=zeros.copy(), q_filtered=zeros.copy(), dq_filtered=zeros.copy(),
                ddq_filtered=zeros.copy(), valid_context=(output_times >= times[0]+1)
                & (output_times <= times[-1]-1))

    r = native_predictions(Predictor(), episode)
    assert not np.any((r['timestamp_s'] > 3) & (r['timestamp_s'] < 5))
    assert not r['valid_context'][0] and not r['valid_context'][-1]
    np.testing.assert_allclose(np.diff(r['timestamp_s'][r['timestamp_s'] <= 3]), .02)


def test_experimental_torque_is_consumed_and_predictor_error_is_separate():
    pytest.importorskip('pinocchio')
    pytest.importorskip('mujoco')
    c = DynamicsComparison(DEFAULT_URDF)
    q = np.full((6, 7), .2)
    dq, ddq = q*.1, np.full((6, 7), .5)
    actual = np.array([c.pin.rnea(a, b, d) for a, b, d in zip(q, dq, ddq)])
    predicted = actual+np.array([.1, .2, .1, .3, .1, .2, .05])
    predictions = dict(timestamp_s=np.arange(6)*.02, valid_context=np.ones(6, bool),
                       q_filtered=q, dq_filtered=dq, ddq_filtered=ddq,
                       tau_measured=actual, tau_free=predicted)
    report, traces = compare_predictions(c, predictions)
    np.testing.assert_array_equal(traces['tau'], predicted)
    assert report['predicted_torque_vs_reported']['rmse'][0] == pytest.approx(.1)
    assert max(report['reported_torque_acceleration_error']['rmse']) < 1e-9
    assert max(report['acceleration_error']['rmse']) > .1
    assert max(report['engine_acceleration_difference']['rmse']) < 1e-8


def test_archived_split_distinguishes_training_replay_and_validation(tmp_path):
    folder = tmp_path/'residual'
    folder.mkdir()
    checkpoint = folder/'experiment.pt'
    split = dict(train_sources=['episode_0001_right'], validation_source='episode_0002_right')
    (folder/'split.json').write_text(json.dumps(split))
    manifest = {'source_sha256': {str(checkpoint): 'unused'}}
    assert training_role(Path('episode_0001.h5'), 'right', manifest)['role'] == 'training_replay'
    assert training_role(Path('episode_0002.h5'), 'right', manifest)['role'] == 'archived_validation'
