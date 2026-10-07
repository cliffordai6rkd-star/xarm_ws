"""Real hold startup on the same clock as committed PINN action tokens."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from inference.core.contracts import ActionChunk
from inference.dp_wm.action_stream import ActionStream
from inference.dp_wm.core import ActionPlan, AsyncDPWM, HistorySnapshot
from inference.dp_wm.mock import MockWM


EPOCH = 1_000_003


def policy_plan(version=1, observation_us=EPOCH+10_000):
    tokens = np.arange(50)+(version-1)*50
    return ActionPlan(ActionChunk(np.repeat(tokens[:, None], 7, axis=1),
                                 'joint', None, observation_us, .04), version, 50)


def measured_history(anchor_us):
    return HistorySnapshot(anchor_us+np.arange(-49, 1, dtype=np.int64)*10_000,
                           {k: np.zeros((50, 7)) for k in ('q', 'dq', 'delta_q', 'tau')})


class ManualWorker:
    def __init__(self):
        self.busy = False
        self.requests = []
        self.result = None
        self.latencies_ms = deque(maxlen=100)

    def submit(self, request):
        assert not self.busy
        self.busy = True
        self.requests.append(request)

    def finish(self, result, latency=100.):
        assert self.busy
        self.result = (result, latency)

    def poll(self):
        if self.result is None:
            return None
        result, self.result = self.result, None
        self.busy = False
        self.latencies_ms.append(result[1])
        return result

    def close(self):
        pass


def pipeline():
    policy = SimpleNamespace(n_obs_steps=1, predict_observations=lambda *a, **k: None)
    runtime = AsyncDPWM(policy, MockWM(), plan_steps=50, policy_refresh_steps=50,
                        policy_open_loop=True, aligned_actions=True,
                        scheduling='continuous', inference_budget_ms=100.,
                        action_sampling='recorded_tokens')
    runtime.close()
    runtime.dp, runtime.wm = ManualWorker(), ManualWorker()
    return runtime


def submit_first_policy(runtime):
    runtime.poll(EPOCH+10_000, observations=(SimpleNamespace(timestamp_us=EPOCH+10_000),))
    runtime.dp.finish(policy_plan().chunk, latency=70.)


def test_bootstrap_prefix_uses_held_index_plus_one_in_every_state_phase():
    stream = ActionStream()
    q = np.arange(7, dtype=float)-10
    expected = q.copy()
    stream.prime_hold(q, EPOCH)
    assert stream.future(EPOCH, sampling='recorded_tokens') is None
    stream.append(policy_plan(), EPOCH+80_000, earliest_start_us=EPOCH+640_000)
    q[:] = 100.  # Subsequent caller changes cannot alter held tokens.
    for state_phase in (0, 10_000, 20_000, 30_000):
        actions, timestamps = stream.future(EPOCH+state_phase, sampling='recorded_tokens')
        np.testing.assert_array_equal(actions, np.repeat(expected[None, :], 8, axis=0))
        np.testing.assert_array_equal(timestamps, EPOCH+np.arange(1, 9)*40_000)
        assert stream.version_at(EPOCH+state_phase) == 0
        assert stream.window_versions(EPOCH+state_phase, sampling='recorded_tokens') == (0,)
    assert stream.future(EPOCH-1, sampling='recorded_tokens') is None


def test_prefix_to_policy_window_preserves_committed_tokens_and_phase():
    stream = ActionStream()
    stream.prime_hold(np.full(7, -1.), EPOCH)
    first = policy_plan()
    start = stream.append(first, EPOCH+80_000, earliest_start_us=EPOCH+633_000)
    assert start == EPOCH+640_000
    assert first.chunk.timestamp_us == EPOCH+10_000
    actions, timestamps = stream.future(EPOCH+590_000, sampling='recorded_tokens')
    np.testing.assert_array_equal(actions[:, 0], [-1, 0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(timestamps, EPOCH+600_000+np.arange(8)*40_000)
    assert stream.window_versions(EPOCH+590_000, sampling='recorded_tokens') == (0, 1)
    np.testing.assert_array_equal(stream.future(start, sampling='recorded_tokens')[0][:, 0],
                                  np.arange(1, 9))
    # A future full chunk cannot replace either old prefix or policy values.
    assert stream.append(policy_plan(2, start+100_000), start+200_000) == start+2_000_000
    later = stream.future(EPOCH+590_000, sampling='recorded_tokens')
    np.testing.assert_array_equal(later[0], actions)
    np.testing.assert_array_equal(later[1], timestamps)
    assert stream.append(policy_plan(3, start+4_000_000), start+4_000_001) is None
    assert len(stream.plans) == 2
    stream.prune(start)
    assert stream.start_us == start
    assert stream.version_at(start) == 1


def test_bootstrap_plan_is_usable_before_activation_and_waits_for_real_history():
    runtime = pipeline()
    runtime.prime_hold(np.full(7, -.2), EPOCH)
    submit_first_policy(runtime)
    event = runtime.poll(EPOCH+80_000)[0]
    assert event['usable_now']
    assert event['execution_start_us'] == EPOCH+640_000
    assert runtime.policy_start_us == EPOCH+640_000
    assert runtime.bootstrap_hold_active(EPOCH+639_999)
    assert not runtime.bootstrap_hold_active(EPOCH+640_000)
    assert runtime.stream.has_pending(EPOCH+80_000)
    np.testing.assert_array_equal(event['policy_timestamps_us'],
                                  runtime.policy_start_us+np.arange(50)*40_000)
    runtime.poll(EPOCH+480_000, history=measured_history(EPOCH+480_000))
    assert not runtime.wm.busy
    assert '50 real state/host-command rows' in runtime.wait_reason
    runtime.poll(EPOCH+500_000, history=measured_history(EPOCH+500_000))
    assert runtime.wm.busy
    request = runtime.wm.requests[-1]
    assert request['plan_version'] == 0
    assert request['action_plan_versions'] == (0, 1)
    np.testing.assert_array_equal(request['actions'][:, 0], [-.2, -.2, -.2, 0, 1, 2, 3, 4])
    np.testing.assert_array_equal(request['action_timestamps_us'],
                                  EPOCH+520_000+np.arange(8)*40_000)
    runtime.wm.finish(MockWM().infer(request))
    forecast = runtime.poll(EPOCH+600_000, active=False)[0]
    assert forecast['kind'] == 'wm'
    assert forecast['anchor_us'] == EPOCH+500_000
    assert not forecast['plan_superseded']
    np.testing.assert_array_equal(forecast['future_timestamps_us'],
                                  EPOCH+500_000+np.arange(1, 33)*10_000)
    assert runtime.latest_prediction['anchor_us'] == forecast['anchor_us']
    assert runtime.report()['bootstrap_hold_epoch_us'] == EPOCH


def test_late_first_receipt_extends_only_initial_hold_on_existing_grid():
    runtime = pipeline()
    runtime.prime_hold(np.zeros(7), EPOCH)
    submit_first_policy(runtime)
    event = runtime.poll(EPOCH+800_000, history=measured_history(EPOCH+800_000))[0]
    assert event['execution_start_us'] == EPOCH+960_000
    assert runtime.policy_start_us == EPOCH+960_000
    assert event['anchor_us'] == EPOCH+10_000
    assert event['usable_now'] and runtime.wm.busy
    assert runtime.bootstrap_hold_active(EPOCH+959_999)


def test_unprimed_pipeline_retains_its_existing_first_receipt_clock():
    runtime = pipeline()
    submit_first_policy(runtime)
    event = runtime.poll(EPOCH+80_000, history=measured_history(EPOCH+80_000))[0]
    assert event['execution_start_us'] == EPOCH+80_000
    assert runtime.wm.busy
    assert runtime.policy_start_us is None
    assert not runtime.bootstrap_hold_active(EPOCH+80_000)


def test_reset_clears_hold_and_discards_both_old_worker_generations():
    runtime = pipeline()
    runtime.prime_hold(np.zeros(7), EPOCH)
    submit_first_policy(runtime)
    runtime.poll(EPOCH+500_000, history=measured_history(EPOCH+500_000))
    old_request = runtime.wm.requests[-1]
    # Make the next chunk in flight before discarding the execution clock.
    runtime.poll(runtime.next_dp_us, observations=(SimpleNamespace(timestamp_us=runtime.next_dp_us),))
    assert runtime.dp.busy
    runtime.reset()
    assert runtime.policy_start_us is None
    assert not runtime.bootstrap_hold_active(EPOCH+2_000_000)
    assert runtime.stream.start_us is None
    assert runtime.plan is None
    runtime.prime_hold(np.ones(7), EPOCH+3_000_000)
    runtime.dp.finish(policy_plan(2, EPOCH+1_000_000).chunk)
    runtime.wm.finish(MockWM().infer(old_request))
    assert runtime.poll(EPOCH+3_010_000, active=False) == []
    assert runtime.policy_start_us is None
    assert runtime.latest_prediction is None
    assert not runtime.stream.plans
    runtime.poll(EPOCH+3_020_000, observations=(SimpleNamespace(timestamp_us=EPOCH+3_020_000),))
    runtime.dp.finish(policy_plan(3, EPOCH+3_020_000).chunk)
    event = runtime.poll(EPOCH+3_100_000)[0]
    assert event['execution_start_us'] == EPOCH+3_640_000
    np.testing.assert_array_equal(runtime.stream.future(EPOCH+3_010_000,
                                                       sampling='recorded_tokens')[0], np.ones((8, 7)))


@pytest.mark.parametrize('q,epoch', [(np.zeros(6), EPOCH), (np.full(7, np.nan), EPOCH),
                                   (np.zeros(7), -1), (np.zeros(7), True)])
def test_bootstrap_hold_rejects_invalid_position_or_timestamp(q, epoch):
    runtime = pipeline()
    with pytest.raises(ValueError):
        runtime.prime_hold(q, epoch)
    assert not runtime.bootstrap_hold_active(EPOCH)


def test_bootstrap_hold_cannot_replace_a_committed_execution_stream():
    runtime = pipeline()
    runtime.prime_hold(np.zeros(7), EPOCH)
    with pytest.raises(RuntimeError):
        runtime.prime_hold(np.ones(7), EPOCH+10_000)
    submit_first_policy(runtime)
    runtime.poll(EPOCH+80_000)
    with pytest.raises(RuntimeError):
        runtime.prime_hold(np.ones(7), EPOCH+100_000)
