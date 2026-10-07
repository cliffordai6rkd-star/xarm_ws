"""Full pi0 chunks retain token order through WM lookahead and reset boundaries."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from inference.core.contracts import ActionChunk
from inference.dp_wm.action_stream import ActionStream
from inference.dp_wm.core import ActionPlan, AsyncDPWM, HistorySnapshot
from inference.dp_wm.mock import MockPolicy, MockWM


def plan(version, observation_us):
    tokens = np.arange((version-1)*50, version*50)
    return ActionPlan(ActionChunk(np.repeat(tokens[:, None], 7, axis=1),
                                 'joint', None, observation_us, .04), version, 50)


def history(anchor):
    return HistorySnapshot(anchor + np.arange(-49, 1, dtype=np.int64)*10_000,
                           {key: np.zeros((50, 7)) for key in ('q', 'dq', 'tau', 'delta_q')})


def test_stream_crosses_boundary_without_replacing_or_skipping_tokens():
    stream = ActionStream()
    first = plan(1, 500_000)
    assert stream.append(first, 1_000_000) == 1_000_000
    assert first.chunk.timestamp_us == 500_000  # Source time never changes.
    assert stream.append(plan(2, 2_000_000), 2_100_000) == 3_000_000
    actions, times = stream.future(2_840_000)
    np.testing.assert_array_equal(actions[:, 0], np.arange(47, 55))
    np.testing.assert_array_equal(times, 2_840_000 + np.arange(1, 9)*40_000)
    # The second chunk is buffered; current chunk/version is still the first.
    assert stream.version_at(2_840_000) == 1
    assert stream.version_at(3_000_000) == 2
    fractional, _ = stream.future(2_980_000)
    np.testing.assert_array_equal(fractional[:, 0], np.arange(50.5, 58.5))
    stream.prune(3_000_000)
    assert len(stream.plans) == 1
    boundary, _ = stream.future(3_000_000)
    np.testing.assert_array_equal(boundary[:, 0], np.arange(51, 59))


def test_stream_refuses_missing_lookahead_and_late_chunk_without_retiming():
    stream = ActionStream()
    stream.append(plan(1, 500_000), 1_000_000)
    assert stream.future(999_999) is None
    assert stream.future(2_640_000) is not None
    assert stream.future(2_640_001) is None
    assert stream.append(plan(2, 2_900_000), 3_000_001) is None
    assert len(stream.plans) == 1
    assert stream.future(3_000_001) is None


class ManualWorker:
    """Advance worker completion separately from the simulated host clock."""
    def __init__(self):
        self.busy = False
        self.requests = []
        self.result = None
        self.latencies_ms = deque(maxlen=100)

    def submit(self, request):
        assert not self.busy
        self.busy = True
        self.requests.append(request)

    def finish(self, result, latency=200.):
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


def scheduler(**options):
    policy = MockPolicy()
    policy.n_obs_steps = 1
    runtime = AsyncDPWM(policy, MockWM(), plan_steps=50, aligned_actions=True,
                        policy_refresh_steps=50, policy_open_loop=True, **options)
    runtime.close()
    runtime.dp, runtime.wm = ManualWorker(), ManualWorker()
    return runtime


def observations(stamp):
    return (SimpleNamespace(timestamp_us=stamp),)


def start(runtime):
    runtime.poll(500_000, observations(500_000))
    runtime.dp.finish(plan(1, 500_000).chunk, latency=500.)
    runtime.poll(1_000_000, history=history(1_000_000))


def test_scheduler_prefetches_and_uses_cross_chunk_wm_window():
    runtime = scheduler()
    start(runtime)
    assert runtime.next_dp_us < 2_640_000 - 500_000
    # Fresh images during the middle of a chunk do not replace committed actions.
    runtime.poll(runtime.next_dp_us-1, observations(runtime.next_dp_us-1))
    assert len(runtime.dp.requests) == 1
    trigger = runtime.next_dp_us
    runtime.poll(trigger, observations(trigger))
    assert len(runtime.dp.requests) == 2
    runtime.dp.finish(plan(2, trigger).chunk, latency=500.)
    events = runtime.poll(trigger+500_000, history=history(trigger+500_000))
    event = next(e for e in events if e['kind'] == 'dp')
    assert event['anchor_us'] == trigger
    assert event['execution_start_us'] == 3_000_000
    np.testing.assert_equal(event['policy_timestamps_us'],3_000_000+np.arange(50)*40_000)
    assert event['policy_clock']=='execution'
    assert event['usable_now']
    assert len(runtime.stream.plans) == 2
    # Complete the prior WM request before asking for a boundary window.
    request = runtime.wm.requests[-1]
    runtime.wm.finish(MockWM().infer(request), latency=20.)
    events = runtime.poll(2_840_000, observations(2_840_000), history(2_840_000))
    assert not next(e for e in events if e['kind'] == 'wm')['plan_superseded']
    assert len(runtime.dp.requests) == 2  # Only one future chunk may be buffered.
    # Ensure the clocked WM worker reaches its next scheduling slot.
    runtime.next_wm_deadline_us = 2_840_000
    runtime.poll(2_840_000, history=history(2_840_000))
    request = runtime.wm.requests[-1]
    np.testing.assert_array_equal(request['actions'][:, 0], np.arange(47, 55))
    assert request['plan_version'] == 1
    assert request['action_plan_versions'] == (1, 2)
    assert runtime.report()['policy_chunk_ms'] == 2000.


def test_missing_next_chunk_does_not_supply_fake_tail_actions():
    runtime = scheduler()
    start(runtime)
    runtime.wm.finish(MockWM().infer(runtime.wm.requests[-1]), latency=20.)
    runtime.next_wm_deadline_us = 2_650_000
    runtime.poll(2_650_000, history=history(2_650_000))
    assert len(runtime.wm.requests) == 1  # Only the earlier, fully covered request.
    assert runtime.counts['action_underflows'] == 1
    assert 'lacks eight future actions' in runtime.wait_reason


def test_reset_discards_committed_stream_and_inflight_next_chunk():
    runtime = scheduler()
    start(runtime)
    trigger = runtime.next_dp_us
    runtime.poll(trigger, observations(trigger))
    runtime.reset()
    runtime.dp.finish(plan(2, trigger).chunk)
    runtime.wm.finish(MockWM().infer(runtime.wm.requests[-1]), latency=20.)
    events = runtime.poll(trigger+200_000, active=False)
    assert not events
    assert not runtime.stream.plans
    assert runtime.plan is None
    runtime.poll(4_000_000, observations(4_000_000))
    runtime.dp.finish(plan(3, 4_000_000).chunk)
    events = runtime.poll(4_200_000)
    assert events[0]['execution_start_us'] == 4_200_000
    assert len(runtime.stream.plans) == 1


@pytest.mark.parametrize('settings', [dict(policy_open_loop='true'),
                                    dict(policy_open_loop=True, policy_refresh_steps=8),
                                    dict(policy_open_loop=True, aligned_actions=False),
                                    dict(policy_prefetch_margin_ms=-1.)])
def test_invalid_streaming_settings_are_rejected(settings):
    options = dict(plan_steps=50, policy_refresh_steps=50, aligned_actions=True)
    options.update(settings)
    with pytest.raises(ValueError):
        AsyncDPWM(MockPolicy(), MockWM(), **options)


def test_recorded_tokens_preserve_training_indices_phase_and_chunk_boundary():
    stream = ActionStream()
    stream.append(plan(1, 500_000), 1_000_000)
    stream.append(plan(2, 2_000_000), 2_100_000)
    actions, times = stream.future(2_980_000, sampling='recorded_tokens')
    np.testing.assert_array_equal(actions[:, 0], np.arange(50, 58))
    np.testing.assert_array_equal(times, 3_000_000 + np.arange(8)*40_000)
    assert stream.window_versions(2_980_000, sampling='recorded_tokens') == (1, 2)
    np.testing.assert_array_equal(stream.future(3_000_000, sampling='recorded_tokens')[0][:, 0],
                                  np.arange(51, 59))
    tail = ActionStream()
    tail.append(plan(1, 500_000), 1_000_000)
    assert tail.covers(2_679_999, sampling='recorded_tokens')
    assert not tail.covers(2_680_000, sampling='recorded_tokens')


def test_continuous_wm_overrun_relaunches_immediately_without_queued_history():
    runtime = scheduler(scheduling='continuous', inference_budget_ms=100.,
                        action_sampling='recorded_tokens', execute_steps=10)
    start(runtime)
    first = runtime.wm.requests[-1]
    for offset in range(10_000, 110_000, 10_000):
        runtime.poll(1_000_000+offset, history=history(1_000_000+offset))
        assert len(runtime.wm.requests) == 1
    runtime.wm.finish(MockWM().infer(first), latency=105.)
    events = runtime.poll(1_105_000, history=history(1_100_000))
    forecast = next(event for event in events if event['kind'] == 'wm')
    assert forecast['deadline_missed']
    assert forecast['anchor_us'] == 1_000_000
    np.testing.assert_array_equal(forecast['future_timestamps_us'],
                                  1_000_000 + np.arange(1,33)*10_000)
    assert runtime.wm.busy and len(runtime.wm.requests) == 2
    second = runtime.wm.requests[-1]
    assert second['history'].anchor_us == 1_100_000
    assert second['deadline_us'] == 1_205_000
    assert first['history'].anchor_us == 1_000_000  # In-flight snapshot was never mutated.
    np.testing.assert_array_equal(second['actions'][:, 0], np.arange(3,11))
    np.testing.assert_array_equal(second['action_timestamps_us'],
                                  1_000_000 + np.arange(3,11)*40_000)
    report = runtime.report()
    assert report['target_wm_hz'] is None
    assert report['wm_reference_age_budget_ms'] == 243.
    assert runtime.counts['wm_unserved_deadlines'] == 0


def test_continuous_wm_does_not_resubmit_stale_history_and_discards_reset_results():
    runtime = scheduler(scheduling='continuous')
    start(runtime)
    first = runtime.wm.requests[-1]
    runtime.wm.finish(MockWM().infer(first), latency=100.)
    runtime.poll(1_100_000, history=history(1_000_000))
    assert len(runtime.wm.requests) == 1 and not runtime.wm.busy
    runtime.poll(1_110_000, history=history(1_110_000))
    assert len(runtime.wm.requests) == 2
    runtime.reset()
    runtime.wm.finish(MockWM().infer(runtime.wm.requests[-1]), latency=100.)
    assert runtime.poll(1_210_000, active=False) == []
    assert runtime.latest_prediction is None


def test_continuous_wm_does_not_publish_forecasts_past_the_two_step_horizon():
    runtime = scheduler(scheduling='continuous')
    start(runtime)
    first = runtime.wm.requests[-1]
    runtime.wm.finish(MockWM().infer(first), latency=310.)
    assert runtime.poll(1_310_000, active=False) == []
    assert runtime.latest_prediction is None
    assert runtime.counts['wm_expired_results'] == 1


def test_100ms_async_wm_keeps_100hz_commands_continuous_with_latency_jitter():
    from pathlib import Path
    from inference.core.wm_control import WmControlLayer
    from inference.controllers.wm_two_step import ControlState
    from test_wm_two_step_controllers import settings

    options = settings()
    options.update(execution='active', maximum_forecast_age_ms=240., maximum_compute_ms=100.)
    options['limits'].update(max_velocity_rad_s=.5, max_acceleration_rad_s2=1.,
                             max_jerk_rad_s3=5., q_only_smoothing='jerk_tracking')
    urdf = Path(__file__).resolve().parents[1] / 'gello_teleop/models/xarm7_dynamics_g2.urdf'
    layer = WmControlLayer(options, urdf, allow_active=True)
    runtime = scheduler(scheduling='continuous', inference_budget_ms=100.,
                        action_sampling='recorded_tokens', execute_steps=10)
    start(runtime)
    delays = (100_000, 110_000, 90_000, 100_000)
    completion = 1_100_000
    next_delay = 1
    q = np.zeros(7)
    commands = []
    for now in range(1_010_000, 1_710_000, 10_000):
        if now >= completion:
            request = runtime.wm.requests[-1]
            forecast = MockWM().infer(request)
            # Changing forecasts exercise switching and command continuity.
            forecast['q'][:] = .02 if len(runtime.wm.requests) % 2 else -.02
            runtime.wm.finish(forecast, latency=delays[(next_delay-1) % len(delays)]/1000)
        count = len(runtime.wm.requests)
        events = runtime.poll(now, history=history(now-20_000))
        if len(runtime.wm.requests) > count:
            completion = now + delays[next_delay % len(delays)]
            next_delay += 1
        for event in events:
            if event['kind'] == 'wm':
                layer.update_forecast(event)
        state = ControlState(now, q.copy(), np.zeros(7))
        command = layer.poll(now, state)
        if layer.forecast is None:
            assert command is None
            continue
        assert command is not None and command['valid'], command
        command['sent'] = True
        layer.mark_sent(command)
        q = command['first_q_command'].copy()
        commands.append(command)
    np.testing.assert_array_equal(np.diff([c['state_timestamp_us'] for c in commands]), 10_000)
    assert len(commands) == 61
    assert layer.counts['sent'] == 61 and layer.counts['invalid'] == 0
    assert max(c['forecast_age_ms'] for c in commands) <= 240.
    velocities = np.stack([c['command_v_plan'][0] for c in commands])
    accelerations = np.stack([c['command_a_plan'][0] for c in commands])
    assert np.max(np.abs(velocities)) <= .5
    assert np.max(np.abs(accelerations)) <= 1.
    assert np.max(np.abs(np.diff(accelerations, axis=0))) <= .05 + 1e-10
