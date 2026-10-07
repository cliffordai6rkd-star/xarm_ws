"""Clocked action plans and queue-free asynchronous inference workers."""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import numpy as np

from inference.core.contracts import ActionChunk
from inference.dp_wm.action_stream import ActionStream

DP_PLAN_STEPS = 15  # Full horizon=16 after the n_obs_steps-1 action start.
WM_ACTION_STEPS = 8


@dataclass(frozen=True)
class HistorySnapshot:
    timestamps_us: np.ndarray
    values: dict[str, np.ndarray]

    @property
    def anchor_us(self):
        return int(self.timestamps_us[-1])

    def validate(self):
        stamps = np.asarray(self.timestamps_us)
        if stamps.shape != (50,) or stamps.dtype.kind not in 'iu':
            raise ValueError('WM requires 50 integer microsecond history timestamps')
        if np.any(np.abs(np.diff(stamps) - 10_000) > 4_000):
            raise ValueError('WM history violates the checkpoint 100 Hz cadence')
        for key in ('q', 'dq', 'delta_q', 'tau'):
            value = np.asarray(self.values[key])
            if value.shape != (50, 7) or not np.isfinite(value).all():
                raise ValueError(f'WM history {key} must be finite [50,7]')


@dataclass(frozen=True)
class ActionPlan:
    chunk: ActionChunk
    version: int
    plan_steps: int = DP_PLAN_STEPS

    def __post_init__(self):
        if self.chunk.semantic != 'joint' or self.chunk.values.shape != (self.plan_steps, 7):
            raise ValueError(f'Policy plan must contain {self.plan_steps} seven-joint actions')
        if self.chunk.step_s is None or not np.isclose(self.chunk.step_s, .04):
            raise ValueError('DP plan must use the trained 25 Hz action rate')

    def future(self, anchor_us, *, aligned=False):
        times = self.chunk.timestamp_us + np.arange(self.plan_steps, dtype=np.int64) * 40_000
        if aligned:
            targets = anchor_us + np.arange(1, WM_ACTION_STEPS+1, dtype=np.int64)*40_000
            if anchor_us < times[0] or targets[-1] > times[-1]:
                return None
            # Native contact WM has ordinal positions, not timestamp inputs.
            # Interpolate joint targets to its anchor-relative 25 Hz grid,
            # then the adapter applies FK. Never extrapolate the plan tail.
            values = np.stack([np.interp(targets-times[0], times-times[0], self.chunk.values[:,j])
                               for j in range(7)], axis=1)
            return values, targets
        # offset=1: skip the currently held token, even exactly on a boundary.
        start = int(np.searchsorted(times, anchor_us, side='right'))
        if anchor_us < times[0] or start + WM_ACTION_STEPS > len(times):
            return None
        return self.chunk.values[start:start+WM_ACTION_STEPS].copy(), times[start:start+WM_ACTION_STEPS].copy()

    def covers(self, anchor_us, *, aligned=False):
        age = anchor_us-self.chunk.timestamp_us
        maximum = (self.plan_steps-WM_ACTION_STEPS-(1 if aligned else 0))*40_000
        return bool(0<=age and (age<=maximum if aligned else age<maximum))


class Worker:
    """One in-flight request, zero queued requests, explicit error propagation."""
    def __init__(self, name, function):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self.function, self.future = function, None
        self.latencies_ms = deque(maxlen=100)

    @property
    def busy(self):
        return self.future is not None

    def submit(self, request):
        if self.busy:
            return False
        self.future = self.executor.submit(self._call, request)
        return True

    def _call(self, request):
        start = time.perf_counter()
        result = self.function(request)
        return result, (time.perf_counter() - start) * 1000

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        result, latency = future.result()
        self.latencies_ms.append(latency)
        return result, latency

    def close(self):
        # Join before the model is released; shutdown cannot cancel a CUDA call.
        self.executor.shutdown(wait=True, cancel_futures=True)


class AsyncDPWM:
    """Source snapshots in, forecasts out. Deliberately has no controller API."""
    def __init__(self, policy, wm, *, execute_steps=2, margin_ms=3., initial_latency_ms=17., aligned_actions=False, plan_steps=DP_PLAN_STEPS, policy_refresh_steps=1, policy_open_loop=False, policy_prefetch_margin_ms=100., wait_for_window_exhaustion=False, scheduling='periodic', inference_budget_ms=100., action_sampling='anchor_grid'):
        if scheduling not in ('periodic', 'continuous'):
            raise ValueError('WM scheduling must be periodic or continuous')
        if action_sampling not in ('anchor_grid', 'recorded_tokens'):
            raise ValueError('WM action_sampling must be anchor_grid or recorded_tokens')
        if not np.isfinite(inference_budget_ms) or inference_budget_ms <= 0:
            raise ValueError('WM inference_budget_ms must be positive and finite')
        self.scheduling = scheduling
        self.inference_budget_ms = float(inference_budget_ms)
        self.action_sampling = action_sampling
        if type(plan_steps) is not int or plan_steps <= WM_ACTION_STEPS:
            raise ValueError('plan_steps must exceed the eight WM action tokens')
        if type(policy_refresh_steps) is not int or not 1 <= policy_refresh_steps <= plan_steps:
            raise ValueError('policy_refresh_steps must be a positive integer fitting the plan')
        if type(policy_open_loop) is not bool:
            raise ValueError('policy_open_loop must be a boolean')
        if type(wait_for_window_exhaustion) is not bool or (wait_for_window_exhaustion and policy_open_loop):
            raise ValueError('DP window exhaustion requires a boolean setting and non-open-loop scheduling')
        if policy_open_loop and (not aligned_actions or policy_refresh_steps != plan_steps):
            raise ValueError('Open-loop WM streaming requires aligned actions and full-chunk consumption')
        if not np.isfinite(policy_prefetch_margin_ms) or policy_prefetch_margin_ms < 0:
            raise ValueError('policy_prefetch_margin_ms must be finite and nonnegative')
        self.policy_open_loop = policy_open_loop
        self.wait_for_window_exhaustion = wait_for_window_exhaustion
        self.policy_prefetch_margin_us = round(policy_prefetch_margin_ms * 1000)
        self.stream = ActionStream()
        self.policy_refresh_steps = policy_refresh_steps
        self.policy_period_us = policy_refresh_steps * 40_000
        self.plan_steps = plan_steps
        self.n_obs_steps = getattr(policy, 'n_obs_steps', 2)
        if type(execute_steps) is not int or not 1 <= execute_steps <= 32:
            raise ValueError('execute_steps must be an integer in [1,32]')
        if not np.isfinite([margin_ms, initial_latency_ms]).all() or min(margin_ms, initial_latency_ms) < 0:
            raise ValueError('latency and margin must be finite and nonnegative')
        self.execute_steps = execute_steps
        self.period_us = execute_steps * 10_000
        self.margin_us = round(margin_ms * 1000)
        self.initial_latency_ms = initial_latency_ms
        self.estimated_wm_latency_ms = initial_latency_ms
        self.aligned_actions = aligned_actions
        self.dp = Worker('dp-plan', lambda obs: policy.predict_observations(obs, plan_steps=self.plan_steps))
        def infer_wm(request):
            result = wm.infer(request)
            # Preserve the exact physical action window consumed by this WM
            # request. Otherwise q_pred alone cannot diagnose policy intent.
            if isinstance(result,dict) and 'actions' in request:
                result['policy_condition_q']=np.asarray(request['actions']).copy()
                result['policy_condition_timestamps_us']=np.asarray(request['action_timestamps_us']).copy()
                result['wm_history_q_anchor']=request['history'].values['q'][-1].copy()
                result['wm_history_delta_q_anchor']=request['history'].values['delta_q'][-1].copy()
            if 'action_plan_versions' in request:
                result['action_plan_versions'] = request['action_plan_versions']
            return result

        self.wm = Worker('latent-wm', infer_wm)
        self.plan = None
        self.version = 0
        self.next_dp_us = self.next_wm_deadline_us = None
        self.last_dp_anchor = self.last_wm_anchor = -1
        self.generation = 0
        self.dp_generation = self.wm_generation = 0
        self.dp_observation_timestamps_us = np.empty(0,dtype=np.int64)
        self.counts = Counter()
        self.wait_reason = 'waiting for real observations'
        self.latest_prediction = None
        self._bootstrap_epoch_us = None
        self.policy_start_us = None

    def prime_hold(self, q, epoch_us):
        """Opt into real-command startup before permitting policy execution.

        The caller must hold this exact position at the control rate until
        ``policy_start_us``. At that fixed boundary it must either have a valid
        forecast or stop; delaying the boundary would change committed tokens.
        """
        if not self.policy_open_loop:
            raise ValueError('Bootstrap hold requires the open-loop policy stream')
        if self.plan is not None or self._bootstrap_epoch_us is not None:
            raise RuntimeError('Bootstrap hold must precede the first policy plan')
        self.stream.prime_hold(q, epoch_us)
        self._bootstrap_epoch_us = int(epoch_us)

    def bootstrap_hold_active(self, now_us):
        return (self._bootstrap_epoch_us is not None
                and (self.policy_start_us is None or now_us < self.policy_start_us))

    def reset(self):
        self.generation += 1
        self.dp_observation_timestamps_us = np.empty(0,dtype=np.int64)
        self.plan = self.latest_prediction = None
        self.stream = ActionStream()
        self.next_dp_us = self.next_wm_deadline_us = None
        self.last_dp_anchor = self.last_wm_anchor = -1
        self._bootstrap_epoch_us = None
        self.policy_start_us = None

    def warmup(self, observations, request, *, steps=3):
        """Warm thread-local CUDA/library state, without live plans or timing samples."""
        if self.dp.busy or self.wm.busy:
            raise RuntimeError('Warmup must precede live inference')
        if type(steps) is not int or steps < 1:
            raise ValueError('Warmup steps must be a positive integer')
        for _ in range(steps):
            dp = self.dp.executor.submit(self.dp.function, observations)
            wm = self.wm.executor.submit(self.wm.function, request)
            dp.result()
            wm.result()

    def policy_refresh_period_us(self):
        if self.wait_for_window_exhaustion:
            return self.policy_period_us
        if self.policy_open_loop:
            latency = float(np.percentile(self.dp.latencies_ms, 95)) if self.dp.latencies_ms else 0.
            # Have the next chunk ready before WM's eight-token window reaches
            # the current tail, including history skew and observation margin.
            lead = 9*40_000 + round(latency*1000) + 30_000 + self.margin_us + self.policy_prefetch_margin_us
            return max(0, self.policy_period_us - lead)
        # Preserve the full plan for WM lookahead even when using a short prefix.
        # Long prefixes must refresh before the final usable eight-action window.
        coverage = (self.plan_steps-WM_ACTION_STEPS-(1 if self.aligned_actions else 0))*40_000
        latency = float(np.percentile(self.dp.latencies_ms, 95)) if self.dp.latencies_ms else 0.
        return max(40_000, min(self.policy_period_us, coverage-round(latency*1000)-30_000-self.margin_us))

    def poll(self, now_us, observations=(), history=None, *, active=True):
        events = []
        dp_result = self.dp.poll()
        if dp_result is not None and self.dp_generation == self.generation:
            chunk, latency = dp_result
            self.version += 1
            plan = ActionPlan(chunk, self.version, self.plan_steps)
            execution_start = None
            if self.policy_open_loop:
                # Observation time stays in ActionChunk. Only the execution
                # stream receives a separate start time; later boundaries are fixed.
                earliest_start = None
                if self._bootstrap_epoch_us is not None and self.policy_start_us is None:
                    # Fifty new 100 Hz host-command rows precede the first WM
                    # request, then reserve its complete inference budget plus
                    # the permitted history skew. All slots retain epoch phase.
                    earliest_start = max(self._bootstrap_epoch_us + 500_000, now_us)
                    earliest_start += round(self.inference_budget_ms*1000) + 30_000 + self.margin_us
                execution_start = self.stream.append(plan, now_us, earliest_start_us=earliest_start)
                if execution_start is not None and self._bootstrap_epoch_us is not None and self.policy_start_us is None:
                    self.policy_start_us = execution_start
                usable = execution_start is not None
            else:
                usable = plan.covers(now_us, aligned=self.aligned_actions)
            if not usable:
                self.counts['dp_expired'] += 1
            else:
                self.plan = plan
                self.counts['dp_plans'] += 1
                if self.policy_open_loop:
                    self.next_dp_us = execution_start + self.policy_refresh_period_us()
                elif self.wait_for_window_exhaustion:
                    # Preserve the checkpoint's 25 Hz action clock: the first
                    # slot belongs to the input observation timestamp. Do not
                    # stretch eight slots by the DP computation latency.
                    self.next_dp_us = chunk.timestamp_us + self.policy_period_us
                elif self.policy_period_us > 40_000:
                    self.next_dp_us = chunk.timestamp_us + self.policy_refresh_period_us()
            if self.wait_for_window_exhaustion and not usable:
                self.next_dp_us = now_us
            events.append({'kind': 'dp', 'version': self.version, 'anchor_us': chunk.timestamp_us,
                           'latency_ms': latency, 'completed_us': now_us,
                           'plan_age_ms': (now_us-chunk.timestamp_us)/1000,
                           'usable_now': usable,
                           'policy_q':chunk.values.copy(),
                           'policy_input_observation_timestamps_us':self.dp_observation_timestamps_us.copy(),
                           'policy_timestamps_us':(chunk.timestamp_us if execution_start is None else execution_start)
                               +np.arange(len(chunk.values),dtype=np.int64)*round(chunk.step_s*1e6),
                           'policy_clock':('execution' if execution_start is not None else 'observation'),
                           **({'window_start_us': chunk.timestamp_us, 'window_end_us': self.next_dp_us}
                              if self.wait_for_window_exhaustion and usable else {}),
                           **({'execution_start_us': execution_start,
                               'policy_open_loop': True} if self.policy_open_loop else {})})
        wm_result = self.wm.poll()
        if wm_result is not None and self.wm_generation == self.generation:
            if self.wm.latencies_ms:
                self.estimated_wm_latency_ms=float(np.percentile(self.wm.latencies_ms,95))
            result, latency = wm_result
            times = np.asarray(result['future_timestamps_us'])
            valid = np.flatnonzero(times > now_us)[:self.execute_steps]
            result.update(completed_us=now_us, latency_ms=latency,
                          selected_indices=valid, deadline_missed=now_us > result['deadline_us'])
            version = (self.stream.version_at(result['anchor_us']) if self.policy_open_loop
                       else None if self.plan is None else self.plan.version)
            result['plan_superseded'] = version is not None and result.get('plan_version') != version
            self.counts['wm_predictions'] += 1
            self.counts['wm_deadline_misses'] += int(result['deadline_missed'])
            self.counts['wm_superseded_results'] += int(result['plan_superseded'])
            if self.scheduling == 'continuous' and times[-1] < now_us + 20_000:
                # A result cannot be retimed or replace a still usable forecast
                # when it no longer covers the controller's two physical steps.
                self.counts['wm_expired_results'] += 1
            else:
                self.latest_prediction = result
                events.append({'kind': 'wm', **result})
        if not active:
            return events
        if self.policy_open_loop:
            # Keep enough past tokens for every permitted history anchor, even
            # while the source rebuilds its history; buffers remain bounded.
            self.stream.prune(now_us-30_000)
        if self.next_dp_us is None:
            self.next_dp_us = now_us
        if self.wait_for_window_exhaustion and not self.dp.busy and now_us < self.next_dp_us:
            self.counts['dp_window_wait_ticks'] += 1
        if now_us >= self.next_dp_us and not (self.policy_open_loop and self.stream.has_pending(now_us)):
            if self.dp.busy:
                self.counts['dp_busy_ticks'] += 1
                self.next_dp_us += ((now_us-self.next_dp_us)//40_000 + 1)*40_000
            elif len(observations) >= self.n_obs_steps and observations[-1].timestamp_us > self.last_dp_anchor:
                self.dp_generation = self.generation
                self.dp.submit(observations)
                self.dp_observation_timestamps_us=np.array(
                    [o.timestamp_us for o in observations[-self.n_obs_steps:]],dtype=np.int64)
                self.last_dp_anchor = observations[-1].timestamp_us
                # Phase-lock requests to acquisition time, not process startup.
                # Otherwise a 25 Hz poll can consistently pick an almost-expired frame.
                if self.wait_for_window_exhaustion:
                    self.next_dp_us = self.last_dp_anchor + self.policy_period_us
                elif not self.policy_open_loop:
                    self.next_dp_us = self.last_dp_anchor + self.policy_refresh_period_us()
                self.counts['dp_requests'] += 1
        if self.scheduling == 'continuous':
            # Polling a completed future frees the worker above. Submit from
            # the newest history in this same tick; never queue stale snapshots
            # or wait for the next periodic slot after an overrun.
            if self.wm.busy:
                self.wait_reason = 'WM prediction in flight'
                return events
            deadline = now_us + round(self.inference_budget_ms * 1000)
        else:
            deadline = self._periodic_deadline(now_us)
            if deadline is None:
                return events
        if history is None:
            self.wait_reason = 'waiting for 50 valid measured state/command samples'
        elif self.plan is None:
            self.wait_reason = 'waiting for a timely policy plan'
        elif now_us-history.anchor_us > 30_000 or history.anchor_us > now_us:
            self.wait_reason = 'WM history is stale or ahead of host time'
        elif (self._bootstrap_epoch_us is not None
              and history.timestamps_us[0] < self._bootstrap_epoch_us):
            self.wait_reason = 'waiting for 50 real state/host-command rows after bootstrap hold'
        elif (self.policy_open_loop and self.stream.start_us is not None
              and history.anchor_us < self.stream.start_us):
            self.wait_reason = 'waiting for a state anchor on the open-loop execution clock'
        elif history.anchor_us <= self.last_wm_anchor:
            self.wait_reason = 'waiting for a new state anchor'
        else:
            aligned = self.aligned_actions and self.action_sampling == 'anchor_grid'
            window = (self.stream.future(history.anchor_us, sampling=self.action_sampling) if self.policy_open_loop
                      else self.plan.future(history.anchor_us, aligned=aligned))
            covered = (self.stream.covers(now_us, sampling=self.action_sampling) if self.policy_open_loop
                       else self.plan.covers(now_us, aligned=aligned))
            # Check coverage now too: a stale anchor must not revive an expired plan.
            if window is None or not covered:
                self.wait_reason = ('open-loop stream lacks eight future actions; waiting for next policy chunk'
                                    if self.policy_open_loop else 'DP plan lacks eight future actions')
                self.counts['action_underflows'] += 1
            else:
                history.validate()
                actions, action_times = window
                self.wm_generation = self.generation
                self.wm.submit({'history': history, 'actions': actions, 'action_timestamps_us': action_times,
                                'plan_version': (self.stream.version_at(history.anchor_us)
                                                 if self.policy_open_loop else self.plan.version),
                                'deadline_us': deadline,
                                **({'action_plan_versions': self.stream.window_versions(history.anchor_us,
                                                            sampling=self.action_sampling)}
                                   if self.policy_open_loop else {})})
                if self.scheduling == 'periodic':
                    self.next_wm_deadline_us += self.period_us
                self.last_wm_anchor = history.anchor_us
                self.wait_reason = 'WM prediction in flight'
                self.counts['wm_requests'] += 1
        return events

    def _periodic_deadline(self, now_us):
        if self.next_wm_deadline_us is None:
            self.next_wm_deadline_us = now_us + self.period_us
        estimate = self.estimated_wm_latency_ms
        # A single worker cannot pipeline more requests than its throughput.
        lead_us = min(self.period_us, round(estimate*1000) + self.margin_us)
        if now_us < self.next_wm_deadline_us - lead_us:
            return None
        if now_us > self.next_wm_deadline_us:
            skipped = (now_us-self.next_wm_deadline_us)//self.period_us + 1
            self.counts['wm_unserved_deadlines'] += skipped
            self.next_wm_deadline_us += skipped*self.period_us
            if now_us < self.next_wm_deadline_us-lead_us:
                return None
        deadline = self.next_wm_deadline_us
        if self.wm.busy:
            self.counts['wm_busy_ticks'] += 1
            self.next_wm_deadline_us += self.period_us
            return None
        return deadline

    def report(self):
        result = dict(self.counts, target_dp_hz=1e6/self.policy_period_us, policy_action_hz=25.,
                      policy_refresh_ms=self.policy_refresh_period_us()/1000, target_wm_hz=1e6/self.period_us,
                      execute_steps=self.execute_steps, wait_reason=self.wait_reason,
                      wm_scheduling=self.scheduling, wm_action_sampling=self.action_sampling,
                      wm_inference_budget_ms=self.inference_budget_ms,
                      wm_reference_age_budget_ms=(2*max(self.inference_budget_ms, self.estimated_wm_latency_ms)
                          if self.scheduling == 'continuous' else self.period_us/1000 + self.estimated_wm_latency_ms)
                          + 30. + self.margin_us/1000)
        if self.scheduling == 'continuous':
            result['target_wm_hz'] = None  # Completion-driven, not a fixed-rate request clock.
        if self.wait_for_window_exhaustion:
            result.update(dp_request_mode='after_window_exhaustion', dp_window_steps=self.policy_refresh_steps,
                          dp_next_request_us=self.next_dp_us)
        for name, worker in [('dp', self.dp), ('wm', self.wm)]:
            if worker.latencies_ms:
                result[name+'_p95_ms'] = float(np.percentile(worker.latencies_ms, 95))
        if self.wm.latencies_ms:
            budget = result['wm_p95_ms']*1000 + self.margin_us
            result['suggested_execute_steps'] = max(1, int(np.ceil(budget/10_000)))
        if self.policy_open_loop:
            result.update(policy_open_loop=True, policy_chunk_steps=self.plan_steps,
                          policy_chunk_ms=self.policy_period_us/1000,
                          policy_prefetch_lead_ms=(self.policy_period_us-self.policy_refresh_period_us())/1000,
                          buffered_policy_chunks=len(self.stream.plans))
        if self._bootstrap_epoch_us is not None:
            result.update(bootstrap_hold_epoch_us=self._bootstrap_epoch_us,
                          policy_start_us=self.policy_start_us)
        return result

    def close(self):
        self.dp.close()
        self.wm.close()
