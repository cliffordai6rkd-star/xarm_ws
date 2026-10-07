"""Committed open-loop policy chunks on a separate 25 Hz execution clock."""
from dataclasses import dataclass

import numpy as np


ACTION_PERIOD_US = 40_000
LOOKAHEAD_STEPS = 8


@dataclass(frozen=True)
class ScheduledPlan:
    plan: object
    start_us: int

    @property
    def end_us(self):
        return self.start_us + self.plan.plan_steps * ACTION_PERIOD_US


class ActionStream:
    """Never replace committed tokens, skip late tokens, or extrapolate a tail."""
    def __init__(self):
        self.plans = []
        self._hold_q = None
        self._hold_epoch_us = self._hold_end_us = None

    @property
    def start_us(self):
        """Earliest retained execution token, including a startup hold."""
        if self._hold_q is not None:
            return self._hold_epoch_us
        return self.plans[0].start_us if self.plans else None

    def prime_hold(self, q, epoch_us):
        """Record a real fixed hold; its finite tail is committed with plan one.

        The caller must already have successfully sent this position. These
        tokens describe that real startup command, rather than predicted or
        interpolated policy actions. No future window exists until the first
        plan commits the hold's end and its following policy tokens.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Bootstrap hold requires seven finite joint positions')
        if isinstance(epoch_us, (bool, np.bool_)) or not isinstance(epoch_us, (int, np.integer)) or epoch_us < 0:
            raise ValueError('Bootstrap hold epoch must be a nonnegative integer timestamp')
        if self.plans or self._hold_q is not None:
            raise RuntimeError('Bootstrap hold must precede all execution tokens')
        self._hold_q = q.copy()
        self._hold_q.flags.writeable = False
        self._hold_epoch_us = int(epoch_us)

    def append(self, plan, now_us, *, earliest_start_us=None):
        if self.plans:
            start = self.plans[-1].end_us
        elif self._hold_q is not None:
            # Preserve the epoch's 25 Hz phase. Policy computation latency may
            # lengthen only this initial hold, never any later chunk boundary.
            earliest = max(now_us, self._hold_epoch_us + ACTION_PERIOD_US,
                           now_us if earliest_start_us is None else earliest_start_us)
            steps = (earliest-self._hold_epoch_us + ACTION_PERIOD_US-1)//ACTION_PERIOD_US
            start = self._hold_epoch_us + steps*ACTION_PERIOD_US
        else:
            start = now_us
        if now_us > start or plan.chunk.timestamp_us > now_us:
            return None
        self.plans.append(ScheduledPlan(plan, start))
        if self._hold_q is not None and self._hold_end_us is None:
            self._hold_end_us = start
        return start

    def prune(self, anchor_us):
        if self._hold_end_us is not None and anchor_us >= self._hold_end_us:
            self._hold_q = None
            self._hold_epoch_us = self._hold_end_us = None
        while len(self.plans) > 1 and anchor_us >= self.plans[1].start_us:
            self.plans.pop(0)

    def version_at(self, anchor_us):
        if (self._hold_q is not None and self._hold_end_us is not None
                and self._hold_epoch_us <= anchor_us < self._hold_end_us):
            return 0
        return next((p.plan.version for p in self.plans
                     if p.start_us <= anchor_us < p.end_us), None)

    def has_pending(self, now_us):
        return bool(self.plans and now_us < self.plans[-1].start_us)

    def window_versions(self, anchor_us, *, sampling='anchor_grid'):
        end = anchor_us + LOOKAHEAD_STEPS * ACTION_PERIOD_US
        if sampling == 'recorded_tokens':
            window = self.future(anchor_us, sampling=sampling)
            if window is None:
                return ()
            end = int(window[1][-1])
        versions = tuple(p.plan.version for p in self.plans
                         if p.end_us > anchor_us and p.start_us <= end)
        if (self._hold_q is not None and self._hold_end_us is not None
                and self._hold_end_us > anchor_us and self._hold_epoch_us <= end):
            return (0, *versions)
        return versions

    def covers(self, anchor_us, *, sampling='anchor_grid'):
        if sampling == 'recorded_tokens':
            return self.future(anchor_us, sampling=sampling) is not None
        return bool(self.plans and anchor_us >= self.start_us
                    and anchor_us + LOOKAHEAD_STEPS * ACTION_PERIOD_US
                    <= self.plans[-1].end_us - ACTION_PERIOD_US)

    def future(self, anchor_us, *, sampling='anchor_grid'):
        if sampling not in ('anchor_grid', 'recorded_tokens'):
            raise ValueError('Unknown action sampling')
        if not self.plans or anchor_us < self.start_us:
            return None
        if sampling == 'anchor_grid' and not self.covers(anchor_us):
            return None
        times = np.concatenate([p.start_us + np.arange(p.plan.plan_steps, dtype=np.int64)
                                * ACTION_PERIOD_US for p in self.plans])
        values = np.concatenate([p.plan.chunk.values for p in self.plans])
        if self._hold_q is not None:
            # Retain only the local prefix tokens needed by this window. The
            # arithmetic keeps the original phase even after a long startup.
            first = max(0, (anchor_us-self._hold_epoch_us)//ACTION_PERIOD_US)
            last = min(first+LOOKAHEAD_STEPS+2,
                       (self._hold_end_us-self._hold_epoch_us)//ACTION_PERIOD_US)
            hold_times = self._hold_epoch_us + np.arange(first, last, dtype=np.int64)*ACTION_PERIOD_US
            times = np.concatenate((hold_times, times))
            values = np.concatenate((np.repeat(self._hold_q[None, :], len(hold_times), axis=0), values))
        if sampling == 'recorded_tokens':
            # V3 training selects the held action index + action_start_offset=1,
            # then eight exact consecutive tokens, retaining their 25 Hz phase.
            start = int(np.searchsorted(times, anchor_us, side='right'))
            if start + LOOKAHEAD_STEPS > len(times):
                return None
            return values[start:start+LOOKAHEAD_STEPS].copy(), times[start:start+LOOKAHEAD_STEPS].copy()
        targets = anchor_us + np.arange(1, LOOKAHEAD_STEPS + 1, dtype=np.int64) * ACTION_PERIOD_US
        actions = np.stack([np.interp(targets-times[0], times-times[0], values[:, j])
                            for j in range(7)], axis=1)
        return actions, targets
