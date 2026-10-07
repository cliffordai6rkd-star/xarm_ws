"""Explicit position-servo output with a latched command-gap watchdog."""
import threading
import time

import numpy as np

from nero_collection.time_utils import now_us


class ActiveOutput:
    def __init__(self, arm, *, maximum_state_age_ms, timeout_s=.03, command_hz=100,
                 minimum_state_remaining_ms=0.):
        if type(command_hz) not in (int, float) or command_hz not in (50, 100):
            raise ValueError('command_hz must be 50 or 100')
        if (type(minimum_state_remaining_ms) not in (int, float)
                or not np.isfinite(minimum_state_remaining_ms)
                or not 0 <= minimum_state_remaining_ms < maximum_state_age_ms):
            raise ValueError('minimum_state_remaining_ms must leave a positive fresh-state window')
        self.arm = arm
        self.command_hz = command_hz
        self.command_index = 0 if command_hz == 100 else 1
        self.command_period_us = round(1e6/command_hz)
        self.maximum_state_age_us = maximum_state_age_ms * 1000
        self.minimum_state_remaining_us = minimum_state_remaining_ms*1000
        self.timeout_s = timeout_s
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.enabled = False
        self.last_send = None
        self.hold_q = None
        self.hold_state_timestamp_us = None
        self.hold_next_us = None
        self.reason = 'paused'
        self.error = None
        self.thread = threading.Thread(target=self._watch, name='wm-command-watchdog', daemon=True)
        self.thread.start()

    def start(self):
        with self.lock:
            self.check_error()
            self.enabled = True  # Include partial enable failures in cleanup.
            self.last_send = None
            self.hold_q = None
            self.hold_state_timestamp_us = None
            self.hold_next_us = None
            self.arm.enable()
            self.reason = 'waiting for first valid command'

    def check_error(self):
        if self.error is not None:
            raise RuntimeError('Active output watchdog could not stop the arm') from self.error

    def stop(self, reason='paused'):
        with self.lock:
            if self.enabled:
                self.arm.stop_motion()
                self.enabled = False
            self.last_send = None
            self.hold_q = None
            self.hold_state_timestamp_us = None
            self.hold_next_us = None
            self.reason = reason

    def start_hold(self, state):
        """Bootstrap with one actual measured-position send, then hold it.

        The fixed target is retained while fresh measurements populate WM
        history. It must never be refreshed from q on subsequent hold ticks.
        """
        with self.lock:
            if not self.enabled:
                raise RuntimeError('Active output must be enabled before startup hold')
            if self.hold_q is not None:
                return self.hold(state)
            if state is None:
                return None
            target = np.array(state.q, dtype=float, copy=True)
            if target.shape != (7,) or not np.isfinite(target).all():
                raise ValueError('Startup hold requires finite measured joint positions')
            if not 0 <= now_us() - state.timestamp_us <= self.maximum_state_age_us:
                return None
            if self.maximum_state_age_us-(now_us()-state.timestamp_us) < self.minimum_state_remaining_us:
                return None
            self.hold_q = target
            self.hold_state_timestamp_us = None
            return self.hold(state)

    def hold(self, state):
        """Resend the captured target at command_hz using fresh distinct feedback."""
        with self.lock:
            if self.hold_q is None or state is None or not self.enabled:
                return None
            stamp = int(state.timestamp_us)
            if self.hold_state_timestamp_us is not None and stamp <= self.hold_state_timestamp_us:
                return None
            now = now_us()
            if self.command_hz == 50 and self.hold_next_us is not None and now < self.hold_next_us:
                return None
            age = now-stamp
            if age >= 0 and self.maximum_state_age_us-age < self.minimum_state_remaining_us:
                return None
            event = dict(kind='controller', mode='startup_hold', valid=True, sent=False,
                         anchor_us=stamp, computed_at_us=now_us(), plan_version=0,
                         state_timestamp_us=stamp, first_q_command=self.hold_q.copy(),
                         command_hz=self.command_hz, command_index=self.command_index,
                         command_target_timestamp_us=stamp+(self.command_index+1)*10_000,
                         target_timestamps_us=np.array([stamp + 10_000, stamp + 20_000], dtype=np.int64))
            if self.send(event):
                self.hold_state_timestamp_us = stamp
                self.hold_next_us = now + self.command_period_us
                self.reason = 'holding measured bootstrap position; waiting for WM'
            return event

    def end_hold(self):
        with self.lock:
            self.hold_q = None
            self.hold_state_timestamp_us = None
            self.hold_next_us = None

    def send(self, event):
        with self.lock:
            self.check_error()
            if not self.enabled:
                return False
            now = now_us()
            if self.last_send is not None and time.monotonic() - self.last_send > self.timeout_s:
                self.stop('command gap exceeded 30 ms; press s to rearm')
                return False
            if event is None:
                return False
            if not event.get('valid'):
                # While waiting for the first command no servo target has been sent.
                if self.last_send is not None:
                    self.stop(f"invalid controller result: {event.get('reason', 'unspecified')}; press s to rearm")
                return False
            index = event.get('command_index', 0)
            targets = np.asarray(event.get('target_timestamps_us', []))
            if (type(index) is not int or index not in (0, 1) or index != self.command_index or targets.ndim != 1
                    or len(targets) <= index or targets.dtype.kind not in 'iu'):
                event.update(valid=False, reason='Command malformed: invalid command_index or target times')
                if self.last_send is not None:
                    self.stop('malformed command; press s to rearm')
                return False
            target_time = int(targets[index])
            target = np.asarray(event['first_q_command'], dtype=float)
            age = now - event['state_timestamp_us']
            event.update(send_phase='before_feedback_lock', send_state_age_ms=age/1000,
                         remaining_state_budget_ms=(self.maximum_state_age_us-age)/1000,
                         selected_target_remaining_ms=(target_time-now)/1000)
            timely = (0 <= age <= self.maximum_state_age_us
                      and now < target_time)
            if target.shape != (7,) or not np.isfinite(target).all() or not timely:
                event.update(valid=False, reason='Command expired or malformed before send')
                if self.last_send is not None:
                    self.stop('invalid/expired command; press s to rearm')
                return False
            history = getattr(self.arm, 'command_history', None)
            command_lock = None if history is None else history.lock
            if command_lock is not None:
                lock_started = time.monotonic()
                deadline = min(event['state_timestamp_us'] + self.maximum_state_age_us,
                               target_time)
                timeout = max(0., (deadline - now_us()) / 1e6)
                if not command_lock.acquire(timeout=timeout):
                    event.update(feedback_lock_wait_ms=(time.monotonic()-lock_started)*1000,
                                 send_state_age_ms=(now_us()-event['state_timestamp_us'])/1000)
                    event.update(valid=False, reason='Command expired while waiting for feedback acquisition')
                    self.stop('invalid/expired command; press s to rearm')
                    return False
                event['feedback_lock_wait_ms'] = (time.monotonic()-lock_started)*1000
            try:
                # A feedback RPC can occupy the shared lock. Check again after
                # acquisition so an expired target never reaches the SDK.
                now = now_us()
                event.update(send_phase='after_feedback_lock',
                             send_state_age_ms=(now-event['state_timestamp_us'])/1000,
                             remaining_state_budget_ms=(self.maximum_state_age_us-(now-event['state_timestamp_us']))/1000,
                             selected_target_remaining_ms=(target_time-now)/1000)
                timely = (0 <= now-event['state_timestamp_us'] <= self.maximum_state_age_us
                          and now < target_time)
                gap = self.last_send is not None and time.monotonic()-self.last_send > self.timeout_s
                if not timely or gap:
                    event.update(valid=False, reason='Command expired or watchdog elapsed while waiting for feedback')
                    self.stop('invalid/expired command; press s to rearm')
                    return False
                self.arm.command_joint_positions(target)
                self.last_send = time.monotonic()
                command = None if history is None else history.latest
                event.update(sent=True, sent_at_us=now_us() if command is None else command.timestamp_us)
            finally:
                if command_lock is not None:
                    command_lock.release()
            self.reason = 'sending position references'
            return True

    def watchdog_tick(self):
        with self.lock:
            if self.enabled and self.last_send is not None and time.monotonic()-self.last_send > self.timeout_s:
                self.stop('command gap exceeded 30 ms; press s to rearm')

    def _watch(self):
        while not self.closed.wait(.005):
            try:
                self.watchdog_tick()
            except Exception as exc:
                self.error = exc
                return

    def close(self):
        self.closed.set()
        try:
            self.stop('closed')
        finally:
            self.thread.join(timeout=1.)
