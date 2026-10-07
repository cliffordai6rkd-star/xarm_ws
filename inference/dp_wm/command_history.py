"""Causal history of successfully completed host joint-position sends."""
from collections import deque
from dataclasses import dataclass
import threading

import numpy as np

from nero_collection.time_utils import now_us


@dataclass(frozen=True)
class SentCommand:
    timestamp_us: int
    q: np.ndarray
    kind: str = 'servo'


class HostCommandHistory:
    """Select the last successful send before a state read begins.

    A held command remains valid until another successful send. Its age is
    deliberately independent of the freshness of the measured state.
    """

    def __init__(self, capacity=4096):
        if type(capacity) is not int or capacity < 2:
            raise ValueError('Host command history capacity must be at least two')
        self.lock = threading.RLock()
        self.commands = deque(maxlen=capacity)

    def record(self, q, timestamp_us, *, kind='servo'):
        target = np.array(q, dtype=float, copy=True)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError('Host q_cmd must be finite seven-joint radians')
        stamp = int(timestamp_us)
        if stamp <= 0:
            raise ValueError('Host send timestamp must be positive')
        target.setflags(write=False)
        command = SentCommand(stamp, target, kind)
        with self.lock:
            if self.commands and stamp < self.commands[-1].timestamp_us:
                raise ValueError('Host send timestamps must be nondecreasing')
            self.commands.append(command)
        return command

    def at_or_before(self, acquired_timestamp_us):
        with self.lock:
            command = next((command for command in reversed(self.commands)
                            if command.timestamp_us <= acquired_timestamp_us), None)
        if command is None:
            raise ValueError('No successful host q_cmd before state acquisition; waiting for authorized hold')
        return command.q.copy(), command.timestamp_us

    @property
    def latest(self):
        with self.lock:
            return self.commands[-1] if self.commands else None


class RecordingArm:
    """Record servo/reset sends and serialize them with feedback acquisition.

    The adapter raises when the SDK rejects a send. Recording only after its
    return excludes failed or in-flight commands. Sharing the lock with reads
    reproduces collection's command-then-read ordering across worker threads.
    """

    def __init__(self, arm, command_history):
        self._arm = arm
        self.command_history = command_history

    def __getattr__(self, name):
        return getattr(self._arm, name)

    def _send(self, method, q, *, kind, **kwargs):
        target = np.array(q, dtype=float, copy=True)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError('Host q_cmd must be finite seven-joint radians')
        with self.command_history.lock:
            result = method(target, **kwargs)
            self.command_history.record(target, now_us(), kind=kind)
        return result

    def command_joint_positions(self, q):
        return self._send(self._arm.command_joint_positions, q, kind='servo')

    def move_to_reset(self, q, *, speed, acceleration):
        return self._send(self._arm.move_to_reset, q, kind='reset',
                          speed=speed, acceleration=acceleration)

    def read_joint_state(self):
        with self.command_history.lock:
            return self._arm.read_joint_state()

    def read_state(self):
        with self.command_history.lock:
            return self._arm.read_state()
