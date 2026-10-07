"""Optional planned reset motion, separate from DP/WM shadow outputs."""
import time
import xml.etree.ElementTree as ET

import numpy as np

from nero_collection.time_utils import now_us


class ResetMotion:
    def __init__(self, arm, target, settings, urdf):
        self.arm = arm
        self.target = np.asarray(target, dtype=float)
        if self.target.shape != (7,) or not np.isfinite(self.target).all():
            raise ValueError('Reset target must contain seven finite joint angles')
        joints = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint')}
        limits = np.array([[float(joints[f'joint{i}'].find('limit').get(k))
                            for k in ('lower', 'upper')] for i in range(1, 8)])
        if not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError('Invalid reset joint limits')
        if np.any(self.target < limits[:, 0]) or np.any(self.target > limits[:, 1]):
            raise ValueError('Reset target is outside URDF joint limits')
        defaults = dict(reset_speed_rad_s=.2, reset_acc_rad_s2=.5,
                        reset_timeout_s=60., reset_tolerance_rad=.03)
        self.settings = {k: float(settings.get(k, v)) for k, v in defaults.items()}
        if any(not np.isfinite(v) or v <= 0 for v in self.settings.values()):
            raise ValueError('Reset settings must be positive and finite')
        self.samples = settings.get('reset_samples', 5)
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError('reset_samples must be a positive integer')
        self.running = self.touched = False

    def begin(self):
        self.touched = True  # Stop even if enabling or sending the move fails.
        self.arm.enable()
        self.arm.move_to_reset(self.target, speed=self.settings['reset_speed_rad_s'],
                               acceleration=self.settings['reset_acc_rad_s2'])
        self.last_stamp = now_us()
        self.deadline = time.monotonic() + self.settings['reset_timeout_s']
        self.settled = 0
        self.next_status_check = 0.
        self.status = {}
        self.running = True

    def poll(self, state):
        if not self.running:
            return False
        if time.monotonic() >= self.deadline:
            raise TimeoutError('xArm reset timed out')
        if time.monotonic() >= self.next_status_check or self.settled >= self.samples - 1:
            self.status = self.arm.control_status()
            self.next_status_check = time.monotonic() + .1
            if (self.status.get('error_code', 0) or self.status.get('mode') not in (None, 0)
                    or (self.status.get('state') is not None and self.status['state'] >= 3)):
                raise RuntimeError(f'xArm reset stopped: {self.status}')
        if state is None or state.timestamp_us <= self.last_stamp:
            return False
        self.last_stamp = state.timestamp_us
        good = (0 <= now_us() - state.timestamp_us <= 150_000
                and np.isfinite(state.q).all()
                and np.max(np.abs(state.q - self.target)) <= self.settings['reset_tolerance_rad']
                and self.status.get('state') != 1)
        self.settled = self.settled + 1 if good else 0
        if self.settled < self.samples:
            return False
        self.stop()
        return True

    def stop(self):
        if self.touched:
            self.arm.stop_motion()
            self.touched = False
        self.running = False
