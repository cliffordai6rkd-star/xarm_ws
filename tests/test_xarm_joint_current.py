from pathlib import Path

import numpy as np

from nero_collection.arms.base import ArmState
from xarm_stack.test_joint_current import JointCurrentTester


class _Raw:
    def __init__(self):
        self.selectors = []

    def set_report_tau_or_i(self, selector):
        self.selectors.append(selector)
        return 0


class _Adapter:
    def __init__(self, name):
        self.name = name
        self.dof = 2
        self._arm = _Raw()
        self.config = type("Config", (), {"config_kwargs": {"feedback_signal": "current", "execution_enabled": False}})()

    def connect(self):
        pass

    def configure_feedback_report(self):
        self._arm.set_report_tau_or_i(1)

    def read_state(self):
        return ArmState(np.zeros(2), np.zeros(2), np.zeros(2), np.eye(4),
                        np.full(2, np.nan), np.ones(2), 1,
                        current_valid=True, torque_valid=False)

    def _call(self, name, *args, **kwargs):
        return 0

    @staticmethod
    def _check_result(*args, **kwargs):
        pass

    def disconnect(self):
        pass


def test_current_tester_keeps_effort_as_current_and_saves_raw_data(tmp_path: Path):
    tester = JointCurrentTester({}, ("left",), arm_factory=lambda name: _Adapter(name))
    tester.connect()
    tester.sample_once()
    assert tester.samples[0].current_valid
    assert not tester.samples[0].torque_valid
    output = tester.save(tmp_path / "current.h5")
    assert output.is_file()
    tester.close()
