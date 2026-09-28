"""A requested Goal PWM above EEPROM limit must be visible, not silently clipped."""

from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from gello_teleop.gello_hardware import GelloReader


class FakePacket:
    def read2ByteTxRx(self, port, motor_id, address):
        assert motor_id == 2 and address == 36
        return 500, 0, 0


class HoldPwmTest(unittest.TestCase):
    def test_move_timeout_identifies_sagging_joint_not_only_tested_joint(self):
        reader = GelloReader.__new__(GelloReader)
        reader.config = SimpleNamespace(joint_ids=[1, 2, 6])
        reader.lock = threading.Lock()
        reader._checked_position_goal = lambda motor_id, goal: goal
        reader._write = Mock()
        reader._apply_hold_pwm = Mock()
        # J1 reaches its test target; stationary J2 sags by 0.130 radians.
        reader.read = Mock(return_value=np.array([1.02, 0.87, 1.0]))
        with patch('gello_teleop.gello_hardware.time.monotonic', side_effect=[0, 6]):
            with self.assertRaises(TimeoutError) as caught:
                reader.move([1.02, 1.0, 1.0], duration=5, tolerance=0.01)
        message = str(caught.exception)
        self.assertIn('J2/ID 2', message)
        self.assertIn('-7.45 deg', message)
        self.assertNotIn('J1/ID 1', message)
        self.assertNotIn('J3/ID 6', message)

    def test_rejects_above_eeprom_limit_before_writing(self):
        reader = GelloReader.__new__(GelloReader)
        reader.config = SimpleNamespace(joint_ids=[2], hold_pwm_by_joint=[600], hold_pwm=None)
        reader.packet = FakePacket()
        reader.port = object()
        writes = []
        reader._write = lambda *args, **kwargs: writes.append((args, kwargs))
        with self.assertRaisesRegex(ValueError, 'EEPROM PWM Limit=500'):
            reader._apply_hold_pwm()
        self.assertEqual(writes, [])

    def test_hold_wraps_multiturn_feedback_before_enabling_torque(self):
        reader = GelloReader.__new__(GelloReader)
        reader.config = SimpleNamespace(joint_ids=[2, 3], hold_pwm_by_joint=[600, 300])
        reader.port = object()
        reader.lock = threading.Lock()

        class PositionPacket:
            def read1ByteTxRx(self, port, motor_id, address):
                self_test.assertEqual(address, 11)
                return 3, 0, 0

            def read2ByteTxRx(self, port, motor_id, address):
                self_test.assertEqual(address, 36)
                return 885, 0, 0

            def read4ByteTxRx(self, port, motor_id, address):
                if address == 132:
                    return (0xFFFFFF9C if motor_id == 2 else 100), 0, 0
                return ({52: 0, 48: 4095}[address]), 0, 0

        self_test = self
        reader.packet = PositionPacket()
        writes = []
        reader._write = lambda *args, **kwargs: writes.append((args, kwargs))
        reader.hold()
        self.assertEqual(writes[:2], [((2, 116, 3996), {'size': 4}),
                                      ((3, 116, 100), {'size': 4})])
        self.assertEqual([args[:3] for args, _ in writes[-2:]], [(2, 64, 1), (3, 64, 1)])

    def test_hold_rejects_outside_position_limits_before_any_write(self):
        reader = GelloReader.__new__(GelloReader)
        reader.config = SimpleNamespace(joint_ids=[3], hold_pwm_by_joint=[300])
        reader.port = object()
        reader.lock = threading.Lock()

        class PositionPacket:
            def read1ByteTxRx(self, port, motor_id, address):
                return 3, 0, 0

            def read2ByteTxRx(self, port, motor_id, address):
                raise AssertionError('PWM must not be touched before position validation')

            def read4ByteTxRx(self, port, motor_id, address):
                return {132: 100, 52: 200, 48: 4095}[address], 0, 0

        reader.packet = PositionPacket()
        writes = []
        reader._write = lambda *args, **kwargs: writes.append((args, kwargs))
        with self.assertRaisesRegex(ValueError, 'outside configured Goal Position limits'):
            reader.hold()
        self.assertEqual(writes, [])


if __name__ == '__main__':
    unittest.main()
