"""Offline checks for slow automatic return, cancellation and blocked movement."""
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from gello_teleop.gello_hardware import GelloReader


class AutoReturnTest(unittest.TestCase):
    def reader(self, start=1.0, follows=True):
        reader = GelloReader.__new__(GelloReader)
        reader.config = SimpleNamespace(joint_ids=[2], leader_reset_speed_deg=5.0,
                                       leader_reset_max_travel_deg=90.0,
                                       leader_reset_timeout=30.0, alignment_tolerance=0.02)
        reader.lock = threading.Lock()
        reader._checked_position_goal = lambda motor_id, ticks: ticks
        reader._apply_hold_pwm = Mock()
        reader.position = start
        reader.writes = []
        def write(motor_id, address, value, size=1):
            reader.writes.append((motor_id, address, value))
            if address == 116 and follows:
                reader.position = value * np.pi / 2048
        reader._write = write
        reader.read = lambda reset_feedback=False: ((np.array([reader.position]), np.array([342]), np.array([120]))
                                                    if reset_feedback else np.array([reader.position]))
        stop = Mock(is_set=Mock(return_value=False))
        clock = iter(np.arange(0, 200, 0.05))
        return reader, stop, clock

    def test_ramp_has_small_steps_and_leaves_holding_enabled(self):
        reader, stop, clock = self.reader()
        with patch('gello_teleop.gello_hardware.time.monotonic', side_effect=lambda: next(clock)):
            reader.return_to_reference([1.1], stop)
        goals = [value for _, address, value in reader.writes if address == 116]
        self.assertGreater(len(goals), 10)
        self.assertLessEqual(max(np.diff(goals)), 4)  # <= 0.352 degrees per sample
        self.assertAlmostEqual(reader.position, 1.1, delta=np.pi/2048)
        self.assertEqual([v for _, addr, v in reader.writes if addr == 64], [0, 1])

    def test_large_or_boundary_crossing_path_does_not_enable_torque(self):
        reader, stop, _ = self.reader(start=6.27)
        with self.assertRaisesRegex(ValueError, '行程过大'):
            reader.return_to_reference([0.02], stop)
        self.assertEqual([v for _, addr, v in reader.writes if addr == 64], [0])
        self.assertFalse(any(addr == 116 for _, addr, _ in reader.writes))

    def test_blocked_joint_aborts_ramp(self):
        reader, stop, clock = self.reader(follows=False)
        reader._read_register = lambda motor_id, address, size: {
            11: 3, 64: 1, 70: 0, 100: 600, 124: 65536-600, 36: 885, 144: 50, 146: 32,
        }[address]
        with patch('gello_teleop.gello_hardware.time.monotonic', side_effect=lambda: next(clock)):
            with self.assertRaisesRegex(RuntimeError, '回位跟踪偏差过大.*ID 2') as caught:
                reader.return_to_reference([1.5], stop)
        self.assertIn('present_pwm=-600', str(caught.exception))
        self.assertIn('torque=1', str(caught.exception))
        self.assertIn('当前指令=', str(caught.exception))

    def test_failed_diagnostic_read_preserves_original_motion_error(self):
        reader, stop, clock = self.reader(follows=False)
        reader._read_register = Mock(side_effect=RuntimeError('disconnected'))
        with patch('gello_teleop.gello_hardware.time.monotonic', side_effect=lambda: next(clock)):
            with self.assertRaisesRegex(RuntimeError, '回位跟踪偏差过大.*status_read_failed=disconnected'):
                reader.return_to_reference([1.5], stop)
        reader._read_register.assert_called_once()

    def test_cancelled_return_writes_nothing(self):
        reader, stop, _ = self.reader()
        stop.is_set.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            reader.return_to_reference([1.1], stop)
        self.assertEqual(reader.writes, [])

    def test_progress_reports_all_quantities_without_changing_goal(self):
        reader, stop, clock = self.reader()
        progress = Mock()
        with patch('gello_teleop.gello_hardware.time.monotonic', side_effect=lambda: next(clock)):
            reader.return_to_reference([1.1], stop, progress=progress)
        self.assertTrue(progress.called)
        sample = progress.call_args.args[0]
        self.assertEqual(sample['pwm'], [342])
        self.assertEqual(sample['current'], [120])
        self.assertEqual(len(sample['tracking']), 1)
        self.assertEqual(len(sample['remaining']), 1)


if __name__ == '__main__':
    unittest.main()
