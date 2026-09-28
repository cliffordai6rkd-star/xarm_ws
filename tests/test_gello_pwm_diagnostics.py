"""A missing servo must not hide the remaining PWM diagnostics."""

from contextlib import redirect_stdout
from io import StringIO
import threading
from types import SimpleNamespace
import unittest

from gello_teleop.diagnose_gello_pwm import inspect_motors
from gello_teleop.gello_hardware import GelloReader


VALUES = {
    0: 1200, 11: 3, 36: 885, 48: 4095, 52: 0, 64: 1, 70: 0,
    100: 600, 124: 0xFFEC, 116: 0xFFFFFFF0, 132: 0xFFFFFF00,
}


class FakePacket:
    def __init__(self):
        self.calls = []

    def _read(self, port, motor_id, address):
        self.calls.append((motor_id, address))
        if motor_id == 2:
            return 0, -3001, 0
        return VALUES[address], 0, 0

    def read1ByteTxRx(self, port, motor_id, address):
        return self._read(port, motor_id, address)

    def read2ByteTxRx(self, port, motor_id, address):
        return self._read(port, motor_id, address)

    def read4ByteTxRx(self, port, motor_id, address):
        return self._read(port, motor_id, address)


class PwmDiagnosticsTest(unittest.TestCase):
    def test_standalone_inspection_continues_past_timeout(self):
        packet = FakePacket()
        leader = {'port': '/dev/fake', 'joint_ids': [1, 2, 3],
                  'hold_pwm_by_joint': [300, 600, 300], 'gripper_id': 8,
                  'torque_joint_ids': [9]}
        output = StringIO()
        with redirect_stdout(output):
            failed = inspect_motors('left', leader, packet, object())
        self.assertEqual(failed, [2])
        self.assertIn('ID 2: Dynamixel ID 2 read address 0 failed', output.getvalue())
        self.assertIn('COMM_RX_TIMEOUT', output.getvalue())
        self.assertIn('ID 3: model=1200', output.getvalue())
        self.assertIn('ID 8: model=1200', output.getvalue())
        self.assertIn('ID 9: model=1200', output.getvalue())
        self.assertEqual(packet.calls.count((2, 0)), 2)

    def test_live_hold_snapshot_is_read_only_and_sign_correct(self):
        reader = GelloReader.__new__(GelloReader)
        reader.config = SimpleNamespace(joint_ids=[1, 2, 3],
                                        hold_pwm_by_joint=[300, 600, 300])
        reader.packet = FakePacket()
        reader.port = object()
        reader.lock = threading.Lock()
        reader.closed = False
        status = reader.read_hold_status()
        self.assertEqual([item['id'] for item in status], [1, 2, 3])
        self.assertIn('read 11 failed (code=-3001)', status[1]['error'])
        self.assertEqual(status[0]['present_pwm'], -20)
        self.assertEqual(status[0]['position_error_ticks'], 240)
        self.assertEqual(status[2]['goal_pwm'], 600)


if __name__ == '__main__':
    unittest.main()
