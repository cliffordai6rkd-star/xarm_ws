import io
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from gello_teleop.reset_display import ResetDisplay
from gello_teleop.gello_hardware import GelloReader

SAMPLE = dict(tracking=[1.0]*7, remaining=[-2.0]*7, pwm=[342]*7, current=[-120]*7)


class ResetDisplayTest(unittest.TestCase):
    def test_redirected_output_only_has_two_final_status_rows(self):
        stream = io.StringIO()
        with ResetDisplay(['left', 'right'], stream) as display:
            for _ in range(30):
                display.update('left', SAMPLE)
                display.update('right', SAMPLE)
            display.finish('left', 'done')
            display.finish('right', 'done')
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 4)  # two explanatory lines and two results
        self.assertIn('I_mA[-120', lines[-1])
        self.assertNotIn('\x1b', stream.getvalue())
        previous = stream.getvalue()
        display.update('right', SAMPLE)
        self.assertEqual(stream.getvalue(), previous)

    def test_tty_redraws_existing_rows_and_rotates_on_narrow_terminal(self):
        stream = io.StringIO()
        stream.isatty = lambda: True
        with patch('gello_teleop.reset_display.shutil.get_terminal_size', return_value=SimpleNamespace(columns=80)):
            with ResetDisplay(['left', 'right'], stream) as display:
                display.update('left', SAMPLE)
                display.update('right', SAMPLE)
                for offset, label in enumerate(('Edeg', 'Rdeg', 'I_mA', 'PWM')):
                    line = display._line('left', 80, display.started + offset)
                    self.assertIn(label, line)
                    self.assertLess(len(line), 80)
                    self.assertEqual(line.count(' '), 8)
        self.assertIn('\x1b[2A', stream.getvalue())

    def test_feedback_reads_signed_current_pwm_and_position_in_one_transaction(self):
        class Group:
            calls = 0
            def txRxPacket(self):
                self.calls += 1
                return 0
            def isAvailable(self, *args):
                return True
            def getData(self, motor_id, address, size):
                return {124: 65536-342, 126: 65536-120, 132: 2048}[address]
        reader = GelloReader.__new__(GelloReader)
        reader.lock = threading.Lock()
        reader.closed = False
        reader.config = SimpleNamespace(port='test', read_timeout=1)
        reader.ids = [4]
        reader.reset_group = Group()
        position, pwm, current = reader.read(reset_feedback=True)
        self.assertEqual(pwm.tolist(), [-342])
        self.assertEqual(current.tolist(), [-120])
        self.assertAlmostEqual(position[0], 3.141592653589793)
        self.assertEqual(reader.reset_group.calls, 1)


if __name__ == '__main__':
    unittest.main()
