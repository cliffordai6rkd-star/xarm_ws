import tempfile
from pathlib import Path
import unittest

from gello_teleop.tune_gello_j4_pwm import load_leader, save_config, update_j4_pwm


class GelloJ4PwmConfigTest(unittest.TestCase):
    def test_updates_only_j4_for_dual_config(self):
        data = {
            'left': {'TeleoperatorConfig': {
                'port': '/dev/left', 'joint_ids': [1, 2, 3, 4, 5, 6, 7],
                'hold_pwm_by_joint': [300, 600, 300, 400, 300, 300, 300],
            }},
            'right': {'TeleoperatorConfig': {
                'port': '/dev/right', 'joint_ids': [11, 12, 13, 14, 15, 16, 17],
                'hold_pwm_by_joint': [300, 600, 300, 500, 300, 300, 300],
            }},
        }
        motor_id, index = update_j4_pwm(data, 'left', 725)
        self.assertEqual((motor_id, index), (4, 3))
        self.assertEqual(data['left']['TeleoperatorConfig']['hold_pwm_by_joint'],
                         [300, 600, 300, 725, 300, 300, 300])
        self.assertEqual(data['right']['TeleoperatorConfig']['hold_pwm_by_joint'][3], 500)

    def test_creates_per_joint_list_when_only_global_pwm_exists(self):
        data = {'TeleoperatorConfig': {
            'port': '/dev/gello', 'joint_ids': [1, 2, 3, 4, 5], 'hold_pwm': 100,
        }}
        update_j4_pwm(data, None, 450)
        self.assertEqual(data['TeleoperatorConfig']['hold_pwm_by_joint'],
                         [100, 100, 100, 450, 100])

    def test_save_refuses_existing_output(self):
        data = {'TeleoperatorConfig': {'port': '/dev/gello', 'joint_ids': [1, 2, 3, 4]}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.yaml'
            output = Path(directory) / 'output.yaml'
            source.write_text('TeleoperatorConfig: {}\n')
            save_config(source, output, data)
            with self.assertRaises(FileExistsError):
                save_config(source, output, data)

    def test_dual_yaml_requires_side(self):
        data = {'left': {'TeleoperatorConfig': {}}, 'right': {'TeleoperatorConfig': {}}}
        with self.assertRaisesRegex(ValueError, 'requires --side'):
            load_leader(data, None)


if __name__ == '__main__':
    unittest.main()
