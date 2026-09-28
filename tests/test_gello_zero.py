"""Passive calibration never drives leaders and preserves saved zero across power cycles."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import ANY, Mock, patch

import numpy as np

from gello_teleop.calibrate_zero import stable_reference, capture_side
from gello_teleop.calibrate_joint_directions import calibration_values, save_result
from gello_teleop.uf_robot_gello_teleop import JointMapper, Session, load_configs

SOURCE = Path(__file__).resolve().parents[1] / 'tests/fixtures/gello_dual.yaml'


def passive_configs():
    configs = load_configs(SOURCE, dual=True)
    for _, robot, leader in configs:
        reference = np.array([4.0, 2.0, -0.1, 3.0, 9.3, 9.5, 4.0])
        values = calibration_values(robot.reset_q, reference, leader.joint_signs)
        leader.joint_offsets = values['joint_offsets']
        leader.leader_reference_q = values['leader_reference_q']
        leader.leader_passive = True
    return configs


class PassiveZeroTest(unittest.TestCase):
    def test_default_capture_saves_single_turn_reference_for_automatic_return(self):
        side, robot, leader = load_configs(SOURCE, dual=True)[0]
        arm = Mock()
        arm.api.axis = 7
        reader = Mock()
        raw = np.r_[np.array([4., 2., -0.1, 3., 9.3, 9.5, 4.]), 0.]
        with patch('gello_teleop.calibrate_zero.Arm', return_value=arm), \
             patch('gello_teleop.calibrate_zero.GelloReader', return_value=reader), \
             patch('gello_teleop.calibrate_zero.stable_reference', return_value=(np.asarray(robot.reset_q, dtype=float), raw)), \
             patch('builtins.input', side_effect=['', '', 'y']):
            result = capture_side(side, robot, leader)
        reference = np.asarray(result['leader_reference_q'])
        self.assertTrue(np.all((reference >= 0) & (reference < 2*np.pi)))
        self.assertFalse(result['leader_passive'])
        np.testing.assert_allclose((reference - result['joint_offsets']) * result['joint_signs'],
                                   robot.reset_q, atol=1e-12)
        reader.hold.assert_not_called()
        reader.move.assert_not_called()
        reader.return_to_reference.assert_not_called()
        arm._command.assert_not_called()
        self.assertEqual(reader._checked_position_goal.call_count, 7)

    def test_saved_auto_reference_uses_slow_return_and_skips_manual_alignment(self):
        configs = passive_configs()
        session = Session(configs)
        session.stage = 'connected'
        both_started = threading.Barrier(2)
        for _, robot, leader in configs:
            leader.leader_passive = False
            session.arms.append(Mock(joints=Mock(return_value=np.asarray(robot.reset_q, dtype=float))))
            session.readers.append(Mock(config=leader, read=Mock(
                return_value=np.r_[leader.leader_reference_q, 0.0])))
            session.readers[-1].return_to_reference.side_effect = lambda *args, **kwargs: both_started.wait(timeout=2)
            session.mappers.append(JointMapper(robot, leader))
        session.reset()
        self.assertEqual(session.stage, 'aligned')
        for reader in session.readers:
            reader.return_to_reference.assert_called_once_with(
                reader.config.leader_reference_q, stop=session.stop_event, progress=ANY)
            reader.move.assert_not_called()

    def test_power_cycle_branch_and_one_to_one_motion(self):
        _, robot, leader = passive_configs()[0]
        mapper = JointMapper(robot, leader)
        raw = np.r_[np.asarray(leader.leader_reference_q) % (2*np.pi), 0.0]
        mapper.align(raw)
        np.testing.assert_allclose(mapper.target(raw)[0], robot.reset_q, atol=1e-12)
        moved = raw.copy()
        moved[1] += 0.02
        expected = np.asarray(robot.reset_q, dtype=float).copy()
        expected[1] -= 0.02
        np.testing.assert_allclose(mapper.target(moved)[0], expected, atol=1e-12)
        np.testing.assert_allclose(leader.joint_offsets,
                                   calibration_values(robot.reset_q, leader.leader_reference_q,
                                                      leader.joint_signs)['joint_offsets'])

    def test_wrong_start_pose_is_rejected_not_rezeroed(self):
        _, robot, leader = passive_configs()[0]
        mapper = JointMapper(robot, leader)
        raw = np.r_[leader.leader_reference_q, 0.0]
        raw[3] += 0.2
        with self.assertRaisesRegex(ValueError, '不会重算零点'):
            mapper.align(raw)

    def test_start_pose_error_is_preserved_and_caught_up_with_slew_limit(self):
        _, robot, leader = passive_configs()[0]
        mapper = JointMapper(robot, leader)
        raw = np.r_[leader.leader_reference_q, 0.0]
        raw[0] += np.deg2rad(7)
        mapper.align(raw)
        target, _ = mapper.target(raw)
        self.assertAlmostEqual(target[0] - robot.reset_q[0], np.deg2rad(7))
        action = mapper.action(raw, 1 / leader.fps)
        self.assertAlmostEqual(action[0] - robot.reset_q[0], leader.max_joint_velocity / leader.fps)
        self.assertLess(action[0], target[0])

    def test_reset_and_alignment_never_move_or_hold_passive_leaders(self):
        configs = passive_configs()
        session = Session(configs)
        session.stage = 'connected'
        for _, robot, leader in configs:
            session.arms.append(Mock(joints=Mock(return_value=np.asarray(robot.reset_q, dtype=float))))
            session.readers.append(Mock(config=leader, read=Mock(
                return_value=np.r_[leader.leader_reference_q, 0.0])))
            session.mappers.append(JointMapper(robot, leader))
        session.reset()
        self.assertEqual(session.stage, 'reset')
        session.align('left')
        session.align('right')
        self.assertEqual(session.stage, 'aligned')
        for reader in session.readers:
            reader.set_torque.assert_called_once_with(False)
            reader.move.assert_not_called()
            reader.hold.assert_not_called()
            reader.prepare_alignment.assert_not_called()

    @patch('gello_teleop.calibrate_zero.time.sleep')
    def test_capture_rejects_drift(self, sleep):
        arm = Mock(health=Mock(return_value=0), joints=Mock(return_value=np.zeros(7)))
        reader = Mock(read=Mock(side_effect=[np.zeros(8), np.ones(8)*0.1]))
        with self.assertRaisesRegex(ValueError, 'GELLO 采样时发生移动'):
            stable_reference(arm, reader, samples=2)
        arm._command.assert_not_called()
        reader.move.assert_not_called()

    def test_capture_and_saved_file_use_passive_mode(self):
        configs = load_configs(SOURCE, dual=True)
        side, robot, leader = configs[0]
        arm = Mock()
        arm.api.axis = 7
        reader = Mock()
        raw = np.r_[np.linspace(1, 2, 7), 0.0]
        with patch('gello_teleop.calibrate_zero.Arm', return_value=arm), \
             patch('gello_teleop.calibrate_zero.GelloReader', return_value=reader), \
             patch('gello_teleop.calibrate_zero.stable_reference', return_value=(np.asarray(robot.reset_q, dtype=float), raw)), \
             patch('builtins.input', side_effect=['', '', 'y']):
            result = capture_side(side, robot, leader, manual_start=True)
        arm._command.assert_not_called()
        arm.reset.assert_not_called()
        reader.move.assert_not_called()
        reader.hold.assert_not_called()
        reader.set_torque.assert_called_once_with(False)
        results = {}
        for name, robot_cfg, leader_cfg in configs:
            values = calibration_values(robot_cfg.reset_q, raw[:7], leader_cfg.joint_signs)
            values['leader_passive'] = True
            results[name] = values
        self.assertTrue(result['leader_passive'])
        with tempfile.TemporaryDirectory() as folder:
            path = save_result(SOURCE, Path(folder)/'zero.yaml', results)
            loaded = load_configs(path, dual=True, require_calibrated=True)
            self.assertTrue(all(leader_cfg.leader_passive for _, _, leader_cfg in loaded))


if __name__ == '__main__':
    unittest.main()
