"""Saved GELLO mapping can be reused without connecting either robot."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from gello_teleop.calibrate_joint_directions import _move_arm, calibration_values, save_result, prepare_manual_reference
from gello_teleop.uf_robot_gello_teleop import (
    JointMapper, Session, apply_saved_calibration, calibration_file_for, load_configs,
    save_current_alignment,
)


SOURCE = Path(__file__).resolve().parents[1] / 'tests/fixtures/gello_dual.yaml'


class SavedCalibrationTest(unittest.TestCase):
    def test_manual_reference_uses_current_pose_without_old_reset_motion(self):
        configs = load_configs(SOURCE, dual=True)
        session = Session(configs)
        actual = [np.asarray(robot.reset_q)+.01 for _, robot, _ in configs]
        session.arms = [Mock(health=Mock(return_value=0), joints=Mock(return_value=q)) for q in actual]
        with patch('gello_teleop.calibrate_joint_directions.time.sleep'):
            prepare_manual_reference(session)
        for (_, robot, leader), arm, q in zip(configs, session.arms, actual):
            np.testing.assert_allclose(robot.reset_q, q)
            self.assertIsNone(leader.joint_offsets)
            self.assertIsNone(leader.leader_reference_q)
            arm.prepare_reset.assert_called_once()
            arm.reset.assert_not_called()
            arm._command.assert_not_called()

    def test_manual_reference_checks_both_arms_before_enabling_either(self):
        session = Session(load_configs(SOURCE, dual=True))
        q = np.zeros(7)
        session.arms = [Mock(health=Mock(return_value=0), joints=Mock(return_value=q)),
                        Mock(health=Mock(return_value=1))]
        with patch('gello_teleop.calibrate_joint_directions.time.sleep'):
            with self.assertRaisesRegex(ValueError, '正在运动'):
                prepare_manual_reference(session)
        for arm in session.arms:
            arm.prepare_reset.assert_not_called()

    def test_start_follow_reports_drift_and_keeps_torque_until_cleanup(self):
        configs = load_configs(SOURCE, dual=True)
        session = Session(configs)
        session.stage = 'aligned'
        for _, robot, leader in configs:
            q = np.asarray(robot.reset_q, dtype=float)
            session.arms.append(Mock(joints=Mock(return_value=q)))
            session.readers.append(Mock(config=leader))
            target = q.copy()
            target[1] += np.deg2rad(4)
            session.mappers.append(Mock(config=leader, target=Mock(return_value=(target, None))))
        with self.assertRaisesRegex(ValueError, 'left: .*J2/ID 2.*偏差=\\+4.00°'):
            session.start_follow()
        for arm, reader in zip(session.arms, session.readers):
            arm.prepare_follow.assert_not_called()
            reader.set_torque.assert_not_called()
        self.assertEqual(session.stage, 'aligned')

    def test_arm_trial_rejects_successful_command_without_actual_motion(self):
        arm = Mock()
        arm.config.robot_ip = 'test-arm'
        arm.joints.side_effect = [np.zeros(7), np.zeros(7)]
        target = np.zeros(7)
        target[0] = np.deg2rad(1)
        with self.assertRaisesRegex(RuntimeError, 'J1 .*允许误差=0.25'):
            _move_arm(arm, target)

    def test_arm_trial_accepts_measured_target(self):
        arm = Mock()
        arm.config.robot_ip = 'test-arm'
        target = np.zeros(7)
        target[0] = np.deg2rad(1)
        arm.joints.side_effect = [np.zeros(7), target.copy()]
        _move_arm(arm, target)
        self.assertEqual(arm.joints.call_count, 2)

    def test_follow_releases_mapped_joints_and_gripper_before_following(self):
        configs = load_configs(SOURCE, dual=True)
        session = Session(configs)
        session.stage = 'aligned'
        session.arms = []
        session.readers = []
        session.mappers = []
        for _, robot, leader in configs:
            q = np.asarray(robot.reset_q, dtype=float)
            session.arms.append(Mock(joints=Mock(return_value=q)))
            reader = Mock(config=leader)
            session.readers.append(reader)
            session.mappers.append(Mock(config=leader, target=Mock(return_value=(q, 0))))
        session.start_follow()
        self.assertEqual(session.stage, 'following')
        for reader in session.readers:
            reader.set_torque.assert_called_once_with(
                False, ids=list(reader.config.joint_ids) + [reader.config.gripper_id])

    def test_follow_does_not_start_if_torque_release_fails(self):
        configs = load_configs(SOURCE, dual=True)
        session = Session(configs)
        session.stage = 'aligned'
        session.arms = []
        session.readers = []
        session.mappers = []
        for _, robot, leader in configs:
            q = np.asarray(robot.reset_q, dtype=float)
            session.arms.append(Mock(joints=Mock(return_value=q)))
            session.readers.append(Mock(config=leader))
            session.mappers.append(Mock(config=leader, target=Mock(return_value=(q, 0))))
        session.readers[0].set_torque.side_effect = RuntimeError('release failed')
        with self.assertRaisesRegex(RuntimeError, 'release failed'):
            session.start_follow()
        self.assertEqual(session.stage, 'aligned')

    def test_manual_alignment_holds_each_leader_at_its_reference(self):
        configs = load_configs(SOURCE, dual=True)
        session = Session(configs)
        session.stage = 'reset'
        session.mappers = [JointMapper(robot, leader) for _, robot, leader in configs]

        class FakeArm:
            def __init__(self, robot):
                self.robot = robot

            def health(self, active=False):
                return 0

            def joints(self):
                return np.asarray(self.robot.reset_q)

        class FakeReader:
            def __init__(self, leader):
                self.config = leader
                self.hold_calls = 0

            def read(self):
                return np.r_[np.linspace(0.1, 0.7, 7), 0.0]

            def hold(self):
                self.hold_calls += 1

        session.arms = [FakeArm(robot) for _, robot, _ in configs]
        session.readers = [FakeReader(leader) for _, _, leader in configs]
        session.align('left')
        session.align('right')
        self.assertEqual(session.stage, 'aligned')
        self.assertEqual([reader.hold_calls for reader in session.readers], [1, 1])

    def test_manual_alignment_saves_feedback_after_torque_turns_on(self):
        configs = load_configs(SOURCE, dual=True)
        session = Session(configs)
        session.stage = 'reset'
        session.mappers = [JointMapper(robot, leader) for _, robot, leader in configs]

        class FakeArm:
            def __init__(self, robot):
                self.robot = robot

            def health(self, active=False):
                return 0

            def joints(self):
                return np.asarray(self.robot.reset_q)

        class FakeReader:
            def __init__(self, leader):
                self.config = leader
                self.held = False

            def read(self):
                values = np.r_[np.linspace(0.1, 0.7, 7), 0.0]
                if not self.held:
                    values[2] -= 2 * np.pi
                return values

            def hold(self):
                self.held = True

        session.arms = [FakeArm(robot) for _, robot, _ in configs]
        session.readers = [FakeReader(leader) for _, _, leader in configs]
        session.align('left')
        self.assertAlmostEqual(session.mappers[0].leader_reference_q[2], 0.3)
        self.assertEqual(session.stage, 'aligning')

    def test_saved_mapping_reloads_with_signs_offsets_and_reference(self):
        source_data = yaml.safe_load(SOURCE.read_text())
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'dual.yaml'
            saved = Path(folder) / 'dual_calibrated.yaml'
            source.write_text(yaml.safe_dump(source_data))
            results = {}
            references = {}
            for side in ('left', 'right'):
                robot_q = np.asarray(source_data[side]['RobotConfig']['reset_q'])
                signs = np.array([1, -1, 1, -1, 1, -1, 1])
                reference = np.linspace(0.1, 0.7, 7)
                results[side] = calibration_values(robot_q, reference, signs)
                references[side] = reference
            save_result(source, saved, results)

            configs = load_configs(saved, dual=True, require_calibrated=True)
            for side, robot, leader in configs:
                mapper = JointMapper(robot, leader)
                raw = np.r_[references[side], 0.0]  # eighth Dynamixel is the gripper
                mapper.align(raw)
                mapped, _ = mapper.target(raw)
                np.testing.assert_allclose(mapped, robot.reset_q)
                changed = raw.copy()
                changed[1] -= 0.05  # sign=-1 means xArm J2 increases
                mapped, _ = mapper.target(changed)
                self.assertAlmostEqual(mapped[1], robot.reset_q[1] + 0.05)

    def test_manual_alignment_saves_sidecar_and_keeps_new_pwm_settings(self):
        source_data = yaml.safe_load(SOURCE.read_text())
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'dual.yaml'
            saved = Path(folder) / 'dual_calibrated.yaml'
            source.write_text(yaml.safe_dump(source_data))
            configs = load_configs(source, dual=True)
            mappers = []
            for _, robot, leader in configs:
                mapper = JointMapper(robot, leader)
                mapper.align(np.r_[np.linspace(0.1, 0.7, 7), 0.0])
                mappers.append(mapper)
            save_current_alignment(source, saved, configs, mappers)
            self.assertEqual(calibration_file_for(source), saved)

            source_data['left']['TeleoperatorConfig']['hold_pwm_by_joint'][1] = 450
            source.write_text(yaml.safe_dump(source_data))
            reloaded = load_configs(source, dual=True, calibration_path=saved)
            self.assertEqual(reloaded[0][2].hold_pwm_by_joint[1], 450)
            np.testing.assert_allclose(reloaded[0][2].joint_offsets, mappers[0].offsets)

            source_data['left']['TeleoperatorConfig']['joint_ids'][0] = 9
            with self.assertRaisesRegex(ValueError, 'joint_ids differs'):
                apply_saved_calibration(source_data, yaml.safe_load(saved.read_text()))


if __name__ == '__main__':
    unittest.main()
