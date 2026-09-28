"""Hardware-free checks for torque-to-position integration."""

import unittest

import numpy as np

from gello_teleop.mit_position import MitPositionController


class OneJointModel:
    dof = 1
    lower = np.array([-2.0])
    upper = np.array([2.0])

    def terms(self, q, dq):
        # M=2, h=g=5: gravity feed-forward must cancel at rest.
        return np.array([[2.0]]), np.array([5.0]), np.array([5.0])


class MitPositionControllerTest(unittest.TestCase):
    def new_controller(self, torque_limit=20.0, tracking_error_limit=1.0):
        return MitPositionController(
            OneJointModel(), kp=[4.0], kd=[0.0], torque_limit=[torque_limit],
            velocity_limit=2.0, acceleration_limit=10.0,
            tracking_error_limit=tracking_error_limit,
        )

    def test_integrates_acceleration_into_velocity_then_position(self):
        control = self.new_controller()
        control.reset([0.0], 0.0)
        first = control.step([1.0], [0.0], 0.1, 0.1)
        self.assertAlmostEqual(first[0], 0.02)  # ddq=2, dq=0.2, q=0.02
        self.assertAlmostEqual(control.dq_cmd[0], 0.2)
        second = control.step([1.0], [0.02], 0.2, 0.1)
        self.assertAlmostEqual(control.dq_cmd[0], 0.396)
        self.assertAlmostEqual(second[0], 0.0596)

    def test_gravity_compensation_holds_position(self):
        control = self.new_controller()
        control.reset([0.0], 0.0)
        self.assertAlmostEqual(control.step([0.0], [0.0], 0.1, 0.1)[0], 0.0)

    def test_torque_limit_applies_before_forward_dynamics(self):
        control = self.new_controller(torque_limit=6.0)
        control.reset([0.0], 0.0)
        self.assertAlmostEqual(control.step([1.0], [0.0], 0.1, 0.1)[0], 0.005)

    def test_tracking_error_rejects_command_without_advancing_state(self):
        control = self.new_controller(tracking_error_limit=0.01)
        control.reset([0.0], 0.0)
        with self.assertRaisesRegex(ValueError, 'tracking limit'):
            control.step([1.0], [0.0], 0.1, 0.1)
        self.assertAlmostEqual(control.q_cmd[0], 0.0)
        self.assertAlmostEqual(control.dq_cmd[0], 0.0)


if __name__ == '__main__':
    unittest.main()
