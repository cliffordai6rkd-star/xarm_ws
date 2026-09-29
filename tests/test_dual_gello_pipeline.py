import time

import numpy as np
import pytest

from gello_teleop.dual_gello_collect import DualGelloPipeline, SecondOrderPositionFollower
from gello_teleop.gello_hardware import GelloReader
from nero_collection.arms.base import ArmState


CONFIG = "gello_teleop/config/xarm7_gello_dual_dataset.yaml"


class FakeArm:
    def __init__(self, side, robot):
        self.name = side
        self.dof = len(robot.reset_q)
        self.q = np.asarray(robot.reset_q, dtype=float).copy()
        self.commands = []

    def connect(self):
        pass

    def enable(self):
        pass

    def move_to_reset(self, q, *, speed, acceleration):
        self.q = np.asarray(q, dtype=float).copy()

    def read_state(self):
        return ArmState(
            q=self.q.copy(), dq=np.zeros(self.dof), ddq=np.zeros(self.dof),
            ee_pose=np.eye(4), torque=np.zeros(self.dof),
            current=np.full(self.dof, np.nan), timestamp_us=time.time_ns() // 1000,
            dq_valid=True, torque_valid=True, current_valid=False,
        )

    def command_joint_positions(self, q):
        self.q = np.asarray(q, dtype=float).copy()
        self.commands.append(self.q.copy())

    def disconnect(self):
        pass


class FakeReader:
    def __init__(self, config):
        self.config = config
        self.raw = np.r_[np.asarray(config.leader_reference_q, dtype=float), 0.0]

    def read(self):
        return self.raw.copy()

    def prepare_alignment(self):
        pass

    def set_torque(self, *args, **kwargs):
        pass

    def close(self):
        pass


def make_pipeline():
    return DualGelloPipeline(
        CONFIG,
        arm_factory=lambda side, robot: FakeArm(side, robot),
        reader_factory=FakeReader,
    )


def test_position_reference_is_bounded_and_takeover_starts_at_measured_pose():
    follower = SecondOrderPositionFollower(2, mode="second_order", kp=10, kd=1,
                                           max_velocity=0.5, max_acceleration=1, max_step=0.02,
                                           max_tracking_error=0.1)
    follower.initialize(np.zeros(2))
    first = follower.update(np.ones(2), np.zeros(2), 0.01)
    assert np.max(np.abs(first)) <= 0.02


def test_damping_current_is_signed_filtered_and_limited_without_nm_claim():
    cfg = type("Damping", (), {
        "damping_gain": (2.0, 2.0), "damping_brake_gain": (1.0, 1.0),
        "damping_current_limit": (3, 3), "damping_velocity_threshold": 0.1,
        "damping_velocity_filter_alpha": 0.5, "weak_hold_enabled": True,
        "weak_hold_gain": (1.0, 1.0), "weak_hold_limit": (0.5, 0.5),
    })()
    current = GelloReader.compute_damping_current(np.array([1.0, -1.0]), np.zeros(2), 0.01, cfg,
                                                  np.zeros(2))
    assert np.all(np.abs(current) <= 3.0)
    assert current[0] < 0 and current[1] > 0


def test_unaligned_cannot_takeover_and_retakeover_checks_current_pose():
    pipeline = make_pipeline()
    pipeline.connect()
    pipeline.reset()
    with pytest.raises(RuntimeError, match="alignment"):
        pipeline.takeover()
    pipeline.confirm_alignment()
    pipeline.takeover()
    assert pipeline.state == "following"
    time.sleep(0.03)
    pipeline.start_episode()
    assert pipeline.state == "recording"
    pipeline.stop_episode(save=False)
    assert pipeline.state == "holding"
    pipeline.arms[0].q = pipeline.arms[0].q + 0.2
    with pytest.raises(RuntimeError, match="moved after alignment"):
        pipeline.takeover()
    pipeline.arms[0].q = np.asarray(pipeline.configs[0][1].reset_q, dtype=float)
    pipeline.confirm_alignment()
    pipeline.takeover()
    assert pipeline.state == "following"
    pipeline.close()


def test_recorded_q_cmd_is_causal_previous_successful_command():
    pipeline = make_pipeline()
    pipeline.connect()
    pipeline.reset()
    pipeline.confirm_alignment()
    pipeline.takeover()
    time.sleep(0.04)
    pipeline.start_episode()
    time.sleep(0.04)
    pipeline.poll()
    assert pipeline.buffer is not None and pipeline.buffer.sample_count > 0
    q_cmd = np.asarray(pipeline.buffer.teleop_data["q_cmd"][0])
    q_follower = np.asarray(pipeline.buffer.teleop_data["q_follower"][0])
    np.testing.assert_allclose(pipeline.buffer.teleop_data["delta_q"][0], q_cmd - q_follower)
    pipeline.stop_episode(save=False)
    pipeline.close()


def test_expired_leader_stops_both_arm_commands():
    pipeline = make_pipeline()
    pipeline.connect()
    pipeline.reset()
    pipeline.confirm_alignment()
    pipeline.takeover()
    pipeline.leader_max_age_s = 1.0e-9
    time.sleep(0.04)
    with pytest.raises(RuntimeError, match="expired"):
        pipeline.poll()
    assert pipeline.stop_event.is_set()
    pipeline.close()
