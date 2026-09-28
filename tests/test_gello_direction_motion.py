"""Hardware-free checks of supervised direction motion and arming isolation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from gello_teleop.gello_direction_motion import DirectionCheckPlan, DirectionSession, run_active
from gello_teleop.gello_identification import initial_config
from gello_teleop.gello_identification_hardware import IdentificationBus
from gello_teleop.measure_gello_ranges import measured_config, run as measure_ranges


def configuration():
    c = initial_config({'joint_ids': list(range(1, 8))})
    c.update(encoder_offsets_rad=[np.pi]*7, encoder_signs=[1, -1, 1, 1, 1, -1, 1],
             lower_rad=[-1]*7, upper_rad=[1]*7, joint_coordinates_verified=False)
    return c


class FakeClock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def wait(self, duration):
        self.now += duration
        return False

    def is_set(self):
        return False


class FakeBus:
    def __init__(self, clock):
        self.clock = clock
        self.q = np.full(7, np.pi)
        self.enabled = False
        self.closed = False
        self.cleanup_failures = []
        self.commands = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.enabled = False
        self.closed = True

    def inspect(self):
        return {'motors': [{'torque_on': 0}]*7}

    def arm_direction_check(self, plan, *_):
        self.enabled = True
        self.q = plan.center.copy()

    def command(self, q, _):
        self.q = np.asarray(q).copy()
        self.commands.append(self.q.copy())

    def sample(self):
        ns = int(self.clock()*1e9)
        return dict(raw_q_rad=self.q.copy(), raw_dq_rad_s=np.zeros(7), current_a=np.zeros(7),
                    temperature_c=np.full(7, 25), voltage_v=np.full(7, 5),
                    hardware_error=np.zeros(7), torque_on=np.full(7, int(self.enabled)),
                    read_started_ns=ns, read_finished_ns=ns)


class DirectionMotionTest(unittest.TestCase):
    def test_targets_use_captured_raw_pose_and_only_one_joint(self):
        c = configuration()
        raw = np.full(7, np.pi)+.1
        p = DirectionCheckPlan(c, raw)
        np.testing.assert_allclose(p.targets-raw, np.diag(np.asarray(c['encoder_signs'])*np.deg2rad(15)))
        np.testing.assert_allclose(p.interpolate(raw, p.targets[0], 0), raw)
        np.testing.assert_allclose(p.interpolate(raw, p.targets[0], 8), p.targets[0])
        self.assertTrue(np.all(p.velocity_bound < p.max_velocity))

    def test_all_paths_checked_before_enabling(self):
        c = configuration()
        raw = np.full(7, np.pi)
        raw[6] += .9
        with self.assertRaisesRegex(ValueError, 'J7.*exceeds'):
            DirectionCheckPlan(c, raw)
        with self.assertRaisesRegex(ValueError, 'too short'):
            DirectionCheckPlan(c, np.full(7, np.pi), move_seconds=1)

    def test_unverified_coordinates_still_block_general_excitation(self):
        bus = IdentificationBus.__new__(IdentificationBus)
        bus.config = configuration()
        with self.assertRaisesRegex(ValueError, 'Confirm encoder'):
            bus.arm(Mock(), {})
        bus.ids = list(range(1, 8))
        with self.assertRaisesRegex(ValueError, 'bounded calibration plan'):
            bus.arm_direction_check(Mock(), {})

    def test_waiting_for_answer_continues_feedback_and_detects_fault(self):
        clock = FakeClock()
        bus = FakeBus(clock)
        bus.enabled = True
        c = configuration()
        session = DirectionSession(bus, DirectionCheckPlan(c, bus.q), c, clock,
                                   clock=clock, read_line=lambda _: None, answer_timeout=.5)
        session.previous_read = 0
        with self.assertRaises(TimeoutError), contextlib.redirect_stdout(io.StringIO()):
            session.ask('answer', bus.q, 0, 'hold')
        self.assertGreaterEqual(len(bus.commands), 4)
        bus.sample = lambda: dict(FakeBus.sample(bus), hardware_error=np.ones(7))
        with self.assertRaisesRegex(RuntimeError, 'Hardware errors'):
            session.tick(bus.q, 0, 'hold')

    def test_active_arming_validates_last_motor_and_cleans_partial_enable(self):
        clock, c = FakeClock(), configuration()
        fake = FakeBus(clock)
        bus = IdentificationBus.__new__(IdentificationBus)
        bus.config, bus.ids, bus.active = c, list(range(1, 8)), False
        bus.port, bus.write_register, bus.command = Mock(), Mock(), Mock()
        bus.sample = fake.sample
        motors = [dict(id=i, mode=3, drive_mode=0, homing_offset=0, torque_on=0,
                       hardware_error=0, watchdog=0, pwm_limit=885, position_min=0,
                       position_max=4095, goal_pwm=200, profile_acceleration=0,
                       profile_velocity=0) for i in bus.ids]
        bus.inspect = lambda: {'motors': motors}
        plan = DirectionCheckPlan(c, fake.q)
        motors[-1]['mode'] = 0
        with self.assertRaisesRegex(ValueError, 'ID 7'):
            bus.arm_direction_check(plan, fake.sample())
        bus.write_register.assert_not_called()
        motors[-1]['mode'] = 3
        def fail_last(motor_id, address, size, value):
            if motor_id == 7 and address == 64 and value == 1:
                raise RuntimeError('enable failed')
        bus.write_register.side_effect = fail_last
        with self.assertRaisesRegex(RuntimeError, 'enable failed'):
            bus.arm_direction_check(plan, fake.sample())
        self.assertTrue(bus.active)
        self.assertEqual(bus.close(), [])
        for motor_id in bus.ids:
            bus.write_register.assert_any_call(motor_id, 64, 1, 0)
            bus.write_register.assert_any_call(motor_id, 112, 4, 0)

    def exercise(self, interrupt=False):
        c, clock = configuration(), FakeClock()
        bus = FakeBus(clock)
        original_session = DirectionSession
        answers = iter(['n']+['y']*6+[''])
        def line(allow):
            if interrupt and allow:
                return 'q'
            return next(answers) if allow else None
        def session_factory(*args, **kwargs):
            session = original_session(*args, **dict(kwargs, clock=clock, read_line=line))
            session.previous_read = 0
            return session
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'review.json'
            with patch('gello_teleop.gello_direction_motion.IdentificationBus', return_value=bus), \
                 patch('gello_teleop.gello_direction_motion.stable_snapshot', return_value=bus.q.copy()), \
                 patch('gello_teleop.gello_direction_motion.DirectionSession', side_effect=session_factory), \
                 patch('gello_teleop.gello_direction_motion.threading.Event', return_value=clock), \
                 patch('gello_teleop.gello_direction_motion.time.monotonic_ns', side_effect=lambda: int(clock()*1e9)), \
                 patch('gello_teleop.gello_direction_motion.sys.stdin.isatty', return_value=True), \
                 patch('builtins.input', return_value=''), contextlib.redirect_stdout(io.StringIO()):
                if interrupt:
                    with self.assertRaises(InterruptedError):
                        run_active(c, {'comparison_reference_q_deg': [90, 0, -90, 90, 0, 90, -90]}, output)
                else:
                    run_active(c, {'comparison_reference_q_deg': [90, 0, -90, 90, 0, 90, -90]}, output)
            report = json.loads(output.read_text())
            self.assertTrue(bus.closed)
            self.assertFalse(bus.enabled)
            self.assertTrue(output.with_suffix('.npz').exists())
            self.assertFalse(report['joint_coordinates_verified'])
            if interrupt:
                self.assertEqual(report['session_status'], 'aborted')
            else:
                self.assertEqual(report['session_status'], 'complete')
                self.assertEqual(len(report['joint_checks']), 7)
                self.assertEqual(report['joint_checks'][0]['viewer_positive_encoder_sign'], -1)
                np.testing.assert_allclose(bus.commands[-1], np.pi)
                # Every commanded pose differs from the reference in at most one axis.
                self.assertTrue(all(np.count_nonzero(abs(q-np.pi) > 1e-9) <= 1 for q in bus.commands))

    def test_full_sequence_returns_each_axis_and_records_negative_answer(self):
        self.exercise()

    def test_operator_abort_closes_torque_and_saves_partial_telemetry(self):
        self.exercise(interrupt=True)

    def test_invalid_initial_pose_is_logged_without_arming(self):
        c, clock = configuration(), FakeClock()
        bus = FakeBus(clock)
        bus.q[6] += 1.1
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'failed.json'
            with patch('gello_teleop.gello_direction_motion.IdentificationBus', return_value=bus), \
                 patch('gello_teleop.gello_direction_motion.stable_snapshot', return_value=bus.q.copy()), \
                 patch('gello_teleop.gello_direction_motion.sys.stdin.isatty', return_value=True), \
                 patch('builtins.input', return_value=''), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'J7.*raw=.*allowed raw='):
                    run_active(c, {'comparison_reference_q_deg': [90, 0, -90, 90, 0, 90, -90]}, output)
            report = json.loads(output.read_text())
            self.assertFalse(report['arming_started'])
            np.testing.assert_allclose(report['reference_raw_rad'], bus.q)
            self.assertEqual(bus.commands, [])
            self.assertTrue(bus.closed)

    def test_selective_range_remeasurement_keeps_six_existing_intervals(self):
        c, margin = configuration(), np.deg2rad(2)
        old_center = np.full(7, np.pi)
        old_center[6] = 1.6
        endpoints = [[2., 4.]]*6+[[1., 2.2]]
        c = measured_config(c, old_center, endpoints, margin)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'old.ranges.json'
            source.write_text(json.dumps(dict(read_only=True, joint_ids=c['joint_ids'],
                                              center_raw_rad=old_center.tolist(), endpoints_raw_rad=endpoints,
                                              inset_margin_rad=margin, endpoint_full_postures_raw_rad=[None]*7)))
            c['range_measurement_file'] = str(source)
            bus = FakeBus(FakeClock())
            bus.inspect = lambda: {'motors': [dict(torque_on=0, mode=3, homing_offset=0)]*7}
            center = np.full(7, np.pi)
            a, b = center.copy(), center.copy()
            a[6], b[6] = np.pi-.4, np.pi+.4
            output = Path(directory)/'updated.yaml'
            with patch('gello_teleop.measure_gello_ranges.IdentificationBus', return_value=bus), \
                 patch('gello_teleop.measure_gello_ranges.stable_snapshot', side_effect=[center, a, b]) as snap, \
                 patch('builtins.input', return_value=''), contextlib.redirect_stdout(io.StringIO()):
                measure_ranges(c, output, margin, joints=[7])
            self.assertEqual(snap.call_count, 3)
            updated = yaml.safe_load(output.read_text())
            for k in ['lower_rad', 'upper_rad']:
                np.testing.assert_allclose(updated[k][:6], c[k][:6])
            DirectionCheckPlan(updated, center, step_deg=15)
            report = json.loads(output.with_suffix('.ranges.json').read_text())
            self.assertEqual(report['reused_joint_numbers'], [1, 2, 3, 4, 5, 6])
            self.assertEqual(report['remeasured_joint_numbers'], [7])
            self.assertFalse(updated['joint_coordinates_verified'])
