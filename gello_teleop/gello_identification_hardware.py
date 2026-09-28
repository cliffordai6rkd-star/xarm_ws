"""Isolated XL330 bus access for identification; never connects to an xArm."""
import json
import math
from pathlib import Path
import signal
import threading
import time

import numpy as np

from gello_teleop.gello_hardware import check
from gello_teleop.gello_identification import ExcitationPlan, save_npz
from gello_teleop.build_gello_urdf import vector


def signed(value, bits):
    return value-(1 << bits) if value & (1 << (bits-1)) else value


class IdentificationBus:
    """Closing a read-only session closes ONLY the port, without torque writes."""

    def __init__(self, config):
        from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite
        self.config = config
        self.ids = list(config['joint_ids'])
        if not self.ids or len(set(self.ids)) != len(self.ids) or any(
                type(i) is not int or not 0 <= i <= 252 for i in self.ids):
            raise ValueError('Invalid joint_ids')
        if not config.get('port') or int(config['baudrate']) <= 0:
            raise ValueError('Supply a GELLO port and baudrate')
        self.port = PortHandler(config['port'])
        self.packet = PacketHandler(2.0)
        self.active = False
        self.original = None
        self.receive_ns = {}
        bus = self

        class CheckedGroup(GroupSyncRead):
            def rxPacket(self):
                self.last_result = False
                for motor_id in self.data_dict:
                    data, comm, error = self.ph.readRx(self.port, motor_id, self.data_length)
                    check(comm, f'ID {motor_id} sync read')
                    check(error, f'ID {motor_id} device error')
                    self.data_dict[motor_id] = data
                    bus.receive_ns[motor_id] = time.monotonic_ns()
                self.last_result = True
                return 0

        try:
            if not self.port.openPort() or not self.port.setBaudRate(int(config['baudrate'])):
                raise ConnectionError(f'Cannot open/set baudrate on {config["port"]}')
            self.port.ser.exclusive = True
            self.port.ser.write_timeout = float(config.get('max_sample_gap_s', 0.15))
            self.feedback = CheckedGroup(self.port, self.packet, 120, 27)
            self.health = CheckedGroup(self.port, self.packet, 64, 7)
            self.goals = GroupSyncWrite(self.port, self.packet, 116, 4)
            for group in [self.feedback, self.health]:
                for motor_id in self.ids:
                    if not group.addParam(motor_id):
                        raise RuntimeError(f'Cannot add motor {motor_id} to sync read')
            # Check current/velocity register units before collecting even passively.
            for motor_id in self.ids:
                if self.read_register(motor_id, 0, 2) not in (1190, 1200):
                    raise ValueError(f'ID {motor_id}: this collector supports XL330-M077/M288 only')
        except BaseException:
            self.port.closePort()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        cleanup = self.close()
        if cleanup and exc is None:
            raise RuntimeError('Cleanup incomplete: '+ '; '.join(cleanup))
        if cleanup and exc is not None:
            print('Cleanup incomplete: '+ '; '.join(cleanup), flush=True)

    def read_register(self, motor_id, address, size):
        method = {1: self.packet.read1ByteTxRx, 2: self.packet.read2ByteTxRx,
                  4: self.packet.read4ByteTxRx}[size]
        value, comm, error = method(self.port, motor_id, address)
        check(comm, f'ID {motor_id} read {address}')
        check(error, f'ID {motor_id} device error at {address}')
        return value

    def write_register(self, motor_id, address, size, value):
        method = {1: self.packet.write1ByteTxRx, 2: self.packet.write2ByteTxRx,
                  4: self.packet.write4ByteTxRx}[size]
        comm, error = method(self.port, motor_id, address, value)
        check(comm, f'ID {motor_id} write {address}')
        check(error, f'ID {motor_id} device error at {address}')

    def inspect(self):
        registers = [('model', 0, 2), ('drive_mode', 10, 1), ('mode', 11, 1),
                     ('homing_offset', 20, 4), ('pwm_limit', 36, 2), ('current_limit_ma', 38, 2),
                     ('position_max', 48, 4), ('position_min', 52, 4),
                     ('torque_on', 64, 1), ('hardware_error', 70, 1), ('watchdog', 98, 1),
                     ('goal_pwm', 100, 2), ('profile_acceleration', 108, 4),
                     ('profile_velocity', 112, 4), ('goal_position', 116, 4)]
        motors = []
        for motor_id in self.ids:
            status = {'id': motor_id}
            for name, address, size in registers:
                status[name] = self.read_register(motor_id, address, size)
            status['homing_offset'] = signed(status['homing_offset'], 32)
            status['watchdog'] = signed(status['watchdog'], 8)
            motors.append(status)
        return {'port': self.config['port'], 'baudrate': self.config['baudrate'],
                'read_only': True, 'motors': motors,
                'current_measurement': 'XL330 input supply current; not calibrated output torque'}

    def sample(self):
        begin = time.monotonic_ns()
        check(self.feedback.txRxPacket(), 'Telemetry sync read')
        receive_ns = np.array([self.receive_ns[i] for i in self.ids], dtype=np.int64)
        values = {name: [] for name in ['raw_q_rad', 'raw_dq_rad_s', 'current_a', 'pwm_raw',
                                       'voltage_v', 'temperature_c', 'servo_tick_ms', 'hardware_error', 'torque_on']}
        for motor_id in self.ids:
            def read(address, size):
                if not self.feedback.isAvailable(motor_id, address, size):
                    raise RuntimeError(f'ID {motor_id}: missing register {address}')
                return self.feedback.getData(motor_id, address, size)
            values['raw_q_rad'].append(signed(read(132, 4), 32)*np.pi/2048)
            values['raw_dq_rad_s'].append(signed(read(128, 4), 32)*0.229*2*np.pi/60)
            values['current_a'].append(signed(read(126, 2), 16)*0.001)
            values['pwm_raw'].append(signed(read(124, 2), 16))
            values['voltage_v'].append(read(144, 2)*0.1)
            values['temperature_c'].append(read(146, 1))
            values['servo_tick_ms'].append(read(120, 2))
        check(self.health.txRxPacket(), 'Hardware health sync read')
        for motor_id in self.ids:
            if not self.health.isAvailable(motor_id, 70, 1) or not self.health.isAvailable(motor_id, 64, 1):
                raise RuntimeError(f'ID {motor_id}: missing hardware state')
            values['hardware_error'].append(self.health.getData(motor_id, 70, 1))
            values['torque_on'].append(self.health.getData(motor_id, 64, 1))
        values = {name: np.asarray(value) for name, value in values.items()}
        values.update({'read_started_ns': begin, 'read_finished_ns': time.monotonic_ns(),
                       'receive_ns': receive_ns})
        return values

    def arm(self, plan, sample, stop=None, initial_status=None):
        """Validate all motors and full path before the first register write."""
        return self._arm(plan, sample, stop, initial_status, require_verified_coordinates=True)

    def arm_direction_check(self, plan, sample, stop=None, initial_status=None):
        """Bounded encoder-space calibration, before the URDF mapping is known."""
        from gello_teleop.gello_direction_motion import DirectionCheckPlan
        if not isinstance(plan, DirectionCheckPlan) or plan.ids != self.ids:
            raise ValueError('Direction arming requires the matching bounded calibration plan')
        return self._arm(plan, sample, stop, initial_status, require_verified_coordinates=False)

    def _arm(self, plan, sample, stop, initial_status, require_verified_coordinates):
        config = self.config
        if require_verified_coordinates and config.get('joint_coordinates_verified') is not True:
            raise ValueError('Confirm encoder coordinates match the leader URDF before driving')
        self.original = (initial_status if initial_status is not None else self.inspect())['motors']
        pwm = vector(config['pwm_ceiling'], 'pwm_ceiling', plan.n)
        velocity = int(math.floor(plan.max_velocity/(0.229*2*np.pi/60)))
        acceleration = int(math.floor(plan.max_acceleration/(214.577*2*np.pi/3600)))
        watchdog = int(math.ceil(float(config['watchdog_s'])/0.02))
        if not 1 <= velocity <= 32767 or not 1 <= acceleration <= 32767:
            raise ValueError('Requested speed/acceleration cannot be represented by a nonzero XL330 profile')
        if np.any(plan.velocity_bound > velocity*0.229*2*np.pi/60) or np.any(
                plan.acceleration_bound > acceleration*214.577*2*np.pi/3600):
            raise ValueError('Excitation exceeds the quantized profile limits; lengthen duration')
        if not 1 <= watchdog <= 127 or watchdog*0.02 <= float(config['max_sample_gap_s']):
            raise ValueError('Watchdog must be > max_sample_gap_s and <=2.54 s')
        actual = plan.signs*(sample['raw_q_rad']-plan.offsets)
        if np.any(actual < plan.lower) or np.any(actual > plan.upper):
            raise ValueError('Current pose is outside the measured mechanical limits')
        if np.max(np.abs(actual-plan.center)) > float(config['start_tolerance_rad']):
            raise ValueError('Manually align the leader to center_rad before execution; no automatic approach move')
        if np.max(np.abs(sample['raw_dq_rad_s'])) > 0.03:
            raise ValueError('Leader must be stationary before execution')
        for i, motor in enumerate(self.original):
            if motor['mode'] != 3 or motor['drive_mode'] & 12 or motor['homing_offset'] != 0:
                raise ValueError(f'ID {motor["id"]}: need mode 3, velocity-based profile, '
                                 'Torque On by Goal Update disabled and zero Homing Offset')
            if motor['torque_on'] or motor['hardware_error'] or motor['watchdog'] < 0:
                raise ValueError(f'ID {motor["id"]}: require torque off and no hardware/watchdog error')
            if not float(pwm[i]).is_integer() or not 1 <= pwm[i] <= min(885, motor['pwm_limit']):
                raise ValueError(f'ID {motor["id"]}: invalid PWM ceiling')
            path = (plan.raw_goal_bounds[:, i] if hasattr(plan, 'raw_goal_bounds') else
                    plan.offsets[i]+np.array([plan.center[i]-plan.amplitude[i],
                                             plan.center[i]+plan.amplitude[i]])/plan.signs[i])
            ticks = np.rint(path*2048/np.pi)
            if np.min(ticks) < motor['position_min'] or np.max(ticks) > motor['position_max']:
                raise ValueError(f'ID {motor["id"]}: path exceeds EEPROM position limits')
            present_ticks = int(round(sample['raw_q_rad'][i]*2048/np.pi))
            if not motor['position_min'] <= present_ticks <= motor['position_max']:
                raise ValueError(f'ID {motor["id"]}: current position cannot be used as a single-turn goal')
        self.active = True  # Any partial initialization is cleaned up on failure.
        for i, motor_id in enumerate(self.ids):
            if stop is not None and stop.is_set():
                raise InterruptedError('Arming interrupted')
            # Initialize goal to the present pose before torque is enabled.
            goal = int(round(sample['raw_q_rad'][i]*2048/np.pi))
            self.write_register(motor_id, 116, 4, goal)
            self.write_register(motor_id, 100, 2, int(pwm[i]))
            self.write_register(motor_id, 108, 4, acceleration)
            self.write_register(motor_id, 112, 4, velocity)
            self.write_register(motor_id, 98, 1, watchdog)
        # Configuration writes take time at 57600 baud. Reject pose changes and
        # refresh every goal immediately before enabling torque.
        fresh = self.sample()
        check_sample(fresh, config)
        actual = plan.signs*(fresh['raw_q_rad']-plan.offsets)
        if np.any(fresh['torque_on']) or np.any(actual < plan.lower) or np.any(actual > plan.upper) or (
                np.max(np.abs(actual-plan.center)) > float(config['start_tolerance_rad'])) or (
                np.max(np.abs(fresh['raw_dq_rad_s'])) > 0.03):
            raise ValueError('Leader pose/state changed during arming; torque remains disabled')
        self.command(actual, plan)
        for motor_id in self.ids:
            if stop is not None and stop.is_set():
                raise InterruptedError('Arming interrupted')
            self.write_register(motor_id, 64, 1, 1)

    def command(self, q, plan):
        ticks = np.rint((plan.offsets+np.asarray(q)/plan.signs)*2048/np.pi).astype(np.int64)
        self.goals.clearParam()
        for motor_id, goal, original in zip(self.ids, ticks, self.original):
            if not original['position_min'] <= goal <= original['position_max']:
                raise ValueError(f'ID {motor_id}: refusing out-of-range goal; never wrap ticks')
            if not self.goals.addParam(motor_id, list(int(goal).to_bytes(4, 'little'))):
                raise RuntimeError(f'Cannot add ID {motor_id} to command packet')
        check(self.goals.txPacket(), 'Goal Position sync write')

    def close(self):
        failures = []
        if self.active:
            for motor_id in self.ids:
                try:
                    self.write_register(motor_id, 64, 1, 0)
                except Exception as exc:
                    failures.append(f'torque off ID {motor_id}: {exc}')
            for motor in self.original or []:
                for address, size, value in [(98, 1, 0), (100, 2, motor['goal_pwm']),
                                              (108, 4, motor['profile_acceleration']),
                                              (112, 4, motor['profile_velocity']),
                                              (98, 1, motor['watchdog'])]:
                    try:
                        self.write_register(motor['id'], address, size, value)
                    except Exception as exc:
                        failures.append(f'restore ID {motor["id"]} register {address}: {exc}')
            self.active = False
        self.cleanup_failures = failures
        self.port.closePort()
        return failures


def check_sample(sample, config, plan=None, command=None):
    n = len(config['joint_ids'])
    maximum = vector(config['max_current_a'], 'max_current_a', n)
    if np.any(maximum <= 0) or np.any(np.abs(sample['current_a']) > maximum):
        raise RuntimeError('Current exceeds configured threshold')
    if np.any(sample['temperature_c'] >= float(config['max_temperature_c'])):
        raise RuntimeError('Temperature exceeds configured threshold')
    if np.any(sample['voltage_v'] < float(config['min_voltage_v'])) or np.any(
            sample['voltage_v'] > float(config['max_voltage_v'])):
        raise RuntimeError('Voltage outside configured range')
    if np.any(sample['hardware_error']):
        raise RuntimeError(f'Hardware errors: {sample["hardware_error"].tolist()}')
    if plan is not None:
        actual = plan.signs*(sample['raw_q_rad']-plan.offsets)
        if np.any(sample['torque_on'] != 1):
            raise RuntimeError('Unexpected torque disable/shutdown')
        if np.any(actual < plan.lower) or np.any(actual > plan.upper):
            raise RuntimeError('Measured position outside mechanical limits')
        if np.max(np.abs(actual-command)) > float(config['max_tracking_error_rad']):
            raise RuntimeError('Leader tracking error exceeds configured threshold')
        if np.max(np.abs(sample['raw_dq_rad_s'])) > plan.max_velocity+0.025:
            raise RuntimeError('Measured joint velocity exceeds configured threshold')


def collect(config, output, execute=False, duration=None):
    if Path(output).exists():
        raise FileExistsError(output)
    plan = ExcitationPlan(config) if execute else None
    duration = plan.duration if execute else float(duration if duration is not None else config['duration_s'])
    hz = float(config['sample_hz'])
    gap_limit = float(config['max_sample_gap_s'])
    if not np.isfinite([duration, hz, gap_limit]).all() or min(duration, hz, gap_limit) <= 0:
        raise ValueError('duration/sample_hz/max_sample_gap_s must be positive')
    # Validate monitor settings BEFORE opening the port, including passive reads.
    n = len(config['joint_ids'])
    maximum = vector(config['max_current_a'], 'max_current_a', n)
    if np.any(maximum <= 0) or not 0 < float(config['max_temperature_c']) <= 70 or not (
            3.7 <= float(config['min_voltage_v']) < float(config['max_voltage_v']) <= 6.0):
        raise ValueError('Invalid XL330 current/temperature/voltage thresholds')
    for name in ['max_tracking_error_rad', 'start_tolerance_rad']:
        if not np.isfinite(float(config[name])) or float(config[name]) <= 0:
            raise ValueError(f'{name} must be positive')
    stop = threading.Event()
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in [signal.SIGINT, signal.SIGTERM]:
            old_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    rows, targets, command_times = [], [], []
    failure = None
    bus = None
    status = 'aborted'
    metadata = {'joint_names': [f'joint{i+1}' for i in range(n)], 'config': config,
                'joint_coordinates_verified': config.get('joint_coordinates_verified') is True,
                'elastic_elements': config.get('elastic_elements', 'unknown'),
                'execute': execute, 'torque_measured': False,
                'timestamp_note': 'time_s uses last servo response reception relative to session_start_ns. '
                                  'Sync-read servos reply sequentially, not simultaneously. '
                                  'receive_ns and servo_tick_ms preserve timing/skew for analysis.',
                'current_note': 'XL330 supply current in A; cannot be used as output torque without calibration.'}
    try:
        with IdentificationBus(config) as bus:
            metadata['initial_hardware'] = bus.inspect()
            first = bus.sample()
            check_sample(first, config)
            if execute:
                if stop.is_set():
                    raise InterruptedError('Collection interrupted before arming')
                bus.arm(plan, first, stop, metadata['initial_hardware'])
            start = time.monotonic_ns()
            metadata['session_start_ns'] = start
            previous = start
            next_sample = start
            command = plan.center if execute else np.full(n, np.nan)
            while not stop.is_set():
                now = time.monotonic_ns()
                t = min((now-start)*1e-9, duration)
                if execute:
                    command = plan.evaluate(t)[0]
                    bus.command(command, plan)
                sample = bus.sample()
                rows.append(sample)
                targets.append(command.copy())
                command_times.append(t)
                check_sample(sample, config)
                if execute:
                    check_sample(sample, config, plan, command)
                    if (sample['read_finished_ns']-previous)*1e-9 > gap_limit:
                        raise TimeoutError('Telemetry/command gap exceeded; trajectory aborted')
                previous = sample['read_finished_ns']
                if t >= duration:
                    if execute and np.max(np.abs(plan.signs*(sample['raw_q_rad']-plan.offsets)
                                                 -plan.center)) > float(config['start_tolerance_rad']):
                        raise RuntimeError('Leader did not return to the trajectory center')
                    status = 'complete'
                    break
                next_sample += int(1e9/hz)
                stop.wait(max(0, (next_sample-time.monotonic_ns())*1e-9))
            if stop.is_set():
                raise KeyboardInterrupt('Collection interrupted')
    except BaseException as exc:
        failure = exc
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if rows:
            arrays = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
            arrays['time_s'] = (arrays['receive_ns'][:, -1]-metadata['session_start_ns'])*1e-9
            arrays['command_time_s'] = np.asarray(command_times)
            arrays['command_q_rad'] = np.asarray(targets)
            if config.get('encoder_offsets_rad') is not None:
                signs = vector(config['encoder_signs'], 'encoder_signs', n)
                offsets = vector(config['encoder_offsets_rad'], 'encoder_offsets_rad', n)
                arrays['q_rad'] = (arrays['raw_q_rad']-offsets)*signs
                arrays['dq_feedback_rad_s'] = arrays['raw_dq_rad_s']*signs
            else:
                metadata['joint_coordinates_verified'] = False
                arrays['q_rad'] = arrays['raw_q_rad']
            if failure:
                status = 'aborted'
            metadata['session_status'] = status
            metadata['failure'] = str(failure) if failure else None
            metadata['cleanup_failures'] = getattr(bus, 'cleanup_failures', [])
            metadata['sample_count'] = len(rows)
            metadata['actual_sample_hz'] = float(1/np.median(np.diff(arrays['time_s']))) if len(rows)>1 else None
            metadata['maximum_bus_read_s'] = float(np.max(arrays['read_finished_ns']-arrays['read_started_ns'])*1e-9)
            metadata['torque_on'] = bool(np.any(arrays['torque_on']))
            metadata['maximum_response_skew_s'] = float(np.max(np.ptp(arrays['receive_ns'], axis=1))*1e-9)
            save_npz(output, metadata_json=json.dumps(metadata), **arrays)
            print(json.dumps({key: metadata[key] for key in ['session_status', 'sample_count',
                              'actual_sample_hz', 'maximum_bus_read_s', 'failure']}, indent=2))
    if failure:
        raise failure
