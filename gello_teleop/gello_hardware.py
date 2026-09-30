"""Hardware adapters for GELLO. Importing this module never opens a device."""
import math
import threading
import time

import numpy as np


# Control-table variants are selected from the model number returned by
# Dynamixel ping.  A damping request for an unknown model is rejected instead
# of silently applying an XL/XM register map to a different servo.
_KNOWN_CURRENT_MODELS = {
    1000: "XH430-W350",
    1010: "XH430-W210",
    1020: "XM430-W350",
    1030: "XM430-W210",
    1040: "XH430-V350",
    1050: "XH430-V210",
    1190: "XL330-M077",
    1200: "XL330-M288",
    1210: "XC330-T181",
    1220: "XC330-T288",
    1230: "XC330-M181",
    1240: "XC330-M288",
}
_CURRENT_TABLE = {
    "operating_mode": 11,
    "current_limit": 38,
    "torque_enable": 64,
    "watchdog": 98,
    "goal_current": 102,
    "present_velocity": 128,
    "velocity_unit_rad_s": 0.229 * 2 * math.pi / 60.0,
}

# XL330 control-table maximum Current Limit, raw units (1 mA/unit).
_ALIGNMENT_MAX_CURRENT = {1190: 1750, 1200: 1750}


def check(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation} failed (code={code})")


class GelloReader:
    """One synchronous, checked Dynamixel transaction per sample; no cached fallback."""

    def __init__(self, config):
        from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite

        class CheckedSyncRead(GroupSyncRead):
            def rxPacket(self):
                # Upstream GroupSyncRead drops individual servo error bytes.
                self.last_result = False
                for motor_id in self.data_dict:
                    data, comm, error = self.ph.readRx(self.port, motor_id, self.data_length)
                    check(comm, f'Dynamixel {motor_id} response')
                    check(error, f'Dynamixel {motor_id} device error')
                    self.data_dict[motor_id] = data
                self.last_result = True
                return 0

        self.config = config
        self.ids = list(config.joint_ids)
        if config.gripper_id >= 0:
            self.ids.append(config.gripper_id)
        self.ids.extend(config.torque_joint_ids or ())
        self.port = PortHandler(config.port)
        self.packet = PacketHandler(2.0)
        self.held_ids = []
        self.lock = threading.Lock()
        self.closed = False
        self.model_numbers = {}
        try:
            if not self.port.openPort():
                raise ConnectionError(f"Cannot open GELLO port: {config.port}")
            if not self.port.setBaudRate(config.baudrate):
                raise ConnectionError(f"Cannot set baudrate on {config.port}")
            # setBaudRate reopens the port, so apply the lock to the final descriptor.
            self.port.ser.exclusive = True
            self.port.ser.write_timeout = config.read_timeout
            self.group = CheckedSyncRead(self.port, self.packet, 132, 4)
            self.reset_group = CheckedSyncRead(self.port, self.packet, 124, 12)
            self.current_writer = GroupSyncWrite(self.port, self.packet, 102, 2)
            self.current_feedback = CheckedSyncRead(self.port, self.packet, 102, 2)
            for motor_id in self.ids:
                if not self.group.addParam(motor_id):
                    raise RuntimeError(f"Cannot add Dynamixel ID {motor_id}")
                if not self.reset_group.addParam(motor_id):
                    raise RuntimeError(f"Cannot add reset feedback Dynamixel ID {motor_id}")
            for motor_id in config.joint_ids:
                if not self.current_feedback.addParam(motor_id):
                    raise RuntimeError(f'Cannot add current feedback Dynamixel ID {motor_id}')
            self.read()
        except BaseException:
            if self.port.ser is not None:
                self.port.closePort()
            raise

    def detect_capabilities(self):
        """Return model-gated capabilities without changing servo state."""
        with self.lock:
            if self.closed:
                raise RuntimeError('GELLO port is closed')
            result = {}
            for motor_id in self.config.joint_ids:
                model = None
                try:
                    value = self.packet.ping(self.port, motor_id)
                    if isinstance(value, tuple):
                        # Dynamixel SDK returns (model_number, comm_result,
                        # packet_error), but accept wrappers that return just
                        # the model number.
                        if len(value) >= 3:
                            model, comm, error = value[:3]
                            check(comm, f'Dynamixel {motor_id} ping')
                            check(error, f'Dynamixel {motor_id} ping device error')
                        elif value:
                            model = value[0]
                    else:
                        model = value
                except Exception as exc:
                    result[motor_id] = {'model_number': None, 'model_name': None,
                                        'current_control': False, 'error': str(exc)}
                    continue
                model = int(model) if model is not None else None
                self.model_numbers[motor_id] = model
                result[motor_id] = {
                    'model_number': model,
                    'model_name': _KNOWN_CURRENT_MODELS.get(model),
                    'current_control': model in _KNOWN_CURRENT_MODELS,
                }
            return result

    def enable_current_damping(self, config):
        """Switch configured joints to model-supported current control.

        Gains and limits are in the servo's raw current units.  This method
        never describes them as Nm and refuses to proceed until every selected
        servo has a recognized model and the configured vectors match.
        """
        if str(getattr(config, 'damping_mode', 'none')).lower() != 'current':
            return {'enabled': False, 'reason': 'damping_mode_is_not_current'}
        capabilities = self.detect_capabilities()
        unsupported = [i for i, value in capabilities.items() if not value.get('current_control')]
        if unsupported:
            raise RuntimeError(f'GELLO current damping unsupported for motor IDs {unsupported}; '
                               'configure the exact Dynamixel model/control table first')
        n = len(self.config.joint_ids)
        gains = np.asarray(getattr(config, 'damping_gain', None), dtype=float)
        limits = np.asarray(getattr(config, 'damping_current_limit', None), dtype=int)
        if gains.shape != (n,) or limits.shape != (n,) or np.any(gains < 0) or np.any(limits <= 0):
            raise ValueError('damping_gain and damping_current_limit must contain one positive value per joint')
        watchdog_ms = int(getattr(config, 'damping_watchdog_ms', 100))
        if watchdog_ms <= 0 or watchdog_ms > 2540 or watchdog_ms % 20:
            raise ValueError('damping_watchdog_ms must be a positive multiple of 20 up to 2540')
        with self.lock:
            # EEPROM mode/limit writes can take longer than the watchdog.
            # Keep EVERY joint off throughout configuration; enable only once
            # all joints are ready, immediately before the streaming loop.
            for motor_id in self.config.joint_ids:
                self._write(motor_id, _CURRENT_TABLE['torque_enable'], 0)
                self._write(motor_id, _CURRENT_TABLE['watchdog'], 0)
            for motor_id, limit in zip(self.config.joint_ids, limits):
                self._write(motor_id, _CURRENT_TABLE['operating_mode'], 0)
                self._write(motor_id, _CURRENT_TABLE['current_limit'], int(limit), size=2)
                self._write(motor_id, _CURRENT_TABLE['goal_current'], 0, size=2)
            for motor_id in self.config.joint_ids:
                self._write(motor_id, _CURRENT_TABLE['watchdog'], watchdog_ms // 20)
                self._write(motor_id, _CURRENT_TABLE['torque_enable'], 1)
        return {'enabled': True, 'models': capabilities, 'unit': 'raw_current',
                'watchdog_ms': watchdog_ms}

    def write_current_damping(self, current):
        values = np.asarray(current, dtype=float).reshape(len(self.config.joint_ids))
        if not np.isfinite(values).all():
            raise ValueError('damping current contains non-finite values')
        with self.lock:
            if self.closed:
                raise RuntimeError('GELLO port is closed')
            words = [int(np.clip(np.rint(v), -32768, 32767)) & 0xffff for v in values]
            self.current_writer.clearParam()
            try:
                for motor_id, word in zip(self.config.joint_ids, words):
                    if not self.current_writer.addParam(motor_id, [word & 0xff, (word >> 8) & 0xff]):
                        raise RuntimeError(f'Cannot add damping current for Dynamixel {motor_id}')
                check(self.current_writer.txPacket(), 'GELLO sync current write')
                # Broadcast writes have no per-motor ACK. Verify every target
                # via checked sync read, preserving device/error detection.
                check(self.current_feedback.txRxPacket(), 'GELLO current readback')
                for motor_id, expected in zip(self.config.joint_ids, words):
                    if not self.current_feedback.isAvailable(motor_id, 102, 2):
                        raise RuntimeError(f'Missing current readback for Dynamixel {motor_id}')
                    actual = self.current_feedback.getData(motor_id, 102, 2)
                    if actual != expected:
                        raise RuntimeError(f'Dynamixel {motor_id} Goal Current write/read mismatch: '
                                           f'expected={expected}, actual={actual}')
            finally:
                self.current_writer.clearParam()

    def enable_alignment_hold(self, goal_current=80):
        """Current-limited position hold at the CURRENT pose; no homing motion.

        Goal Current stays at this raw value as the position goals advance.
        The model's EEPROM Current Limit remains a separate hardware bound.
        """
        if type(goal_current) is not int or goal_current <= 0:
            raise ValueError('alignment Goal Current must be a positive integer')
        capabilities = self.detect_capabilities()
        unsupported = [i for i, info in capabilities.items()
                       if info.get('model_number') not in _ALIGNMENT_MAX_CURRENT]
        if unsupported:
            raise ValueError(f'Alignment hold has no verified maximum current for IDs {unsupported}')
        if any(goal_current > _ALIGNMENT_MAX_CURRENT[info['model_number']] for info in capabilities.values()):
            raise ValueError('Alignment current exceeds model maximum')
        limits = {i: _ALIGNMENT_MAX_CURRENT[info['model_number']] for i, info in capabilities.items()}
        with self.lock:
            for motor_id in self.config.joint_ids:
                self._write(motor_id, 64, 0)
                self._write(motor_id, 98, 0)
            for motor_id in self.config.joint_ids:
                # Mode 5 uses Goal Current to bound the position loop output.
                self._write(motor_id, 11, 5)
                self._write(motor_id, 38, limits[motor_id], size=2)
                current = self._read_register(motor_id, 132, 4)
                if current > 0x7fffffff:
                    current -= 0x100000000
                lower = self._read_register(motor_id, 52, 4)
                upper = self._read_register(motor_id, 48, 4)
                if not lower <= current % 4096 <= upper:
                    raise ValueError(f'Dynamixel {motor_id} current pose outside position limits')
                pwm = self._read_register(motor_id, 36, 2)
                self._write(motor_id, 100, pwm, size=2)
                self._write(motor_id, 102, goal_current, size=2)
                self._write(motor_id, 116, current & 0xffffffff, size=4)
            for motor_id in self.config.joint_ids:
                self._write(motor_id, 64, 1)
        return {'mode': 'current_based_position', 'goal_current_raw': goal_current,
                'current_limits_raw': limits, 'models': capabilities}

    def alignment_reset_plan(self, reference, steps=15, max_travel_deg=90.):
        """Validate all interpolated goals before commanding any reset movement."""
        n = len(self.config.joint_ids)
        reference = np.asarray(reference, float).reshape(n)
        start = self.read()[:n]
        target = reference+np.rint((start-reference)/(2*np.pi))*(2*np.pi)
        if not np.isfinite(target).all() or np.max(np.abs(target-start)) > math.radians(max_travel_deg):
            raise ValueError('GELLO reset exceeds configured travel; manually approach the saved pose first')
        plan = [start+(target-start)*i/steps for i in range(1, steps+1)]
        with self.lock:
            for j, motor_id in enumerate(self.config.joint_ids):
                if self._read_register(motor_id, 11, 1) != 5:
                    raise ValueError('GELLO alignment reset requires current-based position mode (5)')
                lower = self._read_register(motor_id, 52, 4)
                upper = self._read_register(motor_id, 48, 4)
                for pose in plan:
                    ticks = int(round(pose[j]*2048/math.pi))
                    if not -1048575 <= ticks <= 1048575 or not lower <= ticks % 4096 <= upper:
                        raise ValueError(f'Dynamixel {motor_id} reset plan exceeds position limits')
        return plan

    def command_alignment_positions(self, pose):
        values = np.asarray(pose, float).reshape(len(self.config.joint_ids))
        if not np.isfinite(values).all():
            raise ValueError('GELLO reset target contains invalid values')
        ticks = np.rint(values*2048/math.pi).astype(np.int64)
        if np.any(np.abs(ticks) > 1048575):
            raise ValueError('GELLO reset target exceeds mode 5 range')
        with self.lock:
            for motor_id, value in zip(self.config.joint_ids, ticks):
                self._write(motor_id, 116, int(value) & 0xffffffff, size=4)

    @staticmethod
    def compute_damping_current(mapped, previous_mapped, dt, config, hold_reference=None, previous_velocity=None):
        """Compute bounded raw-current damping from mapped joint velocity."""
        mapped = np.asarray(mapped, dtype=float)
        previous = mapped if previous_mapped is None else np.asarray(previous_mapped, dtype=float)
        dt = max(float(dt), 1e-4)
        raw_dq = (mapped - previous) / dt
        alpha = float(np.clip(getattr(config, 'damping_velocity_filter_alpha', 1.0), 1e-6, 1.0))
        dq = raw_dq if previous_velocity is None else alpha * raw_dq + (1.0 - alpha) * np.asarray(previous_velocity, dtype=float)
        gain = np.asarray(config.damping_gain, dtype=float)
        brake = np.asarray(config.damping_brake_gain or [0.0] * mapped.size, dtype=float)
        limit = np.asarray(config.damping_current_limit, dtype=float)
        current = -gain * dq
        current += -brake * np.sign(dq) * np.maximum(np.abs(dq) - float(config.damping_velocity_threshold), 0.0)
        reference = mapped if hold_reference is None else np.asarray(hold_reference, dtype=float)
        if bool(config.weak_hold_enabled):
            weak = np.asarray(config.weak_hold_gain, dtype=float) * (reference - mapped)
            weak_limit = np.asarray(config.weak_hold_limit, dtype=float)
            current += np.clip(weak, -weak_limit, weak_limit)
        if current.shape != mapped.shape or limit.shape != mapped.shape:
            raise ValueError('damping vectors must match the mapped GELLO joints')
        # Velocity/hold error are in mapped robot coordinates; Goal Current is
        # in encoder coordinates. A flipped mapping must flip current as well.
        signs = np.asarray(getattr(config, 'joint_signs', np.ones(mapped.size)), dtype=float)
        if signs.shape != mapped.shape or not np.all(np.isin(signs, [-1., 1.])):
            raise ValueError('damping joint_signs must contain one +1/-1 per joint')
        return np.clip(current, -limit, limit) * signs

    def disable_current_damping(self):
        with self.lock:
            for motor_id in self.config.joint_ids:
                try:
                    self._write(motor_id, _CURRENT_TABLE['torque_enable'], 0)
                    self._write(motor_id, _CURRENT_TABLE['watchdog'], 0)
                    self._write(motor_id, _CURRENT_TABLE['goal_current'], 0, size=2)
                    self._write(motor_id, _CURRENT_TABLE['torque_enable'], 0)
                    # Leave the control table in position mode so a later
                    # alignment/hold operation cannot accidentally write a
                    # position goal while the servo is still in current mode.
                    self._write(motor_id, _CURRENT_TABLE['operating_mode'], 3)
                except Exception:
                    # Cleanup must still close the port after a disconnected
                    # servo; the original communication error is logged by the
                    # caller.
                    pass

    def read(self, reset_feedback=False):
        with self.lock:
            if self.closed:
                raise RuntimeError('GELLO port is closed')
            started = time.monotonic()
            group = self.reset_group if reset_feedback else self.group
            check(group.txRxPacket(), f"GELLO read {self.config.port}")
            values = []
            pwms, currents = [], []
            for motor_id in self.ids:
                if not group.isAvailable(motor_id, 132, 4):
                    raise RuntimeError(f"Missing Dynamixel ID {motor_id}")
                value = group.getData(motor_id, 132, 4)
                if value > 0x7fffffff:
                    value -= 0x100000000
                values.append(value * math.pi / 2048)
                if reset_feedback:
                    for address, output in ((124, pwms), (126, currents)):
                        if not group.isAvailable(motor_id, address, 2):
                            raise RuntimeError(f'Missing Dynamixel ID {motor_id} register {address}')
                        value = group.getData(motor_id, address, 2)
                        output.append(value - 65536 if value & 0x8000 else value)
            if time.monotonic() - started > self.config.read_timeout:
                raise TimeoutError('GELLO sample exceeded read_timeout')
            if reset_feedback:
                return np.asarray(values), np.asarray(pwms), np.asarray(currents)
            return np.asarray(values)

    def _write(self, motor_id, address, value, size=1):
        if size == 1:
            method = self.packet.write1ByteTxRx
        elif size == 2:
            method = self.packet.write2ByteTxRx
        else:
            method = self.packet.write4ByteTxRx
        comm, error = method(self.port, motor_id, address, value)
        check(comm, f"Dynamixel {motor_id} write {address}")
        if error:
            detail = ''
            if address == _CURRENT_TABLE['goal_current']:
                try:
                    watchdog = self._read_register(motor_id, _CURRENT_TABLE['watchdog'], 1)
                    detail = f'; watchdog={watchdog} (255 means expired)'
                except Exception:
                    pass
            raise RuntimeError(f'Dynamixel {motor_id} write register {address}, value={value}: '
                               f'device error code={error}{detail}')

    def _read_register(self, motor_id, address, size):
        method = {1: self.packet.read1ByteTxRx,
                  2: self.packet.read2ByteTxRx,
                  4: self.packet.read4ByteTxRx}[size]
        value, comm, error = method(self.port, motor_id, address)
        check(comm, f'Dynamixel {motor_id} read {address}')
        check(error, f'Dynamixel {motor_id} device error at {address}')
        return value

    def read_hold_status(self):
        """Read holding state through this reader's existing serial connection."""
        with self.lock:
            if self.closed:
                raise RuntimeError('GELLO port is closed')
            configured = getattr(self.config, 'hold_pwm_by_joint', None)
            if configured is None:
                configured = [getattr(self.config, 'hold_pwm', None)] * len(self.config.joint_ids)
            results = []
            for motor_id, requested in zip(self.config.joint_ids, configured):
                try:
                    read = lambda address, size: self._read_register(motor_id, address, size)
                    mode = read(11, 1)
                    pwm_limit = read(36, 2)
                    torque_on = read(64, 1)
                    hardware_error = read(70, 1)
                    goal_pwm = read(100, 2)
                    present_pwm = read(124, 2)
                    if present_pwm & 0x8000:
                        present_pwm -= 0x10000
                    goal_position = read(116, 4)
                    present_position = read(132, 4)
                    if goal_position & 0x80000000:
                        goal_position -= 0x100000000
                    if present_position & 0x80000000:
                        present_position -= 0x100000000
                    results.append({
                        'id': motor_id, 'configured': requested, 'mode': mode,
                        'pwm_limit': pwm_limit, 'torque_on': torque_on,
                        'hardware_error': hardware_error, 'goal_pwm': goal_pwm,
                        'present_pwm': present_pwm,
                        'position_error_ticks': goal_position - present_position,
                    })
                except RuntimeError as exc:
                    results.append({'id': motor_id, 'error': str(exc)})
            return results

    def _checked_position_goal(self, motor_id, raw, from_present=False):
        """Check a position-mode goal against this motor's configured limits."""
        mode = self._read_register(motor_id, 11, 1)
        if mode != 3:
            raise ValueError(f'Dynamixel {motor_id} requires position mode (3), got {mode}')
        # Present Position may contain whole-turn offsets while torque is off.
        # Position mode accepts one absolute turn as its Goal Position.
        goal = raw % 4096 if from_present else raw
        minimum = self._read_register(motor_id, 52, 4)
        maximum = self._read_register(motor_id, 48, 4)
        if not 0 <= minimum <= goal <= maximum <= 4095:
            raise ValueError(
                f'Dynamixel {motor_id} position goal {goal} ticks (present={raw}) '
                f'is outside configured Goal Position limits [{minimum}, {maximum}]; '
                'adjust the pose or inspect the motor limits before alignment'
            )
        return goal

    def _apply_hold_pwm(self, pwm_limit=None):
        values = None if pwm_limit is not None else getattr(self.config, 'hold_pwm_by_joint', None)
        if values is None:
            value = pwm_limit if pwm_limit is not None else getattr(self.config, 'hold_pwm', None)
            values = [value] * len(self.config.joint_ids) if value is not None else None
        if values is None:
            return
        if len(values) != len(self.config.joint_ids):
            raise ValueError('hold_pwm_by_joint must match joint_ids')
        for motor_id, pwm in zip(self.config.joint_ids, values):
            if pwm is not None and int(pwm) > 0:
                # X-series position mode uses Goal PWM as an OUTPUT CEILING,
                # never as a minimum holding effort. EEPROM PWM Limit (36)
                # may be lower than the control table's absolute maximum 885.
                limit, comm, error = self.packet.read2ByteTxRx(self.port, motor_id, 36)
                check(comm, f'Dynamixel {motor_id} read EEPROM PWM Limit')
                check(error, f'Dynamixel {motor_id} PWM Limit device error')
                if int(pwm) > min(limit, 885):
                    raise ValueError(
                        f'Dynamixel {motor_id} hold_pwm={pwm} exceeds EEPROM PWM Limit={limit}; '
                        'raising Goal PWM cannot increase holding effort beyond that limit'
                    )
                self._write(motor_id, 100, int(pwm), size=2)

    def set_torque(self, enabled, ids=None):
        with self.lock:
            selected = self.ids if ids is None else list(ids)
            for motor_id in selected:
                self._write(motor_id, 64, 1 if enabled else 0)

    def move(self, target, duration=3.0, tolerance=0.02):
        """Move mapped joints to a reference pose in position mode.

        This is deliberately a small, blocking reset helper.  It leaves torque
        enabled so the leader stays at the saved pose until the follow loop
        takes over.
        """
        target = np.asarray(target, dtype=float).reshape(len(self.config.joint_ids))
        if not np.isfinite(target).all():
            raise ValueError('Invalid GELLO target')
        with self.lock:
            goals = [self._checked_position_goal(
                motor_id, int(round(float(radians) * 2048.0 / math.pi)))
                for motor_id, radians in zip(self.config.joint_ids, target)
            ]
            for motor_id, goal in zip(self.config.joint_ids, goals):
                self._write(motor_id, 116, goal, size=4)
            self._apply_hold_pwm()
            for motor_id in self.config.joint_ids:
                self._write(motor_id, 64, 1)
        deadline = time.monotonic() + max(float(duration), 0.1)
        while time.monotonic() < deadline:
            actual = self.read()[:len(target)]
            if np.max(np.abs(actual - target)) <= tolerance:
                return
            time.sleep(0.05)
        actual = self.read()[:len(target)]
        errors = actual - target
        if np.max(np.abs(errors)) > tolerance:
            # A stationary joint can sag while another joint is being tested.
            # Report every failing ID, not just an anonymous maximum error.
            details = '; '.join(
                f'J{i + 1}/ID {motor_id}: target={target[i]:.4f} rad, '
                f'actual={actual[i]:.4f} rad, error={errors[i]:+.4f} rad '
                f'({math.degrees(errors[i]):+.2f} deg)'
                for i, motor_id in enumerate(self.config.joint_ids)
                if abs(errors[i]) > tolerance
            )
            raise TimeoutError(
                f'GELLO did not reach reference (error={np.max(np.abs(errors)):.3f} rad; '
                f'tolerance={math.degrees(tolerance):.2f} deg): {details}'
            )

    def _return_status(self, motor_id):
        """Best-effort snapshot of the failed motor before session torque-off.

        Stop at the first failed read so diagnostics do not repeatedly wait on
        an unplugged motor or replace the original motion failure.
        """
        values = []
        try:
            with self.lock:
                for name, address, size in (
                    ('mode', 11, 1), ('torque', 64, 1), ('hardware_error', 70, 1),
                    ('goal_pwm', 100, 2), ('present_pwm', 124, 2),
                    ('pwm_limit', 36, 2), ('voltage_raw', 144, 2), ('temperature_C', 146, 1),
                ):
                    value = self._read_register(motor_id, address, size)
                    if name in ('goal_pwm', 'present_pwm') and value & 0x8000:
                        value -= 0x10000
                    values.append(f'{name}={value:#04x}' if name == 'hardware_error' else f'{name}={value}')
        except Exception as exc:
            values.append(f'status_read_failed={exc}')
        return ', '.join(values)

    def return_to_reference(self, target, stop, progress=None):
        """Ramp single-turn goals from the current pose; do not wrap the path.

        Completion leaves position holding on. The session releases it only
        after the operator confirms teleoperation. Failures propagate to cleanup.
        """
        n = len(self.config.joint_ids)
        target = np.asarray(target, dtype=float).reshape(n)
        if not np.isfinite(target).all():
            raise ValueError('Invalid GELLO reference')
        if stop.is_set():
            raise RuntimeError('GELLO return cancelled')
        with self.lock:
            goals = np.array([self._checked_position_goal(
                motor_id, int(round(angle * 2048 / math.pi)))
                for motor_id, angle in zip(self.config.joint_ids, target)])
        self.set_torque(False, ids=self.config.joint_ids)
        raw = self.read()[:n]
        start_ticks = np.rint(raw * 2048 / math.pi).astype(int) % 4096
        start = start_ticks * math.pi / 2048
        target = goals * math.pi / 2048
        delta = target - start
        travel = np.rad2deg(np.abs(delta))
        if np.max(travel) > self.config.leader_reset_max_travel_deg:
            raise ValueError(
                f'GELLO 回位行程过大，各轴行程(度)={np.round(travel, 2)}；'
                '请托住主手手动靠近参考姿态后重试，不会绕编码器边界跨圈回位'
            )
        speed = math.radians(self.config.leader_reset_speed_deg)
        ramp_time = float(np.max(np.abs(delta))) / speed
        if ramp_time + 1.0 > self.config.leader_reset_timeout:
            raise ValueError('GELLO 回位超时配置不足以按指定速度完成行程')
        with self.lock:
            for motor_id, ticks in zip(self.config.joint_ids, start_ticks):
                self._checked_position_goal(motor_id, int(ticks))
            self._apply_hold_pwm()
            for motor_id, ticks in zip(self.config.joint_ids, start_ticks):
                self._write(motor_id, 116, int(ticks), size=4)
            if stop.is_set():
                raise RuntimeError('GELLO return cancelled')
            for motor_id in self.config.joint_ids:
                self._write(motor_id, 64, 1)
        last_time = time.monotonic()
        deadline = last_time + self.config.leader_reset_timeout
        elapsed = 0.0
        command = start.copy()
        settled = 0
        last_feedback = float('-inf')
        while True:
            if stop.is_set():
                raise RuntimeError('GELLO return cancelled')
            sample_time = time.monotonic() if progress is not None else None
            if progress is not None and sample_time - last_feedback >= 0.25:
                positions, pwms, currents = self.read(reset_feedback=True)
                actual = positions[:n]
                progress({'tracking': np.rad2deg(actual - command).tolist(),
                          'remaining': np.rad2deg(actual - target).tolist(),
                          'pwm': pwms[:n].tolist(), 'current': currents[:n].tolist()})
                last_feedback = sample_time
            else:
                actual = self.read()[:n]
            error = actual - target
            tracking = actual - command
            # Stop if a joint cannot track the ramp instead of driving farther
            # into a cable/spring obstruction. This is not a force sensor.
            if np.max(np.abs(tracking)) > max(self.config.alignment_tolerance * 3, math.radians(8)):
                i = int(np.argmax(np.abs(tracking)))
                status = self._return_status(self.config.joint_ids[i])
                raise RuntimeError(f'GELLO 回位跟踪偏差过大：J{i+1}/ID {self.config.joint_ids[i]} '
                                   f'偏差={math.degrees(tracking[i]):+.2f}°；'
                                   f'起点={math.degrees(start[i]):.2f}°，'
                                   f'当前指令={math.degrees(command[i]):.2f}°，'
                                   f'实际={math.degrees(actual[i]):.2f}°，'
                                   f'最终目标={math.degrees(target[i]):.2f}°；'
                                   f'{status}；检查线缆、弹簧和保持能力')
            settled = settled + 1 if elapsed >= ramp_time and np.max(np.abs(error)) <= self.config.alignment_tolerance else 0
            if settled >= 3:
                return
            now = time.monotonic()
            if now >= deadline:
                i = int(np.argmax(np.abs(error)))
                status = self._return_status(self.config.joint_ids[i])
                raise TimeoutError(f'GELLO 回位超时：J{i+1}/ID {self.config.joint_ids[i]} '
                                   f'目标={math.degrees(target[i]):.2f}°，实际={math.degrees(actual[i]):.2f}°；{status}')
            elapsed += min(max(now - last_time, 0.0), 0.1)
            last_time = now
            fraction = min(elapsed / ramp_time, 1.0) if ramp_time else 1.0
            command = start + fraction * delta
            with self.lock:
                for motor_id, angle in zip(self.config.joint_ids, command):
                    self._write(motor_id, 116, int(round(angle * 2048 / math.pi)), size=4)
            stop.wait(0.05)

    def hold(self, pwm_limit=None):
        """Hold the current mapped positions and optionally apply Goal PWM."""
        with self.lock:
            goals = [self._checked_position_goal(
                motor_id, self._read_register(motor_id, 132, 4), from_present=True
            ) for motor_id in self.config.joint_ids]
            for motor_id, goal in zip(self.config.joint_ids, goals):
                self._write(motor_id, 116, goal, size=4)
            self._apply_hold_pwm(pwm_limit)
            for motor_id in self.config.joint_ids:
                self._write(motor_id, 64, 1)

    def prepare_alignment(self):
        # Release mapped joints for manual alignment; fixed joints hold their CURRENT pose.
        with self.lock:
            for motor_id in self.ids:
                self._write(motor_id, 64, 0)
            for motor_id in self.config.torque_joint_ids or ():
                current = self._read_register(motor_id, 132, 4)
                goal = self._checked_position_goal(motor_id, current, from_present=True)
                self._write(motor_id, 116, goal, size=4)
                self.held_ids.append(motor_id)
                self._write(motor_id, 64, 1)

    def close(self):
        with self.lock:
            if self.closed:
                return
            try:
                for motor_id in dict.fromkeys(self.ids):
                    try:
                        self._write(motor_id, 64, 0)
                    except Exception:
                        # Always close the descriptor, including after unplugging USB.
                        pass
            finally:
                self.closed = True
                self.port.closePort()


class Arm:
    def __init__(self, config):
        from ufactory_devices.robot import UFRobot

        self.config = config
        self.robot = UFRobot(config, initialize=False)
        self.api = self.robot.real_arm
        self.api.set_timeout(config.command_timeout)
        self.stopping = threading.Event()
        self.command_lock = threading.Lock()

    def _command(self, fn, *args, **kwargs):
        # Serialize stop with writes. A worker returning from a slow read cannot restart an arm.
        with self.command_lock:
            if self.stopping.is_set():
                raise RuntimeError('Arm is stopping')
            check(fn(*args, **kwargs), fn.__name__)

    def health(self, active=False):
        if not self.api.connected:
            raise ConnectionError(f"Robot disconnected: {self.config.robot_ip}")
        code, error = self.api.get_err_warn_code()
        check(code, 'Read robot errors')
        if error[0]:
            raise RuntimeError(f"Robot {self.config.robot_ip} error: {error[0]}")
        code, state = self.api.get_state()
        check(code, 'Read robot state')
        if active and state not in (0, 1, 2):
            raise RuntimeError(f"Robot {self.config.robot_ip} stopped/paused (state={state})")
        return state

    def joints(self):
        code, values = self.api.get_servo_angle(is_radian=True)
        check(code, 'Read robot joints')
        values = np.asarray(values[:self.api.axis], dtype=float)
        if values.shape != (self.api.axis,) or not np.isfinite(values).all():
            raise ValueError('Invalid robot joint feedback')
        return values

    def check_target(self, q):
        if len(q) != self.api.axis or not np.isfinite(q).all():
            raise ValueError('Target dimension/values do not match robot axes')
        code, limited = self.api.is_joint_limit(q.tolist(), is_radian=True)
        check(code, 'Check robot joint limits')
        if limited:
            raise ValueError(f"Joint target exceeds controller limits: {q}")

    def prepare_reset(self):
        self.health()
        self.check_target(np.asarray(self.config.reset_q))
        if self.config.collision_sensitivity is not None:
            self._command(self.api.set_collision_sensitivity, self.config.collision_sensitivity)
        self._command(self.api.set_state, 4)  # clear any previous queued motion before enabling
        self._command(self.api.motion_enable, True)
        self._command(self.api.set_mode, 0)
        self._command(self.api.set_state, 0)

    def reset(self, stop, barrier):
        self.prepare_reset()
        barrier.wait(timeout=self.config.reset_timeout)
        if stop.is_set():
            return
        self._command(self.api.set_servo_angle, angle=list(self.config.reset_q),
                      speed=math.radians(self.config.reset_speed),
                      mvacc=math.radians(self.config.reset_acc), is_radian=True, wait=False)
        deadline = time.monotonic() + self.config.reset_timeout
        settled = 0
        while not stop.is_set():
            state = self.health(active=True)
            error = np.max(np.abs(self.joints() - self.config.reset_q))
            settled = settled + 1 if state != 1 and error <= self.config.reset_tolerance else 0
            if settled >= 3:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Reset timed out: {self.config.robot_ip}")
            stop.wait(0.05)
        raise RuntimeError('Reset cancelled')

    def prepare_follow(self):
        self.health(active=True)
        # Initialize grippers only after the reset/alignment confirmations.
        with self.command_lock:
            if self.stopping.is_set():
                raise RuntimeError('Arm is stopping')
            self.robot.robot_init(enable=False, init_gripper_pose=True)
            self.robot.online_joint_control = True
        self.health(active=True)

    def send(self, action):
        self._command(self.robot.send_action, action)

    def stop(self):
        self.stopping.set()
        with self.command_lock:
            check(self.api.set_state(4), 'Stop robot')

    def close(self):
        self.api.disconnect()
