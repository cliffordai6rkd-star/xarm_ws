"""Safe public-SDK adapter shared by xArm collection and inference paths.

``connect`` is read-only: it opens the SDK connection and does not enable
motion, clear faults, change modes, move to a rest pose, initialize a gripper,
or change collision settings.  Motion-changing operations require the explicit
``execution_enabled`` endpoint option.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nero_collection.arms.base import ArmState, GripperState
from nero_collection.arms.kinematics import pose6_to_matrix
from nero_collection.config import ArmEndpointConfig
from nero_collection.time_utils import now_us

log = logging.getLogger(__name__)


def validate_collision_sensitivity(value) -> int | None:
    """None retains the controller setting; 0 disables collision detection."""
    if value is None:
        return None
    try:
        level = float(value)
    except (TypeError, ValueError):
        raise ValueError('collision_sensitivity must be an integer from 0 to 5, or null') from None
    if isinstance(value, bool) or not np.isfinite(level) or level != int(level) or not 0 <= level <= 5:
        raise ValueError('collision_sensitivity must be an integer from 0 to 5, or null')
    return int(level)


class XArmControllerFault(RuntimeError):
    """A controller protection/fault, distinct from a generic SDK return code."""

    def __init__(self, arm_name, error_code, *, operation=None, api_code=None):
        self.arm_name, self.error_code = arm_name, int(error_code)
        self.operation, self.api_code = operation, api_code
        detail = '碰撞导致电流异常' if self.error_code == 31 else '控制器故障'
        prefix = f'xArm {arm_name}' + (f' {operation}' if operation else '')
        sdk = f'; SDK API code {api_code}' if api_code is not None else ''
        super().__init__(f'{prefix} reports controller error code {self.error_code} '
                         f'(C{self.error_code}: {detail}){sdk}; 手动处理故障后再使能并重新对齐接管')


@dataclass
class XArmAdapter:
    """Common xArm adapter for xArm5, xArm6 and xArm7.

    The public xArm SDK exposes position servoing, but no verified Nero/MIT
    joint torque command.  Impedance requests are rejected unless the caller
    explicitly enables the documented position fallback.
    """

    config: ArmEndpointConfig
    name: str = field(init=False)
    dof: int = field(init=False, default=7)
    _arm: Any = field(init=False, default=None)
    _configured_role: str | None = field(init=False, default=None)
    _last_q: np.ndarray | None = field(init=False, default=None)
    _last_dq: np.ndarray | None = field(init=False, default=None)
    _last_t: float | None = field(init=False, default=None)
    _feedback_available: bool = field(init=False, default=False)
    _gripper: Any = field(init=False, default=None)
    _enabled: bool = field(init=False, default=False)
    _last_commanded_q: np.ndarray | None = field(init=False, default=None)
    _last_command_timestamp_us: int | None = field(init=False, default=None)
    _torque_valid: bool = field(init=False, default=False)
    _current_valid: bool = field(init=False, default=False)
    _velocity_valid: bool = field(init=False, default=False)
    _last_read_wall: float | None = field(init=False, default=None)
    _first_read_wall: float | None = field(init=False, default=None)
    _last_feedback_timestamp_us: int | None = field(init=False, default=None)
    _feedback_updates: int = field(init=False, default=0)
    _repeated_feedback: int = field(init=False, default=0)
    _command_intervals_s: list[float] = field(init=False, default_factory=list)
    _io_lock: threading.RLock = field(init=False, default_factory=threading.RLock)

    def __post_init__(self) -> None:
        self.name = self.config.name
        configured = self.config.config_kwargs.get("dof")
        if configured is None and self.config.rest_q:
            configured = len(self.config.rest_q)
        self.dof = int(configured or 7)
        if self.dof not in {5, 6, 7}:
            raise ValueError("xArm dof must be one of 5, 6, or 7")
        if self.config.rest_q and len(self.config.rest_q) != self.dof:
            raise ValueError(f"rest_q has {len(self.config.rest_q)} values but dof={self.dof}")
        self._configured_gripper_speed()
        validate_collision_sensitivity(self.config.config_kwargs.get('collision_sensitivity'))

    @property
    def feedback_available(self) -> bool:
        return self._feedback_available

    @property
    def reported_tcp_payload(self) -> dict | None:
        """Read the SDK rich-report cache, without sending a hardware command."""
        if self._arm is None:
            return None
        load = getattr(self._arm, 'tcp_load', None)
        try:
            mass, com_mm = load
            mass, com = float(mass), np.asarray(com_mm, dtype=float)
            if not np.isfinite(mass) or mass <= 0 or com.shape != (3,) or not np.isfinite(com).all():
                return None
        except (TypeError, ValueError):
            return None
        return {'mass_kg': mass, 'com_m': (com/1000.).tolist()}

    @property
    def capabilities(self) -> dict[str, bool | str]:
        signal = str(self.config.config_kwargs.get("feedback_signal", "none")).lower()
        return {
            "position_servo": True,
            "joint_torque_command": False,
            "mit_impedance": False,
            "feedback_signal": signal,
            "execution_enabled": bool(self.config.config_kwargs.get("execution_enabled", False)),
        }

    def configure_feedback_report(self) -> None:
        """Select the SDK report field explicitly when firmware supports it.

        xArm firmware names this field ``tau_or_i`` because it is either
        torque or motor current.  We never infer one from the other; the
        configured ``feedback_signal`` determines the selector and the state
        validity flags record what was actually available.
        """
        self._require_arm()
        signal = str(self.config.config_kwargs.get("feedback_signal", "none")).lower()
        if signal == "none":
            return
        selector = 0 if signal == "torque" else 1
        self._check_result(
            self._call("set_report_tau_or_i", selector, required=True),
            "set_report_tau_or_i",
        )

    @property
    def last_commanded_q(self) -> np.ndarray | None:
        return None if self._last_commanded_q is None else self._last_commanded_q.copy()

    @property
    def timing_stats(self) -> dict[str, float | int]:
        intervals = np.asarray(self._command_intervals_s, dtype=np.float64)
        return {
            "feedback_updates": self._feedback_updates,
            "repeated_feedback": self._repeated_feedback,
            "feedback_update_hz": float(self._feedback_updates / max(time.monotonic() - (self._first_read_wall or time.monotonic()), 1e-6)),
            "command_count": len(self._command_intervals_s) + (1 if self._last_commanded_q is not None else 0),
            "command_interval_mean_s": float(np.mean(intervals)) if intervals.size else 0.0,
            "command_interval_max_s": float(np.max(intervals)) if intervals.size else 0.0,
        }

    def connect(self) -> None:
        """Open a read-only SDK connection with no motion side effects."""
        if self._arm is not None:
            return
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise RuntimeError(
                "xArm SDK is not installed; install xArm-Python-SDK before using backend=xarm"
            ) from exc
        kwargs = dict(self.config.config_kwargs)
        robot_ip = kwargs.pop("robot_ip", kwargs.pop("ip", None))
        if not robot_ip:
            raise ValueError(f"xArm endpoint {self.name} requires robot_ip")
        sdk_kwargs = {
            key: kwargs[key]
            for key in ("is_radian", "report_type", "do_not_open", "init_gripper")
            if key in kwargs
        }
        self._arm = XArmAPI(robot_ip, **sdk_kwargs)
        time.sleep(float(kwargs.get("connect_settle_s", 0.2)))
        if not bool(getattr(self._arm, "connected", True)):
            self._arm = None
            raise ConnectionError(f"failed to connect xArm {self.name} at {robot_ip}")
        timeout_s = float(kwargs.get("sdk_timeout_s", kwargs.get("command_timeout_s", 0.1)))
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("sdk_timeout_s must be positive and finite")
        set_timeout = getattr(self._arm, "set_timeout", None)
        if callable(set_timeout):
            self._check_result(set_timeout(timeout_s), "set_timeout")
        self._last_q = self._last_dq = None
        self._last_t = None
        self._feedback_available = False
        self._enabled = False
        self._gripper = None
        self._last_commanded_q = None
        self._last_command_timestamp_us = None
        self._last_read_wall = None
        self._first_read_wall = None
        self._last_feedback_timestamp_us = None
        self._feedback_updates = self._repeated_feedback = 0
        self._command_intervals_s.clear()

    def disconnect(self) -> None:
        if self._arm is None:
            return
        arm, self._arm = self._arm, None
        self._enabled = False
        self._gripper = None
        for method in ("disconnect", "close"):
            fn = getattr(arm, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    log.debug("xArm %s %s failed during disconnect", self.name, method, exc_info=True)
                break

    def enable(self) -> None:
        self._require_arm()
        self._require_execution("enable")
        self._raise_if_faulted()
        sensitivity = validate_collision_sensitivity(self.config.config_kwargs.get('collision_sensitivity'))
        if sensitivity is not None:
            # Apply while stopped; changing this setting can put firmware in
            # state 5, so the normal mode/state setup below must follow it.
            self._check_result(self._call('set_collision_sensitivity', sensitivity, wait=False, required=True),
                               'set_collision_sensitivity')
            if sensitivity == 0:
                log.warning('xArm %s collision_sensitivity=0: 控制器碰撞检测已关闭', self.name)
            else:
                log.info('xArm %s collision_sensitivity=%d (1 least sensitive, 5 most sensitive)',
                         self.name, sensitivity)
        self._check_result(self._call("motion_enable", True, required=True), "motion_enable")
        self._enabled = True
        if str(self.config.config_kwargs.get("feedback_signal", "none")).lower() in {"torque", "current"}:
            self.configure_feedback_report()
        self._check_result(self._call("set_mode", self._servo_mode(), required=True), "set_mode")
        self._check_result(self._call("set_state", 0, required=True), "set_state")
        self._wait_reported_mode(self._servo_mode())

    def _wait_reported_mode(self, mode: int) -> None:
        """Wait for SDK mode cache before issuing mode-dependent motion."""
        arm = self._require_arm()
        if not hasattr(arm, 'mode'):
            return  # small offline SDK mocks do not publish mode reports
        timeout = float(self.config.config_kwargs.get('mode_switch_timeout_s', 1.))
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('mode_switch_timeout_s must be positive and finite')
        deadline = time.monotonic()+timeout
        while int(arm.mode) != mode:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'xArm {self.name} mode switch not reported: target={mode}, cached={arm.mode}')
            self._raise_if_faulted()
            time.sleep(.01)

    def disable(self) -> None:
        self._require_arm()
        self._require_execution("disable")
        self._check_result(self._call("motion_enable", False, required=True), "motion_enable")
        self._check_result(self._call("set_state", 4, required=False), "set_state")
        self._enabled = False

    def set_leader_mode(self) -> None:
        self._configured_role = "leader"

    def set_follower_mode(self) -> None:
        self._configured_role = "follower"
        self.set_normal_mode()

    def set_normal_mode(self) -> None:
        self._require_arm()
        self._require_execution("set_normal_mode")
        self._raise_if_faulted()
        self._check_result(self._call("set_mode", self._servo_mode(), required=True), "set_mode")
        self._check_result(self._call("set_state", 0, required=True), "set_state")
        self._wait_reported_mode(self._servo_mode())

    def read_control_role(self, refresh: bool = False) -> str | None:
        del refresh
        return self._configured_role

    def read_state(self) -> ArmState:
        arm = self._require_arm()
        acquired_us = now_us()
        read_wall = time.monotonic()
        if self._first_read_wall is None:
            self._first_read_wall = read_wall
        self._last_read_wall = read_wall
        q, q_timestamp_us = self._read_q(arm)
        timestamp_us = q_timestamp_us or acquired_us
        dt = None if self._last_t is None else max(time.monotonic() - self._last_t, 1.0e-6)
        dq, torque, current, measured = self._read_feedback(arm, q, dt)
        self._feedback_available = measured
        self._feedback_updates += 1
        if self._last_feedback_timestamp_us == timestamp_us:
            self._repeated_feedback += 1
        self._last_feedback_timestamp_us = timestamp_us
        ee_pose = self._read_ee_pose(arm)
        ddq = np.zeros(self.dof, dtype=np.float64)
        if self._last_dq is not None and dt is not None:
            ddq = (dq - self._last_dq) / dt
        self._last_q, self._last_dq = q.copy(), dq.copy()
        self._last_t = time.monotonic()
        stamps = np.full(self.dof, timestamp_us, dtype=np.int64)
        return ArmState(
            q=q, dq=dq, ddq=ddq, ee_pose=ee_pose, torque=torque, current=current,
            timestamp_us=int(timestamp_us), acquired_timestamp_us=int(acquired_us),
            q_timestamp_us=int(timestamp_us), q_acquired_timestamp_us=int(acquired_us),
            q_component_timestamp_us=stamps.copy(), q_source_before_timestamp_us=stamps.copy(),
            q_source_after_timestamp_us=stamps.copy(), motor_timestamp_us=stamps.copy(),
            motor_acquired_timestamp_us=stamps.copy(), q_valid=True,
            dq_valid=self._velocity_valid, torque_valid=self._torque_valid,
            current_valid=self._current_valid, feedback_source=self._feedback_source(),
            timestamp_source="host_receive",
        )

    def read_leader_joint_positions(self) -> np.ndarray:
        return self.read_state().q

    def command_joint_positions(self, q: np.ndarray) -> None:
        arm = self._require_arm()
        self._require_execution("command_joint_positions")
        if not self._enabled:
            raise RuntimeError(f"xArm {self.name} must be enabled before position commands")
        q = self._validate_vector(q, "q")
        speed = float(self.config.config_kwargs.get("joint_speed_rad_s", 1.0))
        acceleration = float(self.config.config_kwargs.get("joint_acc_rad_s2", 5.0))
        api = str(self.config.config_kwargs.get("servo_api", "set_servo_angle_j"))
        if api == "set_servo_angle_j" and callable(getattr(arm, "set_servo_angle_j", None)):
            # xArm's Mode 1 API names this argument ``angles``.  The SDK
            # documents speed/mvacc as reserved for this endpoint, so the
            # follower's own q/dq/acceleration limits are the effective
            # safety limits for this path.
            result = self._call("set_servo_angle_j", angles=q.tolist(), is_radian=True, wait=False, required=True)
        elif api in {"set_servo_angle_j", "set_servo_angle"}:
            result = self._call("set_servo_angle", angle=q.tolist(), speed=speed, mvacc=acceleration, is_radian=True, wait=False, required=True)
        else:
            raise ValueError("servo_api must be set_servo_angle_j or set_servo_angle")
        self._check_motion_result(result, api)
        self._last_commanded_q = q.copy()
        if self._last_command_timestamp_us is not None:
            self._command_intervals_s.append(max(0.0, (now_us() - self._last_command_timestamp_us) * 1e-6))
        self._last_command_timestamp_us = now_us()

    def command_joint_impedance(self, q: np.ndarray, v_des: np.ndarray, kp: np.ndarray, kd: np.ndarray, t_ff: np.ndarray) -> None:
        del v_des, kp, kd, t_ff
        if not bool(self.config.config_kwargs.get("allow_impedance_position_fallback", False)):
            raise RuntimeError("xArm has no verified joint MIT/torque command; position fallback must be explicitly enabled")
        self.command_joint_positions(q)

    def validate_joint_impedance_support(self) -> None:
        if not bool(self.config.config_kwargs.get("allow_impedance_position_fallback", False)):
            raise RuntimeError("verified xArm impedance control is unavailable; use control_mode=position or explicitly enable the position fallback")

    def configure_joint_impedance_mode(self) -> None:
        self.validate_joint_impedance_support()

    def move_joints(self, q: np.ndarray) -> None:
        self._require_arm()
        self._require_execution("move_joints")
        if not self._enabled:
            raise RuntimeError(f"xArm {self.name} must be enabled before motion")
        q = self._validate_vector(q, "q")
        result = self._call("set_servo_angle", angle=q.tolist(), is_radian=True, wait=True, required=True)
        self._check_result(result, "set_servo_angle")
        self._last_commanded_q, self._last_command_timestamp_us = q.copy(), now_us()

    def move_to_reset(self, q: np.ndarray, *, speed: float, acceleration: float) -> None:
        """Low-speed reset motion used by the GELLO takeover state machine."""
        self._require_arm()
        self._require_execution("move_to_reset")
        if not self._enabled:
            raise RuntimeError(f"xArm {self.name} must be enabled before reset motion")
        q = self._validate_vector(q, "q")
        self._raise_if_faulted()
        # set_servo_angle is a planned move (mode 0), unlike the mode 1
        # set_servo_angle_j stream used after takeover. Stay in mode 0 while
        # the user aligns the leaders; takeover explicitly restores servo mode.
        self._check_result(self._call('set_mode', 0, required=True), 'reset set_mode(0)')
        self._check_result(self._call('set_state', 0, required=True), 'reset set_state(0)')
        self._wait_reported_mode(0)
        result = self._call(
            "set_servo_angle",
            angle=q.tolist(), speed=float(speed), mvacc=float(acceleration),
            is_radian=True, wait=False, required=True,
        )
        self._check_result(result, "move_to_reset")
        self._last_commanded_q = q.copy()
        self._last_command_timestamp_us = now_us()

    def control_status(self) -> dict:
        """Read controller state/error for reset progress; never clears faults."""
        arm = self._require_arm()
        result = self._call('get_state', required=False)
        self._check_result(result, 'get_state')
        state = result[1] if isinstance(result, tuple) and len(result) > 1 else getattr(arm, 'state', None)
        faults = self._call('get_err_warn_code', required=False)
        self._check_result(faults, 'get_err_warn_code')
        errors = faults[1] if isinstance(faults, tuple) and len(faults) > 1 else [0, 0]
        return {'mode': getattr(arm, 'mode', None), 'state': state,
                'error_code': int(_first_numeric(errors, 0.))}

    def hold_position(self, *, speed: float = 0.2, acceleration: float = 0.5) -> np.ndarray:
        """Cancel pending motion, then hold actual q in controller position mode.

        Motor enable is retained. A fault is never cleared or overridden; if
        reading q or restarting position mode fails, the stop request remains.
        """
        arm = self._require_arm()
        self._require_execution('hold_position')
        if not self._enabled:
            raise RuntimeError(f'xArm {self.name} must be enabled before position hold')
        self._check_motion_result(self._call('set_state', 4, required=True), 'hold set_state(4)')
        self._raise_if_faulted()
        q = self._validate_vector(self._read_q(arm)[0], 'hold q')
        self.move_to_reset(q, speed=speed, acceleration=acceleration)
        return q.copy()

    def wait_motion_done(self, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
        self._require_arm()
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            moving = self._call("get_is_moving", required=False)
            if moving is None:
                return True
            if isinstance(moving, tuple):
                moving = moving[-1]
            if not bool(moving):
                return True
            time.sleep(poll_interval_s)
        return False

    def init_gripper(self, effector: str = "xArmGripper") -> None:
        del effector
        self._require_execution("init_gripper")
        speed = self._configured_gripper_speed()
        self._gripper = self._require_arm()
        self._check_result(self._call("set_gripper_enable", True, required=True), "set_gripper_enable")
        self._check_result(self._call("set_gripper_mode", 0, required=False), "set_gripper_mode")
        if speed is not None:
            self._check_result(self._call("set_gripper_speed", speed, required=True), "set_gripper_speed")
            log.info("xArm %s G1 gripper speed set to %d r/min", self.name, speed)

    def reset_gripper(self) -> bool:
        if self._gripper is None:
            self.init_gripper()
        close_position = int(round(float(self.config.config_kwargs.get("gripper_close_position", 0))))
        self._check_result(self._call("set_gripper_position", close_position, wait=True, required=True), "set_gripper_position")
        return True

    def read_gripper_state(self) -> GripperState:
        if self._gripper is None:
            return GripperState(np.nan, np.nan, now_us(), "unavailable")
        result = self._call("get_gripper_position", required=False)
        self._check_result(result, "get_gripper_position")
        raw = _first_numeric(result, np.nan)
        return GripperState(float(self._raw_to_width(raw)), np.nan, now_us(), "width")

    def read_leader_gripper_state(self) -> GripperState:
        return self.read_gripper_state()

    def disable_gripper(self) -> None:
        self._require_execution("disable_gripper")
        self._call("set_gripper_enable", False, required=False)

    def command_gripper(self, value: float, force_n: float, mode: str = "width") -> None:
        del force_n
        if self._gripper is None:
            self.init_gripper()
        raw = float(value) if mode in {"raw", "raw_position"} else self._width_to_raw(float(value))
        self._check_result(self._call("set_gripper_position", int(round(raw)), wait=False, required=True), "set_gripper_position")

    def _read_q(self, arm: Any) -> tuple[np.ndarray, int]:
        result = self._call("get_servo_angle", is_radian=True, required=True)
        values = _extract_vector(result, self.dof)
        if values is None:
            raise RuntimeError(f"xArm {self.name} returned an invalid joint-angle report: {result!r}")
        return values, now_us()

    def _read_feedback(self, arm: Any, q: np.ndarray, dt: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        del q
        signal = str(self.config.config_kwargs.get("feedback_signal", "none")).lower()
        report = self._call("get_joint_states", is_radian=True, num=2 if signal == 'none' else 3, required=False)
        # Public xArm SDK format is (code, [position, velocity, effort]).
        # ``effort`` is selected by set_report_tau_or_i(): it is either a
        # torque report or a motor-current report, never both at once.
        velocity = _extract_joint_state_component(report, 1, self.dof)
        if signal not in {"none", "torque", "current"}:
            raise ValueError("feedback_signal must be none, torque, or current")
        effort = _extract_joint_state_component(report, 2, self.dof)
        torque = effort if signal == "torque" else None
        current = effort if signal == "current" else None
        self._velocity_valid, self._torque_valid, self._current_valid = velocity is not None, torque is not None, current is not None
        if velocity is None:
            velocity = np.zeros(self.dof, dtype=np.float64)
            source = str(self.config.config_kwargs.get("velocity_source", "hardware")).lower()
            if source == "position_difference" and dt is not None and self._last_q is not None:
                velocity = (self._read_q(arm)[0] - self._last_q) / dt
                self._velocity_valid = True
            elif source not in {"hardware", "position_difference"}:
                raise ValueError("velocity_source must be hardware or position_difference")
        if torque is None:
            torque = np.full(self.dof, np.nan, dtype=np.float64)
        if current is None:
            current = np.full(self.dof, np.nan, dtype=np.float64)
        return velocity, torque, current, self._torque_valid or self._current_valid

    def _read_ee_pose(self, arm: Any) -> np.ndarray:
        result = self._call("get_position_aa", is_radian=True, required=False)
        if result is None:
            result = self._call("get_position", is_radian=True, required=True)
        pose = _extract_vector(result, 6)
        if pose is None:
            raise RuntimeError(f"xArm {self.name} returned an invalid end-effector report: {result!r}")
        pose = pose.astype(np.float64, copy=True)
        pose[:3] *= 1.0e-3
        return pose6_to_matrix(pose)

    def _call(self, name: str, *args: Any, required: bool = False, **kwargs: Any) -> Any:
        with self._io_lock:
            arm = self._require_arm()
            fn = getattr(arm, name, None)
            if not callable(fn):
                if required:
                    raise RuntimeError(f"installed xArm SDK does not expose {name}()")
                return None
            return fn(*args, **kwargs)

    def _require_arm(self) -> Any:
        if self._arm is None:
            raise RuntimeError(f"xArm {self.name} is not connected")
        return self._arm

    def _require_execution(self, operation: str) -> None:
        if not bool(self.config.config_kwargs.get("execution_enabled", False)):
            raise RuntimeError(f"xArm execution is disabled; set execution_enabled=true explicitly before {operation}")

    def _raise_if_faulted(self, *, operation=None, api_code=None) -> None:
        result = self._call("get_err_warn_code", required=False)
        if result is None:
            return
        self._check_result(result, 'get_err_warn_code')
        value = result[1] if isinstance(result, tuple) and len(result) >= 2 else result
        error = _first_numeric(value, 0.0)
        if int(error) != 0:
            raise XArmControllerFault(self.name, int(error), operation=operation, api_code=api_code)

    def _check_motion_result(self, result, operation):
        try:
            self._check_result(result, operation)
        except RuntimeError as exc:
            # No extra SDK requests during successful 100 Hz servoing. Query
            # the actual controller cause only when a command was rejected.
            try:
                self._raise_if_faulted(operation=operation, api_code=self._result_code(result))
            except XArmControllerFault as fault:
                raise fault from exc
            except Exception:
                log.debug('Controller fault details unavailable for %s', self.name, exc_info=True)
            raise RuntimeError(f'{self.name}: {exc}') from exc

    def _servo_mode(self) -> int:
        return int(self.config.config_kwargs.get("servo_mode", 1))

    def _feedback_source(self) -> str:
        signal = str(self.config.config_kwargs.get("feedback_signal", "none")).lower()
        velocity = str(self.config.config_kwargs.get("velocity_source", "hardware")).lower()
        if signal != "none":
            return signal
        return velocity if velocity == "position_difference" else "unavailable"

    def _configured_gripper_speed(self) -> int | None:
        """G1 SDK motor speed in r/min; absent/null/-1 retains controller settings."""
        value = self.config.config_kwargs.get("gripper_speed")
        if value is None:
            return None
        try:
            speed = float(value)
        except (TypeError, ValueError):
            raise ValueError("gripper_speed must be a positive integer in r/min, or -1") from None
        if speed == -1:
            return None
        if isinstance(value, bool) or not np.isfinite(speed) or speed <= 0 or speed != int(speed):
            raise ValueError("gripper_speed must be a positive integer in r/min, or -1")
        return int(speed)

    def _width_to_raw(self, width_m: float) -> float:
        kwargs = self.config.config_kwargs
        minimum = float(kwargs.get("gripper_min_width_m", 0.0))
        maximum = float(kwargs.get("gripper_max_width_m", 0.085))
        raw_close = float(kwargs.get("gripper_close_position", 0.0))
        raw_open = float(kwargs.get("gripper_open_position", 800.0))
        if not np.isfinite(width_m) or maximum <= minimum:
            raise ValueError("gripper width calibration is invalid")
        normalized = np.clip((width_m - minimum) / (maximum - minimum), 0.0, 1.0)
        return raw_close + normalized * (raw_open - raw_close)

    def _raw_to_width(self, raw: float) -> float:
        kwargs = self.config.config_kwargs
        minimum = float(kwargs.get("gripper_min_width_m", 0.0))
        maximum = float(kwargs.get("gripper_max_width_m", 0.085))
        raw_close = float(kwargs.get("gripper_close_position", 0.0))
        raw_open = float(kwargs.get("gripper_open_position", 800.0))
        if not np.isfinite(raw) or raw_open == raw_close:
            return float("nan")
        normalized = np.clip((raw - raw_close) / (raw_open - raw_close), 0.0, 1.0)
        return minimum + normalized * (maximum - minimum)

    def _validate_vector(self, value: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
        if result.shape != (self.dof,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite {self.dof}-axis vector")
        return result

    @staticmethod
    def _result_code(result: Any) -> int:
        if isinstance(result, (int, np.integer)):
            code = int(result)
        elif isinstance(result, tuple) and result and isinstance(result[0], (int, np.integer)):
            code = int(result[0])
        elif isinstance(result, dict) and "code" in result:
            code = int(result["code"])
        else:
            code = 0
        return code

    @staticmethod
    def _check_result(result: Any, operation: str) -> None:
        code = XArmAdapter._result_code(result)
        if code != 0:
            raise RuntimeError(f"xArm {operation} failed with code {code}")


def _extract_vector(value: Any, length: int) -> np.ndarray | None:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (int, np.integer)):
        value = value[1]
    if isinstance(value, dict):
        for key in ("angles", "joint_angles", "position", "positions", "data", "value"):
            if key in value:
                candidate = _extract_vector(value[key], length)
                if candidate is not None:
                    return candidate
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if array.size == length and np.isfinite(array).all() else None


def _extract_named_or_indexed(value: Any, names: tuple[str, ...], index: int, length: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return _extract_vector(value[name], length)
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            result = _extract_vector(candidate, length)
            if result is not None:
                return result
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (int, np.integer)):
        value = value[1]
    if isinstance(value, (list, tuple)) and len(value) > index:
        return _extract_vector(value[index], length)
    return None


def _extract_joint_state_component(value: Any, component: int, length: int) -> np.ndarray | None:
    """Extract one component from the SDK's ``get_joint_states`` report.

    A few test doubles and wrappers expose a dict/object instead of the public
    tuple.  Those forms are accepted, but arbitrary list positions outside the
    documented ``[q, dq, effort]`` payload are deliberately ignored.
    """
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (int, np.integer)):
        code, payload = int(value[0]), value[1]
        if code != 0:
            return None
        value = payload
    if isinstance(value, dict):
        names = (("position", "positions", "q") if component == 0 else
                 ("velocity", "vel", "dq") if component == 1 else
                 ("effort", "torque", "current", "iq"))
        for name in names:
            if name in value:
                candidate = _extract_vector(value[name], length)
                if candidate is not None:
                    return candidate
        return None
    if isinstance(value, (list, tuple)) and len(value) > component:
        return _extract_vector(value[component], length)
    return None


def _first_numeric(value: Any, default: float) -> float:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            value = value[1]
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        return float(array[0]) if array.size and np.isfinite(array[0]) else default
    except (TypeError, ValueError):
        return default
