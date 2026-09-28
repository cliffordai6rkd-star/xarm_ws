"""Bounded, operator-supervised single-joint direction checks on a GELLO bus."""
import json
from pathlib import Path
import select
import signal
import sys
import threading
import time

import numpy as np

from gello_teleop.build_gello_urdf import vector
from gello_teleop.gello_identification import save_npz
from gello_teleop.gello_identification_hardware import IdentificationBus, check_sample
from gello_teleop.measure_gello_ranges import stable_snapshot


class DirectionCheckPlan:
    """One-sided encoder paths; does not assume the URDF signs/zero are verified."""

    def __init__(self, config, reference_raw, step_deg=15., move_seconds=8.):
        self.ids = list(config['joint_ids'])
        self.n = len(self.ids)
        if self.n != 7 or len(set(self.ids)) != self.n:
            raise ValueError('Direction check expects seven distinct arm motor IDs')
        self.center = vector(reference_raw, 'reference encoder pose', self.n)
        self.test_signs = vector(config['encoder_signs'], 'existing signs', self.n)
        if not np.all(np.abs(self.test_signs) == 1):
            raise ValueError('Existing encoder signs must be ±1')
        old_offsets = vector(config['encoder_offsets_rad'], 'existing offsets', self.n)
        old_lower = vector(config['lower_rad'], 'measured lower limits', self.n)
        old_upper = vector(config['upper_rad'], 'measured upper limits', self.n)
        if np.any(old_lower >= old_upper):
            raise ValueError('Measured limits must be ordered')
        # Old encoder mapping is only used to recover measured RAW operating
        # intervals; it need not agree with the newly selected model zero.
        endpoints = old_offsets + np.stack([old_lower, old_upper])/self.test_signs
        self.lower, self.upper = endpoints.min(axis=0), endpoints.max(axis=0)
        self.signs, self.offsets = np.ones(self.n), np.zeros(self.n)
        self.step_deg, self.move_seconds = float(step_deg), float(move_seconds)
        if not np.isfinite([self.step_deg, self.move_seconds]).all() or not (
                0 < self.step_deg <= 15 and self.move_seconds > 0):
            raise ValueError('Use a finite step in (0,15] degrees and positive move duration')
        self.amplitude = np.full(self.n, np.deg2rad(self.step_deg))
        self.targets = np.tile(self.center, (self.n, 1))
        self.targets[np.arange(self.n), np.arange(self.n)] += self.test_signs*self.amplitude
        paths = np.vstack([self.center, self.targets])
        self.raw_goal_bounds = np.stack([paths.min(axis=0), paths.max(axis=0)])
        for i in range(self.n):
            if self.center[i] < self.lower[i] or self.center[i] > self.upper[i]:
                raise ValueError(f'J{i+1}: initial pose is outside the measured interval: '
                                 f'raw={self.center[i]:.6f} rad / {self.center[i]*2048/np.pi:.1f} ticks; '
                                 f'allowed raw=[{self.lower[i]:.6f}, {self.upper[i]:.6f}] rad. '
                                 '未使能；请在当前模型中位附近重新测量此轴范围，不能直接放宽限位。')
            if self.raw_goal_bounds[0, i] < self.lower[i] or self.raw_goal_bounds[1, i] > self.upper[i]:
                available = ((self.upper[i]-self.center[i]) if self.test_signs[i] > 0 else
                             (self.center[i]-self.lower[i]))
                raise ValueError(f'J{i+1}: {self.step_deg:g}° exceeds the measured interval '
                                 f'(available {np.rad2deg(available):.2f}°). '
                                 'No torque enabled; choose a smaller --step-deg or remeasure the interval.')
        ticks = np.rint(self.raw_goal_bounds*2048/np.pi)
        if np.any(ticks < 0) or np.any(ticks > 4095):
            raise ValueError('Direction path crosses the single-turn encoder boundary; wrapping is forbidden')
        self.max_velocity = min(float(config['max_velocity_rad_s']), .08)
        self.max_acceleration = min(float(config['max_acceleration_rad_s2']), .5)
        self.velocity_bound = 1.875*self.amplitude/self.move_seconds
        self.acceleration_bound = (10/np.sqrt(3))*self.amplitude/self.move_seconds**2
        if not np.isfinite([self.max_velocity, self.max_acceleration]).all() or min(
                self.max_velocity, self.max_acceleration) <= 0:
            raise ValueError('Velocity/acceleration limits must be finite and positive')
        if np.any(self.velocity_bound > self.max_velocity) or np.any(
                self.acceleration_bound > self.max_acceleration):
            raise ValueError('Move duration is too short for the configured speed/acceleration limits')

    def interpolate(self, start, target, elapsed):
        u = np.clip(float(elapsed)/self.move_seconds, 0., 1.)
        blend = u**3*(10-15*u+6*u*u)
        return np.asarray(start)+(np.asarray(target)-np.asarray(start))*blend


def poll_line(_allow_reply=True):
    """Never block the bus while the operator is thinking or typing."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    line = sys.stdin.readline()
    if not line:
        raise EOFError('Terminal input closed; shutting down the direction check')
    return line.strip().lower()


class DirectionSession:
    def __init__(self, bus, plan, config, stop, clock=time.monotonic,
                 read_line=poll_line, answer_timeout=120.):
        self.bus, self.plan, self.config, self.stop = bus, plan, config, stop
        self.clock, self.read_line = clock, read_line
        self.answer_timeout = float(answer_timeout)
        hz = float(config['sample_hz'])
        gap = float(config['max_sample_gap_s'])
        if not np.isfinite([hz, gap, self.answer_timeout]).all() or min(hz, gap, self.answer_timeout) <= 0:
            raise ValueError('Sample rate, maximum gap and answer timeout must be positive')
        if hz > 10 or 1/hz >= gap:
            raise ValueError('Use sample_hz <=10 and a sample period below max_sample_gap_s')
        self.period = 1/hz
        self.next_tick = self.clock()
        self.previous_read = time.monotonic_ns()
        self.command = plan.center.copy()
        self.last_sample = None
        self.rows, self.commands, self.phases = [], [], []

    def tick(self, command, joint, phase, allow_reply=False):
        if self.stop.wait(max(0., self.next_tick-self.clock())) or self.stop.is_set():
            raise InterruptedError('Direction check interrupted')
        self.bus.command(command, self.plan)
        sample = self.bus.sample()
        self.rows.append(sample)
        self.commands.append(np.asarray(command).copy())
        self.phases.append(phase)
        check_sample(sample, self.config, self.plan, command)
        if (sample['read_finished_ns']-self.previous_read)*1e-9 > float(self.config['max_sample_gap_s']):
            raise TimeoutError('Telemetry/command gap exceeded during direction check')
        self.previous_read = sample['read_finished_ns']
        actual = sample['raw_q_rad']
        others = np.arange(self.plan.n) != joint if joint is not None else np.ones(self.plan.n, bool)
        if np.any(np.abs(actual[others]-self.plan.center[others]) > np.deg2rad(2)):
            raise RuntimeError('A held joint moved more than 2°; aborting')
        self.command, self.last_sample = np.asarray(command).copy(), sample
        self.next_tick = max(self.next_tick+self.period, self.clock())
        line = self.read_line(allow_reply)
        if line == 'q':
            raise InterruptedError('Operator stopped the direction check')
        if line is not None and not allow_reply:
            print('运动中可输入 q 或 Ctrl+C 停止；请等提示后再输入 y/n。', flush=True)
            return None
        return line

    def move(self, target, joint, phase):
        start, began = self.command.copy(), self.clock()
        while True:
            elapsed = self.clock()-began
            self.tick(self.plan.interpolate(start, target, elapsed), joint, phase)
            if elapsed >= self.plan.move_seconds:
                break
        return self.settle(target, joint, phase+'_settle')

    def settle(self, target, joint, phase):
        began, stable = self.clock(), 0
        while self.clock()-began < 6.:
            self.tick(target, joint, phase)
            error = np.max(np.abs(self.last_sample['raw_q_rad']-target))
            speed = np.max(np.abs(self.last_sample['raw_dq_rad_s']))
            stable = stable+1 if error <= float(self.config['start_tolerance_rad']) and speed <= .03 else 0
            if stable >= 4:
                return self.last_sample['raw_q_rad'].copy()
        raise TimeoutError('Joint did not settle at the commanded pose')

    def ask(self, prompt, target, joint, phase, allowed=('y', 'n')):
        print(prompt, end='', flush=True)
        began = self.clock()
        while self.clock()-began < self.answer_timeout:
            line = self.tick(target, joint, phase, allow_reply=True)
            if line is None:
                continue
            if line in allowed:
                return line
            print('请输入 '+('/'.join(allowed) if allowed != ('',) else 'Enter')+'；q 停止：', end='', flush=True)
        raise TimeoutError('Operator answer timed out; shutting down torque')


def run_active(config, geometry, output, step_deg=15., move_seconds=8., answer_timeout=120.):
    """Do not call unattended: the operator aligns, observes and answers at the rig."""
    output, telemetry = Path(output), Path(output).with_suffix('.npz')
    if output.suffix != '.json' or output.exists() or telemetry.exists():
        raise FileExistsError('Use a new .json output and a new .npz companion')
    if not sys.stdin.isatty():
        raise ValueError('Active direction checks require an interactive terminal')
    reference = vector(geometry['comparison_reference_q_deg'], 'model reference degrees', 7)
    # Validate monitor values and durations before opening the bus or enabling torque.
    for name in ['start_tolerance_rad', 'max_tracking_error_rad']:
        value = float(config[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    maximum = vector(config['max_current_a'], 'max_current_a', 7)
    if np.any(maximum <= 0) or not 0 < float(config['max_temperature_c']) <= 70 or not (
            3.7 <= float(config['min_voltage_v']) < float(config['max_voltage_v']) <= 6.):
        raise ValueError('Invalid XL330 current/temperature/voltage thresholds')
    if not np.isfinite([step_deg, move_seconds, answer_timeout]).all() or not (
            0 < step_deg <= 15 and move_seconds > 0 and answer_timeout > 0):
        raise ValueError('Use a step in (0,15] degrees and positive move/answer durations')
    print('主动核对：先手动摆好并读取当前位置，然后使能七个臂电机闭环保持。')
    print(f'逐轴缓慢移动 {step_deg:g}°（每次 {move_seconds:g} 秒），y=与网页正方向一致，n=方向相反。')
    print('若动的不是当前关节，输入 q 停止。运动/等待期间 q、Ctrl+C 都可停止。')
    print('每次回答后自动返回初始位置；结束/失败会关闭力矩，请保持机械支撑。')
    print('网页点击“返回中位”，角度：', reference.tolist())
    stop, handlers = threading.Event(), {}
    session, bus, rows = None, None, []
    report = {'execute': True, 'read_only': False, 'session_status': 'aborted',
              'session_type': 'single_joint_direction_check', 'joint_coordinates_verified': False,
              'encoder_zero_verified': False, 'geometry_verified': geometry.get('geometry_verified') is True,
              'model_comparison_reference_q_deg': reference.tolist(),
              'reference_model_q_rad': np.deg2rad(reference).tolist(),
              'config': config, 'step_deg': float(step_deg), 'move_seconds': float(move_seconds),
              'joint_checks': rows, 'torque_measured': False, 'arming_started': False}
    failure = None
    try:
        with IdentificationBus(config) as bus:
            initial = bus.inspect()
            if any(m['torque_on'] for m in initial['motors']):
                raise ValueError('Arm torque must be off before manual alignment; this tool does not change the initial mode')
            report['hardware_snapshot'] = initial
            input('先将 GELLO 手动摆成网页中位并支撑静止，按 Enter 读取位置并使能保持：')
            raw = stable_snapshot(bus)
            # Preserve the captured pose even when constructing the plan fails.
            report['reference_raw_rad'] = raw.tolist()
            report['reference_raw_ticks'] = (raw*2048/np.pi).tolist()
            report['reference_capture_method'] = 'Operator aligned physical arm to model middle before enabling torque'
            first = bus.sample()
            check_sample(first, config)
            plan = DirectionCheckPlan(config, raw, step_deg, move_seconds)
            old_offsets = vector(config['encoder_offsets_rad'], 'old offsets', 7)
            report['reference_q_in_existing_coordinates_rad'] = ((raw-old_offsets)*plan.test_signs).tolist()
            session = DirectionSession(bus, plan, config, stop, answer_timeout=answer_timeout)
            for sig in [signal.SIGINT, signal.SIGTERM]:
                handlers[sig] = signal.signal(sig, lambda *_: stop.set())
            report['arming_started'] = True
            bus.arm_direction_check(plan, first, stop, initial)
            session.previous_read = time.monotonic_ns()
            session.next_tick = session.clock()
            session.settle(plan.center, None, 'initial_hold')
            print('已使能，七个关节保持初始位置。', flush=True)
            for i, motor_id in enumerate(plan.ids):
                before = session.last_sample['raw_q_rad'].copy()
                print(f'J{i+1} / ID {motor_id}：仅此轴运动，其余关节保持。网页 J{i+1} 从 '
                      f'{reference[i]:g}° 增至 {reference[i]+step_deg:g}° 对照。', flush=True)
                session.move(plan.targets[i], i, f'j{i+1}_out')
                answer = session.ask(f'J{i+1} 运动方向是否与网页增加 {step_deg:g}° 一致？[y/n，q停止]：',
                                     plan.targets[i], i, f'j{i+1}_answer')
                after = session.last_sample['raw_q_rad'].copy()
                delta = after-before
                if abs(delta[i]) < np.deg2rad(max(.5, step_deg-2)) or delta[i]*plan.test_signs[i] <= 0:
                    raise RuntimeError('Measured motion does not match the commanded encoder step')
                inferred = int(np.sign(delta[i]))*(1 if answer == 'y' else -1)
                rows.append({'joint_id': motor_id, 'joint_name': f'joint{i+1}', 'operator_answer': answer,
                             'configured_encoder_sign': int(plan.test_signs[i]),
                             'configured_sign_matches': answer == 'y', 'viewer_positive_encoder_sign': inferred,
                             'before_raw_rad': before.tolist(), 'after_raw_rad': after.tolist(),
                             'raw_delta_deg': np.rad2deg(delta).tolist()})
                print('返回该关节初始位置……', flush=True)
                session.move(plan.center, i, f'j{i+1}_return')
            session.ask('全部已返回初始位置。请托住主手，按 Enter 关闭力矩并保存：',
                        plan.center, None, 'final_support', allowed=('',))
            report['session_status'] = 'complete'
    except BaseException as exc:
        failure = exc
        report['session_status'] = 'aborted'
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        report['failure'] = str(failure) if failure else None
        report['cleanup_failures'] = getattr(bus, 'cleanup_failures', [])
        report['all_configured_signs_match_observation'] = bool(len(rows) == 7 and all(
            r['configured_sign_matches'] for r in rows))
        output.parent.mkdir(parents=True, exist_ok=True)
        if session is not None and session.rows:
            arrays = {name: np.asarray([row[name] for row in session.rows]) for name in session.rows[0]}
            arrays['command_raw_q_rad'] = np.asarray(session.commands)
            arrays['phase'] = np.asarray(session.phases)
            arrays['time_s'] = (arrays['read_finished_ns']-arrays['read_started_ns'][0])*1e-9
            report['telemetry_file'] = str(telemetry.resolve())
            save_npz(telemetry, metadata_json=json.dumps(report), **arrays)
        with output.open('x') as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        cleanup_note = '退出时已尝试关闭力矩' if report['arming_started'] else '未进行使能或电机写入'
        print(f"已保存：{output}；状态：{report['session_status']}，{cleanup_note}。", flush=True)
    if failure:
        raise failure
