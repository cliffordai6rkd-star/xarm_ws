"""Live observations with separately authorized planned reset motion."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
import json
import threading
import time

import numpy as np
import yaml

from inference.dp_wm.core import HistorySnapshot
from inference.pi0_wm.wm import CausalChain
from nero_collection.time_utils import now_us


def _cadence_samples(previous_stamp, previous_raw, stamp, raw, *, command_lookup=None, interpolate=True):
    """Bridge measured feedback, keeping the host command zero-order held.

    ``command_lookup`` maps a bridged feedback timestamp to the corresponding
    acquisition timestamp before looking up its last successful host send.
    """
    if previous_stamp is None:
        return [(stamp, raw)]
    gap = stamp - previous_stamp
    if 0 < gap < 6_000:
        return []  # Duplicate/early poll; wait for a distinct state.
    if 6_000 <= gap <= 14_000:
        return [(stamp, raw)]
    if not interpolate:
        raise ValueError(f'State acquisition gap {gap/1000:.3f} ms violates strict measured 100 Hz cadence')
    if 14_000 < gap <= 24_000:
        middle = (previous_stamp + stamp) // 2
        fraction = (middle - previous_stamp) / gap
        interpolated = {key: previous_raw[key] + fraction * (raw[key] - previous_raw[key])
                        for key in raw if key not in ('q_cmd', 'delta_q')}
        qcmd = (command_lookup(middle)[0] if command_lookup is not None
                else previous_raw['q_cmd'] if 'q_cmd' in previous_raw
                else previous_raw['q'] + previous_raw['delta_q'])
        if 'q_cmd' in raw:
            interpolated['q_cmd'] = np.array(qcmd, copy=True)
        if 'delta_q' in raw:
            interpolated['delta_q'] = qcmd - interpolated['q']
        return [(middle, interpolated), (stamp, raw)]
    raise ValueError(f'State acquisition gap {gap/1000:.3f} ms exceeds the 100 Hz bridge limit')


class LiveSource:
    def __init__(self, config, metadata, *, enable_command=False):
        from nero_collection.config import ArmEndpointConfig, CameraConfig
        from nero_collection.cameras import CameraManager, CameraVisualizer
        from ufactory_devices.robot.xarm_adapter import XArmAdapter
        options = config['observation']
        self.options = options
        self.single_observation = config.get('policy', {}).get('type') == 'pi0'
        path = options['hardware_config']
        hardware = yaml.safe_load(path.read_text())
        calibration = yaml.safe_load((path.parent / hardware['gello_config']).resolve().read_text())
        side = options['arm']
        robot = calibration[side]['RobotConfig']
        kwargs = dict(hardware.get('arms', {}).get(side, {}))
        kwargs.update(robot_ip=robot['robot_ip'], dof=7, execution_enabled=enable_command, feedback_signal='torque',
                      velocity_source='hardware', is_radian=True, init_gripper=False)
        if config.get('controller', {}).get('enabled') and config['controller']['execution'] == 'active':
            if not enable_command:
                raise ValueError('Active output requires explicit command authorization')
            kwargs.update(servo_mode=1, servo_api='set_servo_angle_j')
        self.arm = XArmAdapter(ArmEndpointConfig(side, config_kwargs=kwargs))
        self.command_history = None
        if options.get('command_source', 'controller_reference') == 'host_command':
            from inference.dp_wm.command_history import HostCommandHistory, RecordingArm
            self.command_history = HostCommandHistory()
            self.arm = RecordingArm(self.arm, self.command_history)
        self.reset_motion = None
        self.observation_after_us = 0
        if enable_command:
            from inference.dp_wm.reset import ResetMotion
            target = config.get('execution', {}).get('initial_q', {}).get(side, robot.get('reset_q'))
            self.reset_motion = ResetMotion(self.arm, target, hardware.get('alignment', {}),
                                            config['wm']['urdf_path'])
        self.image_keys = tuple(metadata['images'])
        self.mapping = {key: options.get('camera_map', {}).get(key, key) for key in self.image_keys}
        if len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError('Each DP camera must map to a separate hardware stream')
        cameras = []
        for item in hardware['cameras']:
            item = dict(item)
            item.pop('arm', None)
            if item['name'] in self.mapping.values() and item.get('enabled', True):
                key = next(k for k, v in self.mapping.items() if v == item['name'])
                shape = metadata['images'][key]
                item['output_size'] = [shape[2], shape[1]]
                item['preview_output_size'] = (224, 224)
                item['depth'] = False
                item['visualize'] = options.get('visualize', False)
                cameras.append(CameraConfig(**item))
        if {c.name for c in cameras} != set(self.mapping.values()):
            raise ValueError('Missing DP hardware camera')
        self.cameras = CameraManager.from_config(tuple(cameras), visualizer=CameraVisualizer.from_config(cameras))
        self.frames = {name: deque(maxlen=20) for name in self.mapping.values()}
        self.observations = deque(maxlen=20)
        self.states = deque(maxlen=100)
        self.rows = deque(maxlen=50)
        self.history_cache = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error = None
        self.status = 'waiting for state history'
        self.history_interpolated_rows = 0
        self.history_rebuilds = 0
        self.last_history_gap_ms = None
        self.observation_status = 'waiting for camera observations'
        self.thread = threading.Thread(target=self._read, name='dp-wm-observation', daemon=True)
        training_data = metadata['wm_training']['dataloader']
        self.strict_measured_history = (training_data.get('backend') == 'lerobot'
                                        and training_data.get('v3_only') is True)
        filters = training_data.get('filters', {})
        # Source hardware is raw; replay data's source_already_filtered flag
        # does not apply. Reproduce the causal training filter once here.
        self.operations = {key: spec.get('operations', []) for key, spec in filters.items() if spec.get('enabled')}
        if set(self.operations) - {'dq', 'tau', 'q', 'delta_q'}:
            raise ValueError('Unsupported live WM preprocessing')
        self.filters = {k: CausalChain(self.operations.get(k, [])) for k in ('q', 'dq', 'delta_q', 'tau')}
        self.tau_ext_preview = None
        ext = config.get('visualization', {}).get('tau_ext', {})
        if ext.get('enabled', False):
            from inference.tau_ext_visualization import TauExtPreview
            self.tau_ext_preview = TauExtPreview(ext, hardware, path, side, self.arm)

    def start(self):
        try:
            self.arm.connect()
            self.arm.verify_torque_report()
            self.cameras.start()
            if self.tau_ext_preview is not None:
                self.tau_ext_preview.start()
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def _command(self, state_time):
        if self.options['command_source'] == 'host_command':
            # This timestamp is the start of feedback acquisition, not the
            # later q report receipt. A long-held target is still causal.
            return self.command_history.at_or_before(state_time)
        if self.options['command_source'] == 'controller_reference':
            q, stamp = self.arm.read_controller_reference()
        else:
            packet = json.loads(Path(self.options['command_file']).read_text())
            q, stamp = np.asarray(packet['q_cmd'], dtype=float), int(packet['timestamp_us'])
            if stamp > state_time:
                raise ValueError('External command is newer than measured state')
        if np.shape(q) != (7,) or not np.isfinite(q).all():
            raise ValueError('Command reference must be finite seven-joint radians')
        if abs(stamp-state_time) > self.options['maximum_pair_skew_ms']*1000:
            raise ValueError('Command/reference and measured state timestamps are too far apart')
        return q, stamp

    def _read(self):
        previous = None
        previous_raw = None
        previous_acquired = None
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                state = self.arm.read_joint_state()
                stamp = int(state.q_timestamp_us or state.timestamp_us)
                acquired = int(getattr(state, 'q_acquired_timestamp_us', 0)
                               or getattr(state, 'acquired_timestamp_us', 0) or stamp)
                if getattr(self, 'tau_ext_preview', None) is not None:
                    self.tau_ext_preview.submit(state)
                if not state.q_valid or not np.isfinite(state.q).all():
                    raise ValueError('Invalid measured q')
                with self.lock:
                    self.states.append(state)
                try:
                    host_commands = self.options['command_source'] == 'host_command'
                    strict_history = host_commands or getattr(self, 'strict_measured_history', False)
                    qcmd, command_stamp = self._command(acquired if host_commands else stamp)
                    if not state.dq_valid or not state.torque_valid:
                        raise ValueError('WM requires measured velocity and torque')
                    raw = {'q': state.q, 'dq': state.dq, 'tau': state.torque,
                           'q_cmd': qcmd, 'delta_q': qcmd-state.q}
                    if any(not np.isfinite(value).all() for value in raw.values()):
                        raise ValueError('Nonfinite WM feedback')
                    if strict_history and previous_acquired is not None:
                        acquisition_gap = acquired - previous_acquired
                        if acquisition_gap <= 0 or acquisition_gap > 14_000:
                            raise ValueError(f'State acquisition gap {acquisition_gap/1000:.3f} ms '
                                             'violates strict measured 100 Hz cadence')
                    samples = _cadence_samples(previous, previous_raw, stamp, raw,
                                               interpolate=not strict_history)
                    rows = []
                    last_stamp = previous
                    for sample_stamp, sample in samples:
                        dt = .01 if last_stamp is None else (sample_stamp-last_stamp)/1e6
                        row = {key: self.filters[key].apply(sample[key], dt) for key in self.filters}
                        rows.append((sample_stamp, row))
                        last_stamp = sample_stamp
                    with self.lock:
                        self.rows.extend(rows)
                        self.history_interpolated_rows += max(0, len(rows)-1)
                        self.status = (f'history {len(self.rows)}/50; interpolated={self.history_interpolated_rows}; '
                                       f'rebuilds={self.history_rebuilds}; source={self.options["command_source"]}')
                    if samples:
                        previous, previous_raw = stamp, {key: value.copy() for key, value in raw.items()}
                        previous_acquired = acquired
                except (ValueError, OSError, KeyError) as exc:
                    with self.lock:
                        self.rows.clear()
                        self.history_rebuilds += 1
                        self.last_history_gap_ms = None if previous is None else (stamp-previous)/1000
                        self.status = str(exc)
                    previous = None
                    previous_raw = None
                    previous_acquired = None
                    self.filters = {k: CausalChain(self.operations.get(k, [])) for k in self.filters}
                self.stop_event.wait(max(0., .01-(time.monotonic()-started)))
        except Exception as exc:
            self.error = exc

    def poll(self):
        if self.error is not None:
            raise RuntimeError('Live observation reader failed') from self.error
        for frame in self.cameras.poll():
            if frame.camera_name in self.frames:
                frames = self.frames[frame.camera_name]
                if not frames or frame.timestamp_us > frames[-1].timestamp_us:
                    frames.append(frame)
        with self.lock:
            states, rows = tuple(self.states), tuple(self.rows)
        history = None
        if len(rows) == 50 and rows[0][0] > self.observation_after_us:
            if self.history_cache is None or self.history_cache.anchor_us != rows[-1][0]:
                self.history_cache = HistorySnapshot(np.array([t for t, _ in rows], dtype=np.int64),
                    {key: np.stack([row[key] for _, row in rows]) for key in ('q', 'dq', 'delta_q', 'tau')})
            history = self.history_cache
        else:
            self.history_cache = None
        now = now_us()
        self.observation_status = 'waiting for camera/state observation'
        capture_reason = None
        missing_cameras = [name for name, frames in self.frames.items() if not frames]
        if missing_cameras:
            capture_reason = 'waiting for camera frames: ' + ', '.join(missing_cameras)
        elif not states:
            capture_reason = 'waiting for measured state at camera anchor'
        elif self.frames:
            anchor = min(frames[-1].timestamp_us for frames in self.frames.values())
            state = next((s for s in reversed(states) if s.timestamp_us <= anchor), None)
            chosen = {key: next((f for f in reversed(self.frames[name]) if f.timestamp_us <= anchor), None)
                      for key, name in self.mapping.items()}
            maximum_age_us = self.options['maximum_age_ms']*1000
            if anchor > now:
                capture_reason = f'camera anchor is ahead of host time; ahead_ms={(anchor-now)/1000:.3f}'
            elif now-anchor > maximum_age_us:
                stale = [f'{name} age_ms={(now-frames[-1].timestamp_us)/1000:.3f}'
                         for name, frames in self.frames.items()
                         if now-frames[-1].timestamp_us > maximum_age_us]
                capture_reason = ('camera frames stale: ' + ', '.join(stale)
                                  + f'; maximum_age_ms={self.options["maximum_age_ms"]:.3f}')
            elif state is None:
                capture_reason = 'waiting for causal measured state at camera anchor'
            elif anchor-state.timestamp_us > maximum_age_us:
                capture_reason = (f'camera/state skew_ms={(anchor-state.timestamp_us)/1000:.3f}'
                                  + f' exceeds maximum_age_ms={self.options["maximum_age_ms"]:.3f}')
            else:
                unmatched = [self.mapping[key] for key, frame in chosen.items() if frame is None]
                skewed = [f'{self.mapping[key]} skew_ms={(anchor-frame.timestamp_us)/1000:.3f}'
                          for key, frame in chosen.items() if frame is not None
                          and anchor-frame.timestamp_us > self.options['maximum_camera_skew_ms']*1000]
                if unmatched:
                    capture_reason = 'waiting for camera frames at anchor: ' + ', '.join(unmatched)
                elif skewed:
                    capture_reason = ('camera frames exceed maximum_camera_skew_ms='
                                      + f'{self.options["maximum_camera_skew_ms"]:.3f}: ' + ', '.join(skewed))
            if capture_reason is None and (not self.observations or anchor > self.observations[-1].timestamp_us):
                images = {key: np.array(f.frame, copy=True) for key,f in chosen.items()}
                for image in images.values():
                    image.setflags(write=False)
                self.observations.append(SimpleNamespace(timestamp_us=anchor, q=np.array(state.q),
                    images=images, metadata={}))
        if capture_reason is not None:
            self.observation_status = capture_reason
        observations = ()
        if self.single_observation and self.observations:
            last = self.observations[-1]
            age_ms = (now-last.timestamp_us)/1000
            if last.timestamp_us <= self.observation_after_us:
                self.observation_status = 'waiting for single policy observation after reset/start'
            elif 0 <= age_ms <= self.options['maximum_age_ms']:
                observations = (last,)
                self.observation_status = 'ready; single policy observation'
            elif capture_reason is None:
                self.observation_status = (f'single policy observation is stale or ahead of host time; age_ms={age_ms:.3f}'
                                           + f'; maximum_age_ms={self.options["maximum_age_ms"]:.3f}')
        elif not self.single_observation and self.observations:
            last = self.observations[-1]
            # The policy was trained at 25 Hz, but cameras commonly run at
            # 30 Hz. Requiring an earlier sample at least 40 ms away rejects
            # regular 30 Hz pairs: 33 ms is too close and 67 ms is too old.
            # Selecting 25 fps from 30 fps acquisition also produces 33/67 ms
            # gaps. Keep the nearest distinct past observation with its real
            # timestamp; the 20..70 ms window admits both normal gaps.
            first = min((o for o in self.observations if o.timestamp_us < last.timestamp_us),
                        key=lambda o:abs(o.timestamp_us-(last.timestamp_us-40_000)),default=None)
            spacing=None if first is None else last.timestamp_us-first.timestamp_us
            if capture_reason is None:
                self.observation_status = ('waiting for second camera observation' if spacing is None else
                                           f'waiting for valid policy pair; spacing_ms={spacing/1000:.3f}')
            if (first is not None and first.timestamp_us > self.observation_after_us
                    and 0 <= now-last.timestamp_us <= self.options['maximum_age_ms']*1000
                    and 20_000 <= spacing <= 70_000):
                observations = (first, last)
                self.observation_status = f'ready; pair spacing_ms={spacing/1000:.3f}'
        return observations, history

    def close(self):
        self.stop_event.set()
        try:
            if self.reset_motion is not None:
                self.reset_motion.stop()
        finally:
            if self.thread.ident is not None:
                self.thread.join(timeout=2.)
            try:
                self.cameras.stop()
            finally:
                try:
                    if self.tau_ext_preview is not None:
                        self.tau_ext_preview.close()
                finally:
                    self.arm.disconnect()

    def control_state(self):
        """Latest raw hardware feedback, separate from filtered WM history/camera anchor."""
        from inference.controllers.wm_two_step import ControlState

        with self.lock:
            if not self.states:
                return None
            state = self.states[-1]
            if not state.q_valid or not state.dq_valid:
                return None
            return ControlState(int(state.q_timestamp_us or state.timestamp_us),
                                state.q.copy(), state.dq.copy(), state.torque.copy())
