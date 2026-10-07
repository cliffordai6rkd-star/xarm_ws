"""Shared DP/pi0, contact/latent WM preview integration; no actuator access."""
from __future__ import annotations

import math
import logging
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from inference.mujoco_visualization import MujocoKinematicVisualizer
from nero_collection.cameras import CameraPreviewOverlay

PHASE_NAMES = ('free', 'pre-contact', 'contact')
log = logging.getLogger(__name__)


def configure_visualization(config, directory):
    options = config.setdefault('visualization', {})
    if not isinstance(options, dict):
        raise ValueError('visualization must be a mapping')
    if type(options.setdefault('camera_contact', False)) is not bool:
        raise ValueError('visualization.camera_contact must be a boolean')
    ext = options.setdefault('tau_ext', {})
    if not isinstance(ext, dict):
        raise ValueError('visualization.tau_ext must be a mapping')
    for key, default in (('enabled', False), ('plot', True)):
        if type(ext.setdefault(key, default)) is not bool:
            raise ValueError(f'visualization.tau_ext.{key} must be boolean')
    mj = options.setdefault('mujoco', {})
    if not isinstance(mj, dict):
        raise ValueError('visualization.mujoco must be a mapping')
    defaults = dict(enabled=False, headless=False, render_fps=30.,
                    prediction_visualization_hz=25., point_size=.003,
                    torque_base_length_m=.08, torque_gain=1., torque_baseline_floor_nm=.1,
                    torque_offset_m=.09,
                    torque_label_source='tau_ext' if ext['enabled'] else 'wm',
                    robot_joint_names=[f'joint{i}' for i in range(1, 8)],
                    ee_site_name=None, ee_body_name='link_eef',
                    sample_id=config.get('controller', {}).get('sample_id', 0))
    for key, value in defaults.items():
        mj.setdefault(key, value)
    for key in ('enabled', 'headless'):
        if type(mj[key]) is not bool:
            raise ValueError(f'visualization.mujoco.{key} must be boolean')
    if mj['torque_label_source'] not in ('wm', 'tau_ext', 'both', 'none'):
        raise ValueError('visualization.mujoco.torque_label_source must be wm/tau_ext/both/none')
    for key in ('render_fps', 'prediction_visualization_hz', 'point_size',
                'torque_base_length_m', 'torque_gain', 'torque_baseline_floor_nm', 'torque_offset_m'):
        value = mj[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'visualization.mujoco.{key} must be positive and finite')
    if (type(mj['sample_id']) is not int or not 0 <= mj['sample_id'] < config['wm']['num_samples']):
        raise ValueError('visualization.mujoco.sample_id must identify a WM sample')
    if mj['robot_joint_names'] != [f'joint{i}' for i in range(1, 8)]:
        raise ValueError('visualization requires WM joint1..joint7 order')
    path = Path(mj.get('mujoco_model_path', config['wm']['urdf_path'])).expanduser()
    mj['mujoco_model_path'] = (path if path.is_absolute() else directory / path).resolve()
    if mj['enabled'] and not mj['mujoco_model_path'].is_file():
        raise FileNotFoundError(mj['mujoco_model_path'])


def contact_overlay(forecast, sample_id, now_s):
    probabilities = np.asarray(forecast['contact_probability'], dtype=float)
    times = np.asarray(forecast['future_timestamps_us'], dtype=np.int64)
    if (probabilities.ndim != 3 or probabilities.shape[1:] != (len(times), 3)
            or not 0 <= sample_id < probabilities.shape[0]
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(-1), 1., atol=.01)
            or not len(times) or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid WM three-phase probabilities/timestamps')
    index = min(int(np.searchsorted(times, now_s * 1e6)), len(times)-1)
    p = probabilities[sample_id]
    phase = int(p[index].argmax())
    states = p.argmax(-1)
    anchor = forecast['anchor_us']
    releases = np.flatnonzero((states[:-1] == 2) & (states[1:] != 2)) + 1
    releases = releases[releases >= index]
    release = (f'Predicted release: +{(times[releases[0]]-anchor)/1000:.0f} ms'
               if len(releases) else 'Predicted release: none in remaining forecast')
    lines = (f'WM sample={sample_id} t=+{(times[index]-anchor)/1000:.0f} ms: {PHASE_NAMES[phase]}',
             f'P free={p[index,0]:.1%} pre-contact={p[index,1]:.1%} contact={p[index,2]:.1%}',
             release, 'Future 10..320 ms: green=free yellow=pre-contact red=contact')
    return CameraPreviewOverlay(lines, tuple(states.tolist()), times[-1]/1e6)


class WmVisualization:
    def __init__(self, config, source, *, mock=False):
        options = config['visualization']
        settings = dict(options['mujoco'])
        self.ext = getattr(source, 'tau_ext_preview', None)
        if self.ext is not None:
            settings['tau_ext_stale_s'] = self.ext.stale_s
        if mock:
            settings['headless'] = True
        self.visualizer = MujocoKinematicVisualizer(SimpleNamespace(**settings))
        manager = getattr(source, 'cameras', None)
        self.camera_contact = options['camera_contact']
        self.camera = getattr(manager, 'visualizer', None) if self.camera_contact or self.ext is not None else None
        self.last_ext_key = None
        self.sample_id = settings['sample_id']
        self.forecast = None
        self.prediction_id = 0
        self.last_observed_us = -1
        self.last_overlay_s = -math.inf
        self._overlay_error_logged = False

    def start(self):
        self.visualizer.start()

    def reset(self):
        self.forecast = None
        self.last_observed_us = -1
        self.last_overlay_s = -math.inf
        self.visualizer.reset()
        self.last_ext_key = None
        if self.ext is not None:
            self.ext.reset()
        if self.camera is not None:
            self.camera.set_overlay(CameraPreviewOverlay())

    def update(self, state, events):
        # Existing visualizer uses monotonic seconds; state/WM use UNIX microseconds.
        offset = time.monotonic() - time.time()
        result = None if self.ext is None else self.ext.snapshot()
        ext_key = None if result is None else (result.source_timestamp_us, result.valid)
        if state is not None and (state.timestamp_us > self.last_observed_us or ext_key != self.last_ext_key):
            ext_args = ({} if result is None else dict(
                tau_ext=result.tau_ext if result.valid else None,
                tau_ext_timestamp=result.source_timestamp_us/1e6 + offset))
            self.visualizer.publish_observed(state.timestamp_us/1e6 + offset, state.q, **ext_args)
            self.last_observed_us = state.timestamp_us
            self.last_ext_key = ext_key
        for event in events:
            if event['kind'] != 'wm':
                continue
            self.forecast = event
            self.prediction_id += 1
            self.visualizer.publish_prediction(
                event['anchor_us']/1e6 + offset, event.get('wm_history_q_anchor', np.zeros(7)),
                event['q'], predicted_tau=event['tau'], prediction_id=self.prediction_id,
                future_timestamps=np.asarray(event['future_timestamps_us'])/1e6 + offset)
        wall = time.time()
        if self.camera is not None and wall-self.last_overlay_s >= 1/30:
            try:
                overlay = (contact_overlay(self.forecast, self.sample_id, wall)
                           if self.camera_contact and self.forecast is not None else CameraPreviewOverlay())
                if self.ext is not None:
                    overlay = with_tau_ext_overlay(overlay, result, self.ext.label, self.ext.stale_s, wall)
                self.camera.set_overlay(overlay)
            except (ValueError, KeyError, IndexError):
                self.camera.set_overlay(CameraPreviewOverlay())
                if not self._overlay_error_logged:
                    log.warning('Invalid WM contact preview; clearing overlay', exc_info=True)
                    self._overlay_error_logged = True
            self.last_overlay_s = wall

    def close(self):
        self.visualizer.close()


def with_tau_ext_overlay(overlay, result, label, stale_s, now_s):
    """Keep live residual visible even when WM is paused or its forecast expired."""
    valid = (result is not None and result.valid
             and 0 <= now_s-result.source_timestamp_us/1e6 <= stale_s
             and np.shape(result.tau_ext) == (7,) and np.isfinite(result.tau_ext).all())
    if valid:
        values = result.tau_ext
        live = (f'tau_ext [{label}] Nm | L1={np.abs(values).sum():.2f}',
                'J1..J4: ' + '  '.join(f'{value:+.2f}' for value in values[:4]),
                'J5..J7: ' + '  '.join(f'{value:+.2f}' for value in values[4:]))
        expires = result.source_timestamp_us/1e6 + stale_s
    else:
        live = (f'tau_ext [{label}]: warming up / invalid / stale',)
        expires = now_s + stale_s
    wm_valid = bool(overlay.lines) and now_s <= overlay.expires_at_s
    wm_lines = overlay.lines if wm_valid else (('WM: forecast expired',) if overlay.lines else ())
    return CameraPreviewOverlay(live + wm_lines, overlay.phases if wm_valid else (),
                                min(expires, overlay.expires_at_s) if wm_valid else expires,
                                'tau_ext/WM: preview expired; waiting for fresh data')
