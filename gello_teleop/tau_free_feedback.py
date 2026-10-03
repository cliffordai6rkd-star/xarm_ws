"""Asynchronous free-space torque estimation and bounded GELLO current feedback."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
import logging
import queue
import threading

import numpy as np

from gello_teleop.gello_hardware import XL330_MAX_CURRENT_RAW
from nero_collection.config import SequenceCheckpointConfig
from nero_collection.filters import CausalFilterPipeline
from nero_collection.tau_ext_inference import SequenceTorquePredictor
from nero_collection.time_utils import now_us
from xarm_stack.momentum import PinocchioMomentumModel
from xarm_stack.torque_visualization import DEFAULT_URDF, FirstOrderLowPass

log = logging.getLogger(__name__)


def _vector(config, key, default, *, positive=False):
    value = config.get(key, default)
    array = np.full(7, float(value)) if np.isscalar(value) else np.asarray(value, float)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError(f'force_feedback.{key} must be a finite scalar or seven values')
    if np.any(array <= 0 if positive else array < 0):
        raise ValueError(f'force_feedback.{key} must be {"positive" if positive else "nonnegative"}')
    return array


class ResidualCurrentFeedback:
    """Hard magnitude threshold, then gain in raw servo current units / Nm."""
    def __init__(self, config):
        self.threshold = _vector(config, 'threshold_nm', 1.)
        self.gain = _vector(config, 'gain_raw_per_nm', 5.)
        if 'current_limit_percent' in config and 'current_limit_raw' in config:
            raise ValueError('Choose current_limit_percent or current_limit_raw, not both')
        if 'current_limit_percent' in config:
            percent = _vector(config, 'current_limit_percent', 0.)
            if np.any(percent > 100):
                raise ValueError('force_feedback.current_limit_percent must be in [0, 100]')
            self.limit = np.rint(percent * XL330_MAX_CURRENT_RAW / 100.)
        else:
            # Existing external configs can continue to use raw current limits.
            self.limit = _vector(config, 'current_limit_raw', 15., positive=True)
            if not np.equal(self.limit, np.floor(self.limit)).all():
                raise ValueError('force_feedback.current_limit_raw must contain integer raw current limits')
        self.rate_limit = _vector(config, 'rate_limit_raw_s', 200., positive=True)
        self.sign = np.asarray(config.get('sign', [1.] * 7), float)
        if self.sign.shape == ():
            self.sign = np.full(7, float(self.sign))
        if self.sign.shape != (7,) or not np.isin(self.sign, [-1., 1.]).all():
            raise ValueError('force_feedback.sign must contain +1/-1')
        self.ramp_s = float(config.get('ramp_s', 1.))
        if not np.isfinite(self.ramp_s) or self.ramp_s < 0:
            raise ValueError('force_feedback.ramp_s must be nonnegative')
        self.reset()

    def reset(self):
        self.previous = np.zeros(7)
        self.elapsed = 0.

    def update(self, residual, dt, *, valid=True):
        residual = np.asarray(residual, float)
        if not valid or residual.shape != (7,) or not np.isfinite(residual).all():
            self.reset()
            return np.zeros(7)
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('feedback dt must be positive')
        self.elapsed += dt
        active = np.abs(residual) > self.threshold
        target = np.where(active, residual * self.gain * self.sign, 0.)
        ramp = 1. if self.ramp_s == 0 else min(1., self.elapsed / self.ramp_s)
        target = np.clip(target * ramp, -self.limit, self.limit)
        self.previous = np.where(self.previous * target < 0, 0., self.previous)
        command = self.previous + np.clip(target - self.previous,
                                          -self.rate_limit * dt, self.rate_limit * dt)
        # A dropped/below-threshold signal must release immediately, including
        # sign reversals; the slew limiter only controls the rising feedback.
        command = np.where(active, command, 0.)
        self.previous = command.copy()
        return command


@dataclass(frozen=True)
class TauFreeResult:
    source_timestamp_us: int
    sample_timestamp_us: int
    tau_pred: np.ndarray
    tau_measured: np.ndarray
    tau_ext_raw: np.ndarray
    tau_ext: np.ndarray
    valid: bool
    tau_ext_unaveraged: np.ndarray | None = None


def _empty_result(source_timestamp_us=0, sample_timestamp_us=0):
    return TauFreeResult(source_timestamp_us, sample_timestamp_us,
                         np.full(7, np.nan), np.full(7, np.nan),
                         np.full(7, np.nan), np.full(7, np.nan), False)


class TauFreeEstimator:
    def __init__(self, predictor, sample_rate_hz, *, measured_tau_filter='raw', max_gap_s=.03):
        self.predictor = predictor
        metadata = predictor.metadata
        if (metadata.input_keys != ('q', 'dq', 'delta_q') or metadata.output_key != 'tau'
                or metadata.output_dim != 7 or any(dim != 7 for dim in metadata.input_dims.values())):
            raise ValueError('force feedback requires a q/dq/delta_q -> tau seven-joint checkpoint')
        if metadata.sample_rate_hz is None or not np.isclose(metadata.sample_rate_hz, sample_rate_hz):
            raise ValueError('force feedback sample rate must match checkpoint.sample_rate_hz')
        if measured_tau_filter not in {'raw', 'checkpoint'}:
            raise ValueError('force_feedback.measured_tau_filter must be raw or checkpoint')
        self.measured_tau_filter = measured_tau_filter
        self.max_gap_us = max_gap_s * 1e6
        self.filters = {}
        for key, spec in metadata.dataloader_filters.items():
            if key in metadata.input_keys or (key == 'tau' and measured_tau_filter == 'checkpoint'):
                if spec.get('enabled', False):
                    self.filters[key] = CausalFilterPipeline(spec['operations'])
        self.reset()

    def reset(self):
        self.predictor.reset()
        for pipeline in self.filters.values():
            pipeline.reset()
        self.previous_timestamp_us = None

    def update(self, timestamp_us, state, q_cmd, *, valid=True):
        source_us = int(getattr(state, 'acquired_timestamp_us', 0) or state.timestamp_us)
        features = {'q': np.asarray(state.q), 'dq': np.asarray(state.dq),
                    'delta_q': np.asarray(q_cmd) - np.asarray(state.q),
                    'tau': np.asarray(state.torque)}
        valid = valid and state.q_valid and state.dq_valid and state.torque_valid
        valid = valid and all(v.shape == (7,) and np.isfinite(v).all() for v in features.values())
        if not valid:
            self.reset()
            return _empty_result(source_us, timestamp_us)
        if self.previous_timestamp_us is not None:
            gap = timestamp_us - self.previous_timestamp_us
            if gap <= 0 or gap > self.max_gap_us:
                self.reset()
        self.previous_timestamp_us = timestamp_us
        processed = {key: self.filters[key].apply(value, timestamp_us)
                     if key in self.filters else value.copy() for key, value in features.items()}
        prediction = self.predictor.append_and_predict(processed)
        if prediction is None:
            return _empty_result(source_us, timestamp_us)
        prediction = np.asarray(prediction, float)
        if prediction.shape != (7,) or not np.isfinite(prediction).all():
            raise ValueError('tau_free prediction must be a finite seven-joint vector')
        return TauFreeResult(source_us, timestamp_us, prediction.copy(), processed['tau'],
                             features['tau'] - prediction, processed['tau'] - prediction, True)


def _mean_window(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < 1):
        raise ValueError('force_feedback.tau_ext_mean_window must be a positive integer')
    return int(value)


def _validated_payload(payload):
    """Validate tool settings without importing Pinocchio or talking to the robot."""
    if payload is None or (isinstance(payload, dict) and not payload):
        return None
    if not isinstance(payload, dict):
        raise ValueError('force_feedback.payload must be a mapping')
    source = payload.get('source', 'static')
    if source == 'controller':
        if set(payload) != {'source'}:
            raise ValueError('controller force_feedback.payload supports only source')
        return {'source': 'controller'}
    if source != 'static' or set(payload) - {'source', 'mass_kg', 'com_m', 'inertia_kg_m2'}:
        raise ValueError('force_feedback.payload has unsupported fields or source')
    try:
        mass = float(payload['mass_kg'])
        com = np.asarray(payload['com_m'], float)
        inertia = np.asarray(payload.get('inertia_kg_m2', np.zeros((3, 3))), float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('force_feedback.payload requires mass_kg and com_m') from exc
    if (not np.isfinite(mass) or mass < 0 or com.shape != (3,) or not np.isfinite(com).all()
            or inertia.shape != (3, 3) or not np.isfinite(inertia).all()
            or not np.allclose(inertia, inertia.T) or np.linalg.eigvalsh(inertia).min() < -1e-12):
        raise ValueError('Invalid force_feedback flange payload mass, CoM or inertia')
    return {'mass_kg': mass, 'com_m': com.tolist(), 'inertia_kg_m2': inertia.tolist()}


class URDFTorqueEstimator:
    """Measured torque minus causal rigid-body RNEA, on distinct robot reports.

    The two low-pass filters use report time, matching the torque plot. The
    acquisition time remains attached to the result for feedback expiry.
    """
    def __init__(self, model, *, measured_torque_cutoff_hz=3., acceleration_cutoff_hz=3.,
                 tau_ext_mean_window=1, max_gap_s=.03, payload_provider=None):
        self.model = model
        self.measured_filter = FirstOrderLowPass(measured_torque_cutoff_hz, max_gap_s)
        self.acceleration_filter = FirstOrderLowPass(acceleration_cutoff_hz, max_gap_s,
                                                     setting_name='acceleration_cutoff_hz')
        self.mean_window = _mean_window(tau_ext_mean_window)
        self.max_gap_us = max_gap_s * 1e6
        self.payload_provider = payload_provider
        self.reset()

    def reset(self):
        self.measured_filter.reset()
        self.acceleration_filter.reset()
        self.residuals = deque(maxlen=self.mean_window)
        self.previous_timestamp_us = None
        self.latest = _empty_result()

    def _release(self, source_us, sample_us, *, report_us=None):
        self.reset()
        # A rejected report cannot become a fresh integration step merely by
        # being resubmitted by the collector at the next 100 Hz tick.
        self.previous_timestamp_us = report_us
        self.latest = _empty_result(source_us, sample_us)
        return self.latest

    def update(self, timestamp_us, state, q_cmd=None, *, valid=True):
        source_us = int(getattr(state, 'acquired_timestamp_us', 0) or state.timestamp_us)
        report_us = int(state.timestamp_us)
        q, dq, ddq, tau = (np.asarray(getattr(state, key), float)
                           for key in ('q', 'dq', 'ddq', 'torque'))
        valid = (valid and state.q_valid and state.dq_valid and state.torque_valid
                 and getattr(state, 'ddq_valid', state.dq_valid)
                 and all(value.shape == (7,) and np.isfinite(value).all()
                         for value in (q, dq, ddq, tau)))
        if not valid:
            return self._release(source_us, timestamp_us, report_us=report_us)
        if self.previous_timestamp_us is not None:
            gap_us = report_us - self.previous_timestamp_us
            if gap_us == 0:
                return self.latest
            if gap_us < 0 or gap_us > self.max_gap_us:
                return self._release(source_us, timestamp_us, report_us=report_us)
        if self.payload_provider is not None:
            payload = self.payload_provider()
            if payload is None:
                return self._release(source_us, timestamp_us, report_us=report_us)
            changed = self.model.set_payload(payload)
            if changed and self.previous_timestamp_us is not None:
                return self._release(source_us, timestamp_us, report_us=report_us)
        self.previous_timestamp_us = report_us
        report_s = report_us / 1e6
        measured = self.measured_filter.update(report_s, tau)
        filtered_ddq = self.acceleration_filter.update(report_s, ddq)
        prediction = np.asarray(self.model.rnea(q, dq, filtered_ddq), float)
        if prediction.shape != (7,) or not np.isfinite(prediction).all():
            raise ValueError('URDF RNEA must return a finite seven-joint torque vector')
        residual = measured - prediction
        self.residuals.append(residual.copy())
        ready = len(self.residuals) == self.mean_window
        feedback = np.mean(self.residuals, axis=0) if ready else np.full(7, np.nan)
        self.latest = TauFreeResult(source_us, timestamp_us, prediction.copy(), measured.copy(),
                                    tau - prediction, feedback, ready, residual.copy())
        return self.latest


class TauFreeFeedbackWorker:
    """Own inference history off the position-control and serial-I/O threads."""
    def __init__(self, block, config_path, arm_names, sample_rate_hz, *, predictor_factory=None,
                 model_factory=None):
        allowed = {'enabled', 'checkpoint_path', 'device', 'threshold_nm', 'gain_raw_per_nm',
                   'current_limit_raw', 'current_limit_percent', 'rate_limit_raw_s', 'sign', 'ramp_s', 'measured_tau_filter',
                   'maximum_age_s', 'maximum_sample_gap_s', 'sides', 'source', 'urdf_path',
                   'gravity_m_s2', 'payload', 'measured_torque_cutoff_hz',
                   'acceleration_cutoff_hz', 'tau_ext_mean_window'}
        if not isinstance(block, dict) or set(block) - allowed:
            raise ValueError('force_feedback must be a mapping with supported fields')
        if type(block.get('enabled', False)) is not bool:
            raise ValueError('force_feedback.enabled must be a boolean')
        sides = block.get('sides', {})
        if not isinstance(sides, dict) or set(sides) - {'left', 'right'}:
            raise ValueError('force_feedback.sides must contain left/right settings')
        for side, override in sides.items():
            if not isinstance(override, dict) or set(override) - (allowed - {
                    'enabled', 'sides', 'source', 'maximum_age_s', 'maximum_sample_gap_s'}):
                raise ValueError(f'Unsupported force_feedback override for {side}')
        self.enabled = bool(block.get('enabled', False))
        self.source = block.get('source', 'checkpoint')
        if not isinstance(self.source, str) or self.source not in {'checkpoint', 'urdf'}:
            raise ValueError('force_feedback.source must be checkpoint or urdf')
        self.arm_names = tuple(arm_names)
        self.sample_rate_hz = sample_rate_hz
        self.settings = {}
        self.controllers = []
        self.estimators = []
        self.predictor_factory = predictor_factory or SequenceTorquePredictor
        self.model_factory = model_factory or PinocchioMomentumModel
        self.payload_sources = {}
        self.stale_s = float(block.get('maximum_age_s', .15))
        self.max_gap_s = float(block.get('maximum_sample_gap_s', .03))
        if not np.isfinite([self.stale_s, self.max_gap_s]).all() or min(self.stale_s, self.max_gap_s) <= 0:
            raise ValueError('force feedback maximum age/sample gap must be positive')
        for arm in arm_names:
            settings = {key: value for key, value in block.items() if key != 'sides'}
            override = sides.get(arm, {})
            if 'current_limit_percent' in override and 'current_limit_raw' in override:
                raise ValueError(f'Choose only one force feedback current limit for {arm}')
            if 'current_limit_percent' in override:
                settings.pop('current_limit_raw', None)
            elif 'current_limit_raw' in override:
                settings.pop('current_limit_percent', None)
            settings.update(override)
            source = settings.setdefault('source', 'checkpoint')
            if source not in {'checkpoint', 'urdf'}:
                raise ValueError('force_feedback.source must be checkpoint or urdf')
            if self.enabled and source == 'checkpoint' and not settings.get('checkpoint_path'):
                raise ValueError('force_feedback.checkpoint_path is required')
            if settings.get('measured_tau_filter', 'raw') not in {'raw', 'checkpoint'}:
                raise ValueError('force_feedback.measured_tau_filter must be raw or checkpoint')
            if source == 'checkpoint' or 'checkpoint_path' in settings:
                path = Path(settings.get('checkpoint_path', '')).expanduser()
                if not path.is_absolute():
                    path = Path(config_path).parent / path
                settings['checkpoint_path'] = str(path.resolve())
            settings['tau_ext_mean_window'] = _mean_window(settings.get('tau_ext_mean_window', 1))
            if source == 'urdf':
                path = Path(settings.get('urdf_path', DEFAULT_URDF)).expanduser()
                if not path.is_absolute():
                    path = Path(config_path).resolve().parent / path
                if self.enabled and not path.is_file():
                    raise ValueError(f'{arm} force feedback URDF does not exist: {path}')
                settings['urdf_path'] = str(path.resolve())
                gravity = np.asarray(settings.get('gravity_m_s2', (0., 0., -9.81)), float)
                if gravity.shape != (3,) or not np.isfinite(gravity).all():
                    raise ValueError('force_feedback.gravity_m_s2 must be a finite three-vector')
                settings['gravity_m_s2'] = gravity.tolist()
                for key in ('measured_torque_cutoff_hz', 'acceleration_cutoff_hz'):
                    settings.setdefault(key, 3.)
                    FirstOrderLowPass(settings[key], self.max_gap_s, setting_name=key)
                settings['payload'] = _validated_payload(settings.get('payload'))
            self.settings[arm] = settings
            self.controllers.append(ResidualCurrentFeedback(settings))
        self.queue = queue.Queue(maxsize=4)
        self.lock = threading.Lock()
        self.latest = tuple(_empty_result() for _ in arm_names)
        self.stop = threading.Event()
        self.thread = None
        self.dropped_samples = 0
        self.errors = 0
        self._generation = 0

    def prepare(self):
        if self.estimators or not self.enabled:
            return
        estimators = []
        for arm in self.arm_names:
            settings = self.settings[arm]
            if settings['source'] == 'urdf':
                model = self.model_factory(settings['urdf_path'], settings['gravity_m_s2'])
                payload = settings['payload']
                controller_payload = payload is not None and payload.get('source') == 'controller'
                if not controller_payload and payload is not None:
                    model.set_payload(payload)
                provider = (lambda side=arm: self._controller_payload(side)) if controller_payload else None
                estimators.append(URDFTorqueEstimator(model,
                    measured_torque_cutoff_hz=settings['measured_torque_cutoff_hz'],
                    acceleration_cutoff_hz=settings['acceleration_cutoff_hz'],
                    tau_ext_mean_window=settings['tau_ext_mean_window'], max_gap_s=self.max_gap_s,
                    payload_provider=provider))
                continue
            predictor = self.predictor_factory(SequenceCheckpointConfig(
                checkpoint_path=Path(settings['checkpoint_path']), device=settings.get('device', 'cpu'),
                input_keys=('q', 'dq', 'delta_q'), output_key='tau'), name=f'{arm}_tau_free')
            predictor.warm_up()
            estimators.append(TauFreeEstimator(predictor, self.sample_rate_hz,
                measured_tau_filter=settings.get('measured_tau_filter', 'raw'), max_gap_s=self.max_gap_s))
        self.estimators = estimators

    def bind_payload_source(self, side, arm):
        if side not in self.arm_names:
            raise ValueError(f'Unknown force feedback side: {side}')
        self.payload_sources[side] = arm

    def _controller_payload(self, side):
        source = self.payload_sources.get(side)
        # This property is the adapter's rich-report cache. No SDK method or
        # command is permitted on the asynchronous feedback path.
        payload = getattr(source, 'reported_tcp_payload', None)
        if payload is None:
            return None
        payload = _validated_payload(payload)
        if payload is not None and payload.get('source') == 'controller':
            raise ValueError('Cached controller payload must contain mass_kg and com_m')
        return payload

    def start(self):
        if not self.enabled:
            return
        self.prepare()
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name='tau-free-feedback', daemon=True)
            self.thread.start()

    def invalidate(self):
        with self.lock:
            self._generation += 1
            self.latest = tuple(_empty_result() for _ in self.arm_names)

    def submit(self, sample, active):
        with self.lock:
            generation = self._generation
        try:
            self.queue.put_nowait((sample, active, generation))
        except queue.Full:
            self.dropped_samples += 1
            self.invalidate()
            # Dropping old inputs leaves a gap; estimator history resets before
            # any feedback resumes. Never block the 100 Hz sampling thread.
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

    def snapshot(self, index, timestamp_us=None):
        timestamp_us = now_us() if timestamp_us is None else timestamp_us
        with self.lock:
            result = self.latest[index]
        age_us = timestamp_us - result.source_timestamp_us
        if not result.valid or not 0 <= age_us <= self.stale_s * 1e6:
            return _empty_result(result.source_timestamp_us, result.sample_timestamp_us)
        return result

    def _run(self):
        previous_generation = -1
        while not self.stop.is_set():
            try:
                sample, active, generation = self.queue.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                if generation != previous_generation:
                    for estimator in self.estimators:
                        estimator.reset()
                    previous_generation = generation
                results = []
                offset = 0
                for index, (estimator, state) in enumerate(zip(self.estimators, sample.follower_states)):
                    q_cmd = sample.q_cmd[offset:offset + 7]
                    offset += 7
                    age_us = now_us() - int(getattr(state, 'acquired_timestamp_us', 0) or state.timestamp_us)
                    valid = active and bool(sample.q_cmd_ok[index]) and 0 <= age_us <= self.stale_s * 1e6
                    results.append(estimator.update(sample.timestamp_us, state, q_cmd, valid=valid))
                with self.lock:
                    if generation == self._generation:
                        self.latest = tuple(results)
            except Exception:
                self.errors += 1
                self.invalidate()
                if self.errors == 1 or self.errors % 100 == 0:
                    log.exception('Free-space torque estimation failed; releasing GELLO force feedback')

    def close(self):
        self.invalidate()
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
            if self.thread.is_alive():
                log.warning('tau_free inference is still finishing; force feedback has been invalidated')

    def metadata(self):
        input_filters, horizon = [], []
        for arm, estimator in zip(self.arm_names, self.estimators):
            settings = self.settings[arm]
            if settings['source'] == 'urdf':
                horizon.append(1)
                input_filters.append({key: {'enabled': True, 'cutoff_hz': settings[setting]}
                    for key, setting in [('tau', 'measured_torque_cutoff_hz'),
                                         ('ddq', 'acceleration_cutoff_hz')]})
            else:
                horizon.append(estimator.predictor.metadata.horizon)
                input_filters.append(dict(estimator.predictor.metadata.dataloader_filters))
        return {'enabled': self.enabled, 'settings': self.settings,
                'source': self.source,
                'sources': {arm: settings['source'] for arm, settings in self.settings.items()},
                'urdf_residual_definition': 'lowpass(measured_tau) - RNEA(q, dq, lowpass(ddq))',
                'tau_ext_mean_window': {arm: settings['tau_ext_mean_window']
                                        for arm, settings in self.settings.items()},
                'urdf_mean_requires_full_window': True,
                'percent_reference_max_current_raw': XL330_MAX_CURRENT_RAW,
                'resolved_current_limit_raw': {arm: controller.limit.tolist()
                    for arm, controller in zip(self.arm_names, self.controllers)},
                'horizon': horizon,
                'input_filters': input_filters,
                'sample_rate_hz': self.sample_rate_hz, 'threshold_mode': 'abs(tau_ext) > threshold_nm',
                'delta_q_definition': 'causal q_cmd - measured q',
                'dropped_samples': self.dropped_samples, 'errors': self.errors}
