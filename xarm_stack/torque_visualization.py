"""Seven rows of measured / URDF torque; two columns per selected arm.

Demo: python -m xarm_stack.torque_visualization --demo --headless --duration 3
      --save-plot /tmp/dual_torque.png
"""

from __future__ import annotations

import argparse
from collections import deque
import logging
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time

import numpy as np

from xarm_stack.momentum import PinocchioMomentumModel

log = logging.getLogger(__name__)
SIDES = ('left', 'right')
DEFAULT_URDF = Path(__file__).resolve().parents[1] / 'gello_teleop/models/xarm7_dynamics.urdf'


class FirstOrderLowPass:
    """Seven-channel exponential low-pass, using acquisition time intervals."""

    def __init__(self, cutoff_hz=3., max_gap_s=.15, *, setting_name='measured_torque_cutoff_hz'):
        cutoff = np.asarray(cutoff_hz, dtype=float)
        if cutoff.ndim == 0:
            cutoff = np.full(7, float(cutoff))
        if cutoff.shape != (7,) or not np.isfinite(cutoff).all() or np.any(cutoff <= 0):
            raise ValueError(f'{setting_name} must be a positive scalar or seven-vector')
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('max_gap_s must be positive')
        self.rate, self.max_gap_s = 2*np.pi*cutoff, max_gap_s
        self.reset()

    def reset(self):
        self.timestamp_s = self.value = None

    def update(self, timestamp_s, value, valid=True):
        value = np.asarray(value, dtype=float)
        if (not valid or not np.isfinite(timestamp_s) or value.shape != (7,)
                or not np.isfinite(value).all()):
            self.reset()
            return np.full(7, np.nan)
        if self.timestamp_s is not None and timestamp_s == self.timestamp_s:
            return self.value.copy()
        dt = None if self.timestamp_s is None else timestamp_s-self.timestamp_s
        if dt is None or dt <= 0 or dt > self.max_gap_s:
            self.value = value.copy()
        else:
            alpha = -np.expm1(-self.rate*dt)
            self.value += alpha*(value-self.value)
        self.timestamp_s = timestamp_s
        return self.value.copy()


def plot_config(raw, config_path, active_sides):
    cfg = dict(raw)
    for key, default in [('window_s', 30.), ('update_hz', 10.), ('stale_s', .15)]:
        value = float(cfg.get(key, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'torque_visualization.{key} must be positive')
        cfg[key] = value
    cfg['active_sides'] = tuple(active_sides)
    cfg['sides'] = {}
    for side in active_sides:
        settings = dict(raw)
        settings.update(raw.get('sides', {}).get(side, {}))
        path = Path(settings.get('urdf_path', DEFAULT_URDF)).expanduser()
        if not path.is_absolute():
            path = Path(config_path).resolve().parent / path
        if not path.is_file():
            raise ValueError(f'{side} torque URDF does not exist: {path}')
        settings['urdf_path'] = str(path.resolve())
        # Validate both cutoffs before any process or hardware starts.
        for key in ('measured_torque_cutoff_hz', 'acceleration_cutoff_hz'):
            settings.setdefault(key, 3.)
            FirstOrderLowPass(settings[key], cfg['stale_s'], setting_name=key)
        cfg['sides'][side] = settings
    return cfg


def make_models(cfg):
    models = {}
    for side, settings in cfg['sides'].items():
        model = PinocchioMomentumModel(settings['urdf_path'], settings.get('gravity_m_s2', (0., 0., -9.81)))
        payload = settings.get('payload', {})
        if payload and payload.get('source') != 'controller':
            model.set_payload(payload)
        log.info('%s torque model: %s; moving mass %.4f kg', side, settings['urdf_path'],
                 sum(inertia.mass for inertia in model.model.inertias[1:]))
        models[side] = model
    return models


class TorqueHistory:
    def __init__(self, model, stale_s, window_s, measured_torque_cutoff_hz=3., acceleration_cutoff_hz=3.):
        self.model, self.stale_s, self.window_s = model, stale_s, window_s
        self.measured_filter = FirstOrderLowPass(measured_torque_cutoff_hz, stale_s)
        self.acceleration_filter = FirstOrderLowPass(acceleration_cutoff_hz, stale_s,
                                                    setting_name='acceleration_cutoff_hz')
        self.rows = deque(maxlen=20000)
        self.last_timestamp = None
        self.in_gap = False
        self.status = 'waiting for state'
        self.measured_status = 'waiting for torque'

    def ingest(self, sample, now):
        timestamp = sample['timestamp_s']
        fresh = np.isfinite(timestamp) and 0 <= now-timestamp <= self.stale_s
        if not fresh:
            self.gap(now)
            return
        if timestamp == self.last_timestamp:
            return
        if self.last_timestamp is not None and timestamp-self.last_timestamp > self.stale_s:
            self.gap(self.last_timestamp+self.stale_s)
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            self.gap(now)
            self.last_timestamp = None
            return
        if 'payload' in sample and self.model.set_payload(sample['payload']):
            log.info('Torque model flange payload: %s', sample['payload'])
        q, dq = np.asarray(sample['q'], dtype=float), np.asarray(sample['dq'], dtype=float)
        ddq = np.asarray(sample.get('ddq', np.full(7, np.nan)), dtype=float)
        measured = np.asarray(sample.get('torque', np.full(7, np.nan)), dtype=float)
        measured = self.measured_filter.update(timestamp, measured, sample.get('torque_valid', False))
        if not np.isfinite(measured).all():
            self.measured_status = 'torque unavailable'
        else:
            self.measured_status = 'xArm measured'
        gravity = np.full(7, np.nan)
        valid_motion = (sample['q_valid'] and q.shape == (7,) and np.isfinite(q).all()
                        and sample['dq_valid'] and dq.shape == (7,) and np.isfinite(dq).all())
        ddq = self.acceleration_filter.update(timestamp, ddq,
                                             valid_motion and sample.get('ddq_valid', False))
        if sample['q_valid'] and q.shape == (7,) and np.isfinite(q).all():
            gravity = self.model.gravity_torque(q)
        model = np.full(7, np.nan)
        if not sample['q_valid'] or q.shape != (7,) or not np.isfinite(q).all():
            self.status = 'invalid position'
        elif not sample['dq_valid'] or dq.shape != (7,) or not np.isfinite(dq).all():
            self.status = 'velocity unavailable'
        elif not sample.get('ddq_valid', False) or ddq.shape != (7,) or not np.isfinite(ddq).all():
            self.status = 'acceleration unavailable'
        else:
            model = self.model.rnea(q, dq, ddq)
            self.status = 'RNEA' if np.isfinite(model).all() else 'invalid model output'
        self.rows.append((timestamp, model, measured.copy(), gravity))
        self.last_timestamp, self.in_gap = timestamp, False

    def gap(self, now):
        if not self.in_gap:
            self.rows.append((now, np.full(7, np.nan), np.full(7, np.nan), np.full(7, np.nan)))
            self.measured_filter.reset()
            self.acceleration_filter.reset()
            self.in_gap = True
            self.status = 'state unavailable / stale'
            self.measured_status = 'state unavailable / stale'

    def arrays(self, now):
        if self.last_timestamp is None or now-self.last_timestamp > self.stale_s:
            self.gap(now)
        while self.rows and self.rows[0][0] < now-self.window_s:
            self.rows.popleft()
        if not self.rows:
            return np.empty(0), np.empty((0, 7)), np.empty((0, 7)), np.empty((0, 7))
        return (np.asarray([row[0] for row in self.rows])-now,
                np.stack([row[1] for row in self.rows]), np.stack([row[2] for row in self.rows]),
                np.stack([row[3] for row in self.rows]))


class TorqueWindow:
    def __init__(self, cfg, models, *, headless=False):
        import matplotlib
        if headless:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if not headless and str(matplotlib.get_backend()).lower() == 'agg':
            raise RuntimeError('Torque plot needs a desktop GUI; use --no-torque-plot on a headless collector')
        self.plt, self.cfg = plt, cfg
        self.lock = threading.Lock()
        self.history = {side: TorqueHistory(model, cfg['stale_s'], cfg['window_s'],
                                           cfg['sides'][side]['measured_torque_cutoff_hz'],
                                           cfg['sides'][side]['acceleration_cutoff_hz'])
                        for side, model in models.items()}
        self.layout_sides = cfg['active_sides']
        columns = 2*len(self.layout_sides)
        self.figure, self.axes = plt.subplots(7, columns, figsize=(8.5*len(self.layout_sides), 10),
                                             sharex=True, squeeze=False)
        self.figure.canvas.manager.set_window_title('xArm7 measured / URDF torque')
        self.figure.suptitle('xArm7 measured / theoretical torque [Nm]')
        self.lines, self.labels = {}, {}
        for side_index, side in enumerate(self.layout_sides):
            for kind in range(2):
                column = 2*side_index+kind
                title = 'xArm measured' if kind == 0 else 'URDF theoretical'
                self.axes[0, column].set_title(f'{side.title()} | {title}', fontsize=10)
                for joint in range(7):
                    axis = self.axes[joint, column]
                    self.lines[joint, column], = axis.plot([], [], color='#1976d2' if kind == 0 else '#e67e22', lw=1.2)
                    self.labels[joint, column] = axis.text(.98, .85, '-- Nm', transform=axis.transAxes,
                                                         ha='right', fontsize=8)
                    axis.grid(alpha=.25)
                    axis.set_ylabel(f'J{joint+1} [Nm]', fontsize=8)
                    axis.set_xlim(-cfg['window_s'], 0.)
                    axis.set_ylim(-1., 1.)
                self.axes[-1, column].set_xlabel('Time [s, relative to now]')
        self.figure.tight_layout(rect=(0, 0, 1, .96))
        self.figure.canvas.mpl_connect('key_press_event', self._key)
        if not headless:
            plt.show(block=False)

    def _key(self, event):
        if event.key in {'q', 'Q', 'escape'}:
            self.close()

    def ingest(self, snapshots, now):
        with self.lock:
            for side, sample in snapshots.items():
                if side in self.history:
                    self.history[side].ingest(sample, now)

    def draw(self, now):
        for side_index, side in enumerate(self.layout_sides):
            if side not in self.history:
                continue
            with self.lock:
                t, model, measured, gravity = self.history[side].arrays(now)
            for joint in range(7):
                finite = np.concatenate((model[:, joint], measured[:, joint]))
                finite = finite[np.isfinite(finite)]
                bound = max(.1, float(np.max(np.abs(finite)))*1.2) if finite.size else 1.
                for kind, values in enumerate((measured, model)):
                    column = 2*side_index+kind
                    self.lines[joint, column].set_data(t, values[:, joint])
                    latest = values[-1, joint] if len(values) else np.nan
                    self.labels[joint, column].set_text(f'{latest:+.3f} Nm' if np.isfinite(latest) else '-- Nm')
                    self.axes[joint, column].set_ylim(-bound, bound)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def alive(self):
        return self.plt.fignum_exists(self.figure.number)

    def save(self, path):
        self.figure.savefig(path, dpi=140)

    def close(self):
        self.plt.close(self.figure)


def _plot_worker(cfg, samples, stop, ready):
    window = None
    consumer = None
    try:
        window = TorqueWindow(cfg, make_models(cfg))
        def consume():
            # Dynamics continue at feedback frequency while the GUI renders
            # its axes. Only history copies share a short lock with drawing.
            try:
                while not stop.is_set():
                    try:
                        snapshot = samples.get(timeout=.05)
                    except queue.Empty:
                        continue
                    window.ingest(snapshot, time.time())
            except Exception:
                log.exception('Torque sample consumer stopped')
                stop.set()
        consumer = threading.Thread(target=consume, name='torque-model', daemon=True)
        consumer.start()
        ready.send(None)
        next_draw = time.monotonic()
        while not stop.is_set() and window.alive():
            now = time.monotonic()
            if now >= next_draw:
                window.draw(time.time())
                next_draw = now + 1/cfg['update_hz']
            stop.wait(.005)
    except Exception as exc:
        try:
            ready.send(str(exc))
        except (BrokenPipeError, OSError):
            pass
        log.exception('Torque visualization stopped')
    finally:
        stop.set()
        if consumer is not None:
            consumer.join(timeout=1.)
        ready.close()
        if window is not None:
            window.close()


class TorqueVisualizer:
    """A bounded, nonblocking handoff; no robot or SDK calls in this component."""

    def __init__(self, cfg):
        self.cfg, self.process = cfg, None
        self.samples = self.stop = None
        self.dropped_samples = 0
        self._reported_exit = False
        self.payload_sources = {}

    @classmethod
    def from_config(cls, raw, config_path, active_sides):
        if not raw.get('enabled', False):
            return None
        return cls(plot_config(raw, config_path, active_sides))

    def bind_payload_source(self, side, arm):
        self.payload_sources[side] = arm

    def start(self):
        context = mp.get_context('spawn')
        self.samples, self.stop = context.Queue(maxsize=256), context.Event()
        parent, child = context.Pipe(duplex=False)
        self.process = context.Process(target=_plot_worker, args=(self.cfg, self.samples, self.stop, child),
                                       name='xarm-torque-plot', daemon=True)
        try:
            self.process.start()
            child.close()
            if not parent.poll(15.):
                raise RuntimeError('Torque plot startup timed out')
            error = parent.recv()
            if error is not None:
                raise RuntimeError(error)
        except BaseException:
            self.close()
            raise
        finally:
            parent.close()
            child.close()

    def publish(self, sides, states):
        if self.process is None or not self.process.is_alive():
            if not self._reported_exit:
                log.info('Torque window closed; collection continues')
                self._reported_exit = True
            return
        snapshot = {side: dict(timestamp_s=state.timestamp_us/1e6,
                               q=np.asarray(state.q).copy(), dq=np.asarray(state.dq).copy(),
                               q_valid=state.q_valid, dq_valid=state.dq_valid,
                               ddq=np.asarray(state.ddq).copy(),
                               ddq_valid=getattr(state, 'ddq_valid', state.dq_valid),
                               torque=np.asarray(state.torque).copy(), torque_valid=state.torque_valid)
                    for side, state in zip(sides, states)}
        for side, sample in snapshot.items():
            settings = self.cfg.get('sides', {}).get(side, {})
            if settings.get('payload', {}).get('source') == 'controller':
                source = self.payload_sources.get(side)
                sample['payload'] = getattr(source, 'reported_tcp_payload', None)
        try:
            self.samples.put_nowait(snapshot)
        except queue.Full:
            self.dropped_samples += 1

    def close(self):
        if self.stop is not None:
            self.stop.set()
        if self.process is not None:
            if self.process.pid is not None:
                self.process.join(timeout=2.)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1.)
            self.process = None
        if self.samples is not None:
            self.samples.cancel_join_thread()
            self.samples.close()
            self.samples = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', required=True, help='Synthetic motion; never connect hardware')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--save-plot')
    arms = parser.add_mutually_exclusive_group()
    arms.add_argument('--left', dest='active_sides', action='store_const', const=('left',))
    arms.add_argument('--right', dest='active_sides', action='store_const', const=('right',))
    parser.set_defaults(active_sides=SIDES)
    args = parser.parse_args(argv)
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be positive')
    cfg = plot_config({'window_s': min(30., args.duration)}, __file__, args.active_sides)
    window = TorqueWindow(cfg, make_models(cfg), headless=args.headless)
    start = time.time()
    next_draw = 0.
    try:
        # Analytic synthetic trajectory; RNEA supplies synthetic measured effort.
        for t in np.arange(0., args.duration, .01):
            snapshots = {}
            for side in args.active_sides:
                model = window.history[side].model
                phase = np.arange(7)*.4 + (0.7 if side == 'right' else 0.)
                q = .15*np.sin(1.5*t+phase) + np.array([0., -.4, 0., .6, 0., .4, 0.])
                dq = .225*np.cos(1.5*t+phase)
                ddq = -.3375*np.sin(1.5*t+phase)
                torque = model.rnea(q, dq, ddq)
                snapshots[side] = dict(timestamp_s=start+t, q=q, dq=dq, q_valid=True, dq_valid=True,
                                       ddq=ddq, ddq_valid=True,
                                       torque=torque, torque_valid=True)
            window.ingest(snapshots, start+t)
            if t >= next_draw:
                window.draw(start+t)
                next_draw = t+1/cfg['update_hz']
            if not window.alive():
                break
            if not args.headless:
                time.sleep(max(0., start+t+.01-time.time()))
        window.draw(start+t)
        if args.save_plot:
            window.save(args.save_plot)
    finally:
        window.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
