#!/usr/bin/env python3
"""双 xArm 七轴电流实时图：左臂一列、右臂一列，共七行。

Only receive TCP 30002 rich reports. No motion, enable, reset, or feedback
selector commands are sent, so this window can run beside dual_gello_collect.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import socket
import struct
import threading
import time

import numpy as np
import yaml

log = logging.getLogger('dual_xarm_current_plot')
SIDES = ('left', 'right')
REPORT_PORT = 30002
MAX_PACKET_BYTES = 16384
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'gello_teleop/config/xarm7_gello_dual_dataset.yaml'


def load_robot_ips(config_path):
    """Resolve IPs from a dataset config, calibration config, or current config."""
    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(raw, dict):
        raise ValueError('配置文件必须为 YAML mapping')
    arms = raw.get('arms', {})
    calibration = raw
    if raw.get('gello_config') and any(
            not (arms.get(side, {}).get('robot_ip') or arms.get(side, {}).get('ip')) for side in SIDES):
        reference = Path(raw['gello_config']).expanduser()
        if not reference.is_absolute():
            reference = path.parent / reference
        calibration = yaml.safe_load(reference.read_text(encoding='utf-8')) or {}
    result = {}
    for side in SIDES:
        endpoint = arms.get(side, {})
        robot = calibration.get(side, {}).get('RobotConfig', {})
        address = endpoint.get('robot_ip') or endpoint.get('ip') or robot.get('robot_ip')
        if not isinstance(address, str) or not address.strip():
            raise ValueError(f'未找到 {side} xArm 的 robot_ip')
        result[side] = address.strip()
    if result['left'] == result['right']:
        raise ValueError('左右 xArm 的 robot_ip 必须不同')
    return result


@dataclass(frozen=True)
class CurrentReport:
    received_s: float
    sequence: int
    current_a: np.ndarray
    state: int
    mode: int
    error_code: int
    warn_code: int


def parse_current_report(packet, sequence, received_s=None):
    """Decode the dedicated current field, independent of tau_or_i selection.

    UFACTORY TCP 30002: bytes 356..383 (1-based), seven little-endian FP32
    values in amperes. This is separate from the selectable effort field.
    https://docs.supportarticle.ufactory.cc/support_articles/developer/firmware/data-description-of-tcp-port.html
    """
    if len(packet) < 383:
        raise ValueError(f'详细状态报文仅 {len(packet)} 字节，缺少独立电流字段（需 >=383）')
    size = struct.unpack_from('>I', packet)[0]
    if size != len(packet):
        raise ValueError(f'状态报文长度不一致：header={size}, received={len(packet)}')
    if packet[146] != 7:
        raise ValueError(f'需要七轴 xArm，报文轴数为 {packet[146]}')
    current = np.asarray(struct.unpack_from('<7f', packet, 355), dtype=float)
    if not np.isfinite(current).all():
        raise ValueError('七轴电流反馈包含 NaN/Inf')
    state_mode = packet[4]
    return CurrentReport(time.monotonic() if received_s is None else float(received_s),
                         int(sequence), current, state_mode & 15, state_mode >> 4,
                         packet[89], packet[90])


class RichCurrentReceiver:
    """One receive thread per arm; reconnect without blocking the plot window."""

    def __init__(self, side, robot_ip, *, connect_timeout_s=3.):
        self.side, self.robot_ip = side, robot_ip
        self.connect_timeout_s = connect_timeout_s
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.pending = deque(maxlen=512)
        self.latest = None
        self.connected = False
        self.error = 'connecting'
        self.sequence = 0
        self.thread = None
        self.connection = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name=f'{self.side}-current-report', daemon=True)
        self.thread.start()

    def drain(self):
        with self.lock:
            reports = list(self.pending)
            self.pending.clear()
            return reports, self.latest, self.connected, self.error

    def _run(self):
        while not self.stop.is_set():
            try:
                with socket.create_connection((self.robot_ip, REPORT_PORT), timeout=self.connect_timeout_s) as connection:
                    connection.settimeout(.2)
                    with self.lock:
                        self.connection = connection
                        self.connected = True
                        self.error = 'waiting for first current report'
                    log.info('%s: connected to %s:%d (receive only)', self.side, self.robot_ip, REPORT_PORT)
                    self._receive(connection)
            except (OSError, ValueError) as exc:
                if not self.stop.is_set():
                    with self.lock:
                        self.error = str(exc)
                    log.warning('%s current report unavailable: %s; retrying', self.side, exc)
            finally:
                with self.lock:
                    self.connected = False
                    self.connection = None
            self.stop.wait(1.)

    def _receive(self, connection):
        buffer = bytearray()
        while not self.stop.is_set():
            try:
                chunk = connection.recv(4096)
            except socket.timeout:
                continue  # Preserve partial packets across receive timeouts.
            if not chunk:
                raise ConnectionError('current report connection closed')
            buffer.extend(chunk)
            while len(buffer) >= 4:
                size = struct.unpack_from('>I', buffer)[0]
                if not 4 <= size <= MAX_PACKET_BYTES:
                    raise ValueError(f'无效的电流报文长度：{size}')
                if len(buffer) < size:
                    break
                packet = bytes(buffer[:size])
                del buffer[:size]
                report = parse_current_report(packet, self.sequence)
                with self.lock:
                    self.sequence += 1
                    self.latest = report
                    self.pending.append(report)
                    self.error = ''

    def close(self):
        self.stop.set()
        with self.lock:
            connection = self.connection
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=self.connect_timeout_s+.5)
        # No commands are transmitted when connecting or closing this socket.


class DemoCurrentReceiver:
    def __init__(self, side):
        self.side = side
        self.started = time.monotonic()
        self.latest = None
        self.sequence = 0

    def start(self):
        pass

    def drain(self):
        now = time.monotonic()
        if self.latest is not None and now-self.latest.received_s < .2:
            return [], self.latest, True, ''
        t = now-self.started
        phase = np.arange(7)*.55 + (0.8 if self.side == 'right' else 0.)
        values = (.08+.04*np.arange(7))*np.sin(t*(.7+.08*np.arange(7))+phase)
        values += np.asarray([.06, .22, .08, .38, .04, .12, .02])
        self.latest = CurrentReport(now, self.sequence, values, 2, 0, 0, 0)
        self.sequence += 1
        return [self.latest], self.latest, True, ''

    def close(self):
        pass


class CurrentHistory:
    """Bounded sliding history; a missing sample is a gap, never a zero."""

    def __init__(self, window_s):
        self.window_s = window_s
        self.rows = deque(maxlen=4096)
        self.in_gap = False

    def append(self, received_s, current_a):
        values = np.asarray(current_a, float)
        if values.shape != (7,):
            raise ValueError('电流数组必须按 J1..J7 包含七个值')
        if self.rows and received_s <= self.rows[-1][0]:
            return
        self.rows.append((float(received_s), values.copy()))
        self.in_gap = False
        self.trim(received_s)

    def gap(self, now):
        if not self.in_gap:
            self.append(now, np.full(7, np.nan))
            self.in_gap = True

    def trim(self, now):
        while self.rows and self.rows[0][0] < now-self.window_s:
            self.rows.popleft()

    def arrays(self, now):
        self.trim(now)
        if not self.rows:
            return np.empty(0), np.empty((0, 7))
        return (np.asarray([row[0] for row in self.rows])-now,
                np.stack([row[1] for row in self.rows]))


class DualCurrentWindow:
    def __init__(self, addresses, *, window_s=30., stale_s=1.5, ylim=None, headless=False, demo=False):
        try:
            import matplotlib
            if headless:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError('需要绘图库：python -m pip install "matplotlib>=3.7"') from exc
        if not headless and 'agg' == str(matplotlib.get_backend()).lower():
            raise RuntimeError('当前绘图库没有可用 GUI；请在桌面终端启动，或用 --headless --duration 5')
        self.plt, self.window_s, self.stale_s = plt, window_s, stale_s
        self.headless, self.ylim = headless, ylim
        self.history = {side: CurrentHistory(window_s) for side in SIDES}
        self.figure, self.axes = plt.subplots(7, 2, sharex=True, figsize=(13, 9), squeeze=False)
        self.figure.subplots_adjust(left=.065, right=.985, bottom=.07, top=.87, hspace=.28, wspace=.18)
        self.figure.suptitle('Dual xArm joint currents' + ('  [DEMO]' if demo else ''), fontsize=14, y=.985)
        self.figure.text(.5, .013, 'q / Esc: close window    |    time axis: seconds from now', ha='center', fontsize=9)
        self.lines, self.values, self.status = {}, {}, {}
        for column, side in enumerate(SIDES):
            color = 'tab:blue' if side == 'left' else 'tab:orange'
            self.status[side] = self.figure.text(.065 if column == 0 else .562, .909,
                                                f'{side.upper()} {addresses[side]} | waiting', fontsize=9)
            self.lines[side], self.values[side] = [], []
            self.axes[0, column].set_title(f'{side.upper()} xArm — J1 to J7', fontsize=11, color=color)
            for joint in range(7):
                axis = self.axes[joint, column]
                line, = axis.plot([], [], color=color, linewidth=1.25)
                axis.set_ylabel(f'J{joint+1}\nCurrent [A]', fontsize=9)
                axis.set_xlim(-window_s, 0.)
                axis.set_ylim(-(ylim or .5), ylim or .5)
                axis.grid(True, alpha=.25)
                axis.tick_params(labelsize=8, pad=2)
                self.lines[side].append(line)
                self.values[side].append(axis.text(.985, .88, '-- A', ha='right', va='top',
                                                   transform=axis.transAxes, fontsize=9, color=color,
                                                   bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1.5)))
            self.axes[-1, column].set_xlabel('Time [s]', fontsize=9)
        self.addresses = addresses
        self.figure.canvas.mpl_connect('key_press_event', self._key)
        manager = self.figure.canvas.manager
        if hasattr(manager, 'set_window_title'):
            manager.set_window_title('Dual xArm: 14 joint currents')
        if not headless:
            plt.show(block=False)

    def _key(self, event):
        if event.key in {'q', 'Q', 'escape'}:
            self.plt.close(self.figure)

    def ingest(self, snapshots, now):
        for side, (reports, latest, connected, error) in snapshots.items():
            for report in reports:
                self.history[side].append(report.received_s, report.current_a)
            fresh = connected and latest is not None and 0 <= now-latest.received_s <= self.stale_s
            if not fresh:
                self.history[side].gap(now)
            t, values = self.history[side].arrays(now)
            for joint, line in enumerate(self.lines[side]):
                line.set_data(t, values[:, joint])
                self.values[side][joint].set_text(f'{latest.current_a[joint]:+.3f} A' if fresh else '-- A')
                if self.ylim is None:
                    finite = values[:, joint][np.isfinite(values[:, joint])]
                    bound = max(.1, float(np.max(np.abs(finite)))*1.2) if finite.size else .5
                    self.axes[joint, SIDES.index(side)].set_ylim(-bound, bound)
            if fresh:
                message = f'current | age {now-latest.received_s:.2f}s | state {latest.state} mode {latest.mode}'
                if latest.error_code or latest.warn_code:
                    message += f' | error {latest.error_code} warn {latest.warn_code}'
            else:
                message = 'STALE / waiting' if connected else 'DISCONNECTED'
                if error:
                    message += ' | ' + (error[:42] if error.isascii() else 'see terminal for details')
            self.status[side].set_text(f'{side.upper()} {self.addresses[side]} | {message}')
            self.status[side].set_color('black' if fresh else 'tab:red')

    def refresh(self, wait_s):
        if self.headless:
            time.sleep(max(0., wait_s))
        else:
            self.figure.canvas.draw_idle()
            self.plt.pause(max(.001, wait_s))

    def is_open(self):
        return self.plt.fignum_exists(self.figure.number)

    def save(self, path):
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(destination, dpi=140)
        print(f'曲线图已保存：{destination}', flush=True)

    def close(self):
        self.plt.close(self.figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--window-s', type=float, default=30., help='显示最近多少秒，默认 30')
    parser.add_argument('--update-hz', type=float, default=10., help='窗口刷新频率；不会提高硬件报文频率')
    parser.add_argument('--stale-s', type=float, default=1.5, help='超过此秒数未收到报文时显示断线缺口')
    parser.add_argument('--connect-timeout-s', type=float, default=3.)
    parser.add_argument('--ylim', type=float, help='全部子图的固定对称范围（A），默认逐轴自动缩放')
    parser.add_argument('--duration', type=float, default=0., help='运行秒数；0 为持续运行')
    parser.add_argument('--demo', action='store_true', help='使用模拟电流，不连接硬件')
    parser.add_argument('--headless', action='store_true', help='不打开 GUI，须设置 --duration')
    parser.add_argument('--save-plot', type=Path, help='退出时保存当前曲线 PNG')
    parser.add_argument('--check-config', action='store_true', help='仅打印两侧 IP，不连接硬件')
    args = parser.parse_args(argv)
    for name in ('window_s', 'update_hz', 'stale_s', 'connect_timeout_s'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} 必须为正数')
    if not np.isfinite(args.duration) or args.duration < 0:
        parser.error('--duration 必须 >=0')
    if args.ylim is not None and (not np.isfinite(args.ylim) or args.ylim <= 0):
        parser.error('--ylim 必须为正数（A）')
    if args.headless and not args.duration:
        parser.error('--headless 须指定正数 --duration')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    receivers, window = {}, None
    snapshots = {}
    exit_code = 0
    try:
        addresses = dict.fromkeys(SIDES, 'demo') if args.demo else load_robot_ips(args.config)
        if args.check_config:
            print(json.dumps(dict(arms=addresses, port=REPORT_PORT, layout='7 rows x 2 columns'), indent=2))
            return 0
        window = DualCurrentWindow(addresses, window_s=args.window_s, stale_s=args.stale_s,
                                   ylim=args.ylim, headless=args.headless, demo=args.demo)
        for side in SIDES:
            receivers[side] = (DemoCurrentReceiver(side) if args.demo else
                               RichCurrentReceiver(side, addresses[side], connect_timeout_s=args.connect_timeout_s))
            receivers[side].start()
        print('左列 left J1..J7，右列 right J1..J7；单位 A；窗口 q/Esc 或终端 Ctrl+C 退出。', flush=True)
        if not args.demo:
            print('只接收 TCP 30002 详细状态报文；独立电流字段通常为 5 Hz。可与遥操/数采同时运行。', flush=True)
        started = time.monotonic()
        period = 1/args.update_hz
        while window.is_open():
            tick_started = time.monotonic()
            snapshots = {side: receiver.drain() for side, receiver in receivers.items()}
            # A receiver can produce a sample while snapshots are collected.
            # Check age after draining so that a new sample is never in the future.
            now = time.monotonic()
            window.ingest(snapshots, now)
            if args.duration and now-started >= args.duration:
                break
            window.refresh(max(.001, period-(time.monotonic()-tick_started)))
        if args.headless and any(snapshot[1] is None for snapshot in snapshots.values()):
            log.error('运行期间未收到两侧完整电流报文')
            exit_code = 1
    except KeyboardInterrupt:
        exit_code = 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        log.error('%s', exc)
        exit_code = 1
    finally:
        for receiver in receivers.values():
            receiver.close()
        if window is not None:
            try:
                if args.save_plot:
                    window.save(args.save_plot)
            finally:
                window.close()
        for side, (_, latest, _, _) in snapshots.items():
            if latest is not None:
                print(f'{side}: received {latest.sequence+1} reports; J1..J7 [A] = '
                      f'{np.round(latest.current_a, 4).tolist()}', flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
