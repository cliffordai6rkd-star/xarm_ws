import socket
import struct
import threading
import time

import numpy as np
import pytest
import yaml

from xarm_stack.plot_dual_joint_current import (
    CurrentHistory, CurrentReport, DualCurrentWindow, RichCurrentReceiver,
    load_robot_ips, main, parse_current_report,
)


def packet(currents=None, *, size=587, axis=7, state=2, mode=1, error=0, warn=0):
    data = bytearray(size)
    struct.pack_into('>I', data, 0, size)
    data[4] = (mode << 4) | state
    data[89], data[90], data[146] = error, warn, axis
    if size >= 383:
        struct.pack_into('<7f', data, 355, *(currents if currents is not None else np.arange(7)/10))
    return bytes(data)


def test_currents_come_from_dedicated_ampere_field_with_signed_values():
    expected = [-.4, .5, 0., 1.25, -.1, .3, .7]
    raw = bytearray(packet(expected, error=11, warn=3))
    # The selectable effort field must never be interpreted as current.
    struct.pack_into('<7f', raw, 59, *([99.]*7))
    report = parse_current_report(raw, 12, received_s=42.)
    np.testing.assert_allclose(report.current_a, expected)
    assert (report.sequence, report.received_s, report.state, report.mode,
            report.error_code, report.warn_code) == (12, 42., 2, 1, 11, 3)


@pytest.mark.parametrize('raw, match', [
    (packet(size=245), '缺少独立电流字段'),
    (packet(axis=6), '需要七轴'),
    (packet([np.nan]*7), 'NaN/Inf'),
    (packet([np.inf]*7), 'NaN/Inf'),
    (packet()+b'x', '长度不一致'),
])
def test_invalid_reports_are_not_plotted_as_zero(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_current_report(raw, 0)


def test_robot_ips_resolve_relative_dataset_calibration_path(tmp_path):
    calibration = {'left': {'RobotConfig': {'robot_ip': '192.168.1.203'}},
                   'right': {'RobotConfig': {'robot_ip': '192.168.1.196'}}}
    (tmp_path/'cal.yaml').write_text(yaml.safe_dump(calibration))
    cfg = tmp_path/'dataset.yaml'
    cfg.write_text(yaml.safe_dump({'gello_config': 'cal.yaml',
                                  'arms': {'left': {'feedback_signal': 'torque'}}}))
    expected = dict(left='192.168.1.203', right='192.168.1.196')
    assert load_robot_ips(cfg) == expected
    assert load_robot_ips(tmp_path/'cal.yaml') == expected


def test_explicit_ips_work_without_calibration_or_gello_devices(tmp_path):
    cfg = tmp_path/'current.yaml'
    cfg.write_text(yaml.safe_dump({'gello_config': 'does-not-exist.yaml',
                                  'arms': {'left': {'robot_ip': '127.0.0.1'},
                                           'right': {'ip': '127.0.0.2'}}}))
    assert load_robot_ips(cfg) == dict(left='127.0.0.1', right='127.0.0.2')


def test_same_robot_cannot_fill_both_columns(tmp_path):
    cfg = tmp_path/'same.yaml'
    cfg.write_text(yaml.safe_dump({'arms': {side: {'robot_ip': '127.0.0.1'} for side in ('left', 'right')}}))
    with pytest.raises(ValueError, match='必须不同'):
        load_robot_ips(cfg)


def test_tcp_receive_handles_fragmented_and_coalesced_packets_without_sending_commands():
    receiver = RichCurrentReceiver('left', 'unused')
    client, server = socket.socketpair()
    client.settimeout(.01)
    errors = []
    def receive():
        try:
            receiver._receive(client)
        except OSError as exc:
            if not receiver.stop.is_set():
                errors.append(exc)
    thread = threading.Thread(target=receive)
    thread.start()
    try:
        first = packet([.1]*7)
        server.sendall(first[:2])
        time.sleep(.025)  # Partial header survives receive timeout.
        server.sendall(first[2:80])
        time.sleep(.025)  # Partial payload survives receive timeout.
        server.sendall(first[80:]+packet([.2]*7))
        deadline = time.monotonic()+1.
        while receiver.sequence < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        reports, latest, _, _ = receiver.drain()
        assert [report.sequence for report in reports] == [0, 1]
        np.testing.assert_allclose(latest.current_a, [.2]*7)
        server.settimeout(.025)
        with pytest.raises(socket.timeout):
            server.recv(1)  # No selector, mode, enable, or other outbound data.
        assert not errors
    finally:
        receiver.stop.set()
        client.shutdown(socket.SHUT_RDWR)
        thread.join(timeout=1.)
        client.close()
        server.close()


def test_history_is_bounded_in_time_and_breaks_line_after_disconnect():
    history = CurrentHistory(2.)
    history.append(0., np.zeros(7))
    history.append(1., np.ones(7))
    history.gap(2.)
    history.gap(2.1)
    history.append(3.5, np.full(7, 3.))
    t, values = history.arrays(3.5)
    np.testing.assert_allclose(t, [-1.5, 0.])
    assert np.isnan(values[0]).all()
    np.testing.assert_allclose(values[1], [3.]*7)
    t, values = history.arrays(10.)
    assert t.shape == (0,) and values.shape == (0, 7)


def test_window_places_left_and_right_currents_in_seven_rows(tmp_path):
    pytest.importorskip('matplotlib')
    window = DualCurrentWindow(dict(left='left-ip', right='right-ip'),
                               window_s=10., stale_s=1., headless=True)
    try:
        left = CurrentReport(1., 0, np.arange(7)/10, 2, 0, 0, 0)
        right = CurrentReport(1., 0, -np.arange(7)/10, 2, 0, 0, 0)
        window.ingest({'left': ([left], left, True, ''),
                       'right': ([right], right, True, '')}, now=1.1)
        assert window.axes.shape == (7, 2)
        for joint in range(7):
            assert window.lines['left'][joint].axes is window.axes[joint, 0]
            assert window.lines['right'][joint].axes is window.axes[joint, 1]
            np.testing.assert_allclose(window.lines['left'][joint].get_ydata(), [joint/10])
            np.testing.assert_allclose(window.lines['right'][joint].get_ydata(), [-joint/10])
        window.save(tmp_path/'currents.png')
        assert (tmp_path/'currents.png').stat().st_size > 1000
        window.ingest({'left': ([], left, True, ''), 'right': ([], right, False, 'connection lost')}, now=3.)
        assert all(label.get_text() == '-- A' for labels in window.values.values() for label in labels)
        assert 'STALE' in window.status['left'].get_text()
        assert 'DISCONNECTED' in window.status['right'].get_text()
        assert np.isnan(window.lines['right'][0].get_ydata()[-1])
    finally:
        window.close()


def test_demo_cli_saves_fourteen_plots_without_hardware_connections(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    def forbidden(*args, **kwargs):
        pytest.fail('demo must not open a hardware connection')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    fresh = []
    original_ingest = DualCurrentWindow.ingest
    def inspect_freshness(window, snapshots, now):
        original_ingest(window, snapshots, now)
        fresh.append(all(label.get_text() != '-- A'
                         for labels in window.values.values() for label in labels))
    monkeypatch.setattr(DualCurrentWindow, 'ingest', inspect_freshness)
    image = tmp_path/'demo.png'
    assert main(['--demo', '--headless', '--duration', '.25', '--save-plot', str(image)]) == 0
    assert image.is_file()
    assert fresh and all(fresh)


@pytest.mark.parametrize('args', [['--window-s', '0'], ['--update-hz', '-1'], ['--stale-s', 'nan'],
                                 ['--duration', '-1'], ['--ylim', '0'], ['--headless']])
def test_bad_cli_options_fail_before_connecting(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_check_config_does_not_connect(monkeypatch, tmp_path):
    cfg = tmp_path/'ips.yaml'
    cfg.write_text(yaml.safe_dump({'arms': {'left': {'robot_ip': '127.0.0.1'},
                                           'right': {'robot_ip': '127.0.0.2'}}}))
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: pytest.fail('check-config must not connect'))
    assert main(['-c', str(cfg), '--check-config']) == 0
