"""Exercise the actual SDK on a software serial link, never robot hardware."""

import os
import select
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gello_teleop import gello_hardware


@pytest.fixture
def serial_link():
    pty = pytest.importorskip('pty')
    pytest.importorskip('dynamixel_sdk')
    master, slave = pty.openpty()
    config = SimpleNamespace(port=os.ttyname(slave), baudrate=57600,
                             joint_ids=[1, 2], gripper_id=-1,
                             torque_joint_ids=[], read_timeout=.2)
    try:
        # Skip the constructor's initial request: the software motor responder
        # is started by each test after the serial link is available.
        with patch.object(gello_hardware.GelloReader, 'read', return_value=np.zeros(2)):
            reader = gello_hardware.GelloReader(config)
        try:
            yield reader, master
        finally:
            # GelloReader.close() sends torque writes; close only the PTY.
            reader.port.closePort()
    finally:
        os.close(master)
        os.close(slave)


def status_packet(packet_handler, motor_id, data, error=0):
    length = len(data) + 4
    packet = [255, 255, 253, 0, motor_id, length & 255, length >> 8, 85, error, *data]
    crc = packet_handler.updateCRC(0, packet, len(packet))
    return bytes([*packet, crc & 255, crc >> 8])


def test_serial_open_and_reopen_yield_without_changing_sdk_deadline(serial_link):
    from dynamixel_sdk import port_handler

    reader, _ = serial_link
    port = reader.port
    assert port.ser.timeout == .001
    assert port.ser.exclusive is True
    assert port.ser.write_timeout == reader.config.read_timeout
    latency_before = port_handler.LATENCY_TIMER
    port.setPacketTimeout(91)
    assert port.packet_timeout == pytest.approx(91 * 10 / 57600 * 1000 + 2 * latency_before + 2)
    assert port.setBaudRate(115200)
    assert port.ser.timeout == .001
    assert port_handler.LATENCY_TIMER == latency_before


def test_packet_deadline_uses_monotonic_milliseconds(serial_link, monkeypatch):
    reader, _ = serial_link
    clock = [10.]
    monkeypatch.setattr(gello_hardware.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(gello_hardware.time, 'time', lambda: (_ for _ in ()).throw(
        AssertionError('wall time must not be used for SDK deadlines')))
    reader.port.setPacketTimeoutMillis(20)
    assert reader.port.packet_start_time == 10_000
    clock[0] += .019
    assert not reader.port.isPacketTimeout()
    clock[0] += .002
    assert reader.port.isPacketTimeout()


def test_buffered_bytes_are_returned_without_waiting_for_requested_length(serial_link):
    reader, master = serial_link
    os.write(master, b'abc')
    deadline = time.monotonic() + 1.
    while reader.port.ser.in_waiting < 3:
        assert time.monotonic() < deadline
        time.sleep(.001)
    with patch.object(reader.port.ser, 'read', wraps=reader.port.ser.read) as read:
        assert reader.port.readPort(11) == b'abc'
    read.assert_called_once_with(3)


def test_missing_reply_waits_and_remains_a_receive_timeout(serial_link):
    reader, _ = serial_link
    reader.port.setPacketTimeoutMillis(30)
    started = time.monotonic()
    with patch.object(reader.port.ser, 'read', wraps=reader.port.ser.read) as read:
        data, result, error = reader.packet.readRx(reader.port, 1, 4)
    elapsed = time.monotonic() - started
    assert (data, result, error) == ([], -3001, 0)
    assert elapsed >= .029
    # The old timeout=0 transport spins thousands of times in this interval.
    # Scheduler delays may reduce the count, but cannot make it spin faster.
    assert read.call_count < 100


@pytest.mark.parametrize('reply,code', [('partial', -3002), ('bad_crc', -3002), ('missing', -3001)])
def test_checked_position_read_rejects_incomplete_corrupt_or_missing_reply(serial_link, reply, code):
    reader, master = serial_link
    packet = status_packet(reader.packet, 1, [1, 0, 0, 0])
    if reply == 'partial':
        os.write(master, packet[:8])
    elif reply == 'bad_crc':
        os.write(master, packet[:-1] + bytes([packet[-1] ^ 1]))
    reader.port.setPacketTimeoutMillis(20)
    with pytest.raises(RuntimeError, match=f'code={code}'):
        reader.group.rxPacket()
    assert reader.group.last_result is False


@pytest.mark.parametrize('failure,match', [
    (None, None),
    ('mismatch', 'Goal Current write/read mismatch'),
    ('device', 'device error'),
    ('missing', 'COMM_RX_TIMEOUT'),
    ('bad_crc', 'COMM_RX_CORRUPT'),
])
def test_actual_sdk_current_write_keeps_checked_readback(serial_link, failure, match):
    reader, master = serial_link
    stop = threading.Event()
    responder_errors = []

    def respond():
        buffer = bytearray()
        goals = {}
        try:
            while not stop.is_set():
                ready, _, _ = select.select([master], [], [], .01)
                if not ready:
                    continue
                buffer.extend(os.read(master, 1024))
                while len(buffer) >= 7:
                    packet_length = int.from_bytes(buffer[5:7], 'little') + 7
                    if len(buffer) < packet_length:
                        break
                    packet = bytes(buffer[:packet_length])
                    del buffer[:packet_length]
                    assert packet[:4] == b'\xff\xff\xfd\x00'
                    assert int.from_bytes(packet[8:10], 'little') == 102
                    assert int.from_bytes(packet[10:12], 'little') == 2
                    params = packet[12:-2]
                    if packet[7] == 0x83:
                        goals.update({params[i]: list(params[i+1:i+3])
                                      for i in range(0, len(params), 3)})
                    elif packet[7] == 0x82:
                        for motor_id in params:
                            if failure == 'missing' and motor_id == 2:
                                continue
                            data = [0, 0] if failure == 'mismatch' else goals[motor_id]
                            error = 7 if failure == 'device' and motor_id == 1 else 0
                            reply = status_packet(reader.packet, motor_id, data, error)
                            if failure == 'bad_crc' and motor_id == 1:
                                reply = reply[:-1] + bytes([reply[-1] ^ 1])
                            os.write(master, reply)
                    else:
                        raise AssertionError(f'unexpected instruction {packet[7]}')
        except Exception as exc:
            responder_errors.append(exc)

    worker = threading.Thread(target=respond, daemon=True)
    worker.start()
    try:
        if failure is None:
            reader.write_current_damping([-15, 34])
        else:
            with pytest.raises(RuntimeError, match=match):
                reader.write_current_damping([-15, 34])
    finally:
        stop.set()
        worker.join(timeout=1.)
    assert not worker.is_alive()
    assert responder_errors == []
