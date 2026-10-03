import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gello_teleop.gello_hardware import GelloReader
from gello_teleop.diagnose_gello_serial import packet_evidence


def test_alignment_reports_disconnected_port_and_stops_writes():
    termios = pytest.importorskip('termios')
    reader = GelloReader.__new__(GelloReader)
    reader.config = SimpleNamespace(port='/dev/serial/by-id/left-gello', torque_joint_ids=[])
    reader.ids = [1, 2, 3]
    reader.port = object()
    reader.lock = threading.Lock()
    reader.packet = SimpleNamespace(write1ByteTxRx=Mock(side_effect=termios.error(5, 'Input/output error')))
    with pytest.raises(ConnectionError, match=r'left-gello.*ID=1.*register=64') as error:
        reader.prepare_alignment()
    assert isinstance(error.value.__cause__, termios.error)
    reader.packet.write1ByteTxRx.assert_called_once()


def test_position_read_reports_serial_path_without_replacing_data_with_zero():
    termios = pytest.importorskip('termios')
    reader = GelloReader.__new__(GelloReader)
    reader.config = SimpleNamespace(port='/dev/serial/by-id/right-gello')
    reader.closed = False
    reader.lock = threading.Lock()
    reader.group = Mock()
    reader.group.txRxPacket.side_effect = termios.error(5, 'Input/output error')
    with pytest.raises(ConnectionError, match='right-gello.*读取位置'):
        reader.read()


@pytest.mark.parametrize('packet, code, reason, expected', [
    ([], -3001, 'no_response', None),
    ([255, 255, 253, 0, 4, 8, 0, 85, 0, 1, 2], -3002,
     'incomplete_or_malformed_packet', 15),
    ([255, 255, 253, 0, 4, 8, 0, 85, 0, 1, 2, 3, 4, 0, 0], -3002,
     'crc_or_packet_corruption', 15),
])
def test_diagnostic_distinguishes_empty_partial_and_complete_failed_packets(packet, code, reason, expected):
    evidence = packet_evidence(packet, code)
    assert evidence['reason'] == reason
    assert evidence['expected_bytes'] == expected
    assert evidence['received_bytes'] == len(packet)
    assert evidence['packet_hex'] == bytes(packet).hex()


def test_slow_leader_skips_expired_ticks_without_extra_idle_before_next_request(monkeypatch):
    import numpy as np
    from gello_teleop.dual_gello_collect import _LeaderProducer

    clock = [0.]
    stop = threading.Event()
    waits = []
    def read():
        clock[0] += .12
        return np.zeros(7)
    def wait(delay):
        waits.append(delay)
        stop.set()
    monkeypatch.setattr('gello_teleop.dual_gello_collect.time.monotonic', lambda: clock[0])
    monkeypatch.setattr(stop, 'wait', wait)
    cfg = SimpleNamespace(port='fake', gripper_id=-1)
    mapper = SimpleNamespace(config=cfg, n=7, target=lambda raw: (raw, raw))
    producer = _LeaderProducer(SimpleNamespace(read=read), mapper, .05, stop)
    producer._run()
    assert producer.skipped_ticks == 2
    assert waits == pytest.approx([0.])
    assert producer.sequence == 1
    assert producer.last_io_s == pytest.approx(.12)


def test_failed_leader_reports_failed_transaction_duration_and_stops_damping(monkeypatch):
    from gello_teleop.dual_gello_collect import _LeaderProducer

    clock = [0.]
    stop = threading.Event()
    def read():
        clock[0] += .06
        raise RuntimeError('ID4 incomplete status packet')
    monkeypatch.setattr('gello_teleop.dual_gello_collect.time.monotonic', lambda: clock[0])
    cfg = SimpleNamespace(port='fake', gripper_id=-1)
    reader = SimpleNamespace(read=read, disable_current_damping=Mock())
    producer = _LeaderProducer(reader, SimpleNamespace(config=cfg), .05, stop, damping_config=cfg)
    producer._run()
    assert stop.is_set()
    assert producer.latest is None
    assert producer.last_read_s == pytest.approx(.06)
    assert producer.last_io_s == pytest.approx(.06)
    assert producer.errors == 1
    reader.disable_current_damping.assert_called_once()


def test_bus_diagnostic_reads_and_closes_serial_without_motor_writes(monkeypatch):
    import numpy as np
    from gello_teleop import diagnose_gello_serial as diagnostic

    reader = SimpleNamespace(
        ids=list(range(1, 9)), read=Mock(return_value=np.zeros(8)),
        packet=SimpleNamespace(rxPacket=Mock(), readRx=Mock()),
        current_feedback=SimpleNamespace(txRxPacket=Mock(return_value=0)),
        port=SimpleNamespace(closePort=Mock()),
        close=Mock(side_effect=AssertionError('close() changes motor torque')),
        current_writer=SimpleNamespace(txPacket=Mock()),
    )
    monkeypatch.setattr(diagnostic, 'GelloReader', lambda cfg: reader)
    leader = dict(port='/nonexistent/right-gello', baudrate=57600, joint_ids=list(range(1, 8)))
    result = diagnostic.inspect_bus(leader, samples=2, rate_hz=1000)
    assert result['errors'] == []
    assert result['full_cycle_wire_min_ms'] == pytest.approx(50.17)
    assert reader.read.call_count == 2
    assert reader.current_feedback.txRxPacket.call_count == 2
    reader.port.closePort.assert_called_once()
    reader.close.assert_not_called()
    reader.current_writer.txPacket.assert_not_called()
