import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gello_teleop.gello_hardware import GelloReader


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
