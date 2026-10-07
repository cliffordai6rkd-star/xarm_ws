"""Model verification failures never enable motors or assume a servo model."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from gello_teleop import gello_hardware


@pytest.fixture
def reader(monkeypatch):
    reader = gello_hardware.GelloReader.__new__(gello_hardware.GelloReader)
    reader.config = SimpleNamespace(port='/dev/serial/by-id/test-gello', joint_ids=[3])
    reader.port = object()
    reader.packet = SimpleNamespace(ping=Mock())
    reader.lock = threading.Lock()
    reader.closed = False
    reader.model_numbers = {}
    reader._write = Mock()
    reader._read_register = Mock()
    monkeypatch.setattr(gello_hardware.time, 'sleep', Mock())
    return reader


def current_config():
    return SimpleNamespace(damping_mode='current', damping_gain=[8.],
                           damping_current_limit=[20], damping_watchdog_ms=500)


@pytest.mark.parametrize('failures', [[-3001], [-3002], [-3001, -3002]])
def test_transient_ping_timeout_or_corruption_retries_then_verifies_model(reader, failures):
    reader.packet.ping.side_effect = [(0, code, 0) for code in failures] + [(1200, 0, 0)]
    capabilities = reader.detect_capabilities()
    assert capabilities[3]['model_number'] == 1200
    assert capabilities[3]['model_name'] == 'XL330-M288'
    assert capabilities[3]['current_control'] is True
    assert 'error' not in capabilities[3]
    assert reader.model_numbers[3] == 1200
    assert reader.packet.ping.call_args_list == [call(reader.port, 3)] * (len(failures) + 1)
    assert gello_hardware.time.sleep.call_args_list == [call(.01)] * len(failures)
    reader._write.assert_not_called()


def test_current_damping_can_start_after_a_transient_identity_failure(reader):
    reader.packet.ping.side_effect = [(0, -3001, 0), (1200, 0, 0)]
    metadata = reader.enable_current_damping(current_config())
    assert metadata['enabled'] is True
    assert metadata['models'][3]['model_number'] == 1200
    assert metadata['models'][3]['current_control'] is True
    reader._write.assert_any_call(3, 11, 0)
    reader._write.assert_any_call(3, 64, 1)
    assert reader.packet.ping.call_count == 2


@pytest.mark.parametrize('code,detail', [(-3001, 'COMM_RX_TIMEOUT'), (-3002, 'COMM_RX_CORRUPT')])
def test_persistent_transient_failure_is_bounded_and_records_real_cause(reader, code, detail):
    reader.packet.ping.return_value = (0, code, 0)
    capabilities = reader.detect_capabilities()
    assert capabilities[3]['model_number'] is None
    assert capabilities[3]['current_control'] is False
    assert detail in capabilities[3]['error']
    assert str(code) in capabilities[3]['error']
    assert reader.config.port in capabilities[3]['error']
    assert '3' in capabilities[3]['error']
    assert reader.packet.ping.call_count == 3
    assert gello_hardware.time.sleep.call_args_list == [call(.01), call(.01)]
    assert reader.model_numbers.get(3) is None
    reader._write.assert_not_called()


@pytest.mark.parametrize('operation', ['current', 'hold'])
def test_persistent_ping_failure_rejects_both_modes_before_motor_writes(reader, operation):
    reader.packet.ping.return_value = (0, -3001, 0)
    with pytest.raises((RuntimeError, ValueError)) as exc:
        if operation == 'current':
            reader.enable_current_damping(current_config())
        else:
            reader.enable_alignment_hold(80)
    message = str(exc.value)
    assert reader.config.port in message
    assert '3' in message
    assert 'COMM_RX_TIMEOUT' in message
    assert 'unsupported' not in message.lower()
    assert reader.packet.ping.call_count == 3
    reader._write.assert_not_called()
    reader._read_register.assert_not_called()


def test_verified_model_cache_cannot_override_a_later_failed_ping(reader):
    reader.packet.ping.return_value = (1200, 0, 0)
    assert reader.detect_capabilities()[3]['current_control'] is True
    assert reader.model_numbers[3] == 1200
    reader.packet.ping.reset_mock()
    reader.packet.ping.return_value = (0, -3001, 0)
    with pytest.raises(RuntimeError, match='COMM_RX_TIMEOUT'):
        reader.enable_current_damping(current_config())
    assert reader.packet.ping.call_count == 3
    assert reader.model_numbers[3] is None
    reader._write.assert_not_called()


def test_ping_device_error_is_not_retried_or_treated_as_an_unknown_model(reader):
    reader.packet.ping.return_value = (1200, 0, 128)
    with pytest.raises(RuntimeError) as exc:
        reader.enable_current_damping(current_config())
    message = str(exc.value)
    assert reader.config.port in message
    assert '3' in message
    assert '128' in message
    assert 'device error' in message.lower()
    assert 'unsupported' not in message.lower()
    reader.packet.ping.assert_called_once_with(reader.port, 3)
    gello_hardware.time.sleep.assert_not_called()
    reader._write.assert_not_called()


@pytest.mark.parametrize('code', [-1000, -1001, -1002])
def test_other_communication_errors_are_not_retried(reader, code):
    reader.packet.ping.return_value = (0, code, 0)
    with pytest.raises(RuntimeError) as exc:
        reader.enable_current_damping(current_config())
    assert str(code) in str(exc.value)
    assert reader.config.port in str(exc.value)
    reader.packet.ping.assert_called_once_with(reader.port, 3)
    gello_hardware.time.sleep.assert_not_called()
    reader._write.assert_not_called()


@pytest.mark.parametrize('operation', ['current', 'hold'])
def test_successfully_reported_unknown_model_is_rejected_without_retry(reader, operation):
    reader.packet.ping.return_value = (9999, 0, 0)
    with pytest.raises((RuntimeError, ValueError)) as exc:
        if operation == 'current':
            reader.enable_current_damping(current_config())
        else:
            reader.enable_alignment_hold(80)
    message = str(exc.value)
    assert reader.config.port in message
    assert '3' in message
    assert '9999' in message
    assert 'unsupported' in message.lower() or 'maximum current' in message.lower()
    assert 'COMM_RX_' not in message
    reader.packet.ping.assert_called_once_with(reader.port, 3)
    gello_hardware.time.sleep.assert_not_called()
    reader._write.assert_not_called()
    reader._read_register.assert_not_called()


def test_serial_os_error_keeps_port_and_original_exception_without_retry(reader):
    reader.packet.ping.side_effect = OSError(5, 'Input/output error')
    capabilities = reader.detect_capabilities()
    assert capabilities[3]['model_number'] is None
    assert capabilities[3]['current_control'] is False
    message = capabilities[3]['error']
    assert reader.config.port in message
    assert '3' in message
    assert 'Input/output error' in message
    reader.packet.ping.assert_called_once_with(reader.port, 3)
    gello_hardware.time.sleep.assert_not_called()
    reader._write.assert_not_called()


@pytest.mark.parametrize('model', ['invalid-model-number', True, 1200.5, None, 0, -1, 65536])
def test_invalid_model_number_conversion_becomes_a_verification_error(reader, model):
    reader.packet.ping.return_value = (model, 0, 0)
    capabilities = reader.detect_capabilities()
    assert capabilities[3]['model_number'] is None
    assert capabilities[3]['current_control'] is False
    assert repr(model) in capabilities[3]['error']
    assert reader.config.port in capabilities[3]['error']
    reader.packet.ping.assert_called_once_with(reader.port, 3)
    gello_hardware.time.sleep.assert_not_called()
    reader._write.assert_not_called()


@pytest.mark.parametrize('reply', [(), (1200, -3001)])
def test_incomplete_ping_result_cannot_hide_communication_failure(reader, reply):
    reader.packet.ping.return_value = reply
    capabilities = reader.detect_capabilities()
    assert capabilities[3]['model_number'] is None
    assert capabilities[3]['current_control'] is False
    assert 'error' in capabilities[3]
    reader.packet.ping.assert_called_once_with(reader.port, 3)
    gello_hardware.time.sleep.assert_not_called()
    reader._write.assert_not_called()


@pytest.mark.parametrize('reply', [1200, (1200,)])
def test_model_only_wrappers_remain_supported_without_bypassing_model_gate(reader, reply):
    reader.packet.ping.return_value = reply
    capabilities = reader.detect_capabilities()
    assert capabilities[3]['model_number'] == 1200
    assert capabilities[3]['current_control'] is True
    reader.packet.ping.assert_called_once_with(reader.port, 3)
    gello_hardware.time.sleep.assert_not_called()


def test_one_unverified_motor_prevents_all_current_mode_configuration(reader):
    reader.config.joint_ids = [1, 3]
    reader.packet.ping.side_effect = lambda port, motor: (1200, 0, 0) if motor == 1 else (0, -3001, 0)
    config = current_config()
    config.damping_gain = [8., 8.]
    config.damping_current_limit = [20, 20]
    with pytest.raises(RuntimeError, match='ID 3'):
        reader.enable_current_damping(config)
    assert reader.packet.ping.call_args_list == [call(reader.port, 1)] + [call(reader.port, 3)] * 3
    assert reader.model_numbers[1] == 1200
    assert reader.model_numbers[3] is None
    reader._write.assert_not_called()


def test_closed_reader_never_attempts_model_verification(reader):
    reader.closed = True
    with pytest.raises(RuntimeError, match='closed'):
        reader.detect_capabilities()
    reader.packet.ping.assert_not_called()
    reader._write.assert_not_called()
