from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gello_teleop.gripper_manual_test import GripperConsole, Settings, _read_register, main


def fake_arm():
    backend = SimpleNamespace(get_gripper_status=Mock(return_value=(0, 2)),
        arm_cmd=SimpleNamespace(gripper_modbus_r16s=Mock(return_value=[0, 9, 8, 3, 2, 0, 1])))
    return SimpleNamespace(_arm=backend,
        get_gripper_position=Mock(return_value=(0, 768)),
        get_gripper_err_code=Mock(return_value=(0, 0)),
        get_state=Mock(return_value=(0, 2)), get_err_warn_code=Mock(return_value=(0, [0, 0])),
        set_gripper_mode=Mock(return_value=0), set_gripper_enable=Mock(return_value=0),
        set_gripper_speed=Mock(return_value=0), set_gripper_position=Mock(return_value=0))


def console():
    arm = fake_arm()
    lines = []
    return GripperConsole(arm, Settings('left', '192.168.1.203', 800, 0, 5000, 0., .085), lines.append), lines


def test_reading_feedback_never_initializes_or_moves_gripper():
    value, lines = console()
    value.status()
    for method in ('set_gripper_mode', 'set_gripper_enable', 'set_gripper_speed', 'set_gripper_position'):
        getattr(value.arm, method).assert_not_called()
    assert '81.6mm' in lines[-1] and '实际使能=(0, 1)' in lines[-1]
    value.arm.get_gripper_position.assert_called_once_with(check_baud=False)


def test_enter_alternates_close_open_and_preserves_state_between_commands():
    value, _ = console()
    value.enter()
    assert value.last_command == 0
    value.status(); value.status()
    assert value.arm.set_gripper_position.call_count == 1
    value.enter()
    value.enter()
    assert [(c.args, c.kwargs) for c in value.arm.set_gripper_position.call_args_list] == [
        ((0,), {'wait': False, 'wait_motion': False}),
        ((800,), {'wait': False, 'wait_motion': False}),
        ((0,), {'wait': False, 'wait_motion': False})]
    value.arm.set_gripper_enable.assert_called_once_with(True)


def test_failed_close_does_not_advance_to_open():
    value, _ = console()
    value.arm.set_gripper_position.side_effect = [19, 0]
    with pytest.raises(RuntimeError, match='返回=19'):
        value.enter()
    assert value.next_close and value.last_command is None
    value.enter()
    assert [c.args[0] for c in value.arm.set_gripper_position.call_args_list] == [0, 0]


def test_register_errors_are_printed_without_fabricating_feedback():
    arm = fake_arm()
    arm._arm.arm_cmd.gripper_modbus_r16s.return_value = [19, 0]
    assert _read_register(arm, 0x0100) == [19, 0]
    arm._arm.arm_cmd.gripper_modbus_r16s.return_value = [0, 9, 8, 3, 4, 0, 0, 3, 32]
    assert _read_register(arm, 0x0700, 2) == (0, 800)


def test_check_config_does_not_import_sdk_or_connect(capsys):
    assert main(['--check-config']) == 0
    assert '192.168.1.203' in capsys.readouterr().out
