#!/usr/bin/env python3
"""Read-only X-series Dynamixel hold diagnostics; never changes torque or goals."""

import argparse
from pathlib import Path

import yaml


def signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def read_register(packet, port, motor_id, address, size):
    method = {1: packet.read1ByteTxRx, 2: packet.read2ByteTxRx,
              4: packet.read4ByteTxRx}[size]
    value, comm, error = method(port, motor_id, address)
    if comm != 0 or error != 0:
        detail = ' (COMM_RX_TIMEOUT: no status packet received)' if comm == -3001 else ''
        raise RuntimeError(
            f'Dynamixel ID {motor_id} read address {address} failed '
            f'(communication={comm}{detail}, device={error})'
        )
    return value


def inspect_motors(name, leader, packet, port):
    """Inspect every motor the teleop reader expects, despite individual timeouts."""
    joint_ids = leader['joint_ids']
    configured = leader.get('hold_pwm_by_joint')
    if configured is None:
        configured = [leader.get('hold_pwm')] * len(joint_ids)
    if len(configured) != len(joint_ids):
        raise ValueError(f'{name}: hold_pwm_by_joint must match joint_ids')
    ids = list(joint_ids)
    requested_pwm = list(configured)
    if leader.get('gripper_id', -1) >= 0:
        ids.append(leader['gripper_id'])
        requested_pwm.append(None)
    ids.extend(leader.get('torque_joint_ids') or ())
    requested_pwm.extend([None] * len(leader.get('torque_joint_ids') or ()))
    failed = []
    print(f"{name}: {leader['port']}")
    for motor_id, requested in zip(ids, requested_pwm):
        read = lambda address, size: read_register(packet, port, motor_id, address, size)
        try:
            # One retry for discovery covers an occasional lost status packet.
            try:
                model = read(0, 2)
            except RuntimeError:
                model = read(0, 2)
            mode = read(11, 1)
            pwm_limit = read(36, 2)
            position_max = read(48, 4)
            position_min = read(52, 4)
            torque_on = read(64, 1)
            hardware_error = read(70, 1)
            goal_pwm = signed(read(100, 2), 16)
            present_pwm = signed(read(124, 2), 16)
            goal_position = signed(read(116, 4), 32)
            present_position = signed(read(132, 4), 32)
        except RuntimeError as exc:
            failed.append(motor_id)
            print(f'  ID {motor_id}: {exc}')
            continue
        print(
            f'  ID {motor_id}: model={model} mode={mode} torque={torque_on} '
            f'PWM configured={requested} EEPROM limit={pwm_limit} '
            f'goal={goal_pwm} present={present_pwm} '
            f'goal_position_ticks={goal_position} present_position_ticks={present_position} '
            f'position_limits=[{position_min},{position_max}] '
            f'position_error_ticks={goal_position - present_position} '
            f'hardware_error=0x{hardware_error:02x}'
        )
        if requested is not None and requested > pwm_limit:
            print(f'    Configured PWM exceeds this motor\'s EEPROM limit by {requested - pwm_limit}.')
        if hardware_error:
            print('    Motor reports a hardware error; inspect the model-specific status bits.')
    return failed


def inspect_side(name, leader):
    from dynamixel_sdk import PacketHandler, PortHandler

    port = PortHandler(leader['port'])
    try:
        opened = port.openPort()
    except Exception as exc:
        if 'Permission denied' in str(exc):
            raise PermissionError(
                f"Cannot open {leader['port']}: this process needs serial-device access "
                '(usually membership in the dialout group)'
            ) from exc
        raise
    if not opened:
        raise ConnectionError(f"Cannot open {leader['port']}; stop other GELLO programs first")
    try:
        if not port.setBaudRate(int(leader['baudrate'])):
            raise ConnectionError(f"Cannot set GELLO baudrate on {leader['port']}")
        port.ser.exclusive = True
        packet = PacketHandler(2.0)
        return inspect_motors(name, leader, packet, port)
    finally:
        port.closePort()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, help='Dual GELLO YAML; no xArm connection')
    parser.add_argument('--side', choices=('left', 'right'), help='Omit to read both sides')
    args = parser.parse_args()
    data = yaml.safe_load(Path(args.config).expanduser().read_text())
    try:
        failed = []
        for name in ((args.side,) if args.side else ('left', 'right')):
            failed.extend((name, motor_id) for motor_id in
                          inspect_side(name, data[name]['TeleoperatorConfig']))
        if failed:
            details = ', '.join(f'{name} ID {motor_id}' for name, motor_id in failed)
            parser.exit(1, f'No complete status from: {details}\n')
    except (ConnectionError, PermissionError, RuntimeError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
