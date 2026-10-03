"""Read-only GELLO bus sampling with raw failed-packet evidence.

Stop other GELLO programs first. This tool only sends read requests: it never
enables torque, changes goals, or connects to xArm.
"""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from gello_teleop.gello_hardware import GelloReader, check


def packet_evidence(packet, result):
    """Classify the SDK's shared CRC/partial-packet error from received bytes."""
    raw = bytes(packet or ())
    expected = None
    if len(raw) >= 7 and raw[:4] == b'\xff\xff\xfd\x00':
        expected = int.from_bytes(raw[5:7], 'little') + 7
    if result == -3001 and not raw:
        reason = 'no_response'
    elif expected is not None and len(raw) >= expected and result == -3002:
        reason = 'crc_or_packet_corruption'
    elif result == -3002:
        reason = 'incomplete_or_malformed_packet'
    else:
        reason = 'receive_error'
    return dict(code=result, reason=reason, received_bytes=len(raw),
                expected_bytes=expected, packet_hex=raw.hex())


def leader_config(path, side):
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text())
    if 'gello_config' in data:
        calibration = Path(data['gello_config']).expanduser()
        if not calibration.is_absolute():
            calibration = path.parent / calibration
        data = yaml.safe_load(calibration.read_text())
    return data[side]['TeleoperatorConfig']


def inspect_bus(leader, samples, rate_hz, current_readback=True):
    # GelloReader.close() disables torque. Close only its serial descriptor
    # below so even diagnostic cleanup sends no motor writes.
    reader = GelloReader(SimpleNamespace(**leader))
    failures, packet_failures, durations = [], [], []
    original_rx = reader.packet.rxPacket
    context = {}

    def observed_rx(*args, **kwargs):
        packet, result = original_rx(*args, **kwargs)
        if result:
            packet_failures.append(dict(context, **packet_evidence(packet, result)))
        return packet, result

    reader.packet.rxPacket = observed_rx
    original_read_rx = reader.packet.readRx

    def observed_read_rx(port, motor_id, length):
        context['motor_id'] = motor_id
        return original_read_rx(port, motor_id, length)

    reader.packet.readRx = observed_read_rx
    period = 1. / rate_hz
    next_t = started = time.monotonic()
    skipped = 0
    try:
        for index in range(samples):
            tick = time.monotonic()
            context.update(sample=index, address=132)
            context.pop('motor_id', None)
            try:
                reader.read()
                if current_readback:
                    context['address'] = 102
                    context.pop('motor_id', None)
                    check(reader.current_feedback.txRxPacket(), 'Goal Current readback')
            except RuntimeError as exc:
                failures.append(dict(context, error=str(exc)))
                # Let outstanding status packets finish before another request.
                time.sleep(len(reader.ids)*15*10/leader['baudrate'] + .01)
            durations.append((time.monotonic()-tick)*1000)
            next_t += period
            now = time.monotonic()
            if next_t < now:
                missed = int((now-next_t)/period) + 1
                skipped += missed
                next_t += missed*period
            if index + 1 < samples:
                time.sleep(max(0., next_t-time.monotonic()))
        elapsed = time.monotonic()-started
        tty = Path(leader['port']).resolve().name
        latency = Path('/sys/bus/usb-serial/devices')/tty/'latency_timer'
        n, joints = len(reader.ids), len(leader['joint_ids'])
        # Protocol 2.0, 8N1, no byte stuffing/return delay/USB/scheduling costs.
        read_bytes = 14+n + 15*n
        full_bytes = read_bytes + (14+3*joints + 14+joints+13*joints if current_readback else 0)
        full_wire_ms = full_bytes*10/leader['baudrate']*1000
        return dict(port=leader['port'], baudrate=leader['baudrate'], samples=samples,
                    read_only=True, current_readback=current_readback,
                    requested_rate_hz=rate_hz, achieved_rate_hz=round(samples/elapsed, 2),
                    skipped_ticks=skipped,
                    usb_latency_ms=int(latency.read_text()) if latency.exists() else None,
                    full_cycle_wire_min_ms=round(full_wire_ms, 2),
                    full_cycle_wire_utilization=round(full_wire_ms*rate_hz/1000, 3),
                    read_cycle_ms=dict(zip(('min', 'p50', 'p95', 'p99', 'max'),
                                          np.percentile(durations, (0, 50, 95, 99, 100)).round(2).tolist())),
                    errors=failures, failed_packets=packet_failures,
                    note='Full cycle estimate includes current write; measured cycle only reads. '
                         'Torque-disabled testing cannot reproduce load-dependent power or noise faults.')
    finally:
        reader.packet.rxPacket = original_rx
        reader.packet.readRx = original_read_rx
        reader.port.closePort()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, help='Dataset or dual calibration YAML')
    parser.add_argument('--side', choices=('left', 'right'), default='right')
    parser.add_argument('--samples', type=int, default=600)
    parser.add_argument('--rate-hz', type=float, default=15.)
    parser.add_argument('--position-only', action='store_true')
    parser.add_argument('--output', type=Path, help='Save complete JSON evidence')
    args = parser.parse_args(argv)
    if args.samples <= 0 or not math.isfinite(args.rate_hz) or args.rate_hz <= 0:
        parser.error('samples and rate-hz must be positive and finite')
    try:
        result = inspect_bus(leader_config(args.config, args.side), args.samples,
                             args.rate_hz, not args.position_only)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f'{args.side} GELLO diagnostic failed: {exc}\n')
    result['side'] = args.side
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output+'\n')
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
