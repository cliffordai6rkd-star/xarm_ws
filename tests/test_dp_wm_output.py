from types import SimpleNamespace

import numpy as np
import pytest

from inference.dp_wm.output import ActiveOutput


class Arm:
    def __init__(self):
        self.calls = []

    def enable(self):
        self.calls.append('enable')

    def stop_motion(self):
        self.calls.append('stop')

    def command_joint_positions(self, q):
        self.calls.append(q.copy())


@pytest.fixture
def output(monkeypatch):
    import inference.dp_wm.output as module
    clock = SimpleNamespace(us=1_000_000, seconds=0.)
    monkeypatch.setattr(module, 'now_us', lambda: clock.us)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.seconds)
    arm = Arm()
    sender = ActiveOutput(arm, maximum_state_age_ms=10)
    try:
        yield sender, arm, clock
    finally:
        sender.close()


def event(stamp=1_000_000):
    return dict(valid=True, sent=False, first_q_command=np.ones(7)*.01,
                state_timestamp_us=stamp, target_timestamps_us=np.array([stamp+10_000, stamp+20_000]))


def test_authorized_output_and_latched_watchdog(output):
    sender, arm, clock = output
    assert not sender.send(event())
    assert arm.calls == []
    sender.start()
    packet = event()
    assert sender.send(packet) and packet['sent']
    np.testing.assert_equal(arm.calls[-1], packet['first_q_command'])
    clock.seconds += .031
    sender.watchdog_tick()
    assert arm.calls[-1] == 'stop' and not sender.enabled
    assert not sender.send(event())  # No automatic restart after a gap.
    sender.start()
    assert sender.send(event())


@pytest.mark.parametrize('case', ['expired', 'nan', 'invalid', 'future'])
def test_invalid_output_stops_after_first_send(output, case):
    sender, arm, clock = output
    sender.start()
    assert sender.send(event())
    packet = event()
    if case == 'expired':
        clock.us += 10_001
    elif case == 'nan':
        packet['first_q_command'][0] = np.nan
    elif case == 'future':
        packet['state_timestamp_us'] += 1
    else:
        packet['valid'] = False
        packet['reason'] = 'WM reference exceeded the configured age'
    assert not sender.send(packet)
    assert not packet['sent']
    assert arm.calls[-1] == 'stop'
    if case == 'invalid':
        assert 'WM reference exceeded the configured age' in sender.reason


def test_sdk_error_propagates_and_cleanup_stops(output):
    sender, arm, clock = output
    def fail(_):
        raise RuntimeError('SDK rejected target')
    arm.command_joint_positions = fail
    sender.start()
    with pytest.raises(RuntimeError, match='SDK'):
        sender.send(event())
    sender.close()
    assert arm.calls[-1] == 'stop'


def test_missing_authorization_and_mock_fail_before_model_or_hardware(monkeypatch):
    from inference.dp_wm import cli
    monkeypatch.setattr(cli, 'load_config', lambda _: {'controller': {'enabled': True, 'execution': 'active'}})
    for flags in ([], ['--mock', '--enable-command']):
        with pytest.raises(SystemExit) as error:
            cli.main(['--config', 'unused', '--run', *flags])
        assert error.value.code == 2


def test_waiting_for_fresh_state_does_not_refresh_watchdog(output):
    sender,arm,clock=output
    sender.start()
    assert sender.send(event())
    clock.seconds=.011
    assert not sender.send(None)
    assert sender.enabled and sender.last_send==0.
    clock.seconds=.031
    assert not sender.send(None)
    assert not sender.enabled and arm.calls[-1]=='stop'
