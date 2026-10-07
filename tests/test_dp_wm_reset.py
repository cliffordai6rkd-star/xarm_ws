from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference.dp_wm.config import load_config
from inference.dp_wm.reset import ResetMotion


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'gello_teleop/models/xarm7_dynamics_g2.urdf'
TARGET = np.array([0., 0., 0., 1.570796, 0., 1.570796, -1.570796])


@pytest.mark.parametrize('value', ['0', '-1', 'true', '1.5'])
def test_invalid_warmup_count_rejected_before_model_or_hardware(tmp_path, value):
    config = tmp_path / 'bad.yaml'
    config.write_text(f'execution:\n  warmup_steps: {value}\n')
    with pytest.raises(ValueError, match='warmup_steps'):
        load_config(config)


class Arm:
    def __init__(self):
        self.calls = []
        self.status = dict(mode=0, state=0, error_code=0)

    def enable(self):
        self.calls.append('enable')

    def move_to_reset(self, target, **kwargs):
        self.calls.append(('move', target.copy(), kwargs))

    def stop_motion(self):
        self.calls.append('stop')

    def control_status(self):
        return self.status


def setup_motion(monkeypatch):
    import inference.dp_wm.reset as module
    clock = SimpleNamespace(us=1_000_000, seconds=0.)
    monkeypatch.setattr(module, 'now_us', lambda: clock.us)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.seconds)
    arm = Arm()
    motion = ResetMotion(arm, TARGET, {'reset_samples': 2}, URDF)
    return motion, arm, clock


def test_reset_requires_fresh_distinct_settled_samples(monkeypatch):
    motion, arm, clock = setup_motion(monkeypatch)
    assert arm.calls == []
    motion.begin()
    assert arm.calls[0] == 'enable'
    np.testing.assert_equal(arm.calls[1][1], TARGET)
    state = SimpleNamespace(timestamp_us=clock.us, q=TARGET)
    assert not motion.poll(state)  # Pre-command sample.
    clock.us += 10_000
    state.timestamp_us = clock.us
    assert not motion.poll(state)
    assert not motion.poll(state)  # Same sample cannot count twice.
    clock.us += 10_000
    state.timestamp_us = clock.us
    assert motion.poll(state)
    assert not motion.running
    assert arm.calls[-1] == 'stop'


@pytest.mark.parametrize('failure', ['timeout', 'fault', 'enable'])
def test_reset_failures_can_be_stopped_during_cleanup(monkeypatch, failure):
    motion, arm, clock = setup_motion(monkeypatch)
    if failure == 'enable':
        def fail():
            raise RuntimeError('enable failed')
        arm.enable = fail
        with pytest.raises(RuntimeError, match='enable failed'):
            motion.begin()
    else:
        motion.begin()
        if failure == 'timeout':
            clock.seconds = 61.
        else:
            arm.status['error_code'] = 1
        with pytest.raises((RuntimeError, TimeoutError)):
            motion.poll(None)
    motion.stop()
    assert arm.calls[-1] == 'stop'
    assert not motion.running


@pytest.mark.parametrize('target', [[0.] * 6, [float('nan')] * 7, [100.] * 7])
def test_invalid_reset_target_rejected_before_motion(target):
    arm = Arm()
    with pytest.raises(ValueError):
        ResetMotion(arm, target, {}, URDF)
    assert arm.calls == []


def test_cli_startup_reset_and_i_pause_until_s(monkeypatch, tmp_path):
    from inference.dp_wm import cli, mock
    from nero_collection.keyboard import TerminalKeys

    config = load_config(ROOT / 'inference/configs/dp_wm_xarm.yaml')
    config['logging']['output_dir'] = tmp_path
    config['controller']['enabled'] = False
    monkeypatch.setattr(cli, 'load_config', lambda _: config)
    calls = []

    class Motion:
        running = False

        def begin(self):
            calls.append('begin')
            self.running = True

        def poll(self, _):
            calls.append('complete')
            self.running = False
            return True

        def stop(self):
            calls.append('stop')
            self.running = False

    class Source(mock.MockSource):
        reset_motion = Motion()

    class Pipeline:
        counts = {}

        def __init__(self, *a, **kw):
            pass

        def poll(self, *a, active, **kw):
            calls.append(('active', active))
            return []

        def reset(self):
            calls.append('clear')

        def close(self):
            pass

        def report(self):
            return {}

    keys = iter([None, 's', 'i', 's', 'p', 'q'])
    monkeypatch.setattr(TerminalKeys, 'read_key', lambda *a: next(keys))
    monkeypatch.setattr(mock, 'MockSource', Source)
    monkeypatch.setattr(cli, 'AsyncDPWM', Pipeline)
    cli.main(['--config', 'unused', '--run', '--mock', '--duration', '5'])
    assert calls.count('begin') == 2
    assert calls.count('complete') == 2
    assert [c[1] for c in calls if isinstance(c, tuple)] == [False, True, False, True, False]
