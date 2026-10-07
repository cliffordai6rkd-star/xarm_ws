"""Logical-time live CLI tests with real scheduling/output and synthetic hardware."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from inference.core.contracts import ActionChunk
from inference.dp_wm.command_history import HostCommandHistory, RecordingArm
from inference.dp_wm.core import AsyncDPWM, HistorySnapshot
from inference.dp_wm.output import ActiveOutput


def test_feedback_lock_wait_cannot_send_an_expired_target(monkeypatch):
    import inference.dp_wm.output as module
    clock = SimpleNamespace(us=1_000_000)
    monkeypatch.setattr(module, 'now_us', lambda: clock.us)
    history = HostCommandHistory()
    underlying_lock = history.lock

    class DelayedLock:
        def acquire(self, *, timeout):
            assert timeout <= .01
            underlying_lock.acquire()
            clock.us += 20_000  # Feedback finishes after the command's target time.
            return True

        def release(self):
            underlying_lock.release()

    history.lock = DelayedLock()
    sent = []
    arm = SimpleNamespace(command_history=history, enable=lambda: None,
                          stop_motion=lambda: sent.append('stop'),
                          command_joint_positions=lambda q: sent.append(q))
    sender = ActiveOutput(arm, maximum_state_age_ms=10.)
    try:
        sender.start()
        event = dict(valid=True, state_timestamp_us=1_000_000,
                     first_q_command=np.zeros(7), target_timestamps_us=np.array([1_010_000,1_020_000]))
        assert not sender.send(event)
        assert not event['valid'] and not event.get('sent', False)
        assert sent == ['stop']
    finally:
        sender.close()


@pytest.mark.parametrize('wm_delay_ms, expect_motion', [(100, True), (400, False)])
def test_live_cli_bootstrap_keeps_host_commands_causal_and_stops_if_first_forecast_is_late(
        tmp_path, monkeypatch, capsys, wm_delay_ms, expect_motion):
    from inference.dp_wm import cli, live, models, output, command_history
    from inference.dp_wm.config import load_config
    from test_pi0_contact_wm import configuration_file

    config = load_config(configuration_file(tmp_path))
    config['pi0']['open_loop'] = True
    config['observation']['command_source'] = 'host_command'
    config['wm']['action_sampling'] = 'recorded_tokens'
    config['execution']['reset_on_start'] = False
    config['controller'].update(mode='q_only', maximum_compute_ms=100., maximum_forecast_age_ms=240.)
    config['controller']['limits']['q_only_smoothing'] = 'jerk_tracking'
    config['logging']['output_dir'] = tmp_path
    clock = SimpleNamespace(us=1_000_000)
    wall = time
    monkeypatch.setattr(cli, 'time', SimpleNamespace(
        monotonic=lambda: (clock.us-1_000_000)/1e6,
        perf_counter=wall.perf_counter, time_ns=wall.time_ns, sleep=lambda _: None))
    monkeypatch.setattr(output, 'time', SimpleNamespace(monotonic=lambda: clock.us/1e6))
    for module in (cli, output, command_history):
        monkeypatch.setattr(module, 'now_us', lambda: clock.us)
    monkeypatch.setattr(cli, 'load_config', lambda _: config)
    metadata = {'images': {}, 'wm_model_version': 'carswm_v9', 'wm_action_frame': 'link_eef'}
    monkeypatch.setattr(models, 'inspect_checkpoints', lambda _: metadata)

    class Arm:
        def __init__(self):
            self.q = np.zeros(7)
            self.sent = []
            self.stops = []

        def enable(self):
            pass

        def stop_motion(self):
            self.stops.append(clock.us)

        def command_joint_positions(self, q):
            self.q = q.copy()
            self.sent.append((clock.us, q.copy()))

    arm = Arm()
    sources = []

    class Source:
        def __init__(self, *args, **kwargs):
            self.command_history = HostCommandHistory()
            self.arm = RecordingArm(arm, self.command_history)
            self.rows = deque(maxlen=50)
            self.observation_after_us = 0
            self.status = 'synthetic state reader'
            self.state = None
            sources.append(self)

        def start(self):
            pass

        def close(self):
            pass

        def control_state(self):
            return self.state

        def poll(self):
            clock.us += 10_000
            self.state = SimpleNamespace(timestamp_us=clock.us, q=arm.q.copy(), dq=np.zeros(7), tau=np.zeros(7))
            history = None
            if self.command_history.latest is not None:
                command, stamp = self.command_history.at_or_before(clock.us)
                assert stamp <= clock.us
                row = dict(q=self.state.q, dq=np.zeros(7), tau=np.zeros(7), delta_q=command-self.state.q)
                self.rows.append((clock.us, row))
                if len(self.rows) == 50 and self.rows[0][0] > self.observation_after_us:
                    history = HistorySnapshot(np.array([t for t,_ in self.rows], np.int64),
                        {key:np.stack([row[key] for _,row in self.rows]) for key in row})
            return (SimpleNamespace(timestamp_us=clock.us),), history

    monkeypatch.setattr(live, 'LiveSource', Source)
    policy = SimpleNamespace(n_obs_steps=1, close=lambda: None,
        predict_observations=lambda obs, plan_steps: ActionChunk(
            np.repeat((np.arange(plan_steps)*.001)[:,None],7,axis=1),
            'joint', None, obs[-1].timestamp_us, .04))

    class WM:
        def infer(self, request):
            anchor = request['history'].anchor_us
            stamps = anchor + np.arange(1,33,dtype=np.int64)*10_000
            q = np.stack([np.interp(stamps, request['action_timestamps_us'], request['actions'][:,j])
                          for j in range(7)], axis=-1)
            return dict(q=q[None], tau=np.zeros((1,32,7)), anchor_us=anchor,
                        future_timestamps_us=stamps, plan_version=request['plan_version'],
                        deadline_us=request['deadline_us'])

    monkeypatch.setattr(models, 'load_models', lambda *args: (policy, WM()))

    class ClockWorker:
        def __init__(self, function, delay_us):
            self.function, self.delay_us = function, delay_us
            self.busy = False
            self.latencies_ms = deque(maxlen=100)

        def submit(self, request):
            assert not self.busy
            self.request, self.done_us, self.busy = request, clock.us+self.delay_us, True

        def poll(self):
            if not self.busy or clock.us < self.done_us:
                return None
            self.busy = False
            self.latencies_ms.append(self.delay_us/1000)
            return self.function(self.request), self.delay_us/1000

        def close(self):
            pass

    pipelines = []
    class Pipeline(AsyncDPWM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            super().close()
            self.dp = ClockWorker(lambda obs: policy.predict_observations(obs, plan_steps=50), 70_000)
            self.wm = ClockWorker(args[1].infer, wm_delay_ms*1000)
            pipelines.append(self)

        def warmup(self, *args, **kwargs):
            pass

    monkeypatch.setattr(cli, 'AsyncDPWM', Pipeline)
    from nero_collection.keyboard import TerminalKeys
    monkeypatch.setattr(TerminalKeys, 'read_key', lambda *args: None)
    report = cli.main(['--config', 'unused', '--run', '--enable-command', '--auto-start', '--duration', '1.3'])
    terminal = capsys.readouterr().out
    assert '"wm_requests"' not in terminal  # Periodic JSON stays in the file log by default.
    if not expect_motion:
        assert 'Active output stopped:' in terminal  # Critical stop diagnostics remain visible.
    import json
    events_file = next(tmp_path.rglob('events.jsonl'))
    events = [json.loads(line) for line in events_file.read_text().splitlines()]
    assert any(event['kind'] == 'status' for event in events)
    sent_times = [stamp for stamp,_ in arm.sent]
    assert len(sent_times) > 50
    np.testing.assert_array_equal(np.diff(sent_times), 10_000)
    for stamp, q in arm.sent[:60]:
        np.testing.assert_array_equal(q, np.zeros(7))  # First real hold persists through history accumulation.
    if expect_motion:
        assert report['last_output_stop'] is None
        assert any(np.any(np.abs(q)>1e-8) for _,q in arm.sent)
        assert report['controller']['counts']['startup_hold'] >= 60
    else:
        assert 'First WM forecast unavailable' in report['last_output_stop']['reason']
        assert all(np.all(q==0.) for _,q in arm.sent)
