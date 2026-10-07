"""Offline contract/scheduling tests: never connect hardware or load real networks."""
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest
import torch

from inference.core.contracts import ActionChunk
from inference.dp_wm.core import ActionPlan, AsyncDPWM, HistorySnapshot, Worker
from inference.policies.dp.policy import DiffusionPolicy


@pytest.mark.parametrize('cached', [False, True])
def test_pinn_train_namespace_wins_over_dp_train_script(tmp_path, monkeypatch, cached):
    import importlib
    import sys
    from inference.dp_wm.models import add_pinn_source

    pinn, dp = tmp_path / 'pinn', tmp_path / 'dp'
    (pinn / 'train').mkdir(parents=True)
    (pinn / 'train' / 'base_trainer.py').write_text('SOURCE = "pinn"\n')
    dp.mkdir()
    (dp / 'train.py').write_text('SOURCE = "dp"\n')
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'modules', sys.modules.copy())
    for name in list(sys.modules):
        if name == 'train' or name.startswith('train.'):
            del sys.modules[name]
    monkeypatch.syspath_prepend(str(dp))
    if cached:
        assert importlib.import_module('train').SOURCE == 'dp'
    add_pinn_source(pinn)
    assert importlib.import_module('train.base_trainer').SOURCE == 'pinn'
    # Repeated setup and later search-path edits preserve the selected package.
    add_pinn_source(pinn)
    monkeypatch.syspath_prepend(str(dp))
    assert list(sys.modules['train'].__path__) == [str(pinn / 'train')]


def history(anchor=1_000_000):
    return HistorySnapshot(anchor + np.arange(-49, 1, dtype=np.int64)*10_000,
        {k: np.zeros((50, 7), np.float32) for k in ('q', 'dq', 'delta_q', 'tau')})


def plan(anchor=1_000_000):
    return ActionPlan(ActionChunk(np.arange(105).reshape(15,7), 'joint', None, anchor, .04), 1)


def test_fifteen_step_plan_keeps_absolute_time_and_skips_held_action():
    p = plan()
    values, times = p.future(1_000_000)
    np.testing.assert_equal(values, p.chunk.values[1:9])
    np.testing.assert_equal(times, 1_000_000 + np.arange(1,9)*40_000)
    later_values, later_times = p.future(1_020_000)
    np.testing.assert_equal(later_values, values)
    np.testing.assert_equal(later_times, times)
    assert p.future(1_280_000) is None
    assert p.future(999_999) is None


@pytest.mark.parametrize('age,start', [(0,1), (39_999,1), (40_000,2),
    (80_000,3), (239_999,6), (240_000,7), (279_999,7)])
def test_full_plan_sliding_window_boundaries(age, start):
    p = plan()
    values, times = p.future(1_000_000+age)
    np.testing.assert_equal(values, p.chunk.values[start:start+8])
    np.testing.assert_equal(times, 1_000_000+np.arange(start,start+8)*40_000)
    assert np.all(times > 1_000_000+age)
    assert p.covers(1_000_000+age)


def test_delayed_plan_is_usable_and_new_plan_replaces_without_retiming():
    e = engine()
    e.dp.result = (plan().chunk, 21.)
    events = e.poll(1_080_000, history=history(1_080_000))
    assert events[0]['usable_now'] and events[0]['plan_age_ms'] == 80.
    np.testing.assert_equal(events[0]['policy_q'],plan().chunk.values)
    np.testing.assert_equal(events[0]['policy_timestamps_us'],1_000_000+np.arange(15)*40_000)
    assert events[0]['policy_clock']=='observation'
    old_request = e.wm.request
    np.testing.assert_equal(old_request['actions'], plan().chunk.values[3:11])
    np.testing.assert_equal(old_request['action_timestamps_us'], 1_000_000+np.arange(3,11)*40_000)
    e.wm.busy = False
    e.dp.result = (plan(1_080_000).chunk, 20.)
    e.poll(1_100_000, history=history(1_100_000))
    assert e.wm.request['plan_version'] == 2
    np.testing.assert_equal(e.wm.request['actions'], plan().chunk.values[1:9])
    assert old_request['plan_version'] == 1


def test_contact_action_grid_interpolates_without_tail_extrapolation():
    p = plan()
    values, times = p.future(1_020_000, aligned=True)
    np.testing.assert_equal(times, 1_020_000+np.arange(1,9)*40_000)
    np.testing.assert_allclose(values, (p.chunk.values[1:9]+p.chunk.values[2:10])/2)
    assert p.future(1_240_000, aligned=True) is not None
    assert p.future(1_240_001, aligned=True) is None
    assert p.future(999_999, aligned=True) is None
    assert p.covers(1_240_000, aligned=True)
    assert not p.covers(1_240_001, aligned=True)
    assert not p.covers(999_999, aligned=True)


def test_contact_scheduler_sends_anchor_aligned_actions():
    e = engine()
    e.aligned_actions = True
    e.dp.result = (plan().chunk, 20.)
    e.poll(1_020_000, history=history(1_020_000))
    np.testing.assert_equal(e.wm.request['action_timestamps_us'],1_020_000+np.arange(1,9)*40_000)


def test_history_rejects_missing_samples_and_timestamp_gaps():
    history().validate()
    h = history()
    h.timestamps_us[-1] += 9_000
    with pytest.raises(ValueError, match='cadence'):
        h.validate()
    h = history()
    h.values['tau'][1, 2] = np.nan
    with pytest.raises(ValueError, match='tau'):
        h.validate()


def test_dp_extended_plan_does_not_change_normal_eight_step_output():
    model = SimpleNamespace(image_keys=('front',), n_obs_steps=2, horizon=16,
        n_action_steps=8, action_dim=7, pred_action_steps_only=False)
    policy = DiffusionPolicy(model, device='cpu', step_s=.04, image_shapes={'front':(3,2,2)})
    full = torch.arange(112.).reshape(1,16,7)
    policy._predict_action = lambda _: {'action': full[:,1:9], 'action_pred': full}
    obs = SimpleNamespace(timestamp_us=1_000_000, images={'front':np.zeros((2,2,3),np.uint8)}, metadata={})
    standard = policy.predict_observations((obs,))
    extended = policy.predict_observations((obs,), plan_steps=15)
    np.testing.assert_equal(standard.values, full[0,1:9].numpy())
    np.testing.assert_equal(extended.values, full[0,1:16].numpy())
    assert extended.timestamp_us == obs.timestamp_us
    policy._predict_action = lambda _: {'action': full[:,1:9]}
    with pytest.raises(ValueError, match='action_pred'):
        policy.predict_observations((obs,), plan_steps=15)


class ManualWorker:
    def __init__(self):
        self.request = None
        self.result = None
        self.latencies_ms = []
        self.busy = False

    def submit(self, request):
        assert not self.busy
        self.request = request
        self.busy = True

    def poll(self):
        result, self.result = self.result, None
        if result is not None:
            self.busy = False
        return result

    def close(self):
        pass


def engine(**kwargs):
    e = AsyncDPWM(SimpleNamespace(predict_observations=lambda _:None), SimpleNamespace(infer=lambda _:None), **kwargs)
    e.close()
    e.dp, e.wm = ManualWorker(), ManualWorker()
    return e


def test_expired_dp_result_is_not_reanchored():
    e = engine()
    e.dp.result = (plan().chunk, 280.)
    events = e.poll(1_280_000, history=history(1_280_000))
    assert e.plan is None
    assert not events[0]['usable_now']
    assert events[0]['anchor_us'] == 1_000_000
    assert e.counts['dp_expired'] == 1


def test_wm_reuses_overlapping_windows_until_full_plan_expires():
    e = engine()
    e.dp.result = (plan().chunk, 5.)
    e.poll(1_000_000, history=history())
    first = e.wm.request
    assert first['history'].anchor_us == 1_000_000
    assert first['deadline_us'] == 1_020_000
    assert first['actions'].shape == (8,7)
    e.wm.busy = False
    e.poll(1_020_000, history=history(1_020_000))
    second = e.wm.request
    assert second['history'].anchor_us == 1_020_000
    np.testing.assert_equal(first['action_timestamps_us'], second['action_timestamps_us'])
    e.wm.busy = False
    e.poll(1_280_000, history=history(1_280_000))
    assert not e.wm.busy
    assert e.counts['action_underflows'] == 1


def test_result_selection_skips_expired_points_and_preserves_samples():
    e = engine()
    q = np.zeros((2,32,7))
    e.wm.result = ({'q':q, 'tau':q.copy(), 'future_timestamps_us': 1_000_000+np.arange(1,33)*10_000,
                   'deadline_us':1_020_000, 'anchor_us':1_000_000}, 25.)
    result = e.poll(1_025_000, active=False)[0]
    np.testing.assert_equal(result['selected_indices'], [2,3])
    assert result['q'].shape == (2,32,7)
    assert result['deadline_missed']


def test_dp_requests_follow_camera_phase_not_startup_phase():
    e = engine()
    obs = (SimpleNamespace(timestamp_us=960_000), SimpleNamespace(timestamp_us=1_000_000))
    e.poll(1_030_000, observations=obs)
    assert e.next_dp_us == 1_040_000
    e.dp.result = (plan().chunk, 1.)
    e.poll(1_031_000)
    obs = (SimpleNamespace(timestamp_us=1_000_000), SimpleNamespace(timestamp_us=1_040_000))
    e.poll(1_040_000, observations=obs)
    assert e.dp.request[-1].timestamp_us == 1_040_000
    assert e.next_dp_us == 1_080_000


def test_dp_eight_step_window_requests_next_observation_only_after_expiry():
    e = engine(policy_refresh_steps=8, wait_for_window_exhaustion=True)
    first_obs = (SimpleNamespace(timestamp_us=960_000), SimpleNamespace(timestamp_us=1_000_000))
    e.poll(1_000_000, observations=first_obs)
    assert e.dp.busy
    e.dp.result = (plan().chunk, 66.)
    event = e.poll(1_066_000)[0]
    assert event['usable_now']
    assert event['window_start_us'] == 1_000_000
    assert event['window_end_us'] == 1_320_000
    assert e.report()['dp_request_mode'] == 'after_window_exhaustion'
    fresh_obs = (SimpleNamespace(timestamp_us=1_280_000), SimpleNamespace(timestamp_us=1_320_000))
    e.poll(1_319_999, observations=fresh_obs)
    assert not e.dp.busy
    assert e.counts['dp_window_wait_ticks'] >= 1
    e.poll(1_320_000, observations=fresh_obs)
    assert e.dp.busy
    assert e.dp.request[-1].timestamp_us == 1_320_000


def test_expired_dp_plan_does_not_create_an_eight_step_wait():
    e = engine(policy_refresh_steps=8, wait_for_window_exhaustion=True)
    e.dp.result = (plan().chunk, 280.)
    e.poll(1_280_000)
    assert e.next_dp_us == 1_280_000


def test_waiting_wm_retries_when_dp_result_arrives_within_deadline():
    e = engine()
    e.poll(1_000_000, history=history())
    assert not e.wm.busy
    e.dp.result = (plan().chunk, 1.)
    e.poll(1_001_000, history=history())
    assert e.wm.busy
    assert e.wm.request['deadline_us'] == 1_020_000


def test_pause_discards_results_and_busy_workers_have_no_queue():
    e = engine()
    e.reset()
    e.dp.result = (plan().chunk, 2.)
    assert e.poll(1_000_000, active=False) == []
    assert e.plan is None
    gate = threading.Event()
    worker = Worker('test', lambda _:gate.wait(1))
    try:
        assert worker.submit(1)
        assert not worker.submit(2)
        gate.set()
        deadline = time.monotonic()+2
        result = None
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            time.sleep(.001)
        assert result is not None
    finally:
        gate.set()
        worker.close()


def test_worker_errors_reach_runtime():
    def fail(_):
        raise ValueError('model failure')
    worker = Worker('fail', fail)
    try:
        worker.submit(None)
        while not worker.future.done():
            time.sleep(.001)
        with pytest.raises(ValueError, match='model failure'):
            worker.poll()
    finally:
        worker.close()


@pytest.mark.parametrize('steps', [1, 3])
def test_worker_warmup_never_publishes_synthetic_plans_or_latency_samples(steps):
    seen = []
    policy = SimpleNamespace(predict_observations=lambda *a, **kw: seen.append(threading.current_thread().name))
    wm = SimpleNamespace(infer=lambda *a: seen.append(threading.current_thread().name))
    engine = AsyncDPWM(policy, wm)
    try:
        engine.warmup((), {}, steps=steps)
        assert len(seen) == 2*steps
        assert all(name.startswith(('dp-plan', 'latent-wm')) for name in seen)
        assert engine.plan is None and engine.latest_prediction is None
        assert not engine.counts
        assert not engine.dp.latencies_ms and not engine.wm.latencies_ms
        assert engine.dp.poll() is None and engine.wm.poll() is None
    finally:
        engine.close()


def test_controller_reference_only_uses_getter():
    from ufactory_devices.robot.xarm_adapter import XArmAdapter
    arm = XArmAdapter.__new__(XArmAdapter)
    arm.dof = 7
    calls = []
    def call(name, **kwargs):
        calls.append((name, kwargs))
        return 0, [0.1]*7
    arm._call = call
    q, stamp = arm.read_controller_reference()
    np.testing.assert_allclose(q, .1)
    assert stamp > 0
    assert calls == [('get_servo_angle', {'is_radian':True, 'is_real':False, 'required':True})]


@pytest.mark.parametrize('enable_command', [False, True])
def test_live_source_constructs_endpoints_without_connecting(tmp_path, monkeypatch, enable_command):
    from pathlib import Path
    import yaml
    from nero_collection import cameras
    from ufactory_devices.robot import xarm_adapter
    from inference.dp_wm.live import LiveSource
    calibration = tmp_path/'calibration.yaml'
    calibration.write_text(yaml.safe_dump({'right':{'RobotConfig':{'robot_ip':'127.0.0.1',
        'reset_q': [0., 0., 0., 1.570796, 0., 1.570796, -1.570796]}}}))
    hardware = tmp_path/'hardware.yaml'
    hardware.write_text(yaml.safe_dump({'gello_config':'calibration.yaml',
        'cameras':[{'name':'front', 'backend':'mock', 'width':224, 'height':224}],
        'arms':{'right':{'execution_enabled':True}}}))
    received = []
    class FakeArm:
        def __init__(self, config):
            received.append(config)
        def connect(self):
            raise AssertionError('construction must not connect')
    monkeypatch.setattr(xarm_adapter, 'XArmAdapter', FakeArm)
    monkeypatch.setattr(cameras.CameraManager, 'from_config', lambda *a, **k: object())
    monkeypatch.setattr(cameras.CameraVisualizer, 'from_config', lambda *a, **k: None)
    source = LiveSource({'observation':{'hardware_config':hardware, 'arm':'right'},
                         'wm': {'urdf_path': Path(__file__).resolve().parents[1] /
                                'gello_teleop/models/xarm7_dynamics_g2.urdf'}},
                        {'images':{'front':(3,224,224)}, 'wm_training':{'dataloader':{}}},
                        enable_command=enable_command)
    assert received[0].name == 'right'
    assert received[0].config_kwargs['dof'] == 7
    assert received[0].config_kwargs['execution_enabled'] is enable_command
    assert (source.reset_motion is not None) is enable_command
    assert received[0].config_kwargs['init_gripper'] is False
    assert not source.thread.is_alive()



def test_report_forecast_age_budget_includes_period_latency_and_history_age():
    e = AsyncDPWM(SimpleNamespace(predict_observations=lambda *args, **kwargs: None),
                  SimpleNamespace(infer=lambda _: None), execute_steps=8,
                  initial_latency_ms=57., margin_ms=3.)
    try:
        assert e.report()['wm_reference_age_budget_ms'] == 170.
    finally:
        e.close()


def test_wm_recording_preserves_exact_conditioning_actions_and_history():
    from inference.dp_wm.mock import MockPolicy,MockWM
    e=AsyncDPWM(MockPolicy(),MockWM())
    h=history(1_000_000)
    actions=np.arange(56,dtype=float).reshape(8,7)
    times=1_000_000+np.arange(1,9)*40_000
    try:
        result=e.wm.function(dict(history=h,actions=actions,action_timestamps_us=times,
                                  plan_version=2,deadline_us=1_050_000))
        saved_q=h.values['q'][-1].copy();saved_delta=h.values['delta_q'][-1].copy()
        actions[:]=-99;times[:]=0
        h.values['q'][:]=-100;h.values['delta_q'][:]=-200
        np.testing.assert_equal(result['policy_condition_q'],np.arange(56).reshape(8,7))
        np.testing.assert_equal(result['policy_condition_timestamps_us'],1_000_000+np.arange(1,9)*40_000)
        np.testing.assert_equal(result['wm_history_q_anchor'],saved_q)
        np.testing.assert_equal(result['wm_history_delta_q_anchor'],saved_delta)
    finally:
        e.close()
