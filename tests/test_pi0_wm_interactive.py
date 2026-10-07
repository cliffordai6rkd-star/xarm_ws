"""Public PI0 lifecycle tests use simulated cameras/arms and local models only."""
from pathlib import Path
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from inference import cli
from inference.interactive import run_interactive
from inference.pi0_wm import cli as pi0_cli
from inference.pi0_wm.config import load_config
from inference.pi0_wm.core import Request, Worker
from inference.pi0_wm.runtime import Runtime

def config_fixture():
    return {
        'pi0': {'host': 'localhost', 'port': 8000, 'prompt': 'mock task',
                'action_hz': 25, 'consume_steps': 50, 'output_key': 'actions',
                'quaternion_tolerance': .15, 'interface': {
                    'training_config': 'mock_ee_pose', 'state_key': 'state', 'prompt_key': 'prompt',
                    'images': {'left_wrist': 'images/left', 'right_wrist': 'images/right'},
                    'state_semantic': 'observation.ee_pose', 'action_semantic': 'action.ee_pose',
                    'coordinate_frame': 'tcp', 'representation': 'xyz_quaternion',
                    'quaternion_order': 'xyzw', 'image_format': 'uint8_hwc_rgb',
                    'output_space': 'physical_absolute', 'action_hz': 25}},
        'wm': {'coordinate_frame': 'tcp', 'pinn_root': '/offline/PINN',
               'checkpoint': '/offline/model.pt', 'device': 'cpu', 'use_ema': True,
               'num_samples': 1, 'selected_sample': 0, 'flow_steps': 1, 'solver': 'euler'},
        'control': {'hz': 100, 'execute_steps': 20, 'maximum_steps': 3000,
                    'maximum_state_age_s': .1, 'maximum_camera_age_s': .3},
        'calibration': {'warmup_samples': 1, 'minimum_samples': 1, 'pi_seconds': .001,
                        'wm_seconds': .001, 'wm_margin_s': .02, 'pi_margin_s': .08,
                        'request_timeout_s': 1.},
        'hardware': {'backend': 'mock', 'endpoint': {'name': 'mock', 'config_kwargs': {'dof': 7}},
                     'q_min': [-3.]*7, 'q_max': [3.]*7, 'maximum_step_rad': [.02]*7,
                     'reset_q': None, 'reset_speed_rad_s': 1., 'reset_timeout_s': .8,
                     'reset_tolerance_rad': .002, 'reset_samples': 1},
        'cameras': [{'name': name, 'backend': 'mock', 'width': 10, 'height': 8,
                     'fps': 30, 'depth': False, 'visualize': False}
                    for name in ('left_wrist', 'right_wrist')],
        'mujoco': {'enabled': False, 'headless': True, 'mujoco_model_path': None,
                  'robot_joint_names': [], 'ee_body_name': None, 'ee_site_name': None,
                  'render_fps': 50, 'prediction_visualization_hz': 100, 'point_size': .003},
    }


@pytest.fixture
def mock_config(tmp_path):
    path = tmp_path / 'pi0.yaml'
    path.write_text(yaml.safe_dump(config_fixture()))
    return path


def fixture(*, readonly=False, commands=False):
    config = config_fixture()
    runtime = Runtime(config, mock=True, mock_wm=True)
    runtime.wm.latency = .001
    runtime._mock_pi = Mock(side_effect=lambda obs: np.tile(obs['state'], (50, 1)))
    infer_wm = runtime.wm.infer

    def predict(payload):
        result = infer_wm(payload)
        result['q'][:] = .1
        return result

    runtime.wm.infer = Mock(side_effect=predict)
    runtime.command_enabled = commands  # The already-created arm remains a MockArm.
    if readonly:
        runtime.simulated_arm = False
    runtime.arm.enable = Mock(wraps=runtime.arm.enable)
    runtime.arm.command_joint_positions = Mock(wraps=runtime.arm.command_joint_positions)
    return runtime


def test_public_pi0_dispatch_uses_common_terminal_loop(monkeypatch, mock_config):
    seen = {}

    def dispatch(argv, *, run_loop):
        seen.update(argv=argv, run_loop=run_loop)
        return 'pi0'

    monkeypatch.setattr(pi0_cli, 'main', dispatch)
    args = ['--config', str(mock_config), '--run']
    assert cli.main(args) == 'pi0'
    assert seen == {'argv': args, 'run_loop': run_interactive}


def test_pi0_check_never_constructs_models_or_hardware(monkeypatch, mock_config):
    construct = Mock(side_effect=AssertionError('must not construct runtime'))
    monkeypatch.setattr('inference.pi0_wm.runtime.Runtime', construct)
    report = cli.main(['--config', str(mock_config), '--check', '--mock'])
    assert report['status'] == 'config_checked' and not report['hardware_connected']
    construct.assert_not_called()


def test_pi0_cli_adapts_steps_and_common_duration(monkeypatch, mock_config):
    runtime = Mock()
    monkeypatch.setattr('inference.pi0_wm.runtime.Runtime', Mock(return_value=runtime))

    def loop(adapter, args):
        assert args.auto_start and args.enable_commands is False
        return adapter.run(args.duration, read_key=None, single_step=False, start_paused=False)

    pi0_cli.main(['--config', str(mock_config), '--mock', '--auto-start', '--steps', '20',
                  '--duration', '1'], run_loop=loop)
    runtime.run.assert_called_once_with(20, duration_s=1., read_key=None,
                                       single_step=False, start_paused=False)


@pytest.mark.parametrize('section,key,value', [
    ('pi0', 'action_hz', 30), ('hardware', 'reset_q', [4.]*7),
    ('hardware', 'reset_speed_rad_s', -1.), ('hardware', 'reset_samples', 0),
])
def test_invalid_rate_or_reset_config_is_rejected_before_runtime(
    tmp_path, monkeypatch, section, key, value,
):
    config = config_fixture()
    config[section][key] = value
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(config))
    construct = Mock(side_effect=AssertionError('must not construct runtime'))
    monkeypatch.setattr('inference.pi0_wm.runtime.Runtime', construct)
    with pytest.raises(SystemExit) as error:
        cli.main(['--config', str(path), '--mock'])
    assert error.value.code == 2
    construct.assert_not_called()


@pytest.mark.parametrize('flag', ['--enable-command', '--enable-commands'])
def test_pi0_mock_can_never_enable_real_commands(flag, mock_config):
    with pytest.raises(SystemExit) as error:
        pi0_cli.main(['--config', str(mock_config), '--mock', flag])
    assert error.value.code == 2


def test_waiting_for_s_does_not_predict_enable_or_send():
    runtime = fixture(commands=True)
    result = runtime.run(10, duration_s=.05, start_paused=True)
    assert result['executed_steps'] == 0
    runtime._mock_pi.assert_not_called()
    runtime.wm.infer.assert_not_called()
    runtime.arm.enable.assert_not_called()
    runtime.arm.command_joint_positions.assert_not_called()
    assert not runtime.arm._connected


def test_s_starts_continuous_pi0_wm_execution():
    runtime = fixture()
    keys = iter([None, None, 's'])
    result = runtime.run(10, duration_s=1., start_paused=True,
                         read_key=lambda _: next(keys, None))
    assert result['executed_steps'] == 10
    assert runtime._mock_pi.call_count >= 3 and runtime.wm.infer.call_count >= 3
    assert any(np.max(call.args[0]) > 0 for call in runtime.arm.command_joint_positions.call_args_list)


def test_i_resets_then_waits_and_s_starts_a_fresh_episode():
    runtime = fixture()
    stage = {'phase': 'start'}

    def keys(_):
        if stage['phase'] == 'start':
            stage['phase'] = 'running'
            return 's'
        if stage['phase'] == 'running' and np.max(runtime.held) >= .03:
            stage['phase'] = 'resetting'
            return 'i'
        if stage['phase'] == 'resetting' and runtime.reset_progress is None:
            assert runtime.paused
            assert not runtime.plans.plans and runtime.execution.current is None
            np.testing.assert_allclose(runtime.state.q, runtime.reset_q, atol=.002)
            stage.update(phase='wait', ticks=0, calls=runtime._mock_pi.call_count)
        if stage['phase'] == 'wait':
            assert runtime._mock_pi.call_count == stage['calls']
            stage['ticks'] += 1
            if stage['ticks'] >= 3:
                stage['phase'] = 'resumed'
                return 's'
        if stage['phase'] == 'resumed' and np.max(runtime.held) >= .03:
            stage['phase'] = 'done'
            return 'q'
        return None

    runtime.run(100, duration_s=2., start_paused=True, read_key=keys)
    assert stage['phase'] == 'done'
    np.testing.assert_array_equal(runtime.reset_q, np.zeros(7))


@pytest.mark.parametrize('stop_key', ['i', 'p', 'q'])
def test_calibration_responds_to_stop_and_discards_blocked_pi_result(stop_key):
    runtime = fixture()
    entered, release = threading.Event(), threading.Event()

    def blocked_pi(obs):
        entered.set()
        assert release.wait(timeout=2.)
        return np.tile(obs['state'], (50, 1))

    runtime._mock_pi = Mock(side_effect=blocked_pi)
    stage = {'phase': 'start'}

    def keys(_):
        if stage['phase'] == 'start':
            stage['phase'] = 'predicting'
            return 's'
        if stage['phase'] == 'predicting' and entered.is_set():
            stage['phase'] = 'stopped'
            return stop_key
        if stage['phase'] == 'stopped':
            assert not runtime.plans.plans and runtime.execution.current is None
            assert runtime._mock_pi.call_count == 1
            return 'q'
        return None

    try:
        result = runtime.run(10, duration_s=1., start_paused=True, read_key=keys)
        assert stage['phase'] == 'stopped' and result['executed_steps'] == 0
        assert not runtime.arm._connected
    finally:
        release.set()
        if runtime.pi_worker is not None:
            runtime.pi_worker.thread.join(timeout=.5)
    assert runtime.pi_worker.poll() is None
    runtime.wm.infer.assert_not_called()


def test_readonly_i_does_not_enable_or_move_the_arm():
    runtime = fixture(readonly=True)
    keys = iter(['i', None, 'q'])
    result = runtime.run(10, start_paused=True, read_key=lambda _: next(keys))
    assert result['executed_steps'] == 0
    runtime.arm.enable.assert_not_called()
    runtime.arm.command_joint_positions.assert_not_called()


def test_single_step_only_consumes_a_valid_wm_prediction():
    runtime = fixture()
    stage = {'phase': 'start'}

    def keys(_):
        if stage['phase'] == 'start':
            stage['phase'] = 'running'
            return 's'
        if stage['phase'] == 'running' and runtime.paused:
            stage['phase'] = 'done'
            return 'q'
        return None

    result = runtime.run(10, duration_s=1., single_step=True, read_key=keys)
    assert stage['phase'] == 'done' and result['executed_steps'] == 1


def test_q_interrupts_reset_before_reaching_target():
    runtime = fixture()
    runtime.cfg['hardware']['reset_q'] = [.5] * 7
    keys = iter(['i', None, 'q'])
    result = runtime.run(10, start_paused=True, read_key=lambda _: next(keys))
    assert result['executed_steps'] == 0
    assert np.max(runtime.state.q) < .1
    runtime._mock_pi.assert_not_called()


def test_reset_timeout_cleans_up_without_starting_models():
    runtime = fixture(commands=True)
    runtime.cfg['hardware'].update(reset_q=[.5]*7, reset_timeout_s=.02)
    keys = iter(['i'])
    with pytest.raises(TimeoutError, match='reset did not settle'):
        runtime.run(10, start_paused=True, read_key=lambda _: next(keys, None))
    runtime._mock_pi.assert_not_called()
    assert not runtime.arm._connected
    assert runtime.pi_worker.closed and runtime.wm_worker.closed


@pytest.mark.parametrize('raises', [False, True])
def test_cancelled_worker_cannot_publish_or_run_two_model_calls(raises):
    entered, release = threading.Event(), threading.Event()
    active = {'count': 0, 'maximum': 0}

    def infer(value):
        active['count'] += 1
        active['maximum'] = max(active['maximum'], active['count'])
        try:
            if value == 'old':
                entered.set()
                assert release.wait(timeout=1.)
                if raises:
                    raise ValueError('old failed result')
            return value
        finally:
            active['count'] -= 1

    worker = Worker('pi0-test', infer)
    try:
        worker.submit(Request(1, 0, (), 'old', time.perf_counter()))
        assert entered.wait(timeout=1.)
        worker.cancel()
        assert worker.busy
        with pytest.raises(RuntimeError, match='in-flight'):
            worker.submit(Request(2, 0, (), 'new', time.perf_counter()))
        release.set()
        with worker.condition:
            assert worker.condition.wait_for(lambda: not worker.in_flight, timeout=.5)
        assert worker.poll() is None
        worker.submit(Request(2, 0, (), 'new', time.perf_counter()))
        with worker.condition:
            assert worker.condition.wait_for(lambda: worker.result is not None, timeout=.5)
        assert worker.poll().value == 'new'
        assert active['maximum'] == 1
    finally:
        release.set()
        worker.close()
