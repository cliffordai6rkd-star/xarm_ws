"""PI0 plans feed the native Contact WM bridge without retiming delayed results."""
from pathlib import Path

import numpy as np
import pytest

from inference.dp_wm.config import load_config
from inference.dp_wm.core import ActionPlan, AsyncDPWM, HistorySnapshot
from inference.dp_wm.mock import MockSource, MockWM
from inference.policies.pi0_joint import MockPiConnection, Pi0JointPolicy

CONFIG = Path(__file__).resolve().parents[1] / 'inference/configs/pi0_wm_xarm.yaml'


def configuration_file(directory, *, consume=None):
    import yaml
    config = yaml.safe_load(CONFIG.read_text())
    for section, keys in [('policy', ('client_source_path',)),
                          ('wm', ('pinn_root', 'checkpoint', 'urdf_path')),
                          ('observation', ('hardware_config',)),
                          ('logging', ('output_dir',)),
                          ('pi0', ('local_source_path', 'local_checkpoint'))]:
        for key in keys:
            config[section][key] = str((CONFIG.parent / config[section][key]).resolve())
    config['pi0']['action_horizon'] = 50
    config['pi0']['open_loop'] = False  # Legacy replacement mode tests opt in explicitly below.
    config['observation']['command_source'] = 'controller_reference'
    config['controller']['maximum_forecast_age_ms'] = 180.  # Legacy replacement fixture.
    config['wm']['enabled'] = True
    if consume is not None:
        config['pi0']['consume_steps'] = consume
    path = directory / 'pi0.yaml'
    path.write_text(yaml.safe_dump(config))
    return path


def policy_fixture():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        config = load_config(configuration_file(Path(directory)))
    policy = Pi0JointPolicy({**config['pi0'], 'local': False}, client_factory=MockPiConnection)
    policy.allow_mock_server = True
    return config, policy


def test_pi0_full_plan_keeps_camera_keys_and_absolute_time():
    config, policy = policy_fixture()
    source = MockSource(image_keys=policy.image_keys)
    observations, _ = source.poll()
    chunk = policy.predict_observations(observations, plan_steps=50)
    assert chunk.values.shape == (50, 7)
    assert chunk.timestamp_us == observations[-1].timestamp_us
    assert chunk.step_s == .04
    plan = ActionPlan(chunk, 1, plan_steps=50)
    anchor = chunk.timestamp_us + 1_000_000
    actions, times = plan.future(anchor, aligned=True)
    assert actions.shape == (8, 7)
    np.testing.assert_array_equal(times, anchor + np.arange(1, 9) * 40_000)
    assert plan.covers(anchor, aligned=True)
    assert plan.future(chunk.timestamp_us + 1_640_001, aligned=True) is None
    with pytest.raises(ValueError, match='plan_steps'):
        policy.predict_observations(observations, plan_steps=51)
    policy.close()


def test_pi0_delayed_result_can_feed_contact_wm():
    _, policy = policy_fixture()
    scheduler = AsyncDPWM(policy, MockWM(), plan_steps=50, aligned_actions=True)
    observations, _ = MockSource(image_keys=policy.image_keys).poll()
    try:
        # Finish the worker deterministically, then poll at a one-second delay.
        scheduler.dp.submit(observations[-1:])
        scheduler.dp.future.result(timeout=2)
        anchor = observations[-1].timestamp_us + 1_000_000
        history = HistorySnapshot(anchor + np.arange(-49, 1) * 10_000,
            {key: np.zeros((50, 7)) for key in ('q', 'dq', 'tau', 'delta_q')})
        events = scheduler.poll(anchor, history=history)
        assert events[0]['usable_now']
        assert scheduler.wm.busy
        result, _ = scheduler.wm.future.result(timeout=2)
        assert result['plan_version'] == 1
        assert result['q'].shape == (1, 32, 7)
        np.testing.assert_array_equal(result['future_timestamps_us'], anchor + np.arange(1, 33) * 10_000)
    finally:
        scheduler.close()
        policy.close()


def test_configuration_selects_contact_checkpoint_and_three_camera_mapping():
    config, policy = policy_fixture()
    assert config['pipeline'] == 'pi0_contact_wm'
    assert config['wm']['checkpoint'].parent.parent.name == 'contact_wm'
    assert config['wm']['checkpoint'].suffix == '.pt'
    assert config['wm']['enabled']
    assert config['controller']['execution'] in ('shadow', 'active')
    assert config['controller']['maximum_forecast_age_ms'] == 180.
    assert set(config['observation']['camera_map']) == set(policy.image_keys)
    assert 1 <= config['pi0']['consume_steps'] <= config['pi0']['action_horizon']


@pytest.mark.parametrize('key,value', [('scheduling', 'invalid'), ('inference_budget_ms', 0),
                                     ('inference_budget_ms', float('nan')),
                                     ('inference_budget_ms', True), ('action_sampling', 'invalid')])
def test_config_rejects_invalid_continuous_wm_options(tmp_path, key, value):
    import yaml
    path = configuration_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document['wm'][key] = value
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=f'wm.{key}'):
        load_config(path)


def test_cli_passes_continuous_scheduling_and_recorded_tokens_to_worker(tmp_path, monkeypatch):
    import yaml
    from inference.dp_wm import cli

    path = configuration_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document['pi0']['open_loop'] = True
    document['wm'].update(scheduling='continuous', inference_budget_ms=100.,
                          action_sampling='recorded_tokens')
    document['controller'].update(execution='shadow', maximum_forecast_age_ms=240.)
    document['logging']['output_dir'] = str(tmp_path / 'runs')
    path.write_text(yaml.safe_dump(document))
    captured = []
    def capture(*args, **kwargs):
        pipeline = AsyncDPWM(*args, **kwargs)
        captured.append(pipeline)
        return pipeline
    monkeypatch.setattr(cli, 'AsyncDPWM', capture)
    report = cli.main(['--config', str(path), '--mock', '--auto-start', '--duration', '.1'])
    assert captured[0].scheduling == 'continuous'
    assert captured[0].action_sampling == 'recorded_tokens'
    assert report['wm_inference_budget_ms'] == 100.
    assert report['target_wm_hz'] is None


@pytest.mark.parametrize('consume', [1, 8, 50])
def test_pi0_consumption_prefix_is_independent_of_full_wm_plan(tmp_path, consume):
    path = configuration_file(tmp_path, consume=consume)
    loaded = load_config(path)
    policy = Pi0JointPolicy({**loaded['pi0'], 'local': False}, client_factory=MockPiConnection)
    policy.allow_mock_server = True
    scheduler = AsyncDPWM(policy, MockWM(), plan_steps=50, aligned_actions=True, policy_refresh_steps=consume)
    try:
        observations, _ = MockSource(image_keys=policy.image_keys).poll()
        scheduler.dp.submit(observations)
        scheduler.dp.future.result(timeout=2)
        events = scheduler.poll(observations[-1].timestamp_us, active=False)
        assert events[0]['usable_now']
        assert scheduler.plan.chunk.values.shape == (50, 7)
        assert scheduler.policy_period_us == consume*40_000
        assert scheduler.policy_refresh_period_us() <= 1_640_000 - 30_000 - scheduler.margin_us
        if consume == 8:
            assert scheduler.next_dp_us == observations[-1].timestamp_us + 320_000
    finally:
        scheduler.close()
        policy.close()


def test_eight_total_actions_do_not_cover_the_wm_future_window():
    from inference.core.contracts import ActionChunk
    plan = ActionPlan(ActionChunk(np.zeros((8, 7)), 'joint', None, 1_000_000, .04), 1, plan_steps=8)
    assert plan.future(1_000_000, aligned=True) is None
    assert not plan.covers(1_000_000, aligned=True)



def test_config_rejects_short_trained_horizon_with_lookahead_explanation(tmp_path):
    import yaml
    path = configuration_file(tmp_path, consume=8)
    document = yaml.safe_load(path.read_text())
    document['pi0']['action_horizon'] = 8
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match='current token plus eight future WM actions'):
        load_config(path)


@pytest.mark.parametrize('settings', [dict(open_loop='true'),
                                    dict(open_loop=True, consume_steps=8),
                                    dict(prefetch_margin_ms=-1),
                                    dict(prefetch_margin_ms=float('nan')),
                                    dict(prefetch_margin_ms=True)])
def test_config_rejects_invalid_open_loop_settings(tmp_path, settings):
    import yaml
    path = configuration_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document['pi0'].update(settings)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match='pi0\\.(open_loop|prefetch_margin_ms)'):
        load_config(path)



def test_eight_step_prefix_requests_new_plan_after_320_ms():
    _, policy = policy_fixture()
    scheduler = AsyncDPWM(policy, MockWM(), plan_steps=50, aligned_actions=True, policy_refresh_steps=8)
    observations, _ = MockSource(image_keys=policy.image_keys).poll()
    anchor = observations[-1].timestamp_us
    try:
        scheduler.poll(anchor, observations[-1:])
        scheduler.dp.future.result(timeout=2)
        scheduler.poll(anchor + 80_000)
        assert scheduler.counts['dp_requests'] == 1
        assert scheduler.plan.chunk.values.shape == (50, 7)
        observations[-1].timestamp_us = anchor + 280_000
        scheduler.poll(anchor + 319_000, observations[-1:])
        assert scheduler.counts['dp_requests'] == 1
        observations[-1].timestamp_us = anchor + 320_000
        scheduler.poll(anchor + 320_000, observations[-1:])
        assert scheduler.counts['dp_requests'] == 2
        scheduler.dp.future.result(timeout=2)
    finally:
        scheduler.close()
        policy.close()



@pytest.mark.parametrize('entry', ['common', 'pi0_script'])
def test_disabled_wm_routes_to_direct_pi0_without_loading_wm(tmp_path, monkeypatch, entry):
    import yaml
    from inference import cli, joint_cli
    from inference.pi0_wm import cli as pi0_cli
    path = configuration_file(tmp_path, consume=8)
    document = yaml.safe_load(path.read_text())
    document['wm']['enabled'] = False
    # WM files need not exist when disabled.
    document['wm']['checkpoint'] = '/missing/wm.pt'
    path.write_text(yaml.safe_dump(document))
    captured = []
    monkeypatch.setattr(joint_cli, 'main', lambda argv, **kwargs: captured.append(argv))
    (cli.main if entry == 'common' else pi0_cli.main)(['--config', str(path), '--check'])
    assert captured == [['--config', str(path), '--check']]


def test_disabled_wm_uses_eight_step_direct_pi0_policy(tmp_path):
    import yaml
    from inference.joint_config import JointDeployment, load_joint_policy
    path = configuration_file(tmp_path, consume=8)
    document = yaml.safe_load(path.read_text())
    document['wm']['enabled'] = False
    document['wm']['checkpoint'] = '/missing/wm.pt'
    path.write_text(yaml.safe_dump(document))
    deployment = JointDeployment.load(path)
    policy, mapping = load_joint_policy(deployment)
    policy.client_factory = MockPiConnection
    policy.allow_mock_server = True
    try:
        observations, _ = MockSource(image_keys=policy.image_keys).poll()
        chunk = policy.predict_observations(observations)
        assert chunk.values.shape == (8, 7)
        assert policy.horizon == 50 and policy.action_steps == 8
        assert deployment.active_arms == ('right',)
        assert deployment.execution['open_loop']
        assert deployment.observation['maximum_camera_age_s'] == .15
        assert mapping == document['observation']['camera_map']
    finally:
        policy.close()
