"""The pi0 latent branch shares committed action chunks and controller dispatch."""
from pathlib import Path

import pytest
import yaml

from inference.dp_wm.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / 'inference/configs/pi0_latent_wm_xarm.yaml'


def test_latent_configuration_preserves_pi0_stream_and_controller():
    config = load_config(CONFIG)
    assert config['pipeline'] == 'pi0_latent_wm'
    assert config['policy']['type'] == 'pi0'
    assert config['pi0']['open_loop'] is True
    assert config['pi0']['action_horizon'] == config['pi0']['consume_steps'] == 50
    assert 'latent_wm/checkpoints' in str(config['wm']['checkpoint'])
    assert config['wm']['flow_steps'] == 32
    assert config['controller']['mode'] == 'mit_position'
    assert config['controller']['execution'] == 'active'


def test_named_pi0_v2_config_uses_new_weights():
    config = load_config(CONFIG.parent / 'pi0_wm_v2_xarm.yaml')
    assert config['pipeline'] == 'pi0_latent_wm'
    assert config['wm']['model_version'] == 'latent_carswm_lstm_v2'
    assert 'latent_wm_v2' in str(config['wm']['checkpoint'])
    assert config['pi0']['open_loop'] is True
    assert config['execution']['reset_on_start'] is False


def test_pi0_latent_loads_latent_adapter_and_does_not_load_contact_or_dp(monkeypatch):
    from inference.dp_wm import models
    from inference.policies.pi0_joint import Pi0JointPolicy
    config = load_config(CONFIG)
    sentinel = object()
    received = []
    monkeypatch.setattr(models, 'LatentWM', lambda wm: received.append(wm) or sentinel)
    monkeypatch.setattr(models, 'ContactWM', lambda _: pytest.fail('Contact WM must not load'))
    policy, wm = models.load_models(config, {})
    try:
        assert isinstance(policy, Pi0JointPolicy)
        assert policy.client is None
        assert wm is sentinel
        assert received == [config['wm']]
    finally:
        policy.close()


def test_latent_checkpoint_metadata_rejects_contact_weights(monkeypatch):
    import torch
    from inference.dp_wm.models import inspect_checkpoints
    config = load_config(CONFIG)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'model_version': 'carswm_v9'})
    with pytest.raises(ValueError, match='Expected latent_carswm_lstm_v1'):
        inspect_checkpoints(config)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('entry', ['common', 'pi0_script'])
def test_latent_public_entry_dispatches_enabled_or_direct_branch(tmp_path, monkeypatch, enabled, entry):
    from inference import cli, joint_cli
    from inference.dp_wm import cli as wm_cli
    from inference.pi0_wm import cli as pi0_cli
    document = yaml.safe_load(CONFIG.read_text())
    document['wm']['enabled'] = enabled
    path = tmp_path / 'pi0_latent.yaml'
    path.write_text(yaml.safe_dump(document))
    calls = []
    monkeypatch.setattr(joint_cli, 'main', lambda argv, **kwargs: calls.append(('direct', argv)))
    monkeypatch.setattr(wm_cli, 'main', lambda argv: calls.append(('wm', argv)))
    (cli.main if entry == 'common' else pi0_cli.main)(['--config', str(path), '--check'])
    assert calls == [('wm' if enabled else 'direct', ['--config', str(path), '--check'])]


def test_latent_cli_uses_aligned_open_loop_stream_and_forecasts(tmp_path, monkeypatch):
    from inference.dp_wm import cli
    from inference.dp_wm.core import AsyncDPWM
    config = load_config(CONFIG)
    config['logging']['output_dir'] = tmp_path
    config['controller']['execution'] = 'shadow'
    monkeypatch.setattr(cli, 'load_config', lambda _: config)
    captured = []
    def capture(*args, **kwargs):
        pipeline = AsyncDPWM(*args, **kwargs)
        captured.append(pipeline)
        return pipeline
    monkeypatch.setattr(cli, 'AsyncDPWM', capture)
    report = cli.main(['--config', str(CONFIG), '--run', '--mock', '--auto-start', '--duration', '.2'])
    assert captured[0].aligned_actions
    assert captured[0].policy_open_loop
    assert report['policy_chunk_steps'] == 50
    assert report['dp_plans'] >= 1
    assert report['wm_predictions'] >= 1
    assert report['controller']['counts']['valid'] >= 1



def test_disabled_latent_deployment_reuses_direct_pi0_camera_and_execution_settings(tmp_path):
    from inference.joint_config import JointDeployment, load_joint_policy
    document = yaml.safe_load(CONFIG.read_text())
    document['wm']['enabled'] = False
    document['wm']['checkpoint'] = '/missing/latent.pt'
    for section, keys in [('policy', ('client_source_path',)),
                          ('observation', ('hardware_config',)),
                          ('pi0', ('local_source_path', 'local_checkpoint'))]:
        for key in keys:
            document[section][key] = str((CONFIG.parent / document[section][key]).resolve())
    path = tmp_path / 'direct.yaml'
    path.write_text(yaml.safe_dump(document))
    deployment = JointDeployment.load(path)
    policy, mapping = load_joint_policy(deployment)
    try:
        assert policy.client is None
        assert policy.action_steps == 50
        assert deployment.execution['open_loop'] is True
        assert deployment.active_arms == ('right',)
        assert mapping == document['observation']['camera_map']
    finally:
        policy.close()
