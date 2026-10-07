"""Joint-action/WM contracts and CUDA replay; no hardware is used."""

from pathlib import Path

import pytest
import torch

from inference.dp_wm.models import JointToPose, joint_action_frame


URDF = Path(__file__).resolve().parents[1] / 'gello_teleop/models/xarm7_dynamics_g2.urdf'


def test_recorded_contact_tokens_keep_their_phase_instead_of_interpolating():
    import numpy as np
    from inference.dp_wm.models import validate_contact_action_times, validate_action_sampling

    times = 1_000_000 + np.arange(1,9,dtype=np.int64)*40_000
    validate_contact_action_times(1_020_000, times, 'recorded_tokens')
    with pytest.raises(ValueError, match='history-anchor'):
        validate_contact_action_times(1_020_000, times, 'anchor_grid')
    with pytest.raises(ValueError, match='next committed token'):
        validate_contact_action_times(1_040_000, times, 'recorded_tokens')
    invalid = times.copy()
    invalid[-1] += 10_000
    with pytest.raises(ValueError, match='consecutive'):
        validate_contact_action_times(1_020_000, invalid, 'recorded_tokens')
    contract = {'dataloader': dict(backend='lerobot', v3_only=True, action_start_offset=1,
                                   inference_delay_s=0., action_condition_mode='direct')}
    validate_action_sampling(contract, dict(action_sampling='recorded_tokens'))
    contract['dataloader']['inference_delay_s'] = .1
    with pytest.raises(ValueError, match='no action delay'):
        validate_action_sampling(contract, dict(action_sampling='recorded_tokens'))


def test_checkpoint_resolves_v3_sampling_independently_of_fk_or_position_encoding():
    from inference.dp_wm.models import resolve_action_sampling
    contract = {'dataloader': dict(backend='lerobot', v3_only=True, action_start_offset=1,
        inference_delay_s=0., action_condition_mode='direct',
        action_index_key='timing.action_index', action_anchor_timestamp_key='timing.action_anchor_timestamp_ns')}
    assert resolve_action_sampling(contract, {'action_sampling':'checkpoint'}) == 'recorded_tokens'
    with pytest.raises(ValueError, match='changes trained action indices'):
        resolve_action_sampling(contract, {'action_sampling':'anchor_grid'})
    contract['dataloader']['action_index_key'] = ''
    with pytest.raises(ValueError, match='V3 indexed actions'):
        resolve_action_sampling(contract, {'action_sampling':'checkpoint'})


def deployment(**overrides):
    return {'pinn_root': URDF.parent, 'urdf_path': URDF, **overrides}


def training(*, recorded):
    if recorded:
        return {'action_contract': {'type': 'absolute_ee_pose', 'coordinate_frame': 'link7',
                                    'semantic': 'expert_policy_action_target',
                                    'quaternion_order': 'xyzw', 'absolute_or_relative': 'absolute'},
                'dataloader': {'action_key': 'action.ee_pose', 'action_fk': None}}
    return {'action_contract': {'type': 'absolute_ee_pose', 'coordinate_frame': 'link_base',
                                'quaternion_order': 'xyzw', 'absolute_or_relative': 'absolute'},
            'dataloader': {'action_key': 'action.ee_pose', 'action_fk': {
                'base_frame': 'link_base', 'frame_name': 'link_eef', 'urdf_path': '/missing/training.urdf',
                'joint_names': [f'joint{i}' for i in range(1, 8)]}}}


def test_checkpoint_fk_and_recorded_link7_require_distinct_explicit_adapters():
    assert joint_action_frame(training(recorded=False), deployment()) == 'link_eef'
    assert joint_action_frame(training(recorded=False), deployment(
        training_urdf_path=URDF.parent / 'xarm7_dynamics.urdf')) == 'link_eef'
    assert joint_action_frame(training(recorded=True), deployment(
        joint_action_adapter='joint_fk_link7_approx')) == 'link7'
    with pytest.raises(ValueError, match='joint_fk_link7_approx'):
        joint_action_frame(training(recorded=True), deployment())
    with pytest.raises(ValueError, match='checkpoint FK'):
        joint_action_frame(training(recorded=False), deployment(
            joint_action_adapter='joint_fk_link7_approx'))
    with pytest.raises(FileNotFoundError):
        joint_action_frame(training(recorded=False), deployment(
            training_urdf_path=URDF.parent / 'missing.urdf'))


def test_recorded_action_adapter_rejects_wrong_semantics_and_missing_frame():
    data = training(recorded=True)
    data['action_contract']['semantic'] = 'va_measured_joint_snapshot_fk'
    with pytest.raises(ValueError, match='matching joint-action adapter'):
        joint_action_frame(data, deployment(joint_action_adapter='joint_fk_link7_approx'))
    with pytest.raises(ValueError, match='Missing FK frame'):
        JointToPose(URDF, frame_name='missing_link')


def test_v2_latent_checkpoint_requires_an_explicit_v2_deployment(monkeypatch):
    from inference.dp_wm.config import load_config
    from inference.dp_wm.models import inspect_checkpoints

    root = Path(__file__).resolve().parents[1]
    old = load_config(root / 'inference/configs/pi0_latent_wm_xarm.yaml')
    new = load_config(root / 'inference/configs/pi0_latent_wm_v2_xarm_shadow.yaml')
    payload = {'model_version': 'latent_carswm_lstm_v2', 'config': training(recorded=False)}
    payload['config']['dataloader'].update(high_fps=100, expert_fps=25,
        state_history_horizon=50, prediction_horizon=32, action_condition_horizon=8,
        action_start_offset=1)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: payload)
    with pytest.raises(ValueError, match='Expected latent_carswm_lstm_v1'):
        inspect_checkpoints(old)
    result = inspect_checkpoints(new)
    assert result['wm_model_version'] == 'latent_carswm_lstm_v2'
    assert result['wm_action_frame'] == 'link_eef'


@pytest.mark.parametrize('version,aligned', [
    ('latent_carswm_lstm_v1', False), ('latent_carswm_lstm_v2', True)])
def test_dp_latent_scheduler_uses_v2_anchor_grid(tmp_path, monkeypatch, version, aligned):
    from inference.dp_wm import cli
    from inference.dp_wm.config import load_config
    from inference.dp_wm.core import AsyncDPWM

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'inference/configs/dp_latent_wm_v2_xarm_shadow.yaml')
    config['wm']['model_version'] = version
    config['logging']['output_dir'] = tmp_path
    monkeypatch.setattr(cli, 'load_config', lambda _: config)
    captured = []

    def capture(*args, **kwargs):
        pipeline = AsyncDPWM(*args, **kwargs)
        captured.append(pipeline)
        return pipeline

    monkeypatch.setattr(cli, 'AsyncDPWM', capture)
    cli.main(['--config', 'unused', '--run', '--mock', '--duration', '.05'])
    assert captured[0].aligned_actions is aligned
    assert captured[0].wait_for_window_exhaustion is True
    assert captured[0].policy_period_us == 320_000


def test_active_eight_step_dp_requires_bounded_gap_hold(tmp_path):
    import yaml
    from inference.dp_wm.config import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'inference/configs/dp_contact_wm_xarm.yaml')
    config['controller'].pop('gap_hold')
    def portable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        return value
    path = tmp_path / 'active_without_hold.yaml'
    path.write_text(yaml.safe_dump(portable(config)))
    with pytest.raises(ValueError, match='gap_hold.enabled'):
        load_config(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires a GPU')
def test_conditional_cuda_replay_refreshes_decode_history():
    from inference.dp_wm.cuda_sampler import CudaFlowSampler

    class ConditionalModel:
        conditional_codec = True
        future_horizon = 2
        latent_dim = 7
        latent_mean = torch.zeros(7, device='cuda')
        latent_std = torch.ones(7, device='cuda')

        def encode_conditions(self, batch):
            return {'history': batch['q'][:, -1:], 'action': batch['q'][:, -1:],
                    'action_padding_mask': torch.zeros((1, 1), device='cuda', dtype=torch.bool),
                    'future_pe': torch.zeros((1, 2, 7), device='cuda')}

        def velocity(self, state, time, encoded):
            return torch.zeros_like(state)

        def decode(self, raw, batch):
            return {'q_pred': raw + batch['q'][:, -1, None, None, :]}

    sampler = CudaFlowSampler(ConditionalModel(), steps=2, solver='heun', num_samples=1)
    batch = {key: torch.zeros((1, 50, 7), device='cuda') for key in ('q', 'dq', 'delta_q', 'tau')}
    noise = torch.zeros((1, 1, 2, 7), device='cuda')
    first = sampler.sample(batch, source_noise=noise)['q_pred'].clone()
    batch['q'][:, -1] = 0.4
    second = sampler.sample(batch, source_noise=noise)['q_pred'].clone()
    torch.testing.assert_close(first, torch.zeros_like(first))
    torch.testing.assert_close(second, torch.full_like(second, 0.4))
