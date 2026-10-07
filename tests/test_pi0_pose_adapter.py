from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from inference.joint_config import JointDeployment, load_joint_policy, policy_report
from inference.policies.errors import JointActionContractError
from inference.policies.pi0_joint import Pi0JointPolicy
from inference.pose_action_adapter import PoseActionAdapter
from test_pi0_joint_interface import Client, observation, pi_config, pi_deployment


URDF = Path(__file__).resolve().parents[1] / 'gello_teleop/models/xarm7_dynamics.urdf'
SEED = np.array([.1, -.3, .2, 1.3, -.1, 1.2, .2])


def pose_config(order='xyzw', frame='link7'):
    config = pi_config()
    config['action_type'] = 'eepose'
    config['interface'].update(action_semantic='action.ee_pose', coordinate_frame=frame,
                               representation='xyz_quaternion', quaternion_order=order,
                               units='m', reference_frame='base')
    config['interface'].pop('joint_names')
    return config


def adapter(config=None, **settings):
    pytest.importorskip('pinocchio')
    return PoseActionAdapter({'urdf_path': URDF, **settings}, (config or pose_config())['interface'])


@pytest.mark.parametrize('order', ['xyzw', 'wxyz'])
def test_real_fk_ik_sequence_reproduces_poses_and_seeds_each_step_from_previous_solution(order):
    config = pose_config(order)
    model = adapter(config)
    poses = np.stack([model.forward_pose(SEED + np.array([.005*i, 0, 0, 0, 0, 0, 0])) for i in range(20)])
    seeds = []
    original = model.solve

    def solve(seed, target):
        seeds.append(seed.copy())
        return original(seed, target)

    model.solve = solve
    joints, diagnostics = model.convert(poses, SEED + .02)
    assert joints.shape == (20, 7)
    np.testing.assert_array_equal(seeds[0], SEED + .02)
    np.testing.assert_allclose(seeds[1:], joints[:-1])
    for q, expected in zip(joints, poses):
        actual = model.forward_pose(q)
        assert np.linalg.norm(actual[:3] - expected[:3]) <= model.position_tolerance_m
        # Quaternions may have the equivalent opposite sign.
        assert min(np.linalg.norm(actual[3:] - expected[3:]),
                   np.linalg.norm(actual[3:] + expected[3:])) < model.rotation_tolerance_rad
    assert all(d['position_error_m'] <= 1e-4 for d in diagnostics)


def test_tcp_offset_translation_and_rotation_are_applied_in_flange_frame():
    config = pose_config(frame='tcp')
    offset = [.02, -.01, .08, .1, -.2, .3]
    model = adapter(config, frame_name='link_eef', frame_offset_xyz_rpy=offset)
    target = model.forward_pose(SEED)
    joints, _ = model.convert(target[None], SEED + .03)
    recovered = model.forward_pose(joints[0])
    np.testing.assert_allclose(recovered[:3], target[:3], atol=1e-4)
    assert abs(np.dot(recovered[3:], target[3:])) > .999999
    parent = model.model.snapshot(SEED, np.zeros(7)).pose
    transform = np.eye(4)
    transform[:3, 3] = offset[:3]
    transform[:3, :3] = Rotation.from_euler('xyz', offset[3:]).as_matrix()
    np.testing.assert_allclose(target[:3], (parent @ transform)[:3, 3])


@pytest.mark.parametrize('quat', [[0., 0., 0., 0.], [1., 1., 1., 1.]])
def test_bad_quaternion_is_rejected_instead_of_treating_joint_angles_as_pose(quat):
    model = adapter()
    pose = model.forward_pose(SEED)
    pose[3:] = quat
    with pytest.raises(JointActionContractError, match='quaternion'):
        model.convert(pose[None], SEED)


def test_unreachable_second_target_returns_no_partial_chunk():
    model = adapter(max_iterations=5)
    good = model.forward_pose(SEED)
    bad = good.copy()
    bad[:3] = [10., 10., 10.]
    with pytest.raises(JointActionContractError, match='step 1.*IK failed'):
        model.convert(np.stack([good, bad]), SEED)


def test_joint_bounds_constrain_ik_and_frame_mismatch_requires_offset():
    model = adapter()
    limits = np.column_stack([SEED-.001, SEED+.001])
    constrained = PoseActionAdapter({'urdf_path': URDF}, pose_config()['interface'], joint_limits=limits)
    target = model.forward_pose(SEED + np.array([.1, 0, 0, 0, 0, 0, 0]))
    with pytest.raises(JointActionContractError, match='IK failed'):
        constrained.convert(target[None], SEED)
    with pytest.raises(ValueError, match='frame_offset'):
        adapter(pose_config(frame='tcp'), frame_name='link7')


def test_pose_policy_uses_aligned_measured_q_and_returns_only_joint_chunk():
    config = pose_config()
    model = adapter(config)
    client = Client(config)
    client.actions = np.tile(model.forward_pose(SEED), (50, 1))
    raw = client.actions.copy()
    policy = Pi0JointPolicy(config, client_factory=lambda _: client)
    policy.pose_adapter = model
    obs = observation()
    object.__setattr__(obs, 'q', SEED + .02)
    chunk = policy.predict_observations((obs,))
    assert chunk.semantic == 'joint' and chunk.frame_name is None
    assert chunk.values.shape == (20, 7)
    assert chunk.timestamp_us == obs.timestamp_us
    assert chunk.metadata['model_action_type'] == 'eepose'
    assert len(chunk.metadata['ik']) == 20
    np.testing.assert_array_equal(client.actions, raw)
    np.testing.assert_array_equal(client.requests[0]['observation/state'], np.zeros(7))
    assert np.linalg.norm(model.forward_pose(chunk.values[0])[:3] - raw[0, :3]) < 1e-4


@pytest.mark.parametrize('key,value', [('units', 'mm'), ('quaternion_order', 'bad'),
                                      ('action_semantic', 'action.joint'), ('coordinate_frame', None)])
def test_pose_profile_rejects_inconsistent_physical_contract(key, value):
    config = pose_config()
    config['interface'][key] = value
    with pytest.raises(JointActionContractError):
        Pi0JointPolicy(config)


def test_config_load_routes_joint_without_ik_and_pose_with_validated_model(tmp_path, monkeypatch):
    path = pi_deployment(tmp_path)
    config = JointDeployment.load(path)
    policy, mapping = load_joint_policy(config)
    assert policy.action_type == 'joint' and policy.pose_adapter is None
    document = yaml.safe_load(path.read_text())
    document['pi0'] = pose_config()
    document['ik'] = {'urdf_path': str(URDF), 'frame_name': 'link7'}
    path.write_text(yaml.safe_dump(document))
    config = JointDeployment.load(path)
    policy, mapping = load_joint_policy(config)
    assert policy.action_type == 'eepose' and policy.pose_adapter is not None
    assert 'local IK' in policy_report(config, policy, mapping)['action']
    from inference.cli import main
    monkeypatch.setattr('inference.policies.pi0_joint.OpenPiConnection',
                        lambda _: pytest.fail('check must not connect to server'))
    report = main(['--config', str(path), '--check'])
    assert report['model_action_type'] == 'eepose'


def test_server_joint_metadata_cannot_be_used_with_pose_config():
    config = pose_config()
    client = Client(pi_config())
    policy = Pi0JointPolicy(config, client_factory=lambda _: client)
    with pytest.raises(JointActionContractError, match='interface mismatch'):
        policy.verify_server()
    assert client.closed
