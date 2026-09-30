from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from gello_teleop.selected_joint_calibration import (
    begin_side, save_side, record_origin, infer_direction, update_direction, main,
)
from gello_teleop.uf_robot_gello_teleop import load_configs, JointMapper
from gello_teleop.dual_gello_collect import DualGelloPipeline

SOURCE = Path(__file__).resolve().parents[1]/'tests/fixtures/gello_dual.yaml'


def origin(entry):
    q = np.asarray(entry['RobotConfig']['reset_q'])+.1
    raw = np.r_[np.linspace(2., 3., 7), 1.]
    record_origin(entry, q, raw)
    return q, raw


def test_sign_inference_and_bad_manual_motion():
    raw = np.full(7, 2.)
    after = raw.copy(); after[3] -= np.deg2rad(14)
    assert infer_direction(raw, after, np.deg2rad(15), 3)[0] == -1
    after[3] += 2*np.deg2rad(14)
    assert infer_direction(raw, after, np.deg2rad(15), 3)[0] == 1
    after[1] += np.deg2rad(6)
    with pytest.raises(ValueError, match='其他'):
        infer_direction(raw, after, np.deg2rad(15), 3)
    with pytest.raises(ValueError, match='需要约'):
        infer_direction(raw, raw, np.deg2rad(15), 3)


def test_overwrite_one_side_preserves_other_and_recomputes_offset(tmp_path):
    output = tmp_path/'result.yaml'
    left = begin_side(SOURCE, output, 'left')
    q, raw = origin(left)
    after = raw[:7].copy(); after[3] -= np.deg2rad(15)
    update_direction(left, 3, -1, raw[:7], after, np.deg2rad(15))
    save_side(SOURCE, output, 'left', left)
    right = begin_side(SOURCE, output, 'right')
    origin(right)
    save_side(SOURCE, output, 'right', right)
    saved_right = deepcopy(yaml.safe_load(output.read_text())['right'])
    update_direction(left, 3, 1, raw[:7], after, np.deg2rad(15))
    save_side(SOURCE, output, 'left', left)
    assert yaml.safe_load(output.read_text())['right'] == saved_right
    _, robot, leader = load_configs(output, dual=True, side='left')[0]
    mapper = JointMapper(robot, leader); mapper.align(raw)
    np.testing.assert_allclose(mapper.target(raw)[0], q)
    assert leader.gripper_open_deg != leader.gripper_close_deg


def test_partial_result_rejected_before_pipeline_opens_devices(tmp_path):
    output = tmp_path/'result.yaml'
    entry = begin_side(SOURCE, output, 'left')
    origin(entry); save_side(SOURCE, output, 'left', entry)
    config = tmp_path/'pipeline.yaml'
    config.write_text(yaml.safe_dump(dict(gello_config=str(output))))
    with pytest.raises(ValueError, match='尚未全部标定'):
        DualGelloPipeline(config)


def test_complete_both_sides_reused_by_collector(tmp_path):
    output = tmp_path/'result.yaml'
    for side in ('left', 'right'):
        entry = begin_side(SOURCE, output, side)
        q, raw = origin(entry)
        for i in range(7):
            after = raw[:7].copy(); after[i] += np.deg2rad(15)
            update_direction(entry, i, 1, raw[:7], after, np.deg2rad(15))
        save_side(SOURCE, output, side, entry)
    config = tmp_path/'pipeline.yaml'
    config.write_text(yaml.safe_dump(dict(gello_config=str(output))))
    pipeline = DualGelloPipeline(config)
    assert len(pipeline.configs) == 2
    for _, robot, leader in pipeline.configs:
        mapped = (np.asarray(leader.leader_reference_q)-leader.joint_offsets)*leader.joint_signs
        np.testing.assert_allclose(mapped, robot.reset_q)
    pipeline.close()


@pytest.mark.parametrize('side,ip,other', [('left', '192.0.2.1', 'right'), ('right', '192.0.2.2', 'left')])
@pytest.mark.parametrize('damping_enabled', [False, True])
def test_cli_selects_only_requested_arm_j3_and_never_drives_gello(tmp_path, side, ip, other, damping_enabled):
    config = tmp_path/'dataset.yaml'
    workflow = dict(gello_config='result.yaml', gello_template=str(SOURCE))
    if damping_enabled:
        workflow['gello_damping'] = yaml.safe_load(Path('gello_teleop/config/xarm7_gello_dual_dataset.yaml').read_text())['gello_damping']
    config.write_text(yaml.safe_dump(workflow))
    q = np.array([.1, .2, .3, 1., .1, 1., .1])
    target = q.copy(); target[2] += np.deg2rad(15)
    raw = np.r_[np.full(7, 2.), 1.]
    moved = raw.copy(); moved[2] -= np.deg2rad(15)
    arm = Mock(); arm.api.axis = 7; arm.joints.return_value = q
    reader = Mock()
    with patch('gello_teleop.selected_joint_calibration.Arm', return_value=arm) as constructor, \
         patch('gello_teleop.selected_joint_calibration.GelloReader', return_value=reader), \
         patch('gello_teleop.selected_joint_calibration.stable_reference', side_effect=[(q, raw), (q, raw), (target, moved)]), \
         patch('gello_teleop.selected_joint_calibration._move_arm') as move, \
         patch('gello_teleop.selected_joint_calibration.CalibrationDamping') as damping, \
         patch('builtins.input', side_effect=['', '', '3', '', '', 'q']):
        damping.return_value.error = None
        damping.return_value.capabilities = {'enabled': True}
        assert main(['-c', str(config), '--'+side]) == 0
    if damping_enabled:
        assert damping.call_args.args[1].damping_enabled
        np.testing.assert_allclose(damping.return_value.start.call_args.args[0], raw[:7])
        damping.return_value.close.assert_called_once()
    else:
        damping.assert_not_called()
    assert constructor.call_count == 1
    assert constructor.call_args.args[0].robot_ip == ip
    np.testing.assert_allclose(move.call_args_list[0].args[1], target)
    np.testing.assert_allclose(move.call_args_list[1].args[1], q)
    reader.hold.assert_not_called(); reader.move.assert_not_called()
    reader.prepare_alignment.assert_called_once()
    data = yaml.safe_load((tmp_path/'result.yaml').read_text())
    assert data[side]['TeleoperatorConfig']['joint_signs'][2] == -1
    assert data[side]['CalibrationStatus']['direction_verified'] == [False, False, True, False, False, False, False]
    assert data[other] == yaml.safe_load(SOURCE.read_text())[other]
    arm.stop.assert_called_once(); reader.close.assert_called_once()


def test_calibration_damping_uses_encoder_coordinates_and_releases_on_failure():
    from dataclasses import replace
    from gello_teleop.selected_joint_calibration import CalibrationDamping
    from gello_teleop.gello_hardware import GelloReader
    leader = load_configs(SOURCE, dual=True, require_calibrated=False)[0][2]
    cfg = replace(leader, damping_gain=[8]*7, damping_brake_gain=[0]*7,
                  damping_current_limit=[15]*7, weak_hold_enabled=True)
    reader = Mock()
    worker = CalibrationDamping(reader, cfg, Mock())
    assert worker.config.joint_signs == (1,)*7
    assert not worker.config.weak_hold_enabled
    current = GelloReader.compute_damping_current(np.full(7, .02), np.zeros(7), .1, worker.config)
    assert np.all(current < 0)
    reader.read.side_effect = RuntimeError('USB lost')
    worker._run()
    assert isinstance(worker.error, RuntimeError)
    reader.disable_current_damping.assert_called_once()
    worker.on_failure.assert_called_once()


def test_calibration_damping_mode_switch_origin_change_aborts_and_releases():
    from gello_teleop.selected_joint_calibration import CalibrationDamping
    leader = load_configs(SOURCE, dual=True, require_calibrated=False)[0][2]
    reader = Mock()
    reader.read.return_value = np.full(8, 1.)
    worker = CalibrationDamping(reader, leader, Mock())
    with pytest.raises(ValueError, match='原点变化'):
        worker.start(np.zeros(7))
    reader.disable_current_damping.assert_called_once()
    assert worker.thread is None


@pytest.mark.parametrize('existing', [False, True])
def test_start_and_record_origin_do_not_touch_destination(tmp_path, existing):
    output = tmp_path/'result.yaml'
    previous = SOURCE.read_bytes()
    if existing:
        output.write_bytes(previous)
    entry = begin_side(SOURCE, output, 'right')
    origin(entry)
    if existing:
        assert output.read_bytes() == previous
    else:
        assert not output.exists()
    assert not output.with_suffix('.yaml.lock').exists()


def test_cli_quit_after_origin_preserves_previous_calibration(tmp_path):
    output = tmp_path/'result.yaml'; output.write_bytes(SOURCE.read_bytes())
    previous = output.read_bytes()
    config = tmp_path/'dataset.yaml'
    config.write_text(yaml.safe_dump(dict(gello_config='result.yaml', gello_template=str(SOURCE))))
    arm = Mock(); arm.api.axis = 7
    q = np.array([.1, .2, .3, 1., .1, 1., .1]); arm.joints.return_value = q
    with patch('gello_teleop.selected_joint_calibration.Arm', return_value=arm), \
         patch('gello_teleop.selected_joint_calibration.GelloReader'), \
         patch('gello_teleop.selected_joint_calibration.stable_reference', return_value=(q, np.r_[np.full(7, 2.), 1.])), \
         patch('builtins.input', side_effect=['', '', 'q']):
        assert main(['-c', str(config), '--right']) == 0
    assert output.read_bytes() == previous
