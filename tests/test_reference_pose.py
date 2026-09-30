from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from gello_teleop.capture_reference_pose import main
from gello_teleop.dual_gello_collect import DualGelloPipeline
from gello_teleop.reference_pose import capture_live_pose, live_pose_path, overwrite_reference_pose
from scripts.smoke_dual_gello_pipeline import SimulatedArm, write_simulation_config

ROOT = Path(__file__).resolve().parents[1]


class StaticLeader:
    def __init__(self, config):
        self.config = config
        self.raw = np.r_[config.leader_reference_q, np.deg2rad(config.gripper_open_deg or 0.)]

    def read(self):
        return self.raw.copy()

    def prepare_alignment(self):
        pass

    def set_torque(self, *args, **kwargs):
        pass

    def close(self):
        pass


@pytest.fixture
def live_pipeline(tmp_path, request):
    path = write_simulation_config(ROOT/'gello_teleop/config/xarm7_gello_dual_dataset.yaml', tmp_path)
    workflow = yaml.safe_load(path.read_text())
    workflow['active_arms'] = ['left', 'right']
    workflow['gripper']['mode'] = 'follow'
    workflow.update(cameras=[], require_cameras=False, gripper={'enabled': False, 'command_enabled': False})
    workflow['alignment']['alignment_samples'] = 1
    path.write_text(yaml.safe_dump(workflow))
    calibration = Path(workflow['gello_config'])
    saved = yaml.safe_load(calibration.read_text())
    for side in ('left', 'right'):
        saved[side]['CalibrationStatus']['checks'] = {'1': {'new_sign': 1, 'method': 'previous direction test'}}
        saved[side]['TeleoperatorConfig'].update(gripper_open_deg=180., gripper_close_deg=138.)
    calibration.write_text(yaml.safe_dump(saved))
    pipeline = DualGelloPipeline(path, arm_factory=SimulatedArm, reader_factory=StaticLeader)
    try:
        pipeline.connect(); pipeline.reset(); pipeline.confirm_alignment(); pipeline.takeover()
        if getattr(request, 'param', None) != 'following':
            pipeline.freeze_following()
            for i, (arm, reader) in enumerate(zip(pipeline.arms, pipeline.readers)):
                arm.q += .15*(i+1)
                reader.raw[:7] += .2*(i+1)
        deadline = time.monotonic()+1.
        while not live_pose_path(path).exists():
            if time.monotonic() > deadline:
                raise AssertionError('live feedback publisher did not start')
            time.sleep(.005)
        # Allow both buses to publish a sample from the new stationary pose.
        time.sleep(.07)
        yield pipeline
    finally:
        pipeline.close()


def test_capture_uses_actual_feedback_and_new_collector_starts_at_saved_pose(live_pipeline):
    pipeline = live_pipeline
    before = yaml.safe_load(pipeline.gello_config_path.read_text())
    for arm, reader in zip(pipeline.arms, pipeline.readers):
        arm.command_joint_positions = Mock(side_effect=arm.command_joint_positions)
        reader.set_torque = Mock()
        reader.prepare_alignment = Mock()
    pose = capture_live_pose(pipeline.config_path)
    destination = overwrite_reference_pose(pose)
    saved = yaml.safe_load(destination.read_text())
    for side, arm, reader in zip(('left', 'right'), pipeline.arms, pipeline.readers):
        np.testing.assert_allclose(pose['arms'][side]['xarm_q'], arm.q)
        np.testing.assert_allclose(pose['arms'][side]['gello_q'], reader.raw[:7])
        np.testing.assert_allclose(saved[side]['RobotConfig']['reset_q'], arm.q)
        teleop = saved[side]['TeleoperatorConfig']
        np.testing.assert_allclose(teleop['start_joints'], arm.q)
        np.testing.assert_allclose(teleop['leader_reference_q'], reader.raw[:7])
        np.testing.assert_allclose((np.asarray(teleop['leader_reference_q'])-teleop['joint_offsets'])*teleop['joint_signs'], arm.q)
        assert teleop['joint_signs'] == before[side]['TeleoperatorConfig']['joint_signs']
        for key in ('gripper_open_deg', 'gripper_close_deg'):
            assert teleop[key] == before[side]['TeleoperatorConfig'][key]
        assert saved[side]['CalibrationStatus']['checks'] == before[side]['CalibrationStatus']['checks']
        assert saved[side]['CalibrationStatus']['direction_verified'] == [True]*7
        arm.command_joint_positions.assert_not_called()
        reader.prepare_alignment.assert_not_called()
        reader.set_torque.assert_not_called()
    assert pipeline.state == 'holding'
    assert pose['sample_count'] == 5
    pipeline.close()
    assert not live_pose_path(pipeline.config_path).exists()
    replacement = DualGelloPipeline(pipeline.config_path, arm_factory=SimulatedArm, reader_factory=StaticLeader)
    try:
        replacement.connect(); replacement.reset(); replacement.wait_for_alignment(); replacement.takeover()
        for side, arm, mapper, reader in zip(('left', 'right'), replacement.arms, replacement.mappers, replacement.readers):
            np.testing.assert_allclose(arm.q, pose['arms'][side]['xarm_q'])
            np.testing.assert_allclose(mapper.target(reader.read())[0], arm.q)
    finally:
        replacement.close()


def test_cli_dry_run_leaves_calibration_byte_for_byte_unchanged(live_pipeline, capsys):
    pipeline = live_pipeline
    before = pipeline.gello_config_path.read_bytes()
    assert main(['-c', str(pipeline.config_path), '--dry-run', '--samples', '2']) == 0
    assert pipeline.gello_config_path.read_bytes() == before
    output = capsys.readouterr().out
    assert 'left xarm q_rad' in output and 'right gello q_rad' in output


def test_gripper_inspector_uses_cached_motor_eight_without_changing_calibration(live_pipeline, capsys):
    from gello_teleop.inspect_gello_grippers import main as inspect_main, read_gripper_status
    pipeline = live_pipeline
    before = pipeline.gello_config_path.read_bytes()
    # These endpoint directions differ; either sign must map physical opening to 0..1.
    for reader, mapper, degrees in zip(pipeline.readers, pipeline.mappers, ((180., 138.), (180., 222.))):
        mapper.gripper_open, mapper.gripper_close = np.deg2rad(degrees)
        reader.raw[7] = (mapper.gripper_open+mapper.gripper_close)/2
    deadline = time.monotonic()+1.
    while True:
        status = read_gripper_status(pipeline.config_path)
        if all(value['closure_fraction'] == pytest.approx(.5) for value in status.values()):
            break
        if time.monotonic() > deadline:
            raise AssertionError('gripper status did not refresh')
        time.sleep(.01)
    assert [status[side]['gello_angle_deg'] for side in ('left', 'right')] == pytest.approx([159., 201.])
    for value in status.values():
        assert value['desired_width_m'] == pytest.approx(.0425)
        assert value['gello_motor_id'] == 8
        assert value['actual_width_m'] is None
        assert not value['following']  # F holds the xArm gripper while GELLO angles still update
    assert inspect_main(['-c', str(pipeline.config_path), '--left']) == 0
    output = capsys.readouterr().out
    assert 'left ID8' in output and '闭合=50.0%' in output
    assert 'right ID8' not in output
    assert pipeline.gello_config_path.read_bytes() == before
    # The payload remains JSON-safe when no gripper feedback is available.
    json.dumps(pipeline.reference_pose_snapshot(), allow_nan=False)


def test_gripper_inspector_rejects_expired_cache_and_older_collector(live_pipeline):
    from gello_teleop.inspect_gello_grippers import read_gripper_status
    pipeline = live_pipeline
    source = live_pose_path(pipeline.config_path)
    pose = json.loads(source.read_text())
    pipeline.reference_publisher.close()
    for arm in pose['arms'].values():
        arm.pop('gripper')
    pose['published_monotonic_s'] = time.monotonic()
    source.write_text(json.dumps(pose))
    with pytest.raises(RuntimeError, match='重启新版数采脚本'):
        read_gripper_status(pipeline.config_path)
    pose['published_monotonic_s'] -= 10
    source.write_text(json.dumps(pose))
    with pytest.raises(RuntimeError, match='已过期'):
        read_gripper_status(pipeline.config_path)


def test_gripper_status_copies_successful_commands_and_fresh_feedback_without_hardware_io(live_pipeline):
    pipeline = live_pipeline
    now = time.time_ns()//1000
    pipeline.gripper_workers = [Mock(snapshot=Mock(return_value=(.07, True, .06, True, now-100, now-200))),
                                Mock(snapshot=Mock(return_value=(.07, True, .06, True, now-100, now-1_000_000)))]
    readers = []
    for arm in pipeline.arms:
        arm.read_gripper_state = Mock(side_effect=AssertionError('no extra hardware I/O'))
        readers.append(arm.read_gripper_state)
    try:
        pose = pipeline.reference_pose_snapshot()
        assert pose['valid']
        assert pose['arms']['left']['gripper']['actual_width_m'] == .07
        assert pose['arms']['left']['gripper']['command_width_m'] == .06
        assert pose['arms']['right']['gripper']['actual_width_m'] is None
        assert pose['arms']['right']['gripper']['command_width_m'] == .06
        for reader in readers:
            reader.assert_not_called()
    finally:
        pipeline.gripper_workers = []


def test_cli_reads_running_pipeline_from_a_second_process(live_pipeline):
    pipeline = live_pipeline
    result = subprocess.run([sys.executable, str(ROOT/'gello_teleop/capture_reference_pose.py'),
                             '-c', str(pipeline.config_path), '--samples', '2'],
                            capture_output=True, text=True, timeout=5.)
    assert result.returncode == 0, result.stderr
    assert '已覆盖保存所选臂参考' in result.stdout
    saved = yaml.safe_load(pipeline.gello_config_path.read_text())
    for side, arm in zip(('left', 'right'), pipeline.arms):
        np.testing.assert_allclose(saved[side]['RobotConfig']['reset_q'], arm.q)


@pytest.mark.parametrize('defect', ['right_sign', 'right_nan', 'right_identity'])
def test_invalid_right_reference_never_commits_a_partial_left_update(live_pipeline, defect):
    pipeline = live_pipeline
    pose = capture_live_pose(pipeline.config_path, samples=2)
    if defect == 'right_sign':
        saved = yaml.safe_load(pipeline.gello_config_path.read_text())
        saved['right']['TeleoperatorConfig']['joint_signs'][0] *= -1
        pipeline.gello_config_path.write_text(yaml.safe_dump(saved))
    elif defect == 'right_nan':
        pose['arms']['right']['xarm_q'][0] = np.nan
    else:
        pose['arms']['right']['identity']['port'] = '/dev/a-different-gello'
    before = pipeline.gello_config_path.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        overwrite_reference_pose(pose)
    assert pipeline.gello_config_path.read_bytes() == before


def test_old_cached_q_cannot_be_counted_as_multiple_fresh_samples(live_pipeline):
    pipeline = live_pipeline
    source = live_pose_path(pipeline.config_path)
    pose = json.loads(source.read_text())
    pipeline.reference_publisher.close()
    pose['published_monotonic_s'] = time.monotonic()
    now = time.time_ns()//1000
    for arm in pose['arms'].values():
        arm['gello_timestamp_us'] = arm['xarm_timestamp_us'] = now
    source.write_text(json.dumps(pose))
    before = pipeline.gello_config_path.read_bytes()
    with pytest.raises(TimeoutError, match='足够的所选臂新采样'):
        capture_live_pose(pipeline.config_path, samples=2, timeout_s=.06)
    assert pipeline.gello_config_path.read_bytes() == before


@pytest.mark.parametrize('defect', ['stale', 'busy', 'invalid'])
def test_failed_or_stale_live_feedback_never_overwrites(live_pipeline, defect):
    pipeline = live_pipeline
    source = live_pose_path(pipeline.config_path)
    pose = json.loads(source.read_text())
    pipeline.reference_publisher.close()
    if defect == 'stale':
        pose['published_monotonic_s'] -= 10
    elif defect == 'busy':
        pose.update(valid=False, reason='当前正在复位或对齐')
    else:
        pose['arms']['right']['gello_q'][2] = float('nan')
    source.write_text(json.dumps(pose))
    before = pipeline.gello_config_path.read_bytes()
    assert main(['-c', str(pipeline.config_path), '--samples', '2']) == 1
    assert pipeline.gello_config_path.read_bytes() == before


def test_missing_running_collector_reports_how_to_start(tmp_path, capsys):
    config = tmp_path/'config.yaml'
    config.write_text('gello_config: calibration.yaml\n')
    assert main(['-c', str(config)]) == 1
    assert '先用同一配置启动新版 dual_gello_collect' in capsys.readouterr().err


def test_capture_unwraps_gello_encoder_boundary_and_rejects_real_motion(live_pipeline, monkeypatch):
    pipeline = live_pipeline
    base = json.loads(live_pose_path(pipeline.config_path).read_text())
    pipeline.reference_publisher.close()
    values = iter([2*np.pi-.001, .001, .002])
    counter = [0]
    def feedback(*args, **kwargs):
        item = deepcopy(base)
        angle = next(values)
        counter[0] += 1
        item['published_monotonic_s'] = time.monotonic()
        for arm in item['arms'].values():
            arm['gello_sequence'] += counter[0]
            arm['xarm_timestamp_us'] = arm['gello_timestamp_us'] = time.time_ns()//1000
            arm['gello_q'][0] = angle
        return json.dumps(item)
    original_read = Path.read_text
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_text', lambda path, *a, **k: feedback() if path.suffix == '.json' else original_read(path, *a, **k))
        pose = capture_live_pose(pipeline.config_path, samples=3)
    assert pose['arms']['left']['gello_motion_deg'][0] < 1.
    np.testing.assert_allclose(pose['arms']['left']['gello_q'][0], 2*np.pi+.002/3)
    values = iter([0., .1])
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_text', lambda path, *a, **k: feedback() if path.suffix == '.json' else original_read(path, *a, **k))
        with pytest.raises(RuntimeError, match='采样期间移动超过'):
            capture_live_pose(pipeline.config_path, samples=2)


def test_snapshot_is_unavailable_during_o_reset_or_t_alignment(live_pipeline):
    pipeline = live_pipeline
    pipeline.reference_capture_busy = True
    snapshot = pipeline.reference_pose_snapshot()
    assert snapshot['valid'] is False and '复位或对齐' in snapshot['reason']
    pipeline.reference_capture_busy = False


@pytest.mark.parametrize('live_pipeline', ['following'], indirect=True)
def test_keypress_saves_the_pressed_pose_while_motion_and_recording_continue(live_pipeline, monkeypatch, capsys):
    import threading
    import gello_teleop.dual_gello_collect as collect
    pipeline = live_pipeline
    before = yaml.safe_load(pipeline.gello_config_path.read_text())
    mapper_offsets = [mapper.offsets.copy() for mapper in pipeline.mappers]
    entered, release = threading.Event(), threading.Event()
    original_write = collect.overwrite_reference_pose
    def delayed_write(pose, **kwargs):
        entered.set()
        assert release.wait(timeout=2.)
        return original_write(pose, **kwargs)
    monkeypatch.setattr(collect, 'overwrite_reference_pose', delayed_write)
    pipeline.freeze_following = Mock(side_effect=AssertionError('saving must not freeze'))
    pipeline.start_episode()
    buffer = pipeline.buffer
    tick_before = pipeline.control_tick_count
    try:
        assert pipeline.save_reference_pose()
        pressed = deepcopy(pipeline.reference_save_pose)
        assert entered.wait(timeout=1.)
        assert not pipeline.save_reference_pose()  # one active file write, without blocking input
        deadline = time.monotonic()+.16
        while time.monotonic() < deadline:
            for reader in pipeline.readers:
                reader.raw[:7] += .002
            pipeline.poll()
            time.sleep(.005)
        assert pipeline.reference_save_future is not None
        assert pipeline.control_tick_count > tick_before+5
        assert pipeline.recording and pipeline.state == 'recording' and pipeline.buffer is buffer
        assert not pipeline.follow_stop.is_set() and pipeline.queue_overflow == 0
        assert buffer.sample_count >= 8
    finally:
        release.set()
    deadline = time.monotonic()+2.
    while pipeline.reference_save_future is not None:
        pipeline.poll()
        if time.monotonic() > deadline:
            raise AssertionError('reference save did not complete')
        time.sleep(.005)
    saved = yaml.safe_load(pipeline.gello_config_path.read_text())
    for side, reader, mapper, offsets in zip(('left', 'right'), pipeline.readers, pipeline.mappers, mapper_offsets):
        np.testing.assert_allclose(saved[side]['RobotConfig']['reset_q'], pressed['arms'][side]['xarm_q'])
        section = saved[side]['TeleoperatorConfig']
        np.testing.assert_allclose(section['leader_reference_q'], pressed['arms'][side]['gello_q'])
        assert not np.allclose(section['leader_reference_q'], reader.raw[:7])
        assert section['joint_signs'] == before[side]['TeleoperatorConfig']['joint_signs']
        for key in ('gripper_open_deg', 'gripper_close_deg'):
            assert section[key] == before[side]['TeleoperatorConfig'][key]
        assert saved[side]['CalibrationStatus']['reference_pose_capture']['sample_count'] == 1
        assert saved[side]['CalibrationStatus']['reference_pose_capture']['max_motion_deg'] is None
        np.testing.assert_allclose(mapper.offsets, offsets)
    pipeline.freeze_following.assert_not_called()
    assert pipeline.recording and pipeline.control_thread.is_alive()
    assert [event['event'] for event in buffer.episode_metadata['teleop_events']][-2:] == [
        'reference_pose_save_requested', 'reference_pose_saved']
    assert '当前遥操继续' in capsys.readouterr().out


@pytest.mark.parametrize('live_pipeline', ['following'], indirect=True)
def test_repeated_keypress_overwrites_both_references_and_next_start_uses_last_pose(live_pipeline):
    pipeline = live_pipeline
    captured = []
    for step in (0., .03):
        for reader in pipeline.readers:
            reader.raw[:7] += step
        deadline = time.monotonic()+1.
        while True:
            pose = pipeline.reference_pose_snapshot()
            if pose['valid'] and all(np.allclose(pose['arms'][side]['gello_q'], reader.raw[:7])
                                     for side, reader in zip(('left', 'right'), pipeline.readers)):
                break
            pipeline.poll()
            if time.monotonic() > deadline:
                raise AssertionError('new leader pose was not published')
            time.sleep(.005)
        assert pipeline.save_reference_pose()
        captured.append(deepcopy(pipeline.reference_save_pose))
        while pipeline.reference_save_future is not None:
            pipeline.poll()
            if time.monotonic() > deadline:
                raise AssertionError('reference save did not complete')
            time.sleep(.005)
        saved = yaml.safe_load(pipeline.gello_config_path.read_text())
        for side in ('left', 'right'):
            np.testing.assert_allclose(saved[side]['RobotConfig']['reset_q'], captured[-1]['arms'][side]['xarm_q'])
            np.testing.assert_allclose(saved[side]['TeleoperatorConfig']['leader_reference_q'], captured[-1]['arms'][side]['gello_q'])
    assert captured[0]['arms']['left']['gello_q'] != captured[1]['arms']['left']['gello_q']
    pipeline.close()
    replacement = DualGelloPipeline(pipeline.config_path, arm_factory=SimulatedArm, reader_factory=StaticLeader)
    try:
        replacement.connect(); replacement.reset(); replacement.wait_for_alignment(); replacement.takeover()
        for side, arm, reader, mapper in zip(('left', 'right'), replacement.arms, replacement.readers, replacement.mappers):
            np.testing.assert_allclose(arm.q, captured[-1]['arms'][side]['xarm_q'])
            np.testing.assert_allclose(reader.raw[:7], captured[-1]['arms'][side]['gello_q'])
            np.testing.assert_allclose(mapper.target(reader.read())[0], arm.q)
    finally:
        replacement.close()


@pytest.mark.parametrize('defect', ['busy', 'right_nan', 'right_stale'])
def test_failed_keypress_snapshot_never_schedules_a_partial_save(live_pipeline, monkeypatch, defect):
    pipeline = live_pipeline
    pose = pipeline.reference_pose_snapshot()
    if defect == 'busy':
        pose.update(valid=False, reason='复位或对齐中')
    elif defect == 'right_nan':
        pose['arms']['right']['gello_q'][0] = float('nan')
    else:
        pose['arms']['right']['xarm_timestamp_us'] -= 1_000_000
    before = pipeline.gello_config_path.read_bytes()
    monkeypatch.setattr(pipeline, 'reference_pose_snapshot', lambda: pose)
    with pytest.raises((RuntimeError, ValueError)):
        pipeline.save_reference_pose()
    assert pipeline.reference_save_future is None
    assert pipeline.gello_config_path.read_bytes() == before


def test_keypress_save_fails_immediately_on_locked_calibration_and_keeps_pipeline_running(live_pipeline, capsys):
    import fcntl
    pipeline = live_pipeline
    before = pipeline.gello_config_path.read_bytes()
    with pipeline.gello_config_path.with_suffix('.yaml.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert pipeline.save_reference_pose()
        deadline = time.monotonic()+1.
        while pipeline.reference_save_future is not None:
            pipeline.poll()
            if time.monotonic() > deadline:
                raise AssertionError('reference writer blocked behind an external lock')
            time.sleep(.005)
    assert pipeline.gello_config_path.read_bytes() == before
    assert pipeline.state == 'holding' and not pipeline.stop_event.is_set()
    assert '正在由其他进程写入' in capsys.readouterr().out


def test_online_reference_save_preserves_episode_and_configured_camera_streams(tmp_path):
    import h5py
    from scripts.smoke_dual_gello_pipeline import SimulatedLeader
    config = write_simulation_config(ROOT/'gello_teleop/config/xarm7_gello_dual_dataset.yaml', tmp_path)
    workflow = yaml.safe_load(config.read_text())
    workflow['active_arms'] = ['left', 'right']
    workflow.setdefault('cameras', [dict(name=f'{owner} wrist', backend='mock', width=64,
                                        height=48, fps=30, output_size=[224, 224])
                                    for owner in ('left', 'right')])
    workflow['gripper']['mode'] = 'follow'
    config.write_text(yaml.safe_dump(workflow))
    pipeline = DualGelloPipeline(config, arm_factory=SimulatedArm, reader_factory=SimulatedLeader)
    try:
        pipeline.connect(); pipeline.reset(); pipeline.wait_for_alignment(); pipeline.takeover()
        pipeline.start_episode()
        buffer = pipeline.buffer
        assert pipeline.save_reference_pose()
        pressed = deepcopy(pipeline.reference_save_pose)
        deadline = time.monotonic()+.5
        while time.monotonic() < deadline or pipeline.reference_save_future is not None:
            pipeline.poll()
            assert pipeline.recording and pipeline.buffer is buffer and not pipeline.follow_stop.is_set()
            time.sleep(.005)
        output = pipeline.stop_episode(save=True)
        with h5py.File(output) as episode:
            timestamps = episode['teleop/timestamp_us'][:]
            assert len(timestamps) >= 30 and np.all(np.diff(timestamps) > 0)
            assert set(episode['cameras']) == {camera['name'] for camera in workflow['cameras']
                                              if camera.get('enabled', True)}
            for camera in episode['cameras'].values():
                assert len(camera['frames']) >= 5
            metadata = json.loads(episode['metadata/episode_json'][()])
            events = metadata['teleop_events']
            assert [event['event'] for event in events] == ['reference_pose_save_requested', 'reference_pose_saved']
            for side in ('left', 'right'):
                np.testing.assert_allclose(events[-1]['reference_q'][side]['gello'], pressed['arms'][side]['gello_q'])
        assert pipeline.state == 'following' and pipeline.control_thread.is_alive()
    finally:
        pipeline.close()
