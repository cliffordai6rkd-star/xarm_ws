"""Startup feedback overrides use synthetic mappings and never touch hardware."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import gello_teleop.dual_gello_collect as collect
from gello_teleop.gello_hardware import GelloReader
from scripts.smoke_dual_gello_pipeline import write_simulation_config
from test_dual_gello_pipeline import CONFIG
from test_gello_producer_freshness import FakeClock, ClockStop
from xarm_stack.torque_visualization import DEFAULT_URDF


def forbidden(*_args, **_kwargs):
    pytest.fail('Feedback switch checks must not connect hardware or load a torque model')


def simulation_config(tmp_path, *, enabled, unavailable_model=False):
    path = write_simulation_config(CONFIG, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(active_arms=['left', 'right'], cameras=[], require_cameras=False)
    raw['gripper'] = {'enabled': False, 'command_enabled': False}
    raw['gello_damping'] = dict(enabled=True, damping_mode='current', sample_rate_hz=20,
        damping_gain=[8] * 7, damping_brake_gain=[0] * 7,
        damping_current_limit=[15] * 7, damping_velocity_filter_alpha=1.,
        damping_watchdog_ms=500, weak_hold_enabled=False)
    raw['force_feedback'] = dict(enabled=enabled, source='urdf',
        urdf_path=str(tmp_path / 'unavailable.urdf') if unavailable_model else str(DEFAULT_URDF))
    path.write_text(yaml.safe_dump(raw))
    return path


def make_pipeline(path, **kwargs):
    return collect.DualGelloPipeline(path, arm_factory=forbidden, reader_factory=forbidden,
        feedback_model_factory=forbidden, feedback_predictor_factory=forbidden, **kwargs)


def test_yaml_disabled_feedback_constructs_no_worker_and_preserves_damping(tmp_path, monkeypatch):
    path = simulation_config(tmp_path, enabled=False, unavailable_model=True)
    original = path.read_bytes()
    monkeypatch.setattr(collect, 'TauFreeFeedbackWorker', forbidden)
    pipeline = make_pipeline(path)
    try:
        assert pipeline.force_feedback is None
        assert pipeline.raw['force_feedback']['enabled'] is False
        for _side, _robot, leader in pipeline.configs:
            assert leader.damping_enabled and leader.damping_mode == 'current'
            assert list(leader.damping_gain) == [8] * 7
            assert leader.fps == 20
        assert pipeline.arms == [] and pipeline.readers == []
        assert path.read_bytes() == original
    finally:
        pipeline.close()


def test_disabled_feedback_writes_only_nonzero_local_damping(tmp_path, monkeypatch):
    pipeline = make_pipeline(simulation_config(tmp_path, enabled=False))
    clock = FakeClock()
    stop = ClockStop(clock)
    monkeypatch.setattr(collect, 'time', SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(collect, 'now_us', lambda: round(clock.now * 1e6))
    leader = replace(pipeline.configs[0][2], gripper_id=-1)
    writes = []
    reads = 0

    def read():
        nonlocal reads
        reads += 1
        clock.advance(.01)
        return np.full(7, reads * .02)

    def write(current):
        writes.append(current.copy())
        if len(writes) == 2:
            stop.set()

    reader = SimpleNamespace(read=read, compute_damping_current=GelloReader.compute_damping_current,
                             write_current_damping=write, disable_current_damping=forbidden)
    mapper = SimpleNamespace(config=leader, n=7, target=lambda raw: (raw.copy(), raw))
    producer = collect._LeaderProducer(reader, mapper, 1. / leader.fps, stop,
        damping_config=leader, failure_stop=ClockStop(clock),
        feedback_worker=pipeline.force_feedback, feedback_active=forbidden)
    try:
        producer._run()
        assert producer.errors == 0
        assert len(writes) == 2
        np.testing.assert_array_equal(writes[0], np.zeros(7))
        expected_damping = np.rint(np.full(7, -8 * .02 / .05) * np.asarray(leader.joint_signs))
        np.testing.assert_array_equal(writes[1], expected_damping)
        assert np.any(writes[1] != 0)
        feedback, current, valid = producer.current_snapshot()
        np.testing.assert_array_equal(feedback, np.zeros(7))
        np.testing.assert_array_equal(current, expected_damping)
        assert valid is False
    finally:
        pipeline.close()


@pytest.mark.parametrize('override', [None, True, False])
@pytest.mark.parametrize('yaml_enabled', [True, False])
def test_constructor_override_changes_only_effective_flag_without_loading_model(
        tmp_path, yaml_enabled, override):
    path = simulation_config(tmp_path, enabled=yaml_enabled)
    original = path.read_bytes()
    expected = yaml_enabled if override is None else override
    pipeline = make_pipeline(path, force_feedback_enabled=override)
    try:
        assert (pipeline.force_feedback is not None) is expected
        assert pipeline.raw['force_feedback']['enabled'] is expected
        assert yaml.safe_load(pipeline.collection_config.raw_yaml)['force_feedback']['enabled'] is expected
        if expected:
            assert pipeline.force_feedback.estimators == []
            assert pipeline.force_feedback.thread is None
        assert path.read_bytes() == original
    finally:
        pipeline.close()


@pytest.mark.parametrize('yaml_enabled,flag,expected', [
    (True, '--no-force-feedback', False),
    (False, '--force-feedback', True),
    (True, None, True),
    (False, None, False),
])
def test_cli_check_config_uses_effective_feedback_flag_and_never_connects(
        tmp_path, monkeypatch, capsys, yaml_enabled, flag, expected):
    path = simulation_config(tmp_path, enabled=yaml_enabled,
                             unavailable_model=flag == '--no-force-feedback')
    original = path.read_bytes()
    actual = collect.DualGelloPipeline
    constructed = []

    def factory(config, **kwargs):
        constructed.append(kwargs.copy())
        return actual(config, arm_factory=forbidden, reader_factory=forbidden,
                      feedback_model_factory=forbidden, feedback_predictor_factory=forbidden, **kwargs)

    monkeypatch.setattr(actual, 'connect', forbidden)
    monkeypatch.setattr(collect, 'DualGelloPipeline', factory)
    if flag == '--no-force-feedback':
        monkeypatch.setattr(collect, 'TauFreeFeedbackWorker', forbidden)
    args = ['-c', str(path), '--check-config']
    if flag is not None:
        args.append(flag)
    assert collect.main(args) == 0
    assert constructed == ([{}] if flag is None else [{'force_feedback_enabled': expected}])
    result = json.loads(capsys.readouterr().out)
    assert result['force_feedback_enabled'] is expected
    assert result['state'] == 'disconnected'
    assert path.read_bytes() == original


def test_cli_feedback_flags_are_mutually_exclusive(tmp_path, monkeypatch):
    path = simulation_config(tmp_path, enabled=True)
    monkeypatch.setattr(collect, 'DualGelloPipeline', forbidden)
    with pytest.raises(SystemExit) as error:
        collect.main(['-c', str(path), '--check-config', '--force-feedback', '--no-force-feedback'])
    assert error.value.code == 2
