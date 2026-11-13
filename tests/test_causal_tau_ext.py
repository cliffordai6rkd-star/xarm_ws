"""Replay actual dp_hirol_main labels, independently of its checkout or parquet."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from xarm_stack.causal_tau_ext import CausalTauExt, TauExtConfig, gate_tau_ext, load_processing_record


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "gello_teleop/models/xarm7_dynamics.urdf"
FIXTURES = Path(__file__).parent / "fixtures"
RECORDS = {
    "clean": ROOT / "inference/configs/tau_ext_dp_hirol_clean.json",
    "ma5_l1_12p5": ROOT / "inference/configs/tau_ext_dp_hirol_ma5_l1_12p5.json",
}


def replay(estimator, samples):
    return [estimator.update(q, dq, tau, timestamp)
            for q, dq, tau, timestamp in zip(samples["q"], samples["dq"], samples["tau"],
                                             samples["timestamp_s"])]


def assert_stored_labels(outputs, samples):
    for name in ("tau_ext", "tau_ext_unthresholded", "tau_free", "tau_ext_l1", "contact"):
        actual = np.asarray([output[name] for output in outputs])
        expected = samples[name]
        # Parquet stores all numeric labels as float32, including the scalar L1.
        np.testing.assert_array_equal(actual.astype(expected.dtype), expected, err_msg=name)
    assert all(output["tau_ext"].dtype == np.float32 for output in outputs)
    assert all(output["tau_ext_unthresholded"].dtype == np.float32 for output in outputs)


@pytest.mark.parametrize("task", ["board", "huanggua"])
@pytest.mark.parametrize("version", RECORDS)
def test_replay_matches_actual_saved_labels(task, version):
    pytest.importorskip("pinocchio")
    with np.load(FIXTURES / f"causal_tau_ext_{task}_{version}.npz") as samples:
        estimator = CausalTauExt.from_processing_record(RECORDS[version], URDF)
        outputs = replay(estimator, samples)
        assert_stored_labels(outputs, samples)
        assert samples["frame_index"][0] == 0
        assert samples["contact"].any() and not samples["contact"].all()
        assert outputs[0]["first_sample"]
        assert not any(output["first_sample"] for output in outputs[1:])


@pytest.mark.parametrize("version", RECORDS)
def test_reset_replays_episode_without_previous_filter_or_mean_history(version):
    pytest.importorskip("pinocchio")
    with np.load(FIXTURES / f"causal_tau_ext_board_{version}.npz") as samples:
        estimator = CausalTauExt.from_processing_record(RECORDS[version], URDF)
        replay(estimator, samples)
        estimator.reset()
        outputs = replay(estimator, samples)
        assert_stored_labels(outputs, samples)
        np.testing.assert_array_equal(outputs[0]["ddq"], np.zeros(7))
        np.testing.assert_allclose(outputs[0]["dq_filtered"], samples["dq"][0], atol=1e-14)
        np.testing.assert_allclose(outputs[0]["tau_filtered"], samples["tau"][0], atol=1e-14)


def test_threshold_equality_retains_entire_signed_vector():
    raw = np.array([2., -2., 2., -2., 2., -2., 2.], dtype=np.float32)
    gated, l1, contact = gate_tau_ext(raw, 14.)
    assert contact and l1 == 14.
    np.testing.assert_array_equal(gated, raw)
    gated, _, contact = gate_tau_ext(raw, np.nextafter(14., np.inf))
    assert not contact
    np.testing.assert_array_equal(gated, np.zeros(7))


def test_gate_uses_stored_float32_precision_before_l1_comparison():
    raw = np.zeros(7, dtype=np.float64)
    raw[0] = 1. + 2.**-25
    gated, l1, contact = gate_tau_ext(raw, 1. + 2.**-26)
    assert l1 == 1. and not contact
    np.testing.assert_array_equal(gated, np.zeros(7))


def test_signed_mean_precedes_norm_and_gating():
    pytest.importorskip("pinocchio")
    config = replace(load_processing_record(RECORDS["ma5_l1_12p5"], URDF),
                     cutoff_hz=0., compensation="none")
    estimator = CausalTauExt(URDF, config)
    zeros = np.zeros(7)
    positive = np.array([14., 0., 0., 0., 0., 0., 0.])
    first = estimator.update(zeros, zeros, positive, 0.)
    assert first["contact"]
    second = estimator.update(zeros, zeros, -positive, 1. / config.fps)
    assert not second["contact"]
    np.testing.assert_array_equal(second["tau_ext_unthresholded"], zeros)
    np.testing.assert_array_equal(second["tau_ext"], zeros)


def test_saved_rate_is_required_and_reset_allows_new_timeline():
    pytest.importorskip("pinocchio")
    estimator = CausalTauExt.from_processing_record(RECORDS["clean"], URDF)
    zeros = np.zeros(7)
    estimator.update(zeros, zeros, zeros, 0.)
    with pytest.raises(ValueError, match="fixed sample rate"):
        estimator.update(zeros, zeros, zeros, .2)
    estimator.reset()
    assert estimator.update(zeros, zeros, zeros, .2)["first_sample"]


@pytest.mark.parametrize("version,window,threshold", [("clean", 1, 14.), ("ma5_l1_12p5", 5, 12.5)])
def test_processing_record_selects_actual_dataset_version(version, window, threshold):
    config = load_processing_record(RECORDS[version], URDF)
    assert isinstance(config, TauExtConfig)
    assert config.fps == 25. and config.cutoff_hz == 3.
    assert config.tau_ext_smoothing_window == window
    assert config.contact_threshold_l1 == threshold


def test_processing_record_rejects_wrong_urdf_and_offline_filter(tmp_path):
    wrong_urdf = tmp_path / "wrong.urdf"
    wrong_urdf.write_text("different dynamics")
    with pytest.raises(ValueError, match="URDF differs"):
        load_processing_record(RECORDS["clean"], wrong_urdf)
    record = json.loads(RECORDS["clean"].read_text())
    record["filter_mode"] = "offline"
    offline = tmp_path / "offline.json"
    offline.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="future-dependent"):
        load_processing_record(offline, URDF)


def test_processing_record_check_does_not_import_pinocchio():
    script = """
import builtins
original_import = builtins.__import__
def without_pin(name, *args, **kwargs):
    if name == 'pinocchio' or name.startswith('pinocchio.'):
        raise AssertionError('configuration check imported Pinocchio')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_pin
from xarm_stack.causal_tau_ext import load_processing_record
load_processing_record('inference/configs/tau_ext_dp_hirol_clean.json',
                       'gello_teleop/models/xarm7_dynamics.urdf')
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
