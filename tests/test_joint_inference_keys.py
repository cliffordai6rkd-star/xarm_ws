"""Keyboard transitions use mock arms and never access robot hardware."""

from dataclasses import replace
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from inference.core.contracts import ActionChunk
from nero_collection.time_utils import now_us
from test_joint_policy_inference import runtime_fixture


def reset_fixture(sides=("right",), *, enabled=True):
    runtime, arms, policy, cameras = runtime_fixture(sides, enabled=enabled)
    runtime.config.hardware["alignment"] = {
        "reset_speed_rad_s": .2,
        "reset_acc_rad_s2": .5,
        "reset_timeout_s": .3,
        "reset_tolerance_rad": .005,
        "reset_samples": 1,
    }
    runtime.config.execution["warmup_steps"] = 0
    for arm in arms.values():
        attach_reset_tracking(arm)
    return runtime, arms, policy, cameras


def attach_reset_tracking(arm):
    arm.reset_calls = []
    arm.motion_mode = "stopped"
    enable = arm.enable
    stop = arm.stop_motion
    command = arm.command_joint_positions

    def tracked_enable():
        enable()
        arm.motion_mode = "servo"

    def tracked_stop():
        stop()
        arm.motion_mode = "stopped"

    def planned_reset(q, *, speed, acceleration):
        arm._ensure_connected()
        arm.reset_calls.append((q.copy(), speed, acceleration))
        arm.motion_mode = "planned"
        # Integrate toward the target in read_state instead of instantly
        # teleporting, so tests can exit while a reset is still underway.
        arm._target_q = q.copy()

    def servo_command(q):
        assert arm.motion_mode == "servo", "Policy target sent outside servo mode"
        command(q)

    arm.enable = tracked_enable
    arm.stop_motion = tracked_stop
    arm.move_to_reset = planned_reset
    arm.command_joint_positions = servo_command
    arm.control_status = lambda: {
        "state": 0, "mode": 0 if arm.motion_mode == "planned" else 1, "error_code": 0,
    }


def test_start_paused_waits_for_s_without_predicting_or_enabling():
    runtime, arms, policy, _ = reset_fixture()
    runtime.config.execution["startup_timeout_s"] = .02
    runtime.run(.08, start_paused=True)
    assert not policy.windows
    assert runtime.predictions == runtime.commands == 0
    assert arms["right"].enables == arms["right"].stops == 0
    assert not arms["right"].commands and not arms["right"].reset_calls


def test_s_runs_continuously_by_default():
    runtime, arms, _, _ = reset_fixture()
    keys = iter(["s"])
    runtime.run(.26, read_key=lambda _: next(keys, None), start_paused=True)
    assert runtime.predictions >= 2
    assert runtime.commands > 3
    assert arms["right"].enables == 1


def test_i_resets_then_waits_and_s_restarts_warmup_and_servo(caplog):
    runtime, arms, policy, _ = reset_fixture()
    runtime.config.execution["warmup_steps"] = 1
    stage = {"phase": "start"}

    def keys(_):
        if stage["phase"] == "start":
            stage["phase"] = "running"
            return "s"
        if stage["phase"] == "running" and runtime.commands >= 3:
            stage.update(phase="resetting", commands=runtime.commands)
            return "i"
        if (stage["phase"] == "resetting" and arms["right"].reset_calls
                and runtime.reset_progress is None):
            stage.update(phase="paused", paused_at=time.monotonic(),
                         windows=len(policy.windows), predictions=runtime.predictions)
        if stage["phase"] == "paused" and time.monotonic() - stage["paused_at"] > .04:
            assert len(policy.windows) == stage["windows"]
            assert runtime.commands == stage["commands"]
            assert runtime.plan is None
            stage.update(phase="restarted", restart_us=now_us(),
                         restart_window=len(policy.windows))
            return "s"
        if stage["phase"] == "restarted" and runtime.commands >= stage["commands"] + 3:
            stage["phase"] = "done"
            return "q"
        return None

    with caplog.at_level("INFO", logger="inference.joint_runtime"):
        runtime.run(1., read_key=keys, start_paused=True)
    assert stage["phase"] == "done"
    assert caplog.text.count("Policy warmup completed") >= 2
    assert len(arms["right"].reset_calls) == 1
    q, speed, acceleration = arms["right"].reset_calls[0]
    np.testing.assert_array_equal(q, np.zeros(7))
    assert speed == .2 and acceleration == .5
    assert arms["right"].enables >= 2
    resumed_windows = policy.windows[stage["restart_window"]:]
    assert resumed_windows
    assert all(item.timestamp_us >= stage["restart_us"]
               for window in resumed_windows for item in window)


def test_i_in_read_only_mode_never_enables_or_moves_arm():
    runtime, arms, _, _ = reset_fixture(enabled=False)
    stage = {"phase": "start", "ticks": 0}

    def keys(_):
        if stage["phase"] == "start":
            stage["phase"] = "running"
            return "s"
        if stage["phase"] == "running" and runtime.predictions:
            stage.update(phase="paused", predictions=runtime.predictions)
            return "i"
        if stage["phase"] == "paused":
            stage["ticks"] += 1
            if stage["ticks"] >= 5:
                return "q"
        return None

    runtime.run(.4, read_key=keys, start_paused=True)
    assert stage["phase"] == "paused"
    assert runtime.predictions == stage["predictions"]
    assert runtime.plan is None
    assert arms["right"].enables == arms["right"].stops == 0
    assert not arms["right"].commands and not arms["right"].reset_calls


@pytest.mark.parametrize("open_loop", [False, True])
def test_i_cancels_inflight_result_before_restarted_policy_commands(open_loop):
    runtime, arms, policy, _ = reset_fixture()
    runtime.config.execution["open_loop"] = open_loop
    entered, release = threading.Event(), threading.Event()
    predict = policy.predict_observations
    calls = {"count": 0}

    def blocked_first_prediction(window):
        calls["count"] += 1
        if calls["count"] == 1:
            entered.set()
            assert release.wait(timeout=1.)
            predict(window)
            return ActionChunk(np.full((3, 7), .4), "joint", None,
                               window[-1].timestamp_us, policy.step_s)
        return predict(window)

    policy.predict_observations = blocked_first_prediction
    stage = {"phase": "start"}

    def keys(_):
        if stage["phase"] == "start":
            stage["phase"] = "inflight"
            return "s"
        if stage["phase"] == "inflight" and entered.is_set():
            stage["phase"] = "resetting"
            return "i"
        if (stage["phase"] == "resetting" and arms["right"].reset_calls
                and runtime.reset_progress is None):
            release.set()
            stage["phase"] = "restarted"
            return "s"
        if stage["phase"] == "restarted" and runtime.commands >= 3:
            stage["phase"] = "done"
            return "q"
        return None

    try:
        runtime.run(.7, read_key=keys, start_paused=True)
    finally:
        release.set()
    assert stage["phase"] == "done"
    assert runtime.predictions >= 1
    assert arms["right"].commands
    # A cancelled .4 rad chunk must never enter the bounded servo stream.
    assert max(np.max(np.abs(q)) for q in arms["right"].commands) <= .03 + 1e-9


@pytest.mark.parametrize("start_paused", [False, True])
def test_startup_reset_precedes_every_policy_request(start_paused, caplog):
    runtime, arms, policy, _ = reset_fixture()
    runtime.config.execution.update(reset_on_start=True, open_loop=True, log_timing=True)
    predict = policy.predict_observations
    def require_reset(window):
        assert len(arms["right"].reset_calls) == 1
        assert runtime.reset_progress is None
        return predict(window)
    policy.predict_observations = require_reset
    with caplog.at_level("INFO", logger="inference.joint_runtime"):
        runtime.run(.35, start_paused=start_paused)
    assert len(arms["right"].reset_calls) == 1
    assert "reset_ms=" in caplog.text
    if start_paused:
        assert not policy.windows and runtime.predictions == 0
    else:
        assert policy.windows and runtime.predictions >= 1


def test_startup_reset_in_read_only_does_not_enable_or_move():
    runtime, arms, policy, _ = reset_fixture(enabled=False)
    runtime.config.execution["reset_on_start"] = True
    runtime.run(.1, start_paused=True)
    assert not policy.windows
    assert arms["right"].enables == 0 and not arms["right"].reset_calls


def test_q_can_interrupt_startup_reset_without_model_requests():
    runtime, arms, policy, _ = reset_fixture()
    runtime.config.execution["reset_on_start"] = True
    arms["right"]._q.fill(.4)
    arms["right"]._target_q.fill(.4)
    runtime.run(.3, read_key=lambda _: "q", start_paused=True)
    assert not policy.windows and runtime.predictions == 0
    assert len(arms["right"].reset_calls) == 1 and arms["right"].stops == 1


def test_q_during_reset_stops_both_arms_without_policy_inference():
    runtime, arms, policy, _ = reset_fixture(("left", "right"))
    for arm in arms.values():
        arm._q.fill(.4)
        arm._target_q.fill(.4)
    stage = {"started": False}

    def keys(_):
        if not stage["started"]:
            stage["started"] = True
            return "i"
        if all(arm.reset_calls for arm in arms.values()):
            assert runtime.reset_progress is not None
            return "q"
        return None

    runtime.run(.2, read_key=keys, start_paused=True)
    assert not policy.windows
    assert all(len(arm.reset_calls) == 1 and arm.stops == 1 and not arm._connected
               for arm in arms.values())


def test_second_arm_reset_failure_stops_first_arm():
    runtime, arms, _, _ = reset_fixture(("left", "right"))
    reset = arms["right"].move_to_reset

    def failed_reset(q, *, speed, acceleration):
        reset(q, speed=speed, acceleration=acceleration)
        raise RuntimeError("second arm reset SDK failure")

    arms["right"].move_to_reset = failed_reset
    keys = iter(["i"])
    with pytest.raises(RuntimeError, match="second arm reset SDK failure"):
        runtime.run(.2, read_key=lambda _: next(keys, None), start_paused=True)
    assert len(arms["left"].reset_calls) == 1
    assert all(arm.stops == 1 and not arm._connected for arm in arms.values())


def test_invalid_reset_target_rejects_all_arms_before_motion():
    runtime, arms, _, _ = reset_fixture(("left", "right"))
    runtime.config.endpoints["right"] = replace(
        runtime.config.endpoints["right"], rest_q=(float("nan"),) * 7
    )
    keys = iter(["i"])
    with pytest.raises(ValueError, match="reset|rest_q|finite"):
        runtime.run(.2, read_key=lambda _: next(keys, None), start_paused=True)
    assert all(arm.enables == 0 and not arm.reset_calls and not arm.commands
               for arm in arms.values())


@pytest.mark.parametrize(
    "options,start_paused,single_step",
    [([], True, False), (["--auto-start"], False, False), (["--single-step"], True, True)],
)
def test_cli_passes_keyboard_start_options_without_hardware_or_terminal(
    monkeypatch, options, start_paused, single_step
):
    from inference import joint_cli, joint_runtime
    from nero_collection import keyboard
    import torch

    config = SimpleNamespace(policy={"type": "dp"})
    policy = SimpleNamespace(close=lambda: None)
    mapping = {"camera": "left_wrist"}
    recorded = {}

    class Runtime:
        predictions = commands = 0

        def __init__(self, cfg, loaded_policy, camera_map, **kwargs):
            assert cfg is config and loaded_policy is policy and camera_map is mapping
            recorded["init"] = kwargs

        def run(self, duration, **kwargs):
            recorded["run"] = kwargs

    class Keys:
        def __enter__(self):
            recorded["terminal_entered"] = True
            return self

        def __exit__(self, *args):
            recorded["terminal_exited"] = True

        def read_key(self, timeout):
            return None

    monkeypatch.setattr(joint_cli.JointDeployment, "load", lambda *args, **kwargs: config)
    monkeypatch.setattr(joint_cli, "load_joint_policy", lambda cfg: (policy, mapping))
    monkeypatch.setattr(joint_cli, "policy_report", lambda *args: {"test": True})
    monkeypatch.setattr(joint_runtime, "JointInferenceRuntime", Runtime)
    monkeypatch.setattr(keyboard, "TerminalKeys", Keys)
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    joint_cli.main(["--config", "unused.yaml", "--run", "--backend", "mock", *options])
    assert recorded["init"]["backend"] == "mock"
    assert recorded["init"]["command_enabled"] is False
    assert recorded["run"]["start_paused"] is start_paused
    assert recorded["run"]["single_step"] is single_step
    assert callable(recorded["run"]["read_key"])
    assert recorded["terminal_entered"] and recorded["terminal_exited"]


def test_reset_timeout_stops_every_moving_arm():
    runtime, arms, _, _ = reset_fixture(("left", "right"))
    runtime.config.hardware["alignment"]["reset_timeout_s"] = .03
    for arm in arms.values():
        arm._q.fill(.4)
        arm._target_q.fill(.4)
    keys = iter(["i"])
    with pytest.raises(TimeoutError, match="reset did not settle"):
        runtime.run(.2, read_key=lambda _: next(keys, None), start_paused=True)
    assert all(len(arm.reset_calls) == 1 and arm.stops == 1 and not arm._connected
               for arm in arms.values())


def test_reset_controller_fault_stops_every_moving_arm():
    runtime, arms, _, _ = reset_fixture(("left", "right"))
    arms["right"].control_status = lambda: {"mode": 0, "state": 4, "error_code": 7}
    keys = iter(["i"])
    with pytest.raises(RuntimeError, match="reset stopped on right"):
        runtime.run(.2, read_key=lambda _: next(keys, None), start_paused=True)
    assert all(len(arm.reset_calls) == 1 and arm.stops == 1 and not arm._connected
               for arm in arms.values())


def test_reset_requires_unique_feedback_and_controller_no_longer_moving():
    runtime, arms, _, _ = reset_fixture()
    runtime.config.hardware["alignment"]["reset_samples"] = 2
    arm = arms["right"]
    status = {"mode": 0, "state": 0, "error_code": 0}
    arm.control_status = lambda: dict(status)
    arm.connect()
    try:
        runtime._begin_reset()
        stamp = runtime.execution_observation_after_us + 1_000

        def feedback(timestamp):
            return {"right": [SimpleNamespace(q=np.zeros(7),
                q_acquired_timestamp_us=timestamp, acquired_timestamp_us=timestamp,
                timestamp_us=timestamp)]}

        assert not runtime._poll_reset(feedback(stamp), stamp)
        for _ in range(3):
            assert not runtime._poll_reset(feedback(stamp), stamp)
        assert runtime.reset_progress is not None and arm.stops == 0

        # Feedback is already at the target while the controller still reports
        # an executing planned move. It must not count as reset completion.
        status["state"] = 1
        stamp += 1_000
        assert not runtime._poll_reset(feedback(stamp), stamp)
        assert runtime.reset_progress is not None and arm.stops == 0

        status["state"] = 0
        stamp += 1_000
        assert not runtime._poll_reset(feedback(stamp), stamp)
        stamp += 1_000
        assert runtime._poll_reset(feedback(stamp), stamp)
        assert runtime.reset_progress is None and arm.stops == 1
    finally:
        runtime._stop_motion()
        arm.disconnect()
