"""Public inference entry points share keyboard commands and terminal input."""
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

from inference import cli, joint_cli
from inference.interactive import (
    InferenceCommand, add_interaction_arguments, decode_inference_key, run_interactive,
)


@pytest.mark.parametrize("key,expected", [
    ("s", InferenceCommand.START), ("S", InferenceCommand.START),
    ("i", InferenceCommand.STOP_RESET), ("I", InferenceCommand.STOP_RESET),
    ("p", InferenceCommand.PAUSE), ("c", InferenceCommand.START),
    ("q", InferenceCommand.EXIT), ("\x03", InferenceCommand.EXIT),
    (None, None), ("r", None), ("\n", None),
])
def test_common_key_commands(key, expected):
    assert decode_inference_key(key) is expected


def test_single_step_is_explicit_and_c_always_starts_continuously():
    assert decode_inference_key("s", single_step=True) is InferenceCommand.SINGLE_STEP
    assert decode_inference_key("c", single_step=True) is InferenceCommand.START


@pytest.mark.parametrize("policy_type", ["dp", "act"])
def test_public_entry_routes_policies_through_the_same_interactive_loop(
    tmp_path, monkeypatch, policy_type,
):
    config = tmp_path / "policy.yaml"
    config.write_text(f"policy:\n  type: {policy_type}\n")
    recorded = {}

    def dispatch(argv, *, run_loop):
        recorded.update(argv=argv, run_loop=run_loop)
        return "shared entry"

    monkeypatch.setattr(joint_cli, "main", dispatch)
    argv = ["--config", str(config), "--run"]
    assert cli.main(argv) == "shared entry"
    assert recorded == {"argv": argv, "run_loop": run_interactive}


@pytest.mark.parametrize("flags,wait,single_step", [
    ([], True, False), (["--auto-start"], False, False),
    (["--single-step"], True, True),
])
def test_common_runner_owns_terminal_and_passes_wait_for_start(
    monkeypatch, flags, wait, single_step,
):
    from nero_collection import keyboard

    events = []

    class Keys:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")

        def read_key(self, timeout):
            return "s"

    class Runtime:
        def run(self, duration, *, read_key, single_step, start_paused):
            assert duration == 2.
            assert read_key(0.) == "s"
            assert single_step == args.single_step and start_paused == wait
            events.append("run")
            return 7

    monkeypatch.setattr(keyboard, "TerminalKeys", Keys)
    parser = ArgumentParser()
    add_interaction_arguments(parser)
    args = parser.parse_args(flags)
    assert args.single_step == single_step
    args.duration = 2.
    assert run_interactive(Runtime(), args) == 7
    assert events == ["enter", "run", "exit"]


def test_nero_public_entry_uses_the_same_interactive_loop(tmp_path, monkeypatch):
    from inference import config as configuration, runtime as runtimes

    path = tmp_path / "nero.yaml"
    path.write_text("architecture:\n  enabled: true\n")
    config = object()
    recorded = {}

    class Runtime:
        def __init__(self, actual_config, *, backend, command_enabled):
            assert actual_config is config
            self.collection = SimpleNamespace(teleop=SimpleNamespace(backend="mock"))
            recorded.update(runtime=self, backend=backend, enabled=command_enabled)

    def run_loop(runtime, args):
        assert runtime is recorded["runtime"]
        assert not args.auto_start and not args.single_step
        recorded["called"] = True
        return 0

    monkeypatch.setattr(configuration, "load_inference_config", lambda _: config)
    monkeypatch.setattr(runtimes, "NeroInferenceRuntime", Runtime)
    monkeypatch.setattr(cli, "run_interactive", run_loop)
    cli.main(["--config", str(path), "--run", "--backend", "mock"])
    assert recorded["called"] and not recorded["enabled"]
