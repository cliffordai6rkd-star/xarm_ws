"""Keyboard commands and terminal ownership shared by inference entry points."""
from __future__ import annotations

from enum import Enum


class InferenceCommand(str, Enum):
    START = "start"
    SINGLE_STEP = "single_step"
    STOP_RESET = "stop_reset"
    PAUSE = "pause"
    EXIT = "exit"


INTERACTION_HELP = "s: start inference, i: stop and reset, p: pause, q/Ctrl-C: exit."


def decode_inference_key(key: str | None, *, single_step: bool = False):
    """Give each model/backend the same keyboard intent."""
    if key is None:
        return None
    key = key.lower()
    if key == "s":
        return InferenceCommand.SINGLE_STEP if single_step else InferenceCommand.START
    return {"c": InferenceCommand.START, "i": InferenceCommand.STOP_RESET,
            "p": InferenceCommand.PAUSE, "q": InferenceCommand.EXIT,
            "\x03": InferenceCommand.EXIT}.get(key)


def add_interaction_arguments(parser):
    parser.add_argument("--single-step", action="store_true",
                        help="s executes one inference/action cycle then pauses")
    parser.add_argument("--auto-start", action="store_true",
                        help="start inference without waiting for s")


def run_interactive(runtime, args):
    """The public CLI owns keyboard input; runtimes execute shared commands."""
    from nero_collection.keyboard import TerminalKeys

    with TerminalKeys() as keys:
        return runtime.run(args.duration, read_key=keys.read_key,
                           single_step=args.single_step, start_paused=not args.auto_start)
