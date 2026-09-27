#!/usr/bin/env python3
"""Run the independent π0-WM runtime.

This wrapper keeps the π0-WM entry point separate from the legacy DP runtime.
``--mock`` replaces both cameras and the arm and never sends hardware commands.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.pi0_wm.config import load_config
from inference.pi0_wm.runtime import Runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="xArm π0-WM position runtime")
    parser.add_argument("--config", default="inference/configs/pi0_wm_xarm.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="mock arm/cameras; command output is forbidden")
    parser.add_argument("--mock-wm", action="store_true", help="use the deterministic local WM")
    parser.add_argument("--headless", action="store_true", help="disable visualization windows")
    parser.add_argument("--enable-commands", action="store_true", help="explicitly allow position commands")
    parser.add_argument("--dry-run", action="store_true", help="read-only mode (the default)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.dry_run and args.enable_commands:
        parser.error("--dry-run cannot be combined with --enable-commands")
    config = load_config(args.config)
    if args.headless:
        config['mujoco']['headless'] = True
    runtime = Runtime(config, enable_commands=args.enable_commands, mock=args.mock, mock_wm=args.mock_wm)
    result = runtime.run(args.steps)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
