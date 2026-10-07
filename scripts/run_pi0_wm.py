#!/usr/bin/env python3
"""Compatibility command for PI0-WM using the common inference keyboard loop."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.pi0_wm.cli import main as pi0_main


def main(argv: list[str] | None = None) -> int:
    pi0_main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
