#!/usr/bin/env python3
"""Print an H5 episode manifest without opening a GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="inspect a Nero/xArm episode H5")
    parser.add_argument("episode", type=Path)
    args = parser.parse_args(argv)
    try:
        import h5py
    except Exception as exc:
        parser.error(f"h5py is required to inspect episodes: {exc}")
    with h5py.File(args.episode.expanduser().resolve(), "r") as h5:
        result = {
            "format": h5.attrs.get("format", ""),
            "teleop_datasets": sorted(h5.get("teleop", {}).keys()),
            "cameras": {},
            "metadata": sorted(h5.get("metadata", {}).keys()),
        }
        if "cameras" in h5:
            for name, group in h5["cameras"].items():
                result["cameras"][name] = {
                    "frames": tuple(group["frames"].shape) if "frames" in group else None,
                    "timestamps": tuple(group["timestamp_us"].shape) if "timestamp_us" in group else None,
                    "depth": tuple(group["depth"].shape) if "depth" in group else None,
                }
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
