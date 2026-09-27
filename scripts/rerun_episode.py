#!/usr/bin/env python3
"""Stream an episode to the optional Rerun viewer.

The dependency is optional and is never imported by collection or control
loops.  Missing Rerun produces a direct installation message.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="view xArm episode channels in Rerun")
    parser.add_argument("episode", type=Path)
    parser.add_argument("--camera", default=None)
    args = parser.parse_args(argv)
    try:
        import h5py
        import rerun as rr
    except Exception as exc:
        parser.error(f"Rerun episode viewing requires h5py and rerun-sdk: {exc}")
    rr.init("xarm_episode", spawn=True)
    with h5py.File(args.episode.expanduser().resolve(), "r") as h5:
        teleop = h5.get("teleop")
        if teleop is None or "timestamp_us" not in teleop:
            parser.error("episode has no teleop/timestamp_us timeline")
        timestamps = teleop["timestamp_us"][:]
        for index, timestamp_us in enumerate(timestamps):
            rr.set_time_sequence("episode", int(index))
            for name in ("q_follower", "dq_follower", "tau_follower", "delta_q"):
                if name in teleop and index < len(teleop[name]):
                    values = teleop[name][index]
                    for joint, value in enumerate(values.reshape(-1)):
                        rr.log(f"teleop/{name}/j{joint + 1}", rr.Scalars(float(value)))
            if args.camera and f"cameras/{args.camera}" in h5:
                group = h5[f"cameras/{args.camera}"]
                if index < len(group["frames"]):
                    rr.log(f"camera/{args.camera}", rr.Image(group["frames"][index]))
    print("Rerun stream sent; use the spawned viewer to inspect the episode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
