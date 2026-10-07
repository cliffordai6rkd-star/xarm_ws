#!/usr/bin/env python3
"""Read-only xArm H5 review: RGB, TCP trajectory, joints and seven-axis tau_ext_cal."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nero_collection.h5_schema import dataset_candidates
from scripts.browse_h5_rerun import discover_episodes

TIMELINE = "episode_time"
COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
          (148, 103, 189), (140, 86, 75), (227, 119, 194)]


@dataclass
class EpisodeData:
    path: Path
    arm: str
    time_s: np.ndarray
    signals: dict[str, np.ndarray]
    ee_pose: np.ndarray | None
    cameras: dict[str, tuple[np.ndarray, np.ndarray]]
    sources: dict[str, str]
    notes: list[str]


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _arm_names(group):
    names = group.attrs.get("recorded_arm_names", group.attrs.get("arm_names", ()))
    if isinstance(names, (str, bytes)):
        names = (names,)
    return tuple(_text(name) for name in names)


def read_signal(group, key, arm, *, width=7, pose=False):
    """Resolve public device names or select an arm from legacy concatenated rows."""
    names = _arm_names(group)
    n = len(group["timestamp_us"])
    for candidate in dataset_candidates(arm, key):
        if candidate not in group:
            continue
        ds = group[candidate]
        if bool(ds.attrs.get("placeholder", False)):
            continue
        if not candidate.startswith(arm + "_") and names and arm not in names:
            continue
        values = np.asarray(ds[:], dtype=float)
        if not candidate.startswith(arm + "_") and len(names) > 1:
            if arm not in names:
                continue
            index = names.index(arm)
            if pose and values.shape == (n, len(names), 4, 4):
                values = values[:, index]
            elif not pose and values.shape == (n, width * len(names)):
                values = values[:, index*width:(index+1)*width]
            else:
                raise RuntimeError(f"Cannot select {arm} from legacy teleop/{candidate}: {values.shape}")
        if not pose and width == 1 and values.shape == (n,):
            values = values[:, None]
        expected = (n, 4, 4) if pose else (n, width)
        if values.shape != expected:
            raise RuntimeError(f"teleop/{candidate}: expected {expected}, got {values.shape}")
        validity_path = ds.attrs.get("validity_path")
        if validity_path:
            validity_path = _text(validity_path)
            if validity_path in group.file:
                valid = np.asarray(group.file[validity_path][:], dtype=bool).reshape(n, -1)
                if valid.shape[1] == len(names) > 1 and not candidate.startswith(arm + "_"):
                    valid = valid[:, [names.index(arm)]]
                values[~valid.all(axis=1)] = np.nan
        return values, "teleop/" + candidate
    return None, None


def estimate_external_torque(group, config, arm, urdf_path=None):
    """Replay the deployed estimator on recorded reports; never connect hardware."""
    from gello_teleop.tau_free_feedback import URDFTorqueEstimator, _merge_dynamics_settings, _validated_payload
    from xarm_stack.momentum import PinocchioMomentumModel

    block = config.get("force_feedback", {})
    settings = _merge_dynamics_settings(block, block.get("sides", {}).get(arm, {}))
    if urdf_path is None:
        urdf_path = Path(settings.get("urdf_path", "../models/xarm7_dynamics.urdf")).expanduser()
        if not urdf_path.is_absolute():
            urdf_path = ROOT / "gello_teleop/config" / urdf_path
    urdf_path = Path(urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise RuntimeError(f"Offline torque URDF not found: {urdf_path}; provide --urdf")
    model = PinocchioMomentumModel(urdf_path, settings.get("gravity_m_s2", (0., 0., -9.81)))
    notes = [f"Offline estimate from current URDF: {urdf_path}",
             f"URDF SHA256: {hashlib.sha256(urdf_path.read_bytes()).hexdigest()}"]
    payload = _validated_payload(settings.get("payload"))
    if payload and payload.get("source") == "controller":
        notes.append("Controller TCP payload was not recorded: no extra tool payload added to URDF.")
    elif payload:
        model.set_payload(payload)
        notes.append(f"Static flange payload from recorded config: {payload}")
    else:
        notes.append("No extra flange payload: URDF inertias only.")
    method = settings.get("dynamics_method", "rnea")
    estimator = URDFTorqueEstimator(
        model, dynamics_method=method, tau_ext_mean_window=1,
        measured_torque_cutoff_hz=settings.get("measured_torque_cutoff_hz", 3.),
        acceleration_cutoff_hz=settings.get("acceleration_cutoff_hz", 3.),
        max_gap_s=settings.get("maximum_sample_gap_s", .03),
        state_filters=settings.get("state_filters"),
        measured_torque_filter_enabled=settings.get("measured_torque_filter_enabled", True),
        momentum_observer_bandwidth_hz=settings.get("momentum_observer_bandwidth_hz", 10.),
        friction=settings.get("friction"))
    arrays = {}
    sources = {}
    for key, internal in (("q", "q_follower"), ("dq", "dq_follower"),
                          ("torque", "tau_follower"), ("ddq", "ddq_follower")):
        values, source = read_signal(group, internal, arm)
        if values is None and (key != "ddq" or method == "rnea"):
            raise RuntimeError(f"Offline {method} estimation requires recorded {arm} {key}")
        arrays[key] = values
        if source:
            sources[key] = source
    sample_us = np.asarray(group["timestamp_us"][:], dtype=np.int64)
    reports, _ = read_signal(group, "q_follower_timestamp_us", arm, width=1)
    acquired, _ = read_signal(group, "q_follower_acquired_timestamp_us", arm, width=1)
    if reports is None:
        reports = sample_us[:, None]
        notes.append("Source report timestamps unavailable; using teleop sample timestamps.")
    if acquired is None:
        acquired = reports
    output = np.full((len(sample_us), 7), np.nan)
    for i, timestamp_us in enumerate(sample_us):
        if not np.isfinite(reports[i, 0]) or not np.isfinite(acquired[i, 0]):
            estimator.reset()
            continue
        state = SimpleNamespace(
            timestamp_us=int(reports[i, 0]), acquired_timestamp_us=int(acquired[i, 0]),
            q=arrays["q"][i], dq=arrays["dq"][i], torque=arrays["torque"][i],
            ddq=np.full(7, np.nan) if arrays["ddq"] is None else arrays["ddq"][i],
            q_valid=True, dq_valid=True, torque_valid=True, ddq_valid=True)
        result = estimator.update(int(timestamp_us), state)
        if result.valid:
            output[i] = result.tau_ext_unaveraged
    notes.extend([f"Method: {method}; sign is measured torque minus model baseline and friction.",
                  "No moving mean applied to tau_ext_cal; invalid/initialization samples remain NaN.",
                  f"Inputs: {sources}", f"Friction: {estimator.friction.metadata()}",
                  "Derived for viewing only; original H5 was not modified."])
    return output, notes


def load_episode(path, *, arm="right", camera=None, estimate_tau_ext=True, urdf_path=None, no_images=False):
    import h5py
    import yaml

    path = Path(path).expanduser().resolve()
    notes = []
    with h5py.File(path, "r") as h5:
        if "teleop/timestamp_us" not in h5:
            raise RuntimeError(f"{path.name} has no teleop/timestamp_us")
        group = h5["teleop"]
        timestamps = np.asarray(group["timestamp_us"][:], dtype=np.int64)
        if timestamps.ndim != 1 or not len(timestamps) or np.any(np.diff(timestamps) <= 0):
            raise RuntimeError("Teleop timestamps must be nonempty and strictly increasing")
        signals, sources = {}, {}
        for key, internal in (("q", "q_follower"), ("q_cmd", "q_cmd"), ("tau_ext_cal", "tau_ext_cal")):
            values, source = read_signal(group, internal, arm)
            if values is not None:
                signals[key], sources[key] = values, source
        if "q" not in signals:
            raise RuntimeError(f"No recorded {arm} seven-joint trajectory")
        pose, source = read_signal(group, "ee_pose_follower", arm, pose=True)
        if source:
            sources["ee_pose"] = source
        else:
            notes.append(f"No recorded {arm} TCP pose: 3D trajectory unavailable.")
        if "tau_ext_cal" not in signals:
            if estimate_tau_ext:
                config = yaml.safe_load(h5["config_yaml"][()]) if "config_yaml" in h5 else {}
                values, estimate_notes = estimate_external_torque(group, config or {}, arm, urdf_path)
                signals["tau_ext_cal"] = values
                sources["tau_ext_cal"] = "OFFLINE URDF ESTIMATE (not recorded)"
                notes.extend(estimate_notes)
            else:
                notes.append("Seven-axis tau_ext_cal was not recorded; no substitute torque is plotted.")
        cameras = {}
        available = sorted(name for name, g in h5.get("cameras", {}).items()
                           if "timestamp_us" in g and "frames" in g)
        selected = [] if no_images else (camera.split("/") if camera else available)
        if len(set(selected)) != len(selected) or set(selected) - set(available):
            raise RuntimeError(f"Unknown/duplicate camera selection {selected}; available={available}")
        origin = int(timestamps[0])
        for name in selected:
            g = h5[f"cameras/{name}"]
            stamps = np.asarray(g["timestamp_us"][:], dtype=np.int64)
            frames = np.asarray(g["frames"][:], dtype=np.uint8)
            if (not len(stamps) or stamps.shape != (len(frames),) or np.any(np.diff(stamps) <= 0)
                    or frames.ndim != 4 or frames.shape[-1] != 3):
                raise RuntimeError(f"Invalid RGB frames/timeline in cameras/{name}")
            cameras[name] = ((stamps-origin)*1e-6, frames)
    return EpisodeData(path, arm, (timestamps-timestamps[0])*1e-6, signals, pose, cameras, sources, notes)


def log_episode(data, save_path=None):
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise RuntimeError("Install rerun-sdk==0.26.2, or run with the existing pinn Python environment") from exc
    root = data.arm
    estimated = data.sources.get("tau_ext_cal", "").startswith("OFFLINE")
    views = [rrb.Spatial3DView(origin=f"/{root}/tcp", name=f"{root} TCP trajectory [m]"),
             rrb.TimeSeriesView(origin=f"/{root}/q", name=f"{root} J1..J7 q [rad]"),
             rrb.TimeSeriesView(origin=f"/{root}/tau_ext_cal",
                                name="tau_ext_cal J1..J7 [Nm]" + (" — offline estimate" if estimated else ""))]
    if "q_cmd" in data.signals:
        views.append(rrb.TimeSeriesView(origin=f"/{root}/q_cmd", name=f"{root} J1..J7 target q [rad]"))
    views.append(rrb.TextDocumentView(origin="/info", name="Sources / estimation notes"))
    layout = rrb.Vertical(*views)
    if data.cameras:
        layout = rrb.Horizontal(rrb.Vertical(*(rrb.Spatial2DView(origin=f"/camera/{name}", name=name)
                                             for name in data.cameras)), layout, column_shares=[1, 2])
    blueprint = rrb.Blueprint(layout, collapse_panels=True)
    if save_path is None:
        # An absolute Python command does not activate its conda environment.
        # Rerun's spawn searches PATH, so prefer the matching environment's CLI.
        python_bin = Path(sys.executable).parent
        viewer_executable = python_bin / "rerun"
        if viewer_executable.is_file() and os.access(viewer_executable, os.X_OK):
            os.environ["PATH"] = str(python_bin) + os.pathsep + os.environ.get("PATH", "")
    rr.init("xarm_h5_review", spawn=save_path is None, default_blueprint=blueprint)
    if save_path is not None:
        save_path = Path(save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(save_path, default_blueprint=blueprint)
    summary = "\n".join([str(data.path), f"Arm: {data.arm}", str(data.sources), *data.notes])
    rr.log("info", rr.TextDocument(summary), static=True)
    for key, values in data.signals.items():
        unit = "Nm" if key == "tau_ext_cal" else "rad"
        for i in range(7):
            entity = f"{root}/{key}/J{i+1}"
            label = f"J{i+1} {key} [{unit}]" + (" (offline estimate)" if key == "tau_ext_cal" and estimated else "")
            rr.log(entity, rr.SeriesLines(names=label, colors=[COLORS[i]]), static=True)
            rr.send_columns(entity, indexes=[rr.TimeColumn(TIMELINE, duration=data.time_s)],
                            columns=rr.Scalars.columns(scalars=values[:, i]))
    if data.ee_pose is not None:
        points = data.ee_pose[:, :3, 3]
        valid = np.isfinite(data.ee_pose).all(axis=(1, 2))
        # Separate path segments across missing samples rather than bridging gaps.
        boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        strips = [points[a:b] for a, b in zip(boundaries[::2], boundaries[1::2]) if b-a >= 2]
        rr.log(f"{root}/tcp", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        if strips:
            rr.log(f"{root}/tcp/path", rr.LineStrips3D(strips, colors=[COLORS[0]], radii=[.002]), static=True)
        if valid.any():
            rr.send_columns(f"{root}/tcp/current", indexes=[rr.TimeColumn(TIMELINE, duration=data.time_s[valid])],
                            columns=rr.Points3D.columns(positions=points[valid],
                                                       radii=np.full(valid.sum(), .005)))
    for name, (times, frames) in data.cameras.items():
        for timestamp, frame in zip(times, frames):
            rr.set_time(TIMELINE, duration=float(timestamp))
            rr.log(f"camera/{name}/image", rr.Image(frame, color_model="RGB"))
    rr.disconnect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="H5 file or episode folder")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--camera", "--cam")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--urdf", type=Path, help="override current URDF for offline estimation")
    parser.add_argument("--no-estimate-tau-ext", action="store_true")
    parser.add_argument("--no-images", action="store_true", help="only load/render low-dimensional channels")
    parser.add_argument("--check", action="store_true", help="load and estimate without Rerun GUI")
    parser.add_argument("--save", type=Path, help="export .rrd instead of spawning a viewer")
    args = parser.parse_args(argv)
    if args.save is not None and args.save.suffix.lower() != ".rrd":
        parser.error("--save requires an .rrd filename")
    path = args.path.expanduser().resolve()
    try:
        if path.is_dir():
            if args.episode is None:
                parser.error("directory input requires --episode")
            episodes = discover_episodes(path)
            if args.episode not in episodes:
                parser.error(f"episode {args.episode} not found in {path}")
            path = episodes[args.episode]
        data = load_episode(path, arm=args.arm, camera=args.camera,
                            estimate_tau_ext=not args.no_estimate_tau_ext,
                            urdf_path=args.urdf, no_images=args.no_images)
        report = {"path": str(data.path), "arm": data.arm, "samples": len(data.time_s),
                  "signals": {k: {"shape": list(v.shape), "valid_rows": int(np.isfinite(v).all(axis=1).sum()),
                                  "source": data.sources[k]} for k, v in data.signals.items()},
                  "cameras": {name: len(frames) for name, (_, frames) in data.cameras.items()},
                  "notes": data.notes}
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if not args.check:
            log_episode(data, args.save)
            print("Rerun recording saved." if args.save else "Rerun stream sent; inspect it in the viewer.")
    except (RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
