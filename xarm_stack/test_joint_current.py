#!/usr/bin/env python3
"""独立 xArm 关节电流观测工具。

This tool only selects the SDK report channel and reads state.  It does not
infer current from a tuple position or call a torque command.  ``--hold`` is
the only option that enables the arm and sends a position hold at the current
pose.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from nero_collection.config import ArmEndpointConfig
from nero_collection.keyboard import TerminalKeys
from ufactory_devices.robot.xarm_adapter import XArmAdapter

log = logging.getLogger("xarm_joint_current")


@dataclass(frozen=True)
class CurrentSample:
    arm_name: str
    timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    current: np.ndarray
    torque: np.ndarray
    current_valid: bool
    torque_valid: bool
    phase: str
    mode: str
    error: str = ""


class JointCurrentTester:
    def __init__(self, raw: dict[str, Any], arms: tuple[str, ...], *, arm_factory: Callable | None = None):
        self.raw = raw
        self.arm_names = arms
        self.arm_factory = arm_factory or self._default_arm
        self.adapters: dict[str, Any] = {}
        self.samples: list[CurrentSample] = []
        self.phase = "baseline"
        self._restore_signal: dict[str, str] = {}
        self._current_selectable: dict[str, bool] = {}
        self._firmware: dict[str, str] = {}
        self._hold_names: set[str] = set()

    def _default_arm(self, name: str):
        cfg = dict(self.raw.get("arms", {}).get(name, {}))
        ip = cfg.pop("robot_ip", None) or cfg.pop("ip", None)
        if not ip:
            raise ValueError(f"arms.{name}.robot_ip is required")
        dof = int(cfg.pop("dof", 7))
        cfg.setdefault("execution_enabled", False)
        cfg.setdefault("feedback_signal", "current")
        cfg.setdefault("servo_api", "set_servo_angle_j")
        restore = str(cfg.pop("restore_feedback_signal", "torque")).lower()
        if restore not in {"torque", "current"}:
            raise ValueError("restore_feedback_signal must be torque or current")
        self._restore_signal[name] = restore
        return XArmAdapter(ArmEndpointConfig(name=name, rest_q=(0.0,) * dof, config_kwargs={"robot_ip": ip, **cfg}))

    def connect(self, hold: bool = False):
        for name in self.arm_names:
            arm = self.arm_factory(name)
            arm.connect()
            try:
                self._firmware[name] = str(arm._call("get_version", required=False) or "unavailable")
            except Exception:
                self._firmware[name] = "unavailable"
            # The public SDK exposes this selector explicitly.  Do not label
            # the effort component as current until the selector succeeds.
            if not callable(getattr(getattr(arm, "_arm", None), "set_report_tau_or_i", None)):
                log.error("%s: SDK has no set_report_tau_or_i(); current is unsupported", name)
                self._current_selectable[name] = False
                self.adapters[name] = arm
                continue
            self._current_selectable[name] = True
            arm.config.config_kwargs["feedback_signal"] = "current"
            try:
                arm.configure_feedback_report()
            except Exception as exc:
                log.error("%s: failed to select current report: %s", name, exc)
                self._current_selectable[name] = False
                self.adapters[name] = arm
                continue
            state = arm.read_state()
            if not state.current_valid:
                log.error("%s: current report selected but get_joint_states effort is invalid", name)
            if hold:
                arm.config.config_kwargs["execution_enabled"] = True
                arm.enable()
                arm.command_joint_positions(state.q)
                self._hold_names.add(name)
            self.adapters[name] = arm

    def set_phase(self, phase: str):
        phase = str(phase).lower()
        if phase not in {"baseline", "press", "release"}:
            raise ValueError("phase must be baseline, press, or release")
        self.phase = phase

    def sample_once(self):
        timestamp_us = time.time_ns() // 1000
        for name in self.arm_names:
            arm = self.adapters[name]
            try:
                state = arm.read_state()
                current_ok = self._current_selectable.get(name, False) and state.current_valid
                current = state.current.copy() if current_ok else np.full(arm.dof, np.nan)
                torque = state.torque.copy() if state.torque_valid else np.full(arm.dof, np.nan)
                mode = "current_report" if self._current_selectable.get(name, False) else "unsupported"
                error_status = self._error_status(arm)
                self.samples.append(CurrentSample(name, timestamp_us, state.q.copy(), state.dq.copy(), current, torque,
                                                   bool(current_ok), bool(state.torque_valid), self.phase, mode, error_status))
            except Exception as exc:
                dof = arm.dof
                self.samples.append(CurrentSample(name, timestamp_us, np.full(dof, np.nan), np.full(dof, np.nan),
                                                   np.full(dof, np.nan), np.full(dof, np.nan), False, False,
                                                   self.phase, "error", str(exc)))

    def run(self, duration_s: float, *, interactive: bool = True, hold: bool = False):
        try:
            self.connect(hold=hold)
            started = time.monotonic()
            period = 1.0 / float(self.raw.get("sample_rate_hz", 100.0))
            if interactive:
                with TerminalKeys() as keys:
                    if not keys.is_tty:
                        raise RuntimeError("interactive keyboard unavailable; use --noninteractive")
                    print("b=baseline, p=press, l=release, q=quit", flush=True)
                    self._loop(started, duration_s, period, keys)
            else:
                self._loop(started, duration_s, period, None)
        finally:
            self.close()

    def _loop(self, started, duration_s, period, keys):
        next_t = time.monotonic()
        while time.monotonic() - started < duration_s:
            if keys is not None:
                key = keys.read_key(0.0)
                if key in {"q", "Q", "\x03"}:
                    break
                if key in {"b", "B"}:
                    self.set_phase("baseline")
                elif key in {"p", "P"}:
                    self.set_phase("press")
                elif key in {"l", "L"}:
                    self.set_phase("release")
            self.sample_once()
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))

    def close(self):
        for name, arm in self.adapters.items():
            try:
                selector = 0 if self._restore_signal.get(name, "torque") == "torque" else 1
                arm._check_result(arm._call("set_report_tau_or_i", selector, required=True), "restore_feedback_report")
            except Exception:
                log.exception("failed to restore feedback selector for %s", name)
            try:
                if name in self._hold_names:
                    arm.disable()
                arm.disconnect()
            except Exception:
                log.exception("failed to disconnect %s", name)
        self.adapters.clear()

    @staticmethod
    def _error_status(arm) -> str:
        try:
            result = arm._call("get_err_warn_code", required=False)
            if isinstance(result, tuple) and len(result) >= 2:
                code, values = result[0], result[1]
                if int(code) != 0:
                    return f"sdk_code={int(code)}"
                values = np.asarray(values).reshape(-1)
                return "" if not values.size or int(values[0]) == 0 else f"error_code={int(values[0])}"
            return ""
        except Exception as exc:
            return f"error_status_unavailable={exc}"

    def save(self, path: str | Path):
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.samples:
            raise RuntimeError("no current samples were collected")
        timestamps = np.asarray([s.timestamp_us for s in self.samples], dtype=np.int64)
        q = np.stack([s.q for s in self.samples])
        dq = np.stack([s.dq for s in self.samples])
        current = np.stack([s.current for s in self.samples])
        torque = np.stack([s.torque for s in self.samples])
        phases = np.asarray([s.phase for s in self.samples], dtype="S8")
        modes = np.asarray([s.mode for s in self.samples], dtype="S32")
        arm_names = np.asarray([s.arm_name for s in self.samples], dtype="S16")
        current_valid = np.asarray([s.current_valid for s in self.samples], dtype=np.uint8)
        torque_valid = np.asarray([s.torque_valid for s in self.samples], dtype=np.uint8)
        errors = np.asarray([s.error for s in self.samples], dtype="S256")
        if path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(path, "w") as h5:
                h5.attrs["format"] = "xarm_joint_current_observation/v1"
                h5.attrs["arm_names"] = np.asarray(self.arm_names, dtype=h5py.string_dtype())
                h5.attrs["xarm_firmware"] = json.dumps(self._firmware, sort_keys=True)
                h5.attrs["current_source"] = "SDK get_joint_states effort after set_report_tau_or_i(1)"
                h5.attrs["torque_source"] = "SDK effort only when selector is torque; never inferred with current"
                for name, value in (("arm_name", arm_names), ("timestamp_us", timestamps), ("q", q), ("dq", dq), ("current", current),
                                    ("torque", torque), ("phase", phases), ("mode", modes), ("current_valid", current_valid),
                                    ("torque_valid", torque_valid), ("error", errors)):
                    h5.create_dataset(name, data=value)
        else:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["timestamp_us", "arm_name", "phase", "mode", "current_valid", "torque_valid", "q", "dq", "current", "torque", "error"])
                for sample in self.samples:
                    writer.writerow([sample.timestamp_us, sample.arm_name, sample.phase, sample.mode, sample.current_valid,
                                         sample.torque_valid, sample.q.tolist(), sample.dq.tolist(),
                                         sample.current.tolist(), sample.torque.tolist(), sample.error])
        self._save_plot(path)
        self._print_stats()
        return path

    def _save_plot(self, data_path: Path):
        """Write a lightweight current-versus-time plot beside the raw file."""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            log.info("matplotlib is unavailable; raw current data was still saved")
            return
        figure, axes = plt.subplots(len(self.arm_names), 1, squeeze=False,
                                    sharex=True, figsize=(10, 3 * len(self.arm_names)))
        for row, arm_name in enumerate(self.arm_names):
            rows = [s for s in self.samples if s.arm_name == arm_name]
            if not rows:
                continue
            t = (np.asarray([s.timestamp_us for s in rows]) - rows[0].timestamp_us) * 1e-6
            values = np.stack([s.current for s in rows])
            for joint in range(values.shape[1]):
                axes[row, 0].plot(t, values[:, joint], label=f"J{joint + 1}", linewidth=0.8)
            axes[row, 0].set_title(arm_name)
            axes[row, 0].set_ylabel("raw current")
            axes[row, 0].grid(True, alpha=0.25)
            axes[row, 0].legend(loc="upper right", ncol=min(values.shape[1], 4), fontsize="small")
        axes[-1, 0].set_xlabel("time (s)")
        figure.tight_layout()
        figure.savefig(data_path.with_suffix(".png"), dpi=120)
        plt.close(figure)

    def _print_stats(self):
        for arm_name in self.arm_names:
            valid = [s for s in self.samples if s.arm_name == arm_name and s.current_valid and s.phase == "baseline"]
            press = [s for s in self.samples if s.arm_name == arm_name and s.current_valid and s.phase == "press"]
            release = [s for s in self.samples if s.arm_name == arm_name and s.current_valid and s.phase == "release"]
            if not valid:
                print(f"{arm_name} current: unsupported or invalid for all baseline samples", flush=True)
                continue
            base = np.stack([s.current for s in valid])
            base_mean = np.nanmean(base, axis=0)
            print(f"{arm_name} current baseline mean:", base_mean,
                  "noise std:", np.nanstd(base, axis=0), flush=True)
            for label, rows in (("press", press), ("release", release)):
                if rows:
                    delta = np.nanmean(np.stack([s.current for s in rows]), axis=0) - base_mean
                    print(f"{arm_name} {label} minus baseline (signed raw current units):", delta, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--hold", action="store_true", help="enable and hold current pose")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--noninteractive", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(Path(args.config).expanduser().read_text(encoding="utf-8")) or {}
    names = ("left", "right") if args.arm == "both" else (args.arm,)
    tester = JointCurrentTester(raw, names)
    try:
        tester.run(args.duration, interactive=not args.noninteractive, hold=args.hold)
        tester.save(args.output)
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
