"""Dual-GELLO teleoperation entry that records native-resolution RealSense frames.

The original ``dual_gello_collect`` entry is untouched.  This entry reuses the
same pipeline, keyboard workflow and H5 schema, but records every enabled camera
at its native acquisition resolution instead of the policy ``output_size``.  By
default it keeps every acquired frame instead of applying ``output_fps``
selection; pass ``--raw-fps`` to subsample.

Usage::

    python -m gello_teleop.dual_gello_collect_raw_obs \
        -c gello_teleop/config/xarm7_gello_dual_dataset.yaml \
        --output-dir 3cam_data/h5_raw/cut_cucumber

The H5 layout is identical to the normal collection entry; camera groups store
the native-resolution ``frames`` (RGB uint8) and, when configured, ``depth``
(uint16).  Uncompressed native frames are large, so ``--compression lzf`` or
``--compression gzip`` is recommended for long episodes.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import replace
from math import isfinite
from pathlib import Path

import yaml

from gello_teleop.dual_gello_collect import DualGelloPipeline, _exit_interrupt_handler
from nero_collection.config import CollectionConfig
from ufactory_devices.robot.xarm_adapter import XArmControllerFault

log = logging.getLogger("dual_gello_collect_raw_obs")

DEFAULT_OUTPUT_ROOT = Path("3cam_data/h5_raw")


class RawObservationPipeline(DualGelloPipeline):
    """Existing pipeline with camera recording overridden to native resolution."""

    def __init__(self, config_path: str | Path, *, output_dir: str | Path | None = None,
                 output_fps: float | None = None, compression: str = "keep", **kwargs):
        config_path = Path(config_path).expanduser().resolve()
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_ROOT / config_path.stem
        self.raw_output_dir = Path(output_dir).expanduser().resolve()
        if output_fps is not None:
            if isinstance(output_fps, bool) or not isfinite(output_fps) or output_fps <= 0:
                raise ValueError('--raw-fps must be positive and finite')
            output_fps = float(output_fps)
        self.raw_output_fps = output_fps
        if compression not in {"keep", "none", "lzf", "gzip"}:
            raise ValueError('--compression must be keep, none, lzf, or gzip')
        self.raw_compression = compression
        super().__init__(config_path, **kwargs)

    def _make_collection_config(self) -> CollectionConfig:
        config = super()._make_collection_config()
        # output_size=None keeps the native acquisition resolution and lets
        # depth follow the same path; preview_output_size is a separate setting.
        if self.raw_output_fps is not None:
            for camera in config.cameras:
                if self.raw_output_fps > float(camera.fps):
                    raise ValueError(
                        f'--raw-fps {self.raw_output_fps:g} exceeds camera {camera.name} '
                        f'acquisition fps {float(camera.fps):g}')
        cameras = tuple(replace(camera, output_size=None, output_fps=self.raw_output_fps)
                        for camera in config.cameras)
        compression = config.output.camera_compression
        if self.raw_compression != "keep":
            compression = None if self.raw_compression == "none" else self.raw_compression
        output = replace(config.output, directory=self.raw_output_dir,
                         camera_compression=compression)
        # Persist the effective recording settings in the H5 config metadata
        # instead of the policy-size values from the source YAML.
        raw = copy.deepcopy(self.raw)
        for item in raw.get("cameras", []):
            if not isinstance(item, dict) or not item.get("enabled", True):
                continue
            item.pop("output_size", None)
            item["output_fps"] = self.raw_output_fps
        raw.setdefault("output", {}).update(directory=str(self.raw_output_dir),
                                            camera_compression=compression)
        return replace(config, cameras=cameras, output=output,
                       raw_yaml=yaml.safe_dump(raw, sort_keys=False))

    def raw_recording_summary(self) -> list[dict]:
        result = []
        for camera in self.collection_config.cameras:
            result.append({
                "name": camera.name,
                "width": int(camera.width),
                "height": int(camera.height),
                "depth": bool(camera.depth),
                "fps": camera.effective_fps,
                "capture_fps": float(camera.fps),
            })
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="遥操并录制 RealSense 原始分辨率观测（不修改原 dual_gello_collect 入口）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--output-dir", default=None,
                        help=f"原始分辨率 H5 输出目录（相对当前目录）；默认 {DEFAULT_OUTPUT_ROOT}/<配置文件名>")
    parser.add_argument("--raw-fps", type=float, default=None,
                        help="录制帧率上限；默认保留相机原生帧率（不按 output_fps 抽帧）")
    parser.add_argument("--compression", choices=("keep", "none", "lzf", "gzip"), default="keep",
                        help="H5 相机压缩；keep 沿用配置，none 为不压缩")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--auto-save", action="store_true")
    parser.add_argument("--reset-q", action="store_true",
                        help="照常启动遥操；按 s 覆盖保存所选主从臂当前 q 为下次启动参考")
    arms = parser.add_mutually_exclusive_group()
    arms.add_argument('--left', dest='active_arms', action='store_const', const=['left'], help='仅左臂，覆盖 YAML 的 active_arms')
    arms.add_argument('--right', dest='active_arms', action='store_const', const=['right'], help='仅右臂，覆盖 YAML 的 active_arms')
    arms.add_argument('--both', dest='active_arms', action='store_const', const=['left', 'right'], help='双臂，覆盖 YAML 的 active_arms')
    plot = parser.add_mutually_exclusive_group()
    plot.add_argument('--torque-plot', dest='torque_plot', action='store_true', help='开启七轴实测 / URDF 力矩对比图')
    plot.add_argument('--no-torque-plot', dest='torque_plot', action='store_false', help='关闭力矩实时图（无桌面采集）')
    feedback = parser.add_mutually_exclusive_group()
    feedback.add_argument('--force-feedback', dest='force_feedback', action='store_true',
                          help='本次启用 GELLO 力反馈，覆盖 YAML 的 force_feedback.enabled')
    feedback.add_argument('--no-force-feedback', dest='force_feedback', action='store_false',
                          help='本次关闭 GELLO 力反馈，保留独立的主手阻尼')
    parser.set_defaults(torque_plot=None, force_feedback=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = {}
    if args.torque_plot is not None:
        overrides['torque_plot_enabled'] = args.torque_plot
    if args.active_arms is not None:
        overrides['active_arms'] = args.active_arms
    if args.force_feedback is not None:
        overrides['force_feedback_enabled'] = args.force_feedback
    pipeline = RawObservationPipeline(args.config, output_dir=args.output_dir,
                                      output_fps=args.raw_fps,
                                      compression=args.compression, **overrides)
    summary = pipeline.raw_recording_summary()
    if args.check_config:
        try:
            print(json.dumps({"state": pipeline.state,
                              "arm_names": list(pipeline.ARM_NAMES),
                              "sample_rate_hz": pipeline.sample_rate_hz,
                              "force_feedback_enabled": pipeline.force_feedback is not None,
                              "cameras": summary,
                              "output_directory": str(pipeline.collection_config.output.directory),
                              "camera_compression": pipeline.collection_config.output.camera_compression,
                              "gello_config": pipeline.raw.get("gello_config")}, indent=2))
        finally:
            pipeline.close()
        return 0
    for camera in summary:
        log.info('原始分辨率录制 %s：%dx%d @ %.1f fps（采集 %.1f fps），depth=%s',
                 camera['name'], camera['width'], camera['height'], camera['fps'],
                 camera['capture_fps'], camera['depth'])
    log.info('原始分辨率 H5 输出目录：%s（压缩=%s）',
             pipeline.collection_config.output.directory,
             pipeline.collection_config.output.camera_compression or '无')
    user_exit = False
    exit_code = 0
    try:
        pipeline.connect()
        exit_code = pipeline.interactive(auto_save=args.auto_save, reset_q=args.reset_q)
        user_exit = True
    except KeyboardInterrupt:
        user_exit = True
        exit_code = 130
    except ConnectionError as exc:
        log.error('数采连接中断：%s', exc)
        exit_code = 1
    except XArmControllerFault as exc:
        log.error('数采因控制器保护停止：%s', exc)
        exit_code = 1
    finally:
        with _exit_interrupt_handler(pipeline):
            try:
                if user_exit and not pipeline.shutdown_reset_and_hold():
                    exit_code = 1
            finally:
                pipeline.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
