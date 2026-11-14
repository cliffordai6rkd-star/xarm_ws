#!/usr/bin/env python3
"""Export camera observations from Nero/xArm episode H5 files to MP4.

Each camera group is encoded to a high-quality H.264 MP4 (libx264, CRF 14 by
default) that can be sent through WeChat. ``--combined`` additionally writes a
single side-by-side video, which is usually the easiest to share.

Usage::

    # one episode
    python scripts/export_h5_video.py path/to/episode.h5 --out ~/Desktop/share

    # every episode in a directory, plus one combined video per episode
    python scripts/export_h5_video.py 3cam_data/h5_raw/cut_cucumber \
        --out ~/Desktop/share --combined

    # include aligned depth as normalized grayscale (real units in H5 are mm)
    python scripts/export_h5_video.py episode.h5 --depth --depth-max-mm 3000

Send the resulting ``.mp4`` files in WeChat as 文件/File, not as a 视频/Video
message: the latter is transcoded again and loses quality. The source frames
are written at their native resolution; upscaling would only add file size, not
detail.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _group_fps(group) -> float:
    for key in ('measured_fps', 'fps'):
        value = group.attrs.get(key)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                return value
    if 'timestamp_us' in group:
        stamps = np.asarray(group['timestamp_us'][:], dtype=np.int64)
        if len(stamps) > 1 and stamps[-1] > stamps[0]:
            return (len(stamps)-1)*1e6/(stamps[-1]-stamps[0])
    return 30.0


class _Writer:
    """RGB uint8 frames -> MP4. Prefers libx264; falls back to OpenCV."""

    def __init__(self, path: Path, width: int, height: int, fps: float,
                 crf: int, encoder: str):
        self.path, self.width, self.height = path, int(width), int(height)
        self.fps, self.kind, self.proc, self.cv, self.cv_writer = fps, None, None, None, None
        path.parent.mkdir(parents=True, exist_ok=True)
        use_ffmpeg = encoder in ('auto', 'ffmpeg') and shutil.which('ffmpeg') is not None
        if use_ffmpeg:
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', f'{self.width}x{self.height}', '-r', f'{fps:.6f}', '-i', '-',
                '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf),
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path),
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.kind = 'ffmpeg'
            return
        if encoder == 'ffmpeg':
            raise RuntimeError('ffmpeg is not installed; install it or use --encoder opencv')
        import cv2
        self.cv = cv2
        self.cv_writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.width, self.height))
        if not self.cv_writer.isOpened():
            raise RuntimeError(f'OpenCV could not open video writer: {path}')
        self.kind = 'opencv'

    def write(self, rgb: np.ndarray) -> None:
        frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        if self.kind == 'ffmpeg':
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv_writer.write(frame[..., ::-1])

    def close(self) -> None:
        if self.kind == 'ffmpeg':
            if self.proc is not None and self.proc.stdin is not None:
                self.proc.stdin.close()
                code = self.proc.wait()
                if code:
                    raise RuntimeError(f'ffmpeg failed with exit code {code}: {self.path}')
            self.proc = None
        if self.cv_writer is not None:
            self.cv_writer.release()
            self.cv_writer = None


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f'unsupported camera frame shape {frame.shape}')
    return frame


def _depth_to_rgb(depth: np.ndarray, max_mm: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    gray = np.clip(depth/max_mm, 0., 1.)*255.
    return np.repeat(gray.astype(np.uint8)[..., None], 3, axis=2)


def _export_frames(frames, out: Path, fps: float, crf: int, encoder: str,
                   total: int, label: str) -> None:
    first = _as_rgb(frames[0])
    height, width = first.shape[:2]
    writer = _Writer(out, width, height, fps, crf, encoder)
    try:
        for index in range(total):
            frame = _as_rgb(frames[index])
            if frame.shape[:2] != (height, width):
                raise RuntimeError(f'{label}: frame {index} changed shape '
                                   f'{frame.shape[:2]} != {(height, width)}')
            writer.write(frame)
            if (index+1) % 300 == 0 or index+1 == total:
                print(f'  {label}: {index+1}/{total}', flush=True)
    finally:
        writer.close()


def _export_combined(groups, out: Path, fps: float, crf: int, encoder: str,
                     label: str) -> None:
    heights = [int(group['frames'].shape[1]) for _, group, _ in groups]
    widths = [int(group['frames'].shape[2]) for _, group, _ in groups]
    width, height = sum(widths), max(heights)
    counts = [int(group['frames'].shape[0]) for _, group, _ in groups]
    total = min(counts)
    writer = _Writer(out, width, height, fps, crf, encoder)
    try:
        for index in range(total):
            canvas = np.zeros((height, width, 3), np.uint8)
            offset = 0
            for (name, group, _), camera_width in zip(groups, widths):
                frame = _as_rgb(group['frames'][index])
                canvas[:frame.shape[0], offset:offset+frame.shape[1]] = frame
                offset += camera_width
            writer.write(canvas)
            if (index+1) % 300 == 0 or index+1 == total:
                print(f'  {label}: {index+1}/{total}', flush=True)
    finally:
        writer.close()


def export_file(path: Path, out_dir: Path, *, cameras, crf: int, encoder: str,
                combined: bool, depth: bool, depth_max_mm: float) -> list[Path]:
    import h5py

    written: list[Path] = []
    with h5py.File(path, 'r') as h5:
        camera_root = h5.get('cameras')
        if camera_root is None:
            print(f'{path}: no cameras group; skipped', flush=True)
            return written
        available = [name for name in sorted(camera_root.keys())
                     if 'frames' in camera_root[name]]
        if cameras:
            missing = [name for name in cameras if name not in available]
            if missing:
                raise KeyError(f'{path}: missing cameras {missing}; available={available}')
            selected = list(cameras)
        else:
            selected = available
        if not selected:
            print(f'{path}: no camera frames; skipped', flush=True)
            return written

        groups = []
        for name in selected:
            group = camera_root[name]
            fps = _group_fps(group)
            groups.append((name, group, fps))
            individual = out_dir/f'{path.stem}_{name}.mp4'
            _export_frames(group['frames'], individual, fps, crf, encoder,
                           int(group['frames'].shape[0]), f'{path.name}/{name}')
            written.append(individual)
            if depth and 'depth' in group:
                depth_out = out_dir/f'{path.stem}_{name}_depth.mp4'
                depth_frames = group['depth']
                total = int(depth_frames.shape[0])
                # Depth cannot stream through _export_frames (uint16 -> uint8).
                writer = _Writer(depth_out, depth_frames.shape[2], depth_frames.shape[1],
                                 fps, crf, encoder)
                try:
                    for index in range(total):
                        writer.write(_depth_to_rgb(depth_frames[index], depth_max_mm))
                        if (index+1) % 300 == 0 or index+1 == total:
                            print(f'  {path.name}/{name}/depth: {index+1}/{total}', flush=True)
                finally:
                    writer.close()
                written.append(depth_out)

        if combined and len(groups) > 1:
            merged = out_dir/f'{path.stem}_combined.mp4'
            _export_combined(groups, merged, groups[0][2], crf, encoder,
                             f'{path.name}/combined')
            written.append(merged)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Export camera observations from episode H5 files to high-quality MP4')
    parser.add_argument('paths', nargs='+', type=Path,
                        help='H5 file(s) or directory(ies) containing *.h5')
    parser.add_argument('--out', type=Path, default=None,
                        help='output directory (default: <input>/videos)')
    parser.add_argument('--camera', action='append', default=None,
                        help='camera name to export; repeatable (default: all)')
    parser.add_argument('--crf', type=int, default=14,
                        help='x264 quality: lower is clearer and larger (default 14; 10=max for sharing)')
    parser.add_argument('--combined', action='store_true',
                        help='also write one side-by-side video per episode')
    parser.add_argument('--depth', action='store_true',
                        help='also export aligned depth as normalized grayscale')
    parser.add_argument('--depth-max-mm', type=float, default=3000.,
                        help='depth white point in millimetres (default 3000)')
    parser.add_argument('--encoder', choices=('auto', 'ffmpeg', 'opencv'), default='auto')
    parser.add_argument('--recursive', action='store_true',
                        help='recurse into subdirectories when a directory is given')
    parser.add_argument('--latest', type=int, default=None,
                        help='export only the N most recently modified H5 files')
    args = parser.parse_args(argv)
    if not 0 <= args.crf <= 51:
        parser.error('--crf must be in [0, 51]')
    if not args.depth_max_mm > 0:
        parser.error('--depth-max-mm must be positive')
    if args.latest is not None and args.latest <= 0:
        parser.error('--latest must be positive')

    files: list[Path] = []
    for path in args.paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.rglob('*.h5') if args.recursive else path.glob('*.h5')))
        elif path.is_file():
            files.append(path)
        else:
            parser.error(f'no such file or directory: {path}')
    if not files:
        parser.error('no .h5 files found')
    if args.latest is not None:
        files = sorted(files, key=lambda item: item.stat().st_mtime,
                       reverse=True)[:args.latest]
        files.sort(key=lambda item: item.stat().st_mtime)
    first_dir = args.paths[0].expanduser().resolve()
    default_out = (first_dir if first_dir.is_dir() else first_dir.parent)/'videos'
    out_dir = (args.out or default_out).expanduser().resolve()

    for path in files:
        print(f'exporting {path} -> {out_dir}', flush=True)
        written = export_file(path, out_dir, cameras=args.camera, crf=args.crf,
                              encoder=args.encoder, combined=args.combined,
                              depth=args.depth, depth_max_mm=args.depth_max_mm)
        for item in written:
            print(f'  wrote {item}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
