"""Isolated H5 writer with bounded transfers from a frozen episode.

Only configuration, metadata and compact numeric blocks cross the pipe.
No SDK, serial port, robot object or whole EpisodeBuffer is sent to the child.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
import traceback

import numpy as np

_TRANSFER_BYTES = 256 * 1024
_TRANSFER_ROWS = 1024
log = logging.getLogger(__name__)


class EpisodeSaveProcess:
    def __init__(self):
        self.process = None
        self.connection = None
        self._lock = threading.RLock()

    @property
    def pid(self):
        return self.process.pid if self.process is not None else None

    def start(self):
        with self._lock:
            if self.process is not None and self.process.is_alive():
                return
            self.close()
            context = mp.get_context('spawn')
            parent, child = context.Pipe()
            process = context.Process(target=_writer_worker, args=(child,),
                                      name='episode-h5-writer', daemon=True)
            try:
                process.start()
                child.close()
                self.process, self.connection = process, parent
                if not parent.poll(30.):
                    raise TimeoutError('episode H5 writer did not start within 30 seconds')
                status, detail = parent.recv()
                if status != 'ready':
                    raise RuntimeError(f'episode H5 writer startup failed: {detail}')
                log.info('独立 H5 保存进程已就绪，pid=%s', process.pid)
            except BaseException:
                child.close()
                parent.close()
                if process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=2.)
                self.process = self.connection = None
                raise

    def save(self, buffer, target):
        """Run in the transfer thread; H5 finalization/writing runs in the child."""
        if buffer.online_tau_ext is not None:
            raise ValueError('isolated dual-GELLO saving requires online_tau_ext disabled')
        self.start()
        with self._lock:
            try:
                self.connection.send(('begin', {
                    'config': buffer.config, 'arm_names': buffer.arm_names,
                    'target': str(target), 'metadata': buffer.episode_metadata,
                    'state_names': buffer.teleop_state_names,
                    'input_frame_count': buffer.input_frame_count,
                    'duplicate_input_frame_count': buffer.duplicate_input_frame_count,
                    'collector_process_id': os.getpid(),
                }))
                self._send_values('timeline', '', buffer.teleop_timestamps_us)
                for name, values in buffer.teleop_data.items():
                    self._send_values('rows', name, values)
                for name, frames in buffer.camera_frames.items():
                    self._send_values('rgb', name, frames)
                    self._send_values('depth', name, buffer.camera_depth_frames.get(name, ()))
                    self._send_values('camera_time', name, buffer.camera_timestamps_us[name])
                self.connection.send(('commit',))
                while not self.connection.poll(.25):
                    if not self.process.is_alive():
                        raise RuntimeError(f'episode H5 writer exited with code {self.process.exitcode}')
                status, detail = self.connection.recv()
            except BaseException:
                # A partial transfer must never be merged into a later episode.
                self.close()
                raise
            if status != 'saved':
                raise RuntimeError(f'独立 H5 保存失败：{detail}')
            return Path(detail)

    def _send_values(self, kind, name, values):
        if not values:
            return
        first = np.asarray(values[0])
        rows = max(1, min(_TRANSFER_ROWS, _TRANSFER_BYTES//max(first.nbytes, 1)))
        start = 0
        while start < len(values):
            batch = values[start:start+rows]
            if kind in {'timeline', 'camera_time'}:
                block = np.asarray(batch, dtype=np.int64)
            else:
                block = np.stack(batch, axis=0)
            if block.nbytes > _TRANSFER_BYTES and len(batch) > 1:
                # Later samples can promote the dtype beyond that of row 0.
                rows = max(1, min(rows-1, _TRANSFER_BYTES//block[0].nbytes))
                continue
            self.connection.send((kind, name, block))
            start += len(batch)
            time.sleep(0)  # Yield between bounded copies/serialization.

    def close(self):
        with self._lock:
            process, connection = self.process, self.connection
            self.process = self.connection = None
            if connection is not None:
                try:
                    if process is not None and process.is_alive():
                        connection.send(('shutdown',))
                except (OSError, EOFError):
                    pass
                connection.close()
            if process is not None and process.pid is not None:
                process.join(timeout=2.)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.)


def _writer_worker(connection):
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    try:
        # Give the collector preference under CPU load; no elevated privileges.
        if hasattr(os, 'nice'):
            os.nice(5)
        from nero_collection.h5_writer import EpisodeBuffer
        connection.send(('ready', os.getpid()))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
        connection.close()
        return
    buffer, target, error = None, None, None
    try:
        while True:
            message = connection.recv()
            kind = message[0]
            if kind == 'shutdown':
                break
            if kind == 'begin':
                header = message[1]
                error = None
                try:
                    buffer = EpisodeBuffer(header['config'], tuple(header['arm_names']), enable_online_tau_ext=False)
                    buffer.episode_metadata = dict(header['metadata'])
                    buffer.episode_metadata.update(writer_process_id=os.getpid(),
                                                   collector_process_id=header['collector_process_id'])
                    buffer.teleop_state_names = dict(header['state_names'])
                    buffer.input_frame_count = header['input_frame_count']
                    buffer.duplicate_input_frame_count = header['duplicate_input_frame_count']
                    target = header['target']
                except Exception:
                    error = traceback.format_exc()
            elif kind == 'commit':
                try:
                    if error is not None:
                        raise RuntimeError(error)
                    if buffer is None:
                        raise RuntimeError('episode transfer has no header')
                    path = buffer.save(target)
                    connection.send(('saved', str(path)))
                except Exception:
                    connection.send(('error', traceback.format_exc()))
                buffer, target, error = None, None, None
            elif error is None:
                try:
                    if buffer is None:
                        raise RuntimeError('episode transfer has no header')
                    _, name, block = message
                    if kind == 'timeline':
                        buffer.teleop_timestamps_us.extend(block)
                    elif kind == 'rows':
                        buffer.teleop_data[name].extend(block)
                    elif kind == 'rgb':
                        buffer.camera_frames[name].extend(block)
                    elif kind == 'depth':
                        buffer.camera_depth_frames[name].extend(block)
                    elif kind == 'camera_time':
                        buffer.camera_timestamps_us[name].extend(block)
                    else:
                        raise RuntimeError(f'unknown episode packet: {kind}')
                except Exception:
                    # Drain the rest of this transfer before returning its error.
                    error = traceback.format_exc()
    except (EOFError, OSError):
        pass
    finally:
        connection.close()
