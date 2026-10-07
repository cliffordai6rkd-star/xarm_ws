"""Bounded asynchronous prediction/controller recording, with explicit errors."""
from collections import Counter, deque
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np


class EventWriter:
    def __init__(self, output, *, mock, command_source, asynchronous=True, maximum_pending=256):
        self.output = Path(output)
        self.mock, self.command_source = mock, command_source
        self.queue = queue.Queue(maxsize=maximum_pending)
        self.closed = threading.Event()
        self.error = None
        self.counts = Counter()
        self.latencies = deque(maxlen=100)
        self.metrics_lock = threading.Lock()
        self.log = (self.output/'events.jsonl').open('w')
        self.thread = None
        if asynchronous:
            self.thread = threading.Thread(target=self._run, name='dp-wm-event-writer', daemon=True)
            self.thread.start()

    def check_error(self):
        if self.error is not None:
            raise RuntimeError('DP/WM event recording failed') from self.error

    def write(self, event):
        self.check_error()
        if self.closed.is_set():
            raise RuntimeError('DP/WM event writer is closed')
        # Models/controller can reuse buffers; the writer owns its snapshots.
        event = {key: value.copy() if isinstance(value,np.ndarray) else value for key,value in event.items()}
        if self.thread is None:
            self._write(event)
        else:
            try:
                self.queue.put_nowait(event)
            except queue.Full as exc:
                raise RuntimeError('DP/WM event recording queue is full') from exc

    def _write(self, event):
        started = time.perf_counter()
        if event['kind'] in ('dp','wm','controller'):
            kind = event['kind']
            prefix = {'dp':'policy','wm':'prediction','controller':'controller'}[kind]
            filename = f'{prefix}_{self.counts[kind]:06d}.npz'
            arrays = {key:value for key,value in event.items() if isinstance(value,np.ndarray)}
            np.savez(self.output/filename, **arrays, anchor_us=event['anchor_us'],
                plan_version=event.get('plan_version',event.get('version')),
                completed_us=event.get('completed_us',event.get('computed_at_us')),
                mock=self.mock, command_source=self.command_source)
            event = {key:value for key,value in event.items() if not isinstance(value,np.ndarray)}
            event['file'] = filename
            self.counts[kind] += 1
        self.log.write(json.dumps(event)+'\n')
        if event['kind']=='status':
            self.log.flush()
        with self.metrics_lock:
            self.latencies.append((time.perf_counter()-started)*1000)

    def _run(self):
        try:
            while not self.closed.is_set() or not self.queue.empty():
                try:
                    event = self.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                self._write(event)
        except Exception as exc:
            self.error = exc

    def report(self):
        with self.metrics_lock:
            values = list(self.latencies)
        return dict(pending=self.queue.qsize(), write_p95_ms=float(np.percentile(values,95)) if values else None,
                    asynchronous=self.thread is not None)

    def close(self):
        self.closed.set()
        try:
            if self.thread is not None:
                self.thread.join()
            self.check_error()
        finally:
            self.log.close()
