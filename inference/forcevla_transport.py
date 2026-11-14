"""Bounded WebSocket requests using the existing OpenPI NumPy/msgpack protocol."""
from __future__ import annotations

import math
import socket
import threading
import time

import msgpack
import numpy as np


def _pack_array(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(),
                b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot encode {type(value).__name__} as an OpenPI message")


def _unpack_array(value):
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]),
                          shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack_message(value) -> bytes:
    """Encode precisely the byte-key ndarray representation used by OpenPI."""
    return msgpack.packb(value, default=_pack_array, use_bin_type=True)


def unpack_message(payload: bytes):
    return msgpack.unpackb(payload, object_hook=_unpack_array, raw=False)


def _shutdown_socket(connection):
    # WebSocket send holds its protocol mutex while socket.sendall is running.
    # Raw shutdown wakes that send without waiting for the same mutex in close().
    try:
        connection.socket.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class ForceVLAClient:
    """One connection and one outstanding inference request, without reconnects.

    Connect in the runtime's main thread before starting its policy worker.
    ``close`` may be called from another thread to interrupt an outstanding recv.
    A timeout closes the connection so its late reply can never become the next
    request's result.
    """

    def __init__(self, host: str, port: int = 8000, *, connect_timeout_s: float = 10.,
                 request_timeout_s: float = 60.):
        host = str(host).strip()
        if not host or "/" in host or any(char.isspace() for char in host):
            raise ValueError("host must be a hostname or Tailscale IP without a URL scheme")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        for name, value in (("connect_timeout_s", connect_timeout_s),
                            ("request_timeout_s", request_timeout_s)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        self.uri = f"ws://{host}:{int(port)}"
        self.connect_timeout_s = float(connect_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self._state_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._socket = None
        self._metadata = {}
        self._closed = False

    @property
    def metadata(self) -> dict:
        with self._state_lock:
            return dict(self._metadata)

    def connect(self) -> dict:
        """Connect once and consume the server's initial metadata greeting."""
        from websockets.sync.client import connect

        with self._request_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("ForceVLA client is closed")
                if self._socket is not None:
                    return dict(self._metadata)
            connection = None
            try:
                connection = connect(
                    self.uri, open_timeout=self.connect_timeout_s,
                    close_timeout=min(1., self.connect_timeout_s, self.request_timeout_s),
                    proxy=None, compression=None, max_size=None,
                    # The first inference can block the server's event loop for
                    # minutes while JAX compiles. Keepalive pings would then fail
                    # with 1011 before request_timeout_s elapses. The explicit
                    # request watchdog below already bounds every call.
                    ping_interval=None, ping_timeout=None,
                )
                with self._state_lock:
                    if self._closed:
                        raise RuntimeError("ForceVLA client was closed while connecting")
                    self._socket = connection
                metadata = self._receive(connection, self.connect_timeout_s)
                with self._state_lock:
                    if self._closed:
                        raise RuntimeError("ForceVLA client was closed while connecting")
                    self._metadata = metadata
                return dict(metadata)
            except Exception:
                self.close()
                if connection is not None:
                    connection.close()
                raise

    def start(self) -> dict:
        return self.connect()

    @staticmethod
    def _receive(connection, timeout_s):
        response = connection.recv(timeout=timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"ForceVLA server error:\n{response}")
        result = unpack_message(response)
        if not isinstance(result, dict):
            raise RuntimeError("ForceVLA server returned a non-dictionary message")
        return result

    def infer(self, observation: dict) -> dict:
        with self._request_lock:
            with self._state_lock:
                connection = self._socket
            if connection is None:
                raise RuntimeError("ForceVLA client is not connected; call connect() first")
            expired = threading.Event()
            deadline = time.monotonic() + self.request_timeout_s

            def timeout():
                expired.set()
                _shutdown_socket(connection)

            watchdog = threading.Timer(self.request_timeout_s, timeout)
            watchdog.daemon = True
            watchdog.start()
            try:
                connection.send(pack_message(observation))
                result = self._receive(connection, max(0., deadline - time.monotonic()))
                if expired.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("ForceVLA request deadline elapsed")
                return result
            except Exception as exc:
                self.close()
                if expired.is_set() or isinstance(exc, TimeoutError):
                    raise TimeoutError(
                        f"ForceVLA inference timed out after {self.request_timeout_s:g}s at {self.uri}"
                    ) from exc
                from websockets.exceptions import ConnectionClosed
                if isinstance(exc, ConnectionClosed):
                    # Preserve the peer's close code and reason: a 1011 keepalive
                    # timeout means the server event loop was blocked and the
                    # server must answer heartbeats; a longer recv() cannot help.
                    raise ConnectionError(
                        f"ForceVLA server closed the connection at {self.uri}: "
                        f"code={exc.code}, reason={exc.reason!r}; "
                        "first-inference compilation may have blocked the server event loop "
                        "(disable server-side websockets ping_interval or warm the model first)"
                    ) from exc
                raise
            finally:
                watchdog.cancel()

    def close(self) -> None:
        """Close without taking the request lock, including during a blocked recv."""
        with self._state_lock:
            self._closed = True
            connection, self._socket = self._socket, None
        if connection is not None:
            _shutdown_socket(connection)
            connection.close()
