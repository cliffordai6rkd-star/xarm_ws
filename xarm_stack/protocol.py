from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

import numpy as np


def encode(value: Any) -> bytes:
    return json.dumps(_json_value(value), separators=(",", ":")).encode("utf-8")


def decode(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


class RpcServer:
    def __init__(self, endpoint: str, methods: dict[str, Callable[..., Any]]) -> None:
        import zmq

        self.endpoint = endpoint
        self.methods = methods
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.linger = 0
        self.socket.bind(endpoint)
        self._closed = False

    def serve_forever(self) -> None:
        while not self._closed:
            request = decode(self.socket.recv())
            try:
                method = str(request["method"])
                fn = self.methods[method]
                result = fn(**dict(request.get("params", {})))
                self.socket.send(encode({"ok": True, "result": result}))
            except Exception as exc:
                self.socket.send(encode({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))

    def close(self) -> None:
        self._closed = True
        self.socket.close(0)
        self.context.term()


class RpcClient:
    def __init__(self, endpoint: str, timeout_s: float = 2.0) -> None:
        import zmq

        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self.context = zmq.Context()
        self._lock = threading.Lock()
        self._socket = None
        self._reset_socket()

    def call(self, method: str, **params: Any) -> Any:
        import zmq

        with self._lock:
            socket = self._socket
            socket.send(encode({"method": method, "params": params}))
            if socket.poll(int(self.timeout_s * 1000), zmq.POLLIN) == 0:
                self._reset_socket()
                raise TimeoutError(f"RPC timeout calling {method} at {self.endpoint}")
            response = decode(socket.recv())
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "remote service failed"))
            return response.get("result")

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                self._socket.close(0)
                self._socket = None
            self.context.term()

    def _reset_socket(self) -> None:
        import zmq

        if self._socket is not None:
            self._socket.close(0)
        self._socket = self.context.socket(zmq.REQ)
        self._socket.linger = 0
        self._socket.connect(self.endpoint)


class StatePublisher:
    def __init__(self, endpoint: str) -> None:
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.linger = 0
        self.socket.bind(endpoint)

    def publish(self, state: dict[str, Any]) -> None:
        import zmq

        self.socket.send(encode(state), zmq.NOBLOCK)

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()


class StateSubscriber:
    def __init__(self, endpoint: str) -> None:
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.linger = 0
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(endpoint)
        self._state: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="state-subscriber", daemon=True)
        self._thread.start()

    @property
    def state(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._state is None else dict(self._state)

    def wait(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            state = self.state
            if state is not None:
                return state
            time.sleep(0.002)
        raise TimeoutError(f"no state received from {self.endpoint}")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.socket.close(0)
        self.context.term()

    def _run(self) -> None:
        import zmq

        while not self._stop.is_set():
            if self.socket.poll(100, zmq.POLLIN):
                state = decode(self.socket.recv())
                with self._lock:
                    self._state = state
