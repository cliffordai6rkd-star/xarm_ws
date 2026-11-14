"""Exercise real WebSockets against an independent OpenPI wire implementation."""
from contextlib import contextmanager
import threading
import time

import msgpack
import numpy as np
import pytest
from websockets.sync.server import serve

from inference.forcevla_transport import ForceVLAClient, pack_message, unpack_message


def upstream_pack(value):
    # Keep this fixture independent of the client encoder: these are the exact
    # byte-key records used by the remote openpi_client.msgpack_numpy module.
    def default(item):
        if isinstance(item, np.ndarray):
            return {b"__ndarray__": True, b"data": item.tobytes(),
                    b"dtype": item.dtype.str, b"shape": item.shape}
        if isinstance(item, np.generic):
            return {b"__npgeneric__": True, b"data": item.item(), b"dtype": item.dtype.str}
        raise TypeError(type(item))
    return msgpack.packb(value, default=default)


def upstream_unpack(payload):
    def hook(item):
        if b"__ndarray__" in item:
            return np.ndarray(buffer=item[b"data"], dtype=np.dtype(item[b"dtype"]),
                              shape=item[b"shape"])
        if b"__npgeneric__" in item:
            return np.dtype(item[b"dtype"]).type(item[b"data"])
        return item
    return msgpack.unpackb(payload, object_hook=hook)


@contextmanager
def running_server(handler):
    with serve(handler, "127.0.0.1", 0, compression=None, close_timeout=.2) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1]
        finally:
            server.shutdown()
            thread.join(timeout=1.)


def test_wire_encoding_matches_upstream_and_preserves_numpy_types():
    source = {"state": np.arange(7, dtype=">f4"),
              "tau_ext": np.arange(14, dtype=np.float64)[::2],
              "image": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
              "scalar": np.float32(1.25), "prompt": "擦拭木板"}
    assert pack_message(source) == upstream_pack(source)
    for restored in (upstream_unpack(pack_message(source)), unpack_message(upstream_pack(source))):
        for key in ("state", "tau_ext", "image"):
            np.testing.assert_array_equal(restored[key], source[key])
            assert restored[key].dtype == source[key].dtype
        assert isinstance(restored["scalar"], np.float32)
        assert restored["prompt"] == source["prompt"]


def test_metadata_handshake_and_real_inference(monkeypatch):
    observed = {}
    expected = np.arange(350, dtype=np.float32).reshape(50, 7)

    def handler(connection):
        connection.send(upstream_pack({"model": "fixture", "rate": np.int64(25)}))
        observed.update(upstream_unpack(connection.recv(timeout=1.)))
        connection.send(upstream_pack({"actions": expected, "server_timing": {"infer_ms": 2.}}))

    # A Tailscale peer must be reached directly even in environments with proxies.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    with running_server(handler) as port:
        client = ForceVLAClient("127.0.0.1", port, connect_timeout_s=1., request_timeout_s=1.)
        try:
            assert client.connect()["model"] == "fixture"
            assert client.metadata["rate"] == 25
            client.metadata["model"] = "changed"
            assert client.connect()["model"] == "fixture"
            request = {"state": np.linspace(0, 1, 7, dtype=np.float32),
                       "tau_ext": np.arange(7, dtype=np.float32),
                       "image": np.zeros((8, 10, 3), np.uint8),
                       "right_wrist_image": np.ones((8, 10, 3), np.uint8),
                       "prompt": "Wipe the board."}
            response = client.infer(request)
            np.testing.assert_array_equal(response["actions"], expected)
            assert response["server_timing"]["infer_ms"] == 2.
            assert set(observed) == set(request)
            for key in ("state", "tau_ext", "image", "right_wrist_image"):
                np.testing.assert_array_equal(observed[key], request[key])
        finally:
            client.close()
            client.close()


def test_remote_traceback_is_preserved_and_connection_closed():
    traceback = "Traceback (most recent call last):\nValueError: broken force guidance\n"

    def handler(connection):
        connection.send(upstream_pack({}))
        connection.recv(timeout=1.)
        connection.send(traceback)

    with running_server(handler) as port:
        client = ForceVLAClient("127.0.0.1", port, request_timeout_s=1.)
        client.start()
        with pytest.raises(RuntimeError) as error:
            client.infer({})
        assert traceback in str(error.value)
        with pytest.raises(RuntimeError, match="not connected"):
            client.infer({})
        client.close()


def test_request_timeout_closes_connection_without_reusing_late_reply():
    release = threading.Event()
    requested = threading.Event()

    def handler(connection):
        connection.send(upstream_pack({}))
        connection.recv(timeout=1.)
        requested.set()
        release.wait(timeout=2.)

    with running_server(handler) as port:
        client = ForceVLAClient("127.0.0.1", port, request_timeout_s=.05)
        client.connect()
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError, match="inference timed out"):
                client.infer({"state": np.zeros(7, np.float32)})
            assert requested.is_set()
            assert time.monotonic() - started < 1.
            with pytest.raises(RuntimeError, match="not connected"):
                client.infer({})
        finally:
            release.set()
            client.close()


def test_close_interrupts_inference_receive_from_another_thread():
    release = threading.Event()
    requested = threading.Event()
    errors = []

    def handler(connection):
        connection.send(upstream_pack({}))
        connection.recv(timeout=1.)
        requested.set()
        release.wait(timeout=2.)

    with running_server(handler) as port:
        client = ForceVLAClient("127.0.0.1", port, request_timeout_s=10.)
        client.connect()

        def infer():
            try:
                client.infer({})
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=infer, daemon=True)
        worker.start()
        try:
            assert requested.wait(timeout=1.)
            client.close()
            worker.join(timeout=1.)
            assert not worker.is_alive()
            assert errors
        finally:
            release.set()
            client.close()


def test_connect_timeout_bounds_missing_metadata_greeting():
    release = threading.Event()

    def handler(connection):
        release.wait(timeout=2.)

    with running_server(handler) as port:
        client = ForceVLAClient("127.0.0.1", port, connect_timeout_s=.05)
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                client.connect()
            assert time.monotonic() - started < 1.
        finally:
            release.set()
            client.close()


class BlockedUpload:
    """A sendall stalled while owning the same mutex needed by WS close()."""

    def __init__(self):
        self.socket = self
        self.protocol_mutex = threading.Lock()
        self.sending = threading.Event()
        self.disconnected = threading.Event()
        self.closed = False

    def send(self, payload):
        with self.protocol_mutex:
            self.sending.set()
            if not self.disconnected.wait(timeout=2.):
                raise AssertionError("Upload was never interrupted")
            raise OSError("socket shutdown interrupted sendall")

    def shutdown(self, how):
        self.disconnected.set()

    def recv(self, timeout):
        raise AssertionError("A stalled upload must not reach recv")

    def close(self):
        with self.protocol_mutex:
            self.closed = True


def test_request_deadline_interrupts_stalled_upload_before_receive():
    connection = BlockedUpload()
    client = ForceVLAClient("fixture", request_timeout_s=.05)
    client._socket = connection
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="inference timed out"):
        client.infer({"image": np.zeros((224, 224, 3), np.uint8)})
    assert time.monotonic() - started < 1.
    assert connection.closed
    with pytest.raises(RuntimeError, match="not connected"):
        client.infer({})


def test_close_interrupts_stalled_upload_without_waiting_for_protocol_mutex():
    connection = BlockedUpload()
    client = ForceVLAClient("fixture", request_timeout_s=10.)
    client._socket = connection
    errors = []

    def infer():
        try:
            client.infer({})
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=infer, daemon=True)
    worker.start()
    assert connection.sending.wait(timeout=1.)
    started = time.monotonic()
    client.close()
    worker.join(timeout=1.)
    assert time.monotonic() - started < 1.
    assert not worker.is_alive()
    assert connection.closed and errors
