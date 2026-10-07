#!/usr/bin/env python3
"""Offline openpi websocket interface test. Returns zeros; use --backend mock."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    path = Path(args.config).expanduser().resolve()
    document = yaml.safe_load(path.read_text())
    source = document["policy"].get("client_source_path")
    if source:
        source = Path(source).expanduser()
        sys.path.insert(0, str((source if source.is_absolute() else path.parent / source).resolve()))
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve
    from inference.policies.pi0_joint import MockPiConnection

    mock = MockPiConnection(document["pi0"])

    def handle(socket):
        socket.send(msgpack_numpy.packb(mock.get_server_metadata()))
        for packet in socket:
            request = msgpack_numpy.unpackb(packet)
            response = mock.infer(request)
            socket.send(msgpack_numpy.packb(response))
            print(f"request={mock.calls} keys={list(request)} "
                  f"shapes={ {key: getattr(value, 'shape', None) for key, value in request.items()} }",
                  flush=True)

    port = args.port if args.port is not None else document["pi0"]["port"]
    print(f"Mock pi0 websocket server ws://{args.host}:{port}; no model loaded", flush=True)
    with serve(handle, args.host, port, compression=None, max_size=None) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
