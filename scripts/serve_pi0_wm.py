#!/usr/bin/env python3
"""Serve local pi0 weights in the openpi environment; no robot or camera access."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='inference/configs/pi0_wm_xarm.yaml')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--check', action='store_true',
                        help='validate paths/interface only; do not load model or open port')
    args = parser.parse_args(argv)
    path = Path(args.config).expanduser().resolve()
    document = yaml.safe_load(path.read_text())
    settings = dict(document['pi0'])
    settings['local'] = True  # Server always loads weights; YAML selects client transport.
    for key in ('local_source_path', 'local_checkpoint'):
        source = Path(settings[key]).expanduser()
        settings[key] = (source if source.is_absolute() else path.parent / source).resolve()
    client_source = document['policy'].get('client_source_path')
    if client_source:
        source = Path(client_source).expanduser()
        sys.path.insert(0, str((source if source.is_absolute() else path.parent / source).resolve()))
    from inference.policies.pi0_joint import LocalPiConnection, Pi0JointPolicy

    Pi0JointPolicy(settings)  # Validate contract and checkpoint folders without loading.
    if not (settings['local_source_path'] / 'openpi/training/config.py').is_file():
        parser.error('local_source_path does not contain openpi/training/config.py')
    port = args.port if args.port is not None else settings['port']
    if args.check:
        print(f"Config OK: {settings['local_checkpoint']}; ws://{args.host}:{port}; no model loaded")
        return
    logging.basicConfig(level=logging.INFO)
    policy = LocalPiConnection(settings)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    try:
        logging.info('Serving pi0 on ws://%s:%s', args.host, port)
        WebsocketPolicyServer(policy, host=args.host, port=port,
                              metadata=policy.get_server_metadata()).serve_forever()
    finally:
        policy.close()


if __name__ == '__main__':
    main()
