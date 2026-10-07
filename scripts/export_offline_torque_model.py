"""Package a trusted local PINN_LCP experiment for xarm_ws plotting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trusted experiment .pt (pickle)')
    parser.add_argument('--physics', type=Path, required=True)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    spec = dict(cp['spec'])
    if any(spec.get(k) != v for k, v in dict(arch='bilstm', filter='zero5', rate=50, horizon=51).items()):
        raise ValueError('Only the selected zero5/50 Hz/51-frame BiLSTM model is supported')
    if spec.get('no_delta') or spec.get('ddq') or not spec.get('physics_prior'):
        raise ValueError('Expected q/dq/delta_q inputs with an identified physics prior')
    spec.pop('physics_prior')
    artifact = dict(format_version=1, spec=spec, model=cp['model'],
                    normalization={k: torch.as_tensor(v) for k, v in cp['normalization'].items()},
                    coefficients=torch.as_tensor(np.load(args.physics)['coefficients']),
                    urdf_xml=args.urdf.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    manifest = {'model': spec, 'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.checkpoint, args.physics, args.urdf)},
                'artifact_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'purpose': 'Delayed visualization and offline contact review; right-arm validation',
                'filter': '4th-order 5 Hz Butterworth sosfiltfilt, then resample_poly 100 -> 50 Hz',
                'inputs': ['q', 'dq', 'q_cmd - q'], 'target': 'tau - identified dynamics/friction prior'}
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
