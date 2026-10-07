#!/usr/bin/env python3
"""Read-only audit of the actual openpi training action column and its aliases.

Run with --train-config inside the server's openpi environment. No model inference
or robot connection is performed. A dataset root alone audits local Parquet files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def audit_dataset(root, action_key="action"):
    root = Path(root).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise ValueError(f"No training Parquet files under {root / 'data'}")
    candidates = ("action.joint", "action.ee_pose")
    matches = {key: True for key in candidates}
    seen = {key: 0 for key in candidates}
    rows = 0
    first = None
    errors = {key: 0. for key in candidates}
    for path in paths:
        columns = pq.read_schema(path).names
        if action_key not in columns:
            raise ValueError(f"Training column {action_key!r} absent in {path}; available={columns}")
        selected = list(dict.fromkeys([action_key, *(k for k in candidates if k in columns)]))
        table = pq.read_table(path, columns=selected)
        values = np.asarray(table[action_key].to_pylist(), dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"Nonfinite or invalid training action in {path}: {values.shape}")
        if first is None and len(values):
            first = values[:3].tolist()
        rows += len(values)
        for key in candidates:
            if key not in columns:
                matches[key] = False
                continue
            source = np.asarray(table[key].to_pylist(), dtype=float)
            seen[key] += len(source)
            if source.shape != values.shape or not np.isfinite(source).all():
                matches[key] = False
                continue
            if len(source):
                errors[key] = max(errors[key], float(np.abs(values-source).max()))
            matches[key] &= np.array_equal(values, source)
    matching = [key for key in candidates if matches[key] and seen[key] == rows and rows > 0]
    if rows == 0:
        raise ValueError("Training dataset contains no action rows")
    semantic = {"action.joint": "joint", "action.ee_pose": "eepose"}.get(
        matching[0] if len(matching) == 1 else None, "unverified")
    return {"dataset_root": str(root), "action_column": action_key,
            "parquet_files": len(paths), "rows_checked": rows,
            "fps": info.get("fps"), "action_feature": info.get("features", {}).get(action_key),
            "first_3_actions": first, "exact_matching_columns": matching,
            "candidate_comparison": {key: {"rows_present": seen[key], "matches_all_rows": key in matching,
                                           "max_abs_difference": errors[key] if seen[key] else None}
                                     for key in candidates},
            "data_semantic": semantic,
            "note": "Alias equality proves the target column source, not deployment YAML declarations. "
                    "Review input/output transforms and checkpoint provenance before execution."}


def training_dataset(train_config):
    # Use the SAME metadata resolver as openpi's training data loader.
    from openpi.training import config as configs
    from openpi.training import data_loader
    from openpi.transforms import RepackTransform, flatten_dict

    train = configs.get_config(train_config)
    data = train.data.create(train.assets_dirs, train.model)
    repacks = [t for t in data.repack_transforms.inputs if isinstance(t, RepackTransform)]
    if len(repacks) != 1:
        raise ValueError("Expected one training RepackTransform; inspect custom transforms manually")
    mapping = flatten_dict(repacks[0].structure)
    action_key = mapping.get("actions")
    if tuple(data.action_sequence_keys) != (action_key,):
        raise ValueError(f"Repack action {action_key!r} differs from action_sequence_keys {data.action_sequence_keys}")
    meta = data_loader.lerobot_dataset.LeRobotDatasetMetadata(data.repo_id)
    report = {"training_config": train_config, "repo_id": data.repo_id,
              "repack_mapping": mapping, "action_sequence_keys": list(data.action_sequence_keys),
              "action_horizon": train.model.action_horizon,
              "data_input_transforms": [repr(t) for t in data.data_transforms.inputs],
              "data_output_transforms": [repr(t) for t in data.data_transforms.outputs],
              "model_input_transforms": [repr(t) for t in data.model_transforms.inputs],
              "model_output_transforms": [repr(t) for t in data.model_transforms.outputs]}
    return Path(meta.root), action_key, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-config", help="actual registered config; run in server openpi environment")
    mode.add_argument("--dataset-root", type=Path, help="audit an explicitly supplied local dataset")
    parser.add_argument("--action-key", default="action", help="for --dataset-root only")
    args = parser.parse_args(argv)
    try:
        if args.train_config:
            root, key, report = training_dataset(args.train_config)
        else:
            root, key, report = args.dataset_root, args.action_key, {}
        report.update(audit_dataset(root, key))
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
