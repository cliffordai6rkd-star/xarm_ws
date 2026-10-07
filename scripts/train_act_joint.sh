#!/usr/bin/env bash
# Compatibility entry point; the full training launcher lives in the ACT repo.
set -euo pipefail

act_compat_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
act_compat_source_dir="$(dirname -- "$(dirname -- "$act_compat_script_dir")")/act"
exec bash "$act_compat_source_dir/scripts/train_act_joint.sh" "$@"
