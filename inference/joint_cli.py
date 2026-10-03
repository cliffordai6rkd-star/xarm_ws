"""DP/ACT -> checkpoint-selected RGB observations -> xArm joint servo CLI."""
from __future__ import annotations

import argparse
import json
import logging

from inference.joint_config import JointDeployment, load_joint_policy, policy_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="DP / ACT xArm joint inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="override policy checkpoint file or training output directory")
    parser.add_argument("--device", help="override policy device, e.g. cpu or cuda:0")
    parser.add_argument("--active-arms", nargs="+", choices=("left", "right"),
                        help="active arms and concatenation order, overriding hardware config")
    parser.add_argument("--backend", choices=("xarm", "mock"), default="xarm")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate checkpoint and config without hardware")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--enable-command", action="store_true", help="enable xArm position servo commands")
    parser.add_argument("--single-step", action="store_true", help="start paused; s executes one full action chunk")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = JointDeployment.load(args.config, checkpoint=args.checkpoint,
                                  device=args.device, active_arms=args.active_arms)
    # Torch thread pools can otherwise starve the 100 Hz I/O loop on CPU.
    import torch

    torch.set_num_threads(int(config.policy.get("torch_threads", 1)))
    policy, mapping = load_joint_policy(config)
    report = policy_report(config, policy, mapping)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.check:
        policy.close()
        return report
    from inference.joint_runtime import JointInferenceRuntime
    from nero_collection.keyboard import TerminalKeys

    runtime = JointInferenceRuntime(config, policy, mapping, backend=args.backend,
                                    command_enabled=args.enable_command)
    print(f"Starting {config.policy['type'].upper()} xArm inference "
          f"({'command enabled' if args.enable_command else 'read-only'}). "
          "c: continue, p: pause, s: one action chunk, q/Ctrl-C: stop and exit.", flush=True)
    try:
        with TerminalKeys() as keys:
            runtime.run(args.duration, read_key=keys.read_key, single_step=args.single_step)
    except KeyboardInterrupt:
        print("\nInference interrupted; motion stopped.", flush=True)
    print(f"Stopped: predictions={runtime.predictions}, command_ticks={runtime.commands}.", flush=True)
    return report


if __name__ == "__main__":
    main()
