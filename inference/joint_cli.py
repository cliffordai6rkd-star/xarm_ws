"""DP/ACT -> checkpoint-selected RGB observations -> xArm joint servo CLI."""
from __future__ import annotations

import argparse
import json
import logging

from inference.joint_config import JointDeployment, load_joint_policy, policy_report
from inference.policies.errors import JointActionContractError
from inference.interactive import INTERACTION_HELP, add_interaction_arguments, run_interactive


def main(argv=None, *, run_loop=run_interactive):
    parser = argparse.ArgumentParser(description="DP / ACT / remote pi0 xArm joint inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="override policy checkpoint file or training output directory")
    parser.add_argument("--device", help="override policy device, e.g. cpu or cuda:0")
    parser.add_argument("--active-arms", nargs="+", choices=("left", "right"),
                        help="active arms and concatenation order, overriding hardware config")
    parser.add_argument("--backend", choices=("xarm", "mock"), default="xarm")
    parser.add_argument("--mock-policy", action="store_true",
                        help="pi0 only: use an offline fake server; requires --backend mock")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate checkpoint and config without hardware")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--check-server", action="store_true",
                        help="also connect to pi0 and verify server metadata without hardware")
    parser.add_argument("--enable-command", action="store_true", help="enable xArm position servo commands")
    add_interaction_arguments(parser)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = JointDeployment.load(args.config, checkpoint=args.checkpoint,
                                  device=args.device, active_arms=args.active_arms)
    from inference.logging_utils import quiet_hardware_output

    quiet = getattr(config, "execution", {}).get("quiet", False)
    with quiet_hardware_output(quiet):
        return _run(args, parser, config, run_loop=run_loop, quiet=quiet)


def _run(args, parser, config, *, run_loop, quiet):
    if args.mock_policy and (config.policy["type"] != "pi0" or args.backend != "mock"):
        parser.error("--mock-policy requires policy.type pi0 and --backend mock")
    # Torch thread pools can otherwise starve the 100 Hz I/O loop on CPU.
    import torch

    torch.set_num_threads(int(config.policy.get("torch_threads", 1)))
    try:
        policy, mapping = load_joint_policy(config)
        if config.policy["type"] == "pi0":
            policy.allow_mock_server = args.backend == "mock"
        if args.mock_policy:
            from inference.policies.pi0_joint import MockPiConnection

            if policy.action_type == 'eepose':
                import numpy as np

                side = config.active_arms[0]
                action = policy.pose_adapter.forward_pose(np.asarray(config.endpoints[side].rest_q))
                policy.client_factory = lambda settings: MockPiConnection(settings, action=action)
            else:
                policy.client_factory = MockPiConnection
        if args.check_server:
            if config.policy["type"] != "pi0" or not args.check:
                parser.error("--check-server requires pi0 and --check")
            policy.verify_server()
    except JointActionContractError as exc:
        parser.error(str(exc))
    report = policy_report(config, policy, mapping)
    if args.mock_policy:
        report["mock_policy"] = True
    if args.check or not quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.check:
        policy.close()
        return report
    from inference.joint_runtime import JointInferenceRuntime

    runtime = JointInferenceRuntime(config, policy, mapping, backend=args.backend,
                                    command_enabled=args.enable_command)
    print(f"Starting {config.policy['type'].upper()} xArm inference "
          f"({'command enabled' if args.enable_command else 'read-only'}). "
          + INTERACTION_HELP, flush=True)
    try:
        run_loop(runtime, args)
    except JointActionContractError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("\nInference interrupted; motion stopped.", flush=True)
    print(f"Stopped: predictions={runtime.predictions}, command_ticks={runtime.commands}.", flush=True)
    return report


if __name__ == "__main__":
    main()
