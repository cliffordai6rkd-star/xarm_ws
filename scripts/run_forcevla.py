#!/usr/bin/env python3
"""One config: local xArm ForceVLA client, optionally launch the existing server over SSH."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.forcevla_config import ForceVLADeployment

log = logging.getLogger("forcevla")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "inference/configs/forcevla_xarm.yaml"


def ssh_command(config):
    options = config.server.get("ssh", {})
    host = options.get("host") or config.server["host"]
    target = f"{options['user']}@{host}" if options.get("user") else host
    args = ["ssh", "-tt", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-p", str(options.get("port", 22))]
    if options.get("identity_file"):
        args.extend(["-i", str(Path(options["identity_file"]).expanduser())])
    return [*args, target, "bash -lc " + shlex.quote(config.server_command())]


def connect_client(config, process=None):
    from inference.forcevla_transport import ForceVLAClient

    server = config.server
    deadline = time.monotonic() + (server["startup_timeout_s"] if process else server["connect_timeout_s"])
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Remote server launch exited with code {process.returncode}; see SSH output")
        client = ForceVLAClient(server["host"], server["port"],
                               connect_timeout_s=min(server["connect_timeout_s"], max(.1, deadline-time.monotonic())),
                               request_timeout_s=server["request_timeout_s"])
        try:
            client.connect()
            return client
        except (OSError, TimeoutError):
            client.close()
            if process is None or time.monotonic() >= deadline:
                raise
            time.sleep(min(1., max(0., deadline-time.monotonic())))


def main(argv=None):
    parser = argparse.ArgumentParser(description="xArm -> training-matched causal tau_ext + RGB -> ForceVLA over Tailscale")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate config only; no network/hardware")
    mode.add_argument("--check-server", action="store_true", help="one synthetic inference; no robot/cameras")
    mode.add_argument("--print-server-command", action="store_true", help="print server command; execute nothing")
    parser.add_argument("--start-server", action="store_true", help="start existing remote server over SSH")
    parser.add_argument("--mock", action="store_true", help="synthetic local arm/cameras; real torque processing and remote policy")
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--enable-command", action="store_true", help="override config and execute joint targets")
    command.add_argument("--dry-run", action="store_true", help="override config to read-only inference")
    parser.add_argument("--single-step", action="store_true", help="start paused; s executes one chunk")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.duration is not None and (not 0 < args.duration < float("inf")):
        parser.error("--duration must be positive and finite")
    if args.mock and args.enable_command:
        parser.error("--mock cannot send real robot commands")
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = ForceVLADeployment.load(args.config)
    if args.print_server_command:
        print(config.server_command())
        return 0
    enabled = (args.enable_command or config.execution.get("enable_command", False)) and not args.dry_run and not args.mock
    if args.check:
        print(json.dumps({"config": str(config.path), "server": f"ws://{config.server['host']}:{config.server['port']}",
                          "training_config": config.policy["training_config"], "arm": config.active_arms[0],
                          "robot_ip": next(iter(config.endpoints.values())).config_kwargs["robot_ip"],
                          "camera_map": config.observation["camera_map"],
                          "tau_ext": {"processing_record": str(config.tau_ext["processing_record"]),
                                      **config.tau_ext["processing_config"]},
                          "action": "absolute joint q[7], rad", "action_rate_hz": config.policy["action_rate_hz"],
                          "execute_steps": config.policy["execute_steps"], "command_enabled": enabled}, indent=2))
        return 0

    process = client = None
    try:
        if args.start_server or config.server.get("autostart", False):
            command_line = ssh_command(config)
            log.info("Launching existing remote server: %s", shlex.join(command_line))
            process = subprocess.Popen(command_line, stdin=subprocess.DEVNULL)
        client = connect_client(config, process)
        if args.check_server:
            import numpy as np

            h, w = config.policy["image_size"]
            result = client.infer({"state": np.zeros(7, np.float32), "tau_ext": np.zeros(7, np.float32),
                                   "prompt": config.policy["prompt"],
                                   **{key: np.zeros((h, w, 3), np.uint8) for key in config.observation["camera_map"]}})
            actions = np.asarray(result["actions"])
            if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid server action shape or values: {actions.shape}")
            print(json.dumps({"status": "ok", "actions_shape": list(actions.shape),
                              "metadata": client.metadata, "server_timing": result.get("server_timing", {})}, indent=2))
            return 0
        from inference.forcevla_runtime import ForceVLAInferenceRuntime, RemoteForceVLAPolicy
        from nero_collection.keyboard import TerminalKeys

        policy = RemoteForceVLAPolicy(config, client)
        runtime = ForceVLAInferenceRuntime(config, policy, command_enabled=enabled,
                                          backend="mock" if args.mock else "xarm")
        print(f"ForceVLA {'executing joint targets' if enabled else 'read-only inference'}; "
              "c: continue, p: pause, s: one chunk, q/Ctrl-C: stop.", flush=True)
        with TerminalKeys() as keys:
            runtime.run(args.duration, read_key=keys.read_key, single_step=args.single_step)
        print(f"Stopped: predictions={runtime.predictions}, command_ticks={runtime.commands}.")
        return 0
    except KeyboardInterrupt:
        log.info("Interrupted; stopping inference")
        return 130
    finally:
        if client is not None:
            client.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.)


if __name__ == "__main__":
    raise SystemExit(main())
