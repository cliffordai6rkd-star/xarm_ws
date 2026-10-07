"""PI0-WM configuration branch of the common inference entry point."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
from pathlib import Path

from inference.interactive import INTERACTION_HELP, add_interaction_arguments, run_interactive
from inference.pi0_wm.config import load_config


class InteractiveRuntime:
    """Adapt the step-limited PI0 runtime to the shared duration-based loop."""

    def __init__(self, runtime, steps):
        self.runtime, self.steps = runtime, steps

    def run(self, duration, **interaction):
        return self.runtime.run(self.steps, duration_s=duration, **interaction)


def deployment_issues(config, *, mock=False, mock_wm=False):
    issues = []
    if not mock:
        if not config['pi0']['interface']['training_config']:
            issues.append('pi0.interface.training_config is required')
        for camera in config['cameras']:
            if camera.get('backend') == 'realsense' and (
                not camera.get('serial_number') or 'REPLACE_' in str(camera['serial_number'])
            ):
                issues.append(f"camera {camera['name']} requires its actual serial_number")
        modules = ['openpi_client', 'websockets', 'msgpack']
        if config['hardware']['backend'] == 'xarm_zmq':
            modules.append('zmq')
        for module in modules:
            if importlib.util.find_spec(module) is None:
                issues.append(f'missing Python module: {module}')
    if not (mock or mock_wm):
        for key in ('pinn_root', 'checkpoint'):
            if not config['wm'][key] or not Path(config['wm'][key]).exists():
                issues.append(f"wm.{key} does not exist: {config['wm'][key]}")
    return issues


def main(argv=None, *, run_loop=run_interactive):
    import sys
    import yaml
    arguments = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config', default='inference/configs/pi0_wm_xarm.yaml')
    known, _ = probe.parse_known_args(arguments)
    document = yaml.safe_load(Path(known.config).read_text()) or {}
    if document.get('pipeline') in ('pi0_contact_wm', 'pi0_latent_wm'):
        if '--config' not in arguments:
            arguments.extend(['--config', known.config])
        if document.get('wm', {}).get('enabled') is False:
            from inference.joint_cli import main as joint_main
            return joint_main(arguments, run_loop=run_loop)
        from inference.dp_wm.cli import main as contact_main
        return contact_main(arguments)
    parser = argparse.ArgumentParser(description='PI0-WM xArm position inference')
    parser.add_argument('--config', default='inference/configs/pi0_wm_xarm.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--check', action='store_true', help='check config/files/dependencies without hardware')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--mock', action='store_true', help='simulate models, arm and cameras')
    parser.add_argument('--mock-wm', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--enable-command', '--enable-commands', dest='enable_commands', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='read-only mode (the default)')
    parser.add_argument('--log-level', default='INFO')
    add_interaction_arguments(parser)
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        parser.error('--steps must be positive')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive')
    if args.enable_commands and (args.dry_run or args.mock or args.mock_wm):
        parser.error('--enable-command cannot be combined with --dry-run/--mock/--mock-wm')
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        config = load_config(args.config)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    issues = deployment_issues(config, mock=args.mock, mock_wm=args.mock_wm)
    if issues:
        parser.error('; '.join(issues))
    if args.headless:
        config['mujoco']['headless'] = True
    if args.check:
        report = {'status': 'config_checked', 'hardware_connected': False,
                  'mock': args.mock, 'model_contract_verified': False,
                  'config': str(Path(args.config).resolve())}
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return report
    from inference.pi0_wm.runtime import Runtime

    runtime = Runtime(config, enable_commands=args.enable_commands,
                      mock=args.mock, mock_wm=args.mock_wm)
    print('Starting PI0-WM inference. ' + INTERACTION_HELP, flush=True)
    try:
        result = run_loop(InteractiveRuntime(runtime, args.steps), args)
    except KeyboardInterrupt:
        print('\nPI0-WM inference interrupted.', flush=True)
        return
    print(result, flush=True)
    return result
