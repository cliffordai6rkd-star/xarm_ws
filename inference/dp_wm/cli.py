"""Live DP/WM inference with explicitly enabled position output or shadow evaluation."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from inference.dp_wm.config import load_config
from inference.dp_wm.core import AsyncDPWM, DP_PLAN_STEPS, WM_ACTION_STEPS
from nero_collection.time_utils import now_us


def startup_hold_tick(pipeline, sender, source, control_layer, *, primed):
    """Send real bootstrap commands before enabling the committed policy clock."""
    state = source.control_state()
    holding = not primed or pipeline.bootstrap_hold_active(now_us())
    if not holding:
        return None, primed, False
    event = sender.hold(state) if primed else sender.start_hold(state)
    if event is not None and event.get('sent'):
        if not primed:
            pipeline.prime_hold(event['first_q_command'], event['sent_at_us'])
            source.observation_after_us = event['sent_at_us']
            primed = True
        # The actual sent bootstrap target is constant. Use it as the initial
        # command history when the controller takes over, not measured drift.
        event.update(execution='active', actual_mode='startup_hold',
                     command_v_plan=np.zeros((2,7)), command_a_plan=np.zeros((2,7)))
        control_layer.mark_sent(event)
        control_layer.counts['valid'] += 1
        control_layer.counts['startup_hold'] += 1
    return event, primed, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='CPU checkpoint metadata only; no model/hardware')
    mode.add_argument('--run', action='store_true', help='live shadow inference')
    parser.add_argument('--enable-command', action='store_true', help='authorize reset and configured active position output')
    parser.add_argument('--mock', action='store_true', help='synthetic models and observations, no hardware')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--auto-start', action='store_true')
    args = parser.parse_args(argv)
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive and finite')
    if not args.check:
        args.run = True
    config = load_config(args.config)
    is_pi0 = config.get('pipeline') in ('pi0_contact_wm', 'pi0_latent_wm')
    plan_steps = config['pi0']['action_horizon'] if is_pi0 else DP_PLAN_STEPS
    control_options = config.get('controller', {})
    control_enabled = control_options.get('enabled', False)
    active_output_requested = control_enabled and control_options['execution'] == 'active'
    if args.run and active_output_requested and (not args.enable_command or args.mock):
        parser.error('active output requires --enable-command and live inputs; use shadow for --mock')
    if control_enabled and control_options['execution'] == 'offline' and args.run and not args.mock:
        parser.error('offline does not connect hardware; use the replay tool or --mock, or select shadow for live read-only inputs')
    refresh_steps = config['pi0']['consume_steps'] if is_pi0 else config['policy'].get('refresh_steps', 1)
    metadata = None
    if not args.mock:
        from inference.dp_wm.models import inspect_checkpoints
        metadata = inspect_checkpoints(config)
    elif config['wm']['action_sampling'] == 'checkpoint':
        # Mock has no training metadata; this is a synthetic timing contract.
        config['wm']['action_sampling'] = 'recorded_tokens' if is_pi0 else 'anchor_grid'
    wm_identity = ({} if metadata is None else {
        'wm_model_version': metadata['wm_model_version'],
        'wm_action_frame': metadata['wm_action_frame'],
        'wm_joint_action_adapter': config['wm'].get('joint_action_adapter', 'checkpoint_fk'),
        'wm_checkpoint': str(config['wm']['checkpoint'])})
    if args.check:
        report = {'status': 'checked', 'mock': args.mock, 'hardware_connected': False, 'model_loaded': False,
                  'policy_type': config['policy']['type'], 'policy_plan_steps': plan_steps,
                  'dp_plan_steps': plan_steps, 'wm_action_steps': WM_ACTION_STEPS, 'policy_target_hz': 25/refresh_steps, 'policy_action_hz': 25, 'dp_target_hz': 25/refresh_steps,
                  'policy_refresh_steps': refresh_steps,
                  'dp_request_mode': ('after_window_exhaustion' if not is_pi0 and config['policy'].get('wait_for_window_exhaustion', False)
                                      else 'phase_locked'),
                  'policy_open_loop': is_pi0 and config['pi0']['open_loop'],
                  'wm_target_hz': (None if config['wm']['scheduling'] == 'continuous'
                                   else 100/config['wm']['execute_steps']),
                  'wm_scheduling': config['wm']['scheduling'],
                  'wm_action_sampling': config['wm']['action_sampling'],
                  'wm_inference_budget_ms': config['wm']['inference_budget_ms'],
                  'dp_checkpoint': str(config['pi0']['local_checkpoint'] if is_pi0 else config['policy']['checkpoint']),
                  'wm_checkpoint': str(config['wm']['checkpoint']),
                  'command_source': config['observation']['command_source']}
        report['visualization'] = config['visualization']
        if is_pi0:
            report['pi0_checkpoint'] = report.pop('dp_checkpoint')
            report.pop('dp_plan_steps')
            report.pop('dp_target_hz')
            report.pop('dp_request_mode')
        report['controller'] = ({'mode':control_options['mode'], 'execution':control_options['execution'],
                                'target_hz':control_options.get('command_hz', 100),
                                'prediction_step_s':control_options['dt'],
                                'minimum_state_remaining_ms':control_options.get('minimum_state_remaining_ms', 0.),
                                'forecast_transition':control_options.get('forecast_transition', {'enabled': False})}
                               if control_enabled else 'disabled')
        report.update(wm_identity)
        if metadata is not None:
            report['alignment'] = metadata.get('alignment')
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return report
    if args.mock:
        from inference.dp_wm.mock import MockPolicy, MockWM, MockSource
        policy, wm, source = MockPolicy(), MockWM(), MockSource()
        if is_pi0:
            from inference.policies.pi0_joint import Pi0JointPolicy, MockPiConnection
            policy = Pi0JointPolicy({**config['pi0'], 'local': False}, client_factory=MockPiConnection)
            policy.allow_mock_server = True
            source = MockSource(image_keys=policy.image_keys)

    else:
        from inference.dp_wm.models import load_models
        from inference.dp_wm.live import LiveSource
        # Validate source configuration before allocating either model.
        source = LiveSource(config, metadata, enable_command=args.enable_command)
        print('Loading policy/WM and warming models (including CUDA graph capture)...', flush=True)
        policy, wm = load_models(config, metadata)
    options = config['wm']
    pipeline = AsyncDPWM(policy, wm, execute_steps=options['execute_steps'],
                         margin_ms=options['margin_ms'], initial_latency_ms=options['initial_latency_ms'],
                         aligned_actions=(config['pipeline'] in ('dp_contact_wm', 'pi0_contact_wm', 'pi0_latent_wm')
                                          or config['wm'].get('model_version') == 'latent_carswm_lstm_v2'),
                         plan_steps=plan_steps, policy_refresh_steps=refresh_steps,
                         policy_open_loop=is_pi0 and config['pi0']['open_loop'],
                         policy_prefetch_margin_ms=config['pi0']['prefetch_margin_ms'] if is_pi0 else 100.,
                         wait_for_window_exhaustion=False if is_pi0 else config['policy'].get('wait_for_window_exhaustion', False),
                         scheduling=options['scheduling'], inference_budget_ms=options['inference_budget_ms'],
                         action_sampling=options['action_sampling'])
    control_layer = None
    if control_enabled:
        from inference.core import WmControlLayer

        control_layer = WmControlLayer(control_options, options['urdf_path'], allow_active=True)
    output = config['logging']['output_dir'] / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(config,default=str,ensure_ascii=False,indent=2))
    active = args.auto_start
    reset_motion = getattr(source, 'reset_motion', None)
    sender = None
    host_commands = not args.mock and config['observation']['command_source'] == 'host_command'
    startup_primed = False
    resume_after_reset = False
    last_output_stop = None
    from inference.dp_wm.event_writer import EventWriter
    log = EventWriter(output, mock=args.mock, command_source=config['observation']['command_source'],
                      asynchronous=config['logging'].get('async_write', True))
    timings = defaultdict(lambda: deque(maxlen=1000))
    from inference.wm_visualization import WmVisualization
    visualization = WmVisualization(config, source, mock=args.mock)
    from nero_collection.keyboard import TerminalKeys
    mode_label = 'RESET ENABLED; predictions shadow' if reset_motion else 'READ ONLY'
    if active_output_requested:
        mode_label = 'ACTIVE POSITION OUTPUT'
    print(f'{config["pipeline"]}, {mode_label}, mock={args.mock}. s: start, p: pause, i: reset/pause, q: exit. Output: {output}', flush=True)
    if metadata is not None:
        print(f'WM: {metadata["wm_model_version"]}; action frame={metadata["wm_action_frame"]}; '
              f'joint adapter={config["wm"].get("joint_action_adapter", "checkpoint_fk")}', flush=True)
        for issue in metadata.get('alignment', {}).get('issues', []):
            print(f'Training/deployment: {issue}', flush=True)
    if control_enabled:
        print(f'Controller: {control_options["mode"]}; execution={control_options["execution"]}', flush=True)
        print(f'Position command rate={control_options.get("command_hz", 100)} Hz; '
              f'WM physical grid={1/control_options["dt"]:g} Hz', flush=True)
        if not is_pi0 and config['policy'].get('wait_for_window_exhaustion', False):
            hold=control_options.get('gap_hold', {})
            print(f'DP request interval={refresh_steps * 40} ms from observation timestamp; '
                  f'bounded position hold={hold.get("maximum_ms", "disabled")} ms', flush=True)
        if control_options['mode'] in ('mit_position','tau_only'):
            controller=control_layer.controller
            mit=control_options.get('mit') or {}
            print(f'MIT feedforward={controller.feedforward_source}; WM weight={controller.wm_tau_weight}; '
                  f'integration={controller.integration_base}; jerk={control_options["limits"].get("max_jerk_rad_s3", "disabled")} rad/s^3', flush=True)
            print(f'MIT feedback_space={controller.feedback_space}; velocity_source={controller.velocity_source}; '
                  f'nominal_acceleration={controller.nominal_acceleration_source}; '
                  f'WM residual acceleration limit={mit.get("wm_residual_acceleration_limit_rad_s2", "disabled")} rad/s^2', flush=True)
            if control_options['mode']=='tau_only':
                print('tau_only: WM torque weight fixed at 1; mit.wm_tau_weight ignored; WM q/dq/ddq tracking disabled.', flush=True)
        elif control_options['mode']=='q_only':
            print(f'q_only output shaping={control_options["limits"].get("q_only_smoothing", "legacy")}',flush=True)
            transition = control_options.get('forecast_transition', {})
            if transition.get('enabled', False):
                print(f'WM reference handover={transition["duration_ms"]} ms; '
                      'same-time q/slope continuity, bounded by valid overlap', flush=True)
    if config.get('execution', {}).get('reset_on_start') and reset_motion is None:
        print('Startup reset skipped: requires --enable-command and live hardware.', flush=True)
    if not args.mock and config['observation']['command_source'] == 'controller_reference':
        print('delta_q source: controller reference minus measured q; equivalence to training host q_cmd is unverified.', flush=True)
    try:
        if not args.mock:
            from inference.dp_wm.models import warmup_inputs
            steps = config.get('execution', {}).get('warmup_steps', 3)
            print(f'Warming policy/WM worker threads: {steps} rounds before connecting hardware...', flush=True)
            cache = getattr(policy,'feature_cache',None)
            if cache is not None:
                cache.enabled=False
            try:
                pipeline.warmup(*warmup_inputs(metadata['images']), steps=steps)
            finally:
                if cache is not None:
                    cache.enabled=True
            print('Warmup complete; synthetic outputs discarded. Connecting hardware...', flush=True)
        source.start()
        visualization.start()
        if active_output_requested:
            from inference.dp_wm.output import ActiveOutput
            sender = ActiveOutput(source.arm, maximum_state_age_ms=control_options['maximum_state_age_ms'],
                                  command_hz=control_options.get('command_hz', 100),
                                  minimum_state_remaining_ms=control_options.get('minimum_state_remaining_ms', 0.))
        if reset_motion is not None and config.get('execution', {}).get('reset_on_start', False):
            resume_after_reset, active = active, False
            reset_motion.begin()
            print('Startup reset in progress; p/q interrupts.', flush=True)
        elif active and sender is not None:
            sender.start()
            source.observation_after_us = now_us()
        started = last_report = time.monotonic()
        with TerminalKeys() as keys:
            while args.duration is None or time.monotonic()-started < args.duration:
                loop_started = time.perf_counter()
                log.check_error()
                key = keys.read_key(0.)
                if key in ('s','p','i','q','\x03'):
                    log.write({'kind':'interaction','key':key,'timestamp_us':now_us()})
                if key in ('q', '\x03'):
                    break
                if key in ('p', 'i'):
                    if sender is not None:
                        sender.stop()
                    active = False
                    startup_primed = False
                    resume_after_reset = False
                    pipeline.reset()
                    visualization.reset()
                    if control_layer is not None:
                        control_layer.reset()
                    if reset_motion is not None:
                        reset_motion.stop()
                        source.observation_after_us = now_us()
                        if key == 'i':
                            reset_motion.begin()
                            print('Reset in progress; p/q interrupts.', flush=True)
                elif key == 's' and not (reset_motion and reset_motion.running):
                    if not active:
                        if sender is not None:
                            sender.start()
                            source.observation_after_us = now_us()
                        pipeline.reset()
                        visualization.reset()
                        startup_primed = False
                        if control_layer is not None:
                            control_layer.reset()
                    active = True
                stage_started = time.perf_counter()
                observations, history = source.poll()
                timings['source_poll_ms'].append((time.perf_counter()-stage_started)*1000)
                if reset_motion is not None and reset_motion.running:
                    if reset_motion.poll(source.control_state()):
                        source.observation_after_us = now_us()
                        observations, history = (), None
                        active = resume_after_reset
                        startup_primed = False
                        if active and sender is not None:
                            sender.start()
                            source.observation_after_us = now_us()
                        print('Reset complete; waiting for fresh observations.' if active
                              else 'Reset complete; press s to start inference.', flush=True)
                stage_started = time.perf_counter()
                startup_event = None
                holding_startup = False
                if active and host_commands and sender.enabled:
                    was_primed = startup_primed
                    startup_event, startup_primed, holding_startup = startup_hold_tick(
                        pipeline, sender, source, control_layer, primed=startup_primed)
                    if startup_primed and not was_primed:
                        observations, history = (), None  # All rows must follow this actual successful hold.
                events = pipeline.poll(now_us(), observations, history,
                                       active=active and (not host_commands or startup_primed))
                timings['pipeline_poll_ms'].append((time.perf_counter()-stage_started)*1000)
                stage_started = time.perf_counter()
                control_event = None
                if control_layer is not None:
                    for event in events:
                        if event['kind'] == 'wm':
                            control_layer.update_forecast(event)
                    if active:
                        if holding_startup:
                            control_event = startup_event
                        elif host_commands and sender.hold_q is not None and control_layer.forecast is None:
                            sender.stop('First WM forecast unavailable at committed policy start; press s to rearm')
                        else:
                            control_event = control_layer.poll(now_us(), source.control_state())
                            if sender is not None and sender.send(control_event):
                                control_layer.mark_sent(control_event)
                                sender.end_hold()
                        if control_event is not None and config['logging'].get('record_controller', True):
                            events.append(control_event)
                timings['controller_ms'].append((time.perf_counter()-stage_started)*1000)
                if sender is not None:
                    sender.check_error()
                    if active and not sender.enabled:
                        last_output_stop={'kind':'output_stop','timestamp_us':now_us(),'reason':sender.reason,
                            'controller_reason':None if control_event is None else control_event.get('reason'),
                            'forecast_age_ms':None if control_event is None else control_event.get('forecast_age_ms'),
                            'compute_ms':None if control_event is None else control_event.get('compute_ms'),
                            'send_state_age_ms':None if control_event is None else control_event.get('send_state_age_ms'),
                            'remaining_state_budget_ms':None if control_event is None else control_event.get('remaining_state_budget_ms'),
                            'selected_target_remaining_ms':None if control_event is None else control_event.get('selected_target_remaining_ms'),
                            'feedback_lock_wait_ms':None if control_event is None else control_event.get('feedback_lock_wait_ms'),
                            'send_phase':None if control_event is None else control_event.get('send_phase'),
                            'policy_observation_status':getattr(source,'observation_status',None),
                            'policy_wait_reason':pipeline.wait_reason}
                        events.append(last_output_stop.copy())
                        active = False
                        startup_primed = False
                        pipeline.reset()
                        visualization.reset()
                        control_layer.reset()
                        print(f'Active output stopped: {sender.reason}; '
                              f'state age at send={last_output_stop["send_state_age_ms"]} ms; '
                              f'feedback lock wait={last_output_stop["feedback_lock_wait_ms"]} ms; '
                              f'policy input: {last_output_stop["policy_observation_status"]}; '
                              f'WM: {last_output_stop["policy_wait_reason"]}', flush=True)
                stage_started = time.perf_counter()
                visualization.update(source.control_state(), events)
                for event in events:
                    log.write(event)
                timings['log_enqueue_ms'].append((time.perf_counter()-stage_started)*1000)
                timings['loop_ms'].append((time.perf_counter()-loop_started)*1000)
                wall = time.monotonic()
                if wall-last_report >= 1.:
                    report = pipeline.report()
                    if control_layer is not None:
                        report['controller'] = control_layer.report()
                    report.update(wm_identity)
                    report.update(source_status=source.status, mock=args.mock,
                                  policy_observation_status=getattr(source,'observation_status',None),
                                  active=active, output_status=None if sender is None else sender.reason,
                                  elapsed_s=wall-started,
                                  achieved_wm_hz=pipeline.counts['wm_predictions']/max(wall-started,1e-6),
                                  achieved_dp_hz=pipeline.counts['dp_plans']/max(wall-started,1e-6))
                    report['timing_p95_ms'] = {key: float(np.percentile(values,95)) for key,values in timings.items()}
                    report['logging'] = log.report()
                    if getattr(policy,'feature_cache',None) is not None:
                        report['dp_feature_cache'] = policy.feature_cache.report()
                    if config['logging'].get('console_status', False):
                        print(json.dumps(report, ensure_ascii=False), flush=True)
                    log.write({'kind': 'status', **report})
                    last_report = wall
                time.sleep(.001)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            try:
                if sender is not None:
                    sender.close()
            finally:
                try:
                    source.close()
                finally:
                    visualization.close()
        finally:
            try:
                pipeline.close()
                if is_pi0:
                    policy.close()
                report = pipeline.report()
                report.update(wm_identity)
                if control_layer is not None:
                    report['controller'] = control_layer.report()
                report['timing_p95_ms'] = {key: float(np.percentile(values,95)) for key,values in timings.items()}
                report['last_output_stop'] = last_output_stop
                (output/'summary.json').write_text(json.dumps(report, indent=2))
            finally:
                log.close()
    return report
