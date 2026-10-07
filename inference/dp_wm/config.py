from pathlib import Path
import math
import yaml


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    execution = config.setdefault('execution', {})
    if not isinstance(execution, dict) or type(execution.get('reset_on_start', False)) is not bool:
        raise ValueError('execution.reset_on_start must be a boolean')
    warmup_steps = execution.setdefault('warmup_steps', 3)
    if type(warmup_steps) is not int or warmup_steps < 1:
        raise ValueError('execution.warmup_steps must be a positive integer')
    initial_q = execution.get('initial_q', {})
    if not isinstance(initial_q, dict) or set(initial_q) - {'left', 'right'}:
        raise ValueError('execution.initial_q must map left/right to joint angles')
    for side, values in initial_q.items():
        if (not isinstance(values, list) or len(values) != 7
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise ValueError(f'execution.initial_q.{side} must contain seven finite joint angles')
    is_pi0 = config.get('pipeline') in ('pi0_contact_wm', 'pi0_latent_wm')
    if config.get('pipeline') not in ('dp_latent_wm', 'dp_contact_wm', 'pi0_contact_wm', 'pi0_latent_wm') or config['policy']['type'] != ('pi0' if is_pi0 else 'dp'):
        raise ValueError('Pipeline and policy.type do not match')
    policy_paths = ('client_source_path',) if is_pi0 else ('checkpoint', 'source_path')
    paths = [('policy', policy_paths),
                          ('wm', ('checkpoint', 'pinn_root', 'urdf_path')),
                          ('observation', ('hardware_config',)), ('logging', ('output_dir',))]
    if is_pi0:
        paths.append(('pi0', ('local_source_path', 'local_checkpoint')))
    for section, keys in paths:
        for key in keys:
            value = Path(config[section][key]).expanduser()
            config[section][key] = (value if value.is_absolute() else path.parent / value).resolve()
            if section != 'logging' and not (section == 'pi0' and not config['pi0'].get('local', False)) and not config[section][key].exists():
                raise FileNotFoundError(config[section][key])
    if config['wm'].get('training_urdf_path') is not None:
        value = Path(config['wm']['training_urdf_path']).expanduser()
        config['wm']['training_urdf_path'] = (value if value.is_absolute() else path.parent / value).resolve()
        if not config['wm']['training_urdf_path'].exists():
            raise FileNotFoundError(config['wm']['training_urdf_path'])
    policy_integers = [] if is_pi0 else [('policy', 'num_inference_steps'), ('policy', 'torch_threads')]
    if not is_pi0:
        refresh_steps = config['policy'].get('refresh_steps', 1)
        if type(refresh_steps) is not int or not 1 <= refresh_steps <= 15:
            raise ValueError('policy.refresh_steps must be an integer in [1,15]')
        wait_for_window = config['policy'].get('wait_for_window_exhaustion', False)
        if type(wait_for_window) is not bool or (wait_for_window and refresh_steps != 8):
            raise ValueError('policy.wait_for_window_exhaustion requires refresh_steps: 8')
    if is_pi0:
        if config['pi0']['action_hz'] != 25 or config['pi0'].get('action_type', 'joint') != 'joint':
            raise ValueError('WM bridge requires pi0 joint actions at 25 Hz')
        horizon = config['pi0']['action_horizon']
        if type(horizon) is not int or horizon <= 8:
            raise ValueError('pi0.action_horizon must exceed eight to cover the current token plus eight future WM actions; keep the trained horizon (50 for this checkpoint)')
        steps = config['pi0']['consume_steps']
        if type(steps) is not int or not 1 <= steps <= horizon:
            raise ValueError('pi0.consume_steps must be a positive integer no greater than action_horizon')
        open_loop = config['pi0'].setdefault('open_loop', False)
        if type(open_loop) is not bool:
            raise ValueError('pi0.open_loop must be a boolean')
        if open_loop and steps != horizon:
            raise ValueError('pi0.open_loop requires consume_steps equal to action_horizon')
        prefetch = config['pi0'].setdefault('prefetch_margin_ms', 100.)
        if type(prefetch) not in (int, float) or not math.isfinite(prefetch) or prefetch < 0:
            raise ValueError('pi0.prefetch_margin_ms must be finite and nonnegative')
        if config['wm'].get('enabled') is not True:
            raise ValueError(f"{config['pipeline']} requires wm.enabled=true")
    for section, key in policy_integers + [
                          ('wm', 'execute_steps'), ('wm', 'num_samples'), ('wm', 'flow_steps')]:
        value = config[section][key]
        if type(value) is not int or value < 1:
            raise ValueError(f'{section}.{key} must be a positive integer')
    if config['wm']['execute_steps'] > 32:
        raise ValueError('wm.execute_steps exceeds the predicted trajectory')
    scheduling = config['wm'].setdefault('scheduling', 'periodic')
    if scheduling not in ('periodic', 'continuous'):
        raise ValueError('wm.scheduling must be periodic or continuous')
    budget = config['wm'].setdefault('inference_budget_ms', 100.)
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError('wm.inference_budget_ms must be positive and finite')
    sampling = config['wm'].setdefault('action_sampling', 'checkpoint')
    if sampling not in ('checkpoint', 'anchor_grid', 'recorded_tokens'):
        raise ValueError('wm.action_sampling must be checkpoint, anchor_grid or recorded_tokens')
    for section, keys in [('wm', ('margin_ms', 'initial_latency_ms')),
                          ('observation', ('maximum_age_ms', 'maximum_camera_skew_ms', 'maximum_pair_skew_ms'))]:
        for key in keys:
            value = float(config[section][key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {section}.{key}')
    if config['wm']['solver'] not in ('heun', 'euler') or config['wm']['precision'] not in ('float32', 'bfloat16', 'float16'):
        raise ValueError('Invalid WM solver or precision')
    expected_versions = ({'carswm_v9'} if config['pipeline'] in ('dp_contact_wm', 'pi0_contact_wm')
                         else {'latent_carswm_lstm_v1', 'latent_carswm_lstm_v2'})
    if config['wm'].get('model_version') is not None and config['wm']['model_version'] not in expected_versions:
        raise ValueError('wm.model_version does not match the selected WM pipeline')
    if config['wm'].get('joint_action_adapter', 'checkpoint_fk') not in ('checkpoint_fk', 'joint_fk_link7_approx'):
        raise ValueError('wm.joint_action_adapter must be checkpoint_fk or joint_fk_link7_approx')
    if type(config['wm'].get('cuda_graph', False)) is not bool:
        raise ValueError('wm.cuda_graph must be a boolean')
    if type(config['wm'].get('cache_condition_kv', False)) is not bool:
        raise ValueError('wm.cache_condition_kv must be a boolean')
    if config['policy'].get('precision', 'float32') not in ('float32', 'bfloat16', 'float16'):
        raise ValueError('Invalid policy.precision')
    if type(config['policy'].get('cache_observation_features', False)) is not bool:
        raise ValueError('policy.cache_observation_features must be a boolean')
    if type(config['logging'].get('async_write', True)) is not bool:
        raise ValueError('logging.async_write must be a boolean')
    console_status = config['logging'].setdefault('console_status', False)
    if type(console_status) is not bool:
        raise ValueError('logging.console_status must be a boolean')
    if config['observation']['arm'] not in ('left', 'right'):
        raise ValueError('Select one observation arm')
    if config['observation']['command_source'] not in ('controller_reference', 'external_file', 'host_command'):
        raise ValueError('Unknown command_source')
    if config['observation']['command_source'] == 'host_command':
        controller = config.get('controller', {})
        if not controller.get('enabled') or controller.get('execution') != 'active' or not is_pi0 or not config['pi0']['open_loop']:
            raise ValueError('host_command requires active control and a committed open-loop pi0 stream')
    if config['observation']['command_source'] == 'external_file':
        value = Path(config['observation']['command_file']).expanduser()
        config['observation']['command_file'] = (value if value.is_absolute() else path.parent / value).resolve()
    controller = config.get('controller')
    if controller is not None:
        if not isinstance(controller, dict) or type(controller.get('enabled', False)) is not bool:
            raise ValueError('controller.enabled must be a boolean')
        if controller.get('enabled'):
            gap_hold = controller.get('gap_hold', {})
            if (not is_pi0 and config['policy'].get('wait_for_window_exhaustion', False)
                    and controller.get('execution') == 'active'
                    and (not isinstance(gap_hold, dict) or not gap_hold.get('enabled', False))):
                raise ValueError('Active DP 320 ms windows require controller.gap_hold.enabled=true')
            from inference.core.wm_control import validate_settings

            hardware = yaml.safe_load(config['observation']['hardware_config'].read_text())
            controller.setdefault('limits', {key: hardware['control'][key] for key in (
                'max_velocity_rad_s', 'max_acceleration_rad_s2', 'max_step_rad', 'max_tracking_error_rad')})
            validate_settings(controller, allow_active=True)
    from inference.wm_visualization import configure_visualization
    configure_visualization(config, path.parent)
    return config
