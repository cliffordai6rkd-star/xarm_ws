"""Checkpoint adapters. Models load only when the runtime is explicitly started."""
from __future__ import annotations

from pathlib import Path
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
import sys
import xml.etree.ElementTree as ET

import numpy as np


def fk_chain_signature(path, base_frame, target_frame):
    """Joint transforms defining the trained action frame; inertia/meshes are separate."""
    root = ET.parse(path).getroot()
    links = {node.get('name') for node in root.findall('link')}
    if not {base_frame, target_frame} <= links:
        raise ValueError('FK URDF is missing the trained base/target frame')
    parents = {}
    for joint in root.findall('joint'):
        child = joint.find('child').get('link')
        if child in parents:
            raise ValueError('FK URDF has duplicate parents')
        parents[child] = joint
    chain, seen = [], set()
    frame = target_frame
    while frame != base_frame:
        if frame in seen or frame not in parents:
            raise ValueError('FK URDF has a cycle or disconnected trained frame')
        seen.add(frame)
        joint = parents[frame]
        origin, axis = joint.find('origin'), joint.find('axis')
        xyz = tuple(float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split())
        rpy = tuple(float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split())
        direction = tuple(float(v) for v in (axis.get('xyz', '1 0 0') if axis is not None else '1 0 0').split())
        frame = joint.find('parent').get('link')
        chain.append((joint.get('name'), joint.get('type'), frame, xyz, rpy, direction))
    return tuple(reversed(chain))


def add_pinn_source(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    # PINN's train/ is a namespace package. A later DP train.py otherwise
    # wins normal import resolution, even with PINN first on sys.path.
    # Resolve only within PINN and freeze the namespace search path so it
    # cannot be redirected by subsequent sys.path changes.
    train_path = str(root / 'train')
    train = sys.modules.get('train')
    if train_path not in getattr(train, '__path__', ()):
        spec = PathFinder.find_spec('train', [str(root)])
        if spec is None or spec.submodule_search_locations is None:
            raise ImportError(f'PINN train package not found in {root}')
        if any(name.startswith('train.') for name in sys.modules):
            raise ImportError('A different train package is already loaded; load PINN in a fresh process')
        train = module_from_spec(spec)
        train.__path__ = list(spec.submodule_search_locations)
        sys.modules['train'] = train
        if spec.loader is not None:
            spec.loader.exec_module(train)
    package = sys.modules.get('model')
    if package is not None and hasattr(package, '__path__'):
        package.__path__ = [str(root / 'model'), *package.__path__]


def joint_action_frame(training, deployment):
    """Resolve the explicit joint-to-pose bridge for this checkpoint contract."""
    contract = training['action_contract']
    if any(contract.get(k) != v for k, v in {'type': 'absolute_ee_pose',
            'quaternion_order': 'xyzw', 'absolute_or_relative': 'absolute'}.items()):
        raise ValueError('WM action frame/representation mismatch')
    fk = training['dataloader'].get('action_fk')
    adapter = deployment.get('joint_action_adapter', 'checkpoint_fk')
    if fk is not None:
        if (adapter != 'checkpoint_fk' or contract.get('coordinate_frame') != 'link_base'
                or fk['base_frame'] != 'link_base' or fk['frame_name'] != 'link_eef'):
            raise ValueError('WM action frame/representation mismatch: checkpoint FK requires link_base -> link_eef')
        if fk['joint_names'] != [f'joint{i}' for i in range(1, 8)]:
            raise ValueError('WM joint order mismatch')
        # Preserve the trained base->action-frame kinematics while allowing a
        # fixed gripper's inertia/geometry beyond that action frame.
        original = (Path(deployment['pinn_root']) / fk['urdf_path']).resolve()
        override = deployment.get('training_urdf_path')
        if override is not None:
            override = Path(override).resolve()
            if not override.exists():
                raise FileNotFoundError(override)
            if original.exists() and fk_chain_signature(original, fk['base_frame'], fk['frame_name']) != fk_chain_signature(
                    override, fk['base_frame'], fk['frame_name']):
                raise ValueError('Explicit training URDF differs from checkpoint URDF')
            original = override
        if original.exists() and fk_chain_signature(original, fk['base_frame'], fk['frame_name']) != fk_chain_signature(
                deployment['urdf_path'], fk['base_frame'], fk['frame_name']):
            raise ValueError('Deployment FK chain differs from the training URDF')
        return 'link_eef'
    if (adapter == 'joint_fk_link7_approx' and contract.get('coordinate_frame') == 'link7'
            and contract.get('semantic') == 'expert_policy_action_target'
            and training['dataloader']['action_key'] == 'action.ee_pose'):
        # Training used recorded action.ee_pose; FK(q) is an approximation.
        return 'link7'
    raise ValueError('WM action has no matching joint-action adapter; recorded link7 poses '
                     'require joint_action_adapter: joint_fk_link7_approx')


def inspect_checkpoints(config):
    """Read CPU metadata, never instantiate a network or initialize CUDA."""
    import torch
    from omegaconf import OmegaConf
    from inference.checkpoints import _register_checkpoint_resolvers

    _register_checkpoint_resolvers(OmegaConf)
    is_pi0 = config['pipeline'] in ('pi0_contact_wm', 'pi0_latent_wm')
    if is_pi0:
        from inference.policies.pi0_joint import Pi0JointPolicy
        policy = Pi0JointPolicy(config['pi0'], client_source_path=config['policy'].get('client_source_path'))
        images = policy.image_shapes
        training = None
    else:
        dp = torch.load(config['policy']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        training = OmegaConf.to_container(dp['cfg'], resolve=True)
        p = training['policy']
        if (p['horizon'], p['n_obs_steps'], p['n_action_steps']) != (16, 2, 8):
            raise ValueError('This DP bridge requires horizon=16, observations=2, action prefix=8')
        if p.get('pred_action_steps_only', False) or training['task']['dataset']['action_key'] != 'action.joint':
            raise ValueError('DP bridge requires full-horizon absolute action.joint training')
        images = {key: tuple(spec['shape']) for key, spec in p['shape_meta']['obs'].items()
                  if spec.get('type') == 'rgb'}
        if len(images) != len(p['shape_meta']['obs']) or p['shape_meta']['action']['shape'] != [7]:
            raise ValueError('Current live DP bridge supports image-only seven-joint checkpoints')
    wm = torch.load(config['wm']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    expected_version = ('carswm_v9' if config['pipeline'] in ('dp_contact_wm', 'pi0_contact_wm')
                        else config['wm'].get('model_version', 'latent_carswm_lstm_v1'))
    if wm['model_version'] != expected_version:
        raise ValueError(f'Expected {expected_version}')
    c = wm['config']
    expected = {'high_fps': 100, 'expert_fps': 25, 'state_history_horizon': 50,
                'prediction_horizon': 32, 'action_condition_horizon': 8, 'action_start_offset': 1}
    if any(c['dataloader'].get(k) != v for k, v in expected.items()):
        raise ValueError('WM checkpoint does not match the 100 Hz history / 25 Hz action bridge')
    action_frame = joint_action_frame(c, config['wm'])
    config['wm']['action_sampling'] = resolve_action_sampling(c, config['wm'])
    return {'images': images, 'wm_training': c, 'dp_training': training,
            'alignment': alignment_report(config, c, wm.get('metrics', {})),
            'wm_model_version': wm['model_version'], 'wm_action_frame': action_frame}


def alignment_report(config, training, metrics=None):
    """Expose semantic differences even when shapes and FK agree."""
    data, deployment = training['dataloader'], config['wm']
    controller = config.get('controller', {})
    model = training.get('model', {})
    filters = data.get('filters', {})
    issues = []
    semantic = training.get('action_contract', {}).get('semantic')
    if semantic == 'va_measured_joint_snapshot_fk':
        issues.append('WM trained on FK of measured joint snapshots; deployment conditions on future policy targets. Matching FK does not remove this distribution shift.')
    if config['observation']['command_source'] == 'controller_reference':
        issues.append('delta_q uses SDK controller reference; training collection defines delta_q as last successful host q_cmd minus measured q.')
    tau = filters.get('tau', {})
    operations = tau.get('operations', []) if tau.get('enabled') else []
    cutoff = (operations[0].get('cutoff_hz') if len(operations) == 1
              and operations[0].get('type') == 'lowpass' and operations[0].get('order', 1) == 1 else None)
    if (controller.get('enabled') and controller.get('torque_reference') == 'causal_inverse'
            and cutoff is not None and controller.get('tau_filter_cutoff_hz', 20.) != cutoff):
        issues.append(f"Torque inverse filter uses {controller.get('tau_filter_cutoff_hz', 20.)} Hz but checkpoint labels use {cutoff} Hz.")
    steps = model.get('flow_inference_steps')
    if steps is not None and steps != deployment['flow_steps']:
        issues.append(f"WM sampling uses {deployment['flow_steps']} steps; checkpoint rollout uses {steps}.")
    amp = training.get('train', {}).get('amp', {})
    if amp.get('enabled') and amp.get('dtype') != deployment['precision']:
        issues.append(f"WM precision={deployment['precision']}; training AMP={amp.get('dtype')}.")
    metrics = metrics or {}
    return dict(issues=issues, wm_action_semantic=semantic, history_filters=filters,
                training_sampling_steps=steps, training_solver=model.get('flow_solver'),
                deployment_sampling_steps=deployment['flow_steps'],
                torque_label_cutoff_hz=cutoff, command_source=config['observation']['command_source'],
                execution=controller.get('execution'), controller_mode=controller.get('mode'),
                contact_classes=['free', 'pre-contact', 'contact'] if training.get('contact_gate', {}).get('label_mode') == 'three_phase' else None,
                validation={key: metrics[key] for key in ('q_physical_mae', 'q_physical_rmse',
                    'tau_physical_mae', 'tau_physical_rmse', 'contact_macro_f1',
                    'latent_zero_q_physical_mse', 'q_physical_mse') if key in metrics})


def resolve_action_sampling(training, deployment):
    """V3's indexed dataset contract is independent of FK and position encoding."""
    data = training['dataloader']
    indexed = data.get('backend') == 'lerobot' and data.get('v3_only') is True
    sampling = deployment.get('action_sampling', 'checkpoint')
    if sampling == 'checkpoint':
        sampling = 'recorded_tokens' if indexed else 'anchor_grid'
    if sampling == 'recorded_tokens':
        expected = {'backend': 'lerobot', 'v3_only': True, 'action_start_offset': 1,
                    'inference_delay_s': 0., 'action_condition_mode': 'direct'}
        keys = {'action_index_key': 'timing.action_index',
                'action_anchor_timestamp_key': 'timing.action_anchor_timestamp_ns'}
        if (any(data.get(key) != value for key, value in expected.items())
                or any(data.get(key, default) != default for key, default in keys.items())):
            raise ValueError('recorded_tokens requires V3 indexed actions, offset=1 and no action delay')
    elif sampling != 'anchor_grid':
        raise ValueError('Unknown WM action sampling')
    elif indexed:
        raise ValueError('V3 checkpoint requires recorded_tokens; anchor_grid changes trained action indices')
    return sampling


def validate_action_sampling(training, deployment):
    resolve_action_sampling(training, deployment)


def validate_contact_action_times(anchor_us, times, sampling):
    times = np.asarray(times)
    if (times.shape != (8,) or times.dtype.kind not in 'iu'
            or np.any(np.diff(times) != 40_000)):
        raise ValueError('Contact WM requires eight consecutive 25 Hz action timestamps')
    if sampling == 'recorded_tokens':
        if not 0 < int(times[0])-anchor_us <= 40_000:
            raise ValueError('Recorded action window must start at the next committed token')
    elif not np.array_equal(times, anchor_us + np.arange(1,9,dtype=np.int64)*40_000):
        raise ValueError('Contact WM requires actions on the history-anchor 25 Hz grid')


class JointToPose:
    def __init__(self, urdf, *, frame_name='link_eef'):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        if list(self.model.names[1:]) != [f'joint{i}' for i in range(1, 8)] or self.model.nq != 7:
            raise ValueError('FK URDF joint order must be joint1..joint7')
        for name in ('link_base', frame_name):
            if not self.model.existFrame(name):
                raise ValueError(f'Missing FK frame {name}')
        self.base, self.tip = [self.model.getFrameId(n) for n in ('link_base', frame_name)]

    def __call__(self, actions):
        from scipy.spatial.transform import Rotation
        result = []
        for q in actions:
            self.pin.framesForwardKinematics(self.model, self.data, np.asarray(q, dtype=float))
            pose = self.data.oMf[self.base].inverse() * self.data.oMf[self.tip]
            quat = Rotation.from_matrix(pose.rotation).as_quat()
            if quat[3] < 0:
                quat = -quat
            result.append(np.r_[pose.translation, quat])
        return np.asarray(result, dtype=np.float32)


class ContactWM:
    """Native carswm_v9 adapter, with joint-plan FK and strict action timing."""
    def __init__(self, config):
        import torch
        add_pinn_source(config['pinn_root'])
        from model.pinn_model.contact_world_model import ContactWorldModel
        from model.pinn_model.latent_pretrained import convert_scale
        from data_process.causal_data_filter import normalize_dataloader_filters
        payload = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
        training = payload['config']
        action_frame = joint_action_frame(training, config)
        config['action_sampling'] = resolve_action_sampling(training, config)
        self.model = ContactWorldModel(training)
        self.model.validate_checkpoint(payload)
        if training.get('train', {}).get('ema', {}).get('enabled') and payload.get('ema') is None:
            raise ValueError('Contact WM checkpoint is missing EMA state')
        self.model.load_state_dict(payload['model'], strict=True)
        self.device = torch.device(config['device'])
        self.model.to(self.device).eval()
        data = training['dataloader']
        if payload.get('dataloader_filters') != normalize_dataloader_filters(data):
            raise ValueError('Contact WM checkpoint preprocessing differs from training config')
        for key in ('normalize_mode', 'normalize_lowdim_keys'):
            if key in payload['normalizer'] and payload['normalizer'][key] != data[key]:
                raise ValueError(f'Contact WM normalizer disagrees with {key}')
        self.normalizer = {**payload['normalizer'], 'normalize_mode': data['normalize_mode'],
                           'normalize_lowdim_keys': data['normalize_lowdim_keys']}
        self.convert_scale = convert_scale
        from inference.dp_wm.scaling import CheckpointScaler
        self.scaler = CheckpointScaler(self.normalizer, self.device)
        self.fk = JointToPose(config['urdf_path'], frame_name=action_frame)
        self.torch, self.config = torch, config
        self.sampler = None
        if config.get('cuda_graph', False) and self.device.type == 'cuda':
            from inference.dp_wm.cuda_sampler import CudaContactSampler
            self.sampler = CudaContactSampler(self.model, steps=config['flow_steps'], solver=config['solver'],
                num_samples=config['num_samples'], cache_condition_kv=config.get('cache_condition_kv', False))
        self.stream = None
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream(device=self.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))

    def infer(self, request):
        from contextlib import nullcontext
        history = request['history']
        history.validate()
        validate_contact_action_times(history.anchor_us, request['action_timestamps_us'],
                                      self.config.get('action_sampling', 'anchor_grid'))
        device = self.device
        with (self.torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()), \
                self.torch.inference_mode(), self.torch.autocast(device.type,
                    dtype=self.torch.float16 if self.config['precision']=='float16' else self.torch.bfloat16,
                    enabled=self.config['precision']!='float32', cache_enabled=self.sampler is None):
            batch = {key: self.torch.as_tensor(value[None], dtype=self.torch.float32, device=device)
                     for key, value in history.values.items()}
            batch['action'] = self.torch.as_tensor(self.fk(request['actions'])[None], device=device)
            batch = {key: self.scaler(key, value) for key,value in batch.items()}
            out = (self.sampler.sample(batch) if self.sampler is not None else
                self.model.sample(batch, num_samples=self.config['num_samples'],
                    steps=self.config['flow_steps'], solver=self.config['solver'],
                    cache_condition_kv=self.config.get('cache_condition_kv', False)))
            result = {key: self.scaler(key, out[key+'_pred'].float(),
                       inverse=True)[0].cpu().numpy() for key in ('q','tau')}
            result['contact_probability'] = out['contact_probability'][0].float().cpu().numpy()
            for key, value in result.items():
                width = 3 if key == 'contact_probability' else 7
                if value.shape != (self.config['num_samples'],32,width) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid contact WM output {key}')
        result.update(anchor_us=history.anchor_us, plan_version=request['plan_version'],
            deadline_us=request['deadline_us'], anchor_tau_filtered=history.values['tau'][-1].copy(),
            future_timestamps_us=history.anchor_us+np.arange(1,33,dtype=np.int64)*10_000)
        return result


class LatentWM:
    def __init__(self, config):
        import torch
        add_pinn_source(config['pinn_root'])
        from model.pinn_model.latent_contact_world_model import load_latent_checkpoint
        from model.pinn_model.latent_pretrained import convert_scale
        self.torch, self.convert_scale = torch, convert_scale
        self.config = config
        self.model, payload = load_latent_checkpoint(config['checkpoint'], device=config['device'], use_ema=True)
        config['action_sampling'] = resolve_action_sampling(payload['config'], config)
        del payload
        self.fk = JointToPose(config['urdf_path'])
        self.sampler = None
        if config.get('cuda_graph', False) and self.model.latent_mean.device.type == 'cuda':
            from inference.dp_wm.cuda_sampler import CudaFlowSampler
            self.sampler = CudaFlowSampler(self.model, steps=config['flow_steps'],
                                          solver=config['solver'], num_samples=config['num_samples'])
        self.cuda_stream = None
        if self.model.latent_mean.device.type == 'cuda':
            self.cuda_stream = torch.cuda.Stream(device=self.model.latent_mean.device)
            self.cuda_stream.wait_stream(torch.cuda.current_stream(self.model.latent_mean.device))

    def infer(self, request):
        if self.cuda_stream is not None:
            # Independent DP/WM models must not serialize all kernels on the
            # default stream. CPU result copies below complete this stream.
            with self.torch.cuda.stream(self.cuda_stream):
                return self._infer(request)
        return self._infer(request)

    def _infer(self, request):
        torch = self.torch
        with torch.inference_mode():
            history = request['history']
            history.validate()
            if self.model.learned_positions:
                validate_contact_action_times(history.anchor_us, request['action_timestamps_us'],
                                              self.config['action_sampling'])
            device = self.model.latent_mean.device
            values = {key: torch.as_tensor(value[None], dtype=torch.float32, device=device)
                      for key, value in history.values.items()}
            values['action'] = torch.as_tensor(self.fk(request['actions'])[None], device=device)
            positions = ({} if self.model.learned_positions else self.model.grid.positions(
                history_ns=torch.as_tensor(history.timestamps_us[None]*1000, device=device),
                action_ns=torch.as_tensor(request['action_timestamps_us'][None]*1000, device=device),
                future_horizon=32, history_horizon=50, action_start_offset=1))
            batch = {key: self.convert_scale(key, value, self.model.wm_normalizer) for key, value in values.items()}
            batch.update(positions)
            batch['action_mask'] = torch.ones(1, 8, dtype=torch.bool, device=device)
            precision = self.config.get('precision', 'float32')
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16 if precision=='float16' else torch.bfloat16,
                                enabled=precision != 'float32',
                                cache_enabled=self.sampler is None):
                if self.sampler is None:
                    out = self.model.sample(batch, num_samples=self.config['num_samples'],
                                            steps=self.config['flow_steps'], solver=self.config['solver'])
                else:
                    out = self.sampler.sample(batch)
            result = {key: self.convert_scale(key, out[key+'_pred'].float(), self.model.wm_normalizer,
                                              inverse=True)[0].cpu().numpy() for key in ('q', 'tau')}
            result['contact_probability'] = out['contact_probability'][0].float().cpu().numpy()
            if any(not np.isfinite(value).all() for value in result.values()):
                raise ValueError('WM returned nonfinite predictions')
            result.update(anchor_us=history.anchor_us, plan_version=request['plan_version'],
                          deadline_us=request['deadline_us'],
                          anchor_tau_filtered=history.values['tau'][-1].copy(),
                          future_timestamps_us=history.anchor_us + np.arange(1, 33, dtype=np.int64)*10_000)
            return result


def load_models(config, metadata):
    if config['pipeline'] in ('pi0_contact_wm', 'pi0_latent_wm'):
        from inference.policies.pi0_joint import Pi0JointPolicy
        wm_class = ContactWM if config['pipeline'] == 'pi0_contact_wm' else LatentWM
        return (Pi0JointPolicy(config['pi0'], client_source_path=config['policy'].get('client_source_path')),
                wm_class(config['wm']))
    from inference.dp_wm.core import DP_PLAN_STEPS
    import torch
    from inference.policies.dp.policy import DiffusionPolicy
    torch.set_num_threads(config['policy'].get('torch_threads', 1))
    sys.path.insert(0, str(config['policy']['source_path']))
    options = config['policy']
    policy = DiffusionPolicy.from_checkpoint(options['checkpoint'], device=options['device'],
        sampling_method='ddim', num_inference_steps=options['num_inference_steps'], step_s=.04,
        action_semantic='joint', use_ema=True)
    precision = options.get('precision', 'float32')
    if precision != 'float32':
        original_predict = policy._predict_action
        dtype = torch.float16 if precision=='float16' else torch.bfloat16
        def predict_with_precision(batch):
            with torch.autocast(torch.device(options['device']).type, dtype=dtype):
                return original_predict(batch)
        policy._predict_action = predict_with_precision
    if policy.image_keys != tuple(metadata['images']):
        if set(policy.image_keys) != set(metadata['images']):
            raise ValueError('Restored policy image keys differ from checkpoint metadata')
    if options.get('cache_observation_features', False):
        from inference.dp_wm.vision_cache import ObservationFeatureCache
        policy.feature_cache = ObservationFeatureCache(policy)
    wm = (ContactWM if config['pipeline']=='dp_contact_wm' else LatentWM)(config['wm'])
    # Warm both actual call paths before connecting hardware or submitting live
    # observations. Synthetic outputs are discarded, never used as forecasts.
    observations, request = warmup_inputs(metadata['images'])
    for _ in range(config.get('execution', {}).get('warmup_steps', 3)):
        policy.predict_observations(observations, plan_steps=DP_PLAN_STEPS)
        wm.infer(request)
    if options.get('cache_observation_features', False):
        # Warm the one-new-frame path as well as the initial two-frame path.
        extra,_ = warmup_inputs(metadata['images'])
        extra[-1].timestamp_us = observations[-1].timestamp_us+40_000
        policy.predict_observations((observations[-1],extra[-1]),plan_steps=DP_PLAN_STEPS)
    return policy, wm


def warmup_inputs(images):
    """Synthetic inputs for discarded startup calls; never published to the pipeline."""
    from types import SimpleNamespace
    from inference.dp_wm.core import HistorySnapshot
    anchor = 1_000_000
    observations = tuple(SimpleNamespace(timestamp_us=stamp, q=np.zeros(7), metadata={},
        images={key: np.zeros((shape[1], shape[2], shape[0]), np.uint8)
                for key, shape in images.items()}) for stamp in (anchor-40_000, anchor))
    for obs in observations:
        for image in obs.images.values():
            image.setflags(write=False)
    history = HistorySnapshot(anchor + np.arange(-49, 1, dtype=np.int64)*10_000,
        {key: np.zeros((50, 7), np.float32) for key in ('q','dq','delta_q','tau')})
    request = dict(history=history, actions=np.zeros((8, 7)),
                   action_timestamps_us=anchor + np.arange(1, 9, dtype=np.int64)*40_000,
                   plan_version=0, deadline_us=anchor+40_000)
    return observations, request
