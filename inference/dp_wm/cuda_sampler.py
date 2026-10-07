"""CUDA replay of native contact sampling and latent Heun/Euler integration."""


class CudaContactSampler:
    """Capture the native public sample API, including encoding and per-request K/V.

    Fixed input buffers are refreshed on every request. Nothing from the previous
    history/action window is reused except the CUDA execution graph itself.
    The returned tensors belong to the sampler until its next replay.
    """
    def __init__(self, model, *, steps, solver, num_samples, cache_condition_kv=True):
        if model.training:
            raise ValueError('Contact CUDA replay requires an eval model')
        if model.runtime_checks:
            raise ValueError('Contact CUDA replay requires runtime_checks=false; adapter validates inputs/outputs')
        self.model = model
        self.options = dict(steps=steps, solver=solver, num_samples=num_samples,
                            cache_condition_kv=cache_condition_kv)
        self.graph = None

    def sample(self, batch, *, source_noise=None):
        import torch
        reference = batch[self.model.inputs[0]]
        if reference.device.type != 'cuda':
            raise ValueError('Contact CUDA replay requires CUDA inputs')
        shape = (reference.shape[0], self.options['num_samples'],
                 self.model.future_horizon, self.model.flow_dim)
        noise = torch.randn(shape, device=reference.device, dtype=reference.dtype) if source_noise is None else source_noise
        if noise.shape != shape or noise.device != reference.device or noise.dtype != reference.dtype:
            raise ValueError('Contact source noise shape/device/dtype mismatch')
        if self.graph is None:
            self.batch = {key: value.clone() for key, value in batch.items()}
            self.noise = noise.clone()
            stream = torch.cuda.Stream(device=reference.device)
            stream.wait_stream(torch.cuda.current_stream(reference.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.model.sample(self.batch, source_noise=self.noise, **self.options)
            torch.cuda.current_stream(reference.device).wait_stream(stream)
            torch.cuda.synchronize(reference.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = self.model.sample(self.batch, source_noise=self.noise, **self.options)
        else:
            if batch.keys() != self.batch.keys():
                raise ValueError('Contact CUDA replay input keys changed')
            for key, value in batch.items():
                original = self.batch[key]
                if (value.shape, value.dtype, value.device) != (original.shape, original.dtype, original.device):
                    raise ValueError(f'Contact CUDA replay input contract changed: {key}')
                original.copy_(value)
            self.noise.copy_(noise)
        self.graph.replay()
        return self.output


class CudaFlowSampler:
    KEYS = ('history', 'action', 'action_padding_mask', 'future_pe')

    def __init__(self, model, *, steps, solver, num_samples):
        self.model = model
        self.steps, self.solver, self.num_samples = steps, solver, num_samples
        self.graph = None

    def _integrate(self):
        # Matches LatentContactWorldModel.sample; keep the original NFE, noise,
        # precision, conditioning and decoder. Only Python/kernel launch work
        # is removed from subsequent calls.
        m = self.model
        z = self.noise.clone().reshape(-1, m.future_horizon, m.latent_dim)
        dt = 1. / self.steps
        for i in range(self.steps):
            first = m.velocity(z, i*dt, self.encoded)
            if self.solver == 'euler':
                z = z + dt*first
            else:
                second = m.velocity(z + dt*first, (i+1)*dt, self.encoded)
                z = z + dt*.5*(first+second)
        raw = z.float()*m.latent_std + m.latent_mean
        batch = self.noise.shape[0]
        raw = raw.reshape(batch, self.num_samples, m.future_horizon, m.latent_dim)
        decoded = (m.decode(raw, self.decode_batch) if m.conditional_codec else m.decode(raw))
        return {key: value.reshape(batch, self.num_samples, m.future_horizon, -1)
                for key, value in decoded.items()}

    def sample(self, batch, *, source_noise=None):
        import torch
        encoded = self.model.encode_conditions(batch)
        encoded = {key: encoded[key].repeat_interleave(self.num_samples, 0) for key in self.KEYS}
        shape = (batch['q'].shape[0], self.num_samples, self.model.future_horizon, self.model.latent_dim)
        device = encoded['history'].device
        if device.type != 'cuda':
            raise ValueError('CUDA graph sampler requires CUDA inputs')
        noise = torch.randn(shape, device=device) if source_noise is None else source_noise
        if noise.shape != shape or noise.device != device:
            raise ValueError('Invalid latent source noise shape/device')
        if self.graph is None:
            self.encoded = {key: value.clone() for key, value in encoded.items()}
            self.decode_batch = ({key: batch[key].clone() for key in ('q', 'dq', 'delta_q', 'tau')}
                                 if self.model.conditional_codec else None)
            self.noise = noise.clone()
            # Capture before live worker threads start; CUDA capture cannot
            # coexist with unrelated model launches in another host thread.
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._integrate()
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = self._integrate()
        else:
            for key, value in encoded.items():
                original = self.encoded[key]
                if value.shape != original.shape or value.dtype != original.dtype or value.device != original.device:
                    raise ValueError('CUDA graph conditioning contract changed')
                original.copy_(value)
            if self.decode_batch is not None:
                for key, original in self.decode_batch.items():
                    value = batch[key]
                    if (value.shape, value.dtype, value.device) != (original.shape, original.dtype, original.device):
                        raise ValueError(f'CUDA graph decode history contract changed: {key}')
                    original.copy_(value)
            self.noise.copy_(noise)
        self.graph.replay()
        return self.output
