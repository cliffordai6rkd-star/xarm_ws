"""Reuse deterministic DP image features for unchanged observation objects."""
from collections import OrderedDict


class ObservationFeatureCache:
    def __init__(self, policy, maximum_entries=8):
        self.policy = policy
        self.encoder = policy.model.obs_encoder
        if self.encoder.training:
            raise ValueError('Observation feature caching requires an eval encoder')
        self.original_forward = self.encoder.forward
        self.original_predict = policy.predict_observations
        self.cache = OrderedDict()
        self.maximum_entries = maximum_entries
        self.window = None
        self.enabled = True
        self.hits = self.misses = 0
        self.encoder.forward = self.forward
        policy.predict_observations = self.predict_observations

    def predict_observations(self, observations, **kwargs):
        window = list(observations[-self.policy.n_obs_steps:])
        while window and len(window)<self.policy.n_obs_steps:
            window.insert(0,window[0])
        self.window = window
        try:
            return self.original_predict(observations,**kwargs)
        finally:
            self.window = None

    def forward(self, batch):
        import torch
        reference = next(iter(batch.values()))
        mutable = self.window is not None and any(
            not hasattr(obs.images[k],'flags') or obs.images[k].flags.writeable
            for obs in self.window for k in self.policy.image_keys)
        if mutable or self.encoder.training or torch.is_grad_enabled():
            self.cache.clear()
        if (not self.enabled or self.window is None or self.encoder.training or torch.is_grad_enabled()
                or reference.shape[0]!=len(self.window)
                or mutable):
            return self.original_forward(batch)
        # Retain each observation object, preventing object-id reuse while its
        # features are cached. LiveSource creates immutable image snapshots.
        # Include autocast/device state so changing precision cannot reuse a
        # feature computed under another numerical mode.
        autocast = torch.is_autocast_enabled(reference.device.type)
        precision = torch.get_autocast_dtype(reference.device.type) if autocast else reference.dtype
        keys = [(obs.timestamp_us,id(obs),tuple(id(obs.images[k]) for k in self.policy.image_keys),
                 reference.device,precision) for obs in self.window]
        missing = []
        missing_keys = set()
        for i,key in enumerate(keys):
            if key not in self.cache and key not in missing_keys:
                missing.append(i)
                missing_keys.add(key)
        if missing:
            indices = torch.as_tensor(missing,device=reference.device)
            features = self.original_forward({k:v.index_select(0,indices) for k,v in batch.items()})
            for i,value in zip(missing,features):
                self.cache[keys[i]] = (self.window[i],value.detach().clone())
        result = torch.stack([self.cache[key][1] for key in keys])
        self.misses += len(missing)
        self.hits += len(keys)-len(missing)
        for key in keys:
            self.cache.move_to_end(key)
        while len(self.cache)>self.maximum_entries:
            self.cache.popitem(last=False)
        return result

    def report(self):
        return dict(enabled=self.enabled,hits=self.hits,misses=self.misses,entries=len(self.cache))
