"""Validate immutable checkpoint statistics once and keep them on the model device."""


class CheckpointScaler:
    def __init__(self, envelope, device):
        from model.pinn_model.latent_pretrained import validate_normalizer
        from train.nomalizer import Normalizer
        self.mode = envelope.get('normalize_mode')
        self.keys = frozenset(envelope.get('normalize_lowdim_keys', ()))
        if self.mode is not None:
            for key in self.keys:
                validate_normalizer(envelope, key, 7)
        stats = {key: {name: value.detach().to(device).clone() for name,value in values.items()}
                 for key,values in envelope['stats'].items()}
        self.normalizer = Normalizer(stats, eps=envelope.get('eps',1e-6))

    def __call__(self, key, value, *, inverse=False):
        if self.mode is None or key not in self.keys:
            return value
        if inverse and self.mode=='quantile':
            raise ValueError('Clipped quantile statistics cannot be inverted')
        name = f'{self.mode}_{"denormalize" if inverse else "normalize"}'
        return getattr(self.normalizer,name)(key,value)
