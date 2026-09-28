"""RMSNorm experiment; select norm_type='layernorm' for the baseline ablation.

Reference: Zhang and Sennrich, Root Mean Square Layer Normalization, NeurIPS 2019.
https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html
"""
from torch import nn
from model import GPT


def build_model(config):
    config = dict(config)
    norm_type = config.setdefault('norm_type', 'rmsnorm')
    if norm_type not in ('layernorm', 'rmsnorm'):
        raise ValueError(f'Unknown norm_type: {norm_type!r}')
    # Initialize the original GPT first so shared weights match at the same seed.
    model = GPT(config)
    if norm_type == 'rmsnorm':
        width = config['width']
        for block in model.blocks:
            block.norm1 = nn.RMSNorm(width, eps=1e-5)
            block.norm2 = nn.RMSNorm(width, eps=1e-5)
        model.norm = nn.RMSNorm(width, eps=1e-5)
    return model
