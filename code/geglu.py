"""Parameter-matched GEGLU experiment with the original LayerNorm architecture.

Method: Shazeer, GLU Variants Improve Transformer, https://arxiv.org/abs/2002.05202
We retain biases to match the classroom baseline convention.
"""
from torch import nn
from torch.nn import functional as F
from model import GPT


class GEGLU(nn.Module):
    def __init__(self, width, fusion='multiply'):
        super().__init__()
        if fusion not in ('multiply', 'add'):
            raise ValueError(f'Unknown fusion: {fusion!r}')
        self.fusion = fusion
        hidden = round(8 * width / 3)
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.out = nn.Linear(hidden, width)
        self.apply(GPT.initialize)

    def forward(self, x):
        gate, value = F.gelu(self.gate(x)), self.value(x)
        return self.out(gate * value if self.fusion == 'multiply' else gate + value)


def build_model(config):
    config = dict(config)
    ffn_type = config.setdefault('ffn_type', 'geglu')
    if ffn_type not in ('gelu', 'geglu', 'additive'):
        raise ValueError(f'Unknown ffn_type: {ffn_type!r}')
    if config.get('norm_type', 'layernorm') != 'layernorm':
        raise ValueError('This experiment requires LayerNorm.')
    # Initialize the common architecture first to preserve all shared weights.
    model = GPT(config)
    if ffn_type in ('geglu', 'additive'):
        for block in model.blocks:
            block.mlp = GEGLU(config['width'], fusion='add' if ffn_type == 'additive' else 'multiply')
    return model
