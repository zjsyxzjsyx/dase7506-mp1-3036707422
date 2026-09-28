"""GPT with light residual-branch dropout during training only."""
import torch
from torch import nn
from torch.nn import functional as F

from model import Block, GPT


class DropoutBlock(Block):
    def __init__(self, width, heads, dropout_p):
        super().__init__(width, heads)
        self.residual_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention_output = self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + self.residual_dropout(attention_output)
        return x + self.residual_dropout(self.mlp(self.norm2(x)))


class GPTDropout(GPT):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        dropout_p = float(config['dropout_p'])
        if not 0 <= dropout_p < 1:
            raise ValueError('dropout_p must be in [0, 1)')
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList(
            DropoutBlock(width, config['heads'], dropout_p)
            for _ in range(config['depth'])
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight


def build_model(config):
    return GPTDropout(dict(config))
