"""Causal probability mixture of two independently trained GPTs."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT


class GPTEnsemble2(nn.Module):
    def __init__(self, config):
        super().__init__()
        member_configs = config['member_configs']
        weights = [float(value) for value in config['member_weights']]
        if len(member_configs) != 2 or len(weights) != 2:
            raise ValueError('Exactly two member configurations and weights are required')
        if any(value <= 0 or not math.isfinite(value) for value in weights):
            raise ValueError('Member weights must be finite and positive')
        if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-7):
            raise ValueError('Member weights must sum to one')
        if any(member['context'] != config['context'] or member['vocab'] != config['vocab']
               for member in member_configs):
            raise ValueError('Members must share the evaluator context and vocabulary')
        self.context = config['context']
        self.weights = weights
        self.members = nn.ModuleList(GPT(member) for member in member_configs)

    def forward(self, ids):
        return sum(weight * member(ids) for weight, member in zip(self.weights, self.members))

    def predict_log_probs(self, ids):
        first = F.log_softmax(self.members[0](ids).float(), dim=-1) + math.log(self.weights[0])
        second = F.log_softmax(self.members[1](ids).float(), dim=-1) + math.log(self.weights[1])
        return torch.logaddexp(first, second)


def build_model(config):
    return GPTEnsemble2(dict(config))
