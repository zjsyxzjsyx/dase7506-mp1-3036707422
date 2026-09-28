"""Causal probability mixture of three independently trained GPTs."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT


class GPTEnsemble3(nn.Module):
    def __init__(self, config):
        super().__init__()
        member_configs = config['member_configs']
        weights = [float(value) for value in config['member_weights']]
        if len(member_configs) != 3 or len(weights) != 3:
            raise ValueError('Exactly three member configurations and weights are required')
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
        """Return finite, unnormalized logits with gradients through all members."""
        logits = None
        for weight, member in zip(self.weights, self.members):
            branch = member(ids)
            logits = branch * weight if logits is None else logits + branch * weight
        return logits

    def predict_log_probs(self, ids):
        """Mix probabilities independently for each causal scoring window."""
        mixture = None
        for weight, member in zip(self.weights, self.members):
            branch = F.log_softmax(member(ids).float(), dim=-1) + math.log(weight)
            mixture = branch if mixture is None else torch.logaddexp(mixture, branch)
        return mixture


def build_model(config):
    return GPTEnsemble3(dict(config))
