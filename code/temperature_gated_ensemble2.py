"""Serial confidence-gated students with fixed per-student logit temperatures."""
import math

import torch
from torch.nn import functional as F

from gated_ensemble2 import GatedEnsemble2


class TemperatureGatedEnsemble2(GatedEnsemble2):
    def __init__(self, config):
        super().__init__(config)
        self.member_temperatures = tuple(float(value) for value in config['member_temperatures'])
        if len(self.member_temperatures) != 2 or any(
                not math.isfinite(value) or value <= 0 for value in self.member_temperatures):
            raise ValueError('Two finite positive student temperatures are required')

    def predict_log_probs(self, ids):
        first = F.log_softmax(self.members[0](ids).float() / self.member_temperatures[0], dim=-1)
        if torch.is_grad_enabled():
            second = F.log_softmax(self.members[1](ids).float() / self.member_temperatures[1], dim=-1)
            return torch.logaddexp(first + math.log(self.weights[0]),
                                   second + math.log(self.weights[1]))
        use_second = first[:, 0, :].amax(dim=-1).exp() < self.first_token_max_probability_threshold
        if not use_second.any():
            return first
        second = F.log_softmax(self.members[1](ids[use_second]).float()
                               / self.member_temperatures[1], dim=-1)
        result = first.clone()
        result[use_second] = torch.logaddexp(
            first[use_second] + math.log(self.weights[0]),
            second + math.log(self.weights[1]))
        return result


def build_model(config):
    return TemperatureGatedEnsemble2(dict(config))
