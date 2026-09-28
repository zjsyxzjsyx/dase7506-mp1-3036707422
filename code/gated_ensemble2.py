"""Serial FP32 ensemble that spends the second forward pass on uncertain windows."""
import math

import torch
from torch.nn import functional as F

from ensemble2 import GPTEnsemble2


class GatedEnsemble2(GPTEnsemble2):
    def __init__(self, config):
        super().__init__(config)
        self.first_token_max_probability_threshold = float(
            config['first_token_max_probability_threshold'])
        if not 0 < self.first_token_max_probability_threshold <= 1:
            raise ValueError('Confidence threshold must be in (0, 1]')

    def predict_log_probs(self, ids):
        if torch.is_grad_enabled():
            return super().predict_log_probs(ids)
        first = F.log_softmax(self.members[0](ids).float(), dim=-1)
        # The first prediction depends only on ids[:, 0], so this gate cannot
        # inspect future tokens within a window or other examples in the batch.
        use_second = first[:, 0, :].amax(dim=-1).exp() < self.first_token_max_probability_threshold
        if not use_second.any():
            return first
        second = F.log_softmax(self.members[1](ids[use_second]).float(), dim=-1)
        result = first.clone()
        result[use_second] = torch.logaddexp(
            first[use_second] + math.log(self.weights[0]),
            second + math.log(self.weights[1]))
        return result


def build_model(config):
    return GatedEnsemble2(dict(config))
