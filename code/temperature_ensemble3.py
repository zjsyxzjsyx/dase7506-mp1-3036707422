"""Three-member probability mixture with one shared logit temperature."""
import math

from torch.nn import functional as F

from ensemble3 import GPTEnsemble3


class TemperatureEnsemble3(GPTEnsemble3):
    def __init__(self, config):
        super().__init__(config)
        self.temperature = float(config['temperature'])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Temperature must be finite and positive')

    def predict_log_probs(self, ids):
        mixture = None
        for weight, member in zip(self.weights, self.members):
            branch = F.log_softmax(member(ids).float() / self.temperature, dim=-1)
            branch = branch + math.log(weight)
            mixture = branch if mixture is None else mixture.logaddexp(branch)
        return mixture


def build_model(config):
    return TemperatureEnsemble3(dict(config))
