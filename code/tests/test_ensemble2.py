"""Check the two-model predictor's probability and causal contracts."""
import math
import unittest

import torch
from torch.nn import functional as F

from ensemble2 import build_model


class Ensemble2Tests(unittest.TestCase):
    def test_probability_mixture_and_causality(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        members = [dict(vocab=2048, context=256, width=32, heads=4, depth=2)] * 2
        model = build_model(dict(vocab=2048, context=256, member_configs=members,
                                 member_weights=[0.5, 0.5])).eval()
        ids = torch.randint(0, 2048, (2, 12))
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 19) % 2048
        with torch.no_grad():
            actual = model.predict_log_probs(ids)
            expected = torch.logaddexp(*[F.log_softmax(member(ids), dim=-1) - math.log(2)
                                         for member in model.members])
            future_changed = model.predict_log_probs(changed)
            alone = model.predict_log_probs(ids[:1])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(2, 12), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[:, :7], future_changed[:, :7], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[:1], alone, atol=1e-5, rtol=1e-5)

    def test_forward_gradients_reach_both_members(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        members = [dict(vocab=2048, context=256, width=32, heads=4, depth=2)] * 2
        model = build_model(dict(vocab=2048, context=256, member_configs=members,
                                 member_weights=[0.5, 0.5]))
        ids = torch.randint(0, 2048, (2, 13))
        loss = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
        loss.backward()
        for member in model.members:
            gradients = [parameter.grad for parameter in member.parameters()
                         if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)


if __name__ == '__main__':
    unittest.main()
