"""Contract checks for the mixed-width three-model probability mixture."""
import math
import unittest

import torch
from torch.nn import functional as F

from ensemble3 import build_model


class Ensemble3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        members = [dict(vocab=2048, context=256, width=width, heads=4, depth=2)
                   for width in (32, 48, 32)]
        self.model = build_model(dict(vocab=2048, context=256,
                                      member_configs=members,
                                      member_weights=[1 / 3] * 3)).eval()

    def test_probability_mixture_and_normalization(self):
        ids = torch.randint(0, 2048, (2, 12))
        with torch.no_grad():
            actual = self.model.predict_log_probs(ids)
            branches = [F.log_softmax(member(ids), dim=-1) - math.log(3)
                        for member in self.model.members]
            expected = torch.logsumexp(torch.stack(branches), dim=0)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(2, 12),
                                   atol=1e-6, rtol=1e-6)

    def test_causality_and_window_independence(self):
        ids = torch.randint(0, 2048, (2, 12))
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 19) % 2048
        with torch.no_grad():
            together = self.model.predict_log_probs(ids)
            future_changed = self.model.predict_log_probs(changed)
            self.model.predict_log_probs((ids + 31) % 2048)
            again = self.model.predict_log_probs(ids)
            alone = self.model.predict_log_probs(ids[:1])
        torch.testing.assert_close(together[:, :7], future_changed[:, :7], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(together, again)
        torch.testing.assert_close(together[:1], alone, atol=1e-5, rtol=1e-5)

    def test_forward_gradients_reach_all_members(self):
        ids = torch.randint(0, 2048, (2, 13))
        loss = F.cross_entropy(self.model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
        loss.backward()
        for member in self.model.members:
            gradients = [parameter.grad for parameter in member.parameters()
                         if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)


if __name__ == '__main__':
    unittest.main()
