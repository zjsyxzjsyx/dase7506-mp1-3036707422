"""Check that a window's gate and predictions cannot use future inputs."""
import unittest

import torch
from torch.nn import functional as F

from ensemble2 import build_model as build_original
from gated_ensemble2 import build_model as build_gated


class GatedEnsemble2Tests(unittest.TestCase):
    def test_causality_normalization_and_independence(self):
        torch.set_num_threads(2)
        member = {'vocab': 2048, 'width': 32, 'heads': 4, 'depth': 1, 'context': 256}
        config = {'vocab': 2048, 'context': 256,
                  'member_configs': [member] * 2, 'member_weights': [0.5, 0.5]}
        torch.manual_seed(17)
        original = build_original(config).eval()
        ids = torch.randint(2048, (3, 12))
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 19) % 2048
        with torch.no_grad():
            confidence = F.softmax(original.members[0](ids)[:, 0, :], dim=-1).amax(-1)
        threshold = (confidence.min().item() + confidence.max().item()) / 2
        use_second = confidence < threshold
        self.assertTrue(use_second.any())
        self.assertTrue((~use_second).any())
        gated = build_gated(dict(config, first_token_max_probability_threshold=threshold)).eval()
        gated.load_state_dict(original.state_dict())
        with torch.no_grad():
            actual = gated.predict_log_probs(ids)
            future = gated.predict_log_probs(changed)
            alone = gated.predict_log_probs(ids[:1])
            full_ensemble = original.predict_log_probs(ids)
            first_only = F.log_softmax(original.members[0](ids), dim=-1)
        torch.testing.assert_close(actual[use_second], full_ensemble[use_second], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[~use_second], first_only[~use_second], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[:, :7], future[:, :7], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual[:1], alone, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(3, 12), atol=1e-5, rtol=1e-5)
        self.assertTrue(gated.predict_log_probs(ids).requires_grad)


if __name__ == '__main__':
    unittest.main()
