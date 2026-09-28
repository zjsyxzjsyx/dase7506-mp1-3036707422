"""Checks that the experiment changes normalization, not initialization or I/O."""
import io
import unittest

import torch
from torch import nn

from model import GPT
from student import build_model


class NormTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = dict(vocab=2048, width=128, heads=4, depth=4, context=256)

    def test_layernorm_ablation_matches_original(self):
        torch.manual_seed(17)
        baseline = GPT(self.config).eval()
        torch.manual_seed(17)
        ablation = build_model(self.config | {'norm_type': 'layernorm'}).eval()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, ablation.state_dict()[key], rtol=0, atol=0)
        ids = torch.randint(2048, (2, 17))
        with torch.no_grad():
            torch.testing.assert_close(baseline(ids), ablation(ids), rtol=0, atol=0)

    def test_rmsnorm_preserves_shared_initialization(self):
        torch.manual_seed(17)
        baseline = GPT(self.config)
        torch.manual_seed(17)
        model = build_model(self.config)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, baseline.state_dict()[key], rtol=0, atol=0)
        norms = [m for m in model.modules() if isinstance(m, nn.RMSNorm)]
        self.assertEqual(len(norms), 9)
        self.assertFalse(any(isinstance(m, nn.LayerNorm) for m in model.modules()))
        self.assertTrue(all(m.eps == 1e-5 for m in norms))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1087104)
        self.assertIs(model.head.weight, model.token.weight)

    def test_checkpoint_roundtrip_and_full_context(self):
        model = build_model(self.config).eval()
        buffer = io.BytesIO()
        torch.save({'config': model.config, 'model': model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        loaded = build_model(checkpoint['config']).eval()
        loaded.load_state_dict(checkpoint['model'])
        ids = torch.randint(2048, (1, 256))
        with torch.no_grad():
            expected = model.predict_log_probs(ids)
            actual = loaded.predict_log_probs(ids)
        self.assertEqual(actual.shape, (1, 256, 2048))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(1, 256), atol=1e-6, rtol=0)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_unknown_norm_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown norm_type'):
            build_model(self.config | {'norm_type': 'unknown'})


if __name__ == '__main__':
    unittest.main()
