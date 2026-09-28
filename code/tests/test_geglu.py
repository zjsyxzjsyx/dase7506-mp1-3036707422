import io
import unittest
import torch
from torch import nn
import test_contract
from geglu import build_model
from model import GPT


class GEGLUContractTests(test_contract.ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(dict(vocab=2048, width=32, heads=4, depth=2, context=256)).eval()


class GEGLUImplementationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = dict(vocab=2048, width=128, heads=4, depth=4, context=256)

    def test_ablation_and_common_weights(self):
        torch.manual_seed(17)
        baseline = GPT(self.config).eval()
        torch.manual_seed(17)
        ablation = build_model(self.config | {'ffn_type': 'gelu'}).eval()
        torch.manual_seed(17)
        candidate = build_model(self.config).eval()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, ablation.state_dict()[key], atol=0, rtol=0)
            if '.mlp.' not in key:
                torch.testing.assert_close(value, candidate.state_dict()[key], atol=0, rtol=0)
        ids = torch.randint(2048, (1, 256))
        with torch.no_grad():
            torch.testing.assert_close(baseline(ids), ablation(ids), atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), 1088424)
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in candidate.modules()), 9)
        self.assertIs(candidate.token.weight, candidate.head.weight)

    def test_reload_and_all_branch_gradients(self):
        model = build_model(self.config)
        ids = torch.randint(2048, (1, 257))
        loss = nn.functional.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
        loss.backward()
        for block in model.blocks:
            for layer in (block.mlp.gate, block.mlp.value, block.mlp.out):
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        buffer = io.BytesIO()
        torch.save({'config': model.config, 'state': model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = build_model(checkpoint['config'])
        restored.load_state_dict(checkpoint['state'])
        with torch.no_grad():
            torch.testing.assert_close(model.predict_log_probs(ids[:, :-1]),
                                       restored.predict_log_probs(ids[:, :-1]), atol=0, rtol=0)

    def test_reject_invalid_config(self):
        with self.assertRaisesRegex(ValueError, 'Unknown ffn_type'):
            build_model(self.config | {'ffn_type': 'bad'})
        with self.assertRaisesRegex(ValueError, 'requires LayerNorm'):
            build_model(self.config | {'norm_type': 'rmsnorm'})

    def test_fusion_is_only_difference(self):
        torch.manual_seed(17)
        multiply = build_model(self.config).eval()
        torch.manual_seed(17)
        additive = build_model(self.config | {'ffn_type': 'additive'}).eval()
        self.assertEqual(sum(p.numel() for p in additive.parameters()), 1088424)
        for key, value in multiply.state_dict().items():
            torch.testing.assert_close(value, additive.state_dict()[key], atol=0, rtol=0)
        x = torch.randn(2, 16, 128)
        for model, operation in ((multiply, torch.mul), (additive, torch.add)):
            mlp = model.blocks[0].mlp
            expected = mlp.out(operation(nn.functional.gelu(mlp.gate(x)), mlp.value(x)))
            torch.testing.assert_close(mlp(x), expected, atol=0, rtol=0)


class AdditiveContractTests(test_contract.ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(dict(vocab=2048, width=32, heads=4, depth=2,
                                     context=256, ffn_type='additive')).eval()


if __name__ == '__main__':
    unittest.main()
