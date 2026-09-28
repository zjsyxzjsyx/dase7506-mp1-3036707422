"""Check dropout changes training but preserves the original inference model."""
import unittest

import torch

from dropout_model import build_model
from model import GPT


class DropoutModelTests(unittest.TestCase):
    def test_initial_parameters_and_eval_outputs_match_base(self):
        config = dict(vocab=2048, width=32, heads=4, depth=2, context=256)
        torch.manual_seed(17)
        original = GPT(config).eval()
        torch.manual_seed(17)
        dropped = build_model(dict(config, dropout_p=0.05)).eval()
        for name, parameter in original.state_dict().items():
            torch.testing.assert_close(parameter, dropped.state_dict()[name], rtol=0, atol=0)
        ids = torch.randint(0, 2048, (2, 13))
        with torch.no_grad():
            torch.testing.assert_close(original(ids), dropped(ids), rtol=0, atol=0)

    def test_dropout_is_active_only_in_training(self):
        model = build_model(dict(vocab=2048, width=32, heads=4, depth=2,
                                 context=256, dropout_p=0.5))
        ids = torch.randint(0, 2048, (2, 13))
        model.train()
        first, second = model(ids), model(ids)
        self.assertFalse(torch.equal(first, second))
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(ids), model(ids), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
