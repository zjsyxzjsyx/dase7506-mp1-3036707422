"""Check calibrated serial gating preserves probability and causal contracts."""
import unittest

import torch

from gated_ensemble2 import build_model as build_gate
from temperature_gated_ensemble2 import build_model as build_calibrated


class TemperatureGatedEnsemble2Tests(unittest.TestCase):
    def test_unit_temperature_matches_gate_and_future_inputs_do_not_leak(self):
        torch.set_num_threads(2)
        member = {'vocab': 2048, 'width': 32, 'heads': 4, 'depth': 1, 'context': 256}
        config = {'vocab': 2048, 'context': 256,
                  'member_configs': [member] * 2, 'member_weights': [0.5, 0.5],
                  'first_token_max_probability_threshold': 0.5}
        torch.manual_seed(23)
        gate = build_gate(config).eval()
        calibrated = build_calibrated(dict(config, member_temperatures=[1.0, 1.0])).eval()
        calibrated.load_state_dict(gate.state_dict())
        ids = torch.randint(2048, (3, 12))
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 19) % 2048
        with torch.no_grad():
            actual = calibrated.predict_log_probs(ids)
            expected = gate.predict_log_probs(ids)
            future = calibrated.predict_log_probs(changed)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual[:, :7], future[:, :7], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(3, 12), atol=1e-6, rtol=1e-6)

    def test_rejects_invalid_temperatures(self):
        member = {'vocab': 2048, 'width': 32, 'heads': 4, 'depth': 1, 'context': 256}
        config = {'vocab': 2048, 'context': 256,
                  'member_configs': [member] * 2, 'member_weights': [0.5, 0.5],
                  'first_token_max_probability_threshold': 0.16,
                  'member_temperatures': [1.0, float('nan')]}
        with self.assertRaises(ValueError):
            build_calibrated(config)


if __name__ == '__main__':
    unittest.main()
