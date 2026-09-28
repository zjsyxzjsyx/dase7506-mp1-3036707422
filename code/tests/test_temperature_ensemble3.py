import unittest

import torch

from ensemble3 import build_model as build_plain
from temperature_ensemble3 import build_model as build_temperature


class TemperatureEnsembleTests(unittest.TestCase):
    def setUp(self):
        member = {'vocab': 2048, 'width': 32, 'heads': 4, 'depth': 1, 'context': 256}
        self.config = {'vocab': 2048, 'context': 256,
                       'member_configs': [member] * 3,
                       'member_weights': [1/3] * 3}

    def test_unit_temperature_matches_plain_ensemble(self):
        torch.manual_seed(17)
        plain = build_plain(self.config).eval()
        calibrated = build_temperature(dict(self.config, temperature=1.0)).eval()
        calibrated.load_state_dict(plain.state_dict())
        ids = torch.randint(2048, (2, 12))
        with torch.no_grad():
            torch.testing.assert_close(calibrated.predict_log_probs(ids),
                                       plain.predict_log_probs(ids), rtol=0, atol=0)

    def test_calibration_keeps_normalization_and_causality(self):
        calibrated = build_temperature(dict(self.config, temperature=1.05)).eval()
        ids = torch.randint(2048, (1, 12))
        altered = ids.clone()
        altered[:, 6:] = (altered[:, 6:] + 1) % 2048
        with torch.no_grad():
            logp = calibrated.predict_log_probs(ids)
            changed = calibrated.predict_log_probs(altered)
        torch.testing.assert_close(logp.logsumexp(-1), torch.zeros_like(ids, dtype=torch.float), atol=1e-6, rtol=0)
        torch.testing.assert_close(logp[:, :6], changed[:, :6], atol=1e-5, rtol=1e-5)

    def test_invalid_temperatures(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_temperature(dict(self.config, temperature=value))


if __name__ == '__main__':
    unittest.main()
