"""Select one student's inference temperature on complete validation data."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common import load_data, make_model, setup, sha
from evaluate import score


COARSE = (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25)


class CalibratedStudent(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student
        self.context = student.context
        self.temperature = 1.0

    def predict_log_probs(self, ids):
        return F.log_softmax(self.student(ids).float() / self.temperature, dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    device, precision = setup(args.device, 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['implementation'] != 'model':
        parser.error('Expected one distilled GPT checkpoint')
    student, _ = make_model('model', checkpoint['config'], device)
    student.load_state_dict(checkpoint['model'])
    calibrated = CalibratedStudent(student).eval()
    validation = load_data()['validation']
    rows = []

    def evaluate(temperatures):
        for temperature in temperatures:
            calibrated.temperature = temperature
            result = score(calibrated, *validation, device, precision)
            row = {'temperature': temperature, 'bpb': result['bpb'],
                   'seconds': result['seconds']}
            rows.append(row)
            print(json.dumps(row), flush=True)

    evaluate(COARSE)
    coarse_best = min(rows, key=lambda row: row['bpb'])['temperature']
    fine = tuple(round(coarse_best + delta, 3) for delta in (-0.05, -0.025, 0.025, 0.05)
                 if coarse_best + delta > 0 and
                 round(coarse_best + delta, 3) not in COARSE)
    evaluate(fine)
    best = min(rows, key=lambda row: row['bpb'])
    if not math.isfinite(best['bpb']):
        raise ValueError('Calibration produced a nonfinite score')
    result = {'selection_split': 'validation', 'device': str(device),
              'precision': precision, 'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': sha(args.checkpoint),
              'candidates': rows, 'best': best}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
