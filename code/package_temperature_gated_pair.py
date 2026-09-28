"""Repackage a frozen pair with validation-selected temperatures and gate."""
import argparse
import math
from pathlib import Path

import torch

from common import PROTOCOL, sha
from temperature_gated_ensemble2 import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--temperature-first', type=float, required=True)
    parser.add_argument('--temperature-second', type=float, required=True)
    parser.add_argument('--threshold', type=float, default=0.16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output checkpoint already exists')
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.temperature_first, args.temperature_second)):
        parser.error('Temperatures must be finite and positive')
    if not 0 < args.threshold <= 1:
        parser.error('Threshold must be in (0, 1]')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if source['protocol'] != PROTOCOL or source['implementation'] != 'ensemble2':
        parser.error('Expected a frozen two-student checkpoint')
    config = dict(source['config'], member_temperatures=[args.temperature_first,
                                                         args.temperature_second],
                  first_token_max_probability_threshold=args.threshold)
    model = build_model(config)
    model.load_state_dict(source['model'])
    packaged = dict(source)
    packaged['implementation'] = 'temperature_gated_ensemble2'
    packaged['config'] = config
    packaged['parent_checkpoint_sha256'] = sha(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packaged, args.output)
    print(f'Wrote {args.output} ({args.output.stat().st_size / 2**20:.3f} MiB)')


if __name__ == '__main__':
    main()
