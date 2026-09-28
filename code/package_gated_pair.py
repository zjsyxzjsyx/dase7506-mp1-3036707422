"""Repackage frozen student weights for serial confidence-gated inference."""
import argparse
from pathlib import Path

import torch

from common import PROTOCOL, sha
from ensemble2 import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output checkpoint already exists')
    if not 0 < args.threshold <= 1:
        parser.error('threshold must be in (0, 1]')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if source['protocol'] != PROTOCOL or source['implementation'] != 'ensemble2':
        parser.error('Expected a frozen two-member FP32 checkpoint')
    build_model(source['config']).load_state_dict(source['model'])
    packaged = dict(source)
    packaged['implementation'] = 'gated_ensemble2'
    packaged['config'] = dict(source['config'], first_token_max_probability_threshold=args.threshold)
    packaged['parent_checkpoint_sha256'] = sha(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packaged, args.output)
    print(f'Wrote {args.output} ({args.output.stat().st_size / 2**20:.3f} MiB)')


if __name__ == '__main__':
    main()
