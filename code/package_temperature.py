"""Package an already-trained three-member checkpoint with a fixed temperature."""
import argparse
import math
from pathlib import Path

import torch

from common import PROTOCOL, sha
from temperature_ensemble3 import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output checkpoint already exists')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('Temperature must be finite and positive')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if source['protocol'] != PROTOCOL or source['implementation'] != 'ensemble3':
        parser.error('Expected an ensemble3 checkpoint for this protocol')
    config = dict(source['config'], temperature=args.temperature)
    model = build_model(config)
    model.load_state_dict(source['model'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'protocol': PROTOCOL, 'implementation': 'temperature_ensemble3',
                'config': config, 'model': model.state_dict(),
                'parent_checkpoint_sha256': sha(args.checkpoint),
                'component_seeds': source['component_seeds'],
                'component_train_tokens': source['component_train_tokens']}, args.output)
    print(f'Wrote {args.output} ({args.output.stat().st_size / 2**20:.3f} MiB)')


if __name__ == '__main__':
    main()
