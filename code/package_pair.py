"""Package two independently trained GPT checkpoints as an equal-weight ensemble."""
import argparse
from pathlib import Path

import torch

from common import PROTOCOL, sha
from ensemble2 import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True, type=Path)
    parser.add_argument('--second', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output checkpoint already exists')
    paths = (args.first, args.second)
    checkpoints = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    for checkpoint in checkpoints:
        if checkpoint['protocol'] != PROTOCOL or checkpoint['implementation'] not in ('model', 'dropout_model'):
            raise ValueError('Both checkpoints must contain GPT weights for this protocol')
    config = {'vocab': 2048, 'context': 256,
              'member_configs': [checkpoint['config'] for checkpoint in checkpoints],
              'member_weights': [0.5, 0.5]}
    model = build_model(config)
    for member, checkpoint in zip(model.members, checkpoints):
        member.load_state_dict(checkpoint['model'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'protocol': PROTOCOL, 'implementation': 'ensemble2',
                'config': config, 'model': model.state_dict(),
                'component_seeds': [checkpoint['seed'] for checkpoint in checkpoints],
                'component_train_tokens': [checkpoint['train_tokens'] for checkpoint in checkpoints],
                'parent_checkpoint_sha256': [sha(path) for path in paths]}, args.output)
    print(f'Wrote {args.output} ({args.output.stat().st_size / 2**20:.3f} MiB)')


if __name__ == '__main__':
    main()
