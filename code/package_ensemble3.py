"""Package three GPT checkpoints, including training-only dropout variants."""
import argparse
from pathlib import Path

import torch

from common import PROTOCOL, sha
from ensemble3 import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--member', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.member) != 3:
        parser.error('Pass exactly three --member checkpoints')
    if args.output.exists():
        parser.error('Output checkpoint already exists')
    checkpoints = [torch.load(path, map_location='cpu', weights_only=True)
                   for path in args.member]
    for checkpoint in checkpoints:
        if (checkpoint['protocol'] != PROTOCOL
                or checkpoint['implementation'] not in ('model', 'dropout_model')):
            raise ValueError('Every member must use GPT weights for this protocol')
    member_configs = [checkpoint['config'] for checkpoint in checkpoints]
    config = {'vocab': 2048, 'context': 256, 'member_configs': member_configs,
              'member_weights': [1 / 3] * 3}
    model = build_model(config)
    for member, checkpoint in zip(model.members, checkpoints):
        member.load_state_dict(checkpoint['model'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'protocol': PROTOCOL, 'implementation': 'ensemble3', 'config': config,
                'model': model.state_dict(),
                'component_seeds': [checkpoint['seed'] for checkpoint in checkpoints],
                'component_train_tokens': [checkpoint['train_tokens'] for checkpoint in checkpoints],
                'parent_checkpoint_sha256': [sha(path) for path in args.member]}, args.output)
    print(f'Wrote {args.output} ({args.output.stat().st_size / 2**20:.3f} MiB)')


if __name__ == '__main__':
    main()
