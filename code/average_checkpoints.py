"""Average checkpoints from one training trajectory into one inference model."""
import argparse
from pathlib import Path

import torch

from common import PROTOCOL, sha


def average_checkpoints(paths, method, decay):
    checkpoints = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    first = checkpoints[0]
    steps = [checkpoint['training_step'] for checkpoint in checkpoints]
    if steps != sorted(set(steps)):
        raise ValueError('Input snapshots must have unique, ascending training steps')
    for checkpoint in checkpoints:
        if (checkpoint['protocol'] != PROTOCOL
                or checkpoint['implementation'] != first['implementation']
                or checkpoint['config'] != first['config']
                or checkpoint['seed'] != first['seed']
                or checkpoint.get('teacher_checkpoint_sha256') != first.get('teacher_checkpoint_sha256')
                or checkpoint.get('soft_weight') != first.get('soft_weight')):
            raise ValueError('Snapshots must come from the same model and seed')
        if checkpoint['model'].keys() != first['model'].keys():
            raise ValueError('Snapshot parameters differ')
    result = {}
    for name in first['model']:
        values = [checkpoint['model'][name] for checkpoint in checkpoints]
        if not torch.is_floating_point(values[0]):
            if any(not torch.equal(values[0], value) for value in values[1:]):
                raise ValueError(f'Non-floating buffer differs: {name}')
            result[name] = values[0]
            continue
        values = [value.double() for value in values]
        if method == 'uniform':
            merged = torch.stack(values).mean(0)
        else:
            merged = values[0]
            for value in values[1:]:
                merged = decay * merged + (1 - decay) * value
        result[name] = merged.to(first['model'][name].dtype)
    if ('token.weight' in result and 'head.weight' in result
            and torch.equal(result['token.weight'], result['head.weight'])):
        result['head.weight'] = result['token.weight']
    averaged = {'protocol': PROTOCOL, 'implementation': first['implementation'],
            'config': first['config'], 'model': result, 'seed': first['seed'],
            'train_tokens': checkpoints[-1]['train_tokens'],
            'parent_checkpoint_sha256': [sha(path) for path in paths],
            'source_training_steps': steps, 'averaging_method': method,
            'averaging_decay': decay if method == 'ema' else None}
    for key in ('teacher_checkpoint_sha256', 'soft_weight'):
        if key in first:
            averaged[key] = first[key]
    return averaged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=('uniform', 'ema'), required=True)
    parser.add_argument('--decay', type=float, default=0.7)
    args = parser.parse_args()
    if len(args.input) < 2 or not 0 < args.decay < 1:
        parser.error('Need at least two inputs and 0 < decay < 1')
    if args.output.exists():
        parser.error('Output checkpoint already exists')
    averaged = average_checkpoints(args.input, args.method, args.decay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, args.output)
    print(f'Wrote {args.output} ({args.output.stat().st_size / 2**20:.3f} MiB)')


if __name__ == '__main__':
    main()
