"""Train a GPT from scratch with optional soft targets from a frozen local teacher."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--soft-weight', type=float, default=0.0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=9600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=600)
    parser.add_argument('--peak-lr', type=float, default=0.001)
    parser.add_argument('--min-lr-ratio', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--save-checkpoints-from', type=int, default=0,
                        help='Save model-only snapshots at validation steps from this step; 0 disables.')
    args = parser.parse_args()
    if not 0 <= args.soft_weight <= 1:
        parser.error('soft-weight must be in [0, 1]')
    if bool(args.teacher) != (args.soft_weight > 0):
        parser.error('Pass a teacher exactly when soft-weight is positive')
    if args.steps < 1 or args.batch_size < 1 or args.warmup_steps < 1:
        parser.error('steps, batch-size, and warmup-steps must be positive')
    if args.save_checkpoints_from < 0 or (args.save_checkpoints_from and args.eval_every < 1):
        parser.error('Checkpoint snapshots require --eval-every and a nonnegative start step')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Run directory already contains results')
    device, precision = setup(args.device, args.precision, args.threads)
    data = load_data()
    teacher = None
    teacher_hash = None
    if args.teacher:
        source = torch.load(args.teacher, map_location='cpu', weights_only=True)
        if source['protocol'] != PROTOCOL:
            raise ValueError('Teacher protocol differs')
        teacher, _ = make_model(source['implementation'], source['config'], device)
        teacher.load_state_dict(source['model'])
        teacher.eval()
        teacher.requires_grad_(False)
        teacher_hash = sha(args.teacher)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text())
    student, student_source_hash = make_model('model', config, device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.peak_lr, weight_decay=0.1)
    tokens = data['train'][0].to(device)
    sample_rng = torch.Generator().manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    validation_history = []
    validation_seconds = 0.0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=sample_rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        inputs, targets = batch[:, :-1], batch[:, 1:]
        learning_rate = args.peak_lr * min(1.0, (step + 1) / args.warmup_steps) * (
            args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = student(inputs)
            hard_loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
            if teacher is None:
                loss = hard_loss
                soft_loss = None
            else:
                with torch.no_grad():
                    teacher_probs = teacher.predict_log_probs(inputs).float().exp()
                student_log_probs = F.log_softmax(logits.float(), dim=-1)
                soft_loss = -(teacher_probs * student_log_probs).sum(-1).mean()
                loss = (1 - args.soft_weight) * hard_loss + args.soft_weight * soft_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step + 1}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'hard_loss': hard_loss.item(),
                   'soft_loss': None if soft_loss is None else soft_loss.item(),
                   'loss': loss.item(), 'seconds': time.perf_counter() - started - validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0:
            result = score(student, *data['validation'], device, 'fp32')
            result.pop('window_nll_nats')
            validation_seconds += result['seconds']
            validation_history.append({'step': step + 1, **result})
            print(json.dumps({'validation': validation_history[-1]}), flush=True)
            if args.save_checkpoints_from and step + 1 >= args.save_checkpoints_from:
                snapshot_dir = args.run_dir / 'checkpoints'
                snapshot_dir.mkdir(exist_ok=True)
                snapshot = snapshot_dir / f'step{step + 1:05d}.pt'
                torch.save({'protocol': PROTOCOL, 'implementation': 'model',
                            'config': config,
                            'model': {name: value.detach().cpu() for name, value in student.state_dict().items()},
                            'seed': args.seed,
                            'train_tokens': (step + 1) * args.batch_size * 256,
                            'training_step': step + 1,
                            'teacher_checkpoint_sha256': teacher_hash,
                            'soft_weight': args.soft_weight}, snapshot)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - validation_seconds
    final_validation = score(student, *data['validation'], device, 'fp32')
    final_validation.pop('window_nll_nats')
    checkpoint_path = args.run_dir / 'checkpoint.pt'
    torch.save({'protocol': PROTOCOL, 'implementation': 'model', 'config': config,
                'model': student.cpu().state_dict(), 'seed': args.seed,
                'train_tokens': args.steps * args.batch_size * 256,
                'teacher_checkpoint_sha256': teacher_hash,
                'soft_weight': args.soft_weight}, checkpoint_path)
    result = {'protocol': PROTOCOL, 'implementation': 'model', 'config': config,
              'seed': args.seed, 'parameters': sum(p.numel() for p in student.parameters()),
              'steps': args.steps, 'batch_size': args.batch_size,
              'train_tokens': args.steps * args.batch_size * 256,
              'peak_lr': args.peak_lr, 'min_lr_ratio': args.min_lr_ratio,
              'warmup_steps': args.warmup_steps, 'soft_weight': args.soft_weight,
              'save_checkpoints_from': args.save_checkpoints_from,
              'teacher_checkpoint_sha256': teacher_hash,
              'precision': precision, 'train_seconds': train_seconds,
              'validation': final_validation, 'validation_history': validation_history,
              'history': history, 'checkpoint_sha256': sha(checkpoint_path),
              'student_implementation_sha256': student_source_hash,
              **device_metrics(device)}
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'complete': True, 'validation_bpb': final_validation['bpb'],
                      'train_seconds': train_seconds, 'checkpoint': str(checkpoint_path)}), flush=True)


if __name__ == '__main__':
    main()
