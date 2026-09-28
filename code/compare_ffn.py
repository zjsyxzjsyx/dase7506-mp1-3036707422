"""Paired GELU/GEGLU CUDA FP32 training, scored in a separate CPU environment."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
METHODS = ('gelu', 'geglu')


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--cpu-python', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--methods', nargs=2, choices=('gelu', 'geglu', 'additive'), default=METHODS)
    parser.add_argument('--both-seeds', action='store_true', help='Run both seeds regardless of seed 17 results.')
    parser.add_argument('--reference-run-dir', type=Path, help='Also require the candidate to improve over this shorter-budget experiment.')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive.')
    if args.methods[0] == args.methods[1]:
        parser.error('Choose two different methods.')
    methods = args.methods
    if args.reference_run_dir:
        args.reference_run_dir = args.reference_run_dir.resolve()
        for seed in (17, 23):
            for method in methods:
                reference = args.reference_run_dir / f'{method}-s{seed}'
                if not (reference / 'validation_repeat1.json').is_file():
                    parser.error(f'Missing reference score: {reference}')
                if read(reference / 'metrics.json')['train_tokens'] >= args.steps * 32 * 256:
                    parser.error('Reference training budget must be smaller.')
    args.cpu_python = args.cpu_python.resolve()
    if not args.cpu_python.is_file():
        parser.error('CPU Python executable does not exist.')
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = output / 'comparison.md'

    def note(message):
        with journal.open('a', encoding='utf-8') as handle:
            handle.write(message + '\n')

    def run(label, arguments, python=None):
        command = [str(python or sys.executable), '-u', *map(str, arguments)]
        note(f'\n### {label}\n\n```powershell\n& {subprocess.list2cmdline(command)}\n```')
        started = time.perf_counter()
        with (output / f'{label}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        note(f'\nExit: {result.returncode}; wall time: {elapsed:.2f} s; log: [{label}.log]({label}.log).')
        print(f'{label}: exit={result.returncode}, seconds={elapsed:.1f}', flush=True)
        if result.returncode:
            raise RuntimeError(f'{label} failed; see its log.')

    def paired(seed):
        for method in methods:
            run(f'{method}-s{seed}-train', ['train.py', '--implementation', 'geglu', '--config',
                ROOT / 'configs' / f'ffn_{method}.json', '--device', 'cuda', '--precision', 'fp32',
                '--threads', '4', '--seed', seed, '--steps', args.steps, '--batch-size', '32',
                '--eval-every', '300', '--run-dir', output / f'{method}-s{seed}'])
        # Alternating fresh scoring processes, with no concurrent training.
        for repeat in range(1, 4):
            for method in methods:
                directory = output / f'{method}-s{seed}'
                run(f'{method}-s{seed}-score{repeat}', ['baseline_pipeline.py', 'measure',
                    '--checkpoint', directory / 'checkpoint.pt', '--output',
                    directory / f'validation_repeat{repeat}.json'], python=args.cpu_python)
        rows = {}
        for method in methods:
            directory = output / f'{method}-s{seed}'
            metrics = read(directory / 'metrics.json')
            scores = [read(directory / f'validation_repeat{i}.json') for i in range(1, 4)]
            resources = [read(directory / f'validation_repeat{i}.resources.json') for i in range(1, 4)]
            if any(abs(s['bpb'] - metrics['validation']['bpb']) > 1e-6 for s in scores):
                raise RuntimeError('Reloaded validation BPB mismatch')
            if metrics['train_tokens'] != args.steps * 32 * 256:
                raise RuntimeError('Unexpected training target count')
            rows[method] = {'bpb': scores[0]['bpb'],
                'score_seconds': statistics.median(s['seconds'] for s in scores),
                'peak_gib': max(r['peak_process_rss_bytes'] for r in resources) / 2**30,
                'train_seconds': metrics['train_seconds']}
        note(f'\n## Seed {seed} results\n\n| Method | Validation BPB | Train s | Median score s | Peak GiB |\n|---|---:|---:|---:|---:|')
        for method, row in rows.items():
            note(f"| {method} | {row['bpb']:.8f} | {row['train_seconds']:.2f} | {row['score_seconds']:.3f} | {row['peak_gib']:.3f} |")
        delta = rows[methods[1]]['bpb'] - rows[methods[0]]['bpb']
        ratio = rows[methods[1]]['score_seconds'] / rows[methods[0]]['score_seconds']
        note(f'\n{methods[1]} minus {methods[0]} BPB: {delta:+.8f}; CPU time ratio: {ratio:.3f}.')
        print(f'seed={seed}, BPB delta={delta:+.8f}, CPU ratio={ratio:.3f}', flush=True)
        budget_improved = True
        if args.reference_run_dir:
            for method in methods:
                reference_score = read(args.reference_run_dir / f'{method}-s{seed}' / 'validation_repeat1.json')
                change = rows[method]['bpb'] - reference_score['bpb']
                note(f'\n{method} BPB change versus shorter budget: {change:+.8f}.')
                if method == methods[1]:
                    budget_improved = change < 0
        return delta < 0 and ratio <= 5 and budget_improved

    note('# GPU feed-forward comparison\n\nStarted: ' + time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    note(f'\nCUDA FP32 training; CPU FP32 scoring, 4 threads, {args.steps} updates, batch 32, validation every 300 updates. '
         'Each run starts from random initialization; no parent checkpoint. Compare final checkpoints only. No test scoring.')
    note(f'\nCompared methods: {methods[0]}, {methods[1]}. ' +
         ('Both seeds 17 and 23 run unconditionally.' if args.both_seeds else
          'Seed 23 runs only if the second method improves seed 17 BPB within 5x CPU time.'))
    note('\nThe original trainer rescales its cosine learning-rate schedule to the requested total steps; '
         'warmup remains 100 steps. This is not a continuation of a shorter checkpoint.')
    if args.reference_run_dir:
        note(f'\nShorter-budget reference: `{args.reference_run_dir}`. '
             'The second method must also improve over its same-seed shorter-budget BPB to pass the screen.')
    protected = ('model.py', 'common.py', 'evaluate.py', 'train.py', 'configs/baseline.json')
    sources = (*protected, 'geglu.py', 'compare_ffn.py', 'baseline_pipeline.py',
               *(f'configs/ffn_{method}.json' for method in methods))
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    release = read(ROOT / 'PACKAGE_MANIFEST.json')
    if any(hashes[p] != release['code/' + p] for p in protected):
        raise RuntimeError('Original trainer/model/scorer changed')
    note('\n## Source hashes\n\n' + '\n'.join(f'- `{p}`: `{h}`' for p, h in hashes.items()))
    try:
        first_passed = paired(17)
        if first_passed or args.both_seeds:
            confirmed = paired(23)
            note('\nBoth seeds meet the BPB/time screen.' if first_passed and confirmed else
                 '\nThe second method does not meet the BPB/time screen on both seeds; inspect the paired results.')
        else:
            note('\nSeed 17 does not meet the BPB/time screen; seed 23 was not run, per the predefined rule.')
        if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
            raise RuntimeError('Source changed during the comparison')
        note('\nStatus: complete. Resource measurements are validation-stage evidence, not final test compliance.')
    except BaseException as error:
        note(f'\nStatus: failed. {error}')
        raise


if __name__ == '__main__':
    main()
