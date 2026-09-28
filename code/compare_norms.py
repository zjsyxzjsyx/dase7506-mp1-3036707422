"""Paired LayerNorm/RMSNorm runs; validation only, with a Markdown run record."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
METHODS = ('layernorm', 'rmsnorm')


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = output / 'comparison.md'

    def note(message):
        with journal.open('a', encoding='utf-8') as handle:
            handle.write(message + '\n')

    def run(label, arguments):
        command = [sys.executable, '-u', *map(str, arguments)]
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
        for method in METHODS:
            run(f'{method}-s{seed}-train', ['train.py', '--implementation', 'student', '--config',
                ROOT / 'configs' / f'norm_{method}.json', '--device', 'cpu', '--precision', 'fp32',
                '--threads', '4', '--seed', seed, '--steps', '1200', '--batch-size', '32',
                '--eval-every', '300', '--run-dir', output / f'{method}-s{seed}'])
        # Alternating fresh scoring processes, with no concurrent training.
        for repeat in range(1, 4):
            for method in METHODS:
                directory = output / f'{method}-s{seed}'
                run(f'{method}-s{seed}-score{repeat}', ['baseline_pipeline.py', 'measure',
                    '--checkpoint', directory / 'checkpoint.pt', '--output',
                    directory / f'validation_repeat{repeat}.json'])
        rows = {}
        for method in METHODS:
            directory = output / f'{method}-s{seed}'
            metrics = read(directory / 'metrics.json')
            scores = [read(directory / f'validation_repeat{i}.json') for i in range(1, 4)]
            resources = [read(directory / f'validation_repeat{i}.resources.json') for i in range(1, 4)]
            if any(abs(s['bpb'] - metrics['validation']['bpb']) > 1e-6 for s in scores):
                raise RuntimeError('Reloaded validation BPB mismatch')
            if metrics['train_tokens'] != 1200 * 32 * 256:
                raise RuntimeError('Unexpected training target count')
            rows[method] = {'bpb': scores[0]['bpb'],
                'score_seconds': statistics.median(s['seconds'] for s in scores),
                'peak_gib': max(r['peak_process_rss_bytes'] for r in resources) / 2**30,
                'train_seconds': metrics['train_seconds']}
        note(f'\n## Seed {seed} results\n\n| Method | Validation BPB | Train s | Median score s | Peak GiB |\n|---|---:|---:|---:|---:|')
        for method, row in rows.items():
            note(f"| {method} | {row['bpb']:.8f} | {row['train_seconds']:.2f} | {row['score_seconds']:.3f} | {row['peak_gib']:.3f} |")
        delta = rows['rmsnorm']['bpb'] - rows['layernorm']['bpb']
        ratio = rows['rmsnorm']['score_seconds'] / rows['layernorm']['score_seconds']
        note(f'\nRMSNorm minus LayerNorm BPB: {delta:+.8f}; CPU time ratio: {ratio:.3f}.')
        print(f'seed={seed}, BPB delta={delta:+.8f}, CPU ratio={ratio:.3f}', flush=True)
        return delta < 0 and ratio <= 5

    note('# Normalization comparison\n\nStarted: ' + time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    note('\nCPU FP32, 4 threads, 1200 updates, batch 32, validation every 300 updates. '
         'Each run starts from random initialization; no parent checkpoint. Compare final checkpoints only. '
         'Seed 23 is run only if seed 17 improves BPB and CPU time is at most 5x. No test scoring.')
    protected = ('model.py', 'common.py', 'evaluate.py', 'train.py', 'configs/baseline.json')
    sources = (*protected, 'student.py', 'compare_norms.py', 'baseline_pipeline.py',
               'configs/norm_layernorm.json', 'configs/norm_rmsnorm.json')
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    release = read(ROOT / 'PACKAGE_MANIFEST.json')
    if any(hashes[p] != release['code/' + p] for p in protected):
        raise RuntimeError('Original trainer/model/scorer changed')
    note('\n## Source hashes\n\n' + '\n'.join(f'- `{p}`: `{h}`' for p, h in hashes.items()))
    try:
        if paired(17):
            confirmed = paired(23)
            note('\nBoth seeds meet the BPB/time screen.' if confirmed else '\nSeed 23 does not confirm the seed 17 screen; benefit is not consistent.')
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
