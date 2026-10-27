"""Controlled magnitude-MLP training experiment; no larger inference model."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sofa', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--steps', type=int, default=3000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('run_experiment.py')
    conditions = ('baseline', 'paired_control', 'source_contrastive', 'angular')
    rows, reference_examples = [], None
    for seed in range(3):
        for objective in conditions:
            directory = a.output/f'{objective}_s{seed}'
            command = [sys.executable, '-u', str(script), '--sofa', str(a.sofa), '--output', str(directory),
                       '--mode', 'magnitude', '--training-objective', objective, '--select-by', 'angular',
                       '--contrastive-weight', '.1', '--angular-weight', '1.0',
                       '--steps', str(a.steps), '--batch', '256', '--eval-every', '250',
                       '--preroll', '4096', '--seed', str(seed), '--protocol', 'seen', '--require-cuda']
            print(f'START {objective} seed={seed}', flush=True)
            with (a.output/f'{directory.name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            info = json.loads((directory/'training.json').read_text())['info']
            metrics = json.loads((directory/'metrics.json').read_text())['metrics']['core']
            with np.load(directory/'examples.npz') as ex:
                examples = {k: ex[f'core/{k}'] for k in ('true_idx', 'source')}
            if reference_examples is None:
                reference_examples = examples
            else:
                for k in examples:
                    if not np.array_equal(examples[k], reference_examples[k]):
                        raise ValueError('Evaluation examples do not align')
            if info['n_params'] != 277750:
                raise ValueError('Expected unchanged 277750-parameter MLP')
            row = dict(objective=objective, seed=seed, val_mean=info['best_val_err_mean'],
                       test_mean=metrics['gc_mean'], test_median=metrics['gc_median'],
                       test_p95=metrics['gc_p95'], pct_lt5=metrics['pct_lt5'],
                       pct_gt45=metrics['pct_gt45'], by_source=metrics['by_source'],
                       n_params=info['n_params'], best_step=info['best_step'], path=str(directory))
            rows.append(row)
            (a.output/'progress.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(f"DONE {objective} s{seed}: validation={row['val_mean']:.3f} deg "
                  f"test={row['test_mean']:.3f} deg median={row['test_median']:.3f} deg", flush=True)
    summary = {}
    for objective in conditions:
        chosen = [r for r in rows if r['objective']==objective]
        means = [r['test_mean'] for r in chosen]
        summary[objective] = dict(seed_test_means=means, mean=float(np.mean(means)),
                                  seed_sd=float(np.std(means, ddof=1)),
                                  val_mean=float(np.mean([r['val_mean'] for r in chosen])))
    (a.output/'summary.json').write_text(json.dumps(dict(summary=summary, runs=rows), indent=2)+'\n')
    for objective, result in summary.items():
        print(f"SUMMARY {objective}: mean={result['mean']:.4f} seed_sd={result['seed_sd']:.4f}", flush=True)
    print('COMPLETE: all models use 277750 parameters and identical inference code', flush=True)


if __name__ == '__main__':
    main()
