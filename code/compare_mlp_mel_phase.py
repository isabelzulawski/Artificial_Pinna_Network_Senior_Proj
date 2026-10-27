"""Small MLP pilot: original-style direct phase channels, pooled to 128 inputs."""
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
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--steps', type=int, default=3000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('run_experiment.py')
    records, reference = [], None
    for seed in a.seeds:
        for mode in ('mlp_mel_zero', 'mlp_mel_phase'):
            directory = a.output/f'{mode}_s{seed}'
            command = [sys.executable, '-u', str(script), '--sofa', str(a.sofa), '--output', str(directory),
                       '--mode', mode, '--hidden', '232', '--select-by', 'angular', '--steps', str(a.steps),
                       '--batch', '256', '--eval-every', '250', '--patience', str(a.steps//250+2),
                       '--preroll', '4096', '--seed', str(seed), '--protocol', 'seen', '--require-cuda']
            print(f'START MLP {mode} seed={seed} hidden=232', flush=True)
            with (a.output/f'{directory.name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            info = json.loads((directory/'training.json').read_text())['info']
            metrics = json.loads((directory/'metrics.json').read_text())['metrics']['core']
            if info['n_params'] != 276190:
                raise ValueError('Expected lightweight 276190-parameter MLP')
            with np.load(directory/'examples.npz') as ex:
                arrays = {k: ex[f'core/{k}'] for k in ('source', 'true_idx')}
            if reference is None:
                reference = arrays
            if any(not np.array_equal(reference[k], arrays[k]) for k in reference):
                raise ValueError('Test examples do not align')
            row = dict(mode=mode, seed=seed, val_mean=info['best_val_err_mean'],
                       test_mean=metrics['gc_mean'], test_median=metrics['gc_median'],
                       test_p95=metrics['gc_p95'], pct_lt5=metrics['pct_lt5'],
                       by_source=metrics['by_source'], n_params=info['n_params'])
            records.append(row)
            (a.output/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
            print(f"DONE MLP {mode} s{seed}: validation={row['val_mean']:.3f} test={row['test_mean']:.3f}", flush=True)
    (a.output/'summary.json').write_text(json.dumps(dict(records=records), indent=2)+'\n')
    print('COMPLETE MLP mel-phase pilot; no CNN trained', flush=True)


if __name__ == '__main__':
    main()
