"""Coarse reference-frequency sweep, choosing by validation rather than test error."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sofa', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--frequencies', nargs='+', type=float, default=[250, 500, 1000, 2000, 4000, 8000])
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--previous-sweep', type=Path, help='Reuse completed seed-0 candidates from a prior sweep')
    p.add_argument('--screen-only', action='store_true', help='Run candidates only, without controls or extra seeds')
    a = p.parse_args()
    previous = []
    if a.previous_sweep:
        old = json.loads((a.previous_sweep/'summary.json').read_text())
        previous = old['candidates']
        digest = hashlib.sha256(a.sofa.read_bytes()).hexdigest()
        for row in previous:
            env = json.loads((a.previous_sweep/Path(row['path']).name/'environment.json').read_text())
            expected = dict(mode='raw_phase', seed=0, steps=a.steps, batch=256,
                            eval_every=250, protocol='seen', preroll=4096, head='classification', lr=.002)
            if env['sofa_sha256'] != digest or any(env['args'].get(k) != v for k,v in expected.items()):
                p.error('Prior candidates must use the same SOFA and training/evaluation settings')
            if env['args'].get('select_by', 'loss') != 'loss':
                p.error('Prior candidates must use validation-loss checkpoint selection')
    a.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('run_experiment.py')
    records = []

    def run(mode, hz, seed):
        folder = a.output/f'{mode}_{hz:g}hz_s{seed}'
        command = [sys.executable, '-u', str(script), '--sofa', str(a.sofa),
                   '--mode', mode, '--phase-reference-hz', str(hz), '--seed', str(seed),
                   '--steps', str(a.steps), '--eval-every', '250', '--batch', '256',
                   '--protocol', 'seen', '--preroll', '4096', '--require-cuda', '--output', str(folder)]
        print(f'START {mode} reference={hz:g} Hz seed={seed}', flush=True)
        with (a.output/f'{folder.name}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        training = json.loads((folder/'training.json').read_text())['info']
        metrics = json.loads((folder/'metrics.json').read_text())['metrics']['core']
        record = dict(mode=mode, requested_reference_hz=hz, seed=seed,
                      val_mean=training['best_val_err_mean'], val_loss=training['best_val_loss'],
                      test_mean=metrics['gc_mean'], test_median=metrics['gc_median'],
                      test_pct_lt5=metrics['pct_lt5'], n_params=training['n_params'], path=str(folder))
        records.append(record)
        (a.output/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(f"DONE {mode} {hz:g}Hz s{seed}: validation={record['val_mean']:.3f} deg "
              f"test={record['test_mean']:.3f} deg", flush=True)
        return record

    # Lock frequency selection before confirmation seeds or encoding comparison.
    existing = {row['requested_reference_hz'] for row in previous}
    candidates = previous + [run('raw_phase', hz, 0) for hz in dict.fromkeys(a.frequencies) if hz not in existing]
    best = min(candidates, key=lambda r: r['val_mean'])
    hz = best['requested_reference_hz']
    selection = dict(selected_reference_hz=hz, criterion='seed-0 validation mean angular error',
                     frequencies_tested=[r['requested_reference_hz'] for r in candidates], candidates=candidates)
    (a.output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    print(f'SELECTED {hz:g} Hz by validation only', flush=True)
    if a.screen_only:
        selection.update(records=records, previous_sweep=str(a.previous_sweep) if a.previous_sweep else None,
                         note='Exploratory seed-0 screening only; no new controls or confirmation seeds.')
        (a.output/'summary.json').write_text(json.dumps(selection, indent=2)+'\n')
        for row in sorted(candidates, key=lambda r: r['requested_reference_hz']):
            print(f"REFERENCE {row['requested_reference_hz']:g}Hz validation={row['val_mean']:.3f} "
                  f"test={row['test_mean']:.3f}", flush=True)
        print(f'COMPLETE screening: selected={hz:g}Hz across {len(candidates)} references', flush=True)
        return
    run('raw_phase_zero', hz, 0)
    for seed in (1, 2):
        run('raw_phase', hz, seed)
    run('raw_phase_sincos', hz, 0)
    selected = [r['test_mean'] for r in records if r['mode']=='raw_phase' and r['requested_reference_hz']==hz]
    selection.update(records=records, selected_test_seed_means=selected,
                     selected_test_mean=float(np.mean(selected)),
                     selected_test_seed_sd=float(np.std(selected, ddof=1)),
                     note='Exploratory seen-direction sweep; not a global optimum or independent final test.')
    (a.output/'summary.json').write_text(json.dumps(selection, indent=2)+'\n')
    print(f'COMPLETE selected={hz:g}Hz seed_means={selected} average={np.mean(selected):.4f}', flush=True)


if __name__ == '__main__':
    main()
