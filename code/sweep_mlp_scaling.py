"""Serial magnitude MLP scaling: validation-only screen, then fresh top-three reruns."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pinna import config, data, model, train, evaluate, report
from run_experiment import distribution


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def curves(path, history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = [r['step'] for r in history]
    for key in ('train_loss', 'val_loss'):
        axes[0].plot(steps, [r[key] for r in history], label=key)
    axes[1].plot(steps, [r['val_err_mean'] for r in history], label='validation mean')
    axes[0].set_ylabel('Soft-label cross entropy')
    axes[1].set_ylabel('Great-circle error (degrees)')
    for ax in axes:
        ax.set_xlabel('Optimizer step'); ax.legend()
    fig.tight_layout(); fig.savefig(path / 'curves.png', dpi=150); plt.close(fig)


def summarize(root):
    records = []
    for path in sorted(root.glob('*/*/result.json')):
        r = json.loads(path.read_text()); r['directory'] = str(path.parent.relative_to(root)); records.append(r)
    write_json(root / 'summary.json', {'records': records})
    lines = ['# MLP scaling results', '', 'Selection uses validation mean angular error only. Screening never evaluates test data.', '',
             '| Stage | Width | Steps | Seed | Best step | Validation ° | Parameters | Training s | Test ° (report only) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(records, key=lambda r: (r['stage'], r['val_mean'], r['hidden'], r['seed'])):
        test = f"{r['test_mean']:.3f}" if r.get('test_mean') is not None else '—'
        lines.append(f"| {r['stage']} | {r['hidden']} | {r['steps']} | {r['seed']} | {r['best_step']} | {r['val_mean']:.3f} | {r['n_params']} | {r['train_seconds']:.1f} | {test} |")
    confirm = [r for r in records if r['stage'] == 'confirm']
    if confirm:
        lines += ['', '## Confirmation across seeds', '', '| Width | Steps | Seeds completed | Validation mean ± sample SD (°) |', '|---|---:|---:|---:|']
        for width, steps in sorted({(r['hidden'], r['steps']) for r in confirm}):
            vals = [r['val_mean'] for r in confirm if (r['hidden'], r['steps']) == (width, steps)]
            sd = f'{np.std(vals, ddof=1):.3f}' if len(vals) > 1 else '—'
            lines.append(f'| {width} | {steps} | {len(vals)} / 3 | {np.mean(vals):.3f} ± {sd} |')
    (root / 'summary.md').write_text('\n'.join(lines) + '\n')
    return records


def run_one(root, stage, width, steps, seed, ir, vectors, tr, te, az, el, sr, val, smoke):
    path = root / stage / f'w{width}_steps{steps}_s{seed}'
    path.mkdir(parents=True, exist_ok=True)
    if (path / 'result.json').exists():
        return json.loads((path / 'result.json').read_text())
    cfg = train.TrainConfig(hidden=width, steps=steps, seed=seed, feature_mode='magnitude',
                            output_kind='classification', training_objective='baseline', select_by='angular',
                            preroll=4096, batch=8 if smoke else 256, eval_every=5 if smoke else 250,
                            patience=steps + 1)
    write_json(path / 'config.json', asdict(cfg))
    with (path / 'train.log').open('a') as log_file:
        def log(message):
            print(message, flush=True); print(message, file=log_file, flush=True)
        fitted, history, info = train.fit(ir, vectors, tr, cfg, val=val, log=log,
                                          checkpoint=path / 'latest.pt', resume=(path / 'latest.pt').exists())
    classes = vectors[torch.as_tensor(tr, device=config.DEVICE)]
    model.save(path / 'model.pt', fitted, feature_mode='magnitude', class_vectors=classes.cpu(), info=info)
    write_json(path / 'training.json', {'history': history, 'info': info})
    curves(path, history)
    test_mean = None
    if stage == 'confirm':
        # Configurations are frozen in selected.json before any test call.
        result = evaluate.run_battery(train.predictor(fitted, classes), 'seen', ir, az, el, tr, te, sr, quick=True)
        report.save_run(path, result, history, info)
        distribution(result['examples']['core']['err'], path)
        test_mean = result['metrics']['core']['gc_mean']
    result = dict(stage=stage, hidden=width, steps=steps, seed=seed, val_mean=info['best_val_err_mean'],
                  best_step=info['best_step'], n_params=info['n_params'], train_seconds=info['train_seconds'],
                  test_mean=test_mean)
    write_json(path / 'result.json', result)
    summarize(root)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sofa', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    a = p.parse_args()
    if a.summarize_only:
        if not a.output.is_dir(): p.error('Output does not exist')
        summarize(a.output); return
    if not a.sofa or not a.sofa.is_file(): p.error('Provide an existing --sofa')
    if a.require_cuda and not torch.cuda.is_available(): p.error('CUDA unavailable')
    root = a.output
    # Hash implementation and SOFA: reject resume with changed measurements/code.
    code = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for f in sorted(list((code / 'pinna').glob('*.py')) + [Path(__file__).resolve()]):
        digest.update(f.name.encode()); digest.update(f.read_bytes())
    identity = dict(sofa_sha256=hashlib.sha256(a.sofa.read_bytes()).hexdigest(), code_sha256=digest.hexdigest(),
                    smoke=a.smoke, widths=[256, 512, 1024], steps=[3000, 6000, 12000],
                    protocol='seen', preroll=4096, objective='baseline', select_by='angular',
                    torch=str(torch.__version__), device=str(config.DEVICE))
    if a.resume:
        if json.loads((root / 'manifest.json').read_text())['identity'] != identity:
            p.error('Resume identity differs; use a new output directory')
    else:
        root.mkdir(parents=True, exist_ok=False)
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        packages = subprocess.run([os.sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
        (root / 'packages.txt').write_text(packages.stdout or packages.stderr)
        write_json(root / 'manifest.json', dict(identity=identity, git_commit=git.stdout.strip(),
                    git_status=status.stdout, python=os.sys.version, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    slurm_job_id=os.getenv('SLURM_JOB_ID'), started=time.time()))
    ir, az, el, sr = data.load_sofa(a.sofa)
    tr, te = data.split('seen', az, el)
    vectors = torch.as_tensor(data.unit_vec(az, el), dtype=torch.float32, device=config.DEVICE)
    val = train.make_validation_set(ir, tr, train.TrainConfig(preroll=4096))
    write_json(root / 'directions.json', dict(train=tr.tolist(), test=te.tolist(), val_seed=90004))
    if a.smoke:
        run_one(root, 'smoke', 256, 20, 0, ir, vectors, tr, te, az, el, sr, val, True)
        print('SMOKE COMPLETE', flush=True); return
    screen = [run_one(root, 'screen', w, n, 0, ir, vectors, tr, te, az, el, sr, val, False)
              for w in identity['widths'] for n in identity['steps']]
    selected = [(r['hidden'], r['steps']) for r in sorted(screen, key=lambda r: (r['val_mean'], r['n_params'], r['steps']))[:3]]
    selection_path = root / 'selected.json'
    selection = {'metric': 'validation mean angular error', 'configurations': selected}
    if selection_path.exists() and json.loads(selection_path.read_text()) != json.loads(json.dumps(selection)):
        raise ValueError('Frozen selection changed')
    write_json(selection_path, selection)
    for w, n in selected:
        for seed in (0, 1, 2):
            run_one(root, 'confirm', w, n, seed, ir, vectors, tr, te, az, el, sr, val, False)
    print(f'COMPLETE: {root / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
