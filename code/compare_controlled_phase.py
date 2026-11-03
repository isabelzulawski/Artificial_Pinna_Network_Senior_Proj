"""Serial five-way phase experiment. Validation selects; test is never used.

Dedicated adapter reuses train.fit without changing old phase modes or results.
All trials see identical wave/noise RNG streams within a seed. Screening uses
its own OneCycle horizon; confirmations start fresh with a 12000-step horizon.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pinna import config, data, model, train
from pinna import controlled_phase as phase


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def run(root, stage, mode, seed, steps, ir, vectors, idx, batch, evaluation):
    out = root / stage / f'{mode}_s{seed}'
    out.mkdir(parents=True, exist_ok=False)
    cfg = train.TrainConfig(feature_mode=mode, hidden=512, depth=2,
        steps=steps, seed=seed, batch=batch, preroll=4096,
        select_by='angular', patience=steps+1, eval_every=evaluation)
    with (out / 'training.log').open('x') as log:
        def emit(s):
            print(s, flush=True); print(s, file=log, flush=True)
        fitted, history, info = train.fit(ir, vectors, idx, cfg,
                                         checkpoint=out/'latest.pt', log=emit)
    cls = vectors[torch.as_tensor(idx, device=config.DEVICE)]
    model.save(out/'model.pt', fitted, feature_mode='controlled_'+mode,
               class_vectors=cls.cpu(), info=info)
    record = dict(stage=stage, mode=mode, seed=seed,
                  validation=info['best_val_err_mean'], parameters=info['n_params'],
                  seconds=info['train_seconds'], best_step=info['best_step'],
                  history=history, snr={})
    # Paired validation diagnostics: same generator seed and same RNG calls at
    # each SNR (tuple rather than scalar). NOT used for model selection.
    if stage == 'confirm':
        for snr in (0, 5, 10, 20, 30, 40):
            val = train.make_validation_set(ir, idx, replace(cfg, snr_db=(snr,snr)))
            _, err = train.evaluate_set(fitted, val['x'], val['dir'], vectors, cls)
            record['snr'][str(snr)] = dict(mean=float(err.mean()), median=float(np.median(err)))
    write(out/'result.json', record)
    return record


def summarize(root, records, selected):
    lines = ['# Controlled magnitude/phase MLP comparison', '',
      'Validation angular error selects all checkpoints and configurations. No test predictions were made.', '',
      '| Stage | Features | Seeds | Validation mean ± seed SD (°) | Parameters | Mean training seconds |',
      '|---|---|---|---|---|---|']
    fig, ax = plt.subplots(figsize=(10,6))
    for stage in ('smoke', 'screen', 'confirm'):
        for mode in phase.MODES:
            rows = [r for r in records if r['stage']==stage and r['mode']==mode]
            if not rows: continue
            values = [r['validation'] for r in rows]
            sd = f'{np.std(values,ddof=1):.3f}' if len(rows)>1 else 'not estimated'
            lines.append(f"| {stage} | {mode} | {','.join(str(r['seed']) for r in rows)} | {np.mean(values):.3f} ± {sd} | {rows[0]['parameters']} | {np.mean([r['seconds'] for r in rows]):.1f} |")
            if stage=='confirm':
                for r in rows:
                    ax.plot([h['step'] for h in r['history']], [h['val_err_mean'] for h in r['history']], label=f'{mode} s{r["seed"]}')
    ax.set(xlabel='Training step', ylabel='Validation mean angular error (°)', ylim=(0,15))
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(root/'validation_curves.png'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for r in records:
        if r['stage']=='confirm':
            for ax, key in zip(axes, ('train_loss','val_loss')):
                ax.plot([h['step'] for h in r['history']], [h[key] for h in r['history']], label=f'{r["mode"]} s{r["seed"]}')
    for ax, title in zip(axes, ('Training CE','Validation CE')): ax.set(xlabel='Step',ylabel=title); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(root/'loss_curves.png'); plt.close(fig)
    lines += ['', '![Validation curves](validation_curves.png)', '', '![Loss curves](loss_curves.png)', '',
              '## Paired SNR validation diagnostics', '', '| Features | SNR dB | Mean across seeds (°) | Seed SD (°) |', '|---|---|---|---|']
    for mode in selected:
        rr = [r for r in records if r['stage']=='confirm' and r['mode']==mode]
        for snr in (0,5,10,20,30,40):
            vals=[r['snr'][str(snr)]['mean'] for r in rr]
            if vals: lines.append(f'| {mode} | {snr} | {np.mean(vals):.3f} | {np.std(vals,ddof=1):.3f} |')
    baseline = {r['seed']:r for r in records if r['stage']=='confirm' and r['mode']=='zero'}
    lines += ['', '## Phase versus matched zero-input control', '']
    for mode in selected:
        if mode in ('zero','magnitude'): continue
        rr=[r for r in records if r['stage']=='confirm' and r['mode']==mode]
        delta=[r['validation']-baseline[r['seed']]['validation'] for r in rr if r['seed'] in baseline]
        if delta:
            lines.append(f'- {mode}: paired validation differences (phase − zero, degrees) {delta}; mean {np.mean(delta):.4f}; wins {sum(d<0 for d in delta)}/{len(delta)}.')
    lines += ['', 'Three seeds describe optimization variability, not independent measurement-session uncertainty. Low-SNR checks reuse validation source/noise realizations and are exploratory, not an untouched test set.', '',
      'Continue phase only if gains versus zero control repeat across seeds and survive new source recordings. Otherwise prioritize denser direction measurements, held-out-direction validation and source/room robustness. Raspberry Pi latency/power have not been measured.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    write(root/'summary.json', dict(records=records, selected=selected))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sofa', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--screen-steps',type=int,default=3000)
    p.add_argument('--steps',type=int,default=12000)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args()
    if a.require_cuda and config.DEVICE.type!='cuda': raise RuntimeError('CUDA required')
    a.output.mkdir(parents=True,exist_ok=False)
    ir,az,el,sr=data.load_sofa(a.sofa)
    idx,_=data.split('seen',az,el)
    vectors=torch.as_tensor(data.unit_vec(az,el),dtype=torch.float32,device=config.DEVICE)
    write(a.output/'manifest.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        sofa_sha256=hashlib.sha256(a.sofa.read_bytes()).hexdigest(),torch=torch.__version__,device=str(config.DEVICE),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), directions=idx.tolist(),
        selection='validation angular error',snr_note='paired validation only; no test tuning'))
    # Process-local adapter leaves all historical feature implementations intact.
    train.extract,train.feature_count=phase.extract_adapter,phase.feature_count
    rows=[]
    counts={m:sum(v.numel() for v in model.DirectionMLP(phase.feature_count(m),len(idx),512,2).parameters()) for m in phase.MODES}
    assert len({counts[m] for m in phase.MODES if m!='magnitude'})==1
    write(a.output/'parameter_counts.json',counts)
    for mode in phase.MODES:
        print(f'START smoke {mode}',flush=True)
        rows.append(run(a.output,'smoke',mode,0,5,ir,vectors,idx,8,5))
        summarize(a.output,rows,[])
    if a.smoke_only: return
    for mode in phase.MODES:
        print(f'START screen {mode}',flush=True)
        rows.append(run(a.output,'screen',mode,0,a.screen_steps,ir,vectors,idx,a.batch,250))
        summarize(a.output,rows,[])
    control=next(r['validation'] for r in rows if r['stage']=='screen' and r['mode']=='zero')
    ranked=sorted([r for r in rows if r['stage']=='screen' and r['mode'] not in ('zero','magnitude')],key=lambda r:r['validation'])
    # Predeclared permissive screen: confirm phase within 10% of zero control;
    # always confirm the best phase even if none pass, to check seed variability.
    promising=[r['mode'] for r in ranked if r['validation']<=1.10*control] or [ranked[0]['mode']]
    selected=['magnitude','zero',*promising]
    write(a.output/'selection.json',dict(selected=selected,ranked=ranked,rule='within 10% of zero or best phase fallback; validation only'))
    for seed in (0,1,2):
        for mode in selected:
            print(f'START confirm {mode} seed={seed}',flush=True)
            rows.append(run(a.output,'confirm',mode,seed,a.steps,ir,vectors,idx,a.batch,250))
            summarize(a.output,rows,selected)
    print(f'COMPLETE {a.output}/report.md',flush=True)


if __name__=='__main__': main()
