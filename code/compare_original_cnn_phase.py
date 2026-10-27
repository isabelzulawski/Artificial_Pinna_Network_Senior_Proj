"""Matched phase-channel ablation for the earlier PinnaResNet architecture."""
import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pinna import config, data, evaluate, report, simulate
from pinna.original_cnn import PinnaResNet
from pinna.original_features import original_features
from run_experiment import distribution


def phase_input(features, enabled):
    if enabled:
        return features
    result = features.clone()
    result[:, 2:] = 0
    return result


@torch.no_grad()
def validation_features(ir, vectors, directions, chunk=16):
    """Shared CPU feature cache for all modes/seeds; ~1 GiB on the full grid."""
    gen = torch.Generator().manual_seed(90004)
    features, targets = [], []
    for source in config.SOURCE_TYPES:
        for start in range(0, len(directions), chunk):
            idx = directions[start:start+chunk]
            w = simulate.make_batch(ir, idx, [source]*len(idx), gen, (10., 40.), preroll=4096)
            features.append(original_features(w).cpu())
            targets.append(vectors[torch.as_tensor(idx, device=config.DEVICE)].cpu())
        print(f'Validation source {source}: {len(directions)} examples cached', flush=True)
    return torch.cat(features), torch.cat(targets)


@torch.no_grad()
def validate(model, validation, enabled, chunk=16):
    model.eval()
    errors = []
    for start in range(0, len(validation[0]), chunk):
        x = phase_input(validation[0][start:start+chunk].to(config.DEVICE), enabled)
        target = validation[1][start:start+chunk].to(config.DEVICE)
        pred = model(x)
        errors.append(torch.rad2deg(torch.acos(F.cosine_similarity(pred, target).clamp(-1, 1))).cpu().numpy())
    model.train()
    return float(np.concatenate(errors).mean())


def fit(ir, vectors, directions, validation, enabled, seed, steps, batch, eval_every, log=print):
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=config.DEVICE).manual_seed(1000+seed)
    model = PinnaResNet(in_ch=4, dropout=.3).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, .001, total_steps=steps, pct_start=.1)
    best, best_state, best_step = float('inf'), None, 0
    history, train_loss, count = [], 0., 0
    start = time.time()
    for step in range(1, steps+1):
        idx = rng.choice(directions, batch)
        kinds = rng.choice(config.SOURCE_TYPES, batch)
        w = simulate.make_batch(ir, idx, kinds, gen, (10., 40.), preroll=4096)
        x = original_features(w, include_phase=enabled)
        target = vectors[torch.as_tensor(idx, device=config.DEVICE)]
        model.train()
        prediction = model(x)
        loss = (1-F.cosine_similarity(prediction, target, dim=-1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        train_loss, count = train_loss+float(loss.detach()), count+1
        if step % eval_every == 0 or step == steps:
            error = validate(model, validation, enabled)
            history.append(dict(step=step, train_loss=train_loss/count, val_err_mean=error,
                                seconds=round(time.time()-start, 1)))
            train_loss, count = 0., 0
            if error < best-1e-4:
                best, best_step, best_state = error, step, copy.deepcopy(model.state_dict())
            log(f'step {step} validation={error:.3f} deg best={best:.3f} elapsed={time.time()-start:.0f}s', flush=True)
    model.load_state_dict(best_state)
    info = dict(best_step=best_step, best_val_err_mean=best, n_params=sum(p.numel() for p in model.parameters()),
                train_seconds=round(time.time()-start, 1), include_phase=enabled, seed=seed,
                steps=steps, batch=batch, selection_metric='val_err_mean')
    return model.eval(), history, info


def predictor(model, enabled, chunk=16):
    @torch.no_grad()
    def predict(w):
        predictions = [model(original_features(w[start:start+chunk], include_phase=enabled))
                       for start in range(0, len(w), chunk)]
        return torch.cat(predictions), None
    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sofa', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    a = p.parse_args()
    if not torch.cuda.is_available():
        p.error('CUDA required; refusing accidental CPU CNN training')
    if min(a.steps, a.batch, a.eval_every) < 1:
        p.error('Steps, batch and eval-every must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    ir, az, el, sr = data.load_sofa(a.sofa)
    directions = np.arange(len(az))
    vectors = torch.as_tensor(data.unit_vec(az, el), dtype=torch.float32, device=config.DEVICE)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    environment = dict(sofa_sha256=hashlib.sha256(a.sofa.read_bytes()).hexdigest(), git_commit=git.stdout.strip(),
                       args={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                       python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                       source_types=config.SOURCE_TYPES, protocol='seen', shared_validation_seed=90004,
                       train_snr_db=[10, 40], test_snr_db=30, preroll=4096, selection='validation angular error',
                       notebook_training_reproduced=False)
    (a.output/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    print('Building shared validation features', flush=True)
    validation = validation_features(ir, vectors, directions)
    print(f'Validation cached: {len(validation[0])} examples', flush=True)
    records, reference_examples, expected_count = [], None, None
    for seed in a.seeds:
        for enabled in (False, True):
            name = 'phase' if enabled else 'phase_zero'
            directory = a.output/f'{name}_s{seed}'
            directory.mkdir()
            print(f'START original_cnn {name} seed={seed}', flush=True)
            with (directory/'training.log').open('w') as log:
                fitted, history, info = fit(ir, vectors, directions, validation, enabled, seed,
                                           a.steps, a.batch, a.eval_every,
                                           log=lambda *args, **kwargs: print(*args, file=log, **kwargs))
            if expected_count is None:
                expected_count = info['n_params']
            assert info['n_params'] == expected_count
            torch.save(dict(state_dict=fitted.state_dict(), architecture='original_pinnaresnet',
                            model_config=dict(in_ch=4, dropout=.3), info=info, environment=environment),
                       directory/'model.pt')
            result = evaluate.run_battery(predictor(fitted, enabled), 'seen', ir, az, el,
                                          directions, directions, sr, quick=True)
            ex = result['examples']['core']
            if reference_examples is None:
                reference_examples = {k: ex[k] for k in ('true_idx', 'source')}
            for k in reference_examples:
                assert np.array_equal(reference_examples[k], ex[k]), 'Unpaired evaluation examples'
            report.save_run(directory, result, history, info)
            distribution(ex['err'], directory)
            m = result['metrics']['core']
            row = dict(mode=name, seed=seed, val_mean=info['best_val_err_mean'], test_mean=m['gc_mean'],
                       test_median=m['gc_median'], test_p95=m['gc_p95'], pct_lt5=m['pct_lt5'],
                       pct_gt45=m['pct_gt45'], by_source=m['by_source'], n_params=info['n_params'])
            records.append(row)
            (a.output/'progress.json').write_text(json.dumps(records, indent=2)+'\n')
            print(f"DONE {name} s{seed}: validation={row['val_mean']:.3f} test={row['test_mean']:.3f}", flush=True)
            del fitted
            torch.cuda.empty_cache()
    summary = {}
    for name in ('phase_zero', 'phase'):
        means = [r['test_mean'] for r in records if r['mode']==name]
        summary[name] = dict(seed_means=means, mean=float(np.mean(means)),
                             seed_sd=float(np.std(means, ddof=1)) if len(means)>1 else None)
        print(f"SUMMARY {name}: mean={np.mean(means):.4f}", flush=True)
    (a.output/'summary.json').write_text(json.dumps(dict(summary=summary, records=records), indent=2)+'\n')
    print('COMPLETE original CNN phase-channel ablation', flush=True)


if __name__ == '__main__':
    main()
