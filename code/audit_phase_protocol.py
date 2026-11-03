"""Read-only evaluation of an existing magnitude model; never invokes fit()."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinna import config, data, evaluate, features, model, train


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sofa', type=Path, required=True)
    p.add_argument('--scaling-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-cuda', action='store_true')
    a = p.parse_args()
    if a.require_cuda and not torch.cuda.is_available(): p.error('CUDA unavailable')
    a.output.mkdir(parents=True, exist_ok=False)
    root = a.scaling_root
    summary = json.loads((root/'summary.json').read_text())
    assert len(summary['records']) == 18, 'Expected all 18 completed scaling trials'
    # Preserve checkpoints in place and snapshot provenance without rewriting old outputs.
    before = {str(f.relative_to(root)):dict(bytes=f.stat().st_size, sha256=sha(f)) for f in root.rglob('*.pt')}
    (a.output/'checkpoint_inventory_before.json').write_text(json.dumps(before,indent=2))
    for name in ['summary.json','summary.md','selected.json','manifest.json','directions.json']:
        shutil.copy2(root/name, a.output/('scaling_'+name))
    ir, az, el, sr = data.load_sofa(a.sofa)
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['identity']['sofa_sha256'] == sha(a.sofa), 'SOFA differs from scaling data'
    tr, te = data.split('seen',az,el)
    v = torch.as_tensor(data.unit_vec(az,el),dtype=torch.float32,device=config.DEVICE)
    checkpoint = root/'confirm/w512_steps12000_s0/model.pt'
    fitted, ck = model.load(checkpoint,config.DEVICE)
    classes = ck['class_vectors'].to(config.DEVICE)
    assert torch.allclose(classes,v[torch.as_tensor(tr,device=config.DEVICE)])
    cfg = train.TrainConfig(**ck['info']['config'])
    results = {}; validation_hashes = set(); all_val = None
    # Same CPU generator calls in every condition: (30,30) still consumes the
    # uniform SNR draw, preserving sources/noise realizations across conditions.
    for name,snr in [('validation_10_40',cfg.snr_db),('paired_10db',(10.,10.)),
                     ('paired_30db',(30.,30.)),('paired_40db',(40.,40.))]:
        val = train.make_validation_set(ir,tr,replace(cfg,snr_db=snr))
        loss, err = train.evaluate_set(fitted,val['x'],val['dir'],v,classes)
        results[name] = dict(n=len(err),mean_deg=float(err.mean()),median_deg=float(np.median(err)),
                             p95_deg=float(np.percentile(err,95)),loss=loss,
                             by_source={k:float(err[val['source']==k].mean()) for k in config.SOURCE_TYPES})
        if name == 'validation_10_40':
            all_val = val
            validation_hashes = {hashlib.sha256(x.tobytes()).hexdigest() for x in val['x'].cpu().numpy()}
        print(name,results[name]['mean_deg'],flush=True)
    observed = ck['info']['best_val_err_mean']
    assert abs(results['validation_10_40']['mean_deg']-observed) < .02, 'Original validation score did not reproduce'
    # Exact feature-duplicate scan on original core sources. No new test predictions.
    core = evaluate.conditions('seen',quick=True)[0]
    test_hashes = set(); ntest = 0
    for wave,idx,kind in evaluate.batches(core,te,ir,sr,data.mirror_elevation_index(az,el)):
        for x in features.features(wave).cpu().numpy():
            test_hashes.add(hashlib.sha256(x.tobytes()).hexdigest()); ntest += 1
    test_metric = json.loads((checkpoint.parent/'metrics.json').read_text())['metrics']['core']
    after = {str(f.relative_to(root)):dict(bytes=f.stat().st_size, sha256=sha(f)) for f in root.rglob('*.pt')}
    assert before == after, 'Checkpoint preservation check failed'
    (a.output/'checkpoint_inventory_after.json').write_text(json.dumps(after,indent=2))
    result = dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),preserved_checkpoints=len(before),
                  checkpoint_inventory_unchanged=True,train_test_direction_overlap=len(set(tr)&set(te)),
                  unique_validation_features=len(validation_hashes),unique_test_features=len(test_hashes),
                  n_test_features=ntest,exact_val_test_feature_duplicates=len(validation_hashes&test_hashes),
                  feature_hash_limit='Exact float32 feature equality only; not a proof of independent sessions or no near duplicates.',
                  training_seed=cfg.seed,training_wave_seed=1000+cfg.seed,validation_seed=cfg.val_seed,test_seed=core.gen_seed(),
                  protocol='seen',shared_measurement_ir_bank=True,validation=results,
                  saved_core_test_mean_deg=test_metric['gc_mean'],new_test_predictions_performed=False,
                  neural_training_performed=False)
    (a.output/'protocol_audit.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(figsize=(9,4))
    names=list(results); means=[results[k]['mean_deg'] for k in names]
    ax.bar(names,means,color='#305C82'); ax.axhline(test_metric['gc_mean'],ls='--',color='black',label='Previously saved core test (different draws, 30 dB)')
    ax.set(ylabel='Mean angular error (deg)',title='Saved 512-width model: paired validation sounds, sensor SNR changed only')
    ax.tick_params(axis='x',labelrotation=12); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(a.output/'validation_snr_audit.png',dpi=170); plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__': main()
