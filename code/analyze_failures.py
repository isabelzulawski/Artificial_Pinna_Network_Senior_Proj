"""Recreate baseline core waves and inspect frequency coverage on difficult sources.

Response comparisons use known SOFA directions for offline diagnosis only. They
are not model inputs, impossibility proofs, or calibrated uncertainty estimates.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pinna import config, data, evaluate
from pinna.features import BAND, stft_power


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sofa",type=Path,required=True)
    p.add_argument("--baseline-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    saved = json.loads((a.baseline_run/"metrics.json").read_text())
    if saved["protocol"] != "seen":
        p.error("This diagnostic currently requires a seen-direction core run")
    examples = np.load(a.baseline_run/"examples.npz")
    errs = examples["core/err"]
    true = examples["core/true_idx"]
    src = examples["core/source"]
    ir,az,el,sr = data.load_sofa(a.sofa)
    dirs = np.arange(len(ir))
    vectors = torch.as_tensor(data.unit_vec(az,el),dtype=torch.float32,device=config.DEVICE)
    h = torch.as_tensor(ir,device=config.DEVICE)
    hdb = 20*torch.log10(torch.fft.rfft(h,n=512)[:,torch.as_tensor(BAND,device=config.DEVICE)].abs()+1e-10)
    offset = 0
    rows = []
    cond = evaluate.conditions("seen",quick=True)[0]
    gen = torch.Generator().manual_seed(cond.gen_seed())
    def paired_batches():
        for kind in cond.sources:
            for _ in range(cond.draws):
                for idx in np.array_split(dirs,max(1,len(dirs)//256)):
                    state = gen.get_state()
                    noisy = evaluate.synth(cond,ir,idx,kind,sr,gen)
                    clean = None
                    if kind in ("band","harmonic"):
                        clean_gen = torch.Generator().set_state(state)
                        clean = evaluate.synth(replace(cond,snr_db=None),ir,idx,kind,sr,clean_gen)
                    yield noisy,clean,idx,kind
    for w,clean,idx,label in paired_batches():
        stop = offset+len(idx)
        if not np.array_equal(true[offset:stop],idx) or not np.all(src[offset:stop] == label):
            raise ValueError("Saved example order does not match the recreated core battery")
        if label in ("band","harmonic"):
            power = stft_power(w).mean(-1)
            clean_power = stft_power(clean).mean(-1).sum(-1)
            noise_power = stft_power(w-clean).mean(-1).sum(-1).clamp_min(1e-20)
            band_snr = (10*torch.log10(clean_power.clamp_min(1e-20)/noise_power)).cpu().numpy()
            active = power >= power.amax(-1,keepdim=True)*.01
            weights = active.float()/active.sum(-1,keepdim=True).clamp_min(1)
            gaps = []
            for start in range(0,len(idx),32):
                indices = torch.as_tensor(idx[start:start+32],device=config.DEVICE)
                delta = hdb[None] - hdb[indices,None]
                weight = weights[start:start+32,None,:]
                mse = (delta.square()*weight).sum(-1) - (delta*weight).sum(-1).square()
                distance = mse.clamp_min(0).sqrt()
                far = vectors[indices] @ vectors.T < np.cos(np.deg2rad(30))
                gaps.append(distance.masked_fill(~far,float("inf")).amin(-1))
            coverage = active.float().mean(-1).cpu().numpy()
            gap = torch.cat(gaps).cpu().numpy()
            for i in range(len(idx)):
                rows.append({"source":label,"direction":int(idx[i]),"error_deg":float(errs[offset+i]),
                             "occupied_fraction":float(coverage[i]),"far_response_gap_db":float(gap[i]),
                             "band_snr_db":float(band_snr[i])})
        offset = stop
    if offset != len(errs):
        raise ValueError("Saved test set length does not match recreated test set")
    out = {"scope":"Offline diagnostic; response gap is not a bound on localization accuracy",
           "coverage_definition":"2-20 kHz Welch bins within 20 dB of peak observed band power",
           "gap_definition":"Nearest measured response >30 degrees away, weighted level-invariant log-magnitude RMS over occupied bins",
           "snr_definition":"Exact paired clean/noisy synthetic waves, 2-20 kHz STFT power ratio; offline oracle diagnostic only",
           "summary":{},"examples":rows}
    for source in ("band","harmonic"):
        out["summary"][source] = {}
        for group,select in (("all",lambda r:True),("within_5",lambda r:r["error_deg"]<5),
                             ("above_45",lambda r:r["error_deg"]>45)):
            group_rows = [r for r in rows if r["source"]==source and select(r)]
            out["summary"][source][group] = {"n":len(group_rows)}
            if group_rows:
                for k in ("error_deg","occupied_fraction","far_response_gap_db","band_snr_db"):
                    out["summary"][source][group]["median_"+k] = float(np.median([r[k] for r in group_rows]))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"coverage.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out["summary"],indent=2),flush=True)
    print("Observed spectra include sensor noise and pinna coloration; gaps are descriptive, not cue sufficiency proofs.",flush=True)


if __name__ == "__main__":
    main()
