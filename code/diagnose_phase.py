"""Exploratory feature diagnostic, not a trained model or an accuracy bound.

Compare direction templates across independent source realizations, and measure
sensitivity to a small analysis-window shift. No clean source is a feature input.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from pinna import config, data, simulate
from pinna.features import N_FEATURES
from pinna.phase_features import TEMPORAL_FRAMES, extract


def representations(w):
    temporal = extract(w, "temporal_phase")
    compact = extract(w, "phase")
    return {"magnitude": temporal[:, :N_FEATURES],
            "compact_phase_only": compact[:, N_FEATURES:],
            "temporal_phase_only": temporal[:, N_FEATURES:],
            "temporal_mean_only": temporal[:, N_FEATURES:].reshape(
                len(w), 4, N_FEATURES, TEMPORAL_FRAMES).mean(-1).flatten(1)}


def template_scores(train, test, shifted, labels, n_dirs):
    """Centered cosine nearest-class-mean diagnostic; fit centering on train only."""
    center = train.mean(0, keepdim=True)
    templates = torch.stack([train[labels == d].mean(0) for d in range(n_dirs)]) - center
    templates = F.normalize(templates, dim=-1)
    pred = F.normalize(test-center, dim=-1) @ templates.T
    spred = F.normalize(shifted-center, dim=-1) @ templates.T
    predictions = pred.argmax(-1)
    groups = torch.stack([train[labels == d].mean(0) for d in range(n_dirs)])
    between = (groups-center).square().mean()
    within = (train - groups[labels]).square().mean()
    return {"accuracy_pct": float(100*(predictions == labels).float().mean()),
            "shifted_accuracy_pct": float(100*(spred.argmax(-1) == labels).float().mean()),
            "between_to_within_variance": float(between / within.clamp_min(1e-20)),
            "confusion": torch.bincount(labels*n_dirs+predictions,
                        minlength=n_dirs*n_dirs).reshape(n_dirs,n_dirs).cpu().tolist()}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sofa", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=8, help="Independent train and test draws per direction/family")
    p.add_argument("--directions", type=int, default=18)
    p.add_argument("--samples", type=int, default=config.WINDOW)
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    if a.draws < 2 or a.directions < 2 or a.samples < 2048:
        p.error("Require at least 2 draws/directions and 2048 samples")
    if a.require_cuda and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    ir, az, el, sr = data.load_sofa(a.sofa)
    horizon = np.flatnonzero(np.isclose(el, 0))
    if len(horizon) < a.directions:
        p.error("Not enough horizon measurements for requested direction count")
    idx = horizon[np.linspace(0,len(horizon)-1,a.directions,dtype=int)]
    labels = torch.arange(a.directions, device=config.DEVICE).repeat(a.draws)
    ids = np.tile(idx, a.draws)
    # IR energy spread is a time-scale description, not a claim that an STFT
    # cannot encode delays shorter than its window or hop.
    energy = np.cumsum(ir.astype(np.float64)**2, axis=-1)
    energy /= np.maximum(energy[:, -1:], 1e-30)
    lo, hi = (energy >= .05).argmax(-1), (energy >= .95).argmax(-1)
    report = {"scope": "Exploratory nearest-template diagnostic on horizon only; not a neural-model bound",
              "torch": torch.__version__,
              "sofa_sha256": hashlib.sha256(a.sofa.read_bytes()).hexdigest(),
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip(),
              "directions": idx.tolist(), "azimuths": az[idx].tolist(), "draws": a.draws,
              "samples": a.samples, "sample_rate": sr, "device": str(config.DEVICE),
              "chance_pct": 100/a.directions, "snr_db": 30,
              "shift_samples": 37, "shift_ms": 37000/sr,
              "ir_5_to_95_energy_width_ms_median": float(np.median(hi-lo)*1000/sr),
              "ir_5_to_95_energy_width_ms_p95": float(np.percentile(hi-lo,95)*1000/sr),
              "stft_window_ms": 512000/sr, "stft_hop_ms": 256000/sr,
              "pooled_region_ms_approx": 1000*a.samples/sr/TEMPORAL_FRAMES,
              "results": {}}
    a.output.mkdir(parents=True, exist_ok=False)
    for k, kind in enumerate(config.SOURCE_TYPES):
        sets = []
        shifted = None
        for split in range(2):
            gen = torch.Generator(device=config.DEVICE).manual_seed(120000+k*100+split)
            chunks, schunks = {}, {}
            for start in range(0, len(ids), 36):
                di = ids[start:start+36]
                w = simulate.make_batch(ir, di, [kind]*len(di), gen, 30,
                                        n=a.samples+37, preroll=4096)
                x = representations(w[:, :a.samples])
                for name, feat in x.items():
                    chunks.setdefault(name, []).append(feat)
                if split == 1:
                    for name, feat in representations(w[:, 37:]).items():
                        schunks.setdefault(name, []).append(feat)
            sets.append({name: torch.cat(parts) for name, parts in chunks.items()})
            if split == 1:
                shifted = {name: torch.cat(parts) for name, parts in schunks.items()}
        report["results"][kind] = {}
        for name in sets[0]:
            result = template_scores(sets[0][name], sets[1][name], shifted[name], labels, a.directions)
            report["results"][kind][name] = result
            print(f"{kind:10s} {name:24s} independent={result['accuracy_pct']:5.1f}% "
                  f"shifted={result['shifted_accuracy_pct']:5.1f}% "
                  f"between/within={result['between_to_within_variance']:.4f}", flush=True)
        (a.output/"diagnostic.json").write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(figsize=(10,5))
    kinds = list(report["results"])
    names = list(report["results"][kinds[0]])
    x = np.arange(len(kinds))
    for j, name in enumerate(names):
        ax.bar(x+(j-1.5)*.2, [report["results"][k][name]["accuracy_pct"] for k in kinds],
               width=.2, label=name.replace("_", " "))
    ax.axhline(report["chance_pct"], color="black", linestyle=":", label="chance")
    ax.set(xticks=x, xticklabels=kinds, ylabel="Direction-template accuracy (%)", ylim=(0,100),
           title="Independent source realizations: exploratory feature diagnostic")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.output/"feature_diagnostic.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k != "results"}, indent=2), flush=True)
    print(f"Saved diagnostic to {a.output}", flush=True)


if __name__ == "__main__":
    main()
