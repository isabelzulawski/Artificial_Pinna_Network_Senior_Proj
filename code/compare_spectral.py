"""Summarize paired baseline/CNN seed runs and retain a comparison error plot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=Path(__file__).resolve().parent/"runs")
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    summary = {"seen":{},"held_out":{},"seed0_sources":{}}
    loaded = {}
    for mode in ("magnitude","spectral_cnn"):
        means,within,medians,gross = [],[],[],[]
        for seed in range(3):
            path = a.runs/f"{mode}_seen_s{seed}"
            d = json.loads((path/"metrics.json").read_text())
            ex = np.load(path/"examples.npz")
            arrays = {k:ex[f"core/{k}"] for k in ("err","true_idx","source")}
            if mode == "spectral_cnn":
                base = loaded[("magnitude",seed)]
                for k in ("true_idx","source"):
                    if not np.array_equal(base[k],arrays[k]):
                        raise ValueError("Test examples do not align across compared models")
            loaded[(mode,seed)] = arrays
            core = d["metrics"]["core"]
            means.append(core["gc_mean"])
            medians.append(core["gc_median"])
            within.append(core["pct_lt5"])
            gross.append(core["pct_gt45"])
            if seed == 0:
                summary["seed0_sources"][mode] = core["by_source"]
                summary.setdefault("parameters",{})[mode] = d["info"]["n_params"]
        summary["seen"][mode] = {"seed_means_deg":means,"mean_deg":float(np.mean(means)),
            "seed_sd_deg":float(np.std(means,ddof=1)),"mean_of_medians_deg":float(np.mean(medians)),
            "mean_pct_lt5":float(np.mean(within)),"mean_pct_gt45":float(np.mean(gross))}
        for protocol in ("unseen_el","unseen_az"):
            path = a.runs/f"{mode}_{protocol}_s0"/"metrics.json"
            if path.exists():
                summary["held_out"].setdefault(protocol,{})[mode] = json.loads(path.read_text())["metrics"]["core"]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"comparison.json").write_text(json.dumps(summary,indent=2))
    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    for mode,label,color in (("magnitude","Original MLP","black"),("spectral_cnn","Spectral CNN","#008acb")):
        e = loaded[(mode,0)]["err"]
        axes[0].hist(e,bins=np.linspace(0,180,73),histtype="step",linewidth=1.5,color=color,
                     label=f"{label}: mean {e.mean():.2f}°, median {np.median(e):.2f}°, P95 {np.percentile(e,95):.2f}°")
        axes[1].plot(np.sort(e),np.arange(1,len(e)+1)/len(e),label=label,color=color)
    axes[0].set(xlabel="Great-circle error (deg)",ylabel="Count (log scale)",yscale="log",
                title="Seed 0 error distribution",xlim=(0,180))
    axes[0].legend(fontsize=7)
    axes[1].axvline(5,color="green",linestyle=":",label="5° target")
    axes[1].set(xlabel="Great-circle error (deg)",ylabel="Fraction within error",xlim=(0,20),ylim=(0,1),
                title="Seed 0 cumulative error")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.output/"error_comparison.png",dpi=200)
    fig.savefig(a.output/"error_comparison.pdf")
    plt.close(fig)
    for mode,m in summary["seen"].items():
        print(f"{mode}: seeds={m['seed_means_deg']} mean={m['mean_deg']:.4f} "
              f"seed_sd={m['seed_sd_deg']:.4f} median_avg={m['mean_of_medians_deg']:.4f} "
              f"within5={m['mean_pct_lt5']:.3f}% gross45={m['mean_pct_gt45']:.3f}%",flush=True)
    for protocol,models in summary["held_out"].items():
        for mode,m in models.items():
            print(f"{protocol} {mode}: mean={m['gc_mean']:.4f} median={m['gc_median']:.4f} "
                  f"within5={m['pct_lt5']:.3f}%",flush=True)
    for mode,sources in summary["seed0_sources"].items():
        print(mode,"source_means",{s:round(m["gc_mean"],4) for s,m in sources.items()},flush=True)
    print("parameters",summary["parameters"],flush=True)


if __name__ == "__main__":
    main()
