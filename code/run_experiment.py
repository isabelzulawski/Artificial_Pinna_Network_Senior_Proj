"""Headless training/evaluation entry point; run from any working directory."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinna import config, data, evaluate, model, report, train
from pinna.phase_features import MODES


def distribution(err, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(err, bins=np.linspace(0, 180, 46), color="#239fe5", edgecolor="black", linewidth=.4)
    for value, label, color, style in [
        (err.mean(), "Mean", "black", "--"),
        (np.median(err), "Median", "black", ":"),
        (np.percentile(err, 95), "95th", "red", "-."),
        (5, "Target", "green", "-")]:
        ax.axvline(value, color=color, linestyle=style, label=f"{label} = {value:.1f}°")
    ax.set(xlabel="Great-circle error (deg)", ylabel="Count", title="Test error distribution", xlim=(0, 180))
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"error_distribution.{ext}", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sofa", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are rejected")
    p.add_argument("--mode", choices=MODES, default="magnitude")
    p.add_argument("--protocol", choices=("seen", "unseen_el", "unseen_az"), default="seen")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--preroll", type=int, default=4096, help="0 = original training recipe; 4096 = streaming warmup")
    p.add_argument("--full-battery", action="store_true", help="Default tests core conditions only")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    if min(args.steps, args.batch, args.eval_every) < 1 or args.preroll < 0:
        p.error("steps, batch and eval-every must be positive; preroll nonnegative")
    if args.require_cuda and not torch.cuda.is_available():
        p.error("CUDA is unavailable; refusing an accidental CPU cluster run")
    ir, az, el, sr = data.load_sofa(args.sofa)
    tr, te = data.split(args.protocol, az, el)
    cfg = train.TrainConfig(feature_mode=args.mode, preroll=args.preroll, seed=args.seed,
                            steps=args.steps, batch=args.batch, eval_every=args.eval_every)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    def git(*cmd):
        r = subprocess.run(["git", "-C", str(root), *cmd], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    digest = hashlib.sha256()
    with args.sofa.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "device": str(config.DEVICE), "sofa_sha256": digest.hexdigest(),
                "train_directions": tr.tolist(), "test_directions": te.tolist(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (args.output / "packages.txt").write_text(packages.stdout or packages.stderr)
    v = torch.as_tensor(data.unit_vec(az, el), dtype=torch.float32, device=config.DEVICE)
    fitted, history, info = train.fit(ir, v, tr, cfg)
    vcls = v[torch.as_tensor(tr, device=config.DEVICE)]
    model.save(args.output / "model.pt", fitted, feature_mode=args.mode,
               class_vectors=vcls.cpu(), sample_rate=sr, info=info, environment=metadata)
    (args.output / "training.json").write_text(json.dumps({"history": history, "info": info}, indent=2))
    predict = train.predictor(fitted, vcls, args.mode)
    result = evaluate.run_battery(predict, args.protocol, ir, az, el, tr, te, sr,
                                  quick=not args.full_battery)
    report.save_run(args.output, result, history, info)
    distribution(result["examples"]["core"]["err"], args.output)
    print(f"Saved run to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
