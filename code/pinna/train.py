"""Training with separate training, validation and test data.

  training    new examples drawn every step (default), or a fixed set of train_pool examples drawn once
  validation  a fixed set drawn once with a CPU generator at the training directions, from the training
              distribution; used for the training curves, early stopping and choosing the checkpoint
  test        evaluate.py; never used here

Early stopping: the validation loss is computed every eval_every steps; the weights with the lowest validation
loss are kept, and training stops after `patience` evaluations without improvement. With new examples every
step the model cannot memorize examples, so the validation loss usually keeps falling until the learning-rate
schedule ends; with a fixed train_pool it overfits, and early stopping matters.
"""
import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import model as M
from . import simulate
from .config import DEVICE, SOURCE_TYPES
from .phase_features import extract, feature_count


@dataclass
class TrainConfig:
    feature_mode: str = "magnitude"
    preroll: int = 0  # 0 reproduces upstream; 4096 for matched streaming experiments
    seed: int = 0
    steps: int = 3000  # maximum number of optimizer steps
    batch: int = 256
    lr: float = 2e-3  # peak of the one-cycle schedule (10 % warm-up, then cosine decay)
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    sources: tuple = SOURCE_TYPES  # training source types
    snr_db: tuple = (10.0, 40.0)  # sensor-noise SNR, uniform per example
    train_pool: int | None = None  # None: new examples every step; N: a fixed set of N examples
    eval_every: int = 250  # steps between validation evaluations
    patience: int = 4  # evaluations without a lower validation loss before stopping
    val_seed: int = 90004  # CPU seed of the validation set (the test conditions use other seeds)
    val_draws: int = 1  # validation examples per training direction and source type
    hidden: int = 256
    depth: int = 2


def _feature_set(ir, dir_idx, kinds_per_dir, draws, gen, snr_db, chunk=256,
                 feature_mode="magnitude", preroll=0):
    """Features, true direction index and source label for every (source type, draw, direction)."""
    X, T, K = [], [], []
    for kind in kinds_per_dir:
        for _ in range(draws):
            for ch in np.array_split(dir_idx, max(1, len(dir_idx) // chunk)):
                X.append(extract(simulate.make_batch(ir, ch, [kind] * len(ch), gen, snr_db,
                                                    preroll=preroll), feature_mode))
                T.append(ch)
                K.append(np.full(len(ch), kind))
    return torch.cat(X), np.concatenate(T), np.concatenate(K)


@torch.no_grad()
def make_validation_set(ir, train_idx, cfg):
    """Fixed validation features at the training directions: every training source type, val_draws each."""
    gen = torch.Generator().manual_seed(cfg.val_seed)
    X, T, K = _feature_set(ir, train_idx, cfg.sources, cfg.val_draws, gen, cfg.snr_db,
                            feature_mode=cfg.feature_mode, preroll=cfg.preroll)
    return {"x": X, "dir": T, "source": K}


@torch.no_grad()
def _make_pool(ir, train_idx, cfg, gen, rng, chunk=256):
    """A fixed training set of cfg.train_pool examples: random training directions and source types."""
    pick = rng.choice(len(train_idx), cfg.train_pool)
    kinds = rng.choice(list(cfg.sources), cfg.train_pool)
    X = torch.cat([extract(simulate.make_batch(ir, train_idx[pick[i:i + chunk]], kinds[i:i + chunk], gen,
                                                   cfg.snr_db, preroll=cfg.preroll), cfg.feature_mode)
                   for i in range(0, cfg.train_pool, chunk)])
    return X, pick


@torch.no_grad()
def evaluate_set(model, X, dir_idx, V, V_cls, batch=4096):
    """Mean loss and great-circle errors of a model on precomputed features."""
    model.eval()
    losses, errs = [], []
    for i in range(0, len(X), batch):
        vt = V[torch.as_tensor(dir_idx[i:i + batch], device=DEVICE)]
        out = model(X[i:i + batch])
        losses.append(float(M.loss_fn(out, V_cls, vt)) * len(vt))
        errs.append(M.gc_error_deg(M.decode(out, V_cls)[0], vt).cpu().numpy())
    model.train()
    e = np.concatenate(errs)
    return sum(losses) / len(X), e


def fit(ir, V, train_idx, cfg, val=None, log=print):
    """Train a DirectionMLP. V: unit vectors of every grid direction (D_all, 3) on DEVICE; train_idx: the
    training directions (the model's classes). Returns (model with the best validation weights, history, info)."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=DEVICE).manual_seed(1000 + cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    V_cls = V[torch.as_tensor(train_idx, device=DEVICE)]
    if val is None:
        val = make_validation_set(ir, train_idx, cfg)
    pool = _make_pool(ir, train_idx, cfg, gen, rng) if cfg.train_pool else None

    model = M.DirectionMLP(feature_count(cfg.feature_mode), len(train_idx), cfg.hidden, cfg.depth).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    history, best, best_state, bad, run_loss, n_run = [], float("inf"), None, 0, 0.0, 0
    order, pos = None, 0
    for step in range(1, cfg.steps + 1):
        if pool is None:  # new examples
            b = rng.choice(len(train_idx), cfg.batch)
            x = extract(simulate.make_batch(ir, train_idx[b], rng.choice(list(cfg.sources), cfg.batch), gen,
                                             cfg.snr_db, preroll=cfg.preroll), cfg.feature_mode)
        else:  # the fixed set, reshuffled every pass
            if order is None or pos + cfg.batch > len(order):
                order, pos = rng.permutation(cfg.train_pool), 0
            j = order[pos:pos + cfg.batch]
            pos += cfg.batch
            x, b = pool[0][torch.as_tensor(j, device=DEVICE)], pool[1][j]
        loss = M.loss_fn(model(x), V_cls, V_cls[torch.as_tensor(b, device=DEVICE)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        run_loss, n_run = run_loss + float(loss.detach()), n_run + 1

        if step % cfg.eval_every == 0 or step == cfg.steps:
            v_loss, v_err = evaluate_set(model, val["x"], val["dir"], V, V_cls)
            rec = {"step": step, "train_loss": run_loss / n_run, "val_loss": v_loss,
                   "val_err_mean": float(v_err.mean()), "val_err_median": float(np.median(v_err)),
                   "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
            if pool is not None:  # error on (part of) the training set itself, to show memorization
                k = min(cfg.train_pool, len(val["x"]))
                rec["train_err_mean"] = float(evaluate_set(model, pool[0][:k], pool[1][:k], V_cls, V_cls)[1].mean())
            history.append(rec)
            run_loss, n_run = 0.0, 0
            improved = v_loss < best - 1e-4
            if improved:
                best, best_state, bad = v_loss, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            log(f"step {step:5d}  train loss {rec['train_loss']:.4f}  val loss {v_loss:.4f}  "
                f"val error {rec['val_err_mean']:5.2f} deg (median {rec['val_err_median']:.2f})"
                f"{'  *best' if improved else ''}  {rec['seconds']:.0f} s")
            if bad >= cfg.patience:
                log(f"early stop: no lower validation loss for {cfg.patience} evaluations")
                break
    model.load_state_dict(best_state)
    model.eval()
    best_rec = min(history, key=lambda r: r["val_loss"])
    info = {"best_step": best_rec["step"], "best_val_loss": best_rec["val_loss"],
            "best_val_err_mean": best_rec["val_err_mean"], "stopped_early": history[-1]["step"] < cfg.steps,
            "train_seconds": round(time.time() - t0, 1), "n_params": sum(p.numel() for p in model.parameters()),
            "n_val": len(val["x"]), "config": asdict(cfg)}
    return model, history, info


def predictor(model, V_cls, feature_mode="magnitude"):
    """Wrap a trained model as predict(waves (B, n)) -> (unit vectors (B, 3), confidence (B,))."""
    @torch.no_grad()
    def predict(w):
        return M.decode(model(extract(w, feature_mode)), V_cls)
    return predict
