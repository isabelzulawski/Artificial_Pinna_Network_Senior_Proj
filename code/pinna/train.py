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
from .phase_features import TEMPORAL_MODES, extract, feature_count
from .training_losses import OBJECTIVES, expected_angular_cost, paired_recipe, source_contrastive_loss


@dataclass
class TrainConfig:
    baseline_checkpoint: str | None = None
    output_kind: str = "classification"
    feature_mode: str = "magnitude"
    phase_reference_hz: float = 1000.0
    training_objective: str = 'baseline'
    angular_weight: float = 1.0
    contrastive_weight: float = .1
    select_by: str = 'loss'
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
                 feature_mode="magnitude", preroll=0, phase_reference_hz=1000.0):
    """Features, true direction index and source label for every (source type, draw, direction)."""
    X, T, K = [], [], []
    for kind in kinds_per_dir:
        for _ in range(draws):
            for ch in np.array_split(dir_idx, max(1, len(dir_idx) // chunk)):
                X.append(extract(simulate.make_batch(ir, ch, [kind] * len(ch), gen, snr_db,
                                                    preroll=preroll), feature_mode, phase_reference_hz))
                T.append(ch)
                K.append(np.full(len(ch), kind))
    return torch.cat(X), np.concatenate(T), np.concatenate(K)


@torch.no_grad()
def make_validation_set(ir, train_idx, cfg):
    """Fixed validation features at the training directions: every training source type, val_draws each."""
    gen = torch.Generator().manual_seed(cfg.val_seed)
    X, T, K = _feature_set(ir, train_idx, cfg.sources, cfg.val_draws, gen, cfg.snr_db,
                            feature_mode=cfg.feature_mode, preroll=cfg.preroll,
                            phase_reference_hz=cfg.phase_reference_hz)
    return {"x": X, "dir": T, "source": K}


@torch.no_grad()
def _make_pool(ir, train_idx, cfg, gen, rng, chunk=256):
    """A fixed training set of cfg.train_pool examples: random training directions and source types."""
    pick = rng.choice(len(train_idx), cfg.train_pool)
    kinds = rng.choice(list(cfg.sources), cfg.train_pool)
    X = torch.cat([extract(simulate.make_batch(ir, train_idx[pick[i:i + chunk]], kinds[i:i + chunk], gen,
                                                   cfg.snr_db, preroll=cfg.preroll), cfg.feature_mode, cfg.phase_reference_hz)
                   for i in range(0, cfg.train_pool, chunk)])
    return X, pick


@torch.no_grad()
def evaluate_set(model, X, dir_idx, V, V_cls, batch=4096):
    """Mean loss and great-circle errors of a model on precomputed features."""
    model.eval()
    if isinstance(model, (M.TemporalPhaseClassifier, M.SpectralCNN)):
        batch = min(batch, 128)
    losses, errs = [], []
    for i in range(0, len(X), batch):
        vt = V[torch.as_tensor(dir_idx[i:i + batch], device=DEVICE)]
        out = model(X[i:i + batch])
        losses.append(float(M.direction_loss(model, out, V_cls, vt)) * len(vt))
        errs.append(M.gc_error_deg(M.direction_prediction(model, out, V_cls)[0], vt).cpu().numpy())
    model.train()
    e = np.concatenate(errs)
    return sum(losses) / len(X), e


def fit(ir, V, train_idx, cfg, val=None, log=print):
    """Train a DirectionMLP. V: unit vectors of every grid direction (D_all, 3) on DEVICE; train_idx: the
    training directions (the model's classes). Returns (model with the best validation weights, history, info)."""
    if cfg.training_objective not in OBJECTIVES or cfg.select_by not in ('loss', 'angular'):
        raise ValueError('Unknown training objective or selection metric')
    if cfg.angular_weight < 0 or cfg.contrastive_weight < 0:
        raise ValueError('Auxiliary loss weights must be nonnegative')
    paired = cfg.training_objective in ('paired_control', 'source_contrastive')
    if cfg.training_objective != 'baseline':
        if cfg.feature_mode != 'magnitude' or cfg.output_kind != 'classification' or cfg.train_pool:
            raise ValueError('Training-only ablations require online magnitude MLP classification')
        if paired and (cfg.batch % 2 or len(cfg.sources) < 2):
            raise ValueError('Paired training requires even batch and at least two source families')
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=DEVICE).manual_seed(1000 + cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    V_cls = V[torch.as_tensor(train_idx, device=DEVICE)]
    temporal = cfg.feature_mode in TEMPORAL_MODES
    angular_selection = temporal or cfg.select_by == 'angular'
    if temporal:
        if not cfg.baseline_checkpoint or cfg.output_kind != "classification":
            raise ValueError("Temporal modes require a magnitude baseline checkpoint and classification head")
        baseline, ck = M.load(cfg.baseline_checkpoint, DEVICE)
        if ck.get("feature_mode") != "magnitude" or not isinstance(baseline, M.DirectionMLP):
            raise ValueError("Expected an original magnitude-only baseline checkpoint")
        classes = ck.get("class_vectors")
        if classes is None or classes.shape != V_cls.shape or not torch.allclose(classes.to(DEVICE), V_cls):
            raise ValueError("Baseline training directions must match this protocol in the same order")
        old_preroll = ck.get("info", {}).get("config", {}).get("preroll")
        if old_preroll != cfg.preroll:
            raise ValueError("Baseline and phase training must use the same preroll")
        model = M.TemporalPhaseClassifier(baseline.config).to(DEVICE)
        model.magnitude.load_state_dict(baseline.state_dict())
    elif cfg.feature_mode == "spectral_cnn":
        if cfg.output_kind != "classification":
            raise ValueError("Spectral CNN uses the classification head")
        model = M.SpectralCNN(len(train_idx)).to(DEVICE)
    else:
        model = M.DirectionMLP(feature_count(cfg.feature_mode), len(train_idx), cfg.hidden, cfg.depth,
                               output_kind=cfg.output_kind).to(DEVICE)
    if val is None:
        val = make_validation_set(ir, train_idx, cfg)
    pool = _make_pool(ir, train_idx, cfg, gen, rng) if cfg.train_pool else None

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    history, best, best_state, bad, run_loss, n_run = [], float("inf"), None, 0, 0.0, 0
    if temporal:
        v_loss, v_err = evaluate_set(model, val["x"], val["dir"], V, V_cls)
        best = float(v_err.mean())
        best_state = copy.deepcopy(model.state_dict())
        history.append({"step": 0, "train_loss": None, "val_loss": v_loss,
                        "val_err_mean": best, "val_err_median": float(np.median(v_err)),
                        "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)})
        log(f"baseline step 0  val loss {v_loss:.4f}  val error {best:.2f} deg")
    order, pos = None, 0
    for step in range(1, cfg.steps + 1):
        if pool is None:  # new examples
            if paired:
                b, kinds = paired_recipe(rng, len(train_idx), cfg.sources, cfg.batch)
            else:
                b, kinds = rng.choice(len(train_idx), cfg.batch), rng.choice(list(cfg.sources), cfg.batch)
            x = extract(simulate.make_batch(ir, train_idx[b], kinds, gen,
                                             cfg.snr_db, preroll=cfg.preroll), cfg.feature_mode, cfg.phase_reference_hz)
        else:  # the fixed set, reshuffled every pass
            if order is None or pos + cfg.batch > len(order):
                order, pos = rng.permutation(cfg.train_pool), 0
            j = order[pos:pos + cfg.batch]
            pos += cfg.batch
            x, b = pool[0][torch.as_tensor(j, device=DEVICE)], pool[1][j]
        labels = torch.as_tensor(b, device=DEVICE)
        targets = V_cls[labels]
        if cfg.training_objective == 'source_contrastive':
            embedding = model.forward_features(x)
            out = model.head(embedding)
        else:
            out = model(x)
        loss = M.direction_loss(model, out, V_cls, targets)
        if cfg.training_objective == 'source_contrastive':
            loss = loss + cfg.contrastive_weight * source_contrastive_loss(embedding, labels, targets)
        elif cfg.training_objective == 'angular':
            loss = loss + cfg.angular_weight * expected_angular_cost(out, V_cls, targets)
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
            selection = rec["val_err_mean"] if angular_selection else v_loss
            improved = selection < best - 1e-4
            if improved:
                best, best_state, bad = selection, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            log(f"step {step:5d}  train loss {rec['train_loss']:.4f}  val loss {v_loss:.4f}  "
                f"val error {rec['val_err_mean']:5.2f} deg (median {rec['val_err_median']:.2f})"
                f"{'  *best' if improved else ''}  {rec['seconds']:.0f} s")
            if bad >= cfg.patience:
                log(f"early stop: no improvement in validation selection metric for {cfg.patience} evaluations")
                break
    model.load_state_dict(best_state)
    model.eval()
    metric = "val_err_mean" if angular_selection else "val_loss"
    # Match the 1e-4 improvement threshold used to retain actual weights.
    best_rec = history[0]
    for rec in history[1:]:
        if rec[metric] < best_rec[metric] - 1e-4:
            best_rec = rec
    info = {"best_step": best_rec["step"], "best_val_loss": best_rec["val_loss"],
            "best_val_err_mean": best_rec["val_err_mean"], "stopped_early": history[-1]["step"] < cfg.steps,
            "train_seconds": round(time.time() - t0, 1), "n_params": sum(p.numel() for p in model.parameters()),
            "n_val": len(val["x"]), "config": asdict(cfg), "selection_metric": metric,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return model, history, info


def predictor(model, V_cls, feature_mode="magnitude", phase_reference_hz=1000.0):
    """Wrap a trained model as predict(waves (B, n)) -> (unit vectors (B, 3), confidence (B,))."""
    @torch.no_grad()
    def predict(w):
        return M.direction_prediction(model, model(extract(w, feature_mode, phase_reference_hz)), V_cls)
    return predict
