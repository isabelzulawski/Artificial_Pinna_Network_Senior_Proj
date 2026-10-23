"""The decoder: an MLP that classifies the input into one of the training directions.

Architecture: LayerNorm(191) -> Linear(191, 256) -> GELU -> Linear(256, 256) -> GELU -> Linear(256, D),
with D = the number of training directions (632 when training on every measured direction).

Training target (soft labels): instead of one-hot, each class c gets weight proportional to
exp(kappa (cos(angle between c and the true direction) - 1)), kappa = 1 / sigma^2, sigma = 6 deg; loss =
cross-entropy with these targets. Neighbouring directions share probability, so the classes are ordered on the
sphere and a direction between two classes can be represented.

Decoder: take the most likely class, then the probability-weighted mean of the unit vectors of the classes within
15 deg of it. The peak probability is returned as a confidence score.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_DEG = 6.0
RADIUS_DEG = 15.0


class DirectionMLP(nn.Module):
    def __init__(self, n_in, n_classes, hidden=256, depth=2):
        super().__init__()
        self.config = dict(n_in=n_in, n_classes=n_classes, hidden=hidden, depth=depth)
        layers, h = [nn.LayerNorm(n_in)], n_in
        for _ in range(depth):
            layers += [nn.Linear(h, hidden), nn.GELU()]
            h = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.head(self.body(x))  # logits over the training directions


def soft_targets(V_cls, v_true, sigma_deg=SIGMA_DEG):
    """(B, D) soft labels for true unit vectors v_true (B, 3), class unit vectors V_cls (D, 3)."""
    kappa = 1.0 / math.radians(sigma_deg) ** 2
    return torch.softmax(kappa * (v_true @ V_cls.T - 1.0), -1)


def loss_fn(logits, V_cls, v_true):
    return -(soft_targets(V_cls, v_true) * F.log_softmax(logits.float(), -1)).sum(-1).mean()


def decode(logits, V_cls, radius_deg=RADIUS_DEG):
    """-> unit vectors (B, 3), confidence = peak probability (B,)."""
    p = torch.softmax(logits.float(), -1)
    top = V_cls[p.argmax(-1)]
    near = (top @ V_cls.T) >= math.cos(math.radians(radius_deg))
    v = (p * near) @ V_cls
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8), p.amax(-1)


def gc_error_deg(v_pred, v_true):
    """Great-circle angle between unit vectors (torch), degrees."""
    return torch.rad2deg(torch.arccos(torch.clamp((v_pred * v_true).sum(-1), -1, 1)))


def count_macs(model):
    """Multiply-accumulates per example (linear layers)."""
    return sum(m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear))


def save(path, model, **meta):
    torch.save({"config": model.config, "state_dict": model.state_dict(), **meta}, path)


def load(path, device=None):
    """-> (model in eval mode, checkpoint dict with the metadata saved alongside it)."""
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    m = DirectionMLP(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device or "cpu").eval(), ck
