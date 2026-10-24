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
    def __init__(self, n_in, n_classes, hidden=256, depth=2, output_kind="classification"):
        super().__init__()
        if output_kind not in ("classification", "regression"):
            raise ValueError(output_kind)
        self.output_kind = output_kind
        self.config = dict(n_in=n_in, n_classes=n_classes, hidden=hidden, depth=depth,
                           output_kind=output_kind)
        layers, h = [nn.LayerNorm(n_in)], n_in
        for _ in range(depth):
            layers += [nn.Linear(h, hidden), nn.GELU()]
            h = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 3 if output_kind == "regression" else n_classes)

    def forward(self, x):
        out = self.head(self.body(x))
        return F.normalize(out, dim=-1) if self.output_kind == "regression" else out


class TemporalPhaseClassifier(nn.Module):
    """Frozen magnitude classifier plus a learned phase-conditioned logit correction.

    A zero-initialized correction head reproduces baseline logits at step zero.
    Phase processing and magnitude normalization are separate. The zero-input
    control uses the identical architecture, including magnitude correction inputs.
    """
    output_kind = "classification"

    def __init__(self, magnitude_config):
        super().__init__()
        from .features import N_FEATURES
        if magnitude_config.get("output_kind", "classification") != "classification":
            raise ValueError("Temporal fusion requires a magnitude classifier")
        if magnitude_config["n_in"] != N_FEATURES:
            raise ValueError("Baseline must use the original magnitude features")
        self.config = {"architecture": "temporal_phase", "magnitude_config": magnitude_config}
        self.magnitude = DirectionMLP(**magnitude_config)
        self.magnitude.requires_grad_(False)
        self.phase = nn.Sequential(
            nn.Conv2d(4, 8, (5, 3), stride=2, padding=(2, 1)),
            nn.GroupNorm(2, 8), nn.GELU(),
            nn.Conv2d(8, 16, (5, 3), stride=2, padding=(2, 1)),
            nn.GroupNorm(4, 16), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 4)), nn.Flatten(),
            nn.Linear(16 * 8 * 4, 128), nn.GELU())
        self.mag_norm = nn.LayerNorm(N_FEATURES)
        self.fusion = nn.Sequential(nn.Linear(N_FEATURES + 128, 128), nn.GELU(), nn.Dropout(.1))
        self.correction = nn.Linear(128, magnitude_config["n_classes"])
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, x):
        from .features import N_FEATURES
        from .phase_features import TEMPORAL_FRAMES
        mag = x[:, :N_FEATURES]
        phase = x[:, N_FEATURES:].reshape(-1, 4, N_FEATURES, TEMPORAL_FRAMES)
        with torch.no_grad():
            baseline = self.magnitude(mag)
        z = torch.cat((self.mag_norm(mag), self.phase(phase)), dim=-1)
        return baseline + self.correction(self.fusion(z))


class SpectralCNN(nn.Module):
    """Independent frequency CNNs at 512/2048 FFT sizes; position retained."""
    output_kind = "classification"

    def __init__(self, n_classes):
        super().__init__()
        from .spectral_features import COUNTS
        self.config = {"architecture": "spectral_cnn", "n_classes": n_classes}
        self.branches = nn.ModuleList([nn.Sequential(
            nn.Conv1d(2,16,7,stride=2,padding=3), nn.GELU(),
            nn.Conv1d(16,16,5,stride=2,padding=2), nn.GELU(), nn.Flatten())
            for _ in COUNTS])
        width = sum(16*((n+3)//4) for n in COUNTS)
        self.head = nn.Sequential(nn.Linear(width,128), nn.GELU(), nn.Linear(128,n_classes))

    def forward(self, x):
        from .spectral_features import COUNTS
        parts, start = [], 0
        for n, branch in zip(COUNTS, self.branches):
            parts.append(branch(x[:, start:start+2*n].reshape(-1,2,n)))
            start += 2*n
        return self.head(torch.cat(parts, -1))


def direction_loss(model, out, V_cls, v_true):
    """Cosine loss for continuous directions; original soft-label CE otherwise."""
    if model.output_kind == "regression":
        return (1 - F.cosine_similarity(out, v_true, dim=-1)).mean()
    return loss_fn(out, V_cls, v_true)


def direction_prediction(model, out, V_cls):
    """Regressors have no calibrated confidence; return None for it."""
    if model.output_kind == "regression":
        return out, None
    return decode(out, V_cls)


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
    cfg = dict(ck["config"])
    architecture = cfg.pop("architecture", "mlp")
    if architecture == "temporal_phase":
        m = TemporalPhaseClassifier(**cfg)
    elif architecture == "spectral_cnn":
        m = SpectralCNN(**cfg)
    elif architecture == "mlp":
        m = DirectionMLP(**cfg)
    else:
        raise ValueError(f"Unknown checkpoint architecture: {architecture}")
    m.load_state_dict(ck["state_dict"])
    return m.to(device or "cpu").eval(), ck
