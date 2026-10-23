"""Compact phase ablation; the original 191 magnitude features are unchanged.

Phase features are energy-weighted circular means of adjacent-frequency phase
differences, not clean-source transfer-function phase. They retain source/timing
dependence. No source reference or known-direction response is used as input.
"""
import torch

from .features import BAND, HOP, NFFT, N_FEATURES, features

MODES = ("magnitude", "phase", "phase_zero")


def feature_count(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown feature mode: {mode}")
    return N_FEATURES if mode == "magnitude" else 3 * N_FEATURES


def extract(w, mode="magnitude"):
    """(B, samples) -> (B, 191) or (B, 573), for real 48-kHz audio.

    phase_zero matches phase model capacity with zeroed phase inputs. Weighting
    each unit cross-product by its amplitude product gives sum(cross)/sum(abs).
    Silence/low energy contributes zero; circular cancellation remains meaningful.
    """
    feature_count(mode)
    mag = features(w)
    if mode == "magnitude":
        return mag
    if mode == "phase_zero":
        return torch.cat((mag, torch.zeros_like(mag), torch.zeros_like(mag)), -1)
    window = torch.hann_window(NFFT, device=w.device, dtype=w.dtype)
    s = torch.stft(w, NFFT, HOP, NFFT, window, center=True,
                   pad_mode="reflect", return_complex=True)
    s = s[:, torch.as_tensor(BAND, device=w.device)]
    cross = s[:, 1:] * s[:, :-1].conj()
    mean = cross.sum(-1) / cross.abs().sum(-1).clamp_min(1e-12)
    return torch.cat((mag, mean.real, mean.imag), -1)
