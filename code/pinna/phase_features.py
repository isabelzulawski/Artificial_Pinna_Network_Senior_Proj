"""Compact phase ablation; the original 191 magnitude features are unchanged.

Phase features are energy-weighted circular means of adjacent-frequency phase
differences, not clean-source transfer-function phase. They retain source/timing
dependence. No source reference or known-direction response is used as input.
"""
import torch
import torch.nn.functional as F

from .features import BAND, HOP, NFFT, N_FEATURES, features
from .spectral_features import N_SPECTRAL, SPECTRAL_MODES, spectral_features
from .raw_phase import N_PHASE, RAW_PHASE_MODES, phase_values

TEMPORAL_FRAMES = 64
TEMPORAL_CHANNELS = 4
TEMPORAL_MODES = ("temporal_phase", "temporal_zero")
MODES = ("magnitude", "phase", "phase_zero", *TEMPORAL_MODES, *SPECTRAL_MODES, *RAW_PHASE_MODES)


def feature_count(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown feature mode: {mode}")
    if mode in SPECTRAL_MODES:
        return N_SPECTRAL
    if mode in TEMPORAL_MODES:
        return N_FEATURES + TEMPORAL_CHANNELS * N_FEATURES * TEMPORAL_FRAMES
    if mode in RAW_PHASE_MODES:
        return N_FEATURES + N_PHASE * (2 if mode == "raw_phase_sincos" else 1)
    return N_FEATURES if mode == "magnitude" else 3 * N_FEATURES


def temporal_phase(s):
    """Local circular summaries, retaining 64 time regions instead of one mean.

    Inputs are complex STFT bins in the original band. Frequency differences and
    temporal phase advances are not source-independent HRTF phase. Suppress bins
    below -80 dB power relative to the window peak; retain coherence in the vector
    length. Remove the expected bin-center temporal phase advance only.
    """
    s = s[:, :, 1:-1]  # omit reflection-padded boundary frames
    amp = s.abs()
    valid = amp > amp.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12) * 1e-4
    frequency = s[:, 1:] * s[:, :-1].conj()
    fm = valid[:, 1:] & valid[:, :-1]
    temporal = s[:, :-1, 1:] * s[:, :-1, :-1].conj()
    tm = valid[:, :-1, 1:] & valid[:, :-1, :-1]
    bins = torch.as_tensor(BAND, device=s.device).nonzero().flatten()[:-1]
    advance = torch.polar(torch.ones_like(bins, dtype=amp.dtype),
                          -2 * torch.pi * bins * HOP / NFFT)
    temporal = temporal * advance[None, :, None]
    channels = []
    for cross, mask in ((frequency, fm), (temporal, tm)):
        cross = cross * mask
        weight = F.adaptive_avg_pool1d(cross.abs(), TEMPORAL_FRAMES).clamp_min(1e-12)
        channels += [F.adaptive_avg_pool1d(cross.real, TEMPORAL_FRAMES) / weight,
                     F.adaptive_avg_pool1d(cross.imag, TEMPORAL_FRAMES) / weight]
    return torch.stack(channels, dim=1)


def extract(w, mode="magnitude", phase_reference_hz=1000.0):
    """(B, samples) -> (B, 191) or (B, 573), for real 48-kHz audio.

    phase_zero matches phase model capacity with zeroed phase inputs. Weighting
    each unit cross-product by its amplitude product gives sum(cross)/sum(abs).
    Silence/low energy contributes zero; circular cancellation remains meaningful.
    """
    feature_count(mode)
    if mode in SPECTRAL_MODES:
        return spectral_features(w)
    mag = features(w)
    if mode == "magnitude":
        return mag
    if mode in RAW_PHASE_MODES:
        return torch.cat((mag, phase_values(w, mode, phase_reference_hz)), -1)
    if mode == "phase_zero":
        return torch.cat((mag, torch.zeros_like(mag), torch.zeros_like(mag)), -1)
    if mode == "temporal_zero":
        return torch.cat((mag, mag.new_zeros((len(w), feature_count(mode) - N_FEATURES))), -1)
    window = torch.hann_window(NFFT, device=w.device, dtype=w.dtype)
    s = torch.stft(w, NFFT, HOP, NFFT, window, center=True,
                   pad_mode="reflect", return_complex=True)
    s = s[:, torch.as_tensor(BAND, device=w.device)]
    if mode == "temporal_phase":
        return torch.cat((mag, temporal_phase(s).flatten(1)), -1)
    cross = s[:, 1:] * s[:, :-1].conj()
    mean = cross.sum(-1) / cross.abs().sum(-1).clamp_min(1e-12)
    return torch.cat((mag, mean.real, mean.imag), -1)
