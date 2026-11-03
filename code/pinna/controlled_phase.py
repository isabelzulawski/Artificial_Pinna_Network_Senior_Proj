"""Matched 573-input ablations; all inputs derive only from microphone audio.

Curvature uses uniformly spaced 2–20 kHz full-window FFT samples, rather than
unwrapping phase. It cancels ideal linear spectral delay modulo 2*pi, but not
unknown source phase or finite-window changes. Masks indicate relative energy,
not an estimate of signal-to-noise ratio.
"""
import torch
from .config import SR
from .features import FREQS, N_FEATURES, features
from .phase_features import extract as original_extract
from .raw_phase import relative_angles

MODES = ('magnitude', 'zero', 'compact', 'reference_1k', 'curvature')
N_MATCHED = 3 * N_FEATURES


def feature_count(mode):
    if mode not in MODES:
        raise ValueError(mode)
    return N_FEATURES if mode == 'magnitude' else N_MATCHED


def extract(w, mode):
    feature_count(mode)
    if mode == 'compact':
        return original_extract(w, 'phase')
    mag = features(w)
    if mode == 'magnitude':
        return mag
    extra = w.new_zeros((len(w), 2 * N_FEATURES))
    if mode == 'reference_1k':
        extra[:, :len(FREQS)] = relative_angles(w, 1000.)[0]
    elif mode == 'curvature':
        n = w.shape[-1]
        z = torch.fft.rfft(w * torch.hann_window(n, device=w.device, dtype=w.dtype))
        # Fixed original band centers are evenly spaced; for WINDOW=65536 each
        # center maps exactly to a bin. Reject incompatible window lengths.
        bins = torch.round(torch.as_tensor(FREQS, device=w.device) * n / SR).long()
        if not torch.all(torch.diff(bins) == bins[1] - bins[0]):
            raise ValueError('Curvature requires uniformly spaced FFT samples')
        t = z[:, bins]
        valid = t.abs() > z.abs().amax(-1, keepdim=True).clamp_min(1e-12) * 1e-4
        unit = t / t.abs().clamp_min(1e-20)
        cross = unit[:, 2:] * unit[:, :-2] * unit[:, 1:-1].conj().square()
        mask = valid[:, 2:] & valid[:, :-2] & valid[:, 1:-1]
        extra[:, :len(FREQS)-2] = torch.where(mask, cross.angle() / torch.pi, 0)
    return torch.cat((mag, extra), -1)


def extract_adapter(w, mode, phase_reference_hz=1000.0):
    return extract(w, mode)
