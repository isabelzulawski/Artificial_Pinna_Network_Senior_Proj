"""One raw phase angle per frequency, relative to a configurable FFT bin.

Uses a Hann-windowed FFT of the complete microphone window, with no clean
source, direction, phase unwrapping, or sine/cosine channels as model inputs.
The phase remains source- and timing-dependent. Weak target/reference bins
are zero-filled at a fixed -80 dB power threshold relative to the FFT peak.
"""
import math

import torch

from .config import SR
from .features import FREQS

RAW_PHASE_MODES = ("raw_phase", "raw_phase_zero", "raw_phase_sincos")
N_PHASE = len(FREQS)


def reference_bin(reference_hz, n_samples):
    if not math.isfinite(reference_hz) or not 0 < reference_hz < SR / 2:
        raise ValueError("Phase reference must be finite and between 0 and Nyquist")
    index = int(math.floor(reference_hz * n_samples / SR + .5))
    if not 0 < index < n_samples // 2 + 1:
        raise ValueError("Phase reference rounds to DC or outside the FFT")
    return index, index * SR / n_samples


def relative_angles(w, reference_hz):
    """Return (normalized wrapped angles in [-1,1], validity), each (B,192).

    Select the full-window FFT bins nearest the original 192 magnitude-band
    bin centers. Subtract the reference phase and wrap with remainder. Angles
    are divided by pi for scale; this does not remove their discontinuity.
    """
    n = w.shape[-1]
    ref, _ = reference_bin(reference_hz, n)
    spectrum = torch.fft.rfft(w * torch.hann_window(n, device=w.device, dtype=w.dtype))
    bins = torch.floor(torch.as_tensor(FREQS, device=w.device) * n / SR + .5).long()
    target = spectrum[:, bins]
    reference = spectrum[:, ref:ref+1]
    threshold = spectrum.abs().amax(-1, keepdim=True).clamp_min(1e-12) * 1e-4
    valid = (target.abs() > threshold) & (reference.abs() > threshold)
    delta = target.angle() - reference.angle()
    angles = (torch.remainder(delta + math.pi, 2 * math.pi) - math.pi) / math.pi
    return torch.where(valid, angles, 0), valid


def phase_values(w, mode, reference_hz):
    if mode == "raw_phase_zero":
        reference_bin(reference_hz, w.shape[-1])
        return w.new_zeros((len(w), N_PHASE))
    angles, valid = relative_angles(w, reference_hz)
    if mode == "raw_phase":
        return angles
    if mode == "raw_phase_sincos":
        # Matched-reference encoding comparison; not used by the raw-angle MLP.
        return torch.cat((torch.where(valid, torch.cos(math.pi*angles), 0),
                          torch.where(valid, torch.sin(math.pi*angles), 0)), -1)
    raise ValueError(mode)
