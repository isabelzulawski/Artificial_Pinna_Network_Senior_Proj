"""Input features: the frequency-differenced Welch spectrum, 2-20 kHz (191 numbers per window).

1. STFT: 512-point Hann window, hop 256 (10.7 ms frames, 94 Hz bins at 48 kHz).
2. Keep the 192 bins from 2 to 20 kHz: below 2 kHz this pinna barely changes the spectrum with direction.
3. Average the power over all frames (Welch estimate) and take log10.
4. Subtract the mean over bins: removes the overall level, so loudness does not matter.
5. Difference neighbouring bins: removes smooth source slopes and keeps the pinna's peaks and notches.
"""
import numpy as np
import torch

from .config import SR

NFFT, HOP = 512, 256
F_LO, F_HI = 2000.0, 20000.0
_freqs = np.fft.rfftfreq(NFFT, 1 / SR)
BAND = (_freqs >= F_LO) & (_freqs <= F_HI)
FREQS = _freqs[BAND]  # centre frequency of each kept bin (Hz)
N_FEATURES = int(BAND.sum()) - 1  # 191

_window = {}


def stft_power(w):
    """|STFT|^2 of waves (B, n) in the 2-20 kHz band -> (B, 192, frames)."""
    if w.device not in _window:
        _window[w.device] = torch.hann_window(NFFT, device=w.device)
    S = torch.stft(w, NFFT, HOP, NFFT, _window[w.device], center=True, pad_mode="reflect", return_complex=True)
    return S[:, torch.as_tensor(BAND, device=w.device)].abs() ** 2


def steps(w):
    """The intermediate results of each step, for plots: dict of (B, ...) tensors."""
    P = stft_power(w)
    welch = torch.log10(P.mean(-1) + 1e-12)
    level_removed = welch - welch.mean(-1, keepdim=True)
    return {"power": P, "welch_db": 10 * welch, "level_removed_db": 10 * level_removed,
            "features": level_removed[:, 1:] - level_removed[:, :-1]}


def features(w):
    """Waves (B, n) at 48 kHz -> features (B, 191)."""
    wr = torch.log10(stft_power(w).mean(-1) + 1e-12)
    wr = wr - wr.mean(-1, keepdim=True)
    return wr[:, 1:] - wr[:, :-1]
