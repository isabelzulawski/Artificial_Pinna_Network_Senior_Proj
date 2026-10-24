"""Two-resolution, magnitude-only spectra and frequency differences.

Unlike the baseline, retain the level-normalized spectral shape itself, not just
its derivative. No phase or clean-source information is used.
"""
import numpy as np
import torch

from .config import SR

RESOLUTIONS = (512, 2048)
COUNTS = tuple(int(((np.fft.rfftfreq(n, 1/SR) >= 2000) &
                    (np.fft.rfftfreq(n, 1/SR) <= 20000)).sum()) for n in RESOLUTIONS)
N_SPECTRAL = 2 * sum(COUNTS)
SPECTRAL_MODES = ("spectral_mlp", "spectral_cnn")


def spectral_features(w):
    out = []
    for nfft in RESOLUTIONS:
        win = torch.hann_window(nfft, dtype=w.dtype, device=w.device)
        s = torch.stft(w, nfft, nfft//2, nfft, win, center=True,
                       pad_mode="reflect", return_complex=True)
        f = torch.fft.rfftfreq(nfft, 1/SR, device=w.device)
        power = s[:, (f>=2000) & (f<=20000)].abs().square().mean(-1)
        log = torch.log10(power + 1e-12)
        log = log - log.mean(-1, keepdim=True)
        delta = torch.cat((torch.zeros_like(log[:, :1]), log[:, 1:]-log[:, :-1]), -1)
        out.append(torch.stack((log, delta), 1).flatten(1))
    return torch.cat(out, -1)
