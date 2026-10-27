"""Torch port of the supplied notebook's complex-mel four-channel features.

Preserves complex mel filtering BEFORE magnitude/phase extraction, the
1024-point STFT, Hann window, hop 256, 64 Slaney mel bands, temporal delta
(width 9, order 1, nearest padding), and crop/pad to 256 time frames.
Only the final two channels are zeroed in the control. No phase reference.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import SR

NFFT, HOP, MELS, T_TARGET = 1024, 256, 64, 256


def mel_bank():
    """Slaney-normalized triangular filters; verified against librosa.filters.mel."""
    logstep = math.log(6.4)/27
    def hz_to_mel(hz):
        return hz/(200/3) if hz < 1000 else 15 + math.log(hz/1000)/logstep
    m = np.linspace(hz_to_mel(20), hz_to_mel(SR/2), MELS+2)
    frequencies = np.where(m < 15, m*(200/3), 1000*np.exp((m-15)*logstep))
    fft_f = np.fft.rfftfreq(NFFT, 1/SR)
    differences = np.diff(frequencies)
    ramps = frequencies[:, None] - fft_f[None, :]
    bank = np.maximum(0, np.minimum(-ramps[:-2]/differences[:-1, None],
                                    ramps[2:]/differences[1:, None])).astype(np.float32)
    bank *= (2/(frequencies[2:]-frequencies[:-2]))[:, None]
    return bank


_BANK = mel_bank()
_CACHE = {}


def complex_mel(w):
    w = w.float()
    if w.device not in _CACHE:
        _CACHE[w.device] = (torch.hann_window(NFFT, device=w.device),
                           torch.as_tensor(_BANK, device=w.device).to(torch.complex64),
                           torch.arange(-4, 5, device=w.device, dtype=torch.float32).reshape(1, 1, 9)/60)
    window, bank, _ = _CACHE[w.device]
    spectrum = torch.stft(w, NFFT, HOP, NFFT, window, center=True,
                          pad_mode='reflect', return_complex=True)
    return bank @ spectrum


def original_features(w, include_phase=True):
    mel = complex_mel(w)
    derivative = _CACHE[w.device][2]
    logmel = torch.log10(mel.abs() + 1e-10)
    delta = F.conv1d(F.pad(logmel.reshape(-1, 1, logmel.shape[-1]), (4, 4), mode='replicate'), derivative)
    delta = delta.reshape_as(logmel)
    if include_phase:
        phase = mel.angle()
        cosine, sine = phase.cos(), phase.sin()
    else:
        cosine, sine = torch.zeros_like(logmel), torch.zeros_like(logmel)
    output = torch.stack((logmel, delta, cosine, sine), 1)[..., :T_TARGET]
    if output.shape[-1] < T_TARGET:
        output = F.pad(output, (0, T_TARGET-output.shape[-1]))
    return output
