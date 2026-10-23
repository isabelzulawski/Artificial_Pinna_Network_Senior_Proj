"""Source signals and the forward model: source -> measured pinna response -> microphone.

One microphone signal = source convolved with the impulse response of its direction, normalized to -20 dBFS,
a random gain of +-3 dB, plus white sensor noise at a given SNR. This follows the previous team's data
generation, with 6 source types instead of 3.

Random numbers come from a torch.Generator passed in by the caller. A CUDA generator is fast (training draws
new examples every step); a CPU generator gives the same numbers on every machine (validation and test sets).
Everything else runs on config.DEVICE.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import DEVICE, SR, WINDOW


# ---------------------------------------------------------------- random numbers (drawn on the generator's device)
def randn(gen, *shape):
    return torch.randn(*shape, generator=gen, device=gen.device).to(DEVICE)


def rand(gen, *shape):
    return torch.rand(*shape, generator=gen, device=gen.device).to(DEVICE)


def uniform(gen, lo, hi, B):
    """(B, 1) uniform on [lo, hi)."""
    return rand(gen, B, 1) * (hi - lo) + lo


def randint(gen, lo, hi, B):
    return torch.randint(lo, hi, (B,), generator=gen, device=gen.device).to(DEVICE)


# ---------------------------------------------------------------- spectral helpers
def rfft_len(n):
    return 1 << (n - 1).bit_length()


def apply_filter(x, H, N):
    """Multiply the spectrum of x (B, n), zero-padded to N, by H; keep the first n samples."""
    n = x.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(x, n=N) * H, n=N)[..., :n]


def tilt(y, sr, slope_db_oct):
    """Spectral slope in dB/octave re 1 kHz, one value per example (B,)."""
    n = y.shape[-1]
    N = rfft_len(n)
    f = torch.fft.rfftfreq(N, 1 / sr, device=y.device).clamp(min=50.0)
    g = (f[None] / 1000.0) ** (slope_db_oct[:, None] / (20 * math.log10(2)))
    return apply_filter(y, g, N)


def convolve(x, h):
    """x (B, n) * h (B, taps), 'full' convolution truncated to n samples."""
    n, L = x.shape[-1], h.shape[-1]
    N = 1 << (n + L - 2).bit_length()
    return torch.fft.irfft(torch.fft.rfft(x, n=N) * torch.fft.rfft(h, n=N), n=N)[..., :n]


def normalize(y, target_db=-20.0):
    """RMS to target_db dBFS; peak-limited to 1."""
    r = torch.sqrt((y ** 2).mean(-1, keepdim=True) + 1e-12)
    y = y * (10 ** (target_db / 20) / r)
    peak = y.abs().amax(-1, keepdim=True)
    return torch.where(peak > 0.999, y / peak, y)


def add_noise(y, w, snr_db):
    """y + white noise w scaled to snr_db (scalar or (B, 1)) below the RMS of y."""
    yr = torch.sqrt((y ** 2).mean(-1, keepdim=True))
    return y + w * yr / (10 ** (snr_db / 20) * torch.sqrt((w ** 2).mean(-1, keepdim=True)))


# ---------------------------------------------------------------- sources
def source(kind, B, n, sr, gen):
    """B source signals of n samples, peak-normalized.
    white       white noise
    pink        noise with a random slope (1/f^alpha, alpha in [-0.5, 1])
    band        Gaussian-shaped band of noise (centre 0.5-12 kHz, width 1-8 kHz)
    harmonic    pitched, gliding, vibrato, syllable-like envelope, random roll-off (voice / call stand-in)
    bursts      sloped noise gated on and off in 50 ms blocks
    clicktrain  random clicks at 2-20 per second"""
    t = torch.arange(n, device=DEVICE)[None] / sr
    N = rfft_len(n)
    f = torch.fft.rfftfreq(N, 1 / sr, device=DEVICE)
    f[0] = f[1]
    if kind == "white":
        y = randn(gen, B, n)
    elif kind == "pink":
        x = randn(gen, B, n)
        alpha = uniform(gen, -0.5, 1.0, B)
        y = apply_filter(x, 1 / torch.clamp(f, min=20.0)[None] ** (alpha / 2), N)
    elif kind == "band":
        x = randn(gen, B, n)
        fc = uniform(gen, 500.0, min(12000.0, 0.45 * sr), B)
        bw = uniform(gen, 1000.0, min(8000.0, 0.40 * sr), B)
        y = apply_filter(x, torch.exp(-0.5 * ((f[None] - fc) / (bw / 2 + 1e-8)) ** 2), N)
    elif kind == "harmonic":
        f0 = torch.exp(uniform(gen, math.log(80), math.log(800), B))
        glide = uniform(gen, -0.5, 0.5, B)
        f_inst = f0 * 2 ** (glide * t / t[0, -1]) * (1 + 0.02 * torch.sin(2 * math.pi * uniform(gen, 3, 7, B) * t))
        theta = 2 * math.pi * torch.cumsum(f_inst, -1) / sr + uniform(gen, 0, 2 * math.pi, B)
        K = torch.floor(20000.0 / f_inst.amax(-1, keepdim=True))  # harmonics up to 20 kHz
        half = theta / 2
        den = torch.sin(half)
        ok = den.abs() > 1e-4  # sum_k sin(k theta) -> 0 where sin(theta / 2) -> 0
        y = torch.where(ok, torch.sin(K * half) * torch.sin((K + 1) * half) / torch.where(ok, den, 1.0), 0.0)
        y = tilt(y, sr, uniform(gen, -12, -3, B)[:, 0])
        env = (0.5 + 0.5 * torch.sin(2 * math.pi * uniform(gen, 2, 8, B) * t + uniform(gen, 0, 2 * math.pi, B))) ** 2
        y = y * (0.1 + env)
    elif kind == "bursts":
        y = tilt(randn(gen, B, n), sr, uniform(gen, -6, 1, B)[:, 0])
        blk = int(0.05 * sr)
        gate = (rand(gen, B, n // blk + 1) < uniform(gen, 0.3, 0.7, B)).float()
        gate[:, 0] = 1.0
        gate = gate.repeat_interleave(blk, -1)[:, :n]
        ramp = int(0.005 * sr)  # 5 ms ramps
        gate = F.avg_pool1d(gate[:, None], 2 * ramp + 1, 1, ramp, count_include_pad=False)[:, 0]
        y = y * gate
    elif kind == "clicktrain":
        rate = uniform(gen, 2, 20, B)
        y = (rand(gen, B, n) < rate / sr).float()
        y[torch.arange(B, device=DEVICE), randint(gen, 0, n, B)] = 1.0
        y = y * torch.where(rand(gen, B, n) < 0.5, -1.0, 1.0)
    else:
        raise ValueError(kind)
    return y / (y.abs().amax(-1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------- training and validation examples
@torch.no_grad()
def make_batch(ir, idx, kinds, gen, snr_db, n=WINDOW, sr=SR):
    """Microphone signals (B, n) for directions idx (indices into ir) and one source type per example.
    snr_db: a number, or (lo, hi) for a uniform draw per example. The source starts at the first sample."""
    B = len(idx)
    kinds = np.asarray(kinds)
    x = torch.empty(B, n, device=DEVICE)
    for k in np.unique(kinds):
        m = np.where(kinds == k)[0]
        x[torch.as_tensor(m, device=DEVICE)] = source(k, len(m), n, sr, gen)
    y = normalize(convolve(x, torch.as_tensor(ir[idx], device=DEVICE)))
    y = y * 10 ** (uniform(gen, -3, 3, B) / 20)
    snr = torch.full((B, 1), float(snr_db), device=DEVICE) if np.isscalar(snr_db) else uniform(gen, *snr_db, B)
    return add_noise(y, randn(gen, B, n), snr)
