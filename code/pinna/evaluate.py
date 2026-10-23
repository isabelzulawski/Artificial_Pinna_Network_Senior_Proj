"""The test battery: conditions, test-wave synthesis and metrics.

Every condition has a fixed CPU generator seed, so its waves are identical on any machine and for every model:
comparisons between models are paired. Test waves differ from the training data in one deliberate way: each
window is cut from a source that has already played for 85 ms (as in a continuous stream). A source switched on
at the first sample adds a broadband click, filtered by the pinna, that a model can localize even when the
source itself carries no directional cue.

A model is any function predict(waves (B, n)) -> (unit vectors (B, 3), confidence (B,) or None).
"""
import math
import time
import zlib
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import torch
from scipy.signal import resample_poly
from scipy.stats import rankdata

from . import simulate as sim
from .config import DEVICE, SOURCE_TYPES, SR, WINDOW
from .data import mirror_elevation_index, unit_vec, vec_to_az_el


# ---------------------------------------------------------------- conditions
@dataclass
class Cond:
    """One test condition. sources: source types and/or ('filt', f_lo, f_hi, order): white noise high-passed at
    f_lo and low-passed at f_hi (None = no cut) with a Butterworth-shaped magnitude of that order (4 = 24 dB/octave
    skirts), or a brick-wall cut in the FFT if order is None."""
    group: str
    name: str
    sources: list = field(default_factory=lambda: list(SOURCE_TYPES))
    draws: int = 1  # examples per direction and source
    snr_db: float | None = 30.0  # None = no sensor noise
    level_db: float = 0.0  # extra gain re the nominal -20 dBFS
    window: int = WINDOW
    tilt_db_oct: float = 0.0  # source spectral slope re 1 kHz
    eq_db: float = 0.0  # std of a random smooth source EQ
    drr_db: float | None = None  # direct-to-reverberant ratio of a diffuse tail (None = anechoic)
    rt60: float = 0.3
    refl_db: float | None = None  # one floor reflection (mirrored elevation), level re direct
    pert_db: float = 0.0  # RMS (2-20 kHz) of a random smooth deviation of the pinna response
    pert_oct: float = 1 / 3  # its feature width (octaves)
    ir_scale: float = 1.0  # pinna response shifted in frequency: H(f / s)
    preroll: int = 4096  # samples the source plays before the window starts
    seed: int | None = None  # None = derived from the name
    x: float | None = None  # value on the x axis of the group's plot

    def gen_seed(self):
        return self.seed if self.seed is not None else 10_000 + zlib.crc32(self.name.encode()) % 1_000_000


def conditions(protocol, quick=False):
    """'seen': every group on all test directions. Held-out protocols: 'core' on the held-out directions and
    'core_train' on the training directions (the difference is the interpolation penalty). quick: core only."""
    if protocol != "seen":
        return [Cond("core", "core", draws=4, seed=777, x=0), Cond("core_train", "core_train", draws=2, seed=779, x=0)]
    L = [Cond("core", "core", draws=2, seed=777, x=0)]
    if quick:
        return L
    for fc in (500, 1000, 2000, 4000, 8000, 16000):
        L.append(Cond("octave", f"octave_{fc}", [("filt", fc / 2 ** 0.5, min(fc * 2 ** 0.5, 23000), 4)], x=fc))
    for fc in (2000, 4000, 6000, 8000, 12000, 16000):
        L.append(Cond("lowpass", f"lowpass_{fc}", [("filt", None, fc, 4)], x=fc))
    for fc in (500, 1000, 2000, 4000, 8000):
        L.append(Cond("highpass", f"highpass_{fc}", [("filt", fc, None, 4)], x=fc))
    for fc in (4000, 8000, 12000, 16000, 20000):
        L.append(Cond("cliff", f"cliff_{fc}", [("filt", None, fc, None)], x=fc))
    for snr in (None, 40, 30, 20, 10, 5, 0):
        L.append(Cond("snr", f"snr_{'clean' if snr is None else snr}", snr_db=snr, x=60 if snr is None else snr))
    for lv in (-40, -20, -10, 0, 10, 17):  # -60 ... -3 dBFS
        L.append(Cond("level", f"level_{-20 + lv}dBFS", level_db=lv, x=-20 + lv))
    for w in (4096, 8192, 16384, 32768, 65536):
        L.append(Cond("window", f"window_{w}", window=w, x=w / SR))
    for t in (-6, -3, 0, 3, 6):
        L.append(Cond("tilt", f"tilt_{t}", ["white"], tilt_db_oct=t, x=t))
    for e in (0, 1, 2, 4, 6):
        L.append(Cond("eq", f"eq_{e}", ["white"], eq_db=e, x=e))
    for d in (None, 20, 10, 5, 0):
        L.append(Cond("reverb", f"reverb_drr_{'inf' if d is None else d}", drr_db=d, x=30 if d is None else d))
    for r in (None, -12, -6, -3):
        L.append(Cond("reflection", f"reflection_{'none' if r is None else r}", refl_db=r, x=-20 if r is None else r))
    for d in (0, 0.5, 1, 2, 3, 4):
        L.append(Cond("irpert", f"irpert_{d}", ["white"], draws=3, pert_db=d, x=d))
    for sc in (0.96, 0.98, 0.99, 1.0, 1.01, 1.02, 1.04):
        L.append(Cond("irscale", f"irscale_{sc}", ["white"], draws=3, ir_scale=sc, x=round(100 * (sc - 1), 1)))
    return L


GROUP_INFO = {  # group -> (x-axis label, what it tests)
    "core": ("", "6 source types x 2 draws at every direction, 30 dB SNR, 1.37 s"),
    "core_train": ("", "core condition on the training directions of a held-out protocol"),
    "octave": ("octave-band centre (Hz)", "white noise in one octave: which frequencies carry the cue"),
    "lowpass": ("upper cut-off (Hz)", "white noise low-passed at 24 dB/octave (speech, distant sources)"),
    "highpass": ("lower cut-off (Hz)", "white noise high-passed at 24 dB/octave"),
    "cliff": ("brick-wall cut-off (Hz)", "nothing above a sharp cut-off (audio recorded at a low sample rate)"),
    "snr": ("SNR (dB; 60 = no noise)", "white sensor noise"),
    "level": ("presentation level (dBFS)", "overall loudness (training: -20 dBFS +-3 dB)"),
    "window": ("window length (s)", "analysis window (latency); the same model, shorter input"),
    "tilt": ("source tilt (dB/octave)", "unknown source / loudspeaker / microphone slope"),
    "eq": ("random EQ std (dB)", "smooth unknown coloration of source or microphone"),
    "reverb": ("direct-to-reverberant ratio (dB; 30 = anechoic)", "diffuse tail, RT60 0.3 s (crude room stand-in)"),
    "reflection": ("reflection level (dB; -20 = none)", "one floor reflection 1-10 ms after the direct sound"),
    "irpert": ("response deviation, RMS 2-20 kHz (dB)", ("random smooth deviation of the pinna response "
                                                          "(re-mounting, wear, a re-printed pinna)")),
    "irscale": ("frequency shift of the pinna response (%)", ("pinna a few % larger or smaller, or a different "
                                                              "speed of sound")),
}


# ---------------------------------------------------------------- test-wave synthesis
def _filtered_white(B, n, sr, lo, hi, order, gen):
    x = sim.randn(gen, B, n)
    N = sim.rfft_len(n)
    f = torch.fft.rfftfreq(N, 1 / sr, device=DEVICE)
    if order is None:
        m = ((f >= (lo or 0.0)) & (f <= (hi or sr))).float()
    else:
        f = f.clamp(min=1.0)
        m = torch.ones_like(f)
        if lo:
            m = m / torch.sqrt(1 + (lo / f) ** (2 * order))
        if hi:
            m = m / torch.sqrt(1 + (f / hi) ** (2 * order))
    y = sim.apply_filter(x, m, N)
    return y / (y.abs().amax(-1, keepdim=True) + 1e-8)


def _smooth_eq(B, n, sr, std_db, gen):
    """Random smooth gain: 4 cosines over log2 frequency 100 Hz-24 kHz, scaled to std_db."""
    N = sim.rfft_len(n)
    f = torch.fft.rfftfreq(N, 1 / sr, device=DEVICE).clamp(min=100.0)
    u = (torch.log2(f / 100.0) / math.log2(240.0)).clamp(0, 1)
    k = torch.arange(1, 5, device=DEVICE)[None, :, None]
    a = sim.randn(gen, B, 4, 1) / k
    ph = sim.rand(gen, B, 4, 1) * 2 * math.pi
    g = (a * torch.cos(2 * math.pi * k * u[None, None] * 0.5 + ph)).sum(1)
    g = (g - g.mean(-1, keepdim=True)) / (g.std(-1, keepdim=True) + 1e-8) * std_db
    return 10 ** (g / 20), N


LOGF_LO, LOGF_HI, LOGF_STEP = math.log2(100.0), math.log2(24000.0), 1 / 48  # log-frequency grid (octaves)


def smooth_logf(v, fwhm_oct):
    """Gaussian smoothing along the last axis of values on the log-frequency grid (FWHM in octaves)."""
    sig = fwhm_oct / (2 * math.sqrt(2 * math.log(2))) / LOGF_STEP
    k = torch.arange(-int(4 * sig) - 1, int(4 * sig) + 2, device=v.device, dtype=v.dtype)
    k = torch.exp(-0.5 * (k / sig) ** 2)
    k = k / k.sum()
    pad = len(k) // 2
    v = torch.nn.functional.pad(v[:, None], (pad, pad), mode="replicate")
    return torch.nn.functional.conv1d(v, k[None, None])[:, 0]


def _response_pert(B, n, sr, rms_db, fwhm_oct, gen):
    """Random smooth gain in dB with RMS rms_db over 2-20 kHz, interpolated onto the FFT bins."""
    fg = 2 ** torch.arange(LOGF_LO, LOGF_HI + 1e-9, LOGF_STEP, device=DEVICE)
    g = smooth_logf(sim.randn(gen, B, len(fg)), fwhm_oct)
    band = (fg >= 2000) & (fg <= 20000)
    g = g - g[:, band].mean(-1, keepdim=True)
    g = g / (g[:, band].pow(2).mean(-1, keepdim=True).sqrt() + 1e-12) * rms_db
    N = sim.rfft_len(n)
    u = (torch.log2(torch.fft.rfftfreq(N, 1 / sr, device=DEVICE).clamp(min=100.0)) - LOGF_LO) / LOGF_STEP
    i0 = u.floor().long().clamp(0, len(fg) - 2)
    a = (u - i0).clamp(0, 1)
    return 10 ** ((g[:, i0] * (1 - a) + g[:, i0 + 1] * a) / 20), N


_SCALED = {}


def scaled_irs(ir_all, s):
    """Responses resampled in time so that H(f) becomes H(f / s) (anti-aliased); cached."""
    key = (id(ir_all), float(s))
    if key not in _SCALED:
        fr = Fraction(float(s)).limit_denominator(200)
        _SCALED[key] = resample_poly(np.asarray(ir_all, np.float64), fr.denominator, fr.numerator,
                                     axis=-1).astype(np.float32)
    return _SCALED[key]


def _reverb_ir(ir_all, B, sr, rt60, gen, K=8):
    """Diffuse tail: K decaying noises, each filtered by the response of a random measured direction."""
    L = int(1.2 * rt60 * sr)
    t = torch.arange(L, device=DEVICE) / sr
    env = torch.exp(-6.91 * t / rt60) * (t >= 0.002)
    N = sim.rfft_len(L + ir_all.shape[-1])
    H = 0
    for _ in range(K):
        j = sim.randint(gen, 0, len(ir_all), B)
        H = H + torch.fft.rfft(sim.randn(gen, B, L) * env, n=N) * torch.fft.rfft(ir_all[j], n=N)
    return torch.fft.irfft(H, n=N)[:, :L + ir_all.shape[-1] - 1]


def synth(cond, ir_all, idx, src, sr, gen, mirror_idx=None):
    """Test waves (B, cond.window) for directions idx and one source spec."""
    B, n = len(idx), cond.window
    pre = cond.preroll + (int(1.2 * cond.rt60 * sr) if cond.drr_db is not None else 0)
    x = (_filtered_white(B, n + pre, sr, src[1], src[2], src[3], gen) if isinstance(src, tuple)
         else sim.source(src, B, n + pre, sr, gen))
    if cond.tilt_db_oct:
        x = sim.tilt(x, sr, torch.full((B,), float(cond.tilt_db_oct), device=DEVICE))
    if cond.eq_db:
        x = sim.apply_filter(x, *_smooth_eq(B, n + pre, sr, cond.eq_db, gen))
    if cond.pert_db:  # a zero-phase filter on the source = the same deviation of the pinna response
        x = sim.apply_filter(x, *_response_pert(B, n + pre, sr, cond.pert_db, cond.pert_oct, gen))
    bank = ir_all if cond.ir_scale == 1.0 else scaled_irs(ir_all, cond.ir_scale)
    h = torch.as_tensor(bank[idx], device=DEVICE)
    if cond.refl_db is not None:  # floor reflection: mirrored-elevation response, delayed 1-10 ms
        d = sim.randint(gen, int(0.001 * sr), int(0.010 * sr), B)
        hh = torch.zeros(B, h.shape[-1] + int(0.010 * sr), device=DEVICE)
        hh[:, :h.shape[-1]] = h
        hm = torch.as_tensor(ir_all[mirror_idx[idx]], device=DEVICE) * 10 ** (cond.refl_db / 20)
        hh.scatter_add_(1, d[:, None] + torch.arange(h.shape[-1], device=DEVICE)[None], hm)
        h = hh
    if cond.drr_db is not None:
        r = _reverb_ir(torch.as_tensor(ir_all, device=DEVICE), B, sr, cond.rt60, gen)
        scale = torch.sqrt((h ** 2).sum(-1, keepdim=True) / ((r ** 2).sum(-1, keepdim=True) * 10 ** (cond.drr_db / 10)))
        hh = torch.zeros(B, max(h.shape[-1], r.shape[-1]), device=DEVICE)
        hh[:, :h.shape[-1]] += h
        hh[:, :r.shape[-1]] += r * scale
        h = hh
    y = sim.normalize(sim.convolve(x, h)[:, pre:])
    y = y * 10 ** (sim.uniform(gen, -3, 3, B) / 20)
    if cond.snr_db is not None:
        y = sim.add_noise(y, sim.randn(gen, *y.shape), cond.snr_db)
    return y * 10 ** (cond.level_db / 20)


def batches(cond, dir_idx, ir_all, sr, mirror_idx):
    """Yield (waves, direction indices, source label): per source, per draw, chunks of about 256 directions."""
    g = torch.Generator().manual_seed(cond.gen_seed())
    for src in cond.sources:
        label = src if isinstance(src, str) else "filtered"
        for _ in range(cond.draws):
            for ch in np.array_split(dir_idx, max(1, len(dir_idx) // 256)):
                yield synth(cond, ir_all, ch, src, sr, g, mirror_idx), ch, label


# ---------------------------------------------------------------- metrics
def errors(vpred, tidx, az, el):
    """Great-circle error (deg) of predicted vectors vs the true grid directions tidx."""
    vt = unit_vec(az, el)[tidx]
    vp = np.asarray(vpred, np.float64)
    vp = vp / (np.linalg.norm(vp, axis=-1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.clip((vp * vt).sum(-1), -1, 1)))


def _auroc(score, positive):
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(vpred, tidx, az, el, conf=None, full=True):
    """Metrics for one set of predictions; vpred (N, 3), tidx (N,) indices into the grid.
    Great-circle (GC) error: mean, median, P90, P95, % < 5 / 10 / 20 deg, % > 45 deg (gross errors).
    The previous paper's metrics (Kret et al., Table 1): Az-MAE, El-MAE, azimuth bucket (rounded to the 20 deg
    grid) and azimuth-elevation bucket accuracy; frontal / rear split. Added: Az-MAE within +-60 deg elevation,
    elevation bias, nearest-grid accuracy, mirror confusions (error > 30 deg and within 20 deg of the mirrored
    truth), error by elevation band, directions with mean error > 10 deg, and confidence (AUROC of the
    confidence for errors < 10 deg; mean error of the 50 % / 80 % most confident examples)."""
    V = unit_vec(az, el)
    vt = V[tidx]
    vp = np.asarray(vpred, np.float64)
    vp = vp / (np.linalg.norm(vp, axis=-1, keepdims=True) + 1e-12)
    e = np.degrees(np.arccos(np.clip((vp * vt).sum(-1), -1, 1)))
    m = {"n": len(e), "gc_mean": float(e.mean()), "gc_median": float(np.median(e)),
         "gc_p95": float(np.percentile(e, 95)), "pct_lt5": float(100 * (e < 5).mean()),
         "pct_lt10": float(100 * (e < 10).mean()), "pct_gt45": float(100 * (e > 45).mean())}
    if not full:
        return m
    paz, pel = vec_to_az_el(vp)
    taz, tel = np.mod(az[tidx], 360.0), el[tidx]
    daz = np.abs((paz - taz + 180) % 360 - 180)
    az_hit = np.mod(np.round(paz / 20) * 20, 360) == np.round(taz)
    m.update({
        "gc_p90": float(np.percentile(e, 90)), "pct_lt20": float(100 * (e < 20).mean()),
        "az_mae": float(daz.mean()), "az_mae_el60": float(daz[np.abs(tel) <= 60].mean()),
        "el_mae": float(np.abs(pel - tel).mean()), "el_bias": float((pel - tel).mean()),
        "az_bucket_acc": float(100 * az_hit.mean()),
        "azel_bucket_acc": float(100 * (az_hit & (np.round(pel / 5) * 5 == np.round(tel))).mean()),
        "nearest_grid_acc": float(100 * ((vp @ V.T).argmax(-1) == tidx).mean()),
    })
    for name, ax in (("frontback", 0), ("leftright", 1), ("updown", 2)):
        vm = vt.copy()
        vm[:, ax] *= -1
        em = np.degrees(np.arccos(np.clip((vp * vm).sum(-1), -1, 1)))
        m[f"pct_conf_{name}"] = float(100 * ((e > 30) & (em < 20)).mean())
    front = (taz <= 90) | (taz >= 270)
    m["gc_mean_frontal"], m["gc_mean_rear"] = float(e[front].mean()), float(e[~front].mean())
    for lo, hi in ((-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 91)):
        s = (tel >= lo) & (tel < hi)
        m[f"gc_mean_el[{lo},{min(hi, 90)}{')' if hi < 91 else ']'}"] = float(e[s].mean()) if s.any() else float("nan")
    per_dir = np.bincount(tidx, e, minlength=len(az)) / np.maximum(np.bincount(tidx, minlength=len(az)), 1)
    pd = np.sort(per_dir[np.bincount(tidx, minlength=len(az)) > 0])[::-1]
    m["n_dirs_mean_gt10"] = int((pd > 10).sum())
    m["worst10pct_dirs_share"] = float(100 * pd[:max(1, len(pd) // 10)].sum() / max(pd.sum(), 1e-12))
    if conf is not None:
        conf = np.asarray(conf, np.float64)
        m["conf_auroc_err_lt10"] = _auroc(conf, e < 10)
        o = np.argsort(-conf)
        for cov in (50, 80):
            m[f"gc_mean_top{cov}pct_conf"] = float(e[o[:int(len(e) * cov / 100)]].mean())
    return m


def interpolation_metrics(vpred, tidx, train_idx, az, el, protocol):
    """Held-out protocols. bracket: prediction between the two training grid lines around the true direction
    (elevation +-5 deg, or azimuth +-20 deg); snapped: within 1 deg of a training direction."""
    paz, pel = vec_to_az_el(vpred)
    taz, tel = np.mod(az[tidx], 360.0), el[tidx]
    if protocol == "unseen_el":
        inside, signed = (pel >= tel - 5) & (pel <= tel + 5), pel - tel
    else:
        signed = (paz - taz + 180) % 360 - 180
        inside = np.abs(signed) <= 20
    Vtr = unit_vec(az[train_idx], el[train_idx])
    snapped = np.degrees(np.arccos(np.clip((unit_vec(paz, pel) @ Vtr.T).max(-1), -1, 1))) < 1.0
    return {"pct_bracket": float(100 * inside.mean()), "pct_snapped_to_train_dir": float(100 * snapped.mean()),
            "mean_signed_err_interp_axis": float(signed.mean())}


# ---------------------------------------------------------------- the battery
def run_battery(predict, protocol, ir, az, el, train_idx, test_idx, sr=SR, quick=False, log=print):
    """Score one model on every condition of the protocol.
    -> {"conditions": {name: Cond fields}, "metrics": {name: metrics}, "examples": {name: per-example arrays}}"""
    mirror = mirror_elevation_index(az, el)
    out = {"protocol": protocol, "conditions": {}, "metrics": {}, "examples": {}}
    t0 = time.time()
    for cond in conditions(protocol, quick):
        dirs = train_idx if cond.group == "core_train" else test_idx
        V, C, T, S = [], [], [], []
        for w, ch, label in batches(cond, dirs, ir, sr, mirror):
            v, cf = predict(w)
            V.append(v.float().cpu().numpy())
            if cf is not None:
                C.append(cf.float().cpu().numpy())
            T.append(ch)
            S.append(np.full(len(ch), label))
        v, tidx, src = np.concatenate(V), np.concatenate(T), np.concatenate(S)
        cf = np.concatenate(C) if C else None
        m = metrics(v, tidx, az, el, cf)
        if len(set(src)) > 1:
            m["by_source"] = {s: metrics(v[src == s], tidx[src == s], az, el, full=False) for s in np.unique(src)}
        if cond.group == "core" and protocol != "seen":
            m.update(interpolation_metrics(v, tidx, train_idx, az, el, protocol))
        out["conditions"][cond.name] = {"group": cond.group, "x": cond.x}
        out["metrics"][cond.name] = m
        out["examples"][cond.name] = {"err": errors(v, tidx, az, el).astype(np.float32), "true_idx": tidx,
                                      "source": src, "conf": cf}
        log(f"{cond.name:22s} n={len(tidx):5d}  mean {m['gc_mean']:6.2f} deg  median {m['gc_median']:5.2f}  "
            f"< 5 deg {m['pct_lt5']:5.1f} %  {time.time() - t0:4.0f} s")
    return out
