"""Phase/reference diagnostics only: no model fitting and no test-based selection."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinna import config, data, simulate
from pinna.features import FREQS, features


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def unwrap_valid(phase, valid):
    """Unwrap separately within contiguous reliable bins, never across spectral holes."""
    out = np.full_like(phase, np.nan, dtype=float)
    cuts = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    for start, stop in zip(cuts[::2], cuts[1::2]):
        out[start:stop] = np.unwrap(phase[start:stop])
    return out


def phase_views(z, frequencies, reference_hz=1000., targets=FREQS):
    """Angles in radians. NaNs mark unreliable bins; no true direction/source is used.

    Scaled reference is explicitly branch-dependent, even after numerical unwrapping.
    Detrending removes a fitted affine phase, not a verified physical propagation delay.
    Curvature uses uniformly spaced target bins and cancels affine phase modulo 2pi.
    """
    f = np.asarray(frequencies)
    j = int(np.argmin(abs(f-reference_hz)))
    bins = np.array([np.argmin(abs(f-x)) for x in targets])
    amplitude = abs(z)
    valid = (amplitude > max(float(amplitude.max()), 1e-30)*1e-4) & (f > 20) & (f <= 22000)
    phase = np.angle(z)
    unwrapped = unwrap_valid(phase, valid)
    refmask = valid & valid[j]
    # The integer 2pi branch in the reference segment is unknown from a recording.
    same = refmask.copy()
    if valid[j]:
        left = j
        right = j+1
        while left > 0 and valid[left-1]: left -= 1
        while right < len(f) and valid[right]: right += 1
        same[:left] = False; same[right:] = False
    else:
        same[:] = False
    ref = wrap(phase-phase[j])
    scaled = wrap(unwrapped-f/f[j]*unwrapped[j])
    branch = wrap((unwrapped+2*np.pi)-f/f[j]*(unwrapped[j]+2*np.pi))
    fitmask = valid & (f >= 2000) & (f <= 20000)
    cuts = np.flatnonzero(np.diff(np.r_[False, fitmask, False]))
    segments = list(zip(cuts[::2], cuts[1::2]))
    residual = np.full_like(f, np.nan)
    delay = None
    if segments:
        a, b = max(segments, key=lambda pair: pair[1]-pair[0])
        if b-a >= 8:
            ff = f[a:b]; pp = unwrapped[a:b]
            weights = amplitude[a:b]**2
            weights /= max(weights.max(), 1e-30)
            weights = np.maximum(weights, 1e-8)
            slope, intercept = np.polyfit(ff, pp, 1, w=np.sqrt(weights))
            residual[a:b] = wrap(pp-(slope*ff+intercept))
            delay = float(-slope/(2*np.pi))
    p = phase[bins]
    curvature = wrap(p[2:]+p[:-2]-2*p[1:-1])
    uniform = np.allclose(np.diff(f[bins]), np.diff(f[bins])[0])
    cmask = valid[bins][2:] & valid[bins][:-2] & valid[bins][1:-1] & uniform
    def masked(x, mask): return np.where(mask[bins], x[bins], np.nan)
    result = {
        'raw': masked(phase, valid),
        'reference_1k': masked(ref, refmask),
        'scaled_unwrapped': masked(scaled, same),
        'scaled_branch_plus_2pi': masked(branch, same),
        'delay_detrended': residual[bins],
        'phase_curvature': np.r_[np.nan, np.where(cmask, curvature, np.nan), np.nan],
    }
    return result, dict(actual_reference_hz=float(f[j]), reference_valid=bool(valid[j]),
                        fitted_delay_seconds=delay, reliable_target_fraction=float(valid[bins].mean()))


def vectors(a):
    """Circular embedding for distance diagnostics only; missing values remain zero."""
    return np.concatenate((np.nan_to_num(np.cos(a)), np.nan_to_num(np.sin(a))), -1)


def template_accuracy(x, labels, fit, score):
    """Independent-source nearest-template diagnostic, not a CNN/MLP result."""
    means = np.stack([x[fit & (labels == k)].mean(0) for k in np.unique(labels)])
    xx = x[score]
    distance = ((xx[:, None]-means[None])**2).mean(-1)
    return float(100*(distance.argmin(-1) == labels[score]).mean())


def phase_plot(arrays, x, labels, title, path):
    names = ['raw', 'reference_1k', 'scaled_unwrapped', 'delay_detrended']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names):
        for r, label in zip(arrays, labels):
            ax.plot(x/1000, r[name]/np.pi, lw=.8, alpha=.7, label=label)
        ax.set_title(name.replace('_', ' ')); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('Wrapped phase / pi')
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sofa', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draws', type=int, default=6)
    a = p.parse_args()
    if a.draws < 4: p.error('At least four draws required')
    a.output.mkdir(parents=True, exist_ok=False)
    ir, az, el, sr = data.load_sofa(a.sofa)
    wanted = [(0,0),(60,0),(120,0),(180,0),(0,30),(0,-30)]
    idx = np.array([np.argmin(np.linalg.norm(data.unit_vec(az,el)-data.unit_vec(aa,ee), axis=1)) for aa,ee in wanted])
    n = config.WINDOW
    f = np.fft.rfftfreq(n, 1/sr)
    spectra = np.fft.rfft(ir[idx], n=n)
    oracle = [phase_views(z,f)[0] for z in spectra]
    labels = [f'az {az[i]:g}, el {el[i]:g}' for i in idx]
    phase_plot(oracle,FREQS,labels,'Calibration-only measured IR phase (clean response known)',a.output/'01_ir_phase.png')
    arrays, mags, targets, families, draws, reference_snr = [], [], [], [], [], []
    # Shared source across directions isolates response changes in each draw.
    # Independent draws and held-out families then check source dependence.
    for k, kind in enumerate(config.SOURCE_TYPES):
        for draw in range(a.draws):
            gen = torch.Generator().manual_seed(183000+k*100+draw)
            x = simulate.source(kind, 1, n+4096, sr, gen).repeat(len(idx),1)
            clean = simulate.normalize(simulate.convolve(x, torch.as_tensor(ir[idx],device=config.DEVICE))[:,4096:])
            noisy = simulate.add_noise(clean, simulate.randn(gen,*clean.shape),30.)
            z = np.fft.rfft(noisy.cpu().numpy()*np.hanning(n), axis=-1)
            zc = np.fft.rfft(clean.cpu().numpy()*np.hanning(n), axis=-1)
            j = np.argmin(abs(f-1000))
            reference_snr.extend((10*np.log10((abs(zc[:,j])**2+1e-30)/(abs(z[:,j]-zc[:,j])**2+1e-30))).tolist())
            views = [phase_views(zz,f)[0] for zz in z]
            arrays.extend(views); mags.extend(features(noisy).cpu().numpy())
            targets.extend(range(len(idx))); families.extend([kind]*len(idx)); draws.extend([draw]*len(idx))
    y, family, draw = np.array(targets), np.array(families), np.array(draws)
    phase_plot(arrays[:6],FREQS,labels,'Microphone phase: same white-noise realization, six directions',a.output/'02_microphone_phase.png')
    same = [i for i in range(len(arrays)) if y[i] == 0 and family[i] == 'white']
    phase_plot([arrays[i] for i in same],FREQS,[f'white draw {draw[i]}' for i in same],
               'Same direction, independent source realizations (30 dB full-band SNR)',a.output/'03_source_variability.png')
    diagnostics = {}
    names = ['magnitude','raw','reference_1k','scaled_unwrapped','delay_detrended','phase_curvature']
    for name in names:
        x = np.array(mags) if name == 'magnitude' else vectors(np.stack([r[name] for r in arrays]))
        fit = draw < a.draws//2; score = ~fit
        mean = x[fit].mean(0); scale = x[fit].std(0).clip(.05)
        x = (x-mean)/scale  # calibration fitted only on template-training partition
        held = {}
        for kind in config.SOURCE_TYPES:
            held[kind] = template_accuracy(x,y,family != kind,family == kind)
        group = np.stack([x[fit & (y==k)].mean(0) for k in range(len(idx))])
        within = ((x[fit]-group[y[fit]])**2).mean()
        between = group.var(0).mean()
        diagnostics[name] = dict(independent_draw_accuracy_pct=template_accuracy(x,y,fit,score),
                                 held_family_accuracy_pct=held, between_within_ratio=float(between/max(within,1e-20)))
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(range(len(names)),[diagnostics[x]['independent_draw_accuracy_pct'] for x in names],color='#305C82')
    ax.axhline(100/len(idx),ls=':',color='black',label='six-direction chance')
    ax.set(xticks=range(len(names)),xticklabels=[x.replace('_','\n') for x in names],ylabel='Template accuracy (%)',ylim=(0,100))
    ax.legend(); fig.tight_layout(); fig.savefig(a.output/'04_template_diagnostic.png',dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4))
    rs = np.array(reference_snr)
    ax.boxplot([rs[family==k] for k in config.SOURCE_TYPES],tick_labels=config.SOURCE_TYPES,showfliers=False)
    ax.axhline(0,ls=':',color='black'); ax.set(ylabel='Clean-to-noise ratio at 1 kHz (dB)',title='Calibration diagnostic: 30 dB full-band SNR does not guarantee a reliable reference')
    fig.tight_layout(); fig.savefig(a.output/'05_reference_snr.png',dpi=170); plt.close(fig)
    # Algebraic delay experiment: exact spectrum multiplication, not a shifted/cropped recording.
    shift = 37/sr
    z = spectra[0]
    original = phase_views(z,f)[0]; delayed = phase_views(z*np.exp(-2j*np.pi*f*shift),f)[0]
    sensitivity = {}
    for name in original:
        delta = wrap(delayed[name]-original[name]); finite = np.isfinite(delta)
        sensitivity[name] = float(np.sqrt(np.mean(delta[finite]**2))) if finite.any() else None
    branch = wrap(original['scaled_branch_plus_2pi']-original['scaled_unwrapped'])
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(FREQS/1000,original['scaled_unwrapped']/np.pi,label='unwrapped branch')
    axes[0].plot(FREQS/1000,original['scaled_branch_plus_2pi']/np.pi,label='same phase + 2pi',alpha=.7)
    axes[0].set(title='Scaled reference depends on arbitrary phase branch',xlabel='Frequency (kHz)',ylabel='Wrapped phase / pi'); axes[0].legend(fontsize=8)
    for name in ['reference_1k','phase_curvature']:
        axes[1].plot(FREQS/1000,wrap(delayed[name]-original[name])/np.pi,label=name)
    axes[1].set(title='Exact 37-sample delay: phase change',xlabel='Frequency (kHz)',ylabel='Wrapped difference / pi'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.output/'06_branch_and_delay.png',dpi=170); plt.close(fig)
    # Small actual-generator duplicate check. This does not prove all historical draws distinct.
    fingerprints = {}
    for name, seed, snr in [('training_sample',1000,(10.,40.)),('validation_sample',90004,(10.,40.)),('core_test_sample',777,30.)]:
        gen = torch.Generator().manual_seed(seed)
        wave = simulate.make_batch(ir,idx,['white']*len(idx),gen,snr,preroll=4096)
        fingerprints[name] = [hashlib.sha256(w.cpu().numpy().tobytes()).hexdigest() for w in wave]
    overlap = {f'{x} vs {yy}':len(set(fingerprints[x]) & set(fingerprints[yy]))
               for i,x in enumerate(fingerprints) for yy in list(fingerprints)[i+1:]}
    result = dict(sofa_sha256=hashlib.sha256(a.sofa.read_bytes()).hexdigest(),device=str(config.DEVICE),
                  directions=idx.tolist(),azimuths=az[idx].tolist(),elevations=el[idx].tolist(),draws=a.draws,
                  examples=len(y),chance_pct=100/len(idx),actual_reference_hz=float(f[np.argmin(abs(f-1000))]),
                  diagnostic_only=True,neural_training_performed=False,diagnostics=diagnostics,
                  exact_delay_rms_radians=sensitivity,scaled_branch_rms_radians=float(np.sqrt(np.nanmean(branch**2))),
                  reference_snr={k:dict(median_db=float(np.median(rs[family==k])),fraction_below_0db=float((rs[family==k]<0).mean())) for k in config.SOURCE_TYPES},
                  sampled_waveform_hash_overlap=overlap,fingerprints=fingerprints)
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('fingerprints',)},indent=2),flush=True)
    print(f'Analysis complete: {a.output}',flush=True)


if __name__ == '__main__': main()
