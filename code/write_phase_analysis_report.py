"""Build a read-only Markdown/Jupyter report from saved phase diagnostics and protocol audit."""
import argparse
import json
from pathlib import Path


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
                     +['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    a=p.parse_args()
    d=json.loads((a.analysis_dir/'analysis.json').read_text())
    audit=json.loads(a.audit.read_text())
    score_rows=[[name.replace('_',' '),f"{x['independent_draw_accuracy_pct']:.1f}",
                 f"{x['between_within_ratio']:.3f}"] for name,x in d['diagnostics'].items()]
    snr_rows=[[name,f"{x['median_db']:.1f}",f"{100*x['fraction_below_0db']:.1f}%"] for name,x in d['reference_snr'].items()]
    gap_rows=[[k.replace('_',' '),f"{v['mean_deg']:.3f}"] for k,v in audit['validation'].items()]
    text=f'''# Phase-reference investigation: analysis stage

## Findings

The measured responses contain direction-dependent phase. The sampled microphone
features also contain unknown source phase. None of the phase representations
below has yet demonstrated additional MLP localization accuracy beyond magnitude.
The new results here are signal diagnostics, not trained-network results.
No neural training was launched during this analysis stage.

The next controlled model comparison should retain two 512-unit hidden layers
and a 12,000-step OneCycle budget. Do not extend to width 1024 until a phase
representation consistently beats its matched magnitude control on validation.

## Preservation and evaluation audit

The original scaling output directory remains in place. Before/after SHA-256
inventories matched for **{audit['preserved_checkpoints']} checkpoint files**.
Metadata/results snapshots are in the independent protocol-audit directory.
No weights or original results were rewritten.

The seen protocol shares **{audit['train_test_direction_overlap']} measured directions and their IRs**
between training, validation and test. It uses fresh synthetic sounds, not
independent physical sessions. All six source families occur in each partition.
Online training uses wave-generator seed 1000 + training seed; validation uses
CPU seed 90004, and core testing uses CPU seed 777. No test metric enters fit().

The exact float32 feature scan found **{audit['exact_val_test_feature_duplicates']} validation/test duplicates**
among 3,792 validation and 7,584 core-test examples. This excludes exact feature
reuse in that scan, not near duplicates or common response-bank structure.
There is no complete saved corpus of online training waves to exhaustively hash.
The local representative training/validation/test waveform check also found no
duplicates. None of these checks establishes independent measurement sessions.

### Why validation error exceeds reported test error

Validation uses a uniform 10–40 dB sensor-SNR distribution. Core testing uses a
fixed 30 dB SNR. Both use 4096-sample preroll, the same 65,536-sample window,
48 kHz sampling, six source families and the same direction decoder.
Validation has one draw per source/direction (3,792 examples); core has two (7,584).

Saved 512-width/12,000-step seed-0 model, evaluated without updating weights:

{table(['Validation condition','Mean angular error (deg)'],gap_rows)}

Its previously saved core-test mean is **{audit['saved_core_test_mean_deg']:.3f} deg**.
For the paired conditions, source/noise random draws are identical and only the
noise multiplier changes. Thus the SNR mismatch explains most of the gap in this
single-model check. The remaining difference includes different finite source/noise
draws and draw counts. Other leakage mechanisms are not ruled out by an exact-hash
scan alone. Core test results have been inspected repeatedly during development;
new real recordings/sessions are needed for a final untouched evaluation.

## Existing experiments: avoid duplicate claims

| Previous representation | Observed result | Interpretation |
|---|---|---|
| Compact adjacent-frequency STFT phase (real/imaginary circular means) | Seen test mean about 4.26 deg across three seeds, earlier magnitude about 2.69 deg | This particular representation worsened performance |
| Raw full-window relative phase, 250 Hz–8 kHz reference sweep | 1 kHz: validation 5.001 deg, test 2.957 deg; selected 8 kHz three-seed test mean 3.1008 deg | Simple 1 kHz subtraction was already tested; do not present it as new |
| Same raw-phase input size with zero phase | 8 kHz seed-0 control: validation 4.183 deg, test 2.700 deg | Better than supplied raw phase in that architecture |
| Direct complex-mel sine/cosine, pooled to 128 values in a 232-width MLP | Phase validation 4.298 deg versus zero control 4.104 deg, seed 0 | No demonstrated improvement; pooling is lossy |
| Temporal phase correction | Best correction remained at step-zero baseline | Fallback equality is not a learned phase benefit |

These older methods used smaller models/different budgets. Their numbers do not
substitute for a matched 512-width/12,000-step comparison. The requested extension
of the old reference sweep to 12/16/20 kHz has no verified result in this report.

## What phase information is available

The supplied SOFA file has SimpleFreeFieldHRIR responses: 632 directions,
one receiver, 480 real-valued impulse-response samples per direction at 48 kHz.
Data.Delay is zero. Its Fourier transform supplies a complex **calibration
response**. The file does not store original arbitrary source recordings.
The measured response can include the measurement chain; it is not automatically
an isolated pinna-only transfer function.

The simulator creates a clean source, convolves it with the selected measured
IR, discards the preroll, normalizes level with positive gains, then adds noise.
It preserves the IR's filtering phase. Positive RMS normalization does not remove
delay. The current magnitude frontend discards STFT phase explicitly after a
512-point Hann STFT; its level subtraction/frequency differencing do not align
recordings in time. The earlier complex-mel frontend does retain phase before
pooling. All proposed deployment features must use the microphone waveform only.

## Physical interpretation

For an ideal LTI spectrum, Y(f)=S(f)H_direction(f) exp(-j 2 pi f tau).
For finite windowed STFT/FFT segments, convolution and windowing also introduce
boundary effects, so this identity is a useful model, not an exact division rule.

**Simple reference:** wrap(phi(f)-phi(f_ref)) retains source phase differences
and a delay term -2 pi (f-f_ref) tau. It removes a common angular offset, not an
arbitrary time delay or source spectrum. A 1 kHz reference is outside the existing
2–20 kHz magnitude band but is available in the recorded audio. Availability does
not guarantee an energized, reliable source component.

**Frequency-scaled reference:** wrap(Phi(f)-(f/f_ref)Phi(f_ref)) can cancel a pure
delay on a correctly consistent unwrapped branch. However, adding the equivalent
2 pi k to the entire unwrapped phase changes this feature by 2 pi k(1-f/f_ref).
Numerical unwrapping alone does not resolve that global integer-cycle ambiguity.
Noise and missing spectral regions add further ambiguity. A deterministic branch
convention is computable but is not a source-independent physical calibration.

**Delay-normalized phase:** fit a power-weighted affine phase slope within the
largest contiguous reliable 2–20 kHz segment and subtract slope plus intercept.
This is a mathematical detrend of the microphone phase. With unknown source
phase it is not a trustworthy time-of-flight estimate and may remove useful
direction cues. Fitting each independent spectral segment separately would add
more arbitrary gauges, so this diagnostic retains only the largest fit segment.

**Phase curvature:** wrap(phi[k+1]+phi[k-1]-2 phi[k]) at equally spaced bins cancels
linear delay and constant phase modulo 2 pi without unwrapping. It still retains
source phase curvature and loses the pinna's affine phase component. This is a
physically well-defined timing-invariant candidate, not an established accuracy gain.

Reference: the [DFT shift theorem](https://www.dsprelated.com/freebooks/mDFT/Shift_Theorem.html)
relates time delay to linear phase. [NumPy unwrap](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html)
removes large adjacent discontinuities but cannot choose an absolute physical
phase branch. [SciPy group-delay documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html)
defines delay via the derivative of unwrapped system phase; the unknown-source
microphone spectrum is not itself an identified system response.

## Measured signal diagnostics

{d['examples']} simulated windows: six measured directions, six source families,
six independent source draws, 30 dB full-band sensor SNR, 4096 preroll samples.
The same source is shared across directions within each draw to isolate the
response effect; template-training and evaluation use disjoint draws.
The actual full-window reference bin is **{d['actual_reference_hz']:.6f} Hz**.

Circular nearest-template analysis uses sine/cosine *for diagnostic distances*,
not as a committed MLP input design. Preprocessing is fitted on the template-fit
partition only, including separate fitting when a whole source family is held out.
There are only six directions, with chance accuracy **{d['chance_pct']:.2f}%**.

{table(['Representation','Independent-draw template accuracy (%)','Between/within feature variance'],score_rows)}

These small diagnostics do not measure 3D MLP angular error, conditional information
beyond magnitude, or a neural-network accuracy bound. Near-chance phase-only
scores do not prove a joint magnitude/phase model cannot help. There are only
six independent source realizations per family; directions sharing a source
are correlated. No statistical significance or model-seed uncertainty is claimed.
The saved JSON also contains held-family diagnostics.

### Reference-bin reliability

Using known simulated clean/noise components **only as a calibration diagnostic**:

{table(['Source family','Median 1 kHz clean/noise ratio (dB)','Samples below 0 dB'],snr_rows)}

The -80 dB relative-amplitude mask identifies near-zero bins, not clean-signal SNR.
A noise-dominated 1 kHz bin can pass it. The clean/noise ratio cannot be supplied
to an inference model that has only an unknown microphone signal.

### Timing and wrapping

Adding 2 pi to an equivalent unwrapped branch changed the scaled-reference vector
by **{d['scaled_branch_rms_radians']:.3f} radians circular RMS** in the calibration example.
The exact-spectrum delay check uses a known 37-sample delay and verifies algebra.
An actual 37-sample analysis-window shift of one continuous white-noise recording
also changes the finite-window data. Its mean circular RMS feature changes were:

{table(['Representation','Actual crop-shift change (radians)'],[[k,f'{v:.4f}'] for k,v in d['actual_window_shift_rms_radians'].items()])}

Curvature is comparatively stable to this timing shift, but its independent-source
direction-template score was near chance. Delay invariance alone is insufficient.
The fitted-delay and scaled-unwrapped methods were unstable on the actual crops,
despite favorable exact-spectrum algebra, because the unknown source/unwrap
behavior and finite windows matter.

## Deployment and controlled-training plan

| Method | Single-microphone computable? | Main extra work / limitation |
|---|---|---|
| Raw or 1 kHz relative phase | Yes | Full 65,536-point FFT, mask, angles; source/timing dependence |
| Scaled unwrapped reference | Deterministic convention only | Unwrapping, reference branch/gaps; physically ambiguous |
| Fitted-delay residual | Yes as a detrend | Unwrapping plus weighted fit; estimated slope is not identified propagation delay |
| Phase curvature | Yes | FFT plus complex products; removes affine phase but not unknown source curvature |
| Clean IR phase or clean/noise ratios | Calibration only | Requires information unavailable for arbitrary sounds at inference |

The existing analysis window is 1.365 s before acquisition buffering overhead.
No Raspberry Pi timing/power benchmark has been performed, and no claim of sub-1 W
operation is supported. Parameter/MAC counts exclude FFTs, unwrapping and fitting.

For a later small matched screen, keep two **512-unit** hidden layers, baseline
CE/AdamW, the same split, preroll, batch, LR schedule and validation selection.
Use the original 191-feature magnitude model as an anchor. Give every phase
condition 382 extra input slots, padding unused slots with zeros, and include
the identical 573-input model with all added slots zeroed. This gives equal
capacity within phase/control comparisons and makes the input-size cost explicit:

| Architecture | Parameters | Linear MACs per example |
|---|---:|---:|
| Original 191-input, width 512 | 685,558 | 683,520 |
| Matched 573-input, width 512 phase/control | 881,906 | 879,104 |

That is about 28.6% more parameters than the magnitude anchor; a win against the
anchor alone would not establish phase benefit. Compare each phase method with
the matched zero-input control as well. Keep raw 1 kHz angles as one scalar per
frequency (192 values, divided by pi and zero-filled when masked), not two
channels, and do not average wrapped angles arithmetically.

First candidates: existing compact STFT phase, existing raw 1 kHz reference
(replication at the stronger architecture), and timing-invariant curvature.
Treat scaled/unwrapped and fitted-delay variants as diagnostic controls until
their stability is improved. A short plumbing smoke is not an accuracy screen.
For an affordable accuracy screen, use a fresh common reduced budget across all
conditions, then confirm promising candidates and matched controls at **12,000
steps**, seeds **0/1/2**, without selecting on test results. Compare mean validation
error, seed dispersion and paired per-source changes. The exact threshold for a
useful gain must be set before looking at the new screening results.

New phase-MLP validation errors and seed uncertainty: **not measured yet**.
The existing 512-width magnitude reference is 3.045 +/- 0.128 deg validation
across three seeds. The new phase-only diagnostics above do not justify claiming
an improvement or automatically launching a 1024-width phase sweep.

## Figures

'''
    titles=['Calibration IR phase','Microphone phase, shared source across directions','Unknown-source variation at fixed direction',
            'Independent-draw direction templates','Reference-bin clean/noise ratio','Branch ambiguity and ideal delay',
            'Actual recording-window shift']
    paths=sorted(a.analysis_dir.glob('[0-9][0-9]_*.png'))
    for title,path in zip(titles,paths): text+=f'### {title}\n\n![{title}]({path.name})\n\n'
    text+='Source artifacts: `analysis.json` in this directory; protocol audit and checkpoint SHA inventories on Kahan.\n'
    with (a.analysis_dir/'PHASE_REFERENCE_ANALYSIS.md').open('x') as f:f.write(text)
    nb=dict(nbformat=4,nbformat_minor=5,metadata={},cells=[dict(cell_type='markdown',id='phase-report',metadata={},source=text)])
    with (a.analysis_dir/'PHASE_REFERENCE_ANALYSIS.ipynb').open('x') as f:json.dump(nb,f,indent=2)
    print(a.analysis_dir/'PHASE_REFERENCE_ANALYSIS.md')


if __name__ == '__main__': main()
