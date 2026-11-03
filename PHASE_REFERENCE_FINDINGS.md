# Phase-reference investigation: first-stage findings

No new phase neural-network training was launched. This branch adds analysis and
read-only audit scripts; it does not change magnitude/phase feature extraction,
the scaling branch, or completed scaling checkpoints.

## Evaluation audit on Kahan

The preserved 512-width, 12,000-step seed-0 model was checked without updating
weights. SHA-256 before/after inventories matched for all **36 checkpoint files**.
There were **zero exact feature duplicates** between the 3,792 validation examples
and 7,584 existing core-test examples. Representative source draws also had no
exact waveform duplicates. This does not exhaustively audit all online training
draws or establish independent physical measurements.

| Same model, paired validation source/noise realizations | Mean error |
|---|---:|
| Original validation distribution, 10–40 dB SNR | 2.919° |
| Same validation sounds, 10 dB SNR | 6.948° |
| Same validation sounds, 30 dB SNR | 1.909° |
| Same validation sounds, 40 dB SNR | 1.535° |
| Previously saved core test, different draws at 30 dB | 1.725° |

The SNR mismatch explains most of the validation/test gap in this one-model check.
The seen split intentionally shares all 632 measured directions and their IRs.
All six source families occur in all partitions. It measures new synthetic sounds
at known response directions, not unseen sessions, source categories or directions.
The core test has been repeatedly inspected in development; keep new physical
recordings for a genuinely untouched final evaluation.

Audit artifacts on Kahan:
`Artificial_Pinna_Phase_Reference/code/runs/protocol_audit_20261103T034719Z_3340148/`.

## Signal analysis

216 simulated windows, six directions, six source families, six independent source
draws per family. Chance in the six-direction nearest-template diagnostic is 16.7%.

| Representation | Independent-draw template accuracy |
|---|---:|
| Existing magnitude features | 98.1% |
| Raw recording phase | 11.1% |
| Phase relative to 1 kHz | 20.4% |
| Numerically unwrapped frequency-scaled reference | 19.4% |
| Fitted-delay residual | 14.8% |
| Delay-invariant phase curvature | 15.7% |

These are small signal diagnostics, **not MLP angular errors**, significance tests,
or proof that phase cannot help a joint model. Source realizations are shared
across directions within a draw; the effective independent source sample is small.

Measured IR phase varies with direction. Unknown-source microphone phase varies
strongly across source realizations. The full-window 1 kHz FFT bin is actually
999.755859 Hz; 38.9% of sampled harmonic examples and 11.1% of sampled band-noise
examples had clean/noise ratio below 0 dB there, despite 30 dB full-band SNR.
That clean/noise ratio is a calibration diagnostic, unavailable at deployment.

Simple phase subtraction leaves a frequency-dependent delay term. A scaled
reference can cancel an ideal delay only on a consistent branch; adding an
equivalent 2π changes the scaled features at noninteger frequency ratios.
Unwrapping does not determine that physical branch. Removing a fitted linear
phase from an unknown-source microphone spectrum is not an identified estimate
of the pinna's propagation delay.

An exact-spectrum 37-sample delay verifies algebraic invariance for curvature.
On actual shifted recording windows, circular RMS phase changes were about
0.012 rad for curvature, 1.51 rad for scaled/unwrapped phase, and 1.82 rad for the
fitted-delay residual. Curvature is timing-stable in this example but still did
not produce strong independent-source direction-template performance.

Sources: [DFT shift theorem](https://www.dsprelated.com/freebooks/mDFT/Shift_Theorem.html),
[NumPy unwrap](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html),
[SciPy group delay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html).

## What to train next, after reviewing these findings

Retain two 512-unit hidden layers and a final 12,000-step budget. Use a fresh
magnitude anchor, an equal-input-width zero-phase control, existing compact phase,
raw 1 kHz relative phase (replication at the stronger architecture), and curvature.
All phase/control models can have 573 inputs (191 magnitude + 382 phase/padded
slots): 881,906 parameters, versus 685,558 for the original magnitude anchor.
Wins against the larger zero-phase control isolate phase more fairly than wins
against the smaller anchor alone. Preserve the existing CE objective, generator,
preroll and validation methodology. Use fresh common screening budgets before
three-seed 12,000-step confirmation. No 1024-width phase run until a consistent
validation gain has been established. New phase MLP validation error/seed
uncertainty has **not yet been measured**.

Raw/relative/curvature features are computable from a single microphone.
Unwrapping and weighted fitting add processing and stability concerns. None
removes arbitrary unknown source phase. Clean transfer-function phase is a
calibration-only input. The current 65,536-sample audio window itself is 1.365 s.
No Pi latency or power measurement has been performed.

## Full report and plots

Local output: `outputs/phase-reference-analysis-v2/PHASE_REFERENCE_ANALYSIS.md`
and its `.ipynb` companion, with seven numbered PNG figures and `analysis.json`.
Generate the same report in the isolated Kahan phase worktree, using a new output
directory (no training is performed):

```bash
export OMP_NUM_THREADS=2
python3 code/analyze_phase_reference.py --sofa /absolute/path/to/the.sofa \
  --output code/runs/phase_analysis_NEW_NAME
python3 code/write_phase_analysis_report.py \
  --analysis-dir code/runs/phase_analysis_NEW_NAME \
  --audit code/runs/protocol_audit_20261103T034719Z_3340148/protocol_audit.json
```

Use the previously working Jupyter interpreter if system Python lacks the packages.
The full report explains prior phase experiments so the 1 kHz replication is not
mistakenly described as a new idea. Output directories and report files reject
accidental overwrites. Existing training/feature code is unchanged.
