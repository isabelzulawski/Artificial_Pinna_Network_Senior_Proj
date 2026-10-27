# Raw reference-phase experiment results

Completed on Kahan, October 26, 2026. All 16 tests passed on Kahan before
training. Code commit: 9a64226038d742f1b594bc329db953599ecc15d1.
Results below are from the supplied console output; full-precision artifacts
are in code/runs/raw_reference_sweep_v2 on Kahan.

This uses 191 original magnitude features plus 192 raw relative phase angles
from a full-window Hann FFT, with a configurable reference. Total input width:
383; MLP parameters: 327,286 (632 classes). Details: code/RAW_PHASE_EXPERIMENT.md.

## Coarse frequency sweep, seed 0

| Requested reference | Validation mean error | Test mean error |
|---|---:|---:|
| 250 Hz | 5.012° | 3.125° |
| 500 Hz | 4.874° | 3.026° |
| 1000 Hz | 5.001° | 2.957° |
| 2000 Hz | 4.845° | 3.103° |
| 4000 Hz | 5.037° | 3.081° |
| 8000 Hz | 4.772° | 3.278° |

8000 Hz was selected using validation mean error only, before the controls
and additional seeds. Test scores did not determine the choice. This is the
leading validation candidate in this coarse sweep, not a global optimum.
1000 Hz has the lowest test score in this table, which does not justify
retrospectively changing the selected reference using the test set.

## Selected reference and controls

| Input / encoding | Seed | Validation mean | Test mean |
|---|---:|---:|---:|
| Raw relative phase, 8000 Hz | 0 | 4.772° | 3.278° |
| Raw relative phase, 8000 Hz | 1 | 5.113° | 3.090° |
| Raw relative phase, 8000 Hz | 2 | 4.934° | 2.934° |
| Zero-filled phase inputs, same MLP width | 0 | 4.183° | 2.700° |
| Sine/cosine encoding, same reference and FFT | 0 | 5.244° | 3.328° |

The selected raw-angle model's exact three-seed test means are
3.2782833382005387, 3.0901923305552703, and 2.9339189337349856 degrees;
average 3.1008 degrees. The earlier magnitude-only three-seed mean was
2.6938 degrees, so raw reference phase increased mean error by about 15.1%.
The earlier magnitude-only seed-0 test mean was 2.6257 degrees.

Interpretation: this particular raw reference-phase design did not improve
localization. In the matched seed-0 architecture control, removing the phase
values improved both validation and test error. Raw angles performed slightly
better than matched sine/cosine in seed 0, but that one-seed comparison does not
establish a reliable encoding advantage. Neither beat magnitude-only.

These are seen-direction synthetic development benchmarks. Held-out directions,
real recordings, other FFT/time representations, denser frequency sweeps,
Raspberry Pi inference and power were not evaluated by this sweep. Phase remains
source- and timing-dependent; the results do not prove that phase is inherently
unhelpful. Prior compact STFT phase results use a different representation, so
comparisons with that experiment do not isolate raw-angle encoding.
