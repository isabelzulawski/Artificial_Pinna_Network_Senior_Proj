# Exploratory Kahan results

These synthetic-source tests are development benchmarks, not an untouched final
test set. All use the supplied 632-direction SOFA file. Numbers below are rounded
from the saved metrics or reported run output. No real-recording accuracy is claimed.

## Compact phase summary, seen directions

| Features | Mean angular error, seed 0 | Seed 1 | Seed 2 | Average |
|---|---:|---:|---:|---:|
| Magnitude | 2.63 | 2.65 | 2.81 | 2.70 |
| Magnitude + compact phase | 4.22 | 4.21 | 4.36 | 4.26 |
| Zeroed phase inputs | 2.66 | 2.64 | 2.48 | 2.59 |

The compact phase representation did not help under these settings. This does not
rule out useful phase information or other phase-aware architectures.

## Continuous regression ablation, seed 0

Magnitude features and hidden MLP layers were retained. Regression uses a normalized
3D output with cosine loss; classification uses the existing spherical soft labels
and local decoding. These output heads differ in parameter count and objective.

| Protocol | Magnitude classifier mean error | Magnitude regression mean error |
|---|---:|---:|
| Seen directions | 2.63 | 6.98 |
| Held-out elevations | 5.00 | 13.02 |
| Held-out azimuth columns | 44.08 | 45.19 |

The tested regression setup did not improve the results. This does not isolate
the classifier's 15-degree decoding radius as a cause of unseen-azimuth failures.

## Temporal phase correction, seed 0, seen directions

Implemented at commit `2867370e86f19a0f2415bae38f8474532fcce9f7`.
Both modes start from `magnitude_seen_s0/model.pt`, freeze magnitude weights, and
train a CNN logit correction with batch 64, peak LR 0.001, up to 3000 steps.
The zero-input control has the same architecture and additional training budget.
Unlike earlier runs, selection uses validation mean angular error and includes
the original baseline at step zero. Both stopped at step 1000 after four validation
checks without an improvement of at least 0.0001 degrees.

| Model retained by validation | Selected step | Test mean | Test P95 |
|---|---:|---:|---:|
| Original magnitude baseline | 3000 | 2.6257 | 3.8558 |
| Temporal phase correction | 0 | 2.6257 | 3.8558 |
| Zero-input correction | 0 | 2.6257 | 3.8558 |

The temporal runs retained the zero-correction baseline: equal test errors are
fallback results, not evidence that learned phase processing improved performance.
Held-out direction protocols and extra seeds were not run for this architecture.

Nine automated tests passed locally and on Kahan GPU, including phase masking,
magnitude feature preservation, exact initial baseline logits, frozen baseline
weights, phase-branch gradients, checkpoint round-trip and wrong-protocol rejection.
The full seen-direction training/evaluation pipeline completed for both temporal
modes. Outputs and error histograms are in their respective ignored `code/runs/`
directories on Kahan.

Before another architecture search, investigate phase-feature usefulness on
transient versus stationary source families, sensitivity to crop timing, and how
much short-delay structure is lost by the STFT and local time pooling. Any
clean-source or HRTF-phase oracle diagnostic must remain separate from deployment
inputs: deployment has only the microphone recording.

## Feature diagnostic on Kahan

Diagnostic code commit: `c3cd123ae2d3eb45ba79f1f0f987584ba31b7219`.
18 measured horizon directions, 8 independent training and 8 test realizations
per direction/source family, 30 dB SNR and 1.365-second analysis windows. The
centered cosine nearest-class-mean classifier fits templates only on training
realizations. Chance accuracy is 5.56%; each test row has 144 examples.

| Source | Magnitude | Compact phase-derived | Temporal phase-derived | Time mean of temporal channels |
|---|---:|---:|---:|---:|
| White noise | 100.0% | 99.3% | 100.0% | 100.0% |
| Pink noise | 100.0% | 98.6% | 100.0% | 100.0% |
| Band noise | 83.3% | 47.9% | 54.9% | 80.6% |
| Harmonic | 75.7% | 20.8% | 8.3% | 10.4% |
| Bursts | 100.0% | 72.9% | 33.3% | 99.3% |
| Click train | 100.0% | 18.1% | 25.7% | 81.2% |

The phase-derived representations contain reproducible directional structure in
some conditions. Their masking/weighting also uses magnitude, and STFT phase
statistics can reflect the signal spectrum, so this does NOT establish information
independent of magnitude. Template results are not trained-model accuracy bounds.
Magnitude is strong in cases where phase-derived features are strongest; those
cases offer little headroom. Transient position variability is a plausible reason
that time averaging helps the template classifier, but this is not isolated proof.

A 37-sample (0.771 ms) test-window shift did not cause a consistent broad collapse
in this particular diagnostic. This does not establish invariance to arbitrary
onset/delay changes. Complete shifted results/confusions are saved on Kahan in
`code/runs/phase_feature_diagnostic_v1/diagnostic.json` with the comparison PNG.

Measured 5%-to-95% IR energy width: median 0.375 ms, 95th percentile 0.500 ms.
Current STFT window/hop: 10.667/5.333 ms; temporal pooling region approximately
21.333 ms. These scales motivate further inspection of transient features; phase
can encode sub-window delays, so these numbers alone do not demonstrate loss.

Next investigate which source families dominate the existing model's errors and
test whether short-delay/phase descriptors improve those particular cases on
validation data. Avoid claiming that phase has no useful structure, or that a
shorter analysis window will necessarily improve localization.

## Baseline failure analysis and richer magnitude experiment

The baseline seen-direction seed-0 source means were 0.82 degrees (white/pink/click
train), 0.94 (bursts), 7.20 (band noise), and 5.15 (harmonic). All 101 errors above
45 degrees were in band noise (63/1264 examples) or harmonic sounds (38/1264).

The baseline coverage diagnostic recreated the saved core wave order and examined
those two difficult source families. It also regenerated matched clean/noisy
synthetic waves from the same random-generator states, with clean signals used
ONLY as an offline diagnostic. Saved at `code/runs/baseline_coverage_v2/coverage.json`.

| Source and error group | Count | Median observed occupied fraction | Median true 2–20 kHz SNR |
|---|---:|---:|---:|
| Band noise, error <5 deg | 1104 | 38.5% | 31.2 dB |
| Band noise, error >45 deg | 63 | 12.5% | 31.2 dB |
| Harmonic, error <5 deg | 1128 | 43.2% | 17.2 dB |
| Harmonic, error >45 deg | 38 | 100.0% | -5.5 dB |

Occupied fraction uses bins within 20 dB of the strongest observed bin. The
100% figure for harmonic failures is consistent with the observed spectrum being
noise-dominated, not with a clean broadband source. Large band-noise errors also
had smaller offline nearest-far-direction response gaps (median 0.77 dB versus
1.95 dB in the <5-degree group), suggesting potential ambiguity over the excited
frequencies. This is not a proof of a fundamental accuracy floor.

These findings motivate examining the 2 kHz cutoff and noise reliability instead
of assuming that more phase features will address every difficult example.
Neither observed coverage nor the response-gap statistic is a deployment oracle.

The first richer-magnitude comparison retains spectral shape and its derivative
at 512/2048 FFT sizes. Both models use identical new features and the existing
classification objective, data recipe and checkpoint selection. Seed-0 seen-direction
results: original MLP 2.63 degrees, richer-input MLP 2.90 degrees, frequency CNN
2.39 degrees. This is a modest initial CNN gain requiring additional seeds and
held-out direction checks; the new models have different compute/parameter costs.

### Completed spectral CNN comparison

Additional seeds and held-out direction runs completed on Kahan. Exact means from
saved metrics differ slightly from averages computed from rounded console values.

| Seen-direction statistic across seeds 0, 1, 2 | Original magnitude MLP | Spectral CNN |
|---|---:|---:|
| Mean angular error, average +/- sample SD | 2.6938 +/- 0.1004 deg | 2.3668 +/- 0.0267 deg |
| Average of median errors | 0.7533 deg | 0.7888 deg |
| Average percent within 5 deg | 96.057% | 96.664% |
| Average percent above 45 deg | 1.380% | 1.165% |
| Parameters, seen-direction classifier | 277750 | 576248 |

The mean error improvement is about 12.1%, repeated in all three seeds. Median
error is slightly worse, and the candidate has about twice as many parameters
plus a second FFT resolution. No Pi latency, power or real-recording improvement
has yet been demonstrated.

| Held-out protocol, seed 0 | Original MLP mean | Spectral CNN mean |
|---|---:|---:|
| Elevations | 4.9950 deg | 4.6395 deg |
| Azimuth columns | 44.0774 deg | 39.9707 deg |

The held-out azimuth problem remains severe. These protocols each use one training
seed and are development diagnostics, not independent final validation.

| Seed-0 source mean | Original MLP | Spectral CNN |
|---|---:|---:|
| Band noise | 7.2014 deg | 7.4086 deg |
| Harmonic | 5.1502 deg | 3.4230 deg |
| White | 0.8205 deg | 0.8307 deg |
| Pink | 0.8169 deg | 0.8676 deg |
| Bursts | 0.9435 deg | 0.9829 deg |
| Click train | 0.8218 deg | 0.8221 deg |

The main seed-0 gain is harmonic-source performance; band noise did not improve.
The richer-input MLP's failure and CNN's gain do not separately prove which input
channel or FFT resolution helps: both representation and inductive bias matter.

`code/compare_spectral.py` verified paired source/direction arrays across each
seed's baseline/CNN tests and saved exact metrics, an overlaid error histogram and
a cumulative-error plot under `code/runs/spectral_comparison_v1/` on Kahan.
Both distributions are from seed 0; the JSON separately reports three-seed averages.
Thirteen local automated tests passed. Twelve tests passed on Kahan before the
paired-SNR test was added, and the paired-SNR diagnostic then completed on GPU.

## Raw phase with a reference-frequency sweep

Completed October 26, 2026, code commit `9a64226038d742f1b594bc329db953599ecc15d1`.
Full record: `RAW_REFERENCE_RESULTS.md`. All 16 tests passed on Kahan.

A 383-input MLP (191 magnitude features + 192 raw relative phase angles)
used a full-window Hann FFT. References 250, 500, 1000, 2000, 4000, and 8000 Hz
were compared at seed 0; 8000 Hz was selected by validation mean angular error.
Its three-seed test mean was **3.1008°**, versus the earlier magnitude MLP's
**2.6938°**. At seed 0, raw phase scored 3.278°; the same-width zero-phase
control scored 2.700°; matched-reference sine/cosine scored 3.328°.
This raw-phase/reference design did not improve the magnitude baseline.
Results remain synthetic seen-direction development benchmarks.
