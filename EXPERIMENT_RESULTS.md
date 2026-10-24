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
