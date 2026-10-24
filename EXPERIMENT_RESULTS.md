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
