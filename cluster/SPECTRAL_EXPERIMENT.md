# Richer magnitude frontend experiment

The existing baseline keeps 191 neighboring-bin differences of the normalized
512-point Welch log-power spectrum. New modes retain BOTH the normalized shape
and its first difference, at FFT sizes 512 and 2048 (hops 256 and 1024), in 2–20 kHz.
Each channel has its overall level removed; no phase or source reference is used.

- `spectral_mlp`: same 256x256 MLP/classification formulation, richer 1920 inputs.
- `spectral_cnn`: exactly the same features, two small frequency CNN branches,
  flattened with frequency position preserved, then a 128-unit classifier head.

The feature-only MLP comparison separates frontend changes from the CNN change.
Parameter counts and preprocessing cost differ from baseline; this is not an
equal-parameter or equal-compute benchmark. The CNN has two stride-2 convolutions
per branch, 16 channels, kernels 7 and 5. It has no global frequency pooling.

Use the same synthetic source distribution, seed, training steps, optimizer,
checkpoint rule (validation soft-label loss), and existing core test waveforms:

```bash
for mode in spectral_mlp spectral_cnn; do
  /zooper2/isabel.zulawski/jupyter_venv/bin/python3 -u code/run_experiment.py \
    --sofa "data/ArtificialPinnaB2_v2 1.sofa" --mode "$mode" \
    --protocol seen --seed 0 --steps 3000 --eval-every 250 --batch 256 \
    --require-cuda --output "code/runs/${mode}_seen_s0" || break
done
```

Assess validation results and report per-source test errors and catastrophic-error
rates. Treat the existing tests as exploratory development benchmarks. Do not
claim improved generalization until repeated seeds and independent recordings or
measurement directions support it. Any candidate's Pi feature-extraction time and
network inference time must be measured; higher frequency resolution costs more.

## Frequency-coverage diagnostic

```bash
/zooper2/isabel.zulawski/jupyter_venv/bin/python3 -u code/analyze_failures.py \
  --sofa "data/ArtificialPinnaB2_v2 1.sofa" \
  --baseline-run code/runs/magnitude_seen_s0 \
  --output code/runs/baseline_coverage_v1
```

This recreates core test waves in their saved order and examines band/harmonic
examples. Coverage is the fraction of observed 2–20 kHz Welch bins within 20 dB
of the strongest bin. It includes sensor noise and direction-dependent filtering;
it does not identify exactly which bins contain clean source signal.

The updated diagnostic also recreates matched clean/noisy synthetic waves from
identical generator states, to estimate the actual 2–20 kHz SNR. Clean waveforms
are used only for diagnosis, never as model inputs or training labels. A 30 dB
full-band SNR need not imply a 30 dB SNR in the model's analysis band.

For offline diagnosis only, it computes the closest measured response more than
30 degrees away in a level-invariant log-magnitude distance weighted over those
bins. Small response gaps suggest potential ambiguities, but do not prove a
fundamental accuracy limit. Neither the true direction nor the measured response
comparison is provided to the learned model.
