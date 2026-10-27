# Earlier PinnaResNet phase-channel experiment

For the phase representation adapted to the lightweight MLP, use
`MLP_MEL_PHASE.md` and `compare_mlp_mel_phase.py`. This document describes the
separate, larger CNN diagnostic only.

Purpose: determine whether explicitly supplying the earlier notebook's phase
channels helps its CNN, before attempting a lightweight adaptation.

Source: supplied `Another_copy_of_7_Monaural_Localization_PinnaResNet_Final-2.ipynb`.
SHA256: de79152758962453050cde281df76ed3740f2124c6b4dd9eaaaec6d0853f2082.
All four model class definitions in cell 20 were copied with identical Python
ASTs: ResBlock, SqueezeExcite, SpecAugment and PinnaResNet. Parameter count is
1798243. This larger CNN is a diagnostic model, not the selected Pi model.

## Inputs and matched control

The cell-14 frontend is ported to Torch for batched GPU extraction:
- 48 kHz recording, 1024-point Hann STFT, hop 256, reflection-centered frames.
- Slaney mel bank, 64 bands, 20 Hz to Nyquist.
- Apply filters to the COMPLEX STFT, then take magnitude and phase.
- Channels: log10(abs(complex_mel)+1e-10), its temporal delta, cos(phase), sin(phase).
- Delta: width-9, order-1 Savitzky-Golay derivative, nearest padding.
- Compute delta before crop/pad to 256 time frames, preserving notebook order.

phase_zero zeroes only the final two channels; phase retains all four. Both
models are trained from scratch using the same architecture, seed, audio draws,
augmentation and budget. This is not removal of phase only at inference.
The validation feature cache is shared. Test audio/source/direction arrays are
paired across modes and seeds using the existing deterministic core evaluator.

Because the magnitude comes from a complex mel sum, it is itself affected by
phase cancellation. This ablation measures the benefit of EXPLICIT phase
channels; it does not remove every effect of phase from the control.

## Training and evaluation

This reproduces the architecture and frontend, not the notebook's complete
training recipe or reported accuracy. Use the project's current six synthetic
source families, training SNR uniformly 10–40 dB, 4096-sample preroll, 65536-sample
audio windows and core 30 dB test SNR, at the 632 familiar measured directions.
The notebook originally used three noise families, a fixed dataset, 500 maximum
epochs and cosine warm restarts; this comparison instead uses fresh online
examples, 3000 steps, batch 32 and a common OneCycle schedule (peak LR .001).
AdamW weight decay .0001, gradient clipping 5 and cosine direction-vector loss
match the original choices. Both variants select by validation mean angular
error and run the same complete budget, without early stopping. If learning is
still improving at the budget boundary, an inconclusive negative result should
motivate matched longer training, not a universal conclusion about phase.

Default seeds: 0, 1, 2; six sequential runs. About 1 GiB of host RAM is used for
the shared validation feature cache. Inference is chunked in groups of 16 to
limit GPU memory. Outputs include weights, configuration, validation history,
per-source metrics, per-example errors, error histograms and a combined summary.
CUDA is required. Existing output folders are rejected; no training resume.
The smaller batch/budget and different frontend/head mean this is not a clean
architecture-only comparison with the earlier MLP benchmarks.

## Validation of the port

All 23 local tests passed, including both CNN modes' training, prediction and
weight round-trip; exact equality of control versus full first two channels;
finite silence handling; and comparison with notebook-style Librosa features.
The mel-bank coefficients match Librosa 1.0.0 exactly. On the checked random
window, per-channel RMS feature differences were below 1.6e-5; worst difference
was about .0017 in a phase channel where complex mel components nearly cancel.
Tests constrain typical and worst differences separately. Float32 FFT/matrix
implementations are not bit-identical. The Librosa comparison is optional in
cluster tests: Librosa is not required by the experiment itself.

## Run after other GPU experiments finish

From the Kahan repository root with the existing Python environment:

```sh
python -u code/compare_original_cnn_phase.py \
  --sofa 'data/ArtificialPinnaB2_v2 1.sofa' \
  --output code/runs/original_cnn_phase_v1
```

For a two-run pilot only, add `--seeds 0` and use a distinct output folder.
Weights load via PinnaResNet(**checkpoint['model_config']) followed by
load_state_dict(checkpoint['state_dict']); use checkpoint['info']['include_phase']
to choose input channels. These standalone CNN checkpoints are not in the MLP
checkpoint registry. They do not overwrite older CNN or MLP weights.
