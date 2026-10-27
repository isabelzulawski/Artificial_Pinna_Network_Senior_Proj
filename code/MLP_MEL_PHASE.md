# Direct complex-mel phase features in the lightweight MLP

This experiment adapts the earlier notebook's PHASE REPRESENTATION to an MLP.
It does not construct or train PinnaResNet or any convolutional model.

Keep the partner MLP's original 191 magnitude features, classification head,
spherical soft-label cross entropy, training sources and direction decoder.
Use the earlier notebook's 1024-point complex STFT -> 64 complex mel bands ->
cos(phase) and sin(phase) for the extra inputs. No neighboring-frequency phase
difference or reference-frequency subtraction is used.

The earlier phase map has 2*64*256=32768 values. Directly flattening that into
the existing 256-unit first layer would add over eight million weights. For
this explicitly lightweight adaptation, apply fixed average pooling to an
8-frequency-group by 8-time-region map per channel: 128 phase features. This
is lossy compression of direct phase channels and does not reproduce all the
CNN's time/frequency resolution. Encode sine/cosine before pooling, so wrapped
angles are not averaged arithmetically. No learned phase encoder is added.

Final input width: 191+128=319. Both hidden layers are 232 units instead of 256
to keep the overall budget: 276190 parameters versus the original 277750.
Linear MACs: 274456 versus 276224. These counts exclude the added STFT, complex
mel projection and pooling; frontend compute and Pi power still need measurement.

Two modes:
- mlp_mel_phase: magnitude features plus compressed direct phase channels.
- mlp_mel_zero: same 319-input / 232-hidden MLP with the 128 extra slots zeroed.

Both are trained from scratch. Compare these matched models to isolate the
contribution of the supplied phase values; compare the resulting budget-sized
model to the original MLP separately. The narrower hidden layers and extra
zero slots affect model capacity/normalization, so the zero-input control is
important. Both select checkpoints by validation mean angular error, retain
original CE training (no angular auxiliary cost), and use 3000 steps/batch256.
The default is a seed-0 two-model pilot, with new output directories and logs.
Use --seeds 0 1 2 for confirmation only after assessing the pilot on validation.

```sh
python -u code/compare_mlp_mel_phase.py \
  --sofa 'data/ArtificialPinnaB2_v2 1.sofa' \
  --output code/runs/mlp_mel_phase_v1
```

All 25 local tests passed after this addition, including direct comparison
against pooled original phase channels, unchanged magnitude features, finite
silence handling, parameter count, MLP class construction, short training and
checkpoint/predictor round trips. The older CNN diagnostic experiment is separate
and its error is not evidence about this MLP adaptation. No GPU accuracy result
for this new adaptation has been established yet.
