# Temporal phase correction experiment

`--mode temporal_phase` adds a small CNN correction to a frozen magnitude-only
classifier. It requires `--baseline-checkpoint` from the SAME direction protocol.
The loader verifies class vectors and preroll. Baseline checkpoint SHA256 is saved.
The zero-initialized correction starts with exactly the baseline logits.
Magnitude and phase processing remain separate and magnitude weights are frozen.
This tests correction of a pretrained model, not equal-budget training from scratch.

The four phase channels retain 64 local time regions across the original 2–20 kHz
bins: circular frequency differences and temporal phase advances, real/imaginary
parts, with low-energy masking and amplitude-product weighting. Only expected
bin-center temporal advance is removed. These are source-dependent microphone
features, not the clean pinna phase response, and remain sensitive to timing.
Boundary STFT frames are omitted.

`--mode temporal_zero` uses identical architecture and magnitude inputs but zeros
all phase channels. Both modes can learn magnitude-conditioned corrections;
compare them to distinguish phase information from added training/capacity.

Both temporal modes select checkpoints by validation mean angular error, including
step zero as a fallback. Only the retained VALIDATION mean is guaranteed to be no
worse than step zero; test performance may deteriorate. Existing non-temporal
modes still select by validation loss. The initial baseline is already selected on
this validation set, so independent final evaluation is particularly important.

Inside the Kahan GPU Jupyter terminal after `git pull --ff-only`:

```bash
for mode in temporal_phase temporal_zero; do
  /zooper2/isabel.zulawski/jupyter_venv/bin/python3 -u code/run_experiment.py \
    --sofa "data/ArtificialPinnaB2_v2 1.sofa" \
    --mode "$mode" --head classification --protocol seen --seed 0 \
    --baseline-checkpoint code/runs/magnitude_seen_s0/model.pt \
    --steps 3000 --eval-every 250 --batch 64 --lr 0.001 --require-cuda \
    --output "code/runs/${mode}_seen_s0" || break
done
```

The frozen classifier and correction are saved together. Inference does not need
the original checkpoint file. Temporal tensors are larger, so validation uses
chunks of at most 128. Reduce training batch size if GPU memory is insufficient.
For held-out protocols use their corresponding baseline checkpoints. This model
is not selected by `--head regression`; the temporal model uses classification.

Architecture choices should use validation behavior. Existing tests are development
benchmarks; final accuracy needs untouched recordings/measurements. Timing/crop
randomization and physical microphone tests remain necessary follow-ups.
