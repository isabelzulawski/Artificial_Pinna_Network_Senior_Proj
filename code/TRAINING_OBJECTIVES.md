# Same-size MLP training experiments

Four conditions, three seeds each, fixed before looking at results:

1. baseline: existing magnitude MLP and spherical soft-label cross entropy.
2. paired_control: same loss, 128 sampled directions each with two different
   source families (256 audio windows total per step).
3. source_contrastive: identical paired recipe plus 0.1 times supervised
   contrastive loss on the existing 256-dimensional hidden activations.
4. angular: baseline sampling plus 1.0 times expected class angular distance/pi.

All use the same 191 magnitude features, two 256-unit hidden layers, 632-way
classification head, 277750 parameters, original local decoder and preprocessing.
There is no projection head or extra inference step. Losses operate only during
training. Every condition uses batch 256, 3000 optimizer steps, the same sources,
training SNR range, preroll, optimizer and learning-rate schedule.

The paired control separates a change in batch sampling from the contrastive
objective itself. Same-direction sounds are positives. Different directions at
least 15 degrees apart are negatives; closer different directions are excluded
from the contrastive denominator to avoid unnecessarily separating neighbors.
Temperature is 0.1. Some source observations may be ambiguous; invariance is a
training hypothesis rather than a claim that every sound identifies direction.

The angular cost is sum_c p(c) angle(v_c,v_true)/pi, added to existing spherical
soft-label cross entropy. It targets distant probability mass, not the angle of
a global vector average, and leaves the deployed decoder unchanged. This is an
extension of the existing direction-aware target, not the first use of geometry.

For this comparison, all four conditions select checkpoints by validation mean
angular error (--select-by angular), unlike old magnitude runs selected by
validation cross entropy. A freshly trained baseline is therefore required;
compare against that baseline first. Validation cross entropy is still recorded
and is not numerically comparable with the combined training loss. Original CLI
defaults and old model checkpoints are preserved.

Run from the repository root with the working GPU interpreter:

```sh
python -u code/compare_training_objectives.py \
  --sofa 'data/ArtificialPinnaB2_v2 1.sofa' \
  --output code/runs/training_objectives_v1
```

Twelve sequential runs; progress.json updates after each. summary.json retains
three-seed means and seed standard deviations, full per-source results and error
tails. All test source/direction index arrays must match, and parameter count is
checked. Same protocol/CPU test generator produces the same synthetic waveforms.
No hyperparameter sweep or test-based selection is performed by this script.
No claim of real-recording, held-out-source-category, underwater or Pi performance.

Method inspiration: Khosla et al., Supervised Contrastive Learning (2020),
https://arxiv.org/abs/2004.11362. The direction-neighbor exclusion and paired
synthetic-source setup here are task-specific adaptations, not paper results.
