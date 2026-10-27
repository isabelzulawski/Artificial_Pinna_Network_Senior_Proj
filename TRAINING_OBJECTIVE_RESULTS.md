# Same-size MLP training results

Completed on Kahan; reported console output supplied by the user. Code commit
7daa7527bb13a107434520cd05cfc22412564229. Full artifacts remain under
code/runs/training_objectives_v1 on Kahan. Twenty automated tests and an
end-to-end source-contrastive smoke run passed locally before these experiments.

Each condition uses 191 magnitude features, the same 277750-parameter MLP,
identical inference code, 3000 training steps and batch 256. All select checkpoints
by validation mean angular error. This freshly trained baseline is the correct
comparison because earlier runs selected by validation cross entropy.

| Training | Three-seed test mean | Sample SD across seed means |
|---|---:|---:|
| Baseline | 2.7010° | 0.0731° |
| Paired-source control | 2.7854° | 0.0593° |
| Source contrastive | 2.7529° | 0.0280° |
| Angular cost | 2.6459° | 0.0367° |

Angular-cost test errors by seed: 2.604, 2.669, 2.665 degrees. Corresponding
baseline test errors: 2.626, 2.706, 2.772 degrees (rounded console values).
The aggregate reduction is 0.0551 degrees, about 2.04%, with lower test means
in all three seeds. This is a small development-test gain, not a substantial
accuracy improvement or a statistical-significance claim. Validation errors
were 4.061/4.469/4.632 for angular and 4.038/4.435/4.637 for baseline; validation
did not consistently favor angular cost. New evaluation data is needed to
assess whether the small test gain generalizes.

Contrastive training improves slightly over its paired-source control but does
not beat the ordinary baseline. Pairing changed the sampling recipe, so it is
important to retain both comparisons.

These runs do not establish performance on real recordings, unseen source
categories, underwater sound, or the Raspberry Pi. Inference parameter and
operation counts are unchanged; hardware latency and power were not measured.
