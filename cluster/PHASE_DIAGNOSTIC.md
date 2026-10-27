# Phase feature diagnostic

Run `code/diagnose_phase.py --sofa PATH --output NEW_DIRECTORY --require-cuda`
inside the existing GPU allocation. Defaults: 18 measured horizon directions,
8 independent train draws and 8 test draws per direction for each of six synthetic
source families, 30 dB SNR, 4096-sample source preroll and 65536-sample windows.
Training/test draws have separate fixed random seeds. No clean source is an input.

For each representation, training data fit a feature center and a class-mean
template. Independent test examples choose the nearest centered cosine template.
Representations: magnitude, compact phase-only, locally pooled temporal phase-only,
and time means of the temporal phase channels. A second evaluation shifts the
same test analysis windows by 37 samples (0.77 ms), without shifting templates.

Outputs: JSON results including confusion matrices, source-specific accuracy and
uncorrected between/within feature variance ratio; plus an accuracy comparison PNG.
The ratio includes finite-sample noise in class means. Representations have very
different dimensions and statistical properties, so it is descriptive, not an
information-theoretic test or a directly calibrated comparison.

Important limits:
- This is an exploratory direction-template diagnostic, not a trained-network
  comparison or bound on possible localization accuracy.
- All directions are measured horizon positions. Results do not establish unseen
  direction or real-source generalization.
- Phase-derived features retain magnitude-dependent masking/weighting. Their
  direction accuracy does not prove phase provides information independent of
  magnitude; that requires a controlled model/feature ablation.
- Poor accuracy can reflect a weak template classifier rather than absent cues.
- Window-shift accuracy is not general invariance to source onset or arrival delay.
- IR energy width is reported to compare time scales. An STFT can encode delays
  shorter than its window/hop through phase; a coarse window alone does not prove
  that sub-millisecond information is absent.
