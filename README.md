# Artificial Pinna Network (Senior Project)

Neural decoding and real-time deployment for monaural sound source localization with an artificial pinna: a single microphone inside a passive structure whose direction-dependent filtering lets a neural network estimate the direction of arrival.

Cooper Union ECE Senior Project, 2026-2027.

**Start here:** [`code/`](code/) holds the direction decoder: one notebook (`code/pinna_decoder.ipynb`) for data, preprocessing, training and evaluation, the `pinna` modules it calls, and the results so far.

**Isabel's phase experiments:** [`cluster/README.md`](cluster/README.md) explains
the magnitude baseline, experimental phase features, and Kahan batch jobs.
This branch adds experiment infrastructure; improved localization accuracy has
not yet been demonstrated.
