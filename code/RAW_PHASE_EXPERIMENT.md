# Raw phase with a reference

This MLP appends one wrapped angle for each of 192 frequency locations to the
original 191 magnitude features: 383 inputs total. The original magnitude
representation uses 191 differences between 192 bins; the phase representation
retains all 192 locations. Two hidden layers of 256 units are unchanged.

For the entire 65,536-sample microphone window, apply a Hann window and a real
FFT. At the FFT bin nearest the requested reference frequency, obtain its phase.
At each FFT bin nearest an original 2–20 kHz magnitude-bin center, compute
`wrap(angle(X[f]) - angle(X[ref])) / pi`. The input is a scalar angle in [-1,1],
not sine/cosine channels. No circular averaging over STFT frames or phase
unwrapping is used in this experiment. FFT bin rounding is deterministic; the
actual reference frequency is saved in environment.json. It remains dependent
on source phase and recording timing.

For silence or weak bins (below -80 dB power relative to the FFT peak), phase
features are zero-filled. If the reference is weak, all phase values are zero.
No mask features are appended. Zero is thus ambiguous between a valid zero
phase difference and an invalid value; this is a deliberate compact first test.

Modes:
- raw_phase: 383 inputs, including 192 scalar phase angles.
- raw_phase_zero: identical input size with zero-filled phase slots.
- raw_phase_sincos: matched-reference encoding control, 575 inputs. This is
  separate from the raw-angle model and uses the same FFT, mask and reference.

The full-window FFT introduces extra preprocessing and differs from the older
adjacent-frequency STFT circular summaries. Comparisons with the old phase mode
do not isolate angle encoding; raw_phase_sincos provides the matched comparison.

Run a coarse sweep:

```sh
python -u code/sweep_phase_reference.py \
  --sofa 'data/ArtificialPinnaB2_v2 1.sofa' \
  --output code/runs/raw_reference_sweep_v1
```

Default reference candidates: 250, 500, 1000, 2000, 4000, 8000 Hz. Each trains
3000 steps using seed 0 and the seen-direction protocol. Selection ranks
validation mean angular error from each retained checkpoint, which training
selects by validation loss. Test scores are recorded but not used to pick the
reference. After locking the choice, run its zero-phase control at seed 0,
confirm raw-angle performance at seeds 1 and 2, and run the matched sine/cosine
control at seed 0. Save progress.json, selection.json and summary.json plus
per-run checkpoints, metadata, distributions and logs. Requires CUDA; no GPU
training silently falls back to CPU. Existing output directories are rejected.

This finds the leading candidate in a small, exploratory sweep, not an optimal
reference for all sounds/directions. Validation uses familiar directions and
synthetic source families. Held-out-direction and real-recording validation
remain separate work. Original magnitude baseline runs are unchanged.
