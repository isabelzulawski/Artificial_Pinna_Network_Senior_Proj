# Direction decoder for the artificial pinna

A single microphone inside a 3D-printed asymmetric pinna: the pinna filters each sound differently depending on its
direction, and a small neural network reads the direction back from the spectrum at the microphone. Everything (data,
preprocessing, training, evaluation) runs from one notebook, **`pinna_decoder.ipynb`**, which calls the modules in
`pinna/`. The committed notebook includes its outputs, so the results can be read on GitHub without running anything.

## Quick start
1. Python 3.12 environment: `python -m venv .venv`, activate it, then install as described in `requirements.txt`
   (torch first, matching your GPU or CPU).
2. Get the measured pinna responses (the SOFA file `ArtificialPinnaB2_v2`; it is not in the repository, ask Josh).
   Copy `local_paths.example.json` to `local_paths.json` (untracked) and put the file's path in it, or set the
   environment variable `PINNA_SOFA`.
3. Open `pinna_decoder.ipynb` from this folder and run all cells. With a GPU it takes about 5 minutes (measured on one
   slice of an RTX PRO 6000); on a CPU, training is much slower. Set `QUICK = True` to run only the core test. Outputs go to `runs/<protocol>_s<seed>/`
   (untracked).

## Layout
| Path | What |
|---|---|
| `pinna_decoder.ipynb` | The pipeline: parameters, data, preprocessing, model, training, evaluation, summary |
| `pinna/config.py` | Paths (SOFA file, `runs/`), device, constants (48 kHz, 1.37 s window, source types) |
| `pinna/data.py` | Load the SOFA file; unit vectors; direction splits (`seen`, `unseen_el`, `unseen_az`) |
| `pinna/simulate.py` | The 6 source types and the forward model (source * pinna response, level, gain, sensor noise) |
| `pinna/features.py` | Input features: frequency-differenced Welch spectrum, 2-20 kHz (191 numbers) |
| `pinna/model.py` | The MLP classifier, soft-label loss, decoder, save and load |
| `pinna/train.py` | Training with a fixed validation set, early stopping and best-checkpoint selection |
| `pinna/evaluate.py` | The test battery (73 conditions on `seen`) and its metrics |
| `pinna/plots.py`, `pinna/report.py` | Figures; tables and saved run outputs |
| `reference_results.json` | Earlier results of the same test battery (this design over 3 seeds; the previous team's CNN) |

## The pipeline
- **Data.** 632 measured directions (18 azimuths every 20 deg x 35 elevations every 5 deg, plus the poles).
  Microphone signals are simulated: a source (white, pink or band noise, harmonic, noise bursts, click trains) is
  convolved with the response of its direction, normalized, given a random gain of ±3 dB, and sensor noise is added
  (SNR 10-40 dB).
- **Features.** Welch power spectrum over the 1.37 s window, 2-20 kHz, log, mean removed, differenced along
  frequency. Loudness and smooth source slopes drop out; the pinna's peaks and notches remain.
- **Model.** LayerNorm -> 256 -> 256 (GELU) -> one output per training direction, trained with soft labels
  (neighbouring directions share the target) and decoded as a probability-weighted mean near the peak, so it can
  answer between grid points. 278 k parameters, 0.28 M multiply-accumulates per window.
- **Training, validation, test.** Training draws new examples every step (or a fixed set: `TrainConfig.train_pool`).
  A fixed validation set (own seed) gives the curves, early stopping and the kept checkpoint. The test battery uses
  other seeds and is never used for a choice.
- **Test battery.** Every condition has a fixed CPU seed, so test waves are identical on any machine and for every
  model. Groups: core (6 source types), octave bands, low/high-pass, brick-wall cut-off, SNR, level, window length,
  spectral tilt, random EQ, diffuse reverb, a floor reflection, a perturbed pinna response, a frequency-shifted pinna.
  Metrics: great-circle error (mean, median, P95, % within 5 / 10 deg, gross errors), the previous paper's metrics
  (Az-MAE, El-MAE, bucket accuracies), mirror confusions, error by elevation, confidence.

## Results so far (reference runs, `reference_results.json`; mean ± sd over seeds)
| Test | This MLP (3 seeds) | Previous team's CNN |
|---|---|---|
| Core, seen directions, 30 dB | 2.80 ± 0.05 deg | 25.1 ± 0.7 deg (3 seeds) |
| Held-out elevations (5 deg from training) | 5.38 ± 0.09 deg | 31.4 deg (1 seed) |
| Held-out azimuth columns (20 deg from training) | 42.6 ± 1.4 deg | 49.2 deg (1 seed) |
| Compute per window, one CPU thread | 0.9 ms, 0.28 MMAC | 52 ms, 1,559 MMAC |

The CNN was retrained with the previous team's own recipe, which used white, pink and band noise only. On white and
pink noise it scores 3.2 deg (MLP 0.8) and on band noise 21 deg (MLP 8.2), so much of the gap comes from the source
types it never saw (harmonic 59 deg, click trains 53 deg). Open problems: rooms (reverb and
reflections degrade every model sharply) and directions between the 20 deg azimuth columns (the grid is too coarse
to learn them). All sources are synthetic so far.

## Working on it
- Compare designs over 3 or more seeds, on the same test battery, and make choices on the validation set, never on
  the tests.
- Keep reusable code in `pinna/` and the notebook thin. Do not commit `runs/`, data, SOFA files or weights.
