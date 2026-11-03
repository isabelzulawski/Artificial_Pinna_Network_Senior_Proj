# Magnitude MLP scaling on Kahan

Branch: `experiment/mlp-scaling`, based on `codex/phase-features`.

## Protocol

The screen runs all nine combinations of hidden widths **256, 512, 1024**
and optimizer budgets **3000, 6000, 12000**, seed **0**. Each MLP has the existing
two hidden layers. The 256/3000 condition is the baseline anchor. Existing runs,
`run_experiment.py` defaults, feature extraction, phase code, soft-label CE loss,
AdamW settings, peak LR 0.002, batch 256, seen-direction split, validation seed
90004, six source families, and preroll 4096 are preserved. Validation is evaluated
every 250 updates; early stopping is disabled so each budget is actually completed.
The sole intentional selection change is lowest validation mean great-circle error
(with the existing 1e-4 improvement tolerance), rather than validation CE.
The fixed validation features are shared across the entire sweep.

The screen **does not run the test battery**. After all nine runs, `selected.json`
freezes the three lowest validation-error configurations (ties: smaller parameter
count, then fewer steps). Each is trained afresh with seeds **0, 1, 2** in
`confirm/`, including a fresh deterministic seed-0 repeat, and evaluated using the
existing core test battery. Test numbers are descriptive and never used for
selection. This is 18 full training trials: nine screen and nine confirmation.
The same validation set is used for confirmation; it is not an independent final
holdout. The existing synthetic test has already been inspected during earlier
work; independent physical recordings remain necessary for final validation.

## Checkpoints and schedule fairness

Every validation interval saves an atomic `latest.pt` containing current model,
AdamW, OneCycle scheduler, best model, history, cumulative training seconds,
NumPy generator, synthetic-wave generator, and Torch CPU/CUDA random states.
`model.pt` contains the best validation model for inference. These are different.

OneCycle's warmup and decay depend on the **total** budget. A finished 3000-step
schedule cannot be reused as the first half of an otherwise identical 6000-step
schedule. Each budget therefore starts independently. Interruption recovery reuses
only the same configuration/horizon. Changed code/SOFA/configurations are rejected
on resume; completed trials are skipped. Up to 249 updates may be lost after a
termination. No partial or existing experiment directory is silently overwritten:
new launches require a fresh directory, while `--resume` explicitly continues the
same run and refreshes its derived reports. Do not run two processes against one
output root. Keep code unchanged during the run and on resume.

## Launch (Kahan login terminal)

Finish or stop any jobs using your checkout before switching it. If Git reports
local edits, preserve them before switching; do not reset or clean the repository.

```bash
cd "/zooper2/isabel.zulawski/senior projects/Artificial_Pinna_Network_Senior_Proj"
git fetch origin
git switch --track origin/experiment/mlp-scaling
export PINNA_PYTHON=/zooper2/isabel.zulawski/jupyter_venv/bin/python3
export PINNA_SOFA="$PWD/data/ArtificialPinnaB2_v2 1.sofa"
bash cluster/submit_mlp_scaling.sh
squeue -u "$USER"
```

If the branch already exists locally, use `git switch experiment/mlp-scaling`
and `git pull --ff-only origin experiment/mlp-scaling` instead of `--track`.
No dependency installation should be needed in the previously working environment.

The submission helper creates unique UTC timestamp + UUID output names. It first
submits a 20-update GPU smoke job, then one full sweep job with an `afterok`
dependency. At most one GPU job from these submissions runs at a time. All trials
within the full allocation execute serially. Other unrelated jobs are not stopped.
Smoke checks execution, not accuracy. Smoke failure prevents the sweep starting.
Log files: `/zooper2/$USER/pinna-scaling-JOBID.out`.
The full job requests 12 hours; adjust `sbatch --time` if Kahan's limits require it.
This is not a measured runtime estimate for the larger models.

## Results

The submit helper prints the result root. Replace `YOUR_RESULT_ROOT` below with
that absolute path:

```bash
cat "YOUR_RESULT_ROOT/summary.md"
# Regenerate the table without training or GPU work:
"$PINNA_PYTHON" code/sweep_mlp_scaling.py --output "YOUR_RESULT_ROOT" --summarize-only
# Recover a failed/timed-out full sweep, AFTER the previous job has stopped:
sbatch cluster/mlp_scaling.sbatch --output "YOUR_RESULT_ROOT" --resume
```

`summary.md`/`summary.json`: per-trial validation angle, parameter count, best step,
training seconds, and confirmation-only test mean; confirmation validation means
and sample SDs across completed seeds. Compare validation results when choosing
what to deploy. `selected.json`: frozen top-three screen configurations.
`manifest.json`: SOFA/code hashes, Torch version, device, Git commit and job ID.
`directions.json`: split indices and validation seed.

Each `screen/wWIDTH_stepsSTEPS_s0/` or `confirm/wWIDTH_stepsSTEPS_sSEED/` contains
`config.json`, append-only `train.log`, `training.json` (loss, validation angular
error, learning rate, timing curves), `curves.png`, `latest.pt`, `model.pt`, and
`result.json`. Confirmation also has the existing `metrics.json`, `examples.npz`, and PNG/PDF
error-distribution histograms.
Training time includes validation and feature creation in `fit`, excludes test
battery and shared validation preprocessing, and accumulates across resumes.
The screen may favor occasional low validation checkpoints with a larger budget;
report multi-seed results and size/time costs alongside any apparent gains.

## Local verification (2026-11-02)

A 20-update CPU smoke on the supplied 632-direction SOFA passed, writing all
training artifacts and the summary. It retained 277,750 parameters. Its ~69°
validation error is expected for this tiny execution test and is not a scaling
result. Local Torch was 2.15.0; Kahan GPU behavior is checked separately by the
Slurm smoke dependency. Tests verify exact interrupted/uninterrupted training
agreement (including losses/LR curves and best weights), reject a changed schedule
horizon, and check validation-only top-three selection plus all nine fresh
confirmation runs. The earlier suite passed 26 tests with one optional librosa
reference check skipped; the added sweep selection test also passed.
