# Phase experiments on Kahan

This branch adds a headless runner and optional features while leaving the upstream
notebook's default magnitude model and zero-preroll training behavior intact.
No accuracy improvement has been established. GPU execution must be checked on Kahan.

## Get your branch onto Kahan

SSH to `kahanctrl.ee.cooper.edu` with your uLab username; off campus use port 31415.
These commands run in the Kahan terminal:

```bash
mkdir -p /zooper2/"$USER"
cd /zooper2/"$USER"
git clone --branch codex/phase-features https://github.com/isabelzulawski/Artificial_Pinna_Network_Senior_Proj.git
cd Artificial_Pinna_Network_Senior_Proj
git remote add upstream https://github.com/joshmiao1065/Artificial_Pinna_Network_Senior_Proj.git
mkdir -p data
```

Upload the SOFA file into `data/`. It is ignored by Git. On future visits use
`git pull --ff-only` from this branch before submitting jobs, with no active jobs
using that checkout. Commit/push code changes to synchronize your laptop and Kahan.

## Choose a working Python environment

In your existing GPU Jupyter notebook, inspect the working interpreter:

```python
import sys, torch
print(sys.executable)
print(torch.__version__, torch.cuda.is_available())
```

If that interpreter is available on the compute node under `/zooper2/$USER`, set
`PINNA_PYTHON` to its absolute path. Browser Jupyter and Slurm may have different
environments. The cluster guide's default packages do not establish GPU PyTorch availability.

Otherwise allocate an interactive compute shell and prepare an environment there:

```bash
srun --nodes=1 --mem=8G --gres=gpu:gpu24:1 --pty bash
cd /zooper2/"$USER"/Artificial_Pinna_Network_Senior_Proj
python3 --version
python3 -m venv .venv
source .venv/bin/activate
```

Use Python 3.12 for the upstream pinned requirements. Install the CUDA PyTorch
build supported by the cluster driver, then the non-Torch dependencies. Do not
replace a working cluster PyTorch build just to match a version pin. The upstream
`code/requirements.txt` records its original versions; consult
https://pytorch.org/get-started/locally/ when selecting a compatible CUDA wheel.

If PyTorch is already available in your selected environment, install only missing
non-Torch packages with that same interpreter:

```bash
python -m pip install numpy scipy h5py matplotlib
python -c 'import torch,numpy,scipy,h5py,matplotlib; print(torch.__version__); print(torch.cuda.is_available()); assert torch.cuda.is_available()'
```

If the usable environment is container-only, the supplied native-environment batch
script needs adaptation. Use Kahan's `podman-slurm` on a compute allocation and
mount `/zooper2/$USER`; the guide's CUDA compiler image alone does not include
PyTorch. Do not run GPU training on the login node.

## First run

From the repository root in your login shell:

```bash
export PINNA_SOFA="$PWD/data/ArtificialPinnaB2_v2 1.sofa"
# Optional if using an existing shared interpreter:
# export PINNA_PYTHON=/absolute/path/under/zooper2/to/python
sbatch cluster/kahan.sbatch --mode magnitude --steps 5 --eval-every 5 --batch 8 --output code/runs/smoke
squeue -u "$USER"
```

The smoke run still performs the core test battery; it checks execution, not
accuracy. Logs are `/zooper2/$USER/pinna-JOBID.out`; use `tail -f` to follow them.
Existing output directories are rejected to protect prior runs.

## Matched comparison

After the smoke run succeeds, submit the following serial jobs (one GPU allocation
at a time) or run one at a time manually. Repeat with seeds 1 and 2:

```bash
sbatch cluster/kahan.sbatch --mode magnitude --seed 0 --protocol seen --output code/runs/magnitude_seen_s0
sbatch cluster/kahan.sbatch --mode phase --seed 0 --protocol seen --output code/runs/phase_seen_s0
sbatch cluster/kahan.sbatch --mode phase_zero --seed 0 --protocol seen --output code/runs/phase_zero_seen_s0
```

The commands are independent submissions: Slurm may run them concurrently if you
submit all three immediately. Submit the next only after the previous completes
if you want one allocation at a time.

Repeat for `--protocol unseen_el` and `--protocol unseen_az`, using distinct output
names. All modes default to 4096 samples of source preroll during training and
validation; the existing test battery also uses 4096. For upstream reproduction use
`--mode magnitude --preroll 0`. Keep that run separate from the matched comparison.

Phase features append energy-weighted circular summaries of neighboring-frequency
phase differences to the unchanged magnitude features. The `phase_zero` mode uses
the same input width/model capacity but zeros the additional inputs. It does not
replace tests with randomized start timing, unseen source recordings, and new
physical measurements. Frequency phase differences remain sensitive to delay.

Default evaluation is the core battery; `--full-battery` runs all existing stress
conditions. It is a diagnostic benchmark: repeated tuning on these tests makes
them development data. Keep new source recordings/measurement sessions untouched
for final evaluation. Synthetic source families are not held out by these commands.

Each run saves the checkpoint (including feature mode and class vectors), exact
configuration, Git state, SOFA hash, package versions, training history, metrics,
per-example predictions/errors, and a PNG/PDF error histogram with mean, median,
P95 and the 5-degree target. There is no interrupted-training resume yet: a timed-out
training run must be restarted, and the checkpoint is written after fitting finishes.

Kahan-specific settings follow the supplied July 2026 guide: `/zooper2/$USER`,
Slurm, and `gpu24`. Shell variables do not expand in `#SBATCH` directives;
the log path uses Slurm's `%u` and `%j` substitutions instead.
