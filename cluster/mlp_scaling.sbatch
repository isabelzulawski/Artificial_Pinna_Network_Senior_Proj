#!/bin/bash
#SBATCH --job-name=pinna-scaling
#SBATCH --output=/zooper2/%u/pinna-scaling-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gpu24:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
: "${PINNA_SOFA:?Set PINNA_SOFA}"
: "${PINNA_PYTHON:?Set PINNA_PYTHON}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLBACKEND=Agg
# One allocation, serial trials: never more than one GPU job in this sweep.
"$PINNA_PYTHON" -u code/sweep_mlp_scaling.py --sofa "$PINNA_SOFA" --require-cuda "$@"
