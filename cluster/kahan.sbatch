#!/bin/bash
#SBATCH --job-name=pinna
#SBATCH --output=/zooper2/%u/pinna-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gpu24:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
# Submit from the repository root. Override PINNA_PYTHON with the absolute
# interpreter path from your working Kahan Jupyter kernel if appropriate.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
case "$PWD/" in
  /zooper2/"$USER"/*) ;;
  *) echo 'Repository must be under /zooper2/$USER' >&2; exit 1 ;;
esac
: "${PINNA_SOFA:?Set PINNA_SOFA to your SOFA file under /zooper2/$USER}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLBACKEND=Agg
python_bin="${PINNA_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Python missing: $python_bin. Follow cluster/README.md first." >&2
  exit 1
fi
"$python_bin" -u code/run_experiment.py --require-cuda --sofa "$PINNA_SOFA" "$@"
