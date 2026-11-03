#!/bin/bash
# Run on the Kahan login node, from the repository root.
set -euo pipefail
: "${PINNA_SOFA:?Set PINNA_SOFA to the absolute SOFA path}"
: "${PINNA_PYTHON:?Set PINNA_PYTHON to your working Python interpreter}"
tag="$(date -u +%Y%m%dT%H%M%SZ)-$("$PINNA_PYTHON" -c 'import uuid; print(uuid.uuid4().hex[:8])')"
root="$PWD/code/runs/mlp_scaling_$tag"
smoke="${root}_smoke"
smoke_job=$(sbatch --parsable --time=00:20:00 cluster/mlp_scaling.sbatch --smoke --output "$smoke")
smoke_job="${smoke_job%%;*}"
full_job=$(sbatch --parsable --dependency="afterok:$smoke_job" cluster/mlp_scaling.sbatch --output "$root")
printf 'Smoke job: %s\nFull sweep job: %s\nResults: %s\n' "$smoke_job" "$full_job" "$root"
printf 'If smoke fails, the full sweep will not start. Inspect the smoke log before resubmitting.\n'
