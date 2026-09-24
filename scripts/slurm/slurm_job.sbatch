#!/usr/bin/env bash

set -euo pipefail

repository_root="$SLURM_SUBMIT_DIR"
python="$repository_root/.venv/bin/python"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

cd "$repository_root/src"

"$python" run_job.py "$SLURM_ARRAY_TASK_ID"
