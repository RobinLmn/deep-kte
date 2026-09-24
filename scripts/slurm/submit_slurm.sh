#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_directory/../.." && pwd)"
run_id="${1:-run_$(date +%Y%m%d_%H%M%S)}"
outer_reps="${CNME_OUTER_REPS:-20}"
account="${SLURM_ACCOUNT:-}"
qos="${SLURM_QOS:-}"
gpu_partitions="${SLURM_GPU_PARTITIONS:-gpu}"
time_limit="${SLURM_TIME_LIMIT:-}"
gpu_gres="${SLURM_GPU_GRES:-gpu:1}"
gpu_exclude_nodes="${SLURM_GPU_EXCLUDE_NODES:-}"
gpu_parallel="${SLURM_GPU_PARALLEL:-8}"
job_count="$(grep -c '^[[:space:]]*job(' "$repository_root/src/run_job.py")"
gpu_jobs="${SLURM_GPU_JOBS:-0-$((job_count - 1))}"
log_directory="$repository_root/out/$run_id/slurm"
common_options=(--parsable)
gpu_options=(--partition="$gpu_partitions" --gres="$gpu_gres")

if [[ -n "$account" ]]
then
  common_options+=(--account="$account")
fi
if [[ -n "$qos" ]]
then
  common_options+=(--qos="$qos")
fi
if [[ -n "$time_limit" ]]
then
  common_options+=(--time="$time_limit")
fi
if [[ -n "$gpu_exclude_nodes" ]]
then
  gpu_options+=(--exclude="$gpu_exclude_nodes")
fi

mkdir -p "$log_directory"
cd "$repository_root"

gpu_submission="$(sbatch "${common_options[@]}" "${gpu_options[@]}" --job-name=deep-kte-gpu --cpus-per-task=8 --mem=32G --array="$gpu_jobs%$gpu_parallel" --output="$log_directory/gpu-%A_%a.out" --export="ALL,CNME_RUN_ID=$run_id,CNME_OUTER_REPS=$outer_reps,CNME_SEED_OFFSET=0" "$script_directory/slurm_job.sbatch")"
echo "GPU array: $gpu_submission"

echo "Results: $repository_root/out/$run_id"
