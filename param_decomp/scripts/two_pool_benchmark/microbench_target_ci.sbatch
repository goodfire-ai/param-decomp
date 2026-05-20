#!/bin/bash
#SBATCH --job-name=microbench-ci
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/microbench-ci-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/microbench-ci-%j.err
#SBATCH --comment="Single-GPU microbench: breakdown of target+CI forward time at maxA shapes (batch=66 seq=1024, 1 site/rank). Diagnoses where the 78ms in a/1_target_and_ci_fwd goes."

set -euo pipefail
REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"
"$REPO"/.venv/bin/python -m param_decomp.scripts.two_pool_benchmark.microbench_target_ci
