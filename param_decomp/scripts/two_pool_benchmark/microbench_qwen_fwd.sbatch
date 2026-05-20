#!/bin/bash
#SBATCH --job-name=microbench-qwen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/microbench-qwen-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/microbench-qwen-%j.err
#SBATCH --comment="Probe Qwen3-0.6B-Base forward time across batch/seq shapes to pick the 2-pool bench config."

set -euo pipefail
REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"
"$REPO"/.venv/bin/python -m param_decomp.scripts.two_pool_benchmark.microbench_qwen_fwd
