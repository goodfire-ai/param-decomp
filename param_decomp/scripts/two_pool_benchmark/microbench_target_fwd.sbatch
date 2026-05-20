#!/bin/bash
#SBATCH --job-name=microbench-fwd
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/microbench-fwd-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/microbench-fwd-%j.err
#SBATCH --comment="Single-GPU microbench: TinyTransformer forward at maxA shapes, compare baseline / bf16 / torch.compile / both."

set -euo pipefail
REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"
"$REPO"/.venv/bin/python -m param_decomp.scripts.two_pool_benchmark.microbench_target_fwd
