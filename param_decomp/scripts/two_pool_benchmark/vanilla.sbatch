#!/bin/bash
#SBATCH --job-name=vanilla
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/vanilla-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/vanilla-%j.err
#SBATCH --comment="Vanilla DDP-8 SPD baseline using ComponentModel + PersistentPGDState — apples-to-apples baseline for the 2-pool benchmark (same PPGD config, same losses, same target/CI model)."

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

"$REPO"/.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m param_decomp.scripts.two_pool_benchmark.vanilla
