#!/bin/bash
#SBATCH --job-name=2pool-wider
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/2pool-wider-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/2pool-wider-%j.err
#SBATCH --comment="2-pool benchmark wider: 14 pool-A ranks (1 site-group each, 3 sites/rank) + 2 pool-B DP-2 = 16 GPUs, validates that the prior 14-sites/rank pool-A was the bottleneck"

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29510

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \
    --nnodes=2 \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=8 \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m param_decomp.scripts.two_pool_benchmark.two_pool_wider
'
