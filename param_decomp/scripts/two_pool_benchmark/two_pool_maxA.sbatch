#!/bin/bash
#SBATCH --job-name=2pool-maxA
#SBATCH --nodes=6
#SBATCH --gpus-per-node=8
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/2pool-maxA-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/2pool-maxA-%j.err
#SBATCH --comment="2-pool max-pool-A: 42A (1 site/rank) + 6B (DP-6, one per node) across 6 nodes = 48 GPUs. Push pool A parallelism to the limit on the 6-block model."

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29511

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun --nodes=6 --ntasks=6 --ntasks-per-node=1 bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \
    --nnodes=6 \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=8 \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m param_decomp.scripts.two_pool_benchmark.two_pool_maxA
'
