#!/bin/bash
#SBATCH --job-name=2pool-mn-profile
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/2pool-mn-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/2pool-mn-%j.err
#SBATCH --comment="2-pool architecture multi-node wall-clock profile: 12A + 4B across 2 nodes, every in-block DDP and pool B DP all-reduce crosses the IB"

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

# Master node = first node in allocation
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29501

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

# One srun task per node; each task starts torchrun --nproc_per_node=8
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \
    --nnodes=2 \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=8 \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m nano_param_decomp.two_pool_profile_multinode
'
