#!/bin/bash
#SBATCH --job-name=2pool-18way
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/2pool-18way-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/2pool-18way-%j.err
#SBATCH --comment="2-pool aggressive parallelism: 18A (1 block each, no in-block DDP) + 6B DP-6 on 3 nodes; same 1B/10B model as stress test"

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29504

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun --nodes=3 --ntasks=3 --ntasks-per-node=1 bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \
    --nnodes=3 \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=8 \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m nano_param_decomp.two_pool_profile_stress_18way
'
