#!/bin/bash
#SBATCH --job-name=qwen-2pool
#SBATCH --nodes=15
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/qwen-2pool-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/qwen-2pool-%j.err
#SBATCH --comment="2-pool on Qwen3-0.6B-Base, 112A+8B = 120 GPUs across 15 nodes. 56 block groups (28 attn + 28 mlp, mid-point between layer-grouped and site-grouped) with intra-block DDP-2. batch=48 seq=1024."

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29512
# Keep HF download cache somewhere shared so all nodes see it.
export HF_HOME=${HF_HOME:-/mnt/polished-lake/home/oli/.cache/huggingface}

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT  HF_HOME=$HF_HOME"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun --nodes=15 --ntasks=15 --ntasks-per-node=1 bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \
    --nnodes=15 \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=8 \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m param_decomp.scripts.two_pool_benchmark.qwen_2pool
'
