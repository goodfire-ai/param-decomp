#!/bin/bash
#SBATCH --job-name=2pool-llama12L-coal32-b128
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/2pool-llama12L-coal32-b128-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/2pool-llama12L-coal32-b128-%j.err
#SBATCH --comment="2-pool LlamaSimpleMLP-12L coalesced + batch=128 on 32 GPUs. Tests batch-scaling at fixed topology."

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29521

export PROFILE_MODE=sync
export PROFILE_OUT_DIR=/mnt/polished-lake/home/oli/2pool_profiles/llama12L-coal32-b128-$SLURM_JOB_ID

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "PROFILE_OUT_DIR=$PROFILE_OUT_DIR"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun --nodes=8 --ntasks=8 --ntasks-per-node=1 \
  --export=ALL,PROFILE_MODE=$PROFILE_MODE,PROFILE_OUT_DIR=$PROFILE_OUT_DIR \
  bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \
    --nnodes=8 \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=4 \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m param_decomp.scripts.two_pool_benchmark.lm_2pool_launcher \
    --run_config param_decomp/experiments/lm/llama_simple_mlp_12L_smoke_b128.yaml \
    --topology   param_decomp/experiments/lm/llama_simple_mlp_12L_two_pool_topology_32.yaml \
    --wandb_project param-decomp
'
