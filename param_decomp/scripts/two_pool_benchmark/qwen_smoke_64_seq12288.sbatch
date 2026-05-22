#!/bin/bash
#SBATCH --job-name=2pool-qwen-64-s12288
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/2pool-qwen-64-s12288-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/2pool-qwen-64-s12288-%j.err
#SBATCH --comment="2-pool Qwen3-0.6B at seq=12288 b=8 on 64 H200. Scale-up smoke."

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29543
export PROFILE_MODE=off
export PROFILE_OUT_DIR=/tmp/two_pool_profile
export HF_HOME=/mnt/polished-lake/home/oli/.cache/huggingface

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun --nodes=8 --ntasks=8 --ntasks-per-node=1 \
  --export=ALL,HF_HOME=$HF_HOME,PROFILE_MODE=$PROFILE_MODE,PROFILE_OUT_DIR=$PROFILE_OUT_DIR \
  bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \
    --nnodes=8 \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=8 \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m param_decomp.scripts.two_pool_benchmark.lm_2pool_launcher \
    --run_config param_decomp/experiments/lm/qwen3_0p6b_smoke_seq12288.yaml \
    --topology   param_decomp/experiments/lm/qwen3_0p6b_two_pool_topology_64.yaml \
    --wandb_project param-decomp
'
