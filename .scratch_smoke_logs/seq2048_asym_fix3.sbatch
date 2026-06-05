#!/bin/bash
#SBATCH --job-name=asym-fix3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:25:00
#SBATCH --qos=opportunistic
#SBATCH --comment="llama8b seq2048 asym — validate batched logging p2p fix for step-4 NCCL desync (oli)"
#SBATCH --output=/mnt/home/oli/param-decomp/.scratch_smoke_logs/seq2048_asym_fix3_%j.log
cd /mnt/home/oli/param-decomp
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PD_3POOL_PG_TIMEOUT_S=120
export PD_DISABLE_CHUNK_COMPILE=1
export PD_DISABLE_CI_COMPILE=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export PARAM_DECOMP_OUT_DIR=/mnt/data/artifacts/mechanisms/param-decomp
torchrun --standalone --nproc_per_node=8 \
  -m param_decomp_lab.experiments.lm.three_pool_run \
  param_decomp_lab/experiments/lm/_llama8b/seq2048_asym_smoke.yaml
