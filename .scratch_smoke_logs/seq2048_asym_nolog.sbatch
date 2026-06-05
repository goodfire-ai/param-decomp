#!/bin/bash
#SBATCH --job-name=asym-nolog
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:20:00
#SBATCH --qos=opportunistic
#SBATCH --comment="llama8b seq2048 asym, logging OFF — isolate core training comms from logging-path deadlock (oli)"
#SBATCH --output=/mnt/home/oli/param-decomp/.scratch_smoke_logs/seq2048_asym_nolog_%j.log
cd /mnt/home/oli/param-decomp
export PYTHONUNBUFFERED=1 PD_3POOL_PG_TIMEOUT_S=120 PD_DISABLE_CHUNK_COMPILE=1 PD_DISABLE_CI_COMPILE=1
export PARAM_DECOMP_OUT_DIR=/mnt/data/artifacts/mechanisms/param-decomp
uv run torchrun --standalone --nproc_per_node=8 \
  -m param_decomp_lab.experiments.lm.three_pool_run \
  param_decomp_lab/experiments/lm/_llama8b/seq2048_asym_nolog.yaml
