#!/bin/bash
#SBATCH --job-name=seq2048-mem
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --time=00:40:00
#SBATCH --comment="llama8b L18 seq2048 bl8 per-rank memory smoke (oli)"
#SBATCH --output=/mnt/home/oli/param-decomp/.scratch_smoke_logs/seq2048_mem_%j.log
cd /mnt/home/oli/param-decomp
uv run torchrun --standalone --nproc_per_node=3 \
  -m param_decomp_lab.experiments.lm.three_pool_run \
  param_decomp_lab/experiments/lm/_llama8b/seq2048_memsmoke.yaml
