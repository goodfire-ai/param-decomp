#!/bin/bash
#SBATCH --job-name=seq2048-bl16
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --comment="llama8b seq2048 asymmetric chunk/ci bl16 + ppgd bl8 memory confirm (oli)"
#SBATCH --output=/mnt/home/oli/param-decomp/.scratch_smoke_logs/seq2048_bl16_%j.log
cd /mnt/home/oli/param-decomp
uv run torchrun --standalone --nproc_per_node=4 \
  -m param_decomp_lab.experiments.lm.three_pool_run \
  param_decomp_lab/experiments/lm/_llama8b/seq2048_bl16_memsmoke.yaml
