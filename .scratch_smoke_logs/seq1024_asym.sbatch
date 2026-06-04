#!/bin/bash
#SBATCH --job-name=seq1024-asym
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:40:00
#SBATCH --comment="llama8b seq1024 asym n_ci=2/n_ppgd=4 — discriminate refactor-regression vs seq2048-specific (oli)"
#SBATCH --output=/mnt/home/oli/param-decomp/.scratch_smoke_logs/seq1024_asym_%j.log
cd /mnt/home/oli/param-decomp
uv run torchrun --standalone --nproc_per_node=8 \
  -m param_decomp_lab.experiments.lm.three_pool_run \
  param_decomp_lab/experiments/lm/_llama8b/seq1024_asym_smoke.yaml
