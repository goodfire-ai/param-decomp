#!/bin/bash
#SBATCH --job-name=seq2048-asym-fix
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --comment="llama8b seq2048 asym n_ci=2/n_ppgd=4 — VALIDATE batch_isend_irecv fix (oli)"
#SBATCH --output=/mnt/home/oli/param-decomp/.scratch_smoke_logs/seq2048_asym_fix_%j.log
cd /mnt/home/oli/param-decomp
uv run torchrun --standalone --nproc_per_node=8 \
  -m param_decomp_lab.experiments.lm.three_pool_run \
  param_decomp_lab/experiments/lm/_llama8b/seq2048_asym_smoke.yaml
