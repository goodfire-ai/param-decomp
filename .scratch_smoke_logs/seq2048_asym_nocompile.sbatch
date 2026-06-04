#!/bin/bash
#SBATCH --job-name=seq2048-asym-nc
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:40:00
#SBATCH --comment="llama8b seq2048 asym n_ci=2/n_ppgd=4 COMPILE-OFF — test compile-desync hypothesis (oli)"
#SBATCH --output=/mnt/home/oli/param-decomp/.scratch_smoke_logs/seq2048_asym_nc_%j.log
cd /mnt/home/oli/param-decomp
export PD_DISABLE_LW_COMPILE=1 PD_DISABLE_CI_COMPILE=1
uv run torchrun --standalone --nproc_per_node=8 \
  -m param_decomp_lab.experiments.lm.three_pool_run \
  param_decomp_lab/experiments/lm/_llama8b/seq2048_asym_smoke.yaml
