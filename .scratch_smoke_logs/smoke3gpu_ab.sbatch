#!/bin/bash
#SBATCH --job-name=llama8b-rs-ab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --time=00:40:00
#SBATCH --comment="llama8b L18 residual-start A/B smoke (oli)"
#SBATCH --output=/mnt/home/oli/param-decomp/.scratch_smoke_logs/rs_%x_%j.log
cd /mnt/home/oli/param-decomp
export PD_DISABLE_LW_COMPILE=1 PD_DISABLE_CI_COMPILE=1
echo "PD_DISABLE_RESIDUAL_START=${PD_DISABLE_RESIDUAL_START:-unset}"
uv run torchrun --standalone --nproc_per_node=3 \
  -m param_decomp_lab.experiments.lm.three_pool_run \
  param_decomp_lab/experiments/lm/_llama8b/smoke_l18.yaml
