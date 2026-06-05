#!/bin/bash
#SBATCH --job-name=llama8b-smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --time=00:45:00
#SBATCH --comment="llama8b L18 3-pool end-to-end smoke against live tree (oli)"
#SBATCH --output=/mnt/home/oli/param-decomp/.scratch_smoke_logs/smoke3gpu_%j.log
cd /mnt/home/oli/param-decomp
export PD_DISABLE_LW_COMPILE=1 PD_DISABLE_CI_COMPILE=1
uv run torchrun --standalone --nproc_per_node=3 \
  -m param_decomp_lab.experiments.lm.three_pool_run \
  param_decomp_lab/experiments/lm/_llama8b/smoke_l18.yaml
