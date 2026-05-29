#!/bin/bash
#SBATCH --job-name=pd-lm-prof-ddp
#SBATCH --gpus=8
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/home/dan.braun/param-decomp/profiling_out/slurm-ddp-%j.out

set -euo pipefail
cd /mnt/home/dan.braun/param-decomp
source .venv/bin/activate
set -a
[ -f .env ] && source .env
set +a

# Only rank 0 profiles (the trainer gates on is_main_process), so one trace.
export PD_PROFILE_TRACE_PATH=/mnt/home/dan.braun/param-decomp/profiling_out/llama_gate_proj_dp8_perrank8_trace.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# DDP, 8 ranks on one node: global batch_size=64 -> 8 per GPU (the real config).
torchrun --standalone --nproc_per_node=8 \
  -m param_decomp_lab.experiments.lm.run \
  param_decomp_lab/experiments/lm/llama-3.1-8b-profile.yaml
