#!/bin/bash
#SBATCH --job-name=pd-lm-profile
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/home/dan.braun/param-decomp/profiling_out/slurm-%j.out

set -euo pipefail
cd /mnt/home/dan.braun/param-decomp
source .venv/bin/activate
# Pull in HF / WandB creds etc.
set -a
[ -f .env ] && source .env
set +a

export PD_PROFILE_TRACE_PATH=/mnt/home/dan.braun/param-decomp/profiling_out/llama_gate_proj_perrank8_trace.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single process (no --dp) -> no DDP, batch_size=8 == one rank of the --dp 8 run.
python -m param_decomp_lab.experiments.lm.run \
  param_decomp_lab/experiments/lm/llama-3.1-8b-profile.yaml
