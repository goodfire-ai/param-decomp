#!/bin/bash
#SBATCH --job-name=pd-lm-profile
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/home/dan.braun/param-decomp/profiling_out/slurm-%j.out

set -euo pipefail
cd /mnt/home/dan.braun/param-decomp
source .venv/bin/activate

export HF_HUB_CACHE=/mnt/data/artifacts/hf_cache/hub
export PARAM_DECOMP_OUT_DIR=/mnt/home/dan.braun/param-decomp/profiling_out/run_artifacts
export PD_PROFILE_TRACE_PATH=/mnt/home/dan.braun/param-decomp/profiling_out/llama_gate_proj_bs64_trace.json

# Single process (no --dp) -> batch_size 64 stays intact, no DDP noise.
python -m param_decomp_lab.experiments.lm.run \
  param_decomp_lab/experiments/lm/llama-3.1-8b-profile.yaml
