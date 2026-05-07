#!/bin/bash
#SBATCH --job-name=nano-ss-2L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/polished-lake/home/braun/slurm_logs/slurm-%j.out

set -euo pipefail
umask 002

export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd /mnt/polished-lake/home/braun/param-decomp
source .venv/bin/activate

torchrun --standalone --nproc_per_node=8 -m nano_param_decomp.simplestories_4L
