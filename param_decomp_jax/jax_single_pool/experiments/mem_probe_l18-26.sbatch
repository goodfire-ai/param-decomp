#!/bin/bash
#SBATCH --job-name=memprobe-l18-26
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --qos=opportunistic
#SBATCH --comment="AOT HBM probe: Llama-8B MLP l18-26 (27 sites, C=49152, chunkwise 9/chunk, remat) at 64 B200, B=128 and B=64; no training"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR/../.."   # experiments -> jax_single_pool -> param_decomp_jax
source .venv-cuda/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

SRUN="--kill-on-bad-exit=1 --ntasks-per-node=8 --cpus-per-task=8 --distribution=block:block"

echo "===== PROBE B=128 (per_gpu_batch=2) ====="
srun $SRUN python -m jax_single_pool.experiments.mem_probe \
  --first_layer 18 --last_layer 26 --C 49152 \
  --sites_per_chunk 9 --recon_coeff 1.5 --per_gpu_batch 2

echo "===== PROBE B=64 (per_gpu_batch=1) ====="
srun $SRUN python -m jax_single_pool.experiments.mem_probe \
  --first_layer 18 --last_layer 26 --C 49152 \
  --sites_per_chunk 9 --recon_coeff 1.5 --per_gpu_batch 1

echo "===== ALL PROBES DONE ====="
