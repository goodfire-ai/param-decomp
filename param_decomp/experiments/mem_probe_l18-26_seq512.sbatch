#!/bin/bash
#SBATCH --job-name=memprobe-l18-26-seq512
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --qos=opportunistic
#SBATCH --comment="AOT HBM probe at seq=512: Llama-8B MLP l18-26 (27 sites, C=49152, chunkwise 9/chunk, remat) at 64 B200, B=64/128/256; plus 6-layer (l18-23) B=64 fallback; no training"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR/../.."   # experiments -> jax_single_pool -> param_decomp_jax
source .venv-cuda/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

SRUN="--kill-on-bad-exit=1 --ntasks-per-node=8 --cpus-per-task=8 --distribution=block:block"

for PGB in 1 2 4; do
  echo "===== PROBE 9-layer (l18-26) seq=512 per_gpu_batch=$PGB ====="
  srun $SRUN python -m jax_single_pool.experiments.mem_probe \
    --first_layer 18 --last_layer 26 --C 49152 \
    --sites_per_chunk 9 --recon_coeff 1.5 --per_gpu_batch "$PGB" --seq 512
done

echo "===== PROBE 6-layer (l18-23) seq=512 per_gpu_batch=1 (fallback) ====="
srun $SRUN python -m jax_single_pool.experiments.mem_probe \
  --first_layer 18 --last_layer 23 --C 49152 \
  --sites_per_chunk 6 --recon_coeff 1.5 --per_gpu_batch 1 --seq 512

echo "===== ALL PROBES DONE ====="
