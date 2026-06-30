#!/bin/bash
#SBATCH --job-name=relmse-bypass
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --comment="sweep Lp(p=0.9) sparsity coeff {1e-3,3e-3} on ReLU replacement-MLP neuron acts, WITH learned linear bypass (C_cfc->C_down, zeros-init, no sparsity penalty, MLP-only); fixed n_heads=1 head_dim=128 n_compressed=768 replacing all 4 blocks of s-55ea3f9b; recon = SUM of 8 relative residual-stream MSEs (CI-masked forward); L0 of neuron acts as eval; 100k steps; oli/opus"
#SBATCH --output=sweep_mse_bypass_%A_%a.log
#SBATCH --array=0-1%2

COEFFS=(1e-3 3e-3)
COEFF=${COEFFS[$SLURM_ARRAY_TASK_ID]}

# Resolve the repo root (the dir holding .venv) by walking up from the submit dir, so this
# script carries no hardcoded checkout path and runs wherever the repo is checked out.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/.venv" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[ -d "$REPO_ROOT/.venv" ] || { echo "no .venv found above ${SLURM_SUBMIT_DIR:-$PWD}" >&2; exit 1; }
cd "$REPO_ROOT"
source .venv/bin/activate
python -m param_decomp_lab.experiments.mlp_compress.run_mse --n_heads 1 --head_dim 128 --n_compressed 768 --sparsity_coeff $COEFF --bypass True --p_norm 0.9 --steps 100000 --batch_size 32 --lr 1e-3
