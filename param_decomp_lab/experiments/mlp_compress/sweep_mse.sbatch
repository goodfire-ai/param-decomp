#!/bin/bash
#SBATCH --job-name=relmse-hnsweep
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --comment="sweep n_heads{1,2} x n_compressed{16,64,256,1024} replacing all 4 blocks' attn+mlp of s-55ea3f9b, trained on SUM of 8 relative residual-stream MSEs (post-attn + post-mlp per block) on CI-masked forward; KL shown as eval; oli/opus"
#SBATCH --output=sweep_mse_%A_%a.log
#SBATCH --array=0-7%7

H_VALUES=(1 1 1 1 2 2 2 2)
N_VALUES=(16 64 256 1024 16 64 256 1024)
H=${H_VALUES[$SLURM_ARRAY_TASK_ID]}
N=${N_VALUES[$SLURM_ARRAY_TASK_ID]}

# Resolve the repo root (the dir holding .venv) by walking up from the submit dir, so this
# script carries no hardcoded checkout path and runs wherever the repo is checked out.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/.venv" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[ -d "$REPO_ROOT/.venv" ] || { echo "no .venv found above ${SLURM_SUBMIT_DIR:-$PWD}" >&2; exit 1; }
cd "$REPO_ROOT"
source .venv/bin/activate
python -m param_decomp_lab.experiments.mlp_compress.run_mse --n_heads $H --n_compressed $N --steps 20000 --batch_size 32 --lr 1e-3
