#!/bin/bash
#SBATCH --job-name=comp-mlp-dexp
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --comment="Disentangled component MLP on STOCHASTIC-masked forward (masks ~ U[ci.lower_leaky,1], shared teacher/student). Replaces all 4 blocks' MLPs of s-55ea3f9b with: dense linear bypass (C_in->C_out) + N=C_out*d_expand GeLU neurons each wired to ONE output subcomponent (w_neuron) reading all input subcomponents (W_in). Attention untouched (runs masked normally). Loss = SUM of 8 residual-stream relative MSEs (student vs stoch teacher); NO sparsity. Sweep d_expand in {0,1,2,4}; d_expand=0 = bypass only. 20k steps; oli/opus"
#SBATCH --output=sweep_component_%A_%a.log
#SBATCH --array=0-3%4

D_EXPAND=(0 1 2 4)
DE=${D_EXPAND[$SLURM_ARRAY_TASK_ID]}

# Resolve the repo root (the dir holding .venv) by walking up from the submit dir, so this
# script carries no hardcoded checkout path and runs wherever the repo is checked out.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/.venv" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[ -d "$REPO_ROOT/.venv" ] || { echo "no .venv found above ${SLURM_SUBMIT_DIR:-$PWD}" >&2; exit 1; }
cd "$REPO_ROOT"
source .venv/bin/activate
python -m param_decomp_lab.experiments.mlp_compress.run_component --d_expand $DE --steps 20000 --batch_size 32 --lr 1e-3
