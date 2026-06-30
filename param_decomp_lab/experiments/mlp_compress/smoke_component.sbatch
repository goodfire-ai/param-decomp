#!/bin/bash
#SBATCH --job-name=comp-mlp-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --comment="smoke test: component MLP + 4 new KL-to-target evals (ci/unmasked/rounded/adversarial-PGD); d_expand=2, 6 steps, no wandb; oli/opus"
#SBATCH --output=smoke_component_evals_%j.log

# Resolve the repo root (the dir holding .venv) by walking up from the submit dir, so this
# script carries no hardcoded checkout path and runs wherever the repo is checked out.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/.venv" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[ -d "$REPO_ROOT/.venv" ] || { echo "no .venv found above ${SLURM_SUBMIT_DIR:-$PWD}" >&2; exit 1; }
cd "$REPO_ROOT"
source .venv/bin/activate
python -m param_decomp_lab.experiments.mlp_compress.run_component --d_expand 2 --steps 6 --eval_every 2 --n_eval_batches 2 --save_every 1000000 --use_wandb False
