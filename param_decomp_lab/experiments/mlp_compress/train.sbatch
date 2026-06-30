#!/bin/bash
#SBATCH --job-name=mlp0-compress
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --comment="compress block-0 MLP of s-55ea3f9b to 1/10 neuron dim (KL distill on CI-masked forwards), oli/opus"
#SBATCH --output=train_%j.log

# Resolve the repo root (the dir holding .venv) by walking up from the submit dir, so this
# script carries no hardcoded checkout path and runs wherever the repo is checked out.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/.venv" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[ -d "$REPO_ROOT/.venv" ] || { echo "no .venv found above ${SLURM_SUBMIT_DIR:-$PWD}" >&2; exit 1; }
cd "$REPO_ROOT"
source .venv/bin/activate
python -m param_decomp_lab.experiments.mlp_compress.run --steps 20000 --batch_size 32 --lr 1e-3
