#!/bin/bash
#SBATCH --job-name=allmlp-nsweep
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --comment="sweep n_compressed compressing ALL 4 block MLPs of s-55ea3f9b jointly (KL distill on CI-masked forwards), oli/opus"
#SBATCH --output=sweep_all_%A_%a.log
#SBATCH --array=0-6%7

N_VALUES=(4 8 16 32 48 64 96)
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
python -m param_decomp_lab.experiments.mlp_compress.run_all --n_compressed $N --steps 20000 --batch_size 32 --lr 1e-3
