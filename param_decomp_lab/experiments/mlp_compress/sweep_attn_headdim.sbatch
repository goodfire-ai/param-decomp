#!/bin/bash
#SBATCH --job-name=attnmlp-dsweep
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --comment="sweep attn head_dim (1 head/block, RoPE rebuilt per head_dim) + 8-neuron MLP bottleneck, replacing ALL 4 blocks of s-55ea3f9b jointly (KL distill on CI-masked forwards), 100k steps, oli/opus"
#SBATCH --output=sweep_attn_headdim_%A_%a.log
#SBATCH --array=0-4%5

D_VALUES=(8 16 32 64 128)
D=${D_VALUES[$SLURM_ARRAY_TASK_ID]}

# Resolve the repo root (the dir holding .venv) by walking up from the submit dir, so this
# script carries no hardcoded checkout path and runs wherever the repo is checked out.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/.venv" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[ -d "$REPO_ROOT/.venv" ] || { echo "no .venv found above ${SLURM_SUBMIT_DIR:-$PWD}" >&2; exit 1; }
cd "$REPO_ROOT"
source .venv/bin/activate
python -m param_decomp_lab.experiments.mlp_compress.run_attn --n_heads 1 --head_dim $D --n_compressed 8 --steps 100000 --batch_size 32 --lr 1e-3
