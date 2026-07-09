#!/bin/bash
#SBATCH --job-name=comp-mlp-adv
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --comment="Disentangled component MLP, 2-term loss: 0.5*relmse(stochastic masks) + 0.5*relmse(PERSISTENT ADVERSARIAL masks). Adversary = PPGD-style per-component sources (per_batch_per_position, like the real run's training PPGD), Adam ascent beta1=0/beta2=0.99/lr=0.01 constant, n_warmup=1 (=> 2 source updates/step), maximizing the replacement's residual relmse. Replaces all 4 blocks' MLPs of s-55ea3f9b. Base lr=1e-4 cosine, 100k steps. Sweep d_expand in {0,1,2,4}; oli/opus"
#SBATCH --output=sweep_component_adv_%A_%a.log
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
python -m param_decomp_lab.experiments.mlp_compress.run_component --d_expand $DE --steps 100000 --batch_size 32 --lr 1e-4 \
  --stochastic_coeff 0.5 --adversarial_coeff 0.5 \
  --adv_lr 0.01 --adv_beta1 0.0 --adv_beta2 0.99 --adv_n_warmup 1 \
  --adv_scope per_batch_per_position \
  --eval_every 1000 --save_every 10000
