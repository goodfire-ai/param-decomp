#!/bin/bash
#SBATCH --job-name=comp-mlp-dexp4-ddp-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:25:00
#SBATCH --comment="smoke: d_expand=4, 8-GPU DDP, 4 steps; 3-term output-KL loss 0.33*stochastic-recon-subset + 0.33*persistent-PPGD(per_batch_per_position) + 0.33*unmasked(all components on, no delta); oli/opus"
#SBATCH --output=ddp_smoke_dexp4_%j.log

# Resolve the repo root (the dir holding .venv) by walking up from the submit dir, so this
# script carries no hardcoded checkout path and runs wherever the repo is checked out.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/.venv" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[ -d "$REPO_ROOT/.venv" ] || { echo "no .venv found above ${SLURM_SUBMIT_DIR:-$PWD}" >&2; exit 1; }
cd "$REPO_ROOT"
source .venv/bin/activate
torchrun --standalone --nproc_per_node=8 \
  -m param_decomp_lab.experiments.mlp_compress.run_component \
  --d_expand 4 --steps 4 --batch_size 16 --lr 5e-5 \
  --stochastic_coeff 0.33 --adversarial_coeff 0.33 --unmasked_coeff 0.33 \
  --adv_lr 0.01 --adv_beta1 0.0 --adv_beta2 0.99 --adv_n_warmup 2 \
  --adv_scope per_batch_per_position \
  --eval_every 2 --n_eval_batches 1 --save_every 100000 --use_wandb True
