#!/bin/bash
#SBATCH --job-name=comp-mlp-adv-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --comment="smoke test: component MLP + persistent-adversarial (PPGD) 2nd loss term; d_expand=2, 4 steps, 0.5/0.5 coeffs, no wandb; oli/opus"
#SBATCH --output=/mnt/home/lucius/spd/scratch/mlp_compress/smoke_component_adv_%j.log

cd /mnt/home/lucius/spd
source .venv/bin/activate
cd scratch/mlp_compress
python run_component.py --d_expand 2 --steps 4 \
  --stochastic_coeff 0.5 --adversarial_coeff 0.5 \
  --adv_lr 0.01 --adv_beta1 0.0 --adv_beta2 0.99 --adv_n_warmup 1 \
  --lr 1e-4 --eval_every 2 --n_eval_batches 2 --save_every 1000000 --use_wandb False
