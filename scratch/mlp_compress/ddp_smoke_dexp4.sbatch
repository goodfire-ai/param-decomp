#!/bin/bash
#SBATCH --job-name=comp-mlp-adv-dexp4-ddp-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:25:00
#SBATCH --comment="smoke: d_expand=4, 8-GPU DDP, 4 steps, per_batch_per_position PPGD, no wandb; oli/opus"
#SBATCH --output=/mnt/home/lucius/spd/scratch/mlp_compress/ddp_smoke_dexp4_%j.log

cd /mnt/home/lucius/spd
source .venv/bin/activate
cd scratch/mlp_compress
torchrun --standalone --nproc_per_node=8 run_component.py \
  --d_expand 4 --steps 4 --batch_size 64 --lr 5e-5 \
  --stochastic_coeff 0.5 --adversarial_coeff 0.5 \
  --adv_lr 0.01 --adv_beta1 0.0 --adv_beta2 0.99 --adv_n_warmup 2 \
  --adv_scope per_batch_per_position \
  --eval_every 2 --n_eval_batches 1 --save_every 100000 --use_wandb True
