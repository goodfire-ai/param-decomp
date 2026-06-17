#!/bin/bash
#SBATCH --job-name=comp-mlp-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --comment="smoke test: component MLP + 4 new KL-to-target evals (ci/unmasked/rounded/adversarial-PGD); d_expand=2, 6 steps, no wandb; oli/opus"
#SBATCH --output=/mnt/home/lucius/spd/scratch/mlp_compress/smoke_component_evals_%j.log

cd /mnt/home/lucius/spd
source .venv/bin/activate
cd scratch/mlp_compress
python run_component.py --d_expand 2 --steps 6 --eval_every 2 --n_eval_batches 2 --save_every 1000000 --use_wandb False
