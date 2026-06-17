#!/bin/bash
#SBATCH --job-name=comp-mlp-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --comment="smoke test: component MLP + 4 new KL-to-target evals (ci/unmasked/rounded/adversarial-PGD); d_expand=2, 6 steps, no wandb; oli/opus"
#SBATCH --output=/mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress/scratch/mlp0_compress/smoke_component_evals_%j.log

cd /mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress
source .venv/bin/activate
cd scratch/mlp0_compress
python run_component.py --d_expand 2 --steps 6 --eval_every 2 --n_eval_batches 2 --save_every 1000000 --use_wandb False
