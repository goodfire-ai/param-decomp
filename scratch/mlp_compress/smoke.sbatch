#!/bin/bash
#SBATCH --job-name=mlp0-compress-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --comment="smoke test: compress block-0 MLP of s-55ea3f9b (KL distill on CI-masked forwards), oli/opus"
#SBATCH --output=/mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress/scratch/mlp_compress/smoke_%j.log

cd /mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress
source .venv/bin/activate
python scratch/mlp_compress/run.py --steps 5 --batch_size 4 --eval_every 2 --n_eval_batches 1 --warmup_steps 2 --use_wandb False
