#!/bin/bash
#SBATCH --job-name=mlp0-nsweep
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --comment="sweep n_compressed for block-0 MLP compression of s-55ea3f9b (KL distill on CI-masked forwards), find smallest n with KL<0.01, oli/opus"
#SBATCH --output=/mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress/scratch/mlp0_compress/sweep_%A_%a.log
#SBATCH --array=0-6%7

N_VALUES=(4 8 16 32 48 64 96)
N=${N_VALUES[$SLURM_ARRAY_TASK_ID]}

cd /mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress
source .venv/bin/activate
python scratch/mlp0_compress/run.py --n_compressed $N --steps 20000 --batch_size 32 --lr 1e-3
