#!/bin/bash
#SBATCH --job-name=allmlp-nsweep
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --comment="sweep n_compressed compressing ALL 4 block MLPs of s-55ea3f9b jointly (KL distill on CI-masked forwards), oli/opus"
#SBATCH --output=/mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress/scratch/mlp_compress/sweep_all_%A_%a.log
#SBATCH --array=0-6%7

N_VALUES=(4 8 16 32 48 64 96)
N=${N_VALUES[$SLURM_ARRAY_TASK_ID]}

cd /mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress
source .venv/bin/activate
python scratch/mlp_compress/run_all.py --n_compressed $N --steps 20000 --batch_size 32 --lr 1e-3
