#!/bin/bash
#SBATCH --job-name=relmse-hnsweep
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --comment="sweep n_heads{1,2} x n_compressed{16,64,256,1024} replacing all 4 blocks' attn+mlp of s-55ea3f9b, trained on SUM of 8 relative residual-stream MSEs (post-attn + post-mlp per block) on CI-masked forward; KL shown as eval; oli/opus"
#SBATCH --output=/mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress/scratch/mlp_compress/sweep_mse_%A_%a.log
#SBATCH --array=0-7%7

H_VALUES=(1 1 1 1 2 2 2 2)
N_VALUES=(16 64 256 1024 16 64 256 1024)
H=${H_VALUES[$SLURM_ARRAY_TASK_ID]}
N=${N_VALUES[$SLURM_ARRAY_TASK_ID]}

cd /mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress
source .venv/bin/activate
python scratch/mlp_compress/run_mse.py --n_heads $H --n_compressed $N --steps 20000 --batch_size 32 --lr 1e-3
