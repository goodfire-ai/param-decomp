#!/bin/bash
#SBATCH --job-name=attnmlp-dsweep
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --comment="sweep attn head_dim (1 head/block, RoPE rebuilt per head_dim) + 8-neuron MLP bottleneck, replacing ALL 4 blocks of s-55ea3f9b jointly (KL distill on CI-masked forwards), 100k steps, oli/opus"
#SBATCH --output=/mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress/scratch/mlp_compress/sweep_attn_headdim_%A_%a.log
#SBATCH --array=0-4%5

D_VALUES=(8 16 32 64 128)
D=${D_VALUES[$SLURM_ARRAY_TASK_ID]}

cd /mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress
source .venv/bin/activate
python scratch/mlp_compress/run_attn.py --n_heads 1 --head_dim $D --n_compressed 8 --steps 100000 --batch_size 32 --lr 1e-3
