#!/bin/bash
#SBATCH --job-name=attnmlp-hsweep
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --comment="sweep n_heads replacing ALL 4 blocks' attention (fewer heads, head_dim 128) + all MLPs (16-neuron bottleneck) of s-55ea3f9b jointly (KL distill on CI-masked forwards), oli/opus"
#SBATCH --output=/mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress/scratch/mlp_compress/sweep_attn_%A_%a.log
#SBATCH --array=0-5%6

H_VALUES=(1 2 3 4 5 6)
H=${H_VALUES[$SLURM_ARRAY_TASK_ID]}

cd /mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress
source .venv/bin/activate
python scratch/mlp_compress/run_attn.py --n_heads $H --n_compressed 16 --steps 20000 --batch_size 32 --lr 1e-3
