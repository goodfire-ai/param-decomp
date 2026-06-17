#!/bin/bash
#SBATCH --job-name=relmse-spsweep
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --comment="sweep Lp(p=0.9) sparsity coeff {1e-4,3e-4,1e-3,3e-3,1e-2,3e-2} on ReLU replacement-MLP neuron acts; fixed n_heads=1 head_dim=128 n_compressed=768 replacing all 4 blocks of s-55ea3f9b; recon = SUM of 8 relative residual-stream MSEs (CI-masked forward); L0 of neuron acts as eval; 100k steps; oli/opus"
#SBATCH --output=/mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress/scratch/mlp_compress/sweep_mse_sparsity_%A_%a.log
#SBATCH --array=0-5%6

COEFFS=(1e-4 3e-4 1e-3 3e-3 1e-2 3e-2)
COEFF=${COEFFS[$SLURM_ARRAY_TASK_ID]}

cd /mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress
source .venv/bin/activate
cd scratch/mlp_compress
python run_mse.py --n_heads 1 --head_dim 128 --n_compressed 768 --sparsity_coeff $COEFF --p_norm 0.9 --steps 100000 --batch_size 32 --lr 1e-3
