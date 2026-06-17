#!/bin/bash
#SBATCH --job-name=relmse-bp-nc
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --comment="bypass-only-ish sweep: NO sparsity loss (coeff=0), n_compressed in {0,1,2,4}; learned linear bypass (C_cfc->C_down, zeros-init, no penalty, MLP-only) always on; n_compressed=0 => fit MLPs with bypass alone; fixed n_heads=1 head_dim=128 replacing all 4 blocks of s-55ea3f9b; recon = SUM of 8 relative residual-stream MSEs (CI-masked forward); ReLU; 100k steps; oli/opus"
#SBATCH --output=/mnt/home/lucius/spd/scratch/mlp_compress/sweep_mse_bypass_ncompressed_%A_%a.log
#SBATCH --array=0-3%4

N_COMPRESSED=(0 1 2 4)
NC=${N_COMPRESSED[$SLURM_ARRAY_TASK_ID]}

cd /mnt/home/lucius/spd
source .venv/bin/activate
cd scratch/mlp_compress
python run_mse.py --n_heads 1 --head_dim 128 --n_compressed $NC --sparsity_coeff 0 --bypass True --p_norm 0.9 --steps 100000 --batch_size 32 --lr 1e-3
