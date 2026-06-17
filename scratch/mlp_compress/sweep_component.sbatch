#!/bin/bash
#SBATCH --job-name=comp-mlp-dexp
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --comment="Disentangled component MLP on STOCHASTIC-masked forward (masks ~ U[ci.lower_leaky,1], shared teacher/student). Replaces all 4 blocks' MLPs of s-55ea3f9b with: dense linear bypass (C_in->C_out) + N=C_out*d_expand GeLU neurons each wired to ONE output subcomponent (w_neuron) reading all input subcomponents (W_in). Attention untouched (runs masked normally). Loss = SUM of 8 residual-stream relative MSEs (student vs stoch teacher); NO sparsity. Sweep d_expand in {0,1,2,4}; d_expand=0 = bypass only. 20k steps; oli/opus"
#SBATCH --output=/mnt/home/lucius/spd/scratch/mlp_compress/sweep_component_%A_%a.log
#SBATCH --array=0-3%4

D_EXPAND=(0 1 2 4)
DE=${D_EXPAND[$SLURM_ARRAY_TASK_ID]}

cd /mnt/home/lucius/spd
source .venv/bin/activate
cd scratch/mlp_compress
python run_component.py --d_expand $DE --steps 20000 --batch_size 32 --lr 1e-3
