#!/bin/bash
#SBATCH --job-name=fsdp-tolocal-probe
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --qos=opportunistic
#SBATCH --time=0:10:00
#SBATCH --comment="fsdp delta.to_local grad-preservation probe"
#SBATCH --output=/mnt/home/oli/param-decomp/param_decomp_lab/fsdp/_scratch/tolocal-%j.out

source /mnt/home/oli/param-decomp/.venv/bin/activate
cd /mnt/home/oli/param-decomp/param_decomp_lab/fsdp/_scratch
torchrun --standalone --nproc_per_node=2 probe_tolocal.py
