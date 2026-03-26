#!/bin/bash
#SBATCH --job-name=inspect_3698
#SBATCH --gpus-per-node=1
#SBATCH --time=0:30:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/slurm-%j.out
#SBATCH --comment="Inspect top-20 at iter 3698"

source /mnt/polished-lake/home/oli/spd/.venv/bin/activate
python3 -u /mnt/polished-lake/home/oli/spd/scripts/inspect_merge_3698.py
