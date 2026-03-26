#!/bin/bash
#SBATCH --job-name=inspect_3698_10m
#SBATCH --gpus-per-node=1
#SBATCH --mem=500G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/slurm-%j.out
#SBATCH --comment="Inspect top-20 at iter 3698 using 10M snapshot"

source /mnt/polished-lake/home/oli/spd/.venv/bin/activate
python3 -u /mnt/polished-lake/home/oli/spd/scripts/inspect_merge_3698_10m.py
