#!/bin/bash
#SBATCH --job-name=allthr-rdsc400k
#SBATCH --partition=h100-reserved
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/delicate-frog/artifacts/mechanisms/param-decomp/slurm_logs/allthr-rdsc400k-%j.out
source /mnt/delicate-frog/scripts/login.sh
cd /mnt/delicate-frog/home/braun/param-decomp-rdsc
for rid in p-22b1a4e9 p-91a054a8 p-8c61383a p-97bf6bc4 p-0d6005c6 p-d7704c49; do
  echo "### run $rid ###"
  .venv/bin/python eval_all_thresh.py $rid
done
