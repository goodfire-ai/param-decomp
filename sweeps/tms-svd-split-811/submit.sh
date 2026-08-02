#!/usr/bin/env bash
#SBATCH --job-name=vpd811-svdsplit
#SBATCH --comment="Task 811 function-preserving SVD split basin-rescue falsifier"
#SBATCH --array=0-5%6
#SBATCH --time=04:00:00
#SBATCH --qos=scavenge
#SBATCH --output=sweeps/tms-svd-split-811/logs/%A_%a.log

set -euo pipefail
cd /mnt/home/pd-user/.bridge/crew-kernel/minds/agent-c5xs-e96df6e3/adaptive-split
mapfile -t configs < sweeps/tms-svd-split-811/configs.txt
config=${configs[$SLURM_ARRAY_TASK_ID]}
uv run --no-sync python -m param_decomp.experiments.tms.run "$config" \
  --data_root /mnt/data/artifacts/mechanisms/param-decomp/runs \
  --group tms-svd-split-811 \
  --tags task-811,svd-split,adaptive-discovery
