#!/usr/bin/env bash
#SBATCH --job-name=vpd811-Abatch
#SBATCH --comment="Task 811 confidence-gated block-GradMax case-A acceptance"
#SBATCH --array=0-2%3
#SBATCH --time=02:00:00
#SBATCH --qos=scavenge
#SBATCH --output=sweeps/tms-controller-811-A-batch/logs/%A_%a.log

set -euo pipefail
cd /mnt/home/pd-user/.bridge/crew-kernel/minds/agent-c5xs-e96df6e3/controller-batch
mapfile -t configs < sweeps/tms-controller-811-A-batch/configs.txt
config=${configs[$SLURM_ARRAY_TASK_ID]}
uv run --no-sync python -m param_decomp.experiments.tms.run "$config" \
  --data_root /mnt/data/artifacts/mechanisms/param-decomp/runs \
  --group tms-controller-811-A-batch \
  --tags task-811,controller,capacity-lifecycle,block-gradmax
