#!/usr/bin/env bash
#SBATCH --job-name=vpd811-Ainv
#SBATCH --comment="Task 811 basis-invariant block-GradMax case-A acceptance"
#SBATCH --array=0-2%3
#SBATCH --time=02:00:00
#SBATCH --qos=scavenge
#SBATCH --output=sweeps/tms-controller-811-A-block-invariant/logs/%A_%a.log

set -euo pipefail
cd /mnt/home/pd-user/.bridge/crew-kernel/minds/agent-c5xs-e96df6e3/controller-batch
mapfile -t configs < sweeps/tms-controller-811-A-block-invariant/configs.txt
config=${configs[$SLURM_ARRAY_TASK_ID]}
uv run --no-sync python -m param_decomp.experiments.tms.run "$config" \
  --data_root /mnt/data/artifacts/mechanisms/param-decomp/runs \
  --group tms-controller-811-A-block-invariant \
  --tags task-811,controller,capacity-lifecycle,block-gradmax,basis-invariant-referee
