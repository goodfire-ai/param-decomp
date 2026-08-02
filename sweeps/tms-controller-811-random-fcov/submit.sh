#!/usr/bin/env bash
#SBATCH --job-name=vpd811-rfcov
#SBATCH --comment="Task 811 random-overcomplete fcov controller C=100-800 preservation"
#SBATCH --array=0-11%12
#SBATCH --time=04:00:00
#SBATCH --qos=scavenge
#SBATCH --output=sweeps/tms-controller-811-random-fcov/logs/%A_%a.log

set -euo pipefail
cd /mnt/home/pd-user/.bridge/crew-kernel/minds/agent-c5xs-e96df6e3/controller-batch
mapfile -t configs < sweeps/tms-controller-811-random-fcov/configs.txt
config=${configs[$SLURM_ARRAY_TASK_ID]}
uv run --no-sync python -m param_decomp.experiments.tms.run "$config" \
  --data_root /mnt/data/artifacts/mechanisms/param-decomp/runs \
  --group tms-controller-811-random-fcov \
  --tags task-811,controller,random-overcomplete,function-covariant,c-preservation
