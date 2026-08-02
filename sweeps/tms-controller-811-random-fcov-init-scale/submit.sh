#!/usr/bin/env bash
#SBATCH --job-name=vpd811-rfscale
#SBATCH --comment="Task 811 random-fcov controller initial-scale robustness endpoints"
#SBATCH --array=0-11%12
#SBATCH --time=04:00:00
#SBATCH --qos=scavenge
#SBATCH --output=sweeps/tms-controller-811-random-fcov-init-scale/logs/%A_%a.log

set -euo pipefail
cd /mnt/home/pd-user/.bridge/crew-kernel/minds/agent-c5xs-e96df6e3/controller-batch
mapfile -t configs < sweeps/tms-controller-811-random-fcov-init-scale/configs.txt
config=${configs[$SLURM_ARRAY_TASK_ID]}
uv run --no-sync python -m param_decomp.experiments.tms.run "$config" \
  --data_root /mnt/data/artifacts/mechanisms/param-decomp/runs \
  --group tms-controller-811-random-fcov-init-scale \
  --tags task-811,controller,random-overcomplete,function-covariant,initial-scale-robustness
