#!/usr/bin/env bash
#SBATCH --job-name=vpd811-acquire
#SBATCH --comment="Task 811 split-favoring gamma1/frequency-off structure acquisition"
#SBATCH --array=0-5%6
#SBATCH --time=01:00:00
#SBATCH --qos=scavenge
#SBATCH --output=sweeps/tms-structure-acquisition-811/logs/%A_%a.log
set -euo pipefail
cd /mnt/home/pd-user/.bridge/crew-kernel/minds/agent-c5xs-e96df6e3/logical-random-prefix
mapfile -t configs < sweeps/tms-structure-acquisition-811/configs.txt
config=${configs[$SLURM_ARRAY_TASK_ID]}
uv run --no-sync python -m param_decomp.experiments.tms.run "$config" \
  --data_root /mnt/data/artifacts/mechanisms/param-decomp/runs \
  --group tms-structure-acquisition-811 \
  --tags task-811,structure-acquisition,gamma1,frequency-off,random-prefix
