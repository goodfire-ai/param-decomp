#!/usr/bin/env bash
#SBATCH --job-name=vpd811-formoff
#SBATCH --comment="Task 811 test complexity-OFF vs unit during TMS basis formation"
#SBATCH --array=0-5%6
#SBATCH --time=02:00:00
#SBATCH --qos=scavenge
#SBATCH --output=sweeps/tms-formative-off-811/logs/%A_%a.log
set -euo pipefail
cd /mnt/home/pd-user/.bridge/crew-kernel/minds/agent-c5xs-e96df6e3/logical-random-prefix
mapfile -t configs < sweeps/tms-formative-off-811/configs.txt
config=${configs[$SLURM_ARRAY_TASK_ID]}
uv run --no-sync python -m param_decomp.experiments.tms.run "$config" \
  --data_root /mnt/data/artifacts/mechanisms/param-decomp/runs \
  --group tms-formative-off-811 \
  --tags task-811,formative-force,complexity-off,random-prefix
