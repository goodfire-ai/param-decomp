#!/usr/bin/env bash
#SBATCH --job-name=vpd811-ctrl
#SBATCH --comment="Task 811 TMS recon-budget controller A/B/C acceptance"
#SBATCH --array=0-14%15
#SBATCH --time=04:00:00
#SBATCH --qos=scavenge
#SBATCH --output=sweeps/tms-controller-811/logs/%A_%a.log

set -euo pipefail
cd /mnt/home/pd-user/.bridge/crew-kernel/minds/agent-c5xs-e96df6e3/controller-review
mapfile -t configs < sweeps/tms-controller-811/configs.txt
config=${configs[$SLURM_ARRAY_TASK_ID]}
uv run --no-sync python -m param_decomp.experiments.tms.run "$config" \
  --data_root /mnt/data/artifacts/mechanisms/param-decomp/runs \
  --group tms-controller-811-abc \
  --tags task-811,controller,capacity-lifecycle
