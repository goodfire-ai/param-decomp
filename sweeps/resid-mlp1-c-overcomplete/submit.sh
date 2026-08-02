#!/usr/bin/env bash
#SBATCH --job-name=vpd811-residc
#SBATCH --comment="Task 811 ResidMLP1 overcomplete C x complexity-force interaction grid"
#SBATCH --array=0-44%45
#SBATCH --time=12:00:00
#SBATCH --qos=scavenge
#SBATCH --output=sweeps/resid-mlp1-c-overcomplete/logs/%A_%a.log

set -euo pipefail
cd /mnt/home/pd-user/.bridge/crew-kernel/minds/agent-c5xs-e96df6e3/resid-c-sensitivity
mapfile -t configs < sweeps/resid-mlp1-c-overcomplete/configs.txt
config=${configs[$SLURM_ARRAY_TASK_ID]}
uv run --no-sync python -m param_decomp.experiments.resid_mlp.run "$config" \
  --data_root /mnt/data/artifacts/mechanisms/param-decomp/runs \
  --group resid-mlp1-c-overcomplete-811 \
  --tags task-811,resid-mlp1,c-overcomplete,complexity-interaction
