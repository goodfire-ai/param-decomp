#!/bin/bash
#SBATCH --job-name=pd-arith-ci-p530237f8
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/home/pd-user/.bridge/crew-kernel/minds/task-p-530237f8-arith-ci-abplus-2943a5ff/param-decomp/jobs/arith_ci_%j.out
#SBATCH --comment="CI of p-530237f8 (l18-mlp) subcomponents at '=' pos of 10000 'a+b=' prompts. Lucius req via Slack (task p-530237f8-arith-ci-abplus)."

set -euo pipefail
cd /mnt/home/pd-user/.bridge/crew-kernel/minds/task-p-530237f8-arith-ci-abplus-2943a5ff/param-decomp
unset VIRTUAL_ENV
export PARAM_DECOMP_OUT_DIR=/mnt/home/pd-user/param-decomp-runs

uv run --no-sync python jobs/arith_ci.py
