#!/bin/bash
#SBATCH --job-name=pd-arith-ci-p594db290
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/home/oli/claude-slack/data/workspaces/69f82c920cd2/param-decomp/jobs/arith_ci_%j.out
#SBATCH --comment="CI of p-594db290 (l18-mlp) subcomponents at '=' pos of 10000 'a+b=' prompts. Lucius req via Slack (arithmetic investigation)."

set -euo pipefail
cd /mnt/home/oli/claude-slack/data/workspaces/69f82c920cd2/param-decomp
unset VIRTUAL_ENV

uv run --no-sync python jobs/arith_ci.py
