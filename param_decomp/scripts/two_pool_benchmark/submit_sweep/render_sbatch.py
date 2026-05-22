"""Render the per-point sbatch script.

Hardcoded template for the B200 east-13a cluster: nproc_per_node=8, torchrun
launching :mod:`param_decomp.scripts.two_pool_benchmark.lm_2pool_launcher`
with the rendered ``run.yaml`` + ``topology.yaml``.
"""

from pathlib import Path

from param_decomp.scripts.two_pool_benchmark.submit_sweep.paths import (
    HF_HOME,
    REPO,
    SLURM_LOGS,
)
from param_decomp.scripts.two_pool_benchmark.submit_sweep.schema import RuntimeSpec


def render_sbatch(
    *,
    name: str,
    n_nodes: int,
    job_dir: Path,
    runtime: RuntimeSpec,
    master_port: int,
) -> str:
    qos_line = f"#SBATCH --qos={runtime.qos}\n" if runtime.qos else ""
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --nodes={n_nodes}
#SBATCH --gpus-per-node=8
#SBATCH --time={runtime.time}
{qos_line}#SBATCH --output={SLURM_LOGS}/{name}-%j.out
#SBATCH --error={SLURM_LOGS}/{name}-%j.err
#SBATCH --comment="2-pool sweep point: {name}"

set -euo pipefail

REPO={REPO}
cd "$REPO"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT={master_port}
export PROFILE_MODE=off
export HF_HOME={HF_HOME}
export HF_HUB_OFFLINE=${{HF_HUB_OFFLINE:-0}}

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun --nodes={n_nodes} --ntasks={n_nodes} --ntasks-per-node=1 \\
  --export=ALL,HF_HOME=$HF_HOME,PROFILE_MODE=$PROFILE_MODE,HF_HUB_OFFLINE=$HF_HUB_OFFLINE \\
  bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \\
    --nnodes={n_nodes} \\
    --node_rank=$SLURM_PROCID \\
    --nproc_per_node=8 \\
    --master_addr='"$MASTER_ADDR"' \\
    --master_port='"$MASTER_PORT"' \\
    -m param_decomp.scripts.two_pool_benchmark.lm_2pool_launcher \\
    --run_config {job_dir}/run.yaml \\
    --topology   {job_dir}/topology.yaml \\
    --wandb_project {runtime.wandb_project}
'
"""
