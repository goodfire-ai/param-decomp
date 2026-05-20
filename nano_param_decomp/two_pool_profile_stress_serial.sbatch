#!/bin/bash
#SBATCH --job-name=2pool-stress-serial
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/2pool-stress-serial-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/2pool-stress-serial-%j.err
#SBATCH --comment="2-pool stress test SERIAL baseline: same model/topology as two_pool_profile_stress but pool A waits for pool B before home loss forwards"

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29503

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \
    --nnodes=2 \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=8 \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m nano_param_decomp.two_pool_profile_stress_serial
'
