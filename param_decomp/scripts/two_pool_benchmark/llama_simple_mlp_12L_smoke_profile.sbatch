#!/bin/bash
#SBATCH --job-name=2pool-llama12L-prof
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/2pool-llama12L-prof-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/2pool-llama12L-prof-%j.err
#SBATCH --comment="2-pool LlamaSimpleMLP-12L profile run: PROFILE_MODE=sync to identify per-phase bottlenecks. Same config as smoke (500 steps, no warmup)."

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29513

# Profile spans dumped here per rank — rank0 (pool A leader) and the first
# pool-B rank both write JSON for cross-pool timeline analysis.
export PROFILE_MODE=sync
export PROFILE_OUT_DIR=/mnt/polished-lake/home/oli/2pool_profiles/llama12L-$SLURM_JOB_ID

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "PROFILE_OUT_DIR=$PROFILE_OUT_DIR"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun --nodes=8 --ntasks=8 --ntasks-per-node=1 \
  --export=ALL,PROFILE_MODE=$PROFILE_MODE,PROFILE_OUT_DIR=$PROFILE_OUT_DIR \
  bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \
    --nnodes=8 \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=8 \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m param_decomp.scripts.two_pool_benchmark.lm_2pool_launcher \
    --run_config param_decomp/experiments/lm/llama_simple_mlp_12L_smoke.yaml \
    --topology   param_decomp/experiments/lm/llama_simple_mlp_12L_two_pool_topology.yaml \
    --wandb_project param-decomp
'
