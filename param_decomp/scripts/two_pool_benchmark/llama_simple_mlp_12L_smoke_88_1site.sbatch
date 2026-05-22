#!/bin/bash
#SBATCH --job-name=2pool-llama12L-88-1site
#SBATCH --nodes=11
#SBATCH --gpus-per-node=8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/2pool-llama12L-88-1site-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/2pool-llama12L-88-1site-%j.err
#SBATCH --comment="2-pool LlamaSimpleMLP-12L 1-site-per-block topology (72 groups × 1 + 16B = 88 GPUs) to test whether shrinking pool A's layerwise loop is the right lever."

set -euo pipefail

REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29515

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun --nodes=11 --ntasks=11 --ntasks-per-node=1 bash -c '
  cd '"$REPO"'
  echo "[$(hostname)] starting torchrun, SLURM_PROCID=$SLURM_PROCID"
  '"$REPO"'/.venv/bin/python -m torch.distributed.run \
    --nnodes=11 \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=8 \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m param_decomp.scripts.two_pool_benchmark.lm_2pool_launcher \
    --run_config param_decomp/experiments/lm/llama_simple_mlp_12L_smoke.yaml \
    --topology   param_decomp/experiments/lm/llama_simple_mlp_12L_two_pool_topology_88_1site.yaml \
    --wandb_project param-decomp
'
