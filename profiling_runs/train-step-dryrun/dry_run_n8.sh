#!/bin/bash
#SBATCH --job-name=train-step-prof-baseline-n8
#SBATCH --partition=h200-reserved
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/polished-lake/artifacts/mechanisms/param-decomp/slurm_logs/slurm-%A_%a.out
#SBATCH --array=1-1%1

set -euo pipefail
umask 002  # Ensure files are group-writable

export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set per-task SLURM comment
case $SLURM_ARRAY_TASK_ID in
    1)
        scontrol update job="${SLURM_ARRAY_JOB_ID}_1" comment="baseline_ddp"
        ;;
esac

cd "/mnt/polished-lake/home/braun/param-decomp-fsdp-profile"
source .venv/bin/activate

# Execute the appropriate command based on array task ID
case $SLURM_ARRAY_TASK_ID in
    1)
        cd /mnt/polished-lake/home/braun/param-decomp-fsdp-profile && source .venv/bin/activate && mkdir -p profiling_runs/train-step-dryrun/n8/baseline_ddp && torchrun --standalone --nproc_per_node=8 --master_port=24000 scripts/profile_train_step.py --config-path /mnt/polished-lake/home/braun/param-decomp-fsdp-profile/param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml --out-dir profiling_runs/train-step-dryrun/n8/baseline_ddp --profile-label baseline_ddp --strategy ddp --batch-size 64 --warmup-steps 1 --measure-steps 1 --trace-steps 0
        ;;
esac
