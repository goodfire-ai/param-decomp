#!/bin/bash
#SBATCH --job-name=pretokenize-pile-qwen3-1p7b
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/home/oli/slurm_logs/pretokenize-pile-qwen3-1p7b-%j.out
#SBATCH --error=/mnt/home/oli/slurm_logs/pretokenize-pile-qwen3-1p7b-%j.err
#SBATCH --comment="Pre-tokenize monology/pile-uncopyrighted with Qwen3-1.7B-Base tokenizer at seq_len=2048. CPU-bound; the requested GPU is unused but is the cluster's minimum allocation unit."

set -euo pipefail

REPO=/mnt/home/oli/param-decomp
cd "$REPO"

export HF_HOME=/mnt/data/artifacts/hf_cache
export HF_HUB_OFFLINE=0

echo "[pretokenize] node=$(hostname)"
echo "[pretokenize] HF_HOME=$HF_HOME"

"$REPO"/.venv/bin/python -m param_decomp_lab.scripts.pile.pretokenize_pile \
  --tokenizer Qwen/Qwen3-1.7B-Base \
  --output_dir /mnt/data/artifacts/datasets/pile-uncopyrighted-qwen3-1p7b-tok-2048 \
  --seq_len 2048 \
  --n_tokens 100000000
