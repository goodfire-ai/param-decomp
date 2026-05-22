#!/bin/bash
#SBATCH --job-name=pile-tok-array
#SBATCH --array=0-29
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/pile-tok-%A_%a.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/pile-tok-%A_%a.err
#SBATCH --comment="Tokenize one physical shard of pile-uncopyrighted with the Qwen3 tokenizer. SLURM array — one task per train/NN.jsonl.zst. Idempotent: re-runs are no-ops if shard's dataset_info.json already exists."

# Required env: SEQ_LEN must be set in the launch environment.
#   sbatch --export=ALL,SEQ_LEN=2048 tokenize_pile_array.sbatch
: "${SEQ_LEN:?Set SEQ_LEN before sbatch, e.g. --export=ALL,SEQ_LEN=2048}"
: "${TOKENIZER:=Qwen/Qwen3-0.6B-Base}"
: "${SHARDS_DIR:=/mnt/polished-lake/home/oli/datasets/pile-qwen-tok-${SEQ_LEN}-shards}"

set -euo pipefail
REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

# Speed knobs: hf_transfer for download; local-disk HF cache.
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/mnt/polished-lake/home/oli/.cache/huggingface
# Cap tokenizer thread count to allocated CPU budget.
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=true

echo "[$(date +%H:%M:%S)] shard=$SLURM_ARRAY_TASK_ID seq_len=$SEQ_LEN tokenizer=$TOKENIZER"
echo "  SHARDS_DIR=$SHARDS_DIR"
echo "  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"

"$REPO"/.venv/bin/python -m param_decomp.scripts.pretokenize_pile_shard \
    --shard_idx "$SLURM_ARRAY_TASK_ID" \
    --tokenizer "$TOKENIZER" \
    --seq_len "$SEQ_LEN" \
    --shards_dir "$SHARDS_DIR"
