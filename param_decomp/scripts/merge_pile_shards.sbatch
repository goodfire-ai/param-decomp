#!/bin/bash
#SBATCH --job-name=pile-merge
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/pile-merge-%j.out
#SBATCH --error=/mnt/polished-lake/home/oli/slurm_logs/pile-merge-%j.err
#SBATCH --comment="Concatenate per-shard tokenized pile datasets into one combined Dataset, optionally push to Hub."

# Required env:
#   SEQ_LEN          — must match the array job
#   PUSH_TO_HUB      — optional, HF repo id like goodfire/pile-qwen-tok-2048
#   HUB_PRIVATE      — optional, set to 1 for private repo
: "${SEQ_LEN:?Set SEQ_LEN before sbatch}"
: "${SHARDS_DIR:=/mnt/polished-lake/home/oli/datasets/pile-qwen-tok-${SEQ_LEN}-shards}"
: "${OUTPUT_DIR:=/mnt/polished-lake/home/oli/datasets/pile-qwen-tok-${SEQ_LEN}-full}"

set -euo pipefail
REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/mnt/polished-lake/home/oli/.cache/huggingface

echo "[$(date +%H:%M:%S)] merging shards from $SHARDS_DIR → $OUTPUT_DIR (seq_len=$SEQ_LEN)"

PUSH_ARGS=""
if [[ -n "${PUSH_TO_HUB:-}" ]]; then
    PUSH_ARGS="--push_to_hub $PUSH_TO_HUB"
    if [[ "${HUB_PRIVATE:-0}" == "1" ]]; then
        PUSH_ARGS+=" --hub_private"
    fi
fi

"$REPO"/.venv/bin/python -m param_decomp.scripts.merge_pile_shards \
    --shards_dir "$SHARDS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $PUSH_ARGS
