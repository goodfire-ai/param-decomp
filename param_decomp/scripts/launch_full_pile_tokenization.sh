#!/bin/bash
# Submit all 5 seq_len tokenization pipelines.
#
# For each seq_len:
#   - 30-task SLURM array writes per-shard datasets
#   - dependent merge job concatenates + pushes to HF (private)
#
# Idempotent at the per-shard level: if a shard's dataset_info.json
# already exists, the task is a no-op. So requeue/restart is safe.
set -euo pipefail
REPO=/mnt/polished-lake/home/oli/param-decomp
cd "$REPO"

SEQ_LENS=(2048 4096 8192 12288 16384)
PUSH_BASE=oli-gf/pile-uncopyrighted-qwen-tok

for SEQ in "${SEQ_LENS[@]}"; do
    echo "============================================================"
    echo "Submitting seq_len=$SEQ"
    echo "============================================================"

    ARRAY_JOB=$(sbatch --parsable \
        --export=ALL,SEQ_LEN=$SEQ \
        "$REPO/param_decomp/scripts/tokenize_pile_array.sbatch")
    echo "  array: $ARRAY_JOB"

    MERGE_JOB=$(sbatch --parsable \
        --dependency=afterok:$ARRAY_JOB \
        --export=ALL,SEQ_LEN=$SEQ,PUSH_TO_HUB=${PUSH_BASE}-${SEQ},HUB_PRIVATE=1 \
        "$REPO/param_decomp/scripts/merge_pile_shards.sbatch")
    echo "  merge: $MERGE_JOB (after $ARRAY_JOB)"
done

echo
echo "All submitted. Monitor with:"
echo "  squeue --me --format='%.10i %.30j %.10T %.10M'"
