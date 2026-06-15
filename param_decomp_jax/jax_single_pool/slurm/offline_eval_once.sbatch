#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --qos=opportunistic
#SBATCH --output=/mnt/data/artifacts/mechanisms/param-decomp/slurm_logs/slurm-%j.out
# Offline eval of ONE checkpoint, push-triggered by the trainer right after the
# save (run.py `_submit_offline_eval`): jsp-export (JAX venv) -> pd-offline-eval
# (torch venv, logs slow_eval/* into the run's wandb) -> delete the export.
# The trainer submits with `-J jsp-oeval-<run> --dependency=singleton`, so per-run
# evals serialize and never overlap. `<run_dir>/.offline_eval_done` dedups across
# requeue-resubmissions; `<run_dir>/offline_eval_args` (optional, one line) carries
# run-specific pd-offline-eval flags (e.g. `--micro-batch-size 4` for C=49152,
# where the default mb8 OOMs a B200).
# Usage (normally via the trainer):
#   sbatch -J jsp-oeval-<run> --dependency=singleton --comment="..." \
#     slurm/offline_eval_once.sbatch <run_dir> <step>
set -uo pipefail
RUN_DIR="$1"
STEP="$2"
JAX_REPO="$SLURM_SUBMIT_DIR"
TORCH_REPO="$(dirname "$JAX_REPO")"
MARKER="$RUN_DIR/.offline_eval_done"
touch "$MARKER"

if grep -qx "$STEP" "$MARKER"; then
  echo "step $STEP already evaluated"
  exit 0
fi
EXTRA_EVAL_ARGS=()
if [[ -f "$RUN_DIR/offline_eval_args" ]]; then
  read -r -a EXTRA_EVAL_ARGS < "$RUN_DIR/offline_eval_args"
fi
export_file="$RUN_DIR/export/model_${STEP}.safetensors"
(cd "$JAX_REPO" && .venv-cuda/bin/jsp-export "$RUN_DIR" --step "$STEP")
(cd "$TORCH_REPO" && .venv/bin/pd-offline-eval "$export_file" "$RUN_DIR/experiment_config.yaml" "${EXTRA_EVAL_ARGS[@]}")
echo "$STEP" >> "$MARKER"

# Materialize the torch-contract checkpoint (SavedLMRun reads the latest
# model_<step>.pth from the run dir), then prune to the newest 2 — mirroring the
# trainer's keep_last so the run dir doesn't hoard ~42 GB pickles per checkpoint.
"$TORCH_REPO"/.venv/bin/python - "$export_file" "$RUN_DIR/model_${STEP}.pth" <<'PYEOF'
import sys
import torch
from safetensors.torch import load_file

export_path, pth_path = sys.argv[1], sys.argv[2]
torch.save(load_file(export_path), pth_path)
print(f"materialized {pth_path}")
PYEOF
ls "$RUN_DIR" | grep -E '^model_[0-9]+\.pth$' | sort -t_ -k2 -n | head -n -2 | while read -r old_pth; do
  echo "pruning $old_pth"
  rm -f "$RUN_DIR/$old_pth"
done
rm -f "$export_file"
