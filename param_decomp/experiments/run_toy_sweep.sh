#!/usr/bin/env bash
# Launch one run of each canonical toy decomposition (TMS 5-2 / 40-10, both with and
# without the frozen identity layer; ResidualMLP 1/2/3-layer) under a single shared wandb
# group, so their eval metrics (eval/identity_ci_error/<site> scalars and the permuted-CI
# heatmaps logged per checkpoint) land side by side in one collapsible wandb view.
#
# The toys run synchronously on CPU (pd-tms / pd-resid-mlp) — no SLURM — so this just runs
# them in sequence and prints the wandb group URL to watch.
#
# Usage:
#   ./param_decomp_lab/experiments/run_toy_sweep.sh [GROUP] [TAGS]
#     GROUP  wandb group name shared by all seven runs (default: toy-sweep-<timestamp>)
#     TAGS   optional comma-separated wandb tags applied to every run
#
# The wandb project/entity come from each config's `wandb:` block (project
# `param-decomp-toys`, entity = your default). Keep PROJECT below in sync with the configs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT="param-decomp-toys"

GROUP="${1:-toy-sweep-$(date +%Y%m%d-%H%M%S)}"
TAGS="${2:-}"

source "$REPO_ROOT/.venv/bin/activate"

# Resolve the wandb entity the runs will log under, so we can print a live group URL.
ENTITY="$(python -c 'from param_decomp_lab.infra.wandb import get_wandb_entity; print(get_wandb_entity())')"
GROUP_URL="https://wandb.ai/${ENTITY}/${PROJECT}/groups/${GROUP}/workspace"

# (runner config) pairs — pd-tms for the TMS toys, pd-resid-mlp for the ResidualMLP toys.
RUNS=(
  "pd-tms:$SCRIPT_DIR/tms/configs/tms_5-2.yaml"
  "pd-tms:$SCRIPT_DIR/tms/configs/tms_5-2-id.yaml"
  "pd-tms:$SCRIPT_DIR/tms/configs/tms_40-10.yaml"
  "pd-tms:$SCRIPT_DIR/tms/configs/tms_40-10-id.yaml"
  "pd-resid-mlp:$SCRIPT_DIR/resid_mlp/configs/resid_mlp_1l.yaml"
  "pd-resid-mlp:$SCRIPT_DIR/resid_mlp/configs/resid_mlp_2l.yaml"
  "pd-resid-mlp:$SCRIPT_DIR/resid_mlp/configs/resid_mlp_3l.yaml"
)

echo "Launching ${#RUNS[@]} toy decompositions under wandb group '${GROUP}'"
echo "Watch them here: ${GROUP_URL}"
echo

for entry in "${RUNS[@]}"; do
  runner="${entry%%:*}"
  config="${entry#*:}"
  echo "=== ${runner} $(basename "$config") ==="
  args=("$config" --group "$GROUP")
  if [[ -n "$TAGS" ]]; then
    args+=(--tags "$TAGS")
  fi
  "$runner" "${args[@]}"
  echo
done

echo "All ${#RUNS[@]} toy runs finished."
echo "Eval metrics (identity_ci_error + permuted-CI heatmaps) are in the wandb group: ${GROUP_URL}"
