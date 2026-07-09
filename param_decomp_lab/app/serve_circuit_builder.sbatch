#!/bin/bash
# Serve the PD app (with the circuit-builder tab) on a GPU node + tailnet tunnel.
#
# Usage, from a repo checkout of feature/circuit-builder with a synced .venv
# (`uv sync --all-packages` + `npm ci` in param_decomp_lab/app/frontend):
#
#     sbatch param_decomp_lab/app/serve_circuit_builder.sbatch
#
# The job log prints the tunnel URL (also written to app-serve-<jobid>.url in the
# submit dir). The URL is tailnet-only. The backend is warm-loaded with
# $CIRCUIT_BUILDER_RUN (default p-55ea3f9b) after startup. Job/tunnel last 24h;
# resubmit when it expires.
#SBATCH --job-name=circuit-builder-app
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=app-serve-%j.log

set -u
export DATA_MOUNT="${DATA_MOUNT:-/mnt/data}"
RUN_REF="${CIRCUIT_BUILDER_RUN:-p-55ea3f9b}"
cd "$SLURM_SUBMIT_DIR"
test -x .venv/bin/python || { echo "no .venv — run 'uv sync --all-packages' first"; exit 1; }

.venv/bin/python -u -m param_decomp_lab.app.run_app > "app-run-$SLURM_JOB_ID.log" 2>&1 &
APP_PID=$!

for _ in $(seq 1 60); do
    FPORT=$(grep -oP "Ready: .*http://localhost:\K[0-9]+" "app-run-$SLURM_JOB_ID.log" 2>/dev/null | head -1)
    [ -n "$FPORT" ] && break
    kill -0 $APP_PID 2>/dev/null || { echo "run_app died"; tail -30 "app-run-$SLURM_JOB_ID.log"; exit 1; }
    sleep 3
done
[ -n "${FPORT:-}" ] || { echo "no Ready line after 180s"; tail -30 "app-run-$SLURM_JOB_ID.log"; exit 1; }
echo "frontend port: $FPORT"

/mnt/data/scripts/bin/tunnel-url "$FPORT" -t 24 | tee "app-serve-$SLURM_JOB_ID.url"

echo "warm-loading $RUN_REF..."
curl -s -X POST "http://localhost:$FPORT/api/circuit_builder/load" \
    -H "Content-Type: application/json" \
    -d "{\"source\":\"run\",\"run_ref\":\"$RUN_REF\"}" --max-time 600 && echo " ...loaded"

wait $APP_PID
