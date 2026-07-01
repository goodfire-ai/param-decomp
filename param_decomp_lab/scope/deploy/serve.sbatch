#!/bin/bash
#SBATCH --job-name=scope-serve
#SBATCH --partition=dev
#SBATCH --qos=dev
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10000M
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --comment="scope: always-on PD component viewer (SvelteKit SSR + FastAPI, tailnet port 10010); in-place releases via deploy/RELEASE — oli (via claude)"
#SBATCH --output=/mnt/data/artifacts/mechanisms/param-decomp/slurm_logs/slurm-%j.out

set -uo pipefail
SCOPE_CHECKOUT=/mnt/home/oli/param-decomp/.claude/worktrees/agent-ad6fa023b5d67ffa7
BACKEND_PORT=8741
FRONTEND_PORT=8742
TUNNEL_LOGIN_PORT=10010
RELEASE_FILE="$SCOPE_CHECKOUT/param_decomp_lab/scope/deploy/RELEASE"

cd "$SCOPE_CHECKOUT"
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

backend_loop() {
  while true; do
    python -m param_decomp_lab.scope.backend.server --port $BACKEND_PORT --data-source artifacts
    echo "backend exited; restarting"
    sleep 1
  done
}
frontend_loop() {
  while true; do
    (cd param_decomp_lab/scope/frontend \
      && PORT=$FRONTEND_PORT SCOPE_BACKEND_URL="http://localhost:$BACKEND_PORT" node build)
    echo "frontend exited; restarting"
    sleep 1
  done
}
backend_loop &
frontend_loop &

for _ in $(seq 1 90); do
  curl -sf "http://localhost:$FRONTEND_PORT/" >/dev/null && break
  sleep 2
done
curl -sf -o /dev/null "http://localhost:$FRONTEND_PORT/"
echo "scope up on $(hostname): backend :$BACKEND_PORT, frontend :$FRONTEND_PORT"
/mnt/data/scripts/bin/tunnel-url $FRONTEND_PORT -t 48 -p $TUNNEL_LOGIN_PORT

last_release=$(stat -c %Y "$RELEASE_FILE" 2>/dev/null || echo 0)
renew_at=$(( $(date +%s) + 24 * 3600 ))
while true; do
  mtime=$(stat -c %Y "$RELEASE_FILE" 2>/dev/null || echo 0)
  if (( mtime != last_release )); then
    last_release=$mtime
    what=$(cat "$RELEASE_FILE")
    echo "release: $what"
    [[ "$what" == *backend* || "$what" == *both* ]] && fuser -k -TERM $BACKEND_PORT/tcp 2>/dev/null
    [[ "$what" == *frontend* || "$what" == *both* ]] && fuser -k -TERM $FRONTEND_PORT/tcp 2>/dev/null
  fi
  if (( $(date +%s) > renew_at )); then
    /mnt/data/scripts/bin/tunnel-url $FRONTEND_PORT -t 48 -p $TUNNEL_LOGIN_PORT \
      || echo "tunnel renewal failed; will retry next cycle"
    renew_at=$(( $(date +%s) + 24 * 3600 ))
  fi
  sleep 5
done
