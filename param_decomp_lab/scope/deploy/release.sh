#!/bin/bash
# Release new scope code into the running serve job without touching the tunnel.
#   deploy/release.sh frontend   # npm build + ~1s frontend bounce
#   deploy/release.sh backend    # ~15s backend bounce
#   deploy/release.sh both
set -euo pipefail
WHAT=${1:?usage: release.sh frontend|backend|both}
[[ "$WHAT" == "frontend" || "$WHAT" == "backend" || "$WHAT" == "both" ]]
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$WHAT" != "backend" ]]; then
  (cd "$DEPLOY_DIR/../frontend" && npm run build)
fi
echo "$WHAT $(date -Is)" > "$DEPLOY_DIR/RELEASE"
echo "released: $WHAT (supervisor picks it up within 5s)"
