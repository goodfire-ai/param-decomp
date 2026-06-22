#!/usr/bin/env bash
# Silico library post-sync hook: the Lab runs this after its `uv sync` when this
# repo is provisioned under ~/.silico/libraries/. A default sync installs only the
# root `param_decomp` package; --all-packages also installs the `param_decomp_lab`
# workspace member, putting pd-lm (and the other pd-* CLIs) on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --all-packages --no-dev
