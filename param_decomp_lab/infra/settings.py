"""Lab-wide settings derived from environment variables."""

import os
from pathlib import Path

REPO_ROOT = (
    Path(os.environ["GITHUB_WORKSPACE"])
    if ("CI" in os.environ and "GITHUB_WORKSPACE" in os.environ)
    else Path(__file__).parent.parent.parent
)

_data_mount_env = os.environ.get("DATA_MOUNT")
DATA_MOUNT: Path | None = Path(_data_mount_env) if _data_mount_env else None
ON_CLUSTER = DATA_MOUNT is not None and DATA_MOUNT.exists()
CLUSTER_BASE_PATH: Path | None = (
    DATA_MOUNT / "artifacts/mechanisms/param-decomp"
    if ON_CLUSTER and DATA_MOUNT is not None
    else None
)
if CLUSTER_BASE_PATH is not None:
    CLUSTER_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Base directory for outputs (runs, logs, scripts, etc.).
_default_out_dir = (
    CLUSTER_BASE_PATH if CLUSTER_BASE_PATH is not None else "out"
)
PARAM_DECOMP_OUT_DIR = Path(os.environ.get("PARAM_DECOMP_OUT_DIR", _default_out_dir))
PARAM_DECOMP_OUT_DIR.mkdir(parents=True, exist_ok=True)

# SLURM directories
SLURM_LOGS_DIR = PARAM_DECOMP_OUT_DIR / "slurm_logs"
SBATCH_SCRIPTS_DIR = PARAM_DECOMP_OUT_DIR / "sbatch_scripts"

# The pd-run CLI's `--partition` flag overrides this further for ad-hoc launches.
DEFAULT_PARTITION_NAME = os.environ["PARTITION_RESERVED"]

# Default run for the app to load on startup if set
PARAM_DECOMP_APP_DEFAULT_RUN: str | None = os.environ.get("PARAM_DECOMP_APP_DEFAULT_RUN")
