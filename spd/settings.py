import os
from pathlib import Path

REPO_ROOT = (
    Path(os.environ["GITHUB_WORKSPACE"])
    if ("CI" in os.environ and "GITHUB_WORKSPACE" in os.environ)
    else Path(__file__).parent.parent
)

CLUSTER_BASE_PATH = Path("/mnt/polished-lake/artifacts/mechanisms/spd")
ON_CLUSTER = CLUSTER_BASE_PATH.exists()

# Base directory for SPD outputs (runs, logs, scripts, etc.)
_default_out_dir = CLUSTER_BASE_PATH if ON_CLUSTER else Path.home() / "spd_out"
SPD_OUT_DIR = Path(os.environ.get("SPD_OUT_DIR", _default_out_dir))
SPD_OUT_DIR.mkdir(parents=True, exist_ok=True)

# SLURM directories
SLURM_LOGS_DIR = SPD_OUT_DIR / "slurm_logs"
SBATCH_SCRIPTS_DIR = SPD_OUT_DIR / "sbatch_scripts"

# this is the gpu-enabled partition on the cluster
# Not sure why we call it "default" instead of "gpu" or "compute" but keeping the convention here for consistency
DEFAULT_PARTITION_NAME = "h200-reserved"

DEFAULT_PROJECT_NAME = "spd"

SPD_APP_DEFAULT_RUN: str | None = os.environ.get("SPD_APP_DEFAULT_RUN")
