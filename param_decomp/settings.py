import os
from pathlib import Path

REPO_ROOT = (
    Path(os.environ["GITHUB_WORKSPACE"])
    if ("CI" in os.environ and "GITHUB_WORKSPACE" in os.environ)
    else Path(__file__).parent.parent
)

CLUSTER_BASE_PATH = Path("/mnt/polished-lake/artifacts/mechanisms/param-decomp")
ON_CLUSTER = CLUSTER_BASE_PATH.exists()

# Base directory for outputs (runs, logs, scripts, etc.).
_default_out_dir = CLUSTER_BASE_PATH if ON_CLUSTER else Path.home() / "param_decomp_out"
PARAM_DECOMP_OUT_DIR = Path(os.environ.get("PARAM_DECOMP_OUT_DIR", _default_out_dir))
PARAM_DECOMP_OUT_DIR.mkdir(parents=True, exist_ok=True)

# SLURM directories
SLURM_LOGS_DIR = PARAM_DECOMP_OUT_DIR / "slurm_logs"
SBATCH_SCRIPTS_DIR = PARAM_DECOMP_OUT_DIR / "sbatch_scripts"

# this is the gpu-enabled partition on the cluster
# Not sure why we call it "default" instead of "gpu" or "compute" but keeping the convention here for consistency
DEFAULT_PARTITION_NAME = "h200-reserved"

DEFAULT_PROJECT_NAME = "param-decomp"

# Cluster topology: GPUs per node. Default matches the H200 cluster; override per-cluster
# via `PARAM_DECOMP_GPUS_PER_NODE` env var. Used to compute single-node vs multi-node DDP
# layout in run_slurm.py.
GPUS_PER_NODE = int(os.environ.get("PARAM_DECOMP_GPUS_PER_NODE", "8"))

# Default run for the app to load on startup if set
PARAM_DECOMP_APP_DEFAULT_RUN: str | None = os.environ.get("PARAM_DECOMP_APP_DEFAULT_RUN")
