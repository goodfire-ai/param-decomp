"""Canonical output paths and ID generation for clustering artifacts."""

from pathlib import Path

from param_decomp.settings import PARAM_DECOMP_OUT_DIR
from param_decomp.utils.run_utils import generate_run_id


def clustering_run_dir(run_id: str) -> Path:
    return PARAM_DECOMP_OUT_DIR / "clustering" / "runs" / run_id


def clustering_harvest_dir(harvest_id: str) -> Path:
    return PARAM_DECOMP_OUT_DIR / "clustering" / "harvests" / harvest_id


def clustering_ensemble_dir(ensemble_id: str) -> Path:
    return PARAM_DECOMP_OUT_DIR / "clustering" / "ensembles" / ensemble_id


def new_run_id() -> str:
    return generate_run_id("clustering/runs")


def new_harvest_id() -> str:
    return generate_run_id("clustering/harvests")


def new_ensemble_id() -> str:
    return generate_run_id("clustering/ensembles")
