"""Canonical output paths and ID generation for clustering artifacts."""

from pathlib import Path

from param_decomp.infra.run_files import generate_run_id
from param_decomp.infra.settings import ENV


def clustering_run_dir(run_id: str) -> Path:
    return ENV.output_root / "clustering" / "runs" / run_id


def clustering_harvest_dir(harvest_id: str) -> Path:
    return ENV.output_root / "clustering" / "harvests" / harvest_id


def clustering_ensemble_dir(ensemble_id: str) -> Path:
    return ENV.output_root / "clustering" / "ensembles" / ensemble_id


def new_run_id() -> str:
    return generate_run_id("clustering/runs")


def new_harvest_id() -> str:
    return generate_run_id("clustering/harvests")


def new_ensemble_id() -> str:
    return generate_run_id("clustering/ensembles")
