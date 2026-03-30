"""Canonical output paths for clustering artifacts."""

from pathlib import Path

from spd.settings import SPD_OUT_DIR


def clustering_run_dir(run_id: str) -> Path:
    return SPD_OUT_DIR / "clustering" / "runs" / run_id


def clustering_harvest_dir(harvest_id: str) -> Path:
    return SPD_OUT_DIR / "clustering" / "harvests" / harvest_id


def clustering_ensemble_dir(ensemble_id: str) -> Path:
    return SPD_OUT_DIR / "clustering" / "ensembles" / ensemble_id
