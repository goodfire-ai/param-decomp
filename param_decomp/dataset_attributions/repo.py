"""Dataset attributions data repository.

Owns PARAM_DECOMP_OUT_DIR/dataset_attributions/<run_id>/ and provides read access
to the attribution matrix.

Use AttributionRepo.open() to construct — returns None if no attribution data exists.
Layout: dataset_attributions/<run_id>/da-YYYYMMDD_HHMMSS/dataset_attributions.pt
"""

from pathlib import Path

from param_decomp.dataset_attributions.storage import DatasetAttributionStorage
from param_decomp.settings import PARAM_DECOMP_OUT_DIR

DATASET_ATTRIBUTIONS_DIR = PARAM_DECOMP_OUT_DIR / "dataset_attributions"


def get_attributions_dir(run_id: str) -> Path:
    return DATASET_ATTRIBUTIONS_DIR / run_id


def get_attributions_subrun_dir(run_id: str, subrun_id: str) -> Path:
    return get_attributions_dir(run_id) / subrun_id


class AttributionRepo:
    """Read access to dataset attribution data for a single run.

    Constructed via AttributionRepo.open(). Storage is loaded eagerly at construction.
    """

    def __init__(self, storage: DatasetAttributionStorage, subrun_id: str) -> None:
        self._storage = storage
        self.subrun_id = subrun_id

    @classmethod
    def open(cls, run_id: str) -> "AttributionRepo | None":
        """Open attribution data for a run. Returns None if no attribution data exists."""
        base_dir = get_attributions_dir(run_id)
        if not base_dir.exists():
            return None
        candidates = [
            subrun_dir / "dataset_attributions.pt"
            for subrun_dir in base_dir.iterdir()
            if subrun_dir.is_dir() and subrun_dir.name.startswith("da-")
        ]
        existing = [p for p in candidates if p.exists()]
        if not existing:
            return None
        latest = max(existing, key=lambda p: p.stat().st_mtime)
        return cls(DatasetAttributionStorage.load(latest), subrun_id=latest.parent.name)

    def get_attributions(self) -> DatasetAttributionStorage:
        return self._storage
