"""Dataset attributions data repository.

Owns SPD_OUT_DIR/dataset_attributions/<run_id>/ and provides read access
to the attribution matrix.

Use AttributionRepo.open() to construct — returns None if no attribution data exists.
Layout: dataset_attributions/<run_id>/da-YYYYMMDD_HHMMSS/dataset_attributions.pt
"""

from pathlib import Path

from spd.dataset_attributions.storage import DatasetAttributionStorage
from spd.settings import SPD_OUT_DIR

DATASET_ATTRIBUTIONS_DIR = SPD_OUT_DIR / "dataset_attributions"


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
        candidates = sorted(
            [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("da-")],
            key=lambda d: d.name,
            reverse=True,
        )
        for subrun_dir in candidates:
            path = subrun_dir / "dataset_attributions.pt"
            if path.exists():
                return cls(DatasetAttributionStorage.load(path), subrun_id=subrun_dir.name)
        return None

    def get_attributions(self) -> DatasetAttributionStorage:
        return self._storage
