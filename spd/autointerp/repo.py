"""Autointerp data repository.

Owns SPD_OUT_DIR/autointerp/<run_id>/ and provides read access to
interpretations and evaluation scores.

Each autointerp subrun (a-YYYYMMDD_HHMMSS) has its own interp.db.
Use InterpRepo.open() to construct — returns None if no autointerp data exists.
"""

from pathlib import Path
from typing import Any

import yaml

from spd.autointerp.db import InterpDB
from spd.autointerp.schemas import InterpretationResult, get_autointerp_dir


class InterpRepo:
    """Read access to autointerp data for a single run.

    Constructed via InterpRepo.open(). DB is opened eagerly at construction.
    """

    def __init__(self, db: InterpDB, subrun_dir: Path, run_id: str) -> None:
        self._db = db
        self._subrun_dir = subrun_dir
        self.subrun_id = subrun_dir.name
        self.run_id = run_id

    @classmethod
    def _find_latest_subrun_dir(cls, run_id: str) -> Path | None:
        autointerp_dir = get_autointerp_dir(run_id)
        if not autointerp_dir.exists():
            return None
        candidates = sorted(
            [d for d in autointerp_dir.iterdir() if d.is_dir() and d.name.startswith("a-")],
            key=lambda d: d.name,
        )
        return candidates[-1] if candidates else None

    @classmethod
    def open(cls, run_id: str) -> "InterpRepo | None":
        """Open autointerp data for a run (read-only). Returns None if no data exists."""
        subrun_dir = cls._find_latest_subrun_dir(run_id)
        if subrun_dir is None:
            return None
        db_path = subrun_dir / "interp.db"
        if not db_path.exists():
            return None
        return cls(
            db=InterpDB(db_path, readonly=True),
            subrun_dir=subrun_dir,
            run_id=run_id,
        )

    @classmethod
    def open_or_create(cls, run_id: str) -> "InterpRepo":
        """Open or create a writable autointerp repo for a run."""
        from datetime import datetime

        subrun_dir = cls._find_latest_subrun_dir(run_id)
        if subrun_dir is not None and (subrun_dir / "interp.db").exists():
            return cls(
                db=InterpDB(subrun_dir / "interp.db"),
                subrun_dir=subrun_dir,
                run_id=run_id,
            )
        autointerp_dir = get_autointerp_dir(run_id)
        subrun_dir = autointerp_dir / f"a-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        subrun_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            db=InterpDB(subrun_dir / "interp.db"),
            subrun_dir=subrun_dir,
            run_id=run_id,
        )

    # -- Provenance ------------------------------------------------------------

    def get_config(self) -> dict[str, Any] | None:
        config_path = self._subrun_dir / "config.yaml"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            return yaml.safe_load(f)

    def get_interpretation_count(self) -> int:
        return self._db.get_interpretation_count()

    def get_available_score_types(self) -> list[str]:
        return [st for st in ["detection", "fuzzing"] if self._db.has_scores(st)]

    # -- Interpretations -------------------------------------------------------

    def get_all_interpretations(self) -> dict[str, InterpretationResult]:
        return self._db.get_all_interpretations()

    def get_interpretation(self, component_key: str) -> InterpretationResult | None:
        return self._db.get_interpretation(component_key)

    def save_interpretation(self, result: InterpretationResult) -> None:
        self._db.save_interpretation(result)

    # -- Eval scores (label-dependent only) ------------------------------------

    def get_detection_scores(self) -> dict[str, float] | None:
        scores = self._db.get_scores("detection")
        return scores if scores else None

    def get_fuzzing_scores(self) -> dict[str, float] | None:
        scores = self._db.get_scores("fuzzing")
        return scores if scores else None

    def get_scores(self, score_type: str) -> dict[str, float]:
        scores = self._db.get_scores(score_type)
        return scores if scores else {}

    def save_score(self, component_key: str, score_type: str, score: float, details: str) -> None:
        self._db.save_score(component_key, score_type, score, details)
