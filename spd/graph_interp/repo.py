"""Graph interpretation data repository.

Owns SPD_OUT_DIR/graph_interp/<decomposition_id>/ and provides read access
to output, input, and unified labels.

Use GraphInterpRepo.open() to construct — returns None if no data exists.
"""

from pathlib import Path
from typing import Any

import yaml

from spd.graph_interp.db import DONE_MARKER, GraphInterpDB
from spd.graph_interp.schemas import LabelResult, PromptEdge, get_graph_interp_dir


class GraphInterpRepo:
    """Read access to graph interpretation data for a single run."""

    def __init__(self, db: GraphInterpDB, subrun_dir: Path, run_id: str) -> None:
        self._db = db
        self._subrun_dir = subrun_dir
        self.subrun_id = subrun_dir.name
        self.run_id = run_id

    @classmethod
    def open(cls, run_id: str) -> "GraphInterpRepo | None":
        """Open graph interp data for a run. Returns None if no data exists."""
        base_dir = get_graph_interp_dir(run_id)
        if not base_dir.exists():
            return None
        candidates = sorted(
            [
                d
                for d in base_dir.iterdir()
                if d.is_dir() and d.name.startswith("ti-") and (d / DONE_MARKER).exists()
            ],
            key=lambda d: d.name,
        )
        if not candidates:
            return None
        subrun_dir = candidates[-1]
        db_path = subrun_dir / "interp.db"
        if not db_path.exists():
            return None
        return cls(
            db=GraphInterpDB(db_path, readonly=True),
            subrun_dir=subrun_dir,
            run_id=run_id,
        )

    def get_config(self) -> dict[str, Any] | None:
        config_path = self._subrun_dir / "config.yaml"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            return yaml.safe_load(f)

    # -- Labels ----------------------------------------------------------------

    def get_all_output_labels(self) -> dict[str, LabelResult]:
        return self._db.get_all_output_labels()

    def get_all_input_labels(self) -> dict[str, LabelResult]:
        return self._db.get_all_input_labels()

    def get_all_unified_labels(self) -> dict[str, LabelResult]:
        return self._db.get_all_unified_labels()

    def get_output_label(self, component_key: str) -> LabelResult | None:
        return self._db.get_output_label(component_key)

    def get_input_label(self, component_key: str) -> LabelResult | None:
        return self._db.get_input_label(component_key)

    def get_unified_label(self, component_key: str) -> LabelResult | None:
        return self._db.get_unified_label(component_key)

    # -- Edges -----------------------------------------------------------------

    def get_prompt_edges(self, component_key: str) -> list[PromptEdge]:
        return self._db.get_prompt_edges(component_key)

    def get_all_prompt_edges(self) -> list[PromptEdge]:
        return self._db.get_all_prompt_edges()

    # -- Stats -----------------------------------------------------------------

    def get_label_counts(self) -> dict[str, int]:
        return {
            "output": self._db.get_label_count("output_labels"),
            "input": self._db.get_label_count("input_labels"),
            "unified": self._db.get_label_count("unified_labels"),
        }
