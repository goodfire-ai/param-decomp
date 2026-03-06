"""SQLite database for graph interpretation data. NFS-hosted, single writer then read-only."""

import sqlite3
from pathlib import Path

from spd.autointerp.db import DONE_MARKER
from spd.graph_interp.schemas import LabelResult, PromptEdge
from spd.utils.sqlite import open_nfs_sqlite

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS output_labels (
    component_key TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    confidence TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    prompt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS input_labels (
    component_key TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    confidence TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    prompt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unified_labels (
    component_key TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    confidence TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    prompt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_edges (
    component_key TEXT NOT NULL,
    related_key TEXT NOT NULL,
    pass TEXT NOT NULL,
    attribution REAL NOT NULL,
    related_label TEXT,
    related_confidence TEXT,
    PRIMARY KEY (component_key, related_key, pass)
);

"""


class GraphInterpDB:
    """NFS-hosted. Uses open_nfs_sqlite (no WAL). Single writer, then read-only."""

    def __init__(self, db_path: Path, readonly: bool = False) -> None:
        self._conn = open_nfs_sqlite(db_path, readonly)
        if not readonly:
            self._conn.executescript(_SCHEMA)
        self._db_path = db_path

    def mark_done(self) -> None:
        (self._db_path.parent / DONE_MARKER).touch()

    # -- Output labels ---------------------------------------------------------

    def save_output_label(self, result: LabelResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO output_labels VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.component_key,
                result.label,
                result.confidence,
                result.reasoning,
                result.raw_response,
                result.prompt,
            ),
        )
        self._conn.commit()

    def get_output_label(self, component_key: str) -> LabelResult | None:
        row = self._conn.execute(
            "SELECT * FROM output_labels WHERE component_key = ?", (component_key,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_label_result(row)

    def get_all_output_labels(self) -> dict[str, LabelResult]:
        rows = self._conn.execute("SELECT * FROM output_labels").fetchall()
        return {row["component_key"]: _row_to_label_result(row) for row in rows}

    # -- Input labels ----------------------------------------------------------

    def save_input_label(self, result: LabelResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO input_labels VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.component_key,
                result.label,
                result.confidence,
                result.reasoning,
                result.raw_response,
                result.prompt,
            ),
        )
        self._conn.commit()

    def get_input_label(self, component_key: str) -> LabelResult | None:
        row = self._conn.execute(
            "SELECT * FROM input_labels WHERE component_key = ?", (component_key,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_label_result(row)

    def get_all_input_labels(self) -> dict[str, LabelResult]:
        rows = self._conn.execute("SELECT * FROM input_labels").fetchall()
        return {row["component_key"]: _row_to_label_result(row) for row in rows}

    # -- Unified labels --------------------------------------------------------

    def save_unified_label(self, result: LabelResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO unified_labels VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.component_key,
                result.label,
                result.confidence,
                result.reasoning,
                result.raw_response,
                result.prompt,
            ),
        )
        self._conn.commit()

    def get_unified_label(self, component_key: str) -> LabelResult | None:
        row = self._conn.execute(
            "SELECT * FROM unified_labels WHERE component_key = ?", (component_key,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_label_result(row)

    def get_all_unified_labels(self) -> dict[str, LabelResult]:
        rows = self._conn.execute("SELECT * FROM unified_labels").fetchall()
        return {row["component_key"]: _row_to_label_result(row) for row in rows}

    def get_completed_unified_keys(self) -> set[str]:
        rows = self._conn.execute("SELECT component_key FROM unified_labels").fetchall()
        return {row["component_key"] for row in rows}

    # -- Prompt edges ----------------------------------------------------------

    def save_prompt_edges(self, edges: list[PromptEdge]) -> None:
        rows = [
            (
                e.component_key,
                e.related_key,
                e.pass_name,
                e.attribution,
                e.related_label,
                e.related_confidence,
            )
            for e in edges
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO prompt_edges VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get_prompt_edges(self, component_key: str) -> list[PromptEdge]:
        rows = self._conn.execute(
            "SELECT * FROM prompt_edges WHERE component_key = ?", (component_key,)
        ).fetchall()
        return [_row_to_prompt_edge(row) for row in rows]

    def get_all_prompt_edges(self) -> list[PromptEdge]:
        rows = self._conn.execute("SELECT * FROM prompt_edges").fetchall()
        return [_row_to_prompt_edge(row) for row in rows]

    # -- Config ----------------------------------------------------------------

    # -- Stats -----------------------------------------------------------------

    def get_label_count(self, table: str) -> int:
        assert table in ("output_labels", "input_labels", "unified_labels")
        row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        assert row is not None
        return row[0]

    def close(self) -> None:
        self._conn.close()


def _row_to_label_result(row: sqlite3.Row) -> LabelResult:
    return LabelResult(
        component_key=row["component_key"],
        label=row["label"],
        confidence=row["confidence"],
        reasoning=row["reasoning"],
        raw_response=row["raw_response"],
        prompt=row["prompt"],
    )


def _row_to_prompt_edge(row: sqlite3.Row) -> PromptEdge:
    return PromptEdge(
        component_key=row["component_key"],
        related_key=row["related_key"],
        pass_name=row["pass"],
        attribution=row["attribution"],
        related_label=row["related_label"],
        related_confidence=row["related_confidence"],
    )
