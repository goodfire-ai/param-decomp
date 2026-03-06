"""SQLite database for autointerp data (interpretations and scores). NFS-hosted, single writer then read-only."""

import sqlite3
from pathlib import Path

import orjson

from spd.autointerp.schemas import InterpretationResult
from spd.utils.sqlite import open_nfs_sqlite

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS interpretations (
    component_key TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    confidence TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    prompt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    component_key TEXT NOT NULL,
    score_type TEXT NOT NULL,
    score REAL NOT NULL,
    details TEXT NOT NULL,
    PRIMARY KEY (component_key, score_type)
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

DONE_MARKER = ".done"


class InterpDB:
    def __init__(self, db_path: Path, readonly: bool = False) -> None:
        self._conn = open_nfs_sqlite(db_path, readonly)
        if not readonly:
            self._conn.executescript(_SCHEMA)
        self._db_path = db_path

    def mark_done(self) -> None:
        (self._db_path.parent / DONE_MARKER).touch()

    def save_interpretation(self, result: InterpretationResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO interpretations VALUES (?, ?, ?, ?, ?, ?)",
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

    def save_interpretations(self, results: list[InterpretationResult]) -> None:
        rows = [
            (r.component_key, r.label, r.confidence, r.reasoning, r.raw_response, r.prompt)
            for r in results
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO interpretations VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get_interpretation(self, component_key: str) -> InterpretationResult | None:
        row = self._conn.execute(
            "SELECT * FROM interpretations WHERE component_key = ?",
            (component_key,),
        ).fetchone()
        if row is None:
            return None
        return InterpretationResult(
            component_key=row["component_key"],
            label=row["label"],
            confidence=row["confidence"],
            reasoning=row["reasoning"],
            raw_response=row["raw_response"],
            prompt=row["prompt"],
        )

    def get_all_interpretations(self) -> dict[str, InterpretationResult]:
        rows = self._conn.execute("SELECT * FROM interpretations").fetchall()
        return {
            row["component_key"]: InterpretationResult(
                component_key=row["component_key"],
                label=row["label"],
                confidence=row["confidence"],
                reasoning=row["reasoning"],
                raw_response=row["raw_response"],
                prompt=row["prompt"],
            )
            for row in rows
        }

    def get_completed_keys(self) -> set[str]:
        rows = self._conn.execute("SELECT component_key FROM interpretations").fetchall()
        return {row["component_key"] for row in rows}

    def save_score(self, component_key: str, score_type: str, score: float, details: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
            (component_key, score_type, score, details),
        )
        self._conn.commit()

    def save_scores(self, score_type: str, scores: list[tuple[str, float, str]]) -> None:
        rows = [(key, score_type, score, details) for key, score, details in scores]
        self._conn.executemany(
            "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get_scores(self, score_type: str) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT component_key, score FROM scores WHERE score_type = ?",
            (score_type,),
        ).fetchall()
        return {row["component_key"]: row["score"] for row in rows}

    def save_config(self, key: str, value: object) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO config VALUES (?, ?)",
            (key, orjson.dumps(value).decode()),
        )
        self._conn.commit()

    def has_interpretations(self) -> bool:
        row = self._conn.execute("SELECT EXISTS(SELECT 1 FROM interpretations LIMIT 1)").fetchone()
        assert row is not None
        return bool(row[0])

    def get_interpretation_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM interpretations").fetchone()
        assert row is not None
        return row[0]

    def has_scores(self, score_type: str) -> bool:
        row = self._conn.execute(
            "SELECT EXISTS(SELECT 1 FROM scores WHERE score_type = ? LIMIT 1)",
            (score_type,),
        ).fetchone()
        assert row is not None
        return bool(row[0])

    def close(self) -> None:
        self._conn.close()
