"""SQLite database for component-level harvest data. NFS-hosted, write-once then read-only."""

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import orjson

from spd.harvest.config import HarvestConfig
from spd.harvest.schemas import (
    ActivationExample,
    ComponentData,
    ComponentSummary,
    ComponentTokenPMI,
)
from spd.utils.sqlite import open_nfs_sqlite

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS components (
    component_key TEXT PRIMARY KEY,
    layer TEXT NOT NULL,
    component_idx INTEGER NOT NULL,
    firing_density REAL NOT NULL,
    mean_activations TEXT NOT NULL,
    activation_examples TEXT NOT NULL,
    input_token_pmi TEXT NOT NULL,
    output_token_pmi TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    component_key TEXT NOT NULL,
    score_type TEXT NOT NULL,
    score REAL NOT NULL,
    details TEXT NOT NULL,
    PRIMARY KEY (component_key, score_type)
);
"""


def _serialize_component(
    comp: ComponentData,
) -> tuple[str, str, int, float, bytes, bytes, bytes, bytes]:
    return (
        comp.component_key,
        comp.layer,
        comp.component_idx,
        comp.firing_density,
        orjson.dumps(comp.mean_activations),
        orjson.dumps([ex.model_dump() for ex in comp.activation_examples]),
        orjson.dumps(comp.input_token_pmi.model_dump()),
        orjson.dumps(comp.output_token_pmi.model_dump()),
    )


def _deserialize_component(row: sqlite3.Row) -> ComponentData:
    return ComponentData(
        component_key=row["component_key"],
        layer=row["layer"],
        component_idx=row["component_idx"],
        firing_density=row["firing_density"],
        mean_activations=orjson.loads(row["mean_activations"]),
        activation_examples=[
            ActivationExample(**ex) for ex in orjson.loads(row["activation_examples"])
        ],
        input_token_pmi=ComponentTokenPMI(**orjson.loads(row["input_token_pmi"])),
        output_token_pmi=ComponentTokenPMI(**orjson.loads(row["output_token_pmi"])),
    )


class HarvestDB:
    def __init__(self, db_path: Path, readonly: bool = False) -> None:
        self._conn = open_nfs_sqlite(db_path, readonly)
        if not readonly:
            self._conn.executescript(_SCHEMA)

    def save_component(self, comp: ComponentData) -> None:
        row = _serialize_component(comp)
        self._conn.execute(
            "INSERT OR REPLACE INTO components VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        self._conn.commit()

    def save_components_iter(self, components: Iterable[ComponentData]) -> int:
        """Save components from an iterable, one at a time (constant memory)."""
        n = 0
        for comp in components:
            self._conn.execute(
                "INSERT OR REPLACE INTO components VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                _serialize_component(comp),
            )
            n += 1
        self._conn.commit()
        return n

    def save_config(self, config: HarvestConfig) -> None:
        data = config.model_dump()
        rows = [(k, orjson.dumps(v).decode()) for k, v in data.items()]
        self._conn.executemany(
            "INSERT OR REPLACE INTO config VALUES (?, ?)",
            rows,
        )
        self._conn.commit()

    def get_component(self, component_key: str) -> ComponentData | None:
        row = self._conn.execute(
            "SELECT * FROM components WHERE component_key = ?",
            (component_key,),
        ).fetchone()
        if row is None:
            return None
        return _deserialize_component(row)

    def get_components_bulk(self, keys: list[str]) -> dict[str, ComponentData]:
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        rows = self._conn.execute(
            f"SELECT * FROM components WHERE component_key IN ({placeholders})",
            keys,
        ).fetchall()
        return {row["component_key"]: _deserialize_component(row) for row in rows}

    def get_summary(self) -> dict[str, ComponentSummary]:
        rows = self._conn.execute(
            "SELECT component_key, layer, component_idx, firing_density, mean_activations FROM components"
        ).fetchall()
        return {
            row["component_key"]: ComponentSummary(
                layer=row["layer"],
                component_idx=row["component_idx"],
                firing_density=row["firing_density"],
                mean_activations=orjson.loads(row["mean_activations"]),
            )
            for row in rows
        }

    def get_config_dict(self) -> dict[str, object]:
        rows = self._conn.execute("SELECT key, value FROM config").fetchall()
        return {row["key"]: orjson.loads(row["value"]) for row in rows}

    def get_activation_threshold(self) -> float:
        row = self._conn.execute(
            "SELECT value FROM config WHERE key = 'activation_threshold'"
        ).fetchone()
        assert row is not None, "activation_threshold not found in config table"
        return orjson.loads(row["value"])

    def has_data(self) -> bool:
        row = self._conn.execute("SELECT EXISTS(SELECT 1 FROM components LIMIT 1)").fetchone()
        assert row is not None
        return bool(row[0])

    def get_component_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM components").fetchone()
        assert row is not None
        return row[0]

    def get_all_components(self) -> list[ComponentData]:
        """Load all components."""
        rows = self._conn.execute("SELECT * FROM components").fetchall()
        return [_deserialize_component(row) for row in rows]

    # -- Scores (e.g. intruder eval) ------------------------------------------

    def save_score(self, component_key: str, score_type: str, score: float, details: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
            (component_key, score_type, score, details),
        )
        self._conn.commit()

    def get_scores(self, score_type: str) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT component_key, score FROM scores WHERE score_type = ?",
            (score_type,),
        ).fetchall()
        return {row["component_key"]: row["score"] for row in rows}

    def close(self) -> None:
        self._conn.close()
