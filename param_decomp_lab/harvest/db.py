"""SQLite side-store for a harvest subrun: config + eval scores + intruder prompts.

Per-component example/PMI data lives in the scope artifact shards
(`param_decomp_lab/scope/artifacts.py`), not here — `HarvestRepo` reconstructs
`ComponentData` from those shards. This DB holds only the small global provenance
(config) and the LLM-eval byproducts (intruder scores + prompts). NFS-hosted, WAL mode.
"""

import threading
from pathlib import Path

import orjson

from param_decomp_lab.harvest.config import HarvestConfig
from param_decomp_lab.infra.sqlite import open_nfs_sqlite

_SCHEMA = """\
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

CREATE TABLE IF NOT EXISTS intruder_prompts (
    trial_key TEXT PRIMARY KEY,
    prompt TEXT NOT NULL
);
"""


class HarvestDB:
    # Python's sqlite3 connection is not thread-safe even with check_same_thread=False
    # (module threadsafety=1: "Threads may share the module, but not connections"). The app
    # serves concurrent reads from FastAPI's thread pool, so without this lock interleaved
    # execute/fetch calls corrupt each other's rows.
    def __init__(self, db_path: Path, readonly: bool = False) -> None:
        self._conn = open_nfs_sqlite(db_path, readonly)
        self._lock = threading.Lock()
        if not readonly:
            self._conn.executescript(_SCHEMA)

    def save_config(self, config: HarvestConfig) -> None:
        rows = [(k, orjson.dumps(v).decode()) for k, v in config.model_dump().items()]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO config VALUES (?, ?)", rows)
            self._conn.commit()

    def get_config_dict(self) -> dict[str, object]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM config").fetchall()
        return {row["key"]: orjson.loads(row["value"]) for row in rows}

    def get_activation_threshold(self) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM config WHERE key = 'activation_threshold'"
            ).fetchone()
        assert row is not None, "activation_threshold not found in config table"
        return orjson.loads(row["value"])

    # -- Scores (e.g. intruder eval) ------------------------------------------

    def save_score(self, component_key: str, score_type: str, score: float, details: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
                (component_key, score_type, score, details),
            )
            self._conn.commit()

    def get_scores(self, score_type: str) -> dict[str, float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT component_key, score FROM scores WHERE score_type = ?",
                (score_type,),
            ).fetchall()
        return {row["component_key"]: row["score"] for row in rows}

    # -- Intruder prompts ------------------------------------------------------

    def save_intruder_prompt(self, trial_key: str, prompt: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO intruder_prompts VALUES (?, ?)",
                (trial_key, prompt),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
