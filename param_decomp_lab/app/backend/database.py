"""SQLite database for the app's runs + prompts.

The app is a read-only viewer; the only mutable state is the set of runs the user has
opened and the custom prompts they've saved. Activation contexts / correlations live in
the harvest pipeline output (`PARAM_DECOMP_OUT_DIR/runs/<run_id>/harvest/`);
interpretations live in `.../autointerp/<run_id>/`.
"""

import fcntl
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR

_DEFAULT_DB_PATH = PARAM_DECOMP_OUT_DIR / "app" / "prompt_attr.db"


def get_default_db_path() -> Path:
    """Get the default database path.

    Checks env vars in order:
    1. PARAM_DECOMP_INVESTIGATION_DIR - investigation mode, db at dir/app.db
    2. PARAM_DECOMP_APP_DB_PATH - explicit override
    3. Default: PARAM_DECOMP_OUT_DIR/app/prompt_attr.db
    """
    investigation_dir = os.environ.get("PARAM_DECOMP_INVESTIGATION_DIR")
    if investigation_dir:
        return Path(investigation_dir) / "app.db"
    env_path = os.environ.get("PARAM_DECOMP_APP_DB_PATH")
    if env_path:
        return Path(env_path)
    return _DEFAULT_DB_PATH


class Run(BaseModel):
    """A run record."""

    id: int
    wandb_path: str


class PromptRecord(BaseModel):
    """A stored prompt record containing token IDs."""

    id: int
    run_id: int
    token_ids: list[int]
    is_custom: bool = False


class PromptAttrDB:
    """SQLite store for the app's runs (keyed by wandb_path) and prompts (keyed by run_id).

    Shared across the team via NFS; DELETE journal mode + an `fcntl.flock` write lock make
    concurrent access safe. The `CREATE TABLE IF NOT EXISTS` statements are the schema
    source of truth; migrate the real DB manually.
    """

    def __init__(self, db_path: Path | None = None, check_same_thread: bool = True):
        self.db_path = db_path or get_default_db_path()
        self._lock_path = self.db_path.with_suffix(".db.lock")
        self._check_same_thread = check_same_thread
        self._conn: sqlite3.Connection | None = None

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PromptAttrDB":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @contextmanager
    def _write_lock(self):
        """Acquire an exclusive file lock for write operations (NFS-safe)."""
        with open(self._lock_path, "w") as lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    # -------------------------------------------------------------------------
    # Schema initialization
    # -------------------------------------------------------------------------

    def init_schema(self) -> None:
        """Initialize the database schema. Safe to call multiple times."""
        conn = self._get_conn()
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                wandb_path TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id),
                token_ids TEXT NOT NULL,
                context_length INTEGER NOT NULL,
                is_custom INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_prompts_run_id
                ON prompts(run_id);
        """)

        conn.commit()

    # -------------------------------------------------------------------------
    # Run operations
    # -------------------------------------------------------------------------

    def create_run(self, wandb_path: str) -> int:
        """Create a new run. Returns the run ID."""
        with self._write_lock():
            conn = self._get_conn()
            cursor = conn.execute(
                "INSERT INTO runs (wandb_path) VALUES (?)",
                (wandb_path,),
            )
            conn.commit()
            run_id = cursor.lastrowid
            assert run_id is not None
            return run_id

    def get_run_by_wandb_path(self, wandb_path: str) -> Run | None:
        """Get a run by its wandb path."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, wandb_path FROM runs WHERE wandb_path = ?",
            (wandb_path,),
        ).fetchone()
        if row is None:
            return None
        return Run(id=row["id"], wandb_path=row["wandb_path"])

    def get_run(self, run_id: int) -> Run | None:
        """Get a run by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, wandb_path FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return Run(id=row["id"], wandb_path=row["wandb_path"])

    # -------------------------------------------------------------------------
    # Prompt operations
    # -------------------------------------------------------------------------

    def find_prompt_by_token_ids(
        self,
        run_id: int,
        token_ids: list[int],
        context_length: int,
    ) -> int | None:
        """Find an existing prompt with the same token_ids."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id FROM prompts WHERE run_id = ? AND token_ids = ? AND context_length = ?",
            (run_id, json.dumps(token_ids), context_length),
        ).fetchone()
        return row[0] if row else None

    def add_custom_prompt(
        self,
        run_id: int,
        token_ids: list[int],
        context_length: int,
    ) -> int:
        """Add a custom prompt, or return the existing ID on a duplicate.

        Duplicate is keyed on `(run_id, token_ids, context_length)`.
        """
        with self._write_lock():
            existing_id = self.find_prompt_by_token_ids(run_id, token_ids, context_length)
            if existing_id is not None:
                return existing_id

            conn = self._get_conn()
            cursor = conn.execute(
                "INSERT INTO prompts (run_id, token_ids, context_length, is_custom) VALUES (?, ?, ?, 1)",
                (run_id, json.dumps(token_ids), context_length),
            )
            prompt_id = cursor.lastrowid
            assert prompt_id is not None
            conn.commit()
            return prompt_id

    def get_prompt(self, prompt_id: int) -> PromptRecord | None:
        """Get a prompt by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, run_id, token_ids, is_custom FROM prompts WHERE id = ?",
            (prompt_id,),
        ).fetchone()
        if row is None:
            return None

        return PromptRecord(
            id=row["id"],
            run_id=row["run_id"],
            token_ids=json.loads(row["token_ids"]),
            is_custom=bool(row["is_custom"]),
        )

    def get_prompt_count(self, run_id: int, context_length: int) -> int:
        """Get total number of prompts for a run with a specific context length."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM prompts WHERE run_id = ? AND context_length = ?",
            (run_id, context_length),
        ).fetchone()
        return row["cnt"]

    def get_all_prompt_ids(self, run_id: int, context_length: int) -> list[int]:
        """Get all prompt IDs for a run with a specific context length."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id FROM prompts WHERE run_id = ? AND context_length = ? ORDER BY id",
            (run_id, context_length),
        ).fetchall()
        return [row["id"] for row in rows]

    def delete_prompt(self, prompt_id: int) -> None:
        """Delete a prompt."""
        with self._write_lock():
            conn = self._get_conn()
            conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
            conn.commit()
