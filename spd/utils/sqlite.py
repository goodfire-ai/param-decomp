"""SQLite connection helpers for NFS-mounted databases.

Two environments exist in this codebase:

1. **NFS databases** (harvest, autointerp, graph_interp, dataset_attributions):
   - Live at SPD_OUT_DIR on shared NFS mount
   - WAL mode MUST NOT be used — it requires POSIX advisory locking which
     NFS doesn't support reliably, causing "database is locked" errors
   - Readonly uses ?immutable=1 (no lock files created at all)
   - Write mode uses default DELETE journal

2. **App database** (prompt_attr.db):
   - Lives at SPD_OUT_DIR/app/ on NFS (shared across team)
   - Uses DELETE journal mode with fcntl.flock write locking
   - Managed by PromptAttrDB in spd/app/backend/database.py
"""

import sqlite3
from pathlib import Path


def open_nfs_sqlite(path: Path, readonly: bool) -> sqlite3.Connection:
    """Open a SQLite connection safe for NFS-mounted databases.

    Readonly: ?immutable=1 URI (zero lock files, safe for concurrent readers).
    Write: default DELETE journal (WAL breaks on NFS).
    """
    if readonly:
        conn = sqlite3.connect(f"file:{path}?immutable=1", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
