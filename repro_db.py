"""Concurrent read/write reproduction for autointerp DB.

Uses a barrier to guarantee the read happens while the writer holds uncommitted
data and is actively inserting — no reliance on sleep timing.
"""

import sqlite3
import tempfile
import threading
from pathlib import Path

SCHEMA = """\
CREATE TABLE IF NOT EXISTS interpretations (
    component_key TEXT PRIMARY KEY,
    label TEXT NOT NULL
);
"""

N_WRITES = 20
# Barrier: writer writes half, then both threads sync, then reader reads mid-write.
MIDPOINT = N_WRITES // 2


def open_writer(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def open_reader(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    return conn


def count_rows(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0]


def writer_fn(
    db_path: Path,
    mid_barrier: threading.Barrier,
    read_done: threading.Event,
    end_barrier: threading.Barrier,
):
    conn = open_writer(db_path)
    for i in range(MIDPOINT):
        conn.execute("INSERT INTO interpretations VALUES (?, ?)", (f"comp:{i}", f"label {i}"))
        conn.commit()

    # Signal: first half committed, wait for reader to read
    mid_barrier.wait()
    read_done.wait()

    # Write second half
    for i in range(MIDPOINT, N_WRITES):
        conn.execute("INSERT INTO interpretations VALUES (?, ?)", (f"comp:{i}", f"label {i}"))
        conn.commit()

    end_barrier.wait()
    conn.close()


def reader_fn(
    db_path: Path,
    mid_barrier: threading.Barrier,
    read_done: threading.Event,
    end_barrier: threading.Barrier,
):
    conn = open_reader(db_path)

    # Wait until writer has committed first half
    mid_barrier.wait()
    mid_count = count_rows(conn)
    read_done.set()

    # Wait until writer has committed everything
    end_barrier.wait()
    final_count = count_rows(conn)
    conn.close()

    print(f"Mid-write read:  {mid_count} rows (expected {MIDPOINT})")
    print(f"Post-write read: {final_count} rows (expected {N_WRITES})")
    assert mid_count == MIDPOINT, f"Reader didn't see mid-write rows: got {mid_count}"
    assert final_count == N_WRITES, f"Reader didn't see final rows: got {final_count}"
    print("PASS: reader saw correct counts at both sync points")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "interp.db"

        mid_barrier = threading.Barrier(2)
        read_done = threading.Event()
        end_barrier = threading.Barrier(2)

        args = (db_path, mid_barrier, read_done, end_barrier)
        w = threading.Thread(target=writer_fn, args=args)
        r = threading.Thread(target=reader_fn, args=args)

        w.start()
        r.start()
        w.join()
        r.join()


if __name__ == "__main__":
    main()
