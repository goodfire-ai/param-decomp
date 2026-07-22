"""Schedule + serving tests for `data.py` on synthetic parquet shards: determinism,
exact resume addressing, per-process partitioning."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from param_decomp.data import BatchSchedule, ShardServer, scan_shards

SEQ = 8


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("shards")
    rng = np.random.default_rng(0)
    for i, n_rows in enumerate([37, 53]):
        # encode (shard, row) into the first two tokens so tests can identify rows
        rows = rng.integers(0, 1000, size=(n_rows, SEQ), dtype=np.int32)
        rows[:, 0] = i
        rows[:, 1] = np.arange(n_rows)
        table = pa.table({"input_ids": [row.tolist() for row in rows]})
        pq.write_table(table, d / f"shard_{i:05d}.parquet")
    return d


def test_scan_and_schedule(data_dir: Path):
    shards = scan_shards(data_dir)
    assert [s.n_rows for s in shards] == [37, 53]
    sched = BatchSchedule(shards, global_batch=4, seed=7)
    assert sched.steps_per_epoch == 37 // 4 + 53 // 4

    # every step addresses a unique window; rows within an epoch never repeat
    seen: set[tuple[int, int]] = set()
    for step in range(sched.steps_per_epoch):
        loc, rows = sched.batch_rows(step)
        assert len(rows) == 4
        for r in rows:
            key = (loc.file_idx, int(r))
            assert key not in seen, "row served twice in one epoch"
            seen.add(key)


def test_determinism_and_o1_resume(data_dir: Path):
    shards = scan_shards(data_dir)
    a = BatchSchedule(shards, global_batch=4, seed=7)
    b = BatchSchedule(shards, global_batch=4, seed=7)
    for step in [0, 3, 11, 20, 25]:  # includes epoch wrap (steps_per_epoch == 22)
        la, ra = a.batch_rows(step)
        lb, rb = b.batch_rows(step)
        assert la == lb and (ra == rb).all(), "schedule must be a pure function of (seed, step)"
    c = BatchSchedule(shards, global_batch=4, seed=8)
    assert not all((a.batch_rows(s)[1] == c.batch_rows(s)[1]).all() for s in range(5))


def test_process_slices_partition_the_batch(data_dir: Path):
    shards = scan_shards(data_dir)
    sched = BatchSchedule(shards, global_batch=4, seed=7)
    for step in [0, 9, 13]:
        full = ShardServer(sched, SEQ, process_index=0, process_count=1).local_batch(step)
        parts = [
            ShardServer(sched, SEQ, process_index=p, process_count=2).local_batch(step)
            for p in range(2)
        ]
        assert (np.concatenate(parts) == full).all(), "process slices must tile the global batch"
        loc, rows = sched.batch_rows(step)
        assert (full[:, 0] == loc.file_idx).all()
        assert (full[:, 1] == rows).all(), "served rows must be exactly the scheduled rows"


def test_seq_len_mismatch_asserts(data_dir: Path):
    sched = BatchSchedule(scan_shards(data_dir), global_batch=4, seed=7)
    server = ShardServer(sched, seq_len=16, process_index=0, process_count=1)
    with pytest.raises(AssertionError, match="seq"):
        server.local_batch(0)
