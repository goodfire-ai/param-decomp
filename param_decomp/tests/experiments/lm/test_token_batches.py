from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from param_decomp.lm.batch_data import TokenShardServer, scan_shards
from param_decomp.lm.batch_schedule import BatchSchedule


def test_token_server_preserves_rows_and_process_partition(tmp_path: Path) -> None:
    rows = np.arange(64, dtype=np.int32).reshape(8, 8)
    pq.write_table(
        pa.table({"input_ids": pa.array(rows.tolist(), type=pa.list_(pa.int32(), 8))}),
        tmp_path / "shard.parquet",
    )
    schedule = BatchSchedule(scan_shards(tmp_path), global_batch=4, seed=7)
    servers = [TokenShardServer(schedule, 8, rank, 2) for rank in range(2)]
    for step in (1, 0, 2):
        indices = np.concatenate([selection.rows for selection in schedule.batch_rows(step)])
        actual = np.concatenate([server.local_batch(step).token_ids for server in servers])
        np.testing.assert_array_equal(actual, rows[indices])
        batch = servers[0].local_batch(step).to_device(64)
        np.testing.assert_array_equal(batch.token_ids, actual[:2])


def test_token_server_refuses_document_aware_artifacts(tmp_path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "input_ids": pa.array([[1, 2, 3, 4]], type=pa.list_(pa.int32())),
                "document_ids": pa.array([[0, 0, 1, 1]], type=pa.list_(pa.int32())),
            }
        ),
        tmp_path / "shard.parquet",
    )
    schedule = BatchSchedule(scan_shards(tmp_path), global_batch=1, seed=0)
    with pytest.raises(AssertionError, match="does not support document-aware inputs"):
        TokenShardServer(schedule, 4, 0, 1)


def test_token_batches_cross_groups_and_shards_with_exact_process_slices(tmp_path: Path) -> None:
    source = [
        np.arange(n * 8, dtype=np.int32).reshape(n, 8) + 1000 * i for i, n in enumerate((17, 19))
    ]
    for i, rows in enumerate(source):
        pq.write_table(
            pa.table({"input_ids": pa.array(rows.tolist(), type=pa.list_(pa.int32(), 8))}),
            tmp_path / f"shard-{i}.parquet",
            row_group_size=5,
        )
    schedule = BatchSchedule(scan_shards(tmp_path), global_batch=12, seed=7)
    servers = [TokenShardServer(schedule, 8, rank, 4) for rank in range(4)]
    seen = set()
    for step in (2, 0, 1):
        selections = schedule.batch_rows(step)
        assert len(selections) >= 3
        expected = np.concatenate([source[s.file_idx][s.rows] for s in selections])
        actual = np.concatenate([server.local_batch(step).token_ids for server in servers])
        np.testing.assert_array_equal(actual, expected)
        seen.update((s.file_idx, int(row)) for s in selections for row in s.rows)
    assert len(seen) == 36
