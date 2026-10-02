"""Schedule + serving tests for `data.py` on synthetic parquet shards: determinism,
exact resume addressing, per-process partitioning."""

from pathlib import Path

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from param_decomp.core.sharding import hsdp_mesh
from param_decomp.experiments.lm.data import pack_documents
from param_decomp.lm.batch_data import (
    HostLMBatch,
    ShardServer,
    global_lm_batch,
    scan_shards,
)
from param_decomp.lm.batch_schedule import BatchSchedule

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
        table = pa.table(
            {
                "input_ids": pa.array(rows.tolist(), type=pa.list_(pa.int32())),
                "document_ids": pa.array(
                    (np.arange(SEQ)[None, :] // 3 + rows[:, 1:2] * 3).tolist(),
                    type=pa.list_(pa.int32()),
                ),
            }
        )
        pq.write_table(table, d / f"shard_{i:05d}.parquet", row_group_size=8)
    return d


def test_scan_accepts_source_shard_names(tmp_path: Path) -> None:
    table = pa.table({"input_ids": [[0] * SEQ]})
    pq.write_table(table, tmp_path / "train-00000-of-00001.parquet")

    shards = scan_shards(tmp_path)

    assert [shard.path.name for shard in shards] == ["train-00000-of-00001.parquet"]


def test_scan_and_schedule(data_dir: Path):
    shards = scan_shards(data_dir)
    assert [s.n_rows for s in shards] == [37, 53]
    sched = BatchSchedule(shards, global_batch=4, seed=7)
    assert sched.steps_per_epoch == (37 + 53) // 4
    first_group_by_file = {}
    for step in range(sched.steps_per_epoch):
        for loc in sched.batch_rows(step):
            first_group_by_file.setdefault(loc.file_idx, loc.row_group_idx)
    assert all(
        shards[file_idx].row_groups[row_group_idx].n_rows == 8
        for file_idx, row_group_idx in first_group_by_file.items()
    )

    # every step addresses a unique window; rows within an epoch never repeat
    seen: set[tuple[int, int]] = set()
    for step in range(sched.steps_per_epoch):
        selections = sched.batch_rows(step)
        assert sum(len(selection.rows) for selection in selections) == 4
        for selection in selections:
            for row in selection.rows:
                key = (selection.file_idx, int(row))
                assert key not in seen, "row served twice in one epoch"
                seen.add(key)


def _scheduled_rows(schedule: BatchSchedule, step: int) -> np.ndarray:
    return np.array(
        [
            (selection.file_idx, row)
            for selection in schedule.batch_rows(step)
            for row in selection.rows
        ]
    )


def test_determinism_and_direct_resume(data_dir: Path):
    shards = scan_shards(data_dir)
    schedule = BatchSchedule(shards, global_batch=4, seed=7)
    for step in [25, 0, 20, 3, 11]:  # nonsequential and crosses epoch 22
        direct = BatchSchedule(shards, global_batch=4, seed=7)
        assert np.array_equal(_scheduled_rows(schedule, step), _scheduled_rows(direct, step)), (
            "schedule must be a pure function of (seed, step), independent of lookup history"
        )
    other_seed = BatchSchedule(shards, global_batch=4, seed=8)
    assert not all(
        np.array_equal(_scheduled_rows(schedule, step), _scheduled_rows(other_seed, step))
        for step in range(5)
    )


def test_process_slices_partition_the_batch(data_dir: Path):
    shards = scan_shards(data_dir)
    sched = BatchSchedule(shards, global_batch=4, seed=7)
    for step in [0, 9, 13]:
        full = ShardServer(sched, SEQ, process_index=0, process_count=1).local_batch(step)
        parts = [
            ShardServer(sched, SEQ, process_index=p, process_count=2).local_batch(step)
            for p in range(2)
        ]
        np.testing.assert_array_equal(np.concatenate([p.token_ids for p in parts]), full.token_ids)
        np.testing.assert_array_equal(
            np.concatenate([p.document_ids for p in parts]), full.document_ids
        )
        expected = _scheduled_rows(sched, step)
        np.testing.assert_array_equal(full.token_ids[:, :2], expected)
        np.testing.assert_array_equal(
            full.document_ids, np.arange(SEQ)[None, :] // 3 + expected[:, 1:2] * 3
        )


def test_seq_len_mismatch_asserts(data_dir: Path):
    sched = BatchSchedule(scan_shards(data_dir), global_batch=4, seed=7)
    server = ShardServer(sched, seq_len=16, process_index=0, process_count=1)
    with pytest.raises(AssertionError, match="seq"):
        server.local_batch(0)


def test_row_group_stripes_cover_every_shard_before_revisiting(tmp_path: Path) -> None:
    for file_idx in range(3):
        rows = np.zeros((16, SEQ), dtype=np.int32)
        rows[:, 0] = file_idx
        rows[:, 1] = np.arange(16)
        pq.write_table(
            pa.table({"input_ids": [row.tolist() for row in rows]}),
            tmp_path / f"shard_{file_idx:05d}.parquet",
            row_group_size=8,
        )

    schedule = BatchSchedule(scan_shards(tmp_path), global_batch=4, seed=7)
    first_stripe = [schedule.batch_rows(step)[0] for step in range(6)]

    assert {loc.file_idx for loc in first_stripe} == {0, 1, 2}
    assert all(sum(loc.file_idx == file_idx for loc in first_stripe) == 2 for file_idx in range(3))
    assert all(
        len({loc.row_group_idx for loc in first_stripe if loc.file_idx == file_idx}) == 1
        for file_idx in range(3)
    )


@pytest.mark.parametrize("batch_size", [10, 12, 90])
def test_batches_span_small_row_groups_without_losing_interior_tails(
    data_dir: Path, batch_size: int
) -> None:
    shards = scan_shards(data_dir)
    assert max(group.n_rows for shard in shards for group in shard.row_groups) < batch_size
    schedule = BatchSchedule(shards, batch_size, seed=7)
    assert schedule.steps_per_epoch == 90 // batch_size
    seen = [
        tuple(row)
        for step in range(schedule.steps_per_epoch)
        for row in _scheduled_rows(schedule, step)
    ]
    assert len(seen) == len(set(seen)) == 90 - 90 % batch_size
    assert set(seen) <= {(file, row) for file, size in enumerate((37, 53)) for row in range(size)}
    assert any(
        len({loc.file_idx for loc in schedule.batch_rows(step)}) > 1
        for step in range(schedule.steps_per_epoch)
    )
    for step in (schedule.steps_per_epoch + 1, 0, schedule.steps_per_epoch, 2):
        fresh = BatchSchedule(shards, batch_size, seed=7)
        np.testing.assert_array_equal(_scheduled_rows(schedule, step), _scheduled_rows(fresh, step))


def test_cross_group_document_batches_preserve_full_rows_and_process_slices(data_dir: Path) -> None:
    schedule = BatchSchedule(scan_shards(data_dir), global_batch=12, seed=7)
    source = [pq.read_table(shard.path).to_pydict() for shard in schedule.shards]
    servers = [ShardServer(schedule, SEQ, rank, 4) for rank in range(4)]
    for step in (0, 3, 7, 2):
        rows = _scheduled_rows(schedule, step)
        parts = [server.local_batch(step) for server in servers]
        np.testing.assert_array_equal(
            np.concatenate([part.token_ids for part in parts]),
            np.array([source[file]["input_ids"][row] for file, row in rows], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.concatenate([part.document_ids for part in parts]),
            np.array([source[file]["document_ids"][row] for file, row in rows], dtype=np.int32),
        )


def test_batch_larger_than_corpus_is_rejected(data_dir: Path) -> None:
    with pytest.raises(AssertionError, match="larger than the corpus"):
        BatchSchedule(scan_shards(data_dir), global_batch=91, seed=7)


@pytest.mark.parametrize("batch_size", [4, 12, 90])
def test_batch_size_only_repartitions_the_seeded_row_stream(
    data_dir: Path, batch_size: int
) -> None:
    shards = scan_shards(data_dir)
    individual = BatchSchedule(shards, global_batch=1, seed=7)
    batched = BatchSchedule(shards, global_batch=batch_size, seed=7)
    for epoch in (0, 2):
        stream = np.concatenate(
            [
                _scheduled_rows(individual, epoch * individual.steps_per_epoch + step)
                for step in range(individual.steps_per_epoch)
            ]
        )
        for step in reversed(range(batched.steps_per_epoch)):
            np.testing.assert_array_equal(
                _scheduled_rows(batched, epoch * batched.steps_per_epoch + step),
                stream[step * batch_size : (step + 1) * batch_size],
            )


@pytest.mark.parametrize(
    ("tokens", "documents", "message"),
    [
        ([[1, 2]], [[0]], "identical shapes"),
        ([[1, 2]], [[-1, 0]], "nonnegative"),
        ([[-1, 2]], [[0, 0]], "nonnegative"),
        ([[1, 2, 3]], [[0, 1, 0]], "contiguous and nondecreasing"),
        ([[1, 2, 3]], [[0, 2, 2]], "contiguous and nondecreasing"),
    ],
)
def test_host_batch_refuses_invalid_ids(
    tokens: list[list[int]], documents: list[list[int]], message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        HostLMBatch(np.asarray(tokens, dtype=np.int32), np.asarray(documents, dtype=np.int32))


def test_loader_refuses_wrong_dtype(tmp_path: Path) -> None:
    pq.write_table(
        pa.table({"input_ids": [[1] * SEQ], "document_ids": [[0] * SEQ]}),
        tmp_path / "shard.parquet",
    )
    schedule = BatchSchedule(scan_shards(tmp_path), global_batch=1, seed=7)
    with pytest.raises(AssertionError, match="must be an int32 list"):
        ShardServer(schedule, SEQ, process_index=0, process_count=1).local_batch(0)


def test_document_layout_survives_device_placement() -> None:
    host = HostLMBatch(
        np.tile(np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32), (jax.device_count(), 1)),
        np.tile(np.asarray([[0, 0, 1, 1], [4, 5, 5, 6]], dtype=np.int32), (jax.device_count(), 1)),
    )
    device = host.to_device(vocab_size=9)
    mesh = hsdp_mesh(jax.device_count(), 1, 1)
    placed = global_lm_batch(host, mesh, global_batch=host.token_ids.shape[0], vocab_size=9)
    for batch in (device, placed):
        np.testing.assert_array_equal(batch.batch.token_ids, host.token_ids)
        np.testing.assert_array_equal(batch.sequence.document_ids, host.document_ids)
        assert batch.batch.token_ids.sharding == batch.sequence.document_ids.sharding


def test_host_boundary_refuses_wrong_vocabulary() -> None:
    host = HostLMBatch.from_unsegmented_sequences(np.asarray([[1, 9]], dtype=np.int32))
    with pytest.raises(AssertionError, match="vocabularies differ"):
        host.to_device(vocab_size=9)
    with pytest.raises(AssertionError, match="vocabularies differ"):
        global_lm_batch(host, hsdp_mesh(jax.device_count(), 1, 1), global_batch=1, vocab_size=9)


@pytest.mark.parametrize(
    ("tokens", "documents", "message"),
    [
        ([[1] * 7, [2] * 9], [[0] * 8, [0] * 8], "equal lengths"),
        ([[1] * 8, [2] * 8], [[0] * 7, [0] * 9], "equal lengths"),
        ([[1] * 8], [[0] * 9], "identical shapes"),
        ([[1] * 8], [[0, 0, 1, 1, 0, 0, 0, 0]], "contiguous and nondecreasing"),
    ],
)
def test_loader_refuses_malformed_document_rows(
    tmp_path: Path, tokens: list[list[int]], documents: list[list[int]], message: str
) -> None:
    pq.write_table(
        pa.table(
            {
                "input_ids": pa.array(tokens, type=pa.list_(pa.int32())),
                "document_ids": pa.array(documents, type=pa.list_(pa.int32())),
            }
        ),
        tmp_path / "shard.parquet",
    )
    schedule = BatchSchedule(scan_shards(tmp_path), global_batch=1, seed=7)
    with pytest.raises(AssertionError, match=message):
        ShardServer(schedule, SEQ, process_index=0, process_count=1).local_batch(0)


def test_loader_refuses_token_only_shards(tmp_path: Path) -> None:
    pq.write_table(
        pa.table({"input_ids": pa.array([[1] * SEQ], type=pa.list_(pa.int32()))}),
        tmp_path / "shard.parquet",
    )
    schedule = BatchSchedule(scan_shards(tmp_path), global_batch=1, seed=7)
    with pytest.raises(AssertionError, match="require input_ids and document_ids"):
        ShardServer(schedule, SEQ, process_index=0, process_count=1).local_batch(0)


def test_loader_crops_both_next_token_columns(tmp_path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "input_ids": pa.array([[1] * SEQ + [2]], type=pa.list_(pa.int32())),
                "document_ids": pa.array([[3] * (SEQ - 1) + [4, 4]], type=pa.list_(pa.int32())),
            }
        ),
        tmp_path / "shard.parquet",
    )
    schedule = BatchSchedule(scan_shards(tmp_path), global_batch=1, seed=7)
    host = ShardServer(schedule, SEQ, process_index=0, process_count=1).local_batch(0)
    np.testing.assert_array_equal(host.token_ids, [[1] * SEQ])
    np.testing.assert_array_equal(host.document_ids, [[3] * (SEQ - 1) + [4]])


def test_llama_boundaries_survive_fixed_size_shards(tmp_path: Path) -> None:
    packed = pack_documents(
        ((128000, 10, 128001), (128000, 128001), (128000, 11, 128001)), max_length=8
    )
    pq.write_table(
        pa.table(
            {
                "input_ids": pa.array(packed.token_ids.tolist(), type=pa.list_(pa.int32(), 8)),
                "document_ids": pa.array(
                    packed.document_ids.tolist(), type=pa.list_(pa.int32(), 8)
                ),
            }
        ),
        tmp_path / "shard.parquet",
    )
    schedule = BatchSchedule(scan_shards(tmp_path), global_batch=1, seed=7)
    host = ShardServer(schedule, 8, process_index=0, process_count=1).local_batch(0)
    batch = host.to_device(vocab_size=128256)
    np.testing.assert_array_equal(batch.batch.token_ids, packed.token_ids)
    np.testing.assert_array_equal(batch.sequence.document_ids, [[0, 0, 0, 1, 1, 2, 2, 2]])
    np.testing.assert_array_equal(batch.sequence.position_ids(), [[0, 1, 2, 0, 1, 0, 1, 2]])
    np.testing.assert_array_equal(
        batch.sequence.next_token_mask(), [[True, True, False, True, False, True, True]]
    )
    mask = np.asarray(batch.sequence.attention_mask())[0]
    expected = np.zeros((8, 8), dtype=bool)
    for start, stop in ((0, 3), (3, 5), (5, 8)):
        expected[start:stop, start:stop] = True
    np.testing.assert_array_equal(mask, expected)
