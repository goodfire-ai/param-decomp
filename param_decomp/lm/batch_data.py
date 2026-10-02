"""Read planned Parquet rows into process-local batches and place them on devices."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from numpy.typing import NDArray

from param_decomp.core.placement import batch_axes
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.lm.batch_schedule import (
    BatchRowIndices,
    BatchRowSchedule,
    RowGroupInfo,
    RowGroupSelection,
    ShardInfo,
    slice_batch_rows,
)
from param_decomp.sequence import SequenceLayout


@dataclass(frozen=True)
class HostLMBatch:
    """Packed host rows with validated, contiguous document identity."""

    token_ids: NDArray[np.int32]
    document_ids: NDArray[np.int32]

    def __post_init__(self) -> None:
        assert self.token_ids.dtype == self.document_ids.dtype == np.int32, (
            "token_ids and document_ids must be int32"
        )
        assert self.token_ids.ndim == 2 and self.token_ids.shape[1] > 0, (
            "token_ids must have shape [batch, sequence] with a nonempty sequence"
        )
        assert self.document_ids.shape == self.token_ids.shape, (
            "token_ids and document_ids must have identical shapes"
        )
        assert np.all(self.token_ids >= 0), "token IDs must be nonnegative"
        assert np.all(self.document_ids >= 0), "document IDs must be nonnegative"
        transitions = np.diff(self.document_ids, axis=1)
        assert np.all((transitions == 0) | (transitions == 1)), (
            "document IDs must be contiguous and nondecreasing within each row"
        )

    def validate_vocabulary(self, vocab_size: int) -> None:
        assert vocab_size > 0, "vocab_size must be positive"
        assert np.all(self.token_ids < vocab_size), (
            f"token IDs outside [0, {vocab_size}); dataset and target vocabularies differ"
        )

    def to_device(self, vocab_size: int) -> LMBatchWithDocuments:
        self.validate_vocabulary(vocab_size)
        return LMBatchWithDocuments(
            LMBatch(jax.device_put(self.token_ids)),
            SequenceLayout(jax.device_put(self.document_ids)),
        )

    @classmethod
    def from_unsegmented_sequences(cls, token_ids: NDArray[np.int32]) -> "HostLMBatch":
        """Treat each sequence as unpadded, without internal document boundaries."""
        return cls(token_ids, np.zeros_like(token_ids))


def global_lm_batch(
    local: HostLMBatch, mesh: Mesh, global_batch: int, vocab_size: int
) -> LMBatchWithDocuments:
    """Place both packed arrays on the same global batch axes after host validation."""
    local.validate_vocabulary(vocab_size)
    sharding = NamedSharding(mesh, P(batch_axes(mesh)))
    shape = (global_batch, local.token_ids.shape[1])
    return LMBatchWithDocuments(
        LMBatch(jax.make_array_from_process_local_data(sharding, local.token_ids, shape)),
        SequenceLayout(jax.make_array_from_process_local_data(sharding, local.document_ids, shape)),
    )


def scan_shards(data_dir: Path) -> tuple[ShardInfo, ...]:
    files = sorted(data_dir.glob("*.parquet"))
    assert files, f"no *.parquet under {data_dir}"

    shards = []
    for path in files:
        metadata = pq.ParquetFile(path).metadata
        row_groups = []
        start_row = 0
        for row_group_idx in range(metadata.num_row_groups):
            n_rows = metadata.row_group(row_group_idx).num_rows
            row_groups.append(RowGroupInfo(start_row=start_row, n_rows=n_rows))
            start_row += n_rows
        assert start_row == metadata.num_rows
        shards.append(ShardInfo(path=path, row_groups=tuple(row_groups)))
    return tuple(shards)


def _local_batch_rows(
    schedule: BatchRowSchedule, step: int, process_index: int, process_count: int
) -> BatchRowIndices:
    rows = schedule.batch_rows(step)
    assert sum(len(selection.rows) for selection in rows) == schedule.global_batch
    per_process = schedule.global_batch // process_count
    start = process_index * per_process
    return slice_batch_rows(rows, start, start + per_process)


def _read_ids(table: object, name: str, seq_len: int) -> NDArray[np.int32]:
    ids = pa.table(table).column(name).combine_chunks()
    assert (
        pa.types.is_list(ids.type)
        or pa.types.is_large_list(ids.type)
        or pa.types.is_fixed_size_list(ids.type)
    ) and ids.type.value_type == pa.int32(), f"{name} must be an int32 list"
    assert ids.null_count == 0, f"{name} has null rows"
    lengths = pc.call_function("list_value_length", [ids]).to_numpy(zero_copy_only=False)
    assert len(lengths) > 0 and np.all(lengths == lengths[0]), (
        f"{name} rows must have equal lengths"
    )
    assert lengths[0] in (seq_len, seq_len + 1), (
        f"{name} rows have seq {lengths[0]}, config says {seq_len}"
    )
    flat = ids.flatten()
    assert flat.null_count == 0, f"{name} has null IDs"
    return flat.to_numpy(zero_copy_only=False).reshape(len(lengths), -1)


def _read_row_group(shard: ShardInfo, row_group_idx: int, seq_len: int) -> HostLMBatch:
    """One row group's packed rows, each cut to `seq_len`, as training reads them."""
    table = pq.ParquetFile(shard.path).read_row_group(row_group_idx)
    assert set(table.column_names) == {"input_ids", "document_ids"}, (
        f"{shard.path}: shards require input_ids and document_ids; tokenize older datasets again"
    )
    full = HostLMBatch(
        _read_ids(table, "input_ids", seq_len), _read_ids(table, "document_ids", seq_len)
    )
    return HostLMBatch(full.token_ids[:, :seq_len], full.document_ids[:, :seq_len])


def read_shard_rows(shard: ShardInfo, rows: Sequence[int], seq_len: int) -> HostLMBatch:
    """Rows `rows` of one shard, in that order, as training reads them."""
    assert all(0 <= row < shard.n_rows for row in rows), (rows, shard.n_rows)
    starts = np.array([group.start_row for group in shard.row_groups])
    groups = np.searchsorted(starts, rows, side="right") - 1
    read = {int(group): _read_row_group(shard, int(group), seq_len) for group in set(groups)}
    local_rows = np.asarray(rows) - starts[groups]
    placed = list(zip(groups.tolist(), local_rows.tolist(), strict=True))
    return HostLMBatch(
        np.stack([read[group].token_ids[local] for group, local in placed]),
        np.stack([read[group].document_ids[local] for group, local in placed]),
    )


class ShardServer:
    """Read scheduled row groups and serve this process's global-batch slice."""

    def __init__(
        self,
        schedule: BatchRowSchedule,
        seq_len: int,
        process_index: int,
        process_count: int,
    ):
        assert process_count > 0 and 0 <= process_index < process_count
        assert seq_len > 0
        assert schedule.global_batch % process_count == 0, (
            f"global_batch={schedule.global_batch} not divisible by {process_count} processes"
        )
        self.schedule = schedule
        self.seq_len = seq_len
        self.process_index = process_index
        self.process_count = process_count
        self._loaded_row_group: tuple[int, int] | None = None
        self._batch: HostLMBatch | None = None

    @property
    def per_process(self) -> int:
        return self.schedule.global_batch // self.process_count

    def _load_row_group(self, loc: RowGroupSelection) -> HostLMBatch:
        key = (loc.file_idx, loc.row_group_idx)
        if self._loaded_row_group != key:
            self._batch = _read_row_group(
                self.schedule.shards[loc.file_idx], loc.row_group_idx, self.seq_len
            )
            self._loaded_row_group = key
        assert self._batch is not None
        return self._batch

    def local_batch(self, step: int) -> HostLMBatch:
        """This process's paired `[per_process, seq_len]` int32 rows."""
        tokens, documents = [], []
        for selection in _local_batch_rows(
            self.schedule, step, self.process_index, self.process_count
        ):
            group = self.schedule.shards[selection.file_idx].row_groups[selection.row_group_idx]
            batch = self._load_row_group(selection)
            rows = selection.rows - group.start_row
            tokens.append(batch.token_ids[rows])
            documents.append(batch.document_ids[rows])
        return HostLMBatch(
            np.concatenate(tokens),
            np.concatenate(documents),
        )


@dataclass(frozen=True)
class HostTokenBatch:
    token_ids: NDArray[np.int32]

    def __post_init__(self) -> None:
        assert self.token_ids.dtype == np.int32, "token_ids must be int32"
        assert self.token_ids.ndim == 2 and self.token_ids.shape[1] > 0, (
            "token_ids must have shape [batch, sequence] with a nonempty sequence"
        )
        assert np.all(self.token_ids >= 0), "token IDs must be nonnegative"

    def validate_vocabulary(self, vocab_size: int) -> None:
        assert vocab_size > 0, "vocab_size must be positive"
        assert np.all(self.token_ids < vocab_size), (
            f"token IDs outside [0, {vocab_size}); dataset and target vocabularies differ"
        )

    def to_device(self, vocab_size: int) -> LMBatch:
        self.validate_vocabulary(vocab_size)
        return LMBatch(jax.device_put(self.token_ids))


def global_token_batch(
    local: HostTokenBatch, mesh: Mesh, global_batch: int, vocab_size: int
) -> LMBatch:
    local.validate_vocabulary(vocab_size)
    sharding = NamedSharding(mesh, P(batch_axes(mesh)))
    tokens = jax.make_array_from_process_local_data(
        sharding, local.token_ids, (global_batch, local.token_ids.shape[1])
    )
    return LMBatch(tokens)


def validate_token_shards(shards: tuple[ShardInfo, ...]) -> None:
    for shard in shards:
        assert set(pq.ParquetFile(shard.path).schema_arrow.names) == {"input_ids"}, (
            f"{shard.path}: token-only inputs require exactly input_ids; "
            "this target does not support document-aware inputs"
        )


class TokenShardServer:
    """Serve token-only corpora; document-aware shards are a different input type."""

    def __init__(
        self, schedule: BatchRowSchedule, seq_len: int, process_index: int, process_count: int
    ):
        assert process_count > 0 and 0 <= process_index < process_count
        assert seq_len > 0
        assert schedule.global_batch % process_count == 0
        self.schedule = schedule
        self.seq_len = seq_len
        self.process_index = process_index
        self.process_count = process_count
        self._loaded_row_group: tuple[int, int] | None = None
        self._batch: HostTokenBatch | None = None
        validate_token_shards(schedule.shards)

    @property
    def per_process(self) -> int:
        return self.schedule.global_batch // self.process_count

    def _load_row_group(self, loc: RowGroupSelection) -> HostTokenBatch:
        key = (loc.file_idx, loc.row_group_idx)
        if self._loaded_row_group != key:
            shard = self.schedule.shards[loc.file_idx]
            table = pq.ParquetFile(shard.path).read_row_group(loc.row_group_idx)

            full = HostTokenBatch(_read_ids(table, "input_ids", self.seq_len))
            self._batch = HostTokenBatch(full.token_ids[:, : self.seq_len])
            self._loaded_row_group = key
        assert self._batch is not None
        return self._batch

    def local_batch(self, step: int) -> HostTokenBatch:
        tokens = []
        for selection in _local_batch_rows(
            self.schedule, step, self.process_index, self.process_count
        ):
            group = self.schedule.shards[selection.file_idx].row_groups[selection.row_group_idx]
            batch = self._load_row_group(selection)
            tokens.append(batch.token_ids[selection.rows - group.start_row])
        return HostTokenBatch(np.concatenate(tokens))
