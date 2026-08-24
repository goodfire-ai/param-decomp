"""Deterministic batches over local pre-tokenized Parquet shards (SPEC S18).

Each epoch shuffles equal-sized row groups within each shard, then visits one row group
from every shard before returning to any shard. Short tail groups come last so finite
runs remain balanced across shards. Rows shuffle within their row group. `locate(step)`
is directly addressable, so resume needs no replay. Incomplete row-group tails are
dropped to keep every global batch inside one independently readable row group.

Every process computes the same global schedule and serves its contiguous slice of
each batch; the trainer assembles the global device array.
"""

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _perm(seed_parts: tuple[int, ...], n: int) -> np.ndarray:
    return np.random.default_rng(seed_parts).permutation(n)


@dataclass(frozen=True)
class RowGroupInfo:
    start_row: int
    n_rows: int


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    row_groups: tuple[RowGroupInfo, ...]

    @property
    def n_rows(self) -> int:
        return sum(group.n_rows for group in self.row_groups)


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


@dataclass(frozen=True)
class BatchLocation:
    epoch: int
    file_idx: int
    row_group_idx: int
    batch_in_row_group: int


@dataclass(frozen=True)
class _EpochPlan:
    row_groups: tuple[tuple[int, int], ...]
    batch_ends: tuple[int, ...]


class BatchSchedule:
    """A directly addressable, row-group-striped global batch schedule."""

    def __init__(self, shards: tuple[ShardInfo, ...], global_batch: int, seed: int):
        assert shards
        assert global_batch > 0
        self.shards = shards
        self.global_batch = global_batch
        self.seed = seed
        self._batches_per_row_group = tuple(
            tuple(group.n_rows // global_batch for group in shard.row_groups) for shard in shards
        )
        assert all(
            any(n_batches > 0 for n_batches in shard) for shard in self._batches_per_row_group
        ), f"global_batch={global_batch} larger than every row group in some shard"
        self.steps_per_epoch = sum(sum(shard) for shard in self._batches_per_row_group)
        self._cached_plan: tuple[int, _EpochPlan] | None = None

    def _row_group_order(self, epoch: int, file_idx: int) -> np.ndarray:
        batches = self._batches_per_row_group[file_idx]
        eligible = np.flatnonzero(batches)
        shuffled = eligible[_perm((self.seed, epoch, 0xA0, file_idx), len(eligible))]
        return np.array(sorted(shuffled, key=lambda idx: batches[int(idx)], reverse=True))

    def _epoch_plan(self, epoch: int) -> _EpochPlan:
        if self._cached_plan is not None and self._cached_plan[0] == epoch:
            return self._cached_plan[1]
        row_group_orders = tuple(
            self._row_group_order(epoch, file_idx) for file_idx in range(len(self.shards))
        )
        row_groups = []
        batch_ends = []
        total_batches = 0
        for stripe in range(max(len(order) for order in row_group_orders)):
            active_files = np.array(
                [file_idx for file_idx, order in enumerate(row_group_orders) if stripe < len(order)]
            )
            file_order = active_files[_perm((self.seed, epoch, 0xD5, stripe), len(active_files))]
            for file_idx_value in file_order:
                file_idx = int(file_idx_value)
                row_group_idx = int(row_group_orders[file_idx][stripe])
                row_groups.append((file_idx, row_group_idx))
                total_batches += self._batches_per_row_group[file_idx][row_group_idx]
                batch_ends.append(total_batches)
        assert total_batches == self.steps_per_epoch
        plan = _EpochPlan(row_groups=tuple(row_groups), batch_ends=tuple(batch_ends))
        self._cached_plan = (epoch, plan)
        return plan

    def locate(self, step: int) -> BatchLocation:
        assert step >= 0
        epoch, pos = divmod(step, self.steps_per_epoch)
        plan = self._epoch_plan(epoch)
        plan_idx = bisect_right(plan.batch_ends, pos)
        previous_end = 0 if plan_idx == 0 else plan.batch_ends[plan_idx - 1]
        file_idx, row_group_idx = plan.row_groups[plan_idx]
        return BatchLocation(
            epoch=epoch,
            file_idx=file_idx,
            row_group_idx=row_group_idx,
            batch_in_row_group=pos - previous_end,
        )

    def row_perm(self, loc: BatchLocation) -> np.ndarray:
        group = self.shards[loc.file_idx].row_groups[loc.row_group_idx]
        local_rows = _perm(
            (self.seed, loc.epoch, 0xB0, loc.file_idx, loc.row_group_idx), group.n_rows
        )
        return group.start_row + local_rows

    def batch_rows(self, step: int) -> tuple[BatchLocation, np.ndarray]:
        """Return the location and absolute row indices within its shard."""
        loc = self.locate(step)
        start = loc.batch_in_row_group * self.global_batch
        return loc, self.row_perm(loc)[start : start + self.global_batch]


class ShardServer:
    """Load one scheduled row group and serve this process's global-batch slice."""

    def __init__(
        self,
        schedule: BatchSchedule,
        seq_len: int,
        process_index: int,
        process_count: int,
    ):
        assert schedule.global_batch % process_count == 0, (
            f"global_batch={schedule.global_batch} not divisible by {process_count} processes"
        )
        self.schedule = schedule
        self.seq_len = seq_len
        self.per_process = schedule.global_batch // process_count
        self.process_index = process_index
        self._loaded_row_group: tuple[int, int] | None = None
        self._tokens: np.ndarray | None = None

    def _load_row_group(self, loc: BatchLocation) -> np.ndarray:
        key = (loc.file_idx, loc.row_group_idx)
        if self._loaded_row_group != key:
            shard = self.schedule.shards[loc.file_idx]
            table = pq.ParquetFile(shard.path).read_row_group(
                loc.row_group_idx, columns=["input_ids"]
            )
            ids = table.column("input_ids")
            flat = ids.combine_chunks().flatten().to_numpy(zero_copy_only=False)
            group = shard.row_groups[loc.row_group_idx]
            tokens = flat.reshape(group.n_rows, -1)
            assert tokens.shape[1] in (self.seq_len, self.seq_len + 1), (
                f"{shard.path} rows have seq {tokens.shape[1]}, config says {self.seq_len}"
            )
            self._tokens = tokens[:, : self.seq_len]
            self._loaded_row_group = key
        assert self._tokens is not None
        return self._tokens

    def local_batch(self, step: int) -> np.ndarray:
        """This process's `[per_process, seq_len]` int32 slice of the step's batch."""
        loc, absolute_rows = self.schedule.batch_rows(step)
        group = self.schedule.shards[loc.file_idx].row_groups[loc.row_group_idx]
        local_rows = absolute_rows - group.start_row
        tokens = self._load_row_group(loc)
        lo = self.process_index * self.per_process
        return np.ascontiguousarray(tokens[local_rows[lo : lo + self.per_process]], dtype=np.int32)
