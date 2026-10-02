"""Deterministic row selection from dataset metadata, without reading dataset contents."""

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


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


@dataclass(frozen=True)
class RowGroupSelection:
    """Absolute shard row indices to read together from one physical row group."""

    file_idx: int
    row_group_idx: int
    rows: NDArray[np.int64]


type BatchRowIndices = tuple[RowGroupSelection, ...]


class BatchRowSchedule(Protocol):
    @property
    def shards(self) -> tuple[ShardInfo, ...]: ...

    @property
    def global_batch(self) -> int: ...

    def batch_rows(self, step: int) -> BatchRowIndices: ...


@dataclass(frozen=True)
class _EpochPlan:
    row_groups: tuple[tuple[int, int], ...]
    row_ends: tuple[int, ...]


def _perm(seed_parts: tuple[int, ...], n: int) -> NDArray[np.int64]:
    return np.random.default_rng(seed_parts).permutation(n)


class BatchSchedule:
    """Slice fixed-size batches from a seeded, row-group-striped stream of rows.

    A step determines its rows independently of lookup history. Only the epoch's
    group ordering is cached; there is no cursor or partially assembled batch.
    """

    def __init__(self, shards: tuple[ShardInfo, ...], global_batch: int, seed: int):
        assert shards and all(shard.n_rows > 0 for shard in shards)
        assert global_batch > 0
        self.shards = shards
        self.global_batch = global_batch
        self.seed = seed
        self.steps_per_epoch = sum(shard.n_rows for shard in shards) // global_batch
        assert self.steps_per_epoch > 0, f"global_batch={global_batch} larger than the corpus"
        self._cached_plan: tuple[int, _EpochPlan] | None = None

    def _row_group_order(self, epoch: int, file_idx: int) -> NDArray[np.int64]:
        rows = tuple(group.n_rows for group in self.shards[file_idx].row_groups)
        eligible = np.flatnonzero(rows)
        shuffled = eligible[_perm((self.seed, epoch, 0xA0, file_idx), len(eligible))]
        return np.array(sorted(shuffled, key=lambda idx: rows[int(idx)], reverse=True))

    def _epoch_plan(self, epoch: int) -> _EpochPlan:
        if self._cached_plan is not None and self._cached_plan[0] == epoch:
            return self._cached_plan[1]
        row_group_orders = tuple(
            self._row_group_order(epoch, file_idx) for file_idx in range(len(self.shards))
        )
        row_groups = []
        row_ends = []
        total_rows = 0
        for stripe in range(max(len(order) for order in row_group_orders)):
            active_files = np.array(
                [file_idx for file_idx, order in enumerate(row_group_orders) if stripe < len(order)]
            )
            file_order = active_files[_perm((self.seed, epoch, 0xD5, stripe), len(active_files))]
            for file_idx_value in file_order:
                file_idx = int(file_idx_value)
                row_group_idx = int(row_group_orders[file_idx][stripe])
                row_groups.append((file_idx, row_group_idx))
                total_rows += self.shards[file_idx].row_groups[row_group_idx].n_rows
                row_ends.append(total_rows)
        assert total_rows // self.global_batch == self.steps_per_epoch
        plan = _EpochPlan(row_groups=tuple(row_groups), row_ends=tuple(row_ends))
        self._cached_plan = (epoch, plan)
        return plan

    def batch_rows(self, step: int) -> BatchRowIndices:
        assert step >= 0
        epoch, batch_in_epoch = divmod(step, self.steps_per_epoch)
        plan = self._epoch_plan(epoch)
        start = batch_in_epoch * self.global_batch
        stop = start + self.global_batch
        plan_idx = bisect_right(plan.row_ends, start)
        selections = []
        while start < stop:
            group_start = 0 if plan_idx == 0 else plan.row_ends[plan_idx - 1]
            end = min(stop, plan.row_ends[plan_idx])
            file_idx, row_group_idx = plan.row_groups[plan_idx]
            group = self.shards[file_idx].row_groups[row_group_idx]
            rows = _perm((self.seed, epoch, 0xB0, file_idx, row_group_idx), group.n_rows)
            selections.append(
                RowGroupSelection(
                    file_idx,
                    row_group_idx,
                    group.start_row + rows[start - group_start : end - group_start],
                )
            )
            start = end
            plan_idx += 1
        return tuple(selections)


def slice_batch_rows(rows: BatchRowIndices, start: int, stop: int) -> BatchRowIndices:
    """Take an interval in batch order, preserving physical read boundaries."""
    assert 0 <= start <= stop
    offset = 0
    selected = []
    for selection in rows:
        end = offset + len(selection.rows)
        lo, hi = max(start, offset), min(stop, end)
        if lo < hi:
            selected.append(
                RowGroupSelection(
                    selection.file_idx,
                    selection.row_group_idx,
                    selection.rows[lo - offset : hi - offset],
                )
            )
        offset = end
    assert stop <= offset, (stop, offset)
    return tuple(selected)
