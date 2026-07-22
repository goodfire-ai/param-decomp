from dataclasses import dataclass

import numpy as np
from jaxtyping import Int

from param_decomp_lab.clustering.types import GroupIdxsArray


def _array_summary(arr: np.ndarray) -> str:
    return f"shape={tuple(arr.shape)}, dtype={arr.dtype}"


@dataclass(kw_only=True, slots=True)
class GroupMerge:
    """Canonical component-to-group assignment.

    `group_idxs` is a length-`n_components` integer array; entry `c`
    gives the group index (0 to `k_groups-1`) that contains component `c`.
    """

    group_idxs: GroupIdxsArray
    k_groups: int
    old_to_new_idx: dict[int | None, int | None] | None = None

    def summary(self) -> dict[str, int | str | None]:
        return dict(
            group_idxs=_array_summary(self.group_idxs),
            k_groups=self.k_groups,
            old_to_new_idx=f"len={len(self.old_to_new_idx)}"
            if self.old_to_new_idx is not None
            else None,
        )

    @property
    def _n_components(self) -> int:
        return int(self.group_idxs.shape[0])

    @property
    def components_per_group(self) -> Int[np.ndarray, " k_groups"]:
        return np.bincount(self.group_idxs, minlength=self.k_groups)

    def components_in_group(self, group_idx: int) -> list[int]:
        return np.flatnonzero(self.group_idxs == group_idx).tolist()

    def validate(self, *, require_nonempty: bool = True) -> None:
        v_min: int = int(self.group_idxs.min())
        v_max: int = int(self.group_idxs.max())
        if v_min < 0 or v_max >= self.k_groups:
            raise ValueError("group indices out of range")

        if require_nonempty:
            has_empty_groups: bool = bool((self.components_per_group == 0).any())
            if has_empty_groups:
                raise ValueError("one or more groups are empty")

    @classmethod
    def identity(cls, n_components: int) -> "GroupMerge":
        """Each component is its own group."""
        return cls(
            group_idxs=np.arange(n_components, dtype=np.int64),
            k_groups=n_components,
        )

    def merge_groups(self, group_a: int, group_b: int) -> "GroupMerge":
        if group_a < 0 or group_b < 0 or group_a >= self.k_groups or group_b >= self.k_groups:
            raise ValueError("group indices out of range")
        if group_a == group_b:
            raise ValueError("Cannot merge a group with itself")

        if group_a > group_b:
            group_a, group_b = group_b, group_a

        new_idxs: GroupIdxsArray = self.group_idxs.copy()
        new_idxs[new_idxs == group_b] = group_a
        new_idxs[new_idxs > group_b] -= 1

        old_to_new_idx: dict[int | None, int | None] = dict()
        for i in range(self.k_groups):
            if i in {group_a, group_b}:
                old_to_new_idx[i] = None
            elif i <= group_b:
                old_to_new_idx[i] = i
            else:
                old_to_new_idx[i] = i - 1
        old_to_new_idx[None] = group_a  # the new group index for the merged group

        return GroupMerge(
            group_idxs=new_idxs,
            k_groups=self.k_groups - 1,
            old_to_new_idx=old_to_new_idx,
        )


@dataclass(slots=True)
class BatchedGroupMerge:
    """Batch of merge matrices.

    `group_idxs` has shape `(batch, n_components)`; each row holds the
    group index for every component in that matrix.
    """

    group_idxs: Int[np.ndarray, "batch n_components"]
    k_groups: Int[np.ndarray, " batch"]

    def summary(self) -> dict[str, int | str | None]:
        return dict(
            group_idxs=_array_summary(self.group_idxs),
            k_groups=_array_summary(self.k_groups),
        )

    @classmethod
    def init_empty(cls, batch_size: int, n_components: int) -> "BatchedGroupMerge":
        return cls(
            group_idxs=np.full((batch_size, n_components), -1, dtype=np.int32),
            k_groups=np.zeros(batch_size, dtype=np.int32),
        )

    @property
    def _batch_size(self) -> int:
        return int(self.group_idxs.shape[0])

    @property
    def _n_components(self) -> int:
        return int(self.group_idxs.shape[1])

    def __getitem__(self, idx: int) -> GroupMerge:
        if not (0 <= idx < self._batch_size):
            raise IndexError("index out of range")
        return GroupMerge(group_idxs=self.group_idxs[idx], k_groups=int(self.k_groups[idx]))

    def __setitem__(self, idx: int, value: GroupMerge) -> None:
        if not (0 <= idx < self._batch_size):
            raise IndexError("index out of range")
        if value._n_components != self._n_components:
            raise ValueError("value must have the same number of components as the batch")
        self.group_idxs[idx] = value.group_idxs
        self.k_groups[idx] = value.k_groups

    def __iter__(self):
        for i in range(self._batch_size):
            yield self[i]

    def __len__(self) -> int:
        return self._batch_size
