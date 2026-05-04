from dataclasses import dataclass

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from param_decomp.clustering.consts import GroupIdxsTensor


def _array_summary(arr: Tensor) -> str:
    """Get a brief summary string for a tensor."""
    return f"shape={tuple(arr.shape)}, dtype={arr.dtype}"


# pyright: reportUnnecessaryTypeIgnoreComment=false


@dataclass(kw_only=True, slots=True)
class GroupMerge:
    """Canonical component-to-group assignment.

    `group_idxs` is a length-`n_components` integer tensor; entry `c`
    gives the group index (0 to `k_groups-1`) that contains component `c`.
    """

    group_idxs: GroupIdxsTensor
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
    def components_per_group(self) -> Int[Tensor, " k_groups"]:
        return torch.bincount(self.group_idxs, minlength=self.k_groups)

    def components_in_group_mask(self, group_idx: int) -> Bool[Tensor, " n_components"]:
        """Returns a boolean mask for components in the specified group."""
        if group_idx < 0 or group_idx >= self.k_groups:
            raise ValueError("group index out of range")
        return self.group_idxs == group_idx

    def components_in_group(self, group_idx: int) -> list[int]:
        """Returns a list of component indices in the specified group."""
        indices: Int[Tensor, " n_matches"] = (
            (self.group_idxs == group_idx).nonzero(as_tuple=False).squeeze(-1)
        )
        return indices.tolist()

    def validate(self, *, require_nonempty: bool = True) -> None:
        v_min: int = int(self.group_idxs.min().item())
        v_max: int = int(self.group_idxs.max().item())
        if v_min < 0 or v_max >= self.k_groups:
            raise ValueError("group indices out of range")

        if require_nonempty:
            has_empty_groups: bool = bool(self.components_per_group.eq(0).any().item())
            if has_empty_groups:
                raise ValueError("one or more groups are empty")

    def to_matrix(
        self, device: torch.device | None = None
    ) -> Bool[Tensor, "k_groups n_components"]:
        if device is None:
            device = self.group_idxs.device
        mat: Bool[Tensor, "k_groups n_components"] = torch.zeros(
            (self.k_groups, self._n_components), dtype=torch.bool, device=device
        )
        idxs: Int[Tensor, " n_components"] = torch.arange(
            self._n_components, device=device, dtype=torch.int
        )
        mat[self.group_idxs.to(dtype=torch.int), idxs] = True
        return mat

    @classmethod
    def from_matrix(cls, mat: Bool[Tensor, "k_groups n_components"]) -> "GroupMerge":
        if mat.dtype is not torch.bool:
            raise TypeError("mat must have dtype bool")
        if not mat.sum(dim=0).eq(1).all():
            raise ValueError("each column must contain exactly one True")
        group_idxs: GroupIdxsTensor = mat.argmax(dim=0).to(torch.int64)
        inst: GroupMerge = cls(group_idxs=group_idxs, k_groups=int(mat.shape[0]))
        inst.validate(require_nonempty=False)
        return inst

    @classmethod
    def random(
        cls,
        n_components: int,
        k_groups: int,
        *,
        ensure_groups_nonempty: bool = False,
        device: torch.device | str = "cpu",
    ) -> "GroupMerge":
        if ensure_groups_nonempty and n_components < k_groups:
            raise ValueError("n_components must be >= k_groups when ensure_groups_nonempty is True")

        group_idxs: GroupIdxsTensor

        if ensure_groups_nonempty:
            base: Int[Tensor, " k_groups"] = torch.arange(k_groups, device=device)
            if n_components > k_groups:
                extra: Int[Tensor, " n_extra"] = torch.randint(
                    0, k_groups, (n_components - k_groups,), device=device
                )
                group_idxs = torch.cat((base, extra))
                group_idxs = group_idxs[torch.randperm(n_components, device=device)]
            else:
                group_idxs = base
        else:
            group_idxs = torch.randint(0, k_groups, (n_components,), device=device)
        inst: GroupMerge = cls(group_idxs=group_idxs, k_groups=k_groups)
        inst.validate(require_nonempty=ensure_groups_nonempty)
        return inst

    @classmethod
    def identity(cls, n_components: int) -> "GroupMerge":
        """Creates a GroupMerge where each component is its own group."""
        return cls(
            group_idxs=torch.arange(n_components, dtype=torch.int64),
            k_groups=n_components,
        )

    def merge_groups(self, group_a: int, group_b: int) -> "GroupMerge":
        """Merges two groups into one, returning a new GroupMerge."""
        if group_a < 0 or group_b < 0 or group_a >= self.k_groups or group_b >= self.k_groups:
            raise ValueError("group indices out of range")
        if group_a == group_b:
            raise ValueError("Cannot merge a group with itself")

        # make sure group_a is the smaller index
        if group_a > group_b:
            group_a, group_b = group_b, group_a

        # make a copy
        new_idxs: GroupIdxsTensor = self.group_idxs.clone()
        # wherever its currently b, change it to a
        new_idxs[new_idxs == group_b] = group_a
        # wherever i currently above b, change it to i-1
        new_idxs[new_idxs > group_b] -= 1
        # create a new GroupMerge instance
        merged: GroupMerge = GroupMerge(group_idxs=new_idxs, k_groups=self.k_groups - 1)

        # create a mapping from old to new group indices
        # `None` as a key is for the new group that contains both a and b
        # values of a and b are mapped to `None` since they are merged
        old_to_new_idx: dict[int | None, int | None] = dict()
        for i in range(self.k_groups):
            if i in {group_a, group_b}:
                old_to_new_idx[i] = None
            elif i <= group_b:
                old_to_new_idx[i] = i
            else:
                old_to_new_idx[i] = i - 1
        old_to_new_idx[None] = group_a  # the new group index for the merged group

        # HACK: store the mapping in the instance for later use
        merged.old_to_new_idx = old_to_new_idx  # type: ignore[assignment]

        # validate the new instance
        # merged.validate(require_nonempty=True)
        return merged

    def all_downstream_merged(self) -> "BatchedGroupMerge":
        downstream: list[GroupMerge] = []
        idxs: list[tuple[int, int]] = []
        for i in range(self.k_groups):
            for j in range(i + 1, self.k_groups):
                downstream.append(self.merge_groups(i, j))
                idxs.append((i, j))

        return BatchedGroupMerge.from_list(merge_matrices=downstream)


@dataclass(slots=True)
class BatchedGroupMerge:
    """Batch of merge matrices.

    `group_idxs` has shape `(batch, n_components)`; each row holds the
    group index for every component in that matrix.
    """

    group_idxs: Int[Tensor, "batch n_components"]
    k_groups: Int[Tensor, " batch"]

    def summary(self) -> dict[str, int | str | None]:
        return dict(
            group_idxs=_array_summary(self.group_idxs),
            k_groups=_array_summary(self.k_groups),
            # TODO: re-add metadata (which pairs merged at each step)
            # meta=f"len={len(self.meta)}" if self.meta is not None else None,
        )

    @classmethod
    def init_empty(cls, batch_size: int, n_components: int) -> "BatchedGroupMerge":
        """Initialize an empty BatchedGroupMerge with the given batch size and number of components."""
        return cls(
            group_idxs=torch.full((batch_size, n_components), -1, dtype=torch.int16),
            k_groups=torch.zeros(batch_size, dtype=torch.int16),
        )

    @property
    def _batch_size(self) -> int:
        return int(self.group_idxs.shape[0])

    @property
    def _n_components(self) -> int:
        return int(self.group_idxs.shape[1])

    @property
    def k_groups_unique(self) -> int:
        """Returns the number of groups across all matrices, throws exception if they differ."""
        k_groups_set: set[int] = set(self.k_groups.tolist())
        if len(k_groups_set) != 1:
            raise ValueError("All matrices must have the same number of groups")
        return k_groups_set.pop()

    def to_matrix(
        self, device: torch.device | None = None
    ) -> Bool[Tensor, "batch k_groups n_components"]:
        if device is None:
            device = self.group_idxs.device
        k_groups_u: int = self.k_groups_unique
        mat = torch.nn.functional.one_hot(self.group_idxs, num_classes=k_groups_u)
        return mat.permute(0, 2, 1).to(device=device, dtype=torch.bool)

    @classmethod
    def from_matrix(cls, mat: Bool[Tensor, "batch k_groups n_components"]) -> "BatchedGroupMerge":
        if mat.dtype is not torch.bool:
            raise TypeError("mat must have dtype bool")
        if not mat.sum(dim=1).eq(1).all():
            raise ValueError("each column must have exactly one True per matrix")
        group_idxs = mat.argmax(dim=1).to(torch.int64)
        batch_size: int = int(mat.shape[0])
        inst = cls(
            group_idxs=group_idxs,
            k_groups=torch.full((batch_size,), int(mat.shape[1]), dtype=torch.int64),
        )
        # inst.validate(require_nonempty=False)
        return inst

    @classmethod
    def from_list(
        cls,
        merge_matrices: list[GroupMerge],
    ) -> "BatchedGroupMerge":
        group_idxs: Int[Tensor, "batch n_components"] = torch.stack(
            [mm.group_idxs for mm in merge_matrices], dim=0
        )
        k_groups: Int[Tensor, " batch"] = torch.tensor(
            [mm.k_groups for mm in merge_matrices], dtype=torch.int64
        )
        inst: BatchedGroupMerge = cls(group_idxs=group_idxs, k_groups=k_groups)
        # inst.validate(require_nonempty=False)
        return inst

    def __getitem__(self, idx: int) -> GroupMerge:
        if not (0 <= idx < self._batch_size):
            raise IndexError("index out of range")
        group_idxs: GroupIdxsTensor = self.group_idxs[idx]
        k_groups: int = int(self.k_groups[idx].item())
        return GroupMerge(group_idxs=group_idxs, k_groups=k_groups)

    def __setitem__(self, idx: int, value: GroupMerge) -> None:
        if not (0 <= idx < self._batch_size):
            raise IndexError("index out of range")
        if value._n_components != self._n_components:
            raise ValueError("value must have the same number of components as the batch")
        self.group_idxs[idx] = value.group_idxs
        self.k_groups[idx] = value.k_groups

    def __iter__(self):
        """Iterate over the GroupMerge instances in the batch."""
        for i in range(self._batch_size):
            yield self[i]

    def __len__(self) -> int:
        return self._batch_size
