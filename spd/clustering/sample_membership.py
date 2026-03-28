from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numba import njit
from scipy import sparse

from spd.clustering.consts import ClusterCoactivationShaped

_POPCOUNT_TABLE = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def _index_dtype_for(n_samples: int) -> np.dtype[np.unsignedinteger]:
    return np.dtype(np.uint32) if n_samples <= np.iinfo(np.uint32).max else np.dtype(np.uint64)


def _bytes_for_bitset(n_samples: int) -> int:
    return (n_samples + 7) // 8


def _prefer_sparse(n_samples: int, active_count: int, dtype: np.dtype[np.generic]) -> bool:
    return active_count * np.dtype(dtype).itemsize <= _bytes_for_bitset(n_samples)


def _sample_indices_to_bits(
    sample_indices: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    n_bytes = _bytes_for_bitset(n_samples)
    bits = np.zeros(n_bytes, dtype=np.uint8)
    if sample_indices.size == 0:
        return bits

    sample_indices = sample_indices.astype(np.int64, copy=False)
    byte_indices = sample_indices // 8
    bit_offsets = sample_indices % 8
    np.bitwise_or.at(bits, byte_indices, (1 << bit_offsets).astype(np.uint8))
    return bits


def _count_sparse_sparse_intersection(a: np.ndarray, b: np.ndarray) -> int:
    i = 0
    j = 0
    count = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        a_i = int(a[i])
        b_j = int(b[j])
        if a_i == b_j:
            count += 1
            i += 1
            j += 1
        elif a_i < b_j:
            i += 1
        else:
            j += 1
    return count


def _union_sparse_sparse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    i = 0
    j = 0
    out: list[int] = []
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        a_i = int(a[i])
        b_j = int(b[j])
        if a_i == b_j:
            out.append(a_i)
            i += 1
            j += 1
        elif a_i < b_j:
            out.append(a_i)
            i += 1
        else:
            out.append(b_j)
            j += 1
    while i < len_a:
        out.append(int(a[i]))
        i += 1
    while j < len_b:
        out.append(int(b[j]))
        j += 1
    dtype = a.dtype if a.size > 0 else b.dtype
    return np.asarray(out, dtype=dtype)


def _count_sparse_bitset_intersection(
    sample_indices: np.ndarray,
    bits: np.ndarray,
) -> int:
    if sample_indices.size == 0:
        return 0
    sample_indices = sample_indices.astype(np.int64, copy=False)
    byte_indices = sample_indices // 8
    bit_offsets = sample_indices % 8
    return int(np.count_nonzero(bits[byte_indices] & ((1 << bit_offsets).astype(np.uint8))))


def _bitset_to_sample_indices(bits: np.ndarray, n_samples: int) -> np.ndarray:
    nonzero_byte_idxs = np.flatnonzero(bits)
    if nonzero_byte_idxs.size == 0:
        return np.empty((0,), dtype=_index_dtype_for(n_samples))

    unpacked = np.unpackbits(bits[nonzero_byte_idxs], bitorder="little")
    active_bit_positions = np.flatnonzero(unpacked)
    sample_indices = nonzero_byte_idxs[active_bit_positions // 8].astype(
        np.int64, copy=False
    ) * 8 + (active_bit_positions % 8)
    sample_indices = sample_indices[sample_indices < n_samples]
    return sample_indices.astype(_index_dtype_for(n_samples), copy=False)


@njit(cache=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def _count_group_overlaps_rows_numba(
    merged_rows: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    group_idxs: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    counts = np.zeros(n_groups, dtype=np.int64)
    seen = np.full(n_groups, -1, dtype=np.int64)
    stamp = 0
    for row in merged_rows:
        stamp += 1
        start = indptr[row]
        end = indptr[row + 1]
        for pos in range(start, end):
            group_idx = group_idxs[indices[pos]]
            if seen[group_idx] == stamp:
                continue
            seen[group_idx] = stamp
            counts[group_idx] += 1
    return counts


@dataclass(frozen=True, slots=True)
class CompressedMembership:
    """Exact sample memberships stored sparsely when cheaper, otherwise as a bitset."""

    n_samples: int
    active_count: int
    sample_indices: np.ndarray | None = None
    bits: np.ndarray | None = None

    def __post_init__(self) -> None:
        has_sparse = self.sample_indices is not None
        has_bits = self.bits is not None
        assert has_sparse != has_bits, "Membership must use exactly one representation"

    @classmethod
    def empty(cls, n_samples: int) -> "CompressedMembership":
        dtype = _index_dtype_for(n_samples)
        return cls(n_samples=n_samples, active_count=0, sample_indices=np.empty((0,), dtype=dtype))

    @classmethod
    def from_sample_indices(
        cls,
        sample_indices: np.ndarray,
        n_samples: int,
    ) -> "CompressedMembership":
        dtype = _index_dtype_for(n_samples)
        sample_indices = sample_indices.astype(dtype, copy=False)
        active_count = int(sample_indices.size)
        if _prefer_sparse(n_samples, active_count, dtype):
            return cls(
                n_samples=n_samples, active_count=active_count, sample_indices=sample_indices
            )
        return cls(
            n_samples=n_samples,
            active_count=active_count,
            bits=_sample_indices_to_bits(sample_indices, n_samples),
        )

    @classmethod
    def from_bits(
        cls,
        bits: np.ndarray,
        n_samples: int,
        active_count: int,
    ) -> "CompressedMembership":
        return cls(
            n_samples=n_samples,
            active_count=active_count,
            bits=bits.astype(np.uint8, copy=False),
        )

    @property
    def is_sparse(self) -> bool:
        return self.sample_indices is not None

    def count(self) -> int:
        return self.active_count

    def intersection_count(self, other: "CompressedMembership") -> int:
        assert self.n_samples == other.n_samples, "Memberships must share sample space"
        if self.sample_indices is not None and other.sample_indices is not None:
            return _count_sparse_sparse_intersection(self.sample_indices, other.sample_indices)
        if self.sample_indices is not None:
            assert other.bits is not None
            return _count_sparse_bitset_intersection(self.sample_indices, other.bits)
        if other.sample_indices is not None:
            assert self.bits is not None
            return _count_sparse_bitset_intersection(other.sample_indices, self.bits)

        assert self.bits is not None and other.bits is not None
        overlap = np.bitwise_and(self.bits, other.bits)
        return int(_POPCOUNT_TABLE[overlap].sum(dtype=np.uint64))

    def union(self, other: "CompressedMembership") -> "CompressedMembership":
        assert self.n_samples == other.n_samples, "Memberships must share sample space"
        overlap_count = self.intersection_count(other)
        union_count = self.active_count + other.active_count - overlap_count

        if self.sample_indices is not None and other.sample_indices is not None:
            union_indices = _union_sparse_sparse(self.sample_indices, other.sample_indices)
            return CompressedMembership.from_sample_indices(union_indices, self.n_samples)

        if self.bits is not None and other.bits is not None:
            return CompressedMembership.from_bits(
                bits=np.bitwise_or(self.bits, other.bits),
                n_samples=self.n_samples,
                active_count=union_count,
            )

        if self.bits is not None:
            base_bits = self.bits.copy()
            sparse_indices = other.sample_indices
        else:
            assert other.bits is not None
            base_bits = other.bits.copy()
            sparse_indices = self.sample_indices

        assert sparse_indices is not None
        if sparse_indices.size > 0:
            sparse_indices_i64 = sparse_indices.astype(np.int64, copy=False)
            byte_indices = sparse_indices_i64 // 8
            bit_offsets = sparse_indices_i64 % 8
            np.bitwise_or.at(base_bits, byte_indices, (1 << bit_offsets).astype(np.uint8))

        return CompressedMembership.from_bits(
            bits=base_bits,
            n_samples=self.n_samples,
            active_count=union_count,
        )

    def to_bool_array(self) -> np.ndarray:
        if self.sample_indices is not None:
            result = np.zeros(self.n_samples, dtype=bool)
            result[self.sample_indices.astype(np.int64, copy=False)] = True
            return result

        assert self.bits is not None
        unpacked = np.unpackbits(self.bits, bitorder="little")
        return unpacked[: self.n_samples].astype(bool, copy=False)

    def to_sample_indices(self) -> np.ndarray:
        if self.sample_indices is not None:
            return self.sample_indices

        assert self.bits is not None
        return _bitset_to_sample_indices(self.bits, self.n_samples)


def memberships_to_sample_component_matrix(
    memberships: list[CompressedMembership],
    *,
    fmt: Literal["csr", "csc"] = "csr",
) -> sparse.csr_matrix | sparse.csc_matrix:
    """Build a binary sample-by-component sparse matrix from memberships."""
    n_groups = len(memberships)
    if n_groups == 0:
        empty = sparse.csr_matrix((0, 0), dtype=np.uint8)
        return empty if fmt == "csr" else sparse.csc_matrix(empty)

    n_samples = memberships[0].n_samples
    assert all(membership.n_samples == n_samples for membership in memberships), (
        "Memberships must share sample space"
    )

    nnz = sum(membership.count() for membership in memberships)
    row_indices = np.empty(nnz, dtype=np.int64)
    col_indices = np.empty(nnz, dtype=np.int32)

    offset = 0
    for group_idx, membership in enumerate(memberships):
        sample_indices = membership.to_sample_indices().astype(np.int64, copy=False)
        group_nnz = sample_indices.size
        row_indices[offset : offset + group_nnz] = sample_indices
        col_indices[offset : offset + group_nnz] = group_idx
        offset += group_nnz

    values = np.ones(nnz, dtype=np.uint8)
    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_samples, n_groups),
        dtype=np.uint8,
    )
    return matrix if fmt == "csr" else sparse.csc_matrix(matrix)


def memberships_to_sample_component_csr(
    memberships: list[CompressedMembership],
) -> sparse.csr_matrix:
    """Build a binary sample-by-component CSR matrix from memberships."""
    matrix = memberships_to_sample_component_matrix(memberships, fmt="csr")
    assert isinstance(matrix, sparse.csr_matrix)
    return matrix


def count_group_overlaps_from_component_rows(
    merged_rows: np.ndarray,
    component_activity_csr: sparse.csr_matrix,
    group_idxs: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Count exact merged-group overlaps by scanning original component rows.

    `group_idxs` maps original component indices to the current group index after
    the candidate merge has been applied.
    """
    merged_rows_i64 = merged_rows.astype(np.int64, copy=False)
    indptr = component_activity_csr.indptr.astype(np.int64, copy=False)
    indices = component_activity_csr.indices.astype(np.int32, copy=False)
    group_idxs_i64 = group_idxs.astype(np.int64, copy=False)
    return _count_group_overlaps_rows_numba(
        merged_rows=merged_rows_i64,
        indptr=indptr,
        indices=indices,
        group_idxs=group_idxs_i64,
        n_groups=n_groups,
    )


def compute_coactivation_matrix_from_csr(
    component_activity_csr: sparse.csr_matrix,
) -> ClusterCoactivationShaped:
    """Compute the full coactivation matrix from a sample-by-component CSR matrix."""
    activation_matrix = component_activity_csr.astype(np.int32, copy=False)
    coact = (activation_matrix.T @ activation_matrix).toarray()
    return torch.from_numpy(coact.astype(np.float32, copy=False))


def compute_coactivation_matrix(
    memberships: list[CompressedMembership],
) -> ClusterCoactivationShaped:
    """Compute the full coactivation matrix from compressed memberships.

    This builds a sparse sample-by-component matrix and computes X.T @ X,
    which is much faster than Python-level pairwise intersections in the
    typical highly sparse regime.
    """
    n_groups = len(memberships)
    if n_groups == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    n_samples = memberships[0].n_samples
    assert all(membership.n_samples == n_samples for membership in memberships), (
        "Memberships must share sample space"
    )

    return compute_coactivation_matrix_from_csr(memberships_to_sample_component_csr(memberships))
