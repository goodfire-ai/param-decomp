import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from spd.clustering.consts import ComponentLabels
from spd.clustering.sample_membership import CompressedMembership


@dataclass(frozen=True, slots=True)
class MembershipSnapshot:
    """Disk-friendly sparse membership snapshot for repeatable merge benchmarks."""

    matrix_csc: sparse.csc_matrix
    labels: ComponentLabels

    @property
    def n_samples(self) -> int:
        shape = self.matrix_csc.shape
        assert shape is not None
        return int(shape[0])

    @property
    def n_components(self) -> int:
        shape = self.matrix_csc.shape
        assert shape is not None
        return int(shape[1])

    def to_memberships(self) -> list[CompressedMembership]:
        memberships: list[CompressedMembership] = []
        for col_idx in range(self.n_components):
            sample_indices = self.matrix_csc.indices[
                self.matrix_csc.indptr[col_idx] : self.matrix_csc.indptr[col_idx + 1]
            ].astype(np.int64, copy=False)
            memberships.append(
                CompressedMembership.from_sample_indices(
                    sample_indices=sample_indices,
                    n_samples=self.n_samples,
                )
            )
        return memberships

    def to_csr(self) -> sparse.sparray | sparse.spmatrix:
        return self.matrix_csc.tocsr()


def memberships_to_csc(
    memberships: list[CompressedMembership],
    n_samples: int,
) -> sparse.csc_matrix:
    n_components = len(memberships)
    if n_components == 0:
        return sparse.csc_matrix((n_samples, 0), dtype=np.uint8)

    nnz = sum(membership.count() for membership in memberships)
    row_indices = np.empty(nnz, dtype=np.int64)
    col_indices = np.empty(nnz, dtype=np.int32)

    offset = 0
    for col_idx, membership in enumerate(memberships):
        sample_indices = membership.to_sample_indices().astype(np.int64, copy=False)
        col_nnz = sample_indices.size
        row_indices[offset : offset + col_nnz] = sample_indices
        col_indices[offset : offset + col_nnz] = col_idx
        offset += col_nnz

    values = np.ones(nnz, dtype=np.uint8)
    return sparse.csc_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_samples, n_components),
        dtype=np.uint8,
    )


def save_membership_snapshot(
    output_dir: Path,
    *,
    memberships: list[CompressedMembership],
    labels: ComponentLabels,
    n_samples: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "memberships.npz"
    metadata_path = output_dir / "metadata.json"

    matrix_csc = memberships_to_csc(memberships, n_samples=n_samples)
    sparse.save_npz(matrix_path, matrix_csc)
    metadata_path.write_text(
        json.dumps(
            {
                "n_samples": n_samples,
                "n_components": len(labels),
                "labels": list(labels),
            },
            indent=2,
        )
    )
    return output_dir


def load_membership_snapshot(path: Path) -> MembershipSnapshot:
    matrix_path = path / "memberships.npz"
    metadata_path = path / "metadata.json"
    matrix_csc = sparse.load_npz(matrix_path).tocsc()
    metadata = json.loads(metadata_path.read_text())
    labels = ComponentLabels(metadata["labels"])
    assert matrix_csc.shape[0] == metadata["n_samples"]
    assert matrix_csc.shape[1] == metadata["n_components"]
    return MembershipSnapshot(matrix_csc=matrix_csc, labels=labels)
