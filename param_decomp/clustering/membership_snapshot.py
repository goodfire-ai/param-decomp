import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from param_decomp.clustering.consts import ComponentLabels
from param_decomp.clustering.sample_membership import (
    CompressedMembership,
    memberships_to_sample_component_matrix,
)


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
    matrix = memberships_to_sample_component_matrix(memberships, fmt="csc")
    assert isinstance(matrix, sparse.csc_matrix)
    shape = matrix.shape
    assert shape is not None
    assert shape[0] == n_samples
    return matrix


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
