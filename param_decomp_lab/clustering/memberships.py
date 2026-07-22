"""Compressed membership collection, storage, and serialization.

ProcessedMemberships is the core data type: a sparse boolean membership matrix
(which components fire on which samples) with metadata.

MembershipBuilder streams activations into compressed memberships without
materializing the full dense [n_samples, n_components] matrix.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from scipy import sparse

from param_decomp_lab.clustering.formatting import (
    DeadComponentFilterStat,
    ModuleFilterFunc,
)
from param_decomp_lab.clustering.sample_membership import CompressedMembership
from param_decomp_lab.clustering.types import ComponentLabels


@dataclass(frozen=True)
class ProcessedMemberships:
    """Processed, compressed sample memberships for exact merge iteration."""

    module_component_counts: dict[str, int]
    module_alive_counts: dict[str, int]
    labels: ComponentLabels
    dead_components_lst: ComponentLabels | None
    memberships: list[CompressedMembership]
    n_samples: int

    @property
    def n_components_original(self) -> int:
        return sum(self.module_component_counts.values())

    @property
    def n_components_alive(self) -> int:
        return len(self.labels)

    @property
    def n_components_dead(self) -> int:
        return len(self.dead_components_lst) if self.dead_components_lst else 0

    def validate(self) -> None:
        assert self.n_components_alive == len(self.memberships), (
            f"{self.n_components_alive = } != {len(self.memberships) = }"
        )
        assert self.n_components_alive + self.n_components_dead == self.n_components_original, (
            f"{self.n_components_alive = } + {self.n_components_dead = } != {self.n_components_original = }"
        )

    def save(self, path: Path) -> None:
        from param_decomp_lab.clustering.sample_membership import (
            memberships_to_sample_component_matrix,
        )

        path.mkdir(parents=True, exist_ok=True)

        matrix = memberships_to_sample_component_matrix(self.memberships, fmt="csc")
        assert isinstance(matrix, sparse.csc_matrix)
        sparse.save_npz(path / "memberships.npz", matrix)

        metadata = {
            "n_samples": self.n_samples,
            "labels": list(self.labels),
            "dead_components_lst": list(self.dead_components_lst)
            if self.dead_components_lst
            else None,
            "module_component_counts": self.module_component_counts,
            "module_alive_counts": self.module_alive_counts,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ProcessedMemberships":
        metadata = json.loads((path / "metadata.json").read_text())
        labels = ComponentLabels(metadata["labels"])
        dead = (
            ComponentLabels(metadata["dead_components_lst"])
            if metadata["dead_components_lst"]
            else None
        )

        matrix_csc = sparse.load_npz(path / "memberships.npz").tocsc()
        assert matrix_csc.shape[0] == metadata["n_samples"]
        assert matrix_csc.shape[1] == len(labels)

        memberships: list[CompressedMembership] = []
        for col_idx in range(matrix_csc.shape[1]):
            sample_indices = matrix_csc.indices[
                matrix_csc.indptr[col_idx] : matrix_csc.indptr[col_idx + 1]
            ].astype(np.int64, copy=False)
            memberships.append(
                CompressedMembership.from_sample_indices(
                    sample_indices, n_samples=metadata["n_samples"]
                )
            )

        return cls(
            module_component_counts=metadata["module_component_counts"],
            module_alive_counts=metadata["module_alive_counts"],
            labels=labels,
            dead_components_lst=dead,
            memberships=memberships,
            n_samples=metadata["n_samples"],
        )


class MembershipBuilder:
    """Streaming builder for compressed sample memberships.

    Accumulates thresholded boolean memberships from batches without
    materializing the full dense [n_samples, n_components] matrix.
    """

    def __init__(
        self,
        *,
        activation_threshold: float,
        filter_dead_threshold: float,
        filter_dead_stat: DeadComponentFilterStat,
        filter_modules: ModuleFilterFunc | None,
    ) -> None:
        self.activation_threshold = activation_threshold
        self.filter_dead_threshold = filter_dead_threshold
        self.filter_dead_stat = filter_dead_stat
        self.filter_modules = filter_modules

        self.n_samples = 0
        self.module_component_counts: dict[str, int] = {}
        self.max_activations: dict[str, Float[np.ndarray, " c"]] = {}
        self.sum_activations: dict[str, Float[np.ndarray, " c"]] = {}
        self.module_sample_rows: dict[str, list[np.ndarray]] = {}
        self.module_sample_components: dict[str, list[np.ndarray]] = {}
        self.module_order: list[str] = []

    def _ensure_module(self, key: str, n_components: int) -> None:
        if key in self.module_component_counts:
            assert self.module_component_counts[key] == n_components, (
                f"Inconsistent component count for module '{key}': "
                f"{self.module_component_counts[key]} vs {n_components}"
            )
            return

        self.module_component_counts[key] = n_components
        self.max_activations[key] = np.full((n_components,), -np.inf, dtype=np.float32)
        self.sum_activations[key] = np.zeros((n_components,), dtype=np.float64)
        self.module_sample_rows[key] = []
        self.module_sample_components[key] = []
        self.module_order.append(key)

    def add_batch(self, activations: dict[str, Float[np.ndarray, "samples C"]]) -> None:
        filtered = (
            {key: act for key, act in activations.items() if self.filter_modules(key)}
            if self.filter_modules is not None
            else activations
        )
        if not filtered:
            return

        batch_n_samples = next(iter(filtered.values())).shape[0]
        sample_offset = self.n_samples

        for key, act in filtered.items():
            assert act.ndim == 2, f"Expected 2D activations, got shape {tuple(act.shape)}"
            self._ensure_module(key, act.shape[1])

            self.max_activations[key] = np.maximum(self.max_activations[key], act.max(axis=0))
            self.sum_activations[key] += act.sum(axis=0, dtype=np.float64)

            row_indices, comp_indices = np.nonzero(act > self.activation_threshold)
            if row_indices.size > 0:
                self.module_sample_rows[key].append(
                    row_indices.astype(np.int32, copy=False) + sample_offset
                )
                self.module_sample_components[key].append(comp_indices.astype(np.int32, copy=False))

        self.n_samples += batch_n_samples

    def finalize(self) -> ProcessedMemberships:
        module_alive_counts: dict[str, int] = {}
        alive_labels = ComponentLabels(list())
        dead_labels = ComponentLabels(list())
        memberships: list[CompressedMembership] = []

        for key in self.module_order:
            filter_values = (
                self.max_activations[key]
                if self.filter_dead_stat == "max"
                else (self.sum_activations[key] / self.n_samples).astype(
                    self.max_activations[key].dtype
                )
            )
            n_components = self.module_component_counts[key]
            alive = (
                filter_values >= self.filter_dead_threshold
                if self.filter_dead_threshold > 0
                else np.ones(n_components, dtype=np.bool_)
            )
            n_alive = int(alive.sum())
            module_alive_counts[key] = n_alive

            for comp_idx in range(n_components):
                if not alive[comp_idx]:
                    dead_labels.append(f"{key}:{comp_idx}")

            alive_component_indices = np.flatnonzero(alive).astype(np.int32, copy=False)
            for comp_idx in alive_component_indices:
                alive_labels.append(f"{key}:{int(comp_idx)}")

            row_chunks = self.module_sample_rows.pop(key)
            component_chunks = self.module_sample_components.pop(key)
            if n_alive == 0:
                continue

            if row_chunks:
                sample_rows = np.concatenate(row_chunks).astype(np.int64, copy=False)
                sample_components = np.concatenate(component_chunks).astype(np.int32, copy=False)
                alive_entries = alive[sample_components]
                if alive_entries.any():
                    alive_mapping = np.full(n_components, -1, dtype=np.int32)
                    alive_mapping[alive_component_indices] = np.arange(n_alive, dtype=np.int32)
                    csc = sparse.csc_matrix(
                        (
                            np.ones(int(alive_entries.sum()), dtype=np.uint8),
                            (
                                sample_rows[alive_entries],
                                alive_mapping[sample_components[alive_entries]],
                            ),
                        ),
                        shape=(self.n_samples, n_alive),
                        dtype=np.uint8,
                    )
                else:
                    csc = sparse.csc_matrix((self.n_samples, n_alive), dtype=np.uint8)
            else:
                csc = sparse.csc_matrix((self.n_samples, n_alive), dtype=np.uint8)

            for alive_idx in range(n_alive):
                sample_ids = csc.indices[csc.indptr[alive_idx] : csc.indptr[alive_idx + 1]]
                memberships.append(
                    CompressedMembership.from_sample_indices(
                        sample_indices=sample_ids, n_samples=self.n_samples
                    )
                )

        result = ProcessedMemberships(
            module_component_counts=self.module_component_counts,
            module_alive_counts=module_alive_counts,
            labels=alive_labels,
            dead_components_lst=dead_labels if dead_labels else None,
            memberships=memberships,
            n_samples=self.n_samples,
        )
        result.validate()
        return result


def _lm_sample_positions(
    *,
    batch_size: int,
    n_ctx: int,
    n_tokens_per_seq: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    positions = rng.integers(0, n_ctx, size=(batch_size, n_tokens_per_seq))
    batch_indices = np.broadcast_to(np.arange(batch_size)[:, None], positions.shape)
    return batch_indices, positions


def flatten_lm_activations(
    act: Float[np.ndarray, "batch n_ctx C"],
    *,
    batch_size: int,
    n_ctx: int,
    n_tokens_per_seq: int | None,
    use_all_tokens_per_seq: bool,
    rng: np.random.Generator,
) -> Float[np.ndarray, "samples C"]:
    if use_all_tokens_per_seq:
        return act.reshape(batch_size * n_ctx, -1)
    assert n_tokens_per_seq is not None
    batch_indices, positions = _lm_sample_positions(
        batch_size=batch_size,
        n_ctx=n_ctx,
        n_tokens_per_seq=n_tokens_per_seq,
        rng=rng,
    )
    return act[batch_indices, positions].reshape(batch_size * positions.shape[1], -1)
