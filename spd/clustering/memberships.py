"""Compressed membership collection, storage, and serialization.

ProcessedMemberships is the core data type: a sparse boolean membership matrix
(which components fire on which samples) with metadata and an optional dense preview.

MembershipBuilder streams activations into compressed memberships without
materializing the full dense [n_samples, n_components] matrix.

collect_memberships() dispatches LM vs ResidMLP collection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from scipy import sparse
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.clustering.activations import ProcessedActivations, component_activations
from spd.clustering.consts import ComponentLabels
from spd.clustering.harvest_config import HarvestConfig
from spd.clustering.sample_membership import CompressedMembership
from spd.clustering.util import DeadComponentFilterStat, ModuleFilterFunc
from spd.log import logger
from spd.models.component_model import ComponentModel


@dataclass(frozen=True)
class ProcessedMemberships:
    """Processed, compressed sample memberships for exact merge iteration."""

    module_component_counts: dict[str, int]
    module_alive_counts: dict[str, int]
    labels: ComponentLabels
    dead_components_lst: ComponentLabels | None
    memberships: list[CompressedMembership]
    n_samples: int
    preview: ProcessedActivations | None = None

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
        import json

        from spd.clustering.sample_membership import memberships_to_sample_component_matrix

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

        if self.preview is not None:
            torch.save(self.preview.activations, path / "preview.pt")

    @classmethod
    def load(cls, path: Path) -> "ProcessedMemberships":
        import json

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

        preview: ProcessedActivations | None = None
        preview_path = path / "preview.pt"
        if preview_path.exists():
            preview_acts = torch.load(preview_path, weights_only=True)
            preview = ProcessedActivations(
                module_component_counts=metadata["module_component_counts"],
                module_alive_counts=metadata["module_alive_counts"],
                activations=preview_acts,
                labels=ComponentLabels(list(labels)),
                dead_components_lst=ComponentLabels(list(dead)) if dead else None,
            )

        return cls(
            module_component_counts=metadata["module_component_counts"],
            module_alive_counts=metadata["module_alive_counts"],
            labels=labels,
            dead_components_lst=dead,
            memberships=memberships,
            n_samples=metadata["n_samples"],
            preview=preview,
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
        preview_n_samples: int = 256,
    ) -> None:
        self.activation_threshold = activation_threshold
        self.filter_dead_threshold = filter_dead_threshold
        self.filter_dead_stat = filter_dead_stat
        self.filter_modules = filter_modules
        self.preview_n_samples = preview_n_samples

        self.n_samples = 0
        self.module_component_counts: dict[str, int] = {}
        self.max_activations: dict[str, Float[Tensor, " c"]] = {}
        self.sum_activations: dict[str, Float[Tensor, " c"]] = {}
        self.module_sample_rows: dict[str, list[np.ndarray]] = {}
        self.module_sample_components: dict[str, list[np.ndarray]] = {}
        self.preview_chunks: dict[str, list[Tensor]] = {}
        self.module_order: list[str] = []
        self._preview_rows = 0

    def _ensure_module(self, key: str, n_components: int) -> None:
        if key in self.module_component_counts:
            assert self.module_component_counts[key] == n_components, (
                f"Inconsistent component count for module '{key}': "
                f"{self.module_component_counts[key]} vs {n_components}"
            )
            return

        self.module_component_counts[key] = n_components
        self.max_activations[key] = torch.full((n_components,), float("-inf"))
        self.sum_activations[key] = torch.zeros((n_components,), dtype=torch.float64)
        self.module_sample_rows[key] = []
        self.module_sample_components[key] = []
        self.preview_chunks[key] = []
        self.module_order.append(key)

    def add_batch(self, activations: dict[str, Float[Tensor, "samples C"]]) -> None:
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
            act_local = act.detach()
            assert act_local.ndim == 2, (
                f"Expected 2D activations, got shape {tuple(act_local.shape)}"
            )
            self._ensure_module(key, act_local.shape[1])

            self.max_activations[key] = torch.maximum(
                self.max_activations[key], act_local.max(dim=0).values.cpu()
            )
            self.sum_activations[key] += act_local.sum(dim=0, dtype=torch.float64).cpu()

            if self._preview_rows < self.preview_n_samples:
                remaining = self.preview_n_samples - self._preview_rows
                self.preview_chunks[key].append(act_local[:remaining].cpu().clone())

            row_indices_t, comp_indices_t = torch.nonzero(
                act_local > self.activation_threshold, as_tuple=True
            )
            if row_indices_t.numel() > 0:
                self.module_sample_rows[key].append(
                    row_indices_t.to(dtype=torch.int32).cpu().numpy() + sample_offset
                )
                self.module_sample_components[key].append(
                    comp_indices_t.to(dtype=torch.int32).cpu().numpy()
                )

        self.n_samples += batch_n_samples
        self._preview_rows = min(self.n_samples, self.preview_n_samples)

    def finalize(self) -> ProcessedMemberships:
        module_alive_counts: dict[str, int] = {}
        alive_labels = ComponentLabels(list())
        dead_labels = ComponentLabels(list())
        memberships: list[CompressedMembership] = []

        preview_module_component_counts: dict[str, int] = {}
        preview_module_alive_counts: dict[str, int] = {}
        preview_chunks_alive: list[Tensor] = []

        for key in self.module_order:
            filter_values = (
                self.max_activations[key]
                if self.filter_dead_stat == "max"
                else (self.sum_activations[key] / self.n_samples).to(
                    self.max_activations[key].dtype
                )
            )
            n_components = self.module_component_counts[key]
            alive = (
                filter_values >= self.filter_dead_threshold
                if self.filter_dead_threshold > 0
                else torch.ones(n_components, dtype=torch.bool)
            )
            n_alive = int(alive.sum().item())
            module_alive_counts[key] = n_alive
            preview_module_component_counts[key] = n_components
            preview_module_alive_counts[key] = n_alive

            preview_tensor = (
                torch.cat(self.preview_chunks[key], dim=0)
                if self.preview_chunks[key]
                else torch.empty((0, n_components), dtype=filter_values.dtype)
            )

            for comp_idx in range(n_components):
                if not alive[comp_idx]:
                    dead_labels.append(f"{key}:{comp_idx}")

            alive_np = alive.numpy()
            alive_component_indices = np.flatnonzero(alive_np).astype(np.int32, copy=False)
            for comp_idx in alive_component_indices:
                alive_labels.append(f"{key}:{int(comp_idx)}")

            if n_alive > 0:
                row_chunks = self.module_sample_rows.pop(key)
                component_chunks = self.module_sample_components.pop(key)
                if row_chunks:
                    sample_rows = np.concatenate(row_chunks).astype(np.int64, copy=False)
                    sample_components = np.concatenate(component_chunks).astype(
                        np.int32, copy=False
                    )
                    alive_entries = alive_np[sample_components]
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
            else:
                self.module_sample_rows.pop(key)
                self.module_sample_components.pop(key)

            if n_alive > 0:
                preview_chunks_alive.append(preview_tensor[:, alive])

        preview: ProcessedActivations | None = None
        if preview_chunks_alive:
            preview = ProcessedActivations(
                module_component_counts=preview_module_component_counts,
                module_alive_counts=preview_module_alive_counts,
                activations=torch.cat(preview_chunks_alive, dim=1),
                labels=ComponentLabels(alive_labels.copy()),
                dead_components_lst=ComponentLabels(dead_labels.copy()) if dead_labels else None,
            )

        result = ProcessedMemberships(
            module_component_counts=self.module_component_counts,
            module_alive_counts=module_alive_counts,
            labels=alive_labels,
            dead_components_lst=dead_labels if dead_labels else None,
            memberships=memberships,
            n_samples=self.n_samples,
            preview=preview,
        )
        result.validate()
        return result


# ── Collection functions ───────────────────────────────────────────────────


def _lm_sample_positions(
    *,
    batch_size: int,
    n_ctx: int,
    n_tokens_per_seq: int | None,
    use_all_tokens_per_seq: bool,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if use_all_tokens_per_seq:
        positions = torch.arange(n_ctx).unsqueeze(0).expand(batch_size, -1)
    else:
        assert n_tokens_per_seq is not None
        positions = torch.randint(0, n_ctx, (batch_size, n_tokens_per_seq), generator=rng)
    return torch.arange(batch_size).unsqueeze(1).expand_as(positions), positions


def _flatten_lm_activations(
    act: Float[Tensor, "batch n_ctx C"],
    *,
    batch_size: int,
    n_ctx: int,
    n_tokens_per_seq: int | None,
    use_all_tokens_per_seq: bool,
    rng: torch.Generator,
) -> Float[Tensor, "samples C"]:
    if use_all_tokens_per_seq:
        return act.reshape(batch_size * n_ctx, -1)
    batch_indices, positions = _lm_sample_positions(
        batch_size=batch_size,
        n_ctx=n_ctx,
        n_tokens_per_seq=n_tokens_per_seq,
        use_all_tokens_per_seq=False,
        rng=rng,
    )
    return act[batch_indices, positions].reshape(batch_size * positions.shape[1], -1)


def collect_memberships_lm(
    model: ComponentModel,
    dataloader: DataLoader[Any],
    n_tokens: int,
    n_tokens_per_seq: int | None,
    device: torch.device | str,
    seed: int,
    activation_threshold: float,
    filter_dead_threshold: float,
    filter_dead_stat: DeadComponentFilterStat = "max",
    filter_modules: ModuleFilterFunc | None = None,
    preview_n_samples: int = 256,
    use_all_tokens_per_seq: bool = False,
) -> ProcessedMemberships:
    rng = torch.Generator().manual_seed(seed)
    builder = MembershipBuilder(
        activation_threshold=activation_threshold,
        filter_dead_threshold=filter_dead_threshold,
        filter_dead_stat=filter_dead_stat,
        filter_modules=filter_modules,
        preview_n_samples=preview_n_samples,
    )
    n_collected = 0

    pbar = tqdm(dataloader, desc="Collecting activations", unit="batch")
    for batch_data in pbar:
        input_ids = batch_data["input_ids"]
        batch_size, n_ctx = input_ids.shape
        activations = component_activations(model=model, batch=input_ids, device=device)

        tokens_per_seq = n_ctx if use_all_tokens_per_seq else n_tokens_per_seq
        assert tokens_per_seq is not None

        n_remaining = n_tokens - n_collected
        batch_take = min(batch_size * tokens_per_seq, n_remaining)
        sampled: dict[str, Float[Tensor, "samples C"]] = {
            key: _flatten_lm_activations(
                act,
                batch_size=batch_size,
                n_ctx=n_ctx,
                n_tokens_per_seq=n_tokens_per_seq,
                use_all_tokens_per_seq=use_all_tokens_per_seq,
                rng=rng,
            )[:batch_take]
            for key, act in activations.items()
        }
        builder.add_batch(sampled)
        del sampled, activations

        n_collected += batch_take
        pbar.set_postfix(tokens=f"{n_collected}/{n_tokens}")
        if n_collected >= n_tokens:
            break

    assert n_collected >= n_tokens, (
        f"Dataloader exhausted: collected {n_collected} tokens but needed {n_tokens}"
    )
    logger.info(f"Collected {n_collected} token activations (requested {n_tokens})")
    return builder.finalize()


def collect_memberships_resid_mlp(
    model: ComponentModel,
    dataloader: DataLoader[Any],
    n_samples: int,
    device: torch.device | str,
    activation_threshold: float,
    filter_dead_threshold: float,
    filter_dead_stat: DeadComponentFilterStat = "max",
    filter_modules: ModuleFilterFunc | None = None,
    preview_n_samples: int = 256,
) -> ProcessedMemberships:
    builder = MembershipBuilder(
        activation_threshold=activation_threshold,
        filter_dead_threshold=filter_dead_threshold,
        filter_dead_stat=filter_dead_stat,
        filter_modules=filter_modules,
        preview_n_samples=preview_n_samples,
    )
    n_collected = 0

    pbar = tqdm(dataloader, desc="Collecting activations", unit="batch")
    for batch_data in pbar:
        batch, _ = batch_data
        activations = component_activations(model=model, batch=batch, device=device)
        batch_take = min(batch.shape[0], n_samples - n_collected)
        builder.add_batch({key: act[:batch_take] for key, act in activations.items()})

        n_collected += batch_take
        pbar.set_postfix(samples=f"{n_collected}/{n_samples}")
        if n_collected >= n_samples:
            break

    assert n_collected >= n_samples, (
        f"Dataloader exhausted: collected {n_collected} samples but needed {n_samples}"
    )
    logger.info(f"Collected {n_collected} resid_mlp activations (requested {n_samples})")
    return builder.finalize()


def collect_memberships(
    model: ComponentModel,
    dataloader: DataLoader[Any],
    task_name: str,
    device: torch.device | str,
    config: HarvestConfig,
) -> ProcessedMemberships:
    if task_name == "lm":
        assert config.n_tokens is not None, "n_tokens required for LM tasks"
        assert config.use_all_tokens_per_seq or config.n_tokens_per_seq is not None
        return collect_memberships_lm(
            model=model,
            dataloader=dataloader,
            n_tokens=config.n_tokens,
            n_tokens_per_seq=config.n_tokens_per_seq,
            device=device,
            seed=config.dataset_seed,
            activation_threshold=config.activation_threshold,
            filter_dead_threshold=config.filter_dead_threshold,
            filter_dead_stat=config.filter_dead_stat,
            filter_modules=config.filter_modules,
            use_all_tokens_per_seq=config.use_all_tokens_per_seq,
        )

    n_samples = config.n_samples or config.batch_size
    return collect_memberships_resid_mlp(
        model=model,
        dataloader=dataloader,
        n_samples=n_samples,
        device=device,
        activation_threshold=config.activation_threshold,
        filter_dead_threshold=config.filter_dead_threshold,
        filter_dead_stat=config.filter_dead_stat,
        filter_modules=config.filter_modules,
    )
