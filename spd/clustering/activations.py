from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from jaxtyping import Bool, Float
from scipy import sparse
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.clustering.consts import (
    ActivationsTensor,
    BoolActivationsTensor,
    ClusterCoactivationShaped,
    ComponentLabels,
)
from spd.clustering.sample_membership import CompressedMembership
from spd.clustering.util import DeadComponentFilterStat, ModuleFilterFunc
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache


def component_activations(
    model: ComponentModel,
    device: torch.device | str,
    batch: Tensor,
) -> dict[str, ActivationsTensor]:
    """Get the component activations over a **single** batch."""
    causal_importances: dict[str, ActivationsTensor]
    with torch.no_grad():
        model_output: OutputWithCache = model(
            batch.to(device),
            cache_type="input",
        )

        causal_importances = model.calc_causal_importances(
            pre_weight_acts=model_output.cache,
            sampling="continuous",
            detach_inputs=False,
        ).lower_leaky

    return causal_importances


def _lm_sample_positions(
    *,
    batch_size: int,
    n_ctx: int,
    n_tokens_per_seq: int | None,
    use_all_tokens_per_seq: bool,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Return batch/token position indices for LM activation sampling."""
    if use_all_tokens_per_seq:
        positions = torch.arange(n_ctx).unsqueeze(0).expand(batch_size, -1)
    else:
        assert n_tokens_per_seq is not None, (
            "n_tokens_per_seq must be set when not using all tokens"
        )
        positions = torch.randint(0, n_ctx, (batch_size, n_tokens_per_seq), generator=rng)

    batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(positions)
    return batch_indices, positions


def _flatten_lm_activations(
    act: Float[Tensor, "batch n_ctx C"],
    *,
    batch_size: int,
    n_ctx: int,
    n_tokens_per_seq: int | None,
    use_all_tokens_per_seq: bool,
    rng: torch.Generator,
) -> Float[Tensor, "samples C"]:
    """Flatten LM activations for either full-sequence or random-token collection."""
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


def compute_coactivatons(
    activations: ActivationsTensor | BoolActivationsTensor,
) -> ClusterCoactivationShaped:
    """Compute the coactivations matrix from the activations."""
    return activations.float().T @ activations.float()


def _get_component_filter_values(
    activations: ActivationsTensor,
    filter_stat: DeadComponentFilterStat,
) -> Float[Tensor, " c"]:
    if filter_stat == "max":
        return activations.max(dim=0).values

    assert filter_stat == "mean", f"Unsupported dead component filter stat: {filter_stat}"
    return activations.mean(dim=0)


class FilteredActivations(NamedTuple):
    activations: ActivationsTensor
    "activations after filtering dead components"

    labels: ComponentLabels
    "list of length c with labels for each preserved component"

    dead_components_labels: ComponentLabels | None
    "list of labels for dead components, or None if no filtering was applied"

    @property
    def n_alive(self) -> int:
        """Number of alive components after filtering."""
        n_alive: int = len(self.labels)
        assert n_alive == self.activations.shape[1], (
            f"{n_alive = } != {self.activations.shape[1] = }"
        )
        return n_alive

    @property
    def n_dead(self) -> int:
        """Number of dead components after filtering."""
        return len(self.dead_components_labels) if self.dead_components_labels else 0


def filter_dead_components(
    activations: ActivationsTensor,
    labels: ComponentLabels,
    filter_dead_threshold: float = 0.01,
    filter_dead_stat: DeadComponentFilterStat = "max",
) -> FilteredActivations:
    """Filter out dead components based on a threshold

    if `filter_dead_threshold` is 0, no filtering is applied.
    activations and labels are returned as is, `dead_components_labels` is `None`.

    otherwise, components whose aggregate activation statistic across all samples is below the
    threshold are considered dead and filtered out. The statistic is selected by
    `filter_dead_stat` and the labels of dead components are returned in `dead_components_labels`.
    `dead_components_labels` will also be `None` if no components were below the threshold.
    """
    dead_components_lst: ComponentLabels | None = None
    if filter_dead_threshold > 0:
        dead_components_lst = ComponentLabels(list())
        filter_values: Float[Tensor, " c"] = _get_component_filter_values(
            activations=activations,
            filter_stat=filter_dead_stat,
        )
        dead_components: Bool[Tensor, " c"] = filter_values < filter_dead_threshold

        if dead_components.any():
            activations = activations[:, ~dead_components]
            alive_labels: list[tuple[str, bool]] = [
                (lbl, bool(keep.item()))
                for lbl, keep in zip(labels, ~dead_components, strict=False)
            ]
            # re-assign labels only if we are filtering
            labels = ComponentLabels([label for label, keep in alive_labels if keep])
            dead_components_lst = ComponentLabels(
                [label for label, keep in alive_labels if not keep]
            )

    return FilteredActivations(
        activations=activations,
        labels=labels,
        dead_components_labels=dead_components_lst if dead_components_lst else None,
    )


@dataclass(frozen=True)
class ProcessedActivations:
    """Processed activations after filtering and concatenation"""

    module_component_counts: dict[str, int]
    "total component count per module (including dead), preserving module order"

    module_alive_counts: dict[str, int]
    "alive component count per module, preserving module order"

    activations: ActivationsTensor
    "activations after filtering and concatenation"

    labels: ComponentLabels
    "list of length c with labels for each preserved component, format `{module_name}:{component_index}`"

    dead_components_lst: ComponentLabels | None
    "list of labels for dead components, or None if no filtering was applied"

    def validate(self) -> None:
        """Validate the processed activations"""
        # getting this property will also perform a variety of other checks
        assert self.n_components_alive > 0

    @property
    def n_components_original(self) -> int:
        return sum(self.module_component_counts.values())

    @property
    def n_components_alive(self) -> int:
        n_alive: int = len(self.labels)
        assert n_alive + self.n_components_dead == self.n_components_original, (
            f"({n_alive = }) + ({self.n_components_dead = }) != ({self.n_components_original = })"
        )
        assert n_alive == self.activations.shape[1], (
            f"{n_alive = } != {self.activations.shape[1] = }"
        )

        return n_alive

    @property
    def n_components_dead(self) -> int:
        return len(self.dead_components_lst) if self.dead_components_lst else 0

    @cached_property
    def label_index(self) -> dict[str, int | None]:
        """Create a mapping from label to alive index (`None` if dead)"""
        return {
            **{label: i for i, label in enumerate(self.labels)},
            **(
                {label: None for label in self.dead_components_lst}
                if self.dead_components_lst
                else {}
            ),
        }

    def get_label_index(self, label: str) -> int | None:
        """Get the index of a label in the activations, or None if it is dead"""
        return self.label_index[label]

    def get_label_index_alive(self, label: str) -> int:
        """Get the index of a label in the activations, or raise if it is dead"""
        idx: int | None = self.get_label_index(label)
        if idx is None:
            raise ValueError(f"Label '{label}' is dead and has no index in the activations.")
        return idx

    @property
    def module_keys(self) -> list[str]:
        return list(self.module_component_counts.keys())

    def get_module_indices(self, module_key: str) -> list[int | None]:
        """given a module key, return a list len "num components in that module", with int index in alive components, or None if dead"""
        num_components: int = self.module_component_counts[module_key]
        return [self.label_index[f"{module_key}:{i}"] for i in range(num_components)]

    def get_module_activations(self) -> dict[str, ActivationsTensor]:
        """Reconstruct per-module activation views (alive components only) from the concatenated tensor."""
        result: dict[str, ActivationsTensor] = {}
        offset = 0
        for key, n_alive in self.module_alive_counts.items():
            if n_alive > 0:
                result[key] = self.activations[:, offset : offset + n_alive]
            offset += n_alive
        return result


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
        """Save to directory: sparse memberships + metadata + optional preview tensor."""
        import json

        from scipy import sparse as sp

        from spd.clustering.sample_membership import memberships_to_sample_component_matrix

        path.mkdir(parents=True, exist_ok=True)

        matrix = memberships_to_sample_component_matrix(self.memberships, fmt="csc")
        assert isinstance(matrix, sp.csc_matrix)
        sp.save_npz(path / "memberships.npz", matrix)

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
        """Load from directory saved by .save()."""
        import json

        from scipy import sparse as sp

        metadata = json.loads((path / "metadata.json").read_text())
        labels = ComponentLabels(metadata["labels"])
        dead = (
            ComponentLabels(metadata["dead_components_lst"])
            if metadata["dead_components_lst"]
            else None
        )

        matrix_csc = sp.load_npz(path / "memberships.npz").tocsc()
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

    This stores only active sample ids per component plus a small dense preview
    for plots/logging. It assumes thresholded boolean merge semantics.
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

    def add_batch(
        self,
        activations: dict[str, Float[Tensor, "samples C"]],
    ) -> None:
        """Add a batch of per-module activations shaped [samples, components]."""
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
                act_local > self.activation_threshold,
                as_tuple=True,
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
                label = f"{key}:{comp_idx}"
                if not alive[comp_idx]:
                    dead_labels.append(label)

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
                            sample_indices=sample_ids,
                            n_samples=self.n_samples,
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
    """Collect LM activations across batches into compressed memberships."""
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

        sampled_activations: dict[str, Float[Tensor, "samples C"]] = {}
        n_remaining = n_tokens - n_collected
        batch_take = min(batch_size * tokens_per_seq, n_remaining)
        for key, act in activations.items():
            sampled = _flatten_lm_activations(
                act,
                batch_size=batch_size,
                n_ctx=n_ctx,
                n_tokens_per_seq=n_tokens_per_seq,
                use_all_tokens_per_seq=use_all_tokens_per_seq,
                rng=rng,
            )
            sampled_activations[key] = sampled[:batch_take]

        builder.add_batch(sampled_activations)
        del sampled_activations
        del activations

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
    """Collect ResidMLP activations across batches into compressed memberships."""
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

        n_remaining = n_samples - n_collected
        batch_take = min(batch.shape[0], n_remaining)
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
    activation_threshold: float,
    filter_dead_threshold: float,
    filter_dead_stat: DeadComponentFilterStat,
    filter_modules: ModuleFilterFunc | None,
    *,
    n_tokens: int | None = None,
    n_tokens_per_seq: int | None = None,
    use_all_tokens_per_seq: bool = False,
    n_samples: int | None = None,
    dataset_seed: int = 0,
) -> ProcessedMemberships:
    """Collect compressed memberships from a model. Dispatches by task_name."""
    if task_name == "lm":
        assert n_tokens is not None, "n_tokens required for LM tasks"
        assert use_all_tokens_per_seq or n_tokens_per_seq is not None
        return collect_memberships_lm(
            model=model,
            dataloader=dataloader,
            n_tokens=n_tokens,
            n_tokens_per_seq=n_tokens_per_seq,
            device=device,
            seed=dataset_seed,
            activation_threshold=activation_threshold,
            filter_dead_threshold=filter_dead_threshold,
            filter_dead_stat=filter_dead_stat,
            filter_modules=filter_modules,
            use_all_tokens_per_seq=use_all_tokens_per_seq,
        )

    assert n_samples is not None, f"n_samples required for {task_name} tasks"
    return collect_memberships_resid_mlp(
        model=model,
        dataloader=dataloader,
        n_samples=n_samples,
        device=device,
        activation_threshold=activation_threshold,
        filter_dead_threshold=filter_dead_threshold,
        filter_dead_stat=filter_dead_stat,
        filter_modules=filter_modules,
    )


def process_activations(
    activations: dict[
        str,  # module name to
        Float[Tensor, "samples C"]  # (sample x component gate activations)
        | Float[Tensor, " n_sample n_ctx C"],  # (sample x seq index x component gate activations)
    ],
    filter_dead_threshold: float,
    filter_dead_stat: DeadComponentFilterStat = "max",
    seq_mode: Literal["concat", "seq_mean", None] = None,
    filter_modules: ModuleFilterFunc | None = None,
) -> ProcessedActivations:
    """Concatenate per-module activations and filter dead components.

    Fuses concatenation and filtering into a single pass to avoid holding two full
    copies (~2x total components * n_samples) in memory simultaneously.
    """

    # reshape -- special cases for llms
    # ============================================================
    activations_: dict[str, ActivationsTensor]
    if seq_mode == "concat":
        activations_ = {
            key: act.reshape(act.shape[0] * act.shape[1], act.shape[2])
            for key, act in activations.items()
        }
    elif seq_mode == "seq_mean":
        activations_ = {
            key: act.mean(dim=1) if act.ndim == 3 else act for key, act in activations.items()
        }
    else:
        activations_ = activations

    # filter activations for only the modules we want
    if filter_modules is not None:
        activations_ = {key: act for key, act in activations_.items() if filter_modules(key)}

    # First pass: compute per-module component counts and alive masks
    module_component_counts: dict[str, int] = {}
    alive_masks: dict[str, Bool[Tensor, " c"]] = {}
    total_alive = 0
    for key, act in activations_.items():
        c = act.shape[-1]
        module_component_counts[key] = c
        if filter_dead_threshold > 0:
            filter_values: Float[Tensor, " c"] = _get_component_filter_values(
                activations=act,
                filter_stat=filter_dead_stat,
            )
            alive = filter_values >= filter_dead_threshold
            alive_masks[key] = alive
            total_alive += int(alive.sum().item())
        else:
            total_alive += c

    total_c = sum(module_component_counts.values())

    # Second pass: pre-allocate output and copy alive components one module at a time,
    # freeing each module's tensor after copying to keep peak memory ~= 1x total size.
    first_act = next(iter(activations_.values()))
    n_samples = first_act.shape[0]
    dtype = first_act.dtype
    act_filtered = torch.empty(n_samples, total_alive, dtype=dtype)

    offset = 0
    alive_labels = ComponentLabels(list())
    dead_labels = ComponentLabels(list())
    module_alive_counts: dict[str, int] = {}

    for key in list(activations_.keys()):
        tensor = activations_.pop(key)
        c = tensor.shape[-1]

        if filter_dead_threshold > 0:
            alive = alive_masks[key]
            n_alive = int(alive.sum().item())
            for i in range(c):
                label = f"{key}:{i}"
                if alive[i]:
                    alive_labels.append(label)
                else:
                    dead_labels.append(label)
            if n_alive > 0:
                act_filtered[:, offset : offset + n_alive] = tensor[:, alive]
        else:
            n_alive = c
            alive_labels.extend([f"{key}:{i}" for i in range(c)])
            act_filtered[:, offset : offset + n_alive] = tensor

        module_alive_counts[key] = n_alive
        offset += n_alive
        del tensor

    assert offset == total_alive
    assert list(module_alive_counts.keys()) == list(module_component_counts.keys())
    assert len(alive_labels) + len(dead_labels) == total_c

    return ProcessedActivations(
        module_component_counts=module_component_counts,
        module_alive_counts=module_alive_counts,
        activations=act_filtered,
        labels=alive_labels,
        dead_components_lst=dead_labels if dead_labels else None,
    )
