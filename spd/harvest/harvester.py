"""Harvester for collecting component statistics in a single pass.

All accumulator state lives as tensors on `device` (GPU during harvesting, CPU during merge).
"""

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import tqdm
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.harvest.reservoir import (
    WINDOW_PAD_SENTINEL,
    ActivationExamplesReservoir,
    ActivationWindows,
)
from spd.harvest.sampling import sample_at_most_n_per_group, top_k_pmi
from spd.harvest.schemas import ComponentData, ComponentTokenPMI
from spd.log import logger


def extract_padding_firing_windows(
    batch: Int[Tensor, "B S"],
    firings: Bool[Tensor, "B S C"],
    activations: dict[str, Float[Tensor, "B S C"]],
    max_examples_per_batch_per_component: int,
    context_tokens_per_side: int,
) -> ActivationWindows | None:
    batch_idx, seq_idx, comp_idx = torch.where(firings)
    if len(batch_idx) == 0:
        return None

    keep = sample_at_most_n_per_group(comp_idx, max_examples_per_batch_per_component)
    batch_idx, seq_idx, comp_idx = batch_idx[keep], seq_idx[keep], comp_idx[keep]

    seq_len = batch.shape[1]
    offsets = torch.arange(
        -context_tokens_per_side, context_tokens_per_side + 1, device=batch.device
    )
    window_size = offsets.shape[0]
    assert window_size == 2 * context_tokens_per_side + 1

    window_positions: Int[Tensor, "n_firings window_size"]
    window_positions = seq_idx.unsqueeze(1) + offsets.unsqueeze(0)

    in_bounds = (window_positions >= 0) & (window_positions < seq_len)
    clamped = window_positions.clamp(0, seq_len - 1)

    batch_idx_rep = repeat(batch_idx, "n_firings -> n_firings window_size", window_size=window_size)
    c_idx_rep = repeat(comp_idx, "n_firings -> n_firings window_size", window_size=window_size)

    token_windows = batch[batch_idx_rep, clamped]
    token_windows[~in_bounds] = WINDOW_PAD_SENTINEL

    firing_windows = firings[batch_idx_rep, clamped, c_idx_rep]
    firing_windows[~in_bounds] = False

    activation_windows = {}
    for act_type, act in activations.items():
        activation_windows[act_type] = act[batch_idx_rep, clamped, c_idx_rep]
        activation_windows[act_type][~in_bounds] = 0.0

    return ActivationWindows(
        component_idx=comp_idx,
        token_windows=token_windows,
        firing_windows=firing_windows,
        activation_windows=activation_windows,
    )


class Harvester:
    """Accumulates component statistics in a single pass over data.

    All mutable state is stored as tensors on `device`. Workers on GPU accumulate
    into GPU tensors; the merge job reconstructs on CPU.
    """

    def __init__(
        self,
        layers: list[tuple[str, int]],
        vocab_size: int,
        max_examples_per_component: int,
        context_tokens_per_side: int,
        max_examples_per_batch_per_component: int,
        device: torch.device,
    ):
        self.layers = layers
        self.vocab_size = vocab_size
        self.max_examples_per_component = max_examples_per_component
        self.context_tokens_per_side = context_tokens_per_side
        self.max_examples_per_batch_per_component = max_examples_per_batch_per_component
        self.device = device

        self.layer_offsets: dict[str, int] = {}
        offset = 0
        for layer, c in layers:
            self.layer_offsets[layer] = offset
            offset += c

        n_components = offset

        window_size = 2 * context_tokens_per_side + 1

        # Per-component firing stats
        self.firing_counts = torch.zeros(n_components, device=device)
        self.activation_sums = defaultdict[str, Tensor](
            lambda: torch.zeros(n_components, device=device)
        )
        self.cooccurrence_counts: Float[Tensor, "C C"] = torch.zeros(
            n_components, n_components, device=device, dtype=torch.float32
        )

        # Per-(component, token) stats for PMI computation
        #   input: hard token counts at positions where component fires
        #   output: predicted probability mass at positions where component fires
        self.input_cooccurrence: Int[Tensor, "C vocab"] = torch.zeros(
            n_components, vocab_size, device=device, dtype=torch.long
        )
        self.input_marginals: Int[Tensor, " vocab"] = torch.zeros(
            vocab_size, device=device, dtype=torch.long
        )
        self.output_cooccurrence: Float[Tensor, "C vocab"] = torch.zeros(
            n_components, vocab_size, device=device
        )
        self.output_marginals: Float[Tensor, " vocab"] = torch.zeros(vocab_size, device=device)

        self.reservoir = ActivationExamplesReservoir.create(
            n_components, max_examples_per_component, window_size, device
        )
        self.total_tokens_processed = 0

    @property
    def layer_names(self) -> list[str]:
        return [layer for layer, _ in self.layers]

    @property
    def c_per_layer(self) -> dict[str, int]:
        return {layer: c for layer, c in self.layers}

    @property
    def component_keys(self) -> list[str]:
        return [f"{layer}:{i}" for layer, c in self.layers for i in range(c)]

    # -- Batch processing --------------------------------------------------

    def process_batch(
        self,
        batch: Int[Tensor, "B S"],
        firings: dict[str, Bool[Tensor, "B S C"]],
        activations: dict[str, dict[str, Float[Tensor, "B S C"]]],
        output_probs: Float[Tensor, "B S V"],
    ) -> None:
        self.total_tokens_processed += batch.numel()

        tokens_flat = rearrange(batch, "b s -> (b s)")
        probs_flat = rearrange(output_probs, "b s v -> (b s) v")

        firings_cat = torch.cat([firings[layer] for layer in self.layer_names], dim=-1)
        firings_flat = rearrange(firings_cat, "b s lc -> (b s) lc")

        act_types = list(activations[self.layer_names[0]].keys())
        activations_cat: dict[str, Float[Tensor, "B S LC"]] = {}
        for act_type in act_types:
            activations_cat[act_type] = torch.cat(
                [activations[layer][act_type] for layer in self.layer_names], dim=-1
            )

        self.firing_counts += reduce(firings_cat, "b s lc -> lc", "sum")

        for act_type, act in activations_cat.items():
            self.activation_sums[act_type] += reduce(act, "b s lc -> lc", "sum")

        firings_float = firings_flat.float()
        self.cooccurrence_counts += einsum(firings_float, firings_float, "S c1, S c2 -> c1 c2")
        self._accumulate_token_stats(tokens_flat, probs_flat, firings_float)
        self._collect_activation_examples(batch, firings_cat, activations_cat)

    def _accumulate_token_stats(
        self,
        tokens_flat: Int[Tensor, " S"],
        probs_flat: Float[Tensor, "S vocab"],
        firing_flat: Float[Tensor, "S LC"],
    ) -> None:
        n_components = firing_flat.shape[1]
        token_indices = repeat(tokens_flat, "S -> lc S", lc=n_components)

        # use scatter_add for inputs because inputs are one-hot / token indices
        self.input_cooccurrence.scatter_add_(
            dim=1, index=token_indices, src=rearrange(firing_flat, "S lc -> lc S").long()
        )
        self.input_marginals.scatter_add_(
            dim=0,
            index=tokens_flat,
            src=torch.ones(tokens_flat.shape[0], device=self.device, dtype=torch.long),
        )

        # however, for outputs we need to accumulate probability mass over vocab
        self.output_cooccurrence += einsum(firing_flat, probs_flat, "S lc, S v -> lc v")
        self.output_marginals += reduce(probs_flat, "S v -> v", "sum")

    def _collect_activation_examples(
        self,
        batch: Int[Tensor, "B S"],
        firings: Bool[Tensor, "B S LC"],
        activations: dict[str, Float[Tensor, "B S LC"]],
    ) -> None:
        res = extract_padding_firing_windows(
            batch,
            firings,
            activations,
            self.max_examples_per_batch_per_component,
            self.context_tokens_per_side,
        )
        if res is not None:
            self.reservoir.add(res)

    def save(self, path: Path) -> None:
        data: dict[str, object] = {
            "layers": self.layers,
            "vocab_size": self.vocab_size,
            "max_examples_per_component": self.max_examples_per_component,
            "context_tokens_per_side": self.context_tokens_per_side,
            "max_examples_per_batch_per_component": self.max_examples_per_batch_per_component,
            "total_tokens_processed": self.total_tokens_processed,
            "reservoir": self.reservoir.state_dict(),
            "firing_counts": self.firing_counts.cpu(),
            "activation_sums": {
                act_type: self.activation_sums[act_type].cpu() for act_type in self.activation_sums
            },
            "cooccurrence_counts": self.cooccurrence_counts.cpu(),
            "input_cooccurrence": self.input_cooccurrence.cpu(),
            "input_marginals": self.input_marginals.cpu(),
            "output_cooccurrence": self.output_cooccurrence.cpu(),
            "output_marginals": self.output_marginals.cpu(),
        }
        torch.save(data, path)

    @staticmethod
    def load(path: Path, device: torch.device) -> "Harvester":
        d: dict[str, Any] = torch.load(path, weights_only=False)
        h = Harvester(
            layers=d["layers"],
            vocab_size=d["vocab_size"],
            max_examples_per_component=d["max_examples_per_component"],
            context_tokens_per_side=d["context_tokens_per_side"],
            max_examples_per_batch_per_component=d.get("max_examples_per_batch_per_component", 5),
            device=device,
        )
        h.total_tokens_processed = d["total_tokens_processed"]
        h.firing_counts = d["firing_counts"].to(device)
        h.activation_sums = {k: v.to(device) for k, v in d["activation_sums"].items()}
        h.cooccurrence_counts = d["cooccurrence_counts"].to(device)
        h.input_cooccurrence = d["input_cooccurrence"].to(device)
        h.input_marginals = d["input_marginals"].to(device)
        h.output_cooccurrence = d["output_cooccurrence"].to(device)
        h.output_marginals = d["output_marginals"].to(device)
        h.reservoir = ActivationExamplesReservoir.from_state_dict(d["reservoir"], device)
        return h

    def merge(self, other: "Harvester") -> None:
        assert other.layer_names == self.layer_names
        assert other.c_per_layer == self.c_per_layer
        assert other.vocab_size == self.vocab_size

        self.firing_counts += other.firing_counts
        for act_type in self.activation_sums:
            self.activation_sums[act_type] += other.activation_sums[act_type]
        self.cooccurrence_counts += other.cooccurrence_counts
        self.input_cooccurrence += other.input_cooccurrence
        self.input_marginals += other.input_marginals
        self.output_cooccurrence += other.output_cooccurrence
        self.output_marginals += other.output_marginals
        self.total_tokens_processed += other.total_tokens_processed

        self.reservoir.merge(other.reservoir)

    # -- Result building ---------------------------------------------------

    def build_results(self, pmi_top_k_tokens: int) -> Iterator[ComponentData]:
        """Yield ComponentData objects one at a time (constant memory)."""
        logger.info("  Moving tensors to CPU...")
        mean_activations = {
            act_type: (self.activation_sums[act_type] / self.total_tokens_processed).cpu()
            for act_type in self.activation_sums
        }
        firing_counts = self.firing_counts.cpu()
        input_cooccurrence = self.input_cooccurrence.cpu()
        input_marginals = self.input_marginals.cpu()
        output_cooccurrence = self.output_cooccurrence.cpu()
        output_marginals = self.output_marginals.cpu()

        reservoir_cpu = self.reservoir.to(torch.device("cpu"))

        _log_base_rate_summary(firing_counts, input_marginals)

        for layer, layer_c in self.layers:
            offset = self.layer_offsets[layer]

            for component_idx in tqdm.tqdm(range(layer_c), desc="Building components"):
                flat_idx = offset + component_idx

                n_firings = float(firing_counts[flat_idx])
                if n_firings == 0:
                    continue

                yield ComponentData(
                    component_key=f"{layer}:{component_idx}",
                    layer=layer,
                    component_idx=component_idx,  # as in, the index of the component within the layer
                    firing_density=n_firings / self.total_tokens_processed,
                    mean_activations={
                        act_type: float(mean_activations[act_type][flat_idx].item())
                        for act_type in mean_activations
                    },
                    activation_examples=list(reservoir_cpu.examples(flat_idx)),
                    input_token_pmi=_compute_token_pmi(
                        input_cooccurrence[flat_idx],
                        input_marginals,
                        n_firings,
                        self.total_tokens_processed,
                        pmi_top_k_tokens,
                    ),
                    output_token_pmi=_compute_token_pmi(
                        output_cooccurrence[flat_idx],
                        output_marginals,
                        n_firings,
                        self.total_tokens_processed,
                        pmi_top_k_tokens,
                    ),
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_base_rate_summary(firing_counts: Tensor, input_marginals: Tensor) -> None:
    active_counts = firing_counts[firing_counts > 0]
    if len(active_counts) == 0:
        logger.info("  WARNING: No components fired above threshold!")
        return

    sorted_counts = active_counts.sort().values
    n_active = len(active_counts)
    logger.info("\n  === Base Rate Summary ===")
    logger.info(f"  Components with firings: {n_active} / {len(firing_counts)}")
    logger.info(
        f"  Firing counts - min: {int(sorted_counts[0])}, "
        f"median: {int(sorted_counts[n_active // 2])}, "
        f"max: {int(sorted_counts[-1])}"
    )

    LOW_FIRING_THRESHOLD = 100
    n_sparse = int((active_counts < LOW_FIRING_THRESHOLD).sum())
    if n_sparse > 0:
        logger.info(
            f"  WARNING: {n_sparse} components have <{LOW_FIRING_THRESHOLD} firings "
            f"(stats may be noisy)"
        )

    active_tokens = input_marginals[input_marginals > 0]
    sorted_token_counts = active_tokens.sort().values
    n_tokens = len(active_tokens)
    logger.info(
        f"  Tokens seen: {n_tokens} unique, "
        f"occurrences - min: {int(sorted_token_counts[0])}, "
        f"median: {int(sorted_token_counts[n_tokens // 2])}, "
        f"max: {int(sorted_token_counts[-1])}"
    )

    RARE_TOKEN_THRESHOLD = 10
    n_rare = int((active_tokens < RARE_TOKEN_THRESHOLD).sum())
    if n_rare > 0:
        logger.info(
            f"  Note: {n_rare} tokens have <{RARE_TOKEN_THRESHOLD} occurrences "
            f"(high precision/recall with these may be spurious)"
        )
    logger.info("")


def _compute_token_pmi(
    token_mass_for_component: Tensor,
    token_mass_totals: Tensor,
    component_firing_count: float,
    total_tokens: int,
    top_k: int,
) -> ComponentTokenPMI:
    top, bottom = top_k_pmi(
        cooccurrence_counts=token_mass_for_component,
        marginal_counts=token_mass_totals,
        target_count=component_firing_count,
        total_count=total_tokens,
        top_k=top_k,
    )
    return ComponentTokenPMI(top=top, bottom=bottom)
