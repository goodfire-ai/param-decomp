"""Harvester for collecting component statistics in a single pass.

All accumulator state lives as NumPy arrays on the host. Counts and co-occurrence use
int64 (firings summed over a stream overflow narrower integer types); probability-mass
accumulators use float64.
"""

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int

from param_decomp.log import logger
from param_decomp_lab.harvest.reservoir import (
    WINDOW_PAD_SENTINEL,
    ActivationExamplesReservoir,
    ActivationWindows,
)
from param_decomp_lab.harvest.sampling import sample_at_most_n_per_group, top_k_pmi
from param_decomp_lab.harvest.schemas import ComponentData, ComponentTokenPMI


def extract_padding_firing_windows(
    batch: Int[np.ndarray, "B S"],
    firings: Bool[np.ndarray, "B S C"],
    activations: dict[str, Float[np.ndarray, "B S C"]],
    max_examples_per_batch_per_component: int,
    context_tokens_per_side: int,
    rng: np.random.Generator,
) -> ActivationWindows | None:
    batch_idx, seq_idx, comp_idx = np.nonzero(firings)
    if len(batch_idx) == 0:
        return None

    keep = sample_at_most_n_per_group(comp_idx, max_examples_per_batch_per_component, rng)
    batch_idx, seq_idx, comp_idx = batch_idx[keep], seq_idx[keep], comp_idx[keep]

    seq_len = batch.shape[1]
    offsets = np.arange(-context_tokens_per_side, context_tokens_per_side + 1)
    window_size = offsets.shape[0]
    assert window_size == 2 * context_tokens_per_side + 1

    window_positions: Int[np.ndarray, "n_firings window_size"]
    window_positions = seq_idx[:, None] + offsets[None, :]

    in_bounds = (window_positions >= 0) & (window_positions < seq_len)
    clamped = np.clip(window_positions, 0, seq_len - 1)

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

    All mutable state is stored as NumPy arrays on the host. Workers accumulate into
    their own arrays; the merge job sums them.
    """

    def __init__(
        self,
        layers: list[tuple[str, int]],
        vocab_size: int,
        max_examples_per_component: int,
        context_tokens_per_side: int,
        max_examples_per_batch_per_component: int,
        collect_component_cooccurrence: bool,
        selected_global_indices: Int[np.ndarray, " n_selected"] | None = None,
        seed: int = 0,
    ):
        self.layers = layers
        self.vocab_size = vocab_size
        self.max_examples_per_component = max_examples_per_component
        self.context_tokens_per_side = context_tokens_per_side
        self.max_examples_per_batch_per_component = max_examples_per_batch_per_component
        self.collect_component_cooccurrence = collect_component_cooccurrence
        self.rng = np.random.default_rng(seed)

        self.layer_offsets: dict[str, int] = {}
        offset = 0
        for layer, c in layers:
            self.layer_offsets[layer] = offset
            offset += c
        n_full = offset

        full_layout = [(layer, i) for layer, c in layers for i in range(c)]
        if selected_global_indices is None:
            self.gather_index: Int[np.ndarray, " n_selected"] | None = None
            self.component_layout = full_layout
        else:
            sel = np.asarray(selected_global_indices)
            assert sel.ndim == 1 and sel.size > 0, f"bad selection shape {sel.shape}"
            assert np.issubdtype(sel.dtype, np.integer), (
                f"selection must be integer, got {sel.dtype}"
            )
            assert (sel >= 0).all() and (sel < n_full).all(), "selection out of range"
            assert np.unique(sel).size == sel.size, "selected indices must be unique"
            self.gather_index = sel
            self.component_layout = [full_layout[g] for g in sel.tolist()]

        n_components = len(self.component_layout)

        window_size = 2 * context_tokens_per_side + 1

        # Per-component firing stats
        self.firing_counts = np.zeros(n_components, dtype=np.int64)
        self.activation_sums = defaultdict[str, np.ndarray](
            lambda: np.zeros(n_components, dtype=np.float64)
        )
        self.cooccurrence_counts: Int[np.ndarray, "C C"] | None = (
            np.zeros((n_components, n_components), dtype=np.int64)
            if collect_component_cooccurrence
            else None
        )

        # Per-(component, token) stats for PMI computation
        #   input: hard token counts at positions where component fires
        #   output: predicted probability mass at positions where component fires
        self.input_cooccurrence: Int[np.ndarray, "C vocab"] = np.zeros(
            (n_components, vocab_size), dtype=np.int64
        )
        self.input_marginals: Int[np.ndarray, " vocab"] = np.zeros(vocab_size, dtype=np.int64)
        self.output_cooccurrence: Float[np.ndarray, "C vocab"] = np.zeros(
            (n_components, vocab_size), dtype=np.float64
        )
        self.output_marginals: Float[np.ndarray, " vocab"] = np.zeros(vocab_size, dtype=np.float64)

        self.reservoir = ActivationExamplesReservoir.create(
            n_components, max_examples_per_component, window_size
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
        return [f"{layer}:{i}" for layer, i in self.component_layout]

    # -- Batch processing --------------------------------------------------

    def process_batch(
        self,
        batch: Int[np.ndarray, "B S"],
        firings: dict[str, Bool[np.ndarray, "B S C"]],
        activations: dict[str, dict[str, Float[np.ndarray, "B S C"]]],
        output_probs: Float[np.ndarray, "B S V"],
    ) -> None:
        self.total_tokens_processed += batch.size

        tokens_flat = rearrange(batch, "b s -> (b s)")
        probs_flat = rearrange(output_probs, "b s v -> (b s) v").astype(np.float64)

        firings_cat = np.concatenate([firings[layer] for layer in self.layer_names], axis=-1)

        act_types = list(activations[self.layer_names[0]].keys())
        activations_cat: dict[str, Float[np.ndarray, "B S LC"]] = {}
        for act_type in act_types:
            activations_cat[act_type] = np.concatenate(
                [activations[layer][act_type] for layer in self.layer_names], axis=-1
            )

        if self.gather_index is not None:
            firings_cat = firings_cat[..., self.gather_index]
            activations_cat = {at: a[..., self.gather_index] for at, a in activations_cat.items()}

        firings_flat = rearrange(firings_cat, "b s lc -> (b s) lc")

        self.firing_counts += reduce(firings_cat.astype(np.int64), "b s lc -> lc", "sum")

        for act_type, act in activations_cat.items():
            self.activation_sums[act_type] += reduce(act.astype(np.float64), "b s lc -> lc", "sum")

        if self.cooccurrence_counts is not None:
            firings_int = firings_flat.astype(np.int64)
            self.cooccurrence_counts += einsum(firings_int, firings_int, "S c1, S c2 -> c1 c2")
        self._accumulate_token_stats(tokens_flat, probs_flat, firings_flat)
        self._collect_activation_examples(batch, firings_cat, activations_cat)

    def _accumulate_token_stats(
        self,
        tokens_flat: Int[np.ndarray, " S"],
        probs_flat: Float[np.ndarray, "S vocab"],
        firing_flat: Bool[np.ndarray, "S LC"],
    ) -> None:
        # inputs are hard token counts: add each firing into the (component, token) cell.
        # Index with parallel (component, token) integer arrays over only the firing entries.
        fire_pos, fire_comp = np.nonzero(firing_flat)
        np.add.at(self.input_cooccurrence, (fire_comp, tokens_flat[fire_pos]), 1)
        np.add.at(self.input_marginals, tokens_flat, 1)

        # outputs accumulate predicted probability mass over vocab. A plain matmul
        # (firingᵀ @ probs) dispatches to multithreaded BLAS dgemm; the equivalent
        # np.einsum runs a naive single-threaded loop (~37x slower at LM vocab sizes).
        self.output_cooccurrence += firing_flat.astype(np.float64).T @ probs_flat
        self.output_marginals += reduce(probs_flat, "S v -> v", "sum")

    def _collect_activation_examples(
        self,
        batch: Int[np.ndarray, "B S"],
        firings: Bool[np.ndarray, "B S LC"],
        activations: dict[str, Float[np.ndarray, "B S LC"]],
    ) -> None:
        res = extract_padding_firing_windows(
            batch,
            firings,
            activations,
            self.max_examples_per_batch_per_component,
            self.context_tokens_per_side,
            self.rng,
        )
        if res is not None:
            self.reservoir.add(res, self.rng)

    def save(self, path: Path) -> None:
        data: dict[str, object] = {
            "layers": self.layers,
            "vocab_size": self.vocab_size,
            "max_examples_per_component": self.max_examples_per_component,
            "context_tokens_per_side": self.context_tokens_per_side,
            "max_examples_per_batch_per_component": self.max_examples_per_batch_per_component,
            "gather_index": self.gather_index,
            "total_tokens_processed": self.total_tokens_processed,
            "reservoir": self.reservoir.state_dict(),
            "firing_counts": self.firing_counts,
            "activation_sums": dict(self.activation_sums),
            "collect_component_cooccurrence": self.collect_component_cooccurrence,
            "cooccurrence_counts": self.cooccurrence_counts,
            "input_cooccurrence": self.input_cooccurrence,
            "input_marginals": self.input_marginals,
            "output_cooccurrence": self.output_cooccurrence,
            "output_marginals": self.output_marginals,
        }
        np.savez(path, harvester=np.array(data, dtype=object), allow_pickle=True)

    @staticmethod
    def load(path: Path) -> "Harvester":
        loaded = np.load(path, allow_pickle=True)
        d: dict[str, Any] = loaded["harvester"].item()
        h = Harvester(
            layers=d["layers"],
            vocab_size=d["vocab_size"],
            max_examples_per_component=d["max_examples_per_component"],
            context_tokens_per_side=d["context_tokens_per_side"],
            max_examples_per_batch_per_component=d["max_examples_per_batch_per_component"],
            collect_component_cooccurrence=d["collect_component_cooccurrence"],
            selected_global_indices=d["gather_index"],
        )
        h.total_tokens_processed = d["total_tokens_processed"]
        h.firing_counts = d["firing_counts"]
        for act_type, sums in d["activation_sums"].items():
            h.activation_sums[act_type] = sums
        h.cooccurrence_counts = d["cooccurrence_counts"]
        h.input_cooccurrence = d["input_cooccurrence"]
        h.input_marginals = d["input_marginals"]
        h.output_cooccurrence = d["output_cooccurrence"]
        h.output_marginals = d["output_marginals"]
        h.reservoir = ActivationExamplesReservoir.from_state_dict(d["reservoir"])
        return h

    def merge(self, other: "Harvester") -> None:
        assert other.layer_names == self.layer_names
        assert other.c_per_layer == self.c_per_layer
        assert other.vocab_size == self.vocab_size
        assert other.component_layout == self.component_layout, "mismatched component selection"

        assert (self.cooccurrence_counts is None) == (other.cooccurrence_counts is None), (
            "Cannot merge harvesters with mismatched component-cooccurrence collection"
        )

        self.firing_counts += other.firing_counts
        for act_type in self.activation_sums:
            self.activation_sums[act_type] += other.activation_sums[act_type]
        if self.cooccurrence_counts is not None:
            assert other.cooccurrence_counts is not None
            self.cooccurrence_counts += other.cooccurrence_counts
        self.input_cooccurrence += other.input_cooccurrence
        self.input_marginals += other.input_marginals
        self.output_cooccurrence += other.output_cooccurrence
        self.output_marginals += other.output_marginals
        self.total_tokens_processed += other.total_tokens_processed

        self.reservoir.merge(other.reservoir, self.rng)

    # -- Result building ---------------------------------------------------

    def build_results(self, pmi_top_k_tokens: int) -> Iterator[ComponentData]:
        """Yield ComponentData objects one at a time (constant memory)."""
        mean_activations = {
            act_type: self.activation_sums[act_type] / self.total_tokens_processed
            for act_type in self.activation_sums
        }

        _log_base_rate_summary(self.firing_counts, self.input_marginals)

        for flat_idx, (layer, component_idx) in enumerate(
            tqdm.tqdm(self.component_layout, desc="Building components")
        ):
            n_firings = float(self.firing_counts[flat_idx])
            if n_firings == 0:
                continue

            yield ComponentData(
                component_key=f"{layer}:{component_idx}",
                layer=layer,
                component_idx=component_idx,  # as in, the index of the component within the layer
                firing_density=n_firings / self.total_tokens_processed,
                mean_activations={
                    act_type: float(mean_activations[act_type][flat_idx])
                    for act_type in mean_activations
                },
                activation_examples=list(self.reservoir.examples(flat_idx)),
                input_token_pmi=_compute_token_pmi(
                    self.input_cooccurrence[flat_idx].astype(np.float64),
                    self.input_marginals.astype(np.float64),
                    n_firings,
                    self.total_tokens_processed,
                    pmi_top_k_tokens,
                ),
                output_token_pmi=_compute_token_pmi(
                    self.output_cooccurrence[flat_idx],
                    self.output_marginals,
                    n_firings,
                    self.total_tokens_processed,
                    pmi_top_k_tokens,
                ),
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_base_rate_summary(
    firing_counts: Int[np.ndarray, " C"], input_marginals: Int[np.ndarray, " vocab"]
) -> None:
    active_counts = firing_counts[firing_counts > 0]
    if len(active_counts) == 0:
        logger.info("  WARNING: No components fired above threshold!")
        return

    sorted_counts = np.sort(active_counts)
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
    sorted_token_counts = np.sort(active_tokens)
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
    token_mass_for_component: Float[np.ndarray, " vocab"],
    token_mass_totals: Float[np.ndarray, " vocab"],
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
