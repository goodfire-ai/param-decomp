"""JAX-native attention-pattern reconstruction eval metrics
(`CIMaskedAttnPatternsReconLoss`, `StochasticAttnPatternsReconLoss`), the in-loop
counterparts of the torch eval metrics of the same names
(`param_decomp/eval_metrics/attn_patterns_recon_loss.py`).

Per decomposed attention layer, both compute `KL(target_pattern ‖ masked_pattern)` over
every attention distribution, where a *pattern* is the post-softmax `(B, H, T, T)` causal
attention map. `target_pattern` is built from the clean (frozen `x @ W`) Q/K projections;
`masked_pattern` from the CI-masked (or stochastically-masked) decomposed Q/K. The KL is
torch's `F.kl_div(masked.clamp(1e-12).log(), target, reduction="sum")` per layer, summed
across layers, divided once by `n_distributions = Σ_layers (B · n_heads · T_query)`. Log
keys mirror torch exactly: `"<ClassName>/<q_proj_site>"` per layer plus a combined
`"<ClassName>"` (= Σ sum_kl / Σ n_distributions).

The clean-reference and masked Q/K projections use the same target-owned site-output
points. One clean plan unions Q/K with the CI inputs; the scored pass requests those Q/K
points from `masked_forward`. No attention-map value is captured: production flash
attention never materializes one.

The attention-pattern reproduction is target-specific (RoPE base, GQA, head reshape,
Qwen3's per-layer QK-norm), so it is TARGET-OWNED: the model exposes
`attention_pattern_from_qk(q_site, q_flat, k_flat)` (the `AttnPatternModel` protocol below —
`GLUDecomposedModel` delegates to the layer's own attention module). This file only drives it; nothing here switches on a model family. A target
without the method refuses at step-build time.

Masked and clean Q/K run in COMPUTE_DT (bf16, matching the trained model); the
pattern softmax and the KL reduction are fp32.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from param_decomp.core.jit_util import filter_jit
from param_decomp.core.linear_plan import uniform_like
from param_decomp.core.model import (
    MaterializedMasking,
    PlacedModel,
)


@runtime_checkable
class AttnPatternModel(Protocol):
    """The capability this eval needs from a target: the post-softmax `(B, H, T, T)`
    causal attention map from one layer's flat Q/K projections, identified by the
    layer's `q_proj` site name (the recipe is the target's own — RoPE flavor, GQA,
    any pre-RoPE math like Qwen3's per-layer QK-norm)."""

    def attention_pattern_from_qk(
        self,
        q_site: str,
        q_flat: Float[Array, "B T qd"],
        k_flat: Float[Array, "B T kvd"],
    ) -> Float[Array, "B H T T"]: ...


def _attn_layer_sites(site_names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """The (q_proj_site, k_proj_site) pairs in canonical order — one per decomposed
    attention layer. Both q and k must be decomposed for a layer to contribute."""
    pairs: list[tuple[str, str]] = []
    for name in site_names:
        if not name.endswith("q_proj"):
            continue
        k_name = name[: -len("q_proj")] + "k_proj"
        assert k_name in site_names, (
            f"attn-patterns needs k_proj alongside {name!r}; {k_name!r} not decomposed"
        )
        pairs.append((name, k_name))
    assert pairs, f"attn-patterns found no decomposed q_proj/k_proj sites in {site_names}"
    return tuple(pairs)


@dataclass(frozen=True)
class LayerKLReduction:
    """Per-attention-layer `(Σ sum_kl, Σ n_distributions)` across the eval pass. Combined
    and per-layer means divide once (torch's accumulate-then-`compute()`)."""

    sum_kl: float
    n_distributions: int


def _pattern_kl(target_pattern: Array, masked_pattern: Array) -> Array:
    """`Σ target · (log target − log masked.clamp(1e-12))` in fp32 (torch
    `F.kl_div(masked.clamp(1e-12).log(), target, reduction="sum")`)."""
    target_pattern = target_pattern.astype(jnp.float32)
    masked_pattern = masked_pattern.astype(jnp.float32)
    log_masked = jnp.log(jnp.clip(masked_pattern, min=1e-12))
    log_target = jnp.log(jnp.clip(target_pattern, min=1e-12))
    return jnp.sum(target_pattern * (log_target - log_masked))


AttnPatternsStep = Callable[
    [
        PlacedModel,
        Any,
        Int[Array, "*leading"],
        dict[str, Array],
        dict[str, Array],
        PRNGKeyArray,
    ],
    tuple[dict[str, Array], dict[str, int]],
]
"""`(model, prepared_weights, tokens, ci_lower, clean_site_outputs_by_site, key) ->
({q_site: sum_kl}, {q_site: n_dists})` — one batch's per-layer summed KL (fp32) and
distribution counts. The clean-side Q/K values come from the pass's shared batch context;
only the masked forward runs here. `key` is unused by the deterministic CI step. `model`
(frozen-weight-bearing) is the jit ARG."""


def attn_output_key_by_site(model_static: PlacedModel) -> dict[str, str]:
    """The decomposed q/k sites' canonical output capture keys — the clean-capture demand
    this eval declares on the shared batch context."""
    layer_pairs = _attn_layer_sites(model_static.site_names)
    requested_sites = tuple(site for pair in layer_pairs for site in pair)
    return dict(zip(requested_sites, model_static.site_output_keys(requested_sites), strict=True))


def _attn_pattern_model(model: PlacedModel) -> AttnPatternModel:
    """Narrow the bundle's target to the pattern-capable surface — re-derived from the
    traced model arg each call, never closed over (the HLO-baking rule)."""
    inner = model.model
    assert isinstance(inner, AttnPatternModel)
    return inner


def _attention_patterns(
    model: PlacedModel,
    layer_pairs: tuple[tuple[str, str], ...],
    site_outputs: dict[str, Array],
) -> dict[str, Array]:
    """Per-layer target-owned attention pattern derived from captured Q/K outputs."""
    pattern_model = _attn_pattern_model(model)
    return {
        q: pattern_model.attention_pattern_from_qk(q, site_outputs[q], site_outputs[k])
        for q, k in layer_pairs
    }


def _attention_pattern_kl_by_layer(
    model: PlacedModel,
    layer_pairs: tuple[tuple[str, str], ...],
    masked_outputs: dict[str, Array],
    target_patterns: dict[str, Array],
) -> dict[str, Array]:
    pattern_model = _attn_pattern_model(model)
    return {
        q: _pattern_kl(
            target_patterns[q],
            pattern_model.attention_pattern_from_qk(q, masked_outputs[q], masked_outputs[k]),
        )
        for q, k in layer_pairs
    }


def _assert_position_axis(model_static: PlacedModel) -> None:
    """Attention patterns are `(B, H, T_query, T_key)` causal maps over the position
    axis; the metric only applies to a positioned LM target (the `AttnPatternModel`
    capability assert in the step factories rejects non-attention targets)."""
    assert model_static.has_position_axis, (
        "attn-patterns eval is LM-only (causal attention over the position axis)"
    )


def make_ci_attn_patterns_step(
    model_static: PlacedModel,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> AttnPatternsStep:
    """Deterministic CI-mask attention-pattern step: one masked forward over the shared
    context's clean-side Q/K values."""
    _assert_position_axis(model_static)
    assert isinstance(model_static.model, AttnPatternModel), (
        f"attn-patterns eval needs a target exposing attention_pattern_from_qk; {type(model_static.model).__name__} does not"
    )
    layer_pairs = _attn_layer_sites(model_static.site_names)
    output_key_by_site = attn_output_key_by_site(model_static)
    site_output_keys = tuple(output_key_by_site.values())

    def step(
        model: PlacedModel,
        prepared_weights: Any,
        tokens: Int[Array, "*leading"],
        ci_lower: dict[str, Array],
        clean_site_outputs_by_site: dict[str, Array],
        _key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, int]]:
        target_patterns = _attention_patterns(model, layer_pairs, clean_site_outputs_by_site)
        masked_captures_by_key = model.masked_forward(
            prepared_weights,
            tokens,
            masking=MaterializedMasking(component_masks=ci_lower),
            capture_keys=frozenset(site_output_keys),
            remat=False,
        ).captures
        masked_site_outputs_by_site = {
            site: masked_captures_by_key[key] for site, key in output_key_by_site.items()
        }
        sum_kl = _attention_pattern_kl_by_layer(
            model, layer_pairs, masked_site_outputs_by_site, target_patterns
        )
        n_distributions = {q: int(np.prod(target_patterns[q].shape[:3])) for q, _ in layer_pairs}
        return sum_kl, n_distributions

    return filter_jit(step, compiler_options=compiler_options)


def make_stochastic_attn_patterns_step(
    model_static: PlacedModel,
    n_mask_samples: int,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> AttnPatternsStep:
    """Stochastic-mask attn-patterns step: `n_mask_samples` draws of `mask = ci + (1−ci)·s`
    (with weight deltas) over the shared context's clean-side Q/K values, per-draw
    per-layer pattern KL summed. Per-draw and per-site `fold_in` keeps each random stream
    independent and reproducible."""
    _assert_position_axis(model_static)
    assert isinstance(model_static.model, AttnPatternModel), (
        f"attn-patterns eval needs a target exposing attention_pattern_from_qk; {type(model_static.model).__name__} does not"
    )
    assert n_mask_samples >= 1, n_mask_samples
    site_names = model_static.site_names
    layer_pairs = _attn_layer_sites(site_names)
    output_key_by_site = attn_output_key_by_site(model_static)
    site_output_keys = tuple(output_key_by_site.values())

    def step(
        model: PlacedModel,
        prepared_weights: Any,
        tokens: Int[Array, "*leading"],
        ci_lower: dict[str, Array],
        clean_site_outputs_by_site: dict[str, Array],
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, int]]:
        target_patterns = _attention_patterns(model, layer_pairs, clean_site_outputs_by_site)

        sum_kl = {q: jnp.zeros((), jnp.float32) for q, _ in layer_pairs}
        for draw_idx in range(n_mask_samples):
            mask_key, delta_key = random.split(random.fold_in(key, draw_idx))
            masks = {}
            delta_masks = {}
            # `uniform_like`, never a bare draw: a bare `random.uniform` lowers REPLICATED
            # under the Explicit mesh and the per-kind mask stacks then hold the full
            # eval batch on every rank (value-identical either way — threefry, SPEC D4).
            for site_idx, site in enumerate(site_names):
                ci_site = ci_lower[site]
                source_key = random.fold_in(mask_key, site_idx)
                source = uniform_like(source_key, ci_site)
                masks[site] = ci_site + (1.0 - ci_site) * source
                delta_masks[site] = uniform_like(
                    random.fold_in(delta_key, site_idx), ci_site, drop_last_axis=True
                )
            masked_captures_by_key = model.masked_forward(
                prepared_weights,
                tokens,
                masking=MaterializedMasking(component_masks=masks, weight_delta_masks=delta_masks),
                capture_keys=frozenset(site_output_keys),
                remat=False,
            ).captures
            masked_site_outputs_by_site = {
                site: masked_captures_by_key[key] for site, key in output_key_by_site.items()
            }
            draw_kl = _attention_pattern_kl_by_layer(
                model, layer_pairs, masked_site_outputs_by_site, target_patterns
            )
            sum_kl = {q: sum_kl[q] + draw_kl[q] for q, _ in layer_pairs}

        n_distributions = {
            q: int(np.prod(target_patterns[q].shape[:3])) * n_mask_samples for q, _ in layer_pairs
        }
        return sum_kl, n_distributions

    return filter_jit(step, compiler_options=compiler_options)


def fold_layer_kl(
    accumulated: dict[str, LayerKLReduction],
    batch_sum: dict[str, Array],
    batch_n: dict[str, int],
) -> dict[str, LayerKLReduction]:
    """Fold one batch's `(Σ sum_kl, Σ n)` per layer into the pass accumulator."""
    return {
        site: LayerKLReduction(
            sum_kl=(accumulated[site].sum_kl if site in accumulated else 0.0)
            + float(np.asarray(batch_sum[site])),
            n_distributions=(accumulated[site].n_distributions if site in accumulated else 0)
            + int(batch_n[site]),
        )
        for site in batch_sum
    }


def attn_patterns_log_entries(
    class_name: str, reductions: dict[str, LayerKLReduction]
) -> dict[str, float]:
    """`{class_name/q_proj_site: kl}` per layer plus a combined `{class_name}`: per-layer is
    `sum_kl/n`, combined is `Σ sum_kl / Σ n` (torch `compute_per_module_metrics`)."""
    assert reductions, "no attn-patterns data accumulated"
    log_entries = {
        f"{class_name}/{site}": reduction.sum_kl / reduction.n_distributions
        for site, reduction in reductions.items()
    }
    total_sum = sum(r.sum_kl for r in reductions.values())
    total_n = sum(r.n_distributions for r in reductions.values())
    log_entries[class_name] = total_sum / total_n
    return log_entries
