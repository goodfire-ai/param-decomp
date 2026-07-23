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

The clean (target) and masked Q/K are obtained via the existing `masked_site_outputs`
seam — the same all-false-routes trick `hidden_acts_eval` uses for its clean target (one
all-false forward) plus one masked forward — so `DecomposedModel` gains no new method.

The attention-pattern reproduction is target-specific (RoPE base, GQA, head reshape,
Qwen3's per-layer QK-norm), so it is TARGET-OWNED: the model exposes
`attn_pattern(q_site, q_flat, k_flat)` (the `AttnPatternModel` protocol below —
`GLUDecomposedModel` / `SimpleMLPDecomposedModel` delegate to their own attention
modules). This file only drives it; nothing here switches on a model family. A target
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

from param_decomp.core.components import ComponentStacks
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.model import DecomposedModel, all_false_routes
from param_decomp.core.train import COMPUTE_DT, cast_floating


@runtime_checkable
class AttnPatternModel(Protocol):
    """The capability this eval needs from a target: the post-softmax `(B, H, T, T)`
    causal attention map from one layer's flat Q/K projections, identified by the
    layer's `q_proj` site name (the recipe is the target's own — RoPE flavor, GQA,
    any pre-RoPE math like Qwen3's per-layer QK-norm)."""

    def attn_pattern(
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
    [DecomposedModel, Any, Any, Int[Array, "*leading"], PRNGKeyArray],
    tuple[dict[str, Array], dict[str, int]],
]
"""`(model, components, ci_fn, tokens, key) -> ({q_site: sum_kl}, {q_site: n_dists})`
— one batch's per-layer summed KL (fp32) and distribution counts. `key` is unused by the
deterministic CI step. `model` (frozen-weight-bearing) is the jit ARG."""


def _clean_patterns(
    model: DecomposedModel,
    layer_pairs: tuple[tuple[str, str], ...],
    prepared: Any,
    tokens: Int[Array, "*leading"],
    ci_lower: dict[str, Array],
) -> dict[str, Array]:
    """Per-layer target pattern from the clean (frozen `x @ W`) Q/K — `masked_site_outputs`
    with every site live but routed FALSE everywhere falls onto the frozen path (the same
    seam reuse `hidden_acts_eval` uses for its clean target)."""
    site_names = model.site_names
    leading = tokens.shape
    clean_outputs = model.masked_site_outputs(
        prepared, tokens,
        {s: jnp.ones_like(ci_lower[s]) for s in site_names},
        {s: jnp.zeros(leading, COMPUTE_DT) for s in site_names},
        all_false_routes(site_names, leading), site_names, False,
    )  # fmt: skip
    assert isinstance(model, AttnPatternModel)
    return {q: model.attn_pattern(q, clean_outputs[q], clean_outputs[k]) for q, k in layer_pairs}


def _masked_patterns_kl(
    model: DecomposedModel,
    layer_pairs: tuple[tuple[str, str], ...],
    masked_outputs: dict[str, Array],
    target_patterns: dict[str, Array],
) -> dict[str, Array]:
    assert isinstance(model, AttnPatternModel)
    return {
        q: _pattern_kl(
            target_patterns[q], model.attn_pattern(q, masked_outputs[q], masked_outputs[k])
        )
        for q, k in layer_pairs
    }


def _assert_position_axis(model_static: DecomposedModel) -> None:
    """Attention patterns are `(B, H, T_query, T_key)` causal maps over the position
    axis; the metric only applies to a positioned LM target (the `AttnPatternModel`
    capability assert in the step factories rejects non-attention targets)."""
    assert model_static.has_position_axis, (
        "attn-patterns eval is LM-only (causal attention over the position axis)"
    )


def make_ci_attn_patterns_step(
    model_static: DecomposedModel,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> AttnPatternsStep:
    """Deterministic CI-mask attn-patterns step: `lower_leaky` CI, no delta, one masked
    forward + one clean (all-false) forward."""
    _assert_position_axis(model_static)
    assert isinstance(model_static, AttnPatternModel), (
        f"attn-patterns eval needs a target exposing attn_pattern; {type(model_static).__name__} does not"
    )
    site_names = model_static.site_names
    layer_pairs = _attn_layer_sites(site_names)

    def step(
        model: DecomposedModel,
        components: ComponentStacks,
        ci_fn: Any,
        tokens: Int[Array, "*leading"],
        _key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, int]]:
        taps = model.read_activations(tokens, ci_fn.input_names)
        components_bf16 = cast_floating(components, COMPUTE_DT)
        prepared = model.prepare_compute_weights(components_bf16)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        ci_lower = ci_fn_bf16(taps, remat=False).lower

        target_patterns = _clean_patterns(model, layer_pairs, prepared, tokens, ci_lower)
        leading = tokens.shape
        zeros_delta = {s: jnp.zeros(leading, COMPUTE_DT) for s in site_names}
        masked_outputs = model.masked_site_outputs(
            prepared, tokens, ci_lower, zeros_delta, None, site_names, False
        )
        sum_kl = _masked_patterns_kl(model, layer_pairs, masked_outputs, target_patterns)
        n_distributions = {q: int(np.prod(target_patterns[q].shape[:3])) for q, _ in layer_pairs}
        return sum_kl, n_distributions

    return filter_jit(step, compiler_options=compiler_options)


def make_stochastic_attn_patterns_step(
    model_static: DecomposedModel,
    n_mask_samples: int,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> AttnPatternsStep:
    """Stochastic-mask attn-patterns step: `n_mask_samples` draws of `mask = ci + (1−ci)·s`
    (with weight deltas), per-draw per-layer pattern KL summed. RNG via per-draw / per-site
    `fold_in` (the eval-step discipline, mirrors `hidden_acts_eval`)."""
    _assert_position_axis(model_static)
    assert isinstance(model_static, AttnPatternModel), (
        f"attn-patterns eval needs a target exposing attn_pattern; {type(model_static).__name__} does not"
    )
    assert n_mask_samples >= 1, n_mask_samples
    site_names = model_static.site_names
    layer_pairs = _attn_layer_sites(site_names)

    def step(
        model: DecomposedModel,
        components: ComponentStacks,
        ci_fn: Any,
        tokens: Int[Array, "*leading"],
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, int]]:
        taps = model.read_activations(tokens, ci_fn.input_names)
        components_bf16 = cast_floating(components, COMPUTE_DT)
        prepared = model.prepare_compute_weights(components_bf16)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        ci_lower = ci_fn_bf16(taps, remat=False).lower

        target_patterns = _clean_patterns(model, layer_pairs, prepared, tokens, ci_lower)
        leading = tokens.shape

        sum_kl = {q: jnp.zeros((), jnp.float32) for q, _ in layer_pairs}
        for draw_idx in range(n_mask_samples):
            mask_key, delta_key = random.split(random.fold_in(key, draw_idx))
            masks = {}
            delta_masks = {}
            for site_idx, site in enumerate(site_names):
                ci_site = ci_lower[site]
                source_key = random.fold_in(mask_key, site_idx)
                source = random.uniform(source_key, ci_site.shape, COMPUTE_DT)
                masks[site] = ci_site + (1.0 - ci_site) * source
                delta_masks[site] = random.uniform(
                    random.fold_in(delta_key, site_idx), leading, COMPUTE_DT
                )
            masked_outputs = model.masked_site_outputs(
                prepared, tokens, masks, delta_masks, None, site_names, True
            )
            draw_kl = _masked_patterns_kl(model, layer_pairs, masked_outputs, target_patterns)
            sum_kl = {q: sum_kl[q] + draw_kl[q] for q, _ in layer_pairs}

        n_distributions = {
            q: int(np.prod(target_patterns[q].shape[:3])) * n_mask_samples for q, _ in layer_pairs
        }
        return sum_kl, n_distributions

    return filter_jit(step, compiler_options=compiler_options)


def accumulate_attn_patterns(
    step: AttnPatternsStep,
    model: DecomposedModel,
    components: ComponentStacks,
    ci_fn: Any,
    token_batches: list[Int[Array, "*leading"]],
    base_key: PRNGKeyArray,
) -> dict[str, LayerKLReduction]:
    """Drive `step` over the eval batches, host-accumulating `(Σ sum_kl, Σ n)` per layer
    (the density/mean/hidden-acts accumulator pattern). Per-batch RNG is
    `fold_in(base_key, batch_idx)`."""
    assert token_batches, "attn-patterns eval needs at least one batch"
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch_idx, tokens in enumerate(token_batches):
        batch_sum, batch_n = step(
            model, components, ci_fn, tokens, random.fold_in(base_key, batch_idx)
        )
        for site in batch_sum:
            sums[site] = sums.get(site, 0.0) + float(np.asarray(batch_sum[site]))
            counts[site] = counts.get(site, 0) + int(batch_n[site])
    return {s: LayerKLReduction(sum_kl=sums[s], n_distributions=counts[s]) for s in sums}


def attn_patterns_log_entries(
    class_name: str, reductions: dict[str, LayerKLReduction]
) -> dict[str, float]:
    """`{class_name/q_proj_site: kl}` per layer plus a combined `{class_name}`: per-layer is
    `sum_kl/n`, combined is `Σ sum_kl / Σ n` (torch `compute_per_module_metrics`)."""
    assert reductions, "no attn-patterns data accumulated"
    out = {f"{class_name}/{s}": r.sum_kl / r.n_distributions for s, r in reductions.items()}
    total_sum = sum(r.sum_kl for r in reductions.values())
    total_n = sum(r.n_distributions for r in reductions.values())
    out[class_name] = total_sum / total_n
    return out
