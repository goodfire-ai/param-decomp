"""JAX-native attention-pattern reconstruction eval metrics
(`CIMaskedAttnPatternsReconLoss`, `StochasticAttnPatternsReconLoss`), the in-loop
counterparts of the torch eval metrics of the same names
(`param_decomp_lab/eval_metrics/attn_patterns_recon_loss.py`).

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

The attention-pattern reproduction is target-specific (RoPE base, GQA, head reshape) and
lives HERE, not on `DecomposedModel`: `attn_pattern_for` dispatches on the concrete frozen
target, reusing that target's OWN RoPE helper and attention config. A non-attention target
(no `FrozenAttn`/`inv_freq`) raises — the metric only applies to attention-bearing targets.

Masked and clean Q/K run in COMPUTE_DT (bf16, matching the trained model); the
pattern softmax and the KL reduction are fp32.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from param_decomp.components import DecompVU
from param_decomp.lm import DecomposedModel, all_false_routes
from param_decomp.targets.llama8b import FrozenAttn, LlamaDecomposedModel
from param_decomp.targets.llama_simple_mlp import SimpleMLPDecomposedModel
from param_decomp.train import COMPUTE_DT, cast_floating
from vendored_jax.llama import apply_rope, repeat_kv, rope_cos_sin

AttnPatternFn = Callable[[Float[Array, "B T qd"], Float[Array, "B T kvd"]], Float[Array, "B H T T"]]
"""`(q_flat, k_flat) -> (B, n_heads, T_query, T_key)` post-softmax causal attention map
from a layer's flat Q/K projections — the per-target attention recipe."""


def _attn_pattern_from_config(
    n_head: int, n_kv_head: int, head_dim: int, n_rep: int, inv_freq: Array
) -> AttnPatternFn:
    """The shared RoPE + GQA + causal-softmax pattern recipe, parameterised by a target's
    attention config. Reuses the vendored `rope_cos_sin`/`apply_rope`/`repeat_kv` — never
    a reimplemented RoPE. Scores in fp32, scaled by `1/√head_dim`, causal-masked, softmaxed."""

    def attn_pattern(q_flat: Array, k_flat: Array) -> Array:
        b, t, _ = q_flat.shape
        assert q_flat.shape[-1] == n_head * head_dim, q_flat.shape
        assert k_flat.shape[-1] == n_kv_head * head_dim, k_flat.shape
        q = q_flat.reshape(b, t, n_head, head_dim).transpose(0, 2, 1, 3)
        k = k_flat.reshape(b, t, n_kv_head, head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, q_flat.dtype)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, n_rep)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
        scores = scores / math.sqrt(head_dim)
        causal = jnp.triu(jnp.ones((t, t), bool), k=1)
        scores = jnp.where(causal, -jnp.inf, scores)
        return jax.nn.softmax(scores, axis=-1)

    return attn_pattern


def _frozen_attn(target: LlamaDecomposedModel | SimpleMLPDecomposedModel) -> FrozenAttn:
    """The first layer's attention — every layer shares the same attn config, so
    one `FrozenAttn` fixes `n_head`/`n_kv_head`/`head_dim`/`n_rep` for the recipe."""
    assert target.layers, "attn-patterns metric needs at least one layer"
    return target.layers[0].attn


def attn_pattern_for(target: Any) -> AttnPatternFn:
    """The per-target attention-pattern recipe, dispatched on the concrete frozen target.

    Reuses the target's OWN attention config (`FrozenAttn`) and RoPE frequencies
    (`inv_freq`). A non-attention target raises — the metric only applies to
    attention-bearing targets (localize-and-assert: fail loudly, never silently)."""
    match target:
        case LlamaDecomposedModel() | SimpleMLPDecomposedModel():
            attn = _frozen_attn(target)
            return _attn_pattern_from_config(
                attn.n_head, attn.n_kv_head, attn.head_dim, attn.n_rep, target.inv_freq
            )
        case _:
            raise AssertionError(
                f"attn-patterns metric only applies to attention targets, got {type(target).__name__}"
            )


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
    pattern_fn: AttnPatternFn,
    layer_pairs: tuple[tuple[str, str], ...],
    components_bf16: DecompVU,
    tokens: Int[Array, "*leading"],
    ci_lower: dict[str, Array],
) -> dict[str, Array]:
    """Per-layer target pattern from the clean (frozen `x @ W`) Q/K — `masked_site_outputs`
    with every site live but routed FALSE everywhere falls onto the frozen path (the same
    seam reuse `hidden_acts_eval` uses for its clean target)."""
    site_names = model.site_names
    leading = tokens.shape
    clean_outputs = model.masked_site_outputs(
        components_bf16, tokens,
        {s: jnp.ones_like(ci_lower[s]) for s in site_names},
        {s: jnp.zeros(leading, COMPUTE_DT) for s in site_names},
        all_false_routes(site_names, leading), site_names, False,
    )  # fmt: skip
    return {q: pattern_fn(clean_outputs[q], clean_outputs[k]) for q, k in layer_pairs}


def _masked_patterns_kl(
    pattern_fn: AttnPatternFn,
    layer_pairs: tuple[tuple[str, str], ...],
    masked_outputs: dict[str, Array],
    target_patterns: dict[str, Array],
) -> dict[str, Array]:
    return {
        q: _pattern_kl(target_patterns[q], pattern_fn(masked_outputs[q], masked_outputs[k]))
        for q, k in layer_pairs
    }


def _assert_attention_sequence_axes(lm: DecomposedModel) -> None:
    """Attention patterns are `(B, H, T_query, T_key)` causal maps over a sequence axis;
    the metric only applies to a sequence-axis LM target (complements the per-target
    `attn_pattern_for` dispatch, which rejects non-attention targets)."""
    assert lm.leading_axes == ("sequence",), (
        f"attn-patterns eval is LM-only (causal attention over a sequence axis); model "
        f"has leading_axes={lm.leading_axes}"
    )


def make_ci_attn_patterns_step(lm: DecomposedModel, pattern_fn: AttnPatternFn) -> AttnPatternsStep:
    """Deterministic CI-mask attn-patterns step: `lower_leaky` CI, no delta, one masked
    forward + one clean (all-false) forward."""
    _assert_attention_sequence_axes(lm)
    site_names = lm.site_names
    layer_pairs = _attn_layer_sites(site_names)

    @eqx.filter_jit
    def step(
        model: DecomposedModel,
        components: DecompVU,
        ci_fn: Any,
        tokens: Int[Array, "*leading"],
        _key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, int]]:
        taps = model.read_activations(tokens, ci_fn.input_names)
        components_bf16 = cast_floating(components, COMPUTE_DT)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        ci_lower = ci_fn_bf16(taps, remat=False).lower

        target_patterns = _clean_patterns(
            model, pattern_fn, layer_pairs, components_bf16, tokens, ci_lower
        )
        leading = tokens.shape
        zeros_delta = {s: jnp.zeros(leading, COMPUTE_DT) for s in site_names}
        masked_outputs = model.masked_site_outputs(
            components_bf16, tokens, ci_lower, zeros_delta, None, site_names, False
        )
        sum_kl = _masked_patterns_kl(pattern_fn, layer_pairs, masked_outputs, target_patterns)
        n_distributions = {q: int(np.prod(target_patterns[q].shape[:3])) for q, _ in layer_pairs}
        return sum_kl, n_distributions

    return step


def make_stochastic_attn_patterns_step(
    lm: DecomposedModel, pattern_fn: AttnPatternFn, n_mask_samples: int
) -> AttnPatternsStep:
    """Stochastic-mask attn-patterns step: `n_mask_samples` draws of `mask = ci + (1−ci)·s`
    (with weight deltas), per-draw per-layer pattern KL summed. RNG via per-draw / per-site
    `fold_in` (the eval-step discipline, mirrors `hidden_acts_eval`)."""
    _assert_attention_sequence_axes(lm)
    assert n_mask_samples >= 1, n_mask_samples
    site_names = lm.site_names
    layer_pairs = _attn_layer_sites(site_names)

    @eqx.filter_jit
    def step(
        model: DecomposedModel,
        components: DecompVU,
        ci_fn: Any,
        tokens: Int[Array, "*leading"],
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, int]]:
        taps = model.read_activations(tokens, ci_fn.input_names)
        components_bf16 = cast_floating(components, COMPUTE_DT)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        ci_lower = ci_fn_bf16(taps, remat=False).lower

        target_patterns = _clean_patterns(
            model, pattern_fn, layer_pairs, components_bf16, tokens, ci_lower
        )
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
                components_bf16, tokens, masks, delta_masks, None, site_names, True
            )
            draw_kl = _masked_patterns_kl(pattern_fn, layer_pairs, masked_outputs, target_patterns)
            sum_kl = {q: sum_kl[q] + draw_kl[q] for q, _ in layer_pairs}

        n_distributions = {
            q: int(np.prod(target_patterns[q].shape[:3])) * n_mask_samples for q, _ in layer_pairs
        }
        return sum_kl, n_distributions

    return step


def accumulate_attn_patterns(
    step: AttnPatternsStep,
    model: DecomposedModel,
    components: DecompVU,
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
