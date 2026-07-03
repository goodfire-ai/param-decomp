"""The pure loss terms (SPEC §2) and their schedules — fp32 reductions, no state."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from param_decomp.configs import (
    AnyImportanceMinimalityLossConfig,
    ImportanceMinimalityLossConfig,
    SmoothL0ImportanceMinimalityLossConfig,
)


@jaxtyped(typechecker=beartype)
def kl_per_position(
    masked_output: Float[Array, "*leading vocab"], clean_output: Float[Array, "*leading vocab"]
) -> Float[Array, ""]:
    """`Σ_{*leading} KL(softmax(clean) ‖ softmax(masked)) / Π(*leading)` in fp32
    (SPEC §2.3, N3); `*leading` is every axis but the final logit axis."""
    masked_output = masked_output.astype(jnp.float32)
    clean_output = clean_output.astype(jnp.float32)
    log_q = jax.nn.log_softmax(masked_output, axis=-1)
    log_p = jax.nn.log_softmax(clean_output, axis=-1)
    p = jnp.exp(log_p)
    n_positions = math.prod(masked_output.shape[:-1])
    return jnp.sum(p * (log_p - log_q)) / n_positions


@jaxtyped(typechecker=beartype)
def faithfulness_loss(weight_deltas: dict[str, Float[Array, "_ _"]]) -> Float[Array, ""]:
    """`Σ_s ‖Δ_s‖² / Σ_s numel` over fp32 deltas (SPEC S17). Each `Δ_s` is `(d_out, d_in)`;
    dims are per-site (anonymous, not bound across sites)."""
    numerator = sum(
        ((delta.astype(jnp.float32) ** 2).sum() for delta in weight_deltas.values()),
        start=jnp.zeros((), jnp.float32),
    )
    # float, not int: the full-model param total (Σ d_in·d_out ≈ 7e9) overflows the int32
    # that jax materializes a Python int into under jit. A float normalizer is exact here.
    denominator = float(sum(delta.size for delta in weight_deltas.values()))
    return numerator / denominator


def _imp_min_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]],
    per_value_penalty: Callable[[Float[Array, "*leading _"]], Float[Array, "*leading _"]],
    reference_token_count: int | None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """`(lp, freq)` for any per-value penalty `psi`, with per-site grouping (SPEC S7/S8):

    - `lp = Σ_s Σ_c f_c`, the bare per-component mean firing rate `f_c = (Σ_{b,t} psi(c)) / B·T`.
    - `freq = Σ_s Σ_c f_c · log2(1 + a' · f_c)`, the batch-invariant frequency penalty with
      `a' = reference_token_count`; `0.0` when `reference_token_count is None`.

    The two imp-min penalties (`L_p`, smooth-L0) differ ONLY in `psi`. Under GSPMD the
    `*leading` axes are the global batch, so `jnp.sum` IS the exact global per-component
    sum — XLA reduces across shards inside the graph, so `f_c` is the true full-batch
    frequency inside the convex `log2` (a per-shard `f_c` would give a Jensen bias)."""
    lp = jnp.zeros((), jnp.float32)
    freq = jnp.zeros((), jnp.float32)
    for ci in ci_upper.values():
        ci = ci.astype(jnp.float32)  # (*leading, C)
        leading_axes = tuple(range(ci.ndim - 1))
        n_positions = math.prod(ci.shape[:-1])
        per_component_sums = jnp.sum(per_value_penalty(ci), axis=leading_axes)  # (C,)
        per_component_means = per_component_sums / n_positions  # f_c
        lp = lp + jnp.sum(per_component_means)
        if reference_token_count is not None:
            freq = freq + jnp.sum(
                per_component_means * jnp.log2(1.0 + reference_token_count * per_component_means)
            )
    return lp, freq


@jaxtyped(typechecker=beartype)
def importance_minimality_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]],
    pnorm: Float[Array, ""],
    eps: float,
    reference_token_count: int | None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """`L_p` imp-min terms: per-value penalty `(c + eps)^pnorm`, singular at `c=0` for
    `pnorm < 1` (the `eps` floor caps the gradient there)."""
    return _imp_min_terms(ci_upper, lambda ci: (ci + eps) ** pnorm, reference_token_count)


@jaxtyped(typechecker=beartype)
def smooth_l0_importance_minimality_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]],
    gamma: Float[Array, ""],
    reference_token_count: int | None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Geman–McClure smooth-L0 imp-min terms: per-value penalty `c^2 / (c^2 + gamma^2)`.
    Flat at the origin (`phi'(0)=0`) and bounded (`|phi'| <= 0.65/gamma`) — no singularity,
    no `eps` floor. Approaches the true `L_0` count as `gamma -> 0`."""
    gamma_sq = gamma * gamma
    return _imp_min_terms(ci_upper, lambda ci: ci**2 / (ci**2 + gamma_sq), reference_token_count)


def _linear_anneal(
    step_f32: Array,
    total_steps: int,
    initial: float,
    final: float,
    start_frac: float,
    end_frac: float,
) -> Array:
    span = max(end_frac - start_frac, 1e-9)
    progress = jnp.clip((step_f32 / total_steps - start_frac) / span, 0.0, 1.0)
    return jnp.asarray(initial + (final - initial) * progress)


def annealed_pnorm(step_f32: Array, total_steps: int, cfg: ImportanceMinimalityLossConfig) -> Array:
    """`p` anneals linearly `pnorm → p_anneal_final_p` over
    `[p_anneal_start_frac, p_anneal_end_frac]` of training (SPEC S9)."""
    assert cfg.p_anneal_final_p is not None
    return _linear_anneal(
        step_f32,
        total_steps,
        cfg.pnorm,
        cfg.p_anneal_final_p,
        cfg.p_anneal_start_frac,
        cfg.p_anneal_end_frac,
    )


def annealed_gamma(
    step_f32: Array, total_steps: int, cfg: SmoothL0ImportanceMinimalityLossConfig
) -> Array:
    """`gamma` anneals linearly `gamma → gamma_anneal_final_gamma` over
    `[gamma_anneal_start_frac, gamma_anneal_end_frac]` of training (SPEC S9', the smooth-L0
    analogue of S9)."""
    assert cfg.gamma_anneal_final_gamma is not None
    return _linear_anneal(
        step_f32,
        total_steps,
        cfg.gamma,
        cfg.gamma_anneal_final_gamma,
        cfg.gamma_anneal_start_frac,
        cfg.gamma_anneal_end_frac,
    )


def annealed_imp_min_param(
    step_f32: Array, total_steps: int, cfg: AnyImportanceMinimalityLossConfig
) -> Array:
    """The annealed per-value-penalty parameter at this step (`p` for `L_p`, `gamma` for
    smooth-L0). Pure in the step, so the train step hoists it out of the loss `grad`."""
    match cfg:
        case ImportanceMinimalityLossConfig():
            return annealed_pnorm(step_f32, total_steps, cfg)
        case SmoothL0ImportanceMinimalityLossConfig():
            return annealed_gamma(step_f32, total_steps, cfg)


def imp_min_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]],
    cfg: AnyImportanceMinimalityLossConfig,
    annealed_param: Array,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Dispatch `(lp, freq)` on the imp-min penalty kind, given its annealed parameter; the
    `freq` term is `0.0` unless `cfg.frequency` is configured."""
    ref = cfg.frequency.reference_token_count if cfg.frequency is not None else None
    match cfg:
        case ImportanceMinimalityLossConfig():
            return importance_minimality_terms(ci_upper, annealed_param, cfg.eps, ref)
        case SmoothL0ImportanceMinimalityLossConfig():
            return smooth_l0_importance_minimality_terms(ci_upper, annealed_param, ref)


def warmup_then_constant_lr(
    step_f32: Array, total_steps: int, lr: float, warmup_frac: float
) -> Array:
    warmup_steps = jnp.maximum(jnp.floor(total_steps * warmup_frac), 1.0)
    return jnp.where(step_f32 < warmup_steps, lr * step_f32 / warmup_steps, lr)
