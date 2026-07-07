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
from param_decomp.schedule import ScheduleConfig


def scheduled_value_traced(step_f32: Array, total_steps: int, config: ScheduleConfig) -> Array:
    """jnp twin of `schedule.get_scheduled_value` for a traced `step_f32` (inside the
    jitted step, or as an optax schedule over the update count). Same values pointwise
    (`test_schedule.py` pins the pair); only the warmup gate becomes a `jnp.where`. Lives
    here rather than next to its host twin so the config schema stays jax-free.

    The `decay_steps - 1` denominator is canonical-torch: the decay reaches
    `final_val_frac ×` at `step = total_steps - 1` (SPEC S20). Plain
    `optax.cosine_decay_schedule` divides by `decay_steps` and gets there one update
    later — a genuine ~O(1/steps) per-step divergence this fn must avoid."""
    assert total_steps > 0, f"total_steps must be positive, got {total_steps}"

    warmup_steps = int(total_steps * config.warmup_pct)
    decay_steps = total_steps - warmup_steps

    if decay_steps <= 1:
        decayed = jnp.asarray(config.start_val, jnp.float32)
    else:
        progress = (step_f32 - warmup_steps) / (decay_steps - 1)
        match config.fn_type:
            case "constant":
                multiplier = jnp.asarray(1.0, jnp.float32)
            case "linear":
                multiplier = config.final_val_frac + (1 - config.final_val_frac) * (1 - progress)
            case "cosine":
                multiplier = config.final_val_frac + (1 - config.final_val_frac) * 0.5 * (
                    1 + jnp.cos(jnp.pi * progress)
                )
        decayed = config.start_val * multiplier

    if warmup_steps == 0:
        return decayed
    return jnp.where(step_f32 < warmup_steps, config.start_val * (step_f32 / warmup_steps), decayed)


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


def annealed_imp_min_param(
    step_f32: Array, total_steps: int, cfg: AnyImportanceMinimalityLossConfig
) -> Array:
    """The scheduled per-value-penalty parameter at this step (`p` for `L_p`, `gamma` for
    smooth-L0; SPEC S9/S9′). Pure in the step, so the train step hoists it out of the
    loss `grad`."""
    match cfg:
        case ImportanceMinimalityLossConfig():
            schedule = cfg.pnorm
        case SmoothL0ImportanceMinimalityLossConfig():
            schedule = cfg.gamma
    return scheduled_value_traced(step_f32, total_steps, schedule)


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
