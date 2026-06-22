"""The pure loss terms (SPEC §2) and their schedules — fp32 reductions, no state."""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from param_decomp.configs import ImportanceMinimalityLossConfig


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


@jaxtyped(typechecker=beartype)
def importance_minimality_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]], pnorm: Float[Array, ""], eps: float
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """`(lp, entropy)` with per-site grouping and the global-batch sum inside the log2
    (SPEC S7/S8); the loss is `lp + beta * entropy`. Torch's `_no_beta` diagnostic (`lp`
    alone) is emitted only on the eval path (`Metric.compute`), never from the train
    step, so this trainer does not log a `train/loss/*_no_beta` key.

    Under GSPMD the `*leading` axes are the global batch, so `jnp.sum` IS the exact
    global per-component sum — XLA reduces across shards inside the graph."""
    lp = jnp.zeros((), jnp.float32)
    entropy = jnp.zeros((), jnp.float32)
    for ci in ci_upper.values():
        ci = ci.astype(jnp.float32)  # (*leading, C)
        leading_axes = tuple(range(ci.ndim - 1))
        n_positions = math.prod(ci.shape[:-1])
        per_component_sums = jnp.sum((ci + eps) ** pnorm, axis=leading_axes)  # (C,)
        per_component_means = per_component_sums / n_positions
        lp = lp + jnp.sum(per_component_means)
        entropy = entropy + jnp.sum(per_component_means * jnp.log2(1.0 + per_component_sums))
    return lp, entropy


def annealed_pnorm(step_f32: Array, total_steps: int, cfg: ImportanceMinimalityLossConfig) -> Array:
    """`p` anneals linearly `pnorm → p_anneal_final_p` over
    `[p_anneal_start_frac, p_anneal_end_frac]` of training (SPEC S9)."""
    assert cfg.p_anneal_final_p is not None
    span = max(cfg.p_anneal_end_frac - cfg.p_anneal_start_frac, 1e-9)
    progress = jnp.clip((step_f32 / total_steps - cfg.p_anneal_start_frac) / span, 0.0, 1.0)
    return jnp.asarray(cfg.pnorm + (cfg.p_anneal_final_p - cfg.pnorm) * progress)


def warmup_then_constant_lr(
    step_f32: Array, total_steps: int, lr: float, warmup_frac: float
) -> Array:
    warmup_steps = jnp.maximum(jnp.floor(total_steps * warmup_frac), 1.0)
    return jnp.where(step_f32 < warmup_steps, lr * step_f32 / warmup_steps, lr)
