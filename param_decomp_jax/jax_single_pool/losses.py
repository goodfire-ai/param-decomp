"""The pure loss terms (SPEC §2) and their schedules — fp32 reductions, no state."""

import jax
import jax.numpy as jnp
from jaxtyping import Array

from param_decomp_config.losses import ImportanceMinimalityLossConfig


def kl_per_position(masked_output: Array, clean_output: Array) -> Array:
    """`Σ_{b,t} KL(softmax(clean) ‖ softmax(masked)) / (B·T)` in fp32 (SPEC §2.3, N3)."""
    masked_output = masked_output.astype(jnp.float32)
    clean_output = clean_output.astype(jnp.float32)
    log_q = jax.nn.log_softmax(masked_output, axis=-1)
    log_p = jax.nn.log_softmax(clean_output, axis=-1)
    p = jnp.exp(log_p)
    n_positions = masked_output.shape[0] * masked_output.shape[1]
    return jnp.sum(p * (log_p - log_q)) / n_positions


def faithfulness_loss(weight_deltas: dict[str, Array]) -> Array:
    """`Σ_s ‖Δ_s‖² / Σ_s numel` over fp32 deltas (SPEC S17)."""
    numerator = sum(
        ((delta.astype(jnp.float32) ** 2).sum() for delta in weight_deltas.values()),
        start=jnp.zeros((), jnp.float32),
    )
    denominator = sum(delta.size for delta in weight_deltas.values())
    return numerator / denominator


def importance_minimality_terms(
    ci_upper: dict[str, Array], pnorm: Array, eps: float
) -> tuple[Array, Array]:
    """`(lp, entropy)` with per-site grouping and the global-batch sum inside the log2
    (SPEC S7/S8); the loss is `lp + beta * entropy`. Torch's `_no_beta` diagnostic (`lp`
    alone) is emitted only on the eval path (`Metric.compute`), never from the train
    step, so this trainer does not log a `train/loss/*_no_beta` key.

    Under GSPMD the `(b, t)` axes are the global batch, so `jnp.sum` IS the exact
    global per-component sum — XLA reduces across shards inside the graph."""
    lp = jnp.zeros((), jnp.float32)
    entropy = jnp.zeros((), jnp.float32)
    for ci in ci_upper.values():
        ci = ci.astype(jnp.float32)  # (B, T, C)
        n_positions = ci.shape[0] * ci.shape[1]
        per_component_sums = jnp.sum((ci + eps) ** pnorm, axis=(0, 1))  # (C,)
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
