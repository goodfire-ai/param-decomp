"""The pure loss terms (SPEC §2) and their schedules — fp32 reductions, no state."""

import jax
import jax.numpy as jnp
from jaxtyping import Array

from param_decomp_config.losses import ImportanceMinimalityLossConfig


def kl_per_position(masked_logits: Array, clean_logits: Array) -> Array:
    """`Σ_{b,t} KL(softmax(clean) ‖ softmax(masked)) / (B·T)` in fp32 (SPEC §2.3, N3)."""
    masked_logits = masked_logits.astype(jnp.float32)
    clean_logits = clean_logits.astype(jnp.float32)
    log_q = jax.nn.log_softmax(masked_logits, axis=-1)
    log_p = jax.nn.log_softmax(clean_logits, axis=-1)
    p = jnp.exp(log_p)
    n_positions = masked_logits.shape[0] * masked_logits.shape[1]
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
    (SPEC S7/S8); the loss is `lp + beta * entropy` and `lp` alone is the torch
    `ImportanceMinimalityLoss_no_beta` metric.

    Under GSPMD the `(b, t)` axes are the global batch, so `jnp.sum` IS the exact
    global per-component sum — XLA reduces across shards inside the graph.

    DECISIVE PRECISION TEST (diagnostic, revert after): compute the Lp penalty in bf16 to
    match torch-`main`, which never casts `ci_upper_leaky` to fp32 under autocast. The
    fp32 path (`ci.astype(float32)` + the `(ci+eps)**p` power in fp32) is the suspected
    cause of the low-p CI-fn gradient runaway after ~77.5% of training; torch's bf16
    `(ci+eps)**p` is the (accidentally stabilising) baseline behaviour."""
    lp = jnp.zeros((), jnp.float32)
    entropy = jnp.zeros((), jnp.float32)
    pnorm_bf16 = pnorm.astype(jnp.bfloat16)
    for ci in ci_upper.values():
        ci = ci.astype(jnp.bfloat16)  # (B, T, C) — match torch bf16 Lp (was float32)
        n_positions = ci.shape[0] * ci.shape[1]
        per_component_sums = jnp.sum((ci + eps) ** pnorm_bf16, axis=(0, 1))  # (C,) bf16
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
