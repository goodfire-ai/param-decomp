"""CI-fn training-health metrics — the big-model instrumentation for the CI transformer.

The CI fn is itself a large trained transformer, but the train step surfaces only its
grad norms. This module adds the standard health telemetry, emitted as scalars under
`ci_health/*` on the fast eval cadence (config-gated by the `CIFnHealth` eval metric;
JAX-only keys, no torch-parity analog):

- `ci_health/weights/*` — per (block, matrix) Frobenius norm, spectral norm sigma_max
  (batched power iteration over the chunk axis — never a full SVD, which is minutes-slow
  at 32 chunks x 4096^2), and stable rank `frob^2 / sigma_max^2` (the "weight rank").
- `ci_health/act/*` — from the instrumented forward (`ChunkwiseTransformerCIFn.telemetry`):
  post-RoPE q/k RMS ("QK norm scales"), attention max/RMS logit + softmax entropy on a
  strided query subsample (logit growth / entropy collapse), in_proj / MLP-hidden /
  post-block residual RMS. Per-chunk scalars reduced over chunks: `attn_logit_max` takes
  the max, `attn_entropy` also logs a worst-chunk `_min`, everything else the mean.
- `ci_health/logits/*` — CI-logit distribution health: mean/std and the fractions in the
  lower-squashing zero-gradient region (`logit > 1`) and leak region (`logit < 0`).
- `ci_health/components/*` — per-site mode-collapse scalars off the mean-CI-per-component
  vector: participation fraction `(sum m)^2 / (C * sum m^2)` (1 = uniform mass, ->1/C =
  one component hogging) and dead fraction (`mean CI < 1e-6`), aggregated over sites.

Chunkwise-transformer only: the MLP CI fns have none of these internals; composition
refuses the metric for them.
"""

from typing import Any

import einops
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from param_decomp.ci_fn import ChunkwiseTransformerCIFn, lower_leaky_hard_sigmoid
from param_decomp.jit_util import filter_jit
from param_decomp.lm import DecomposedModel
from param_decomp.sharding import batch_shard_leading
from param_decomp.train import COMPUTE_DT, cast_floating

SIGMA_MAX_POWER_ITERS = 32
"""Power-iteration count for the spectral-norm estimate. The iterate starts from the
deterministic all-ones vector (RNG-free eval); convergence is geometric in
`(sigma_2 / sigma_1)^2` — slow on the near-degenerate spectra of freshly-initialized
matrices (expect ~1% under-estimate there), plenty once training separates sigma_1. A
health trend scalar, not a precision measurement."""

DEAD_COMPONENT_MEAN_CI = 1e-6
"""A component whose mean lower-squashed CI over the eval batch is below this counts as
dead."""


def _sigma_max(w: Float[Array, "nc m n"]) -> Float[Array, " nc"]:
    """Batched top singular value via power iteration on `W^T W` (fp32). A `fori_loop`,
    not an unrolled python loop: unrolled at 32 iterations x ~50 sharded leaves the flat
    graph carried tens of thousands of ops whose per-op collectives each wanted their own
    registered comm buffers — ~50k VMM allocations, OOM'ing a B200 (job 167430). The loop
    body compiles once and reuses its buffers; `w` is already replicated by the caller so
    the body is collective-free."""
    tiny = jnp.finfo(jnp.float32).tiny
    v0 = jnp.ones((w.shape[0], w.shape[2]), jnp.float32) / w.shape[2] ** 0.5

    def body(_: Array, v: Array) -> Array:
        u = einops.einsum(w, v, "nc m n, nc n -> nc m")
        u = u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + tiny)
        v = einops.einsum(w, u, "nc m n, nc m -> nc n")
        return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + tiny)

    v = jax.lax.fori_loop(0, SIGMA_MAX_POWER_ITERS, body, v0)
    return jnp.linalg.norm(einops.einsum(w, v, "nc m n, nc n -> nc m"), axis=-1)


def _weight_family_stats(w: Float[Array, "nc m n"]) -> tuple[Array, Array, Array]:
    """`(whole-leaf frob, per-chunk sigma_max [nc], per-chunk stable rank [nc])`. The
    fp32 copy is pinned REPLICATED (one all-gather per leaf, a few hundred MB at most) so
    the power iteration runs collective-free; the frobenius reduction happens on the
    sharded master before the gather. No-op off-mesh."""
    w32 = w.astype(jnp.float32)
    per_chunk_frob_sq = jnp.sum(w32**2, axis=(1, 2))
    if not jax.sharding.get_abstract_mesh().empty:
        w32 = jax.lax.with_sharding_constraint(w32, P())
    sigma = _sigma_max(w32)
    stable_rank = per_chunk_frob_sq / jnp.maximum(sigma**2, jnp.finfo(jnp.float32).tiny)
    return jnp.sqrt(per_chunk_frob_sq.sum()), sigma, stable_rank


def ci_fn_weight_health(ci_fn: ChunkwiseTransformerCIFn) -> dict[str, Array]:
    """Weight-space health scalars off the fp32 masters (no batch involved). Per matrix
    family: whole-leaf Frobenius norm (matches the grad-norms convention, so
    update-to-weight ratios overlay), worst-chunk `sigma_max`, chunk-mean stable rank.
    The per-slot output heads aggregate into one `out_heads` family (frob over all slots,
    worst / mean over slots x chunks)."""
    chunks = ci_fn.chunks
    out: dict[str, Array] = {}

    def add(name: str, w: Array) -> None:
        frob, sigma, stable_rank = _weight_family_stats(w)
        out[f"ci_health/weights/frob/{name}"] = frob
        out[f"ci_health/weights/sigma_max/{name}"] = sigma.max()
        out[f"ci_health/weights/stable_rank/{name}"] = stable_rank.mean()

    add("in_proj", chunks.in_proj_w)
    for block_idx, block in enumerate(chunks.blocks):
        for matrix_name in ("wq", "wk", "wv", "wo", "w1", "w2"):
            add(f"block{block_idx}/{matrix_name}", getattr(block, matrix_name))

    slot_stats = [_weight_family_stats(w) for w in chunks.out_ws]
    out["ci_health/weights/frob/out_heads"] = jnp.sqrt(sum(frob**2 for frob, _, _ in slot_stats))
    out["ci_health/weights/sigma_max/out_heads"] = jnp.stack(
        [sigma.max() for _, sigma, _ in slot_stats]
    ).max()
    out["ci_health/weights/stable_rank/out_heads"] = jnp.stack(
        [stable_rank.mean() for _, _, stable_rank in slot_stats]
    ).mean()
    return out


def make_ci_weight_health_fn(compiler_options: dict[str, bool | int | str] | None = None):
    return filter_jit(ci_fn_weight_health, compiler_options=compiler_options)


def _reduce_chunk_stats(chunk_stats: dict[str, Float[Array, " nc"]]) -> dict[str, Array]:
    """Reduce the telemetry's per-chunk scalars over the chunk axis: max for
    `attn_logit_max`, mean + worst-chunk `_min` for `attn_entropy`, mean otherwise."""
    out: dict[str, Array] = {}
    for key, per_chunk in chunk_stats.items():
        if key.endswith("attn_logit_max"):
            out[f"ci_health/act/{key}"] = per_chunk.max()
        elif key.endswith("attn_entropy"):
            out[f"ci_health/act/{key}"] = per_chunk.mean()
            out[f"ci_health/act/{key}_min"] = per_chunk.min()
        else:
            out[f"ci_health/act/{key}"] = per_chunk.mean()
    return out


def _logit_distribution_stats(logits: dict[str, Array]) -> dict[str, Array]:
    """Global CI-logit stats across all sites: mean, std, and the fractions in the lower
    squashing's zero-gradient (`> 1`) and leak (`< 0`) regions."""
    zero = jnp.zeros((), jnp.float32)
    n_total = sum(lg.size for lg in logits.values())
    sum1 = sum((lg.sum() for lg in logits.values()), start=zero)
    sum2 = sum(((lg**2).sum() for lg in logits.values()), start=zero)
    n_above = sum(((lg > 1.0).sum() for lg in logits.values()), start=zero)
    n_below = sum(((lg < 0.0).sum() for lg in logits.values()), start=zero)
    mean = sum1 / n_total
    var = jnp.maximum(sum2 / n_total - mean**2, 0.0)
    return {
        "ci_health/logits/mean": mean,
        "ci_health/logits/std": jnp.sqrt(var),
        "ci_health/logits/frac_above_1": n_above / n_total,
        "ci_health/logits/frac_below_0": n_below / n_total,
    }


def _component_collapse_stats(logits: dict[str, Array]) -> dict[str, Array]:
    """Mode-collapse scalars off the per-site mean-CI-per-component vector `m [C]`:
    participation fraction `(sum m)^2 / (C * sum m^2)` (1 = mass spread uniformly,
    -> 1/C = one component carrying everything; 0 when the site is entirely dead) and
    the dead-component fraction, aggregated over sites."""
    participation = []
    dead = []
    for lg in logits.values():
        c = lg.shape[-1]
        mean_ci = lower_leaky_hard_sigmoid(lg).reshape(-1, c).mean(0)
        mass_sq = (mean_ci**2).sum()
        participation.append(jnp.where(mass_sq > 0.0, mean_ci.sum() ** 2 / (c * mass_sq), 0.0))
        dead.append((mean_ci < DEAD_COMPONENT_MEAN_CI).mean())
    participation_arr = jnp.stack(participation)
    dead_arr = jnp.stack(dead)
    return {
        "ci_health/components/participation_frac_mean": participation_arr.mean(),
        "ci_health/components/participation_frac_min": participation_arr.min(),
        "ci_health/components/dead_frac_mean": dead_arr.mean(),
        "ci_health/components/dead_frac_max": dead_arr.max(),
    }


def make_ci_activation_health_step(
    mesh: Mesh | None,
    compiler_options: dict[str, bool | int | str] | None = None,
):
    """Build the jit'd `step(model, ci_fn, batch) -> {key: scalar}`: one instrumented CI
    forward in training precision (bf16 weights, fp32 stats) emitting the `ci_health/act`,
    `ci_health/logits`, and `ci_health/components` scalar families. `model`
    (frozen-weight-bearing) is the jit ARG."""

    def step(
        model: DecomposedModel, ci_fn: ChunkwiseTransformerCIFn, batch: Any
    ) -> dict[str, Array]:
        batch = batch_shard_leading(batch, mesh)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        taps = {
            k: x.astype(COMPUTE_DT)
            for k, x in model.read_activations(batch, ci_fn.input_names).items()
        }
        logits, chunk_stats = ci_fn_bf16.telemetry(taps)
        logits32 = {site: lg.astype(jnp.float32) for site, lg in logits.items()}
        return (
            _reduce_chunk_stats(chunk_stats)
            | _logit_distribution_stats(logits32)
            | _component_collapse_stats(logits32)
        )

    return filter_jit(step, compiler_options=compiler_options)
