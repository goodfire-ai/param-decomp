"""Target-owned output-space reconstruction metrics.

Each metric has two spellings, dispatched by the consumer on the LM output edge's closed
`LMOutput` union (`targets.lm_output`): the materialized kernels take logits, the
`streamed_*` twins take the factored package and stream over its head's vocab axis in
chunks, so the `[*leading, vocab]` logits never materialize. Each chunk's logits are
fp32-ACCUMULATED from the native-dtype operands (`_chunk_logits`): the fp32 chunk
materializes either way, so the accumulation costs nothing. The materialized edge's
logits are the model's native-dtype matmul output (bf16 on a bf16 target — an fp32
full-vocab buffer would double), cast to fp32 at the kernel, so the two edges differ by
exactly one bf16 rounding of the logits, on the materialized side; the KL floor that
rounding sets between two otherwise-identical logit sets sits on the materialized edge
only. Everything past the logits is fp32 on both edges, differing by reassociation.

STREAMED RECURRENCES (fp32 accumulators over vocab chunks; `z = activations @ head.T`).
Online softmax for one stream keeps a running max `m` and rescaled sum-exp `s`:

    m' = max(m, max_chunk(z));   s' = s·e^{m−m'} + Σ_chunk e^{z−m'}
    ⇒ logZ = m + log s

KL(p ‖ q) between two streamed logit sets z_p (clean) and z_q (masked) needs one more
accumulator — the p-weighted cross term `a`, rescaled by p's own max like `s_p`:

    a' = a·e^{m_p−m_p'} + Σ_chunk e^{z_p−m_p'}·(z_p − z_q)
    ⇒ KL = Σ_v p_v(z_pv − z_qv) − logZ_p + logZ_q = a/s_p − (m_p + log s_p) + (m_q + log s_q)

CE at a label accumulates the label's logit alongside `m`/`s` (each label lands in
exactly one chunk):  CE = (m + log s) − z_label.

The gradients need no custom VJP: each scan body is `jax.checkpoint`ed, so the backward
re-derives one chunk's logits at a time from the saved activations and flows the exact
softmax cotangent (∂KL/∂z_q = q − p, ∂CE/∂z = softmax(z) − onehot) through the same
placed matmul — contributions through the running max cancel algebraically because the
results are invariant to `m`.
"""

import math
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Float32, Int, jaxtyped

from param_decomp.targets.lm_output import LMOutput, StreamedLinearOutput


@jaxtyped(typechecker=beartype)
def kl_per_position(
    masked_output: Float[Array, "*leading vocab"], clean_output: Float[Array, "*leading vocab"]
) -> Float32[Array, ""]:
    """Mean `KL(softmax(clean) ‖ softmax(masked))` over every leading position."""
    masked_output = masked_output.astype(jnp.float32)
    clean_output = clean_output.astype(jnp.float32)
    log_q = jax.nn.log_softmax(masked_output, axis=-1)
    log_p = jax.nn.log_softmax(clean_output, axis=-1)
    p = jnp.exp(log_p)
    return jnp.sum(p * (log_p - log_q)) / math.prod(masked_output.shape[:-1])


@jaxtyped(typechecker=beartype)
def _head_chunks(
    output: StreamedLinearOutput,
) -> Float[
    Array,
    "{output.n_chunks} {output.head.shape[0]//output.n_chunks} {output.head.shape[1]}",
]:
    d_out, d_model = output.head.shape
    assert d_out % output.n_chunks == 0, (d_out, output.n_chunks)
    return output.head.reshape(output.n_chunks, d_out // output.n_chunks, d_model)


@jaxtyped(typechecker=beartype)
def _chunk_logits(
    activations: Float[Array, "*leading d_model"],
    head_chunk: Float[Array, "chunk d_model"],
) -> Float32[Array, "*leading chunk"]:
    """One vocab chunk's logits, fp32-accumulated from the native-dtype operands."""
    return jnp.dot(activations, head_chunk.T, preferred_element_type=jnp.float32)


@jaxtyped(typechecker=beartype)
def _accumulator_like(
    activations: Float[Array, "*leading d_model"],
) -> Float32[Array, "*leading"]:
    """A zeroed fp32 per-position accumulator inheriting the activations' leading
    (batch) sharding — the scan carry must enter with the type its body maintains."""
    return jnp.zeros_like(activations[..., 0], dtype=jnp.float32)


@jaxtyped(typechecker=beartype)
class _KLAccumulator(eqx.Module):
    clean_max: Float32[Array, "*leading"]
    clean_sum_exp: Float32[Array, "*leading"]
    cross: Float32[Array, "*leading"]
    masked_max: Float32[Array, "*leading"]
    masked_sum_exp: Float32[Array, "*leading"]


@jaxtyped(typechecker=beartype)
def streamed_position_kl(
    masked: StreamedLinearOutput, clean: StreamedLinearOutput
) -> Float32[Array, "*leading"]:
    """Per-position `KL(softmax(clean) ‖ softmax(masked))`, streamed over vocab chunks
    (the paired-accumulator recurrence in the module docstring)."""
    assert masked.n_chunks == clean.n_chunks, (masked.n_chunks, clean.n_chunks)
    assert masked.head.shape == clean.head.shape, (masked.head.shape, clean.head.shape)
    assert masked.activations.shape == clean.activations.shape, (
        masked.activations.shape,
        clean.activations.shape,
    )
    zeros = _accumulator_like(masked.activations)
    neg_inf = jnp.full_like(zeros, -jnp.inf)

    @jax.checkpoint  # the backward re-derives one chunk's logits at a time
    @jaxtyped(typechecker=beartype)
    def chunk_step(
        carry: _KLAccumulator,
        heads: tuple[Float[Array, "chunk d_model"], Float[Array, "chunk d_model"]],
    ) -> tuple[_KLAccumulator, None]:
        clean_head, masked_head = heads
        z_p = _chunk_logits(clean.activations, clean_head)
        z_q = _chunk_logits(masked.activations, masked_head)
        m_p_next = jnp.maximum(carry.clean_max, jnp.max(z_p, axis=-1))
        m_q_next = jnp.maximum(carry.masked_max, jnp.max(z_q, axis=-1))
        rescale_p = jnp.exp(carry.clean_max - m_p_next)
        exp_p = jnp.exp(z_p - m_p_next[..., None])
        s_p = carry.clean_sum_exp * rescale_p + jnp.sum(exp_p, axis=-1)
        cross = carry.cross * rescale_p + jnp.sum(exp_p * (z_p - z_q), axis=-1)
        s_q = carry.masked_sum_exp * jnp.exp(carry.masked_max - m_q_next) + jnp.sum(
            jnp.exp(z_q - m_q_next[..., None]), axis=-1
        )
        return _KLAccumulator(
            clean_max=m_p_next,
            clean_sum_exp=s_p,
            cross=cross,
            masked_max=m_q_next,
            masked_sum_exp=s_q,
        ), None

    final, _ = jax.lax.scan(
        chunk_step,
        _KLAccumulator(
            clean_max=neg_inf,
            clean_sum_exp=zeros,
            cross=zeros,
            masked_max=neg_inf,
            masked_sum_exp=zeros,
        ),
        (_head_chunks(clean), _head_chunks(masked)),
    )
    position_kl = (
        final.cross / final.clean_sum_exp
        - (final.clean_max + jnp.log(final.clean_sum_exp))
        + (final.masked_max + jnp.log(final.masked_sum_exp))
    )
    assert position_kl.shape == masked.activations.shape[:-1], (
        position_kl.shape,
        masked.activations.shape,
    )
    return position_kl


@jaxtyped(typechecker=beartype)
def streamed_kl_per_position(
    masked: StreamedLinearOutput, clean: StreamedLinearOutput
) -> Float32[Array, ""]:
    """`kl_per_position` on the streamed edge — the same mean over every position."""
    return streamed_position_kl(masked, clean).mean()


@jaxtyped(typechecker=beartype)
class _CEAccumulator(eqx.Module):
    max_logit: Float32[Array, "*leading"]
    sum_exp: Float32[Array, "*leading"]
    label_logit: Float32[Array, "*leading"]


@jaxtyped(typechecker=beartype)
def streamed_position_ce(
    output: StreamedLinearOutput, labels: Int[Array, "*leading"]
) -> Float32[Array, "*leading"]:
    """Per-position `−log softmax(logits)[label]`, streamed over vocab chunks.

    A label outside `[0, vocab)` lands in no chunk and yields NaN — never a finite CE
    (it would be logZ). The materialized twin's `take_along_axis` fills NaN for an
    overflowing label but wraps a negative one, so this spelling is the stricter of the
    two. The loud refusal lives where labels enter the process
    (`lm.batch_data.HostLMBatch.validate_vocabulary`): the repo has no in-jit
    assertion idiom, and a host callback per eval chunk would cost more than the check."""
    assert output.activations.shape[:-1] == labels.shape, (output.activations.shape, labels.shape)
    head_chunks = _head_chunks(output)
    chunk = head_chunks.shape[1]
    zeros = _accumulator_like(output.activations)
    init = _CEAccumulator(
        max_logit=jnp.full_like(zeros, -jnp.inf), sum_exp=zeros, label_logit=zeros
    )

    @jax.checkpoint  # the backward re-derives one chunk's logits at a time
    @jaxtyped(typechecker=beartype)
    def chunk_step(
        carry: _CEAccumulator, inputs: tuple[Float[Array, "chunk d_model"], Int[Array, ""]]
    ) -> tuple[_CEAccumulator, None]:
        head_chunk, start = inputs
        z = _chunk_logits(output.activations, head_chunk)
        m_next = jnp.maximum(carry.max_logit, jnp.max(z, axis=-1))
        s = carry.sum_exp * jnp.exp(carry.max_logit - m_next) + jnp.sum(
            jnp.exp(z - m_next[..., None]), axis=-1
        )
        local = labels - start
        in_chunk = (local >= 0) & (local < chunk)
        picked = jnp.take_along_axis(z, jnp.clip(local, 0, chunk - 1)[..., None], axis=-1)[..., 0]
        z_label = carry.label_logit + jnp.where(in_chunk, picked, 0.0)
        return _CEAccumulator(max_logit=m_next, sum_exp=s, label_logit=z_label), None

    final, _ = jax.lax.scan(chunk_step, init, (head_chunks, jnp.arange(output.n_chunks) * chunk))
    valid_label = (labels >= 0) & (labels < output.head.shape[0])
    return jnp.where(
        valid_label, (final.max_logit + jnp.log(final.sum_exp)) - final.label_logit, jnp.nan
    )


@jaxtyped(typechecker=beartype)
def streamed_position_next_token_ce(
    output: StreamedLinearOutput, token_ids: Int[Array, "B T"]
) -> Float32[Array, "B T-1"]:
    """Per-position CE of positions 0..T−2 predicting tokens 1..T−1 (the materialized
    `next_token_cross_entropy` alignment: labels with the first position ignored)."""
    shifted = replace(output, activations=output.activations[:, :-1])
    return streamed_position_ce(shifted, token_ids[:, 1:])


@jaxtyped(typechecker=beartype)
def dense_position_kl(
    masked_output: Float[Array, "*leading vocab"], clean_output: Float[Array, "*leading vocab"]
) -> Float32[Array, "*leading"]:
    """Per-position `KL(softmax(clean) ‖ softmax(masked))`: `kl_per_position` before its
    mean."""
    log_q = jax.nn.log_softmax(masked_output.astype(jnp.float32), axis=-1)
    log_p = jax.nn.log_softmax(clean_output.astype(jnp.float32), axis=-1)
    return jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1)


@jaxtyped(typechecker=beartype)
def dense_position_next_token_ce(
    output: Float[Array, "B T vocab"], token_ids: Int[Array, "B T"]
) -> Float32[Array, "B T-1"]:
    """Per-position CE of positions 0..T−2 predicting tokens 1..T−1, the materialized twin
    of `streamed_position_next_token_ce`."""
    log_q = jax.nn.log_softmax(output[:, :-1].astype(jnp.float32), axis=-1)
    return -jnp.take_along_axis(log_q, token_ids[:, 1:, None], axis=-1)[..., 0]


@jaxtyped(typechecker=beartype)
def lm_output_position_kl(masked: LMOutput, clean: LMOutput) -> Float32[Array, "*leading"]:
    """`dense_position_kl` in the spelling of the edge both outputs share; a mix of edges is
    refused."""
    match (masked, clean):
        case (StreamedLinearOutput(), StreamedLinearOutput()):
            return streamed_position_kl(masked, clean)
        case (jax.Array(), jax.Array()):
            return dense_position_kl(masked, clean)
        case _:
            raise AssertionError(
                f"mixed model-output edges: {type(masked).__name__} vs {type(clean).__name__}"
            )


@jaxtyped(typechecker=beartype)
def lm_output_position_next_token_ce(
    output: LMOutput, token_ids: Int[Array, "B T"]
) -> Float32[Array, "B T-1"]:
    match output:
        case StreamedLinearOutput():
            return streamed_position_next_token_ce(output, token_ids)
        case jax.Array():
            return dense_position_next_token_ce(output, token_ids)


@jaxtyped(typechecker=beartype)
def lm_output_kl_per_position(masked: LMOutput, clean: LMOutput) -> Float32[Array, ""]:
    """The LM `recon_loss_fn`: `kl_per_position` in the spelling of the edge both outputs
    share; a mix of edges is refused."""
    match (masked, clean):
        case (StreamedLinearOutput(), StreamedLinearOutput()):
            return streamed_kl_per_position(masked, clean)
        case (jax.Array(), jax.Array()):
            return kl_per_position(masked, clean)
        case _:
            raise AssertionError(
                f"mixed model-output edges: {type(masked).__name__} vs {type(clean).__name__}"
            )
