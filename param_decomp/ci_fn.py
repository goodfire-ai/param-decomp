"""CI-fn interface + the chunkwise-transformer impl.

A CI fn maps named INPUT taps to a `CI` bundle over OUTPUT sites:
`dict[InputTap, Array] -> CI` (logits + the two squashings). The input keyspace (opaque
tap keys — the lab authors them, the target's `read_activations` produces them) is
independent of the output keyspace (the decomposition sites). The output sites MUST
partition the model's sites — every site needs exactly one CI value — asserted at
construction. Core treats both keyspaces as OPAQUE dict keys: look up inputs, scatter
outputs, validate the partition. It never parses a key.

The SAME logits are squashed two ways (SPEC S5/S6) in ONE place (`CI.from_logits`):
`lower` (clip[0,1], leaky-below) feeds recon / PPGD / routing masks; `upper`
(leaky-above-1) feeds importance-minimality. `logits` is kept too — the CI histograms /
heatmaps plot the pre-squash view. Params are fp32 masters (SPEC N1); the trainer casts
for bf16 compute.

The chunkwise-transformer (`ChunkwiseTransformerCIFn`) is the LM impl: each chunk reads
one or more residual taps (RMS-normed per tap, then concatenated) and emits CI for the
matrix sites it covers, via an independent pre-norm bidirectional-RoPE transformer. The
per-chunk transformers are stacked along a leading `n_chunks` axis and run under a
`jax.lax.scan` over that axis (so the chunk iteration lowers as a loop — one chunk's FSDP
weight gather live at a time, not all `n_chunks` hoisted into the flat entry computation).
The positionless toys use the MLP impls below (`LayerwiseMLPCIFn` /
`GlobalMLPCIFn`); every impl satisfies the same `CIFn` protocol and is equally core — the
architectures differ by domain (sequence vs positionless), not by status.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.components import SiteSpec
from vendored_jax.llama import apply_rope, rms_norm, rope_cos_sin

CI_FN_RMS_EPS = float(jnp.finfo(jnp.float32).eps)
"""Matches torch's `F.rms_norm` default eps (`finfo(fp32).eps` ~1.19e-7); RMS upcasts to
fp32 internally, so this is the dtype that governs (SPEC S4)."""

SiteDict = dict[str, Float[Array, "*leading C"]]
"""Per-output-site tensor keyed by OUTPUT site name."""


# ----------------------------- squashings (SPEC S5/S6) -----------------------------


@jax.custom_vjp
def lower_leaky_hard_sigmoid(x: Array) -> Array:
    return jnp.clip(x, 0.0, 1.0)


def _lhs_f(x: Array) -> tuple[Array, Array]:
    return jnp.clip(x, 0.0, 1.0), x


def _lhs_b(x: Array, g: Array) -> tuple[Array]:
    leak = jnp.where(g < 0, 0.01 * g, 0.0)
    return (jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0)),)


lower_leaky_hard_sigmoid.defvjp(_lhs_f, _lhs_b)


def upper_leaky_hard_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """`x>1 ? 1+alpha*(x-1) : clamp(x,0,1)` — ordinary autodiff of this expression
    (torch builds its backward the same way; only the lower squashing is a custom VJP)."""
    alpha = 0.01
    return jnp.where(x > 1, 1 + alpha * (x - 1), jnp.clip(x, 0.0, 1.0))


# ----------------------------- the CI bundle + protocol -----------------------------


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CI:
    """The CI fn output: raw logits + both squashings, all keyed by output site. `logits`
    is kept (a consumed view — the histograms / heatmaps plot pre-squash). The squashing
    lives only in `from_logits`, so no impl re-triplicates it."""

    logits: SiteDict
    lower: SiteDict
    upper: SiteDict

    @staticmethod
    def from_logits(logits: SiteDict) -> "CI":
        return CI(
            logits=logits,
            lower={k: lower_leaky_hard_sigmoid(v) for k, v in logits.items()},
            upper={k: upper_leaky_hard_sigmoid(v) for k, v in logits.items()},
        )


@runtime_checkable
class CIFn(Protocol):
    """`dict[InputTap, Array] -> CI`. `output_names` partition the model sites (asserted at
    construction); input taps are unconstrained. `expects_axes` must equal the paired
    `DecomposedModel.leading_axes` (asserted at trainer construction)."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    expects_axes: tuple[str, ...]

    def __call__(self, taps: dict[str, Array], *, remat: bool) -> CI: ...

    def shardings(self, mesh: Mesh) -> "CIFn":
        """Per-leaf `dp` placement matching this CI fn's pytree structure (each array leaf
        → a `NamedSharding`; `P()` to replicate). Asserts every declared shard axis tiles
        the mesh. Applied via `jax.jit(init, out_shardings=...)`."""
        ...


# ----------------------------- transformer building blocks -----------------------------


def _weightless_rms_norm(x: Array, eps: float) -> Array:
    return rms_norm(x, jnp.ones((x.shape[-1],), x.dtype), eps)


class CIBlock(eqx.Module):
    """Pre-norm block: weightless-RMSNorm → bidirectional RoPE MHA → residual;
    weightless-RMSNorm → Linear+b → GELU → Linear+b → residual."""

    wq: Array
    wk: Array
    wv: Array
    wo: Array
    w1: Array
    b1: Array
    w2: Array
    b2: Array
    n_head: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def shardings(self, mesh: Mesh) -> "CIBlock":
        """True ÷N ZeRO-1 PERSISTENCE layout (master + Adam m/v shard over the FULL mesh
        `("replicate","fsdp")`); the leading `n_chunks` axis (axis 0) is UNSHARDED. Every
        weight shards its `d_model` dim ÷N and replicates the head / mlp_hidden dim (no TP /
        Megatron axis):

        - qkv (`[nc, head_out, d_model_in]`): d_model ÷N, head replicated.
        - out-proj (`[nc, d_model_out, head_in]`): d_model ÷N, head replicated.
        - w1 up-proj (`[nc, d_model_in, mlp_hidden_out]`): d_model ÷N, mlp_hidden replicated.
        - w2 down-proj (`[nc, mlp_hidden_in, d_model_out]`): d_model ÷N, mlp_hidden replicated.

        Biases replicate. COMPUTE re-pins to `fsdp`-only before the chunk scan (the ZeRO-1
        reconstruction, once/step in ENTRY — `ChunkwiseTransformerCIFn.__call__`), so the
        per-chunk gather is intra-node NVLink. Asserts each ÷N dim tiles the device count."""
        full = ("replicate", "fsdp")
        din = NamedSharding(mesh, P(None, None, full))  # axis2 (d_model in) ÷N
        dout = NamedSharding(mesh, P(None, full, None))  # axis1 (d_model out) ÷N
        repl = NamedSharding(mesh, P())
        n = mesh.devices.size
        for w in (self.wq, self.wk, self.wv):
            assert w.shape[2] % n == 0, f"CIBlock qkv in (d_model) {w.shape[2]} not ÷ N={n}"
        assert self.wo.shape[1] % n == 0, (
            f"CIBlock out-proj out (d_model) {self.wo.shape[1]} not ÷ N={n}"
        )
        assert self.w1.shape[1] % n == 0, f"CIBlock w1 in (d_model) {self.w1.shape[1]} not ÷ N={n}"
        assert self.w2.shape[2] % n == 0, f"CIBlock w2 out (d_model) {self.w2.shape[2]} not ÷ N={n}"
        return eqx.tree_at(
            lambda b: (b.wq, b.wk, b.wv, b.wo, b.w1, b.b1, b.w2, b.b2),
            self,
            (din, din, din, dout, din, repl, dout, repl),
        )

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Array) -> Array:
        t = x.shape[1]
        h = _weightless_rms_norm(x, self.eps)

        def heads(w: Array) -> Array:  # [b, t, d] -> [b, nh, t, hd]  (RoPE layout)
            proj = einops.einsum(h, w, "b t i, o i -> b t o")
            return einops.rearrange(proj, "b t (nh hd) -> b nh t hd", nh=self.n_head)

        q, k, v = heads(self.wq), heads(self.wk), heads(self.wv)
        cos, sin = rope_cos_sin(inv_freq, t, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        qt, kt, vt = (einops.rearrange(a, "b nh t hd -> b t nh hd") for a in (q, k, v))
        # cuDNN flash: heads are local (tp=1, no Megatron head-sharding), so cuDNN's partitioner
        # accepts the q/k/v layout. Flash never materializes the (B,H,T,T) score — which at
        # d=4096/nh=64 is GBs per chunk, stacked over the chunk scan — so it stays off the peak.
        # Bidirectional.
        y = jax.nn.dot_product_attention(qt, kt, vt, is_causal=False, implementation="cudnn")
        x = x + einops.einsum(
            einops.rearrange(y, "b t nh hd -> b t (nh hd)"), self.wo, "b t i, o i -> b t o"
        )
        h = _weightless_rms_norm(x, self.eps)
        hidden = jax.nn.gelu(
            einops.einsum(h, self.w1, "b t i, i o -> b t o") + self.b1, approximate=False
        )
        return x + einops.einsum(hidden, self.w2, "b t i, i o -> b t o") + self.b2


# ----------------------------- chunkwise transformer -----------------------------


@dataclass(frozen=True)
class Chunk:
    """One resolved chunk: the input taps to concatenate → CI for a group of output sites.
    Authored lab-side (from `blocks_per_chunk` + topology); core treats both keyspaces as
    opaque keys. `input_taps` may name several residual taps (e.g. the residual entering the
    chunk plus earlier read points) — RMS-normed per tap and concatenated as the input."""

    input_taps: tuple[str, ...]
    output_sites: tuple[str, ...]


@dataclass(frozen=True)
class _ChunkMeta:
    """Per-chunk static routing, index-aligned with the stacked `chunks` leading axis."""

    input_taps: tuple[str, ...]  # taps to RMS-norm + concatenate as this chunk's input
    output_sites: tuple[str, ...]  # output sites this chunk scores, in C-per-slot order


@dataclass(frozen=True)
class ChunkwiseTransformerCIArch:
    """Resolved chunkwise-transformer arch: explicit chunks + the CI transformer's dims.

    `input_dim` is the per-chunk concatenated input width — a plain linear-layer input
    dimension. The lab computes it from the taps it authored (their widths summed); core
    stays agnostic to what the taps mean, so no transformer concept (residual width) leaks
    in. All chunks share one `input_dim` (the vmap homogeneity requirement)."""

    chunks: tuple[Chunk, ...]
    input_dim: int
    d_model: int
    n_blocks: int
    n_heads: int
    mlp_hidden: int


class ChunkTransformer(eqx.Module):
    """ONE chunk: its (already RMS-normed, concatenated) input `[*leading, total_d_in]` →
    a TUPLE of per-output-site logits (`out` of `[*leading, C_j]` per site-slot j), via
    in_proj → RoPE blocks → one output head PER site-slot.

    One head per site-slot (`out_ws[j] [d_model, C_j]` / `out_bs[j] [C_j]`) instead of a
    single glued `[d_model, ΣC]` head: each head's output IS that site's CI, born already
    split per site (matching `x@V` / the mask, SPEC §4.1 `site_out`). Under pure HSDP the C
    axis is replicated (not sharded), so the split is a pure layout convenience; it was
    load-bearing under the prior TP layout (a tp-sharded glued ΣC axis sliced mid-site),
    and is kept harmlessly.

    In the bundle every array below carries a leading `n_chunks` axis and the module is
    run under `jax.lax.scan` over that axis, so this body is written for a single chunk."""

    in_proj_w: Float[Array, "total_d_in d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: list[CIBlock]
    out_ws: tuple[Float[Array, "d_model _C"], ...]
    out_bs: tuple[Float[Array, " _C"], ...]

    def shardings(self, mesh: Mesh) -> "ChunkTransformer":
        """True ÷N ZeRO-1 PERSISTENCE layout (master + Adam shard over the FULL mesh); leading
        `n_chunks` axis (axis 0) UNSHARDED. `in_proj_w [nc, total_d_in, d_model]`: d_model ÷N,
        total_d_in replicated. Each `out_ws[j] [nc, d_model, C_j]`: d_model ÷N, C_j REPLICATED
        (no TP axis — the CI output C is never sharded, so the mask multiply needs no reshard).
        Blocks delegate to `CIBlock.shardings`; biases replicate. COMPUTE re-pins fsdp-only
        before the chunk scan (`ChunkwiseTransformerCIFn.__call__`)."""
        full = ("replicate", "fsdp")
        dmodel_out = NamedSharding(mesh, P(None, None, full))  # axis2 (d_model out) ÷N
        dmodel_in = NamedSharding(mesh, P(None, full, None))  # axis1 (d_model in) ÷N
        repl = NamedSharding(mesh, P())
        n = mesh.devices.size
        assert self.in_proj_w.shape[2] % n == 0, (
            f"ChunkTransformer in_proj_w d_model {self.in_proj_w.shape[2]} not ÷ N={n}"
        )
        for slot, w in enumerate(self.out_ws):
            assert w.shape[1] % n == 0, (
                f"ChunkTransformer out_ws[{slot}] d_model {w.shape[1]} not ÷ N={n}"
            )
        return eqx.tree_at(
            lambda ct: (ct.in_proj_w, ct.in_proj_b, ct.blocks, ct.out_ws, ct.out_bs),
            self,
            (
                dmodel_out,
                repl,
                [b.shardings(mesh) for b in self.blocks],
                tuple(dmodel_in for _ in self.out_ws),
                tuple(repl for _ in self.out_bs),
            ),
        )

    def __call__(
        self, x: Float[Array, "*leading total_d_in"], inv_freq: Array
    ) -> tuple[Float[Array, "*leading _C"], ...]:
        x = einops.einsum(x, self.in_proj_w, "... i, i o -> ... o") + self.in_proj_b
        for block in self.blocks:
            x = block(x, inv_freq)
        return tuple(
            einops.einsum(x, w, "... i, i o -> ... o") + b
            for w, b in zip(self.out_ws, self.out_bs, strict=True)
        )


def _reconstruct_ci_compute_weights(chunks: "ChunkTransformer") -> "ChunkTransformer":
    """The ZeRO-1 reconstruction for the CI fn: the stacked per-chunk weights arrive with
    their `d_model` dim sharded ÷N over the FULL mesh (the master is `P(..., ("replicate",
    "fsdp"), ...)`); reconstruct them to the `fsdp`-sharded (÷fsdp) COMPUTE layout here,
    BEFORE the chunk scan, so the cross-`replicate` gather runs ONCE per step in ENTRY
    (landing a SMALL ÷fsdp-resident weight stack, NOT the full CI fn) and the per-chunk scan
    body gathers only on `fsdp` (intra-node NVLink), transiently. Cast to bf16 here so the
    ÷fsdp-resident stack is half-size (no f32 full copy). Mirrors the `.shardings` axis
    positions (leading `n_chunks` axis unsharded) with `"fsdp"` in place of the full-mesh
    tuple. No-op off-mesh."""
    if jax.sharding.get_abstract_mesh().empty:
        return chunks
    d_axis2 = P(None, None, "fsdp")  # d_model is axis2 (matmul input dim)
    d_axis1 = P(None, "fsdp", None)  # d_model is axis1 (matmul output dim)

    def pin(x: Array, spec: "P") -> Array:
        # optimization_barrier: cast bf16 BEFORE the gather (else XLA sinks the convert past
        # the all-gather and moves the f32 master — 2x the comm).
        return jax.lax.with_sharding_constraint(
            jax.lax.optimization_barrier(x.astype(jnp.bfloat16)), spec
        )

    pinned_blocks = [
        eqx.tree_at(
            lambda b: (b.wq, b.wk, b.wv, b.wo, b.w1, b.w2),
            blk,
            (pin(blk.wq, d_axis2), pin(blk.wk, d_axis2), pin(blk.wv, d_axis2),
             pin(blk.wo, d_axis1), pin(blk.w1, d_axis2), pin(blk.w2, d_axis1)),
        )
        for blk in chunks.blocks
    ]  # fmt: skip
    return eqx.tree_at(
        lambda ct: (ct.in_proj_w, ct.blocks, ct.out_ws),
        chunks,
        (
            pin(chunks.in_proj_w, d_axis2),  # [nc, total_d_in, d_model] — d_model axis2 on fsdp
            pinned_blocks,
            tuple(pin(w, d_axis1) for w in chunks.out_ws),  # [nc, d_model, C_j] — d_model axis1
        ),
    )


class ChunkwiseTransformerCIFn(eqx.Module):
    """Per-chunk `ChunkTransformer`s stacked along a leading `n_chunks` axis, iterated by a
    `jax.lax.scan` over that axis (lowers as a loop so one chunk's FSDP weight gather is live
    at a time, not all `n_chunks` at once). Each chunk's input is its `chunk_input_taps`
    RMS-normed per tap and concatenated. Requires homogeneous chunks (equal total input width
    and an identical per-slot C tuple — same C-per-output-site ORDER) so the stack, including
    the per-slot output heads, is rectangular — asserted at init."""

    chunks: ChunkTransformer  # arrays stacked along leading n_chunks
    inv_freq: Array  # shared across chunks (RoPE buffer); NOT mapped

    input_names: tuple[str, ...] = eqx.field(static=True)  # dedup union (for read_activations)
    output_names: tuple[str, ...] = eqx.field(static=True)  # all sites, flat
    chunk_meta: tuple[_ChunkMeta, ...] = eqx.field(static=True)  # per-chunk routing
    eps: float = eqx.field(static=True)
    expects_axes: tuple[str, ...] = eqx.field(static=True)

    def shardings(self, mesh: Mesh) -> "ChunkwiseTransformerCIFn":
        """The stacked per-chunk transformer's HSDP layout (`ChunkTransformer.shardings`,
        leading `n_chunks` axis un-sharded); `inv_freq` (a 1-D RoPE buffer) replicates."""
        return eqx.tree_at(
            lambda f: (f.chunks, f.inv_freq),
            self,
            (self.chunks.shardings(mesh), NamedSharding(mesh, P())),
        )

    def __call__(self, taps: dict[str, Array], *, remat: bool) -> CI:
        per_chunk_in = [
            jnp.concatenate(
                [_weightless_rms_norm(taps[k], self.eps) for k in m.input_taps], axis=-1
            )
            for m in self.chunk_meta
        ]
        stacked_in = jnp.stack(per_chunk_in, axis=0)  # [n_chunks, *leading, total_d_in]
        inv_freq = jax.lax.stop_gradient(self.inv_freq)
        # ZeRO-1 reconstruction: the master shards d_model ÷N over the FULL mesh; pin the
        # compute weights `fsdp`-ONLY here, BEFORE the chunk scan, so GSPMD gathers the
        # `replicate` shard ONCE per step in ENTRY (off the hot path) and the per-chunk scan
        # body gathers only on `fsdp` (intra-node NVLink). No-op off-mesh.
        chunks = _reconstruct_ci_compute_weights(self.chunks)
        # `lax.scan` (not `filter_vmap`) over the leading `n_chunks` axis so XLA lowers the
        # chunk iteration as a loop: one chunk's FSDP weight all-gather (∝ ΣC/tp) is live at
        # a time, then freed, instead of every chunk's gathered weights materialized at once
        # (the vmap unrolls, hoisting all n_chunks gathers into the flat entry computation).
        # Same math as the vmap — scan stacks per-iteration outputs exactly as vmap maps
        # them; results match up to fp32 reassociation (XLA picks different matmul layouts).
        chunk_arrays, chunk_static = eqx.partition(chunks, eqx.is_array)

        def run_chunk(
            _: None, scanned: tuple[ChunkTransformer, Array]
        ) -> tuple[None, tuple[Array, ...]]:
            chunk_array, chunk_input = scanned
            chunk = eqx.combine(chunk_array, chunk_static)
            return None, chunk(chunk_input, inv_freq)

        # Per-CHUNK remat: checkpoint the scan BODY so the backward recomputes one chunk at a
        # time, keeping only the carry — NOT all `n_chunks` chunks' attention scores + MLP
        # hidden states stacked `[n_chunks, ...]`. (Whole-CI-fn checkpointing does not bound
        # the scan: the recompute still stacks every chunk — the `[n_chunks, *, seq, seq]`
        # f32 score slab that dominated the full-model step. Same fix shape as the target's
        # per-layer remat.)
        # Each per-slot head stacks over the chunk axis: `stacked_per_slot[j]` is
        # `[n_chunks, *leading, C_j]`. No glued ΣC axis, so no slice — site `(chunk i, slot j)`
        # is `stacked_per_slot[j][i]` directly (chunks are slot-homogeneous in C-per-site
        # ORDER, asserted at init, so slot j carries one C_j across every chunk).
        import os as _os

        # Per-CHUNK checkpoint of the scan BODY in BOTH modes — `remat` controls ONLY whether
        # the chunk ACTIVATIONS are recomputed; it NEVER controls the ÷fsdp→full weight gather.
        # `remat=True` → nothing_saveable: recompute activations AND re-gather (min memory, the
        # `[n_chunks, *, seq, seq]` f32 score slab never stacks). `remat=False` → dots_saveable:
        # SAVE the activation matmuls, still re-gather the weights (a collective, not a dot) — i.e.
        # plain FSDP. WITHOUT any checkpoint the backward would instead stack every chunk's full
        # gathered weights `[n_chunks, …]` as residuals → DDP-stack OOM, so we always checkpoint.
        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )

        if _os.environ.get("PD_CI_BROADCAST", "") == "1":
            # EXPERIMENT: broadcast all chunks at once (vmap, no loop) so XLA sees one network
            # and consolidates the per-chunk cross-node grad reduces into one. Trades the scan's
            # memory bound for fewer collectives — per-chunk remat still drops intermediates.
            def run_chunk_v(chunk_array: ChunkTransformer, chunk_input: Array) -> tuple[Array, ...]:
                chunk = eqx.combine(chunk_array, chunk_static)
                return chunk(chunk_input, inv_freq)

            fn = jax.checkpoint(run_chunk_v, policy=policy)
            stacked_per_slot = jax.vmap(fn)(chunk_arrays, stacked_in)
        else:
            body = jax.checkpoint(run_chunk, policy=policy)
            _, stacked_per_slot = jax.lax.scan(body, None, (chunk_arrays, stacked_in))
        logits: SiteDict = {}
        for chunk_idx, m in enumerate(self.chunk_meta):
            for slot, site in enumerate(m.output_sites):
                logits[site] = stacked_per_slot[slot][chunk_idx]
        return CI.from_logits(logits)


def _init_chunk_transformer(
    arch: ChunkwiseTransformerCIArch,
    total_d_in: int,
    slot_cs: tuple[int, ...],
    key: PRNGKeyArray,
) -> ChunkTransformer:
    """One chunk's params, same Kaiming scheme as the old global transformer: relu-gain
    (√2) on in_proj / MLP-in, linear gain (1) on out / MLP-out, PyTorch-default
    `U(±1/√fan_in)` on the attention projections, zero biases.

    The per-site output heads are SLICES of a single glued `[d, ΣC]` Kaiming draw (drawn with
    the same `out_key`, `gain 1`): head j = columns `[offset_j : offset_j + C_j]`. This keeps
    the RNG consumption (one `(d, ΣC)` normal + one `(ΣC,)` zero bias) and the values bit-for-
    bit identical to the old single glued head, so the equivalence goldens are unchanged —
    the math is the same, only the partitioning differs.

    Each consumer takes its OWN explicit key — the split count lives next to its use
    (`n_blocks + 2` at the top = in_proj + out + one per block; 6 within a block), so it
    can't silently drift out of sync with the number of draws."""
    relu_gain = 2.0**0.5
    d, mlp = arch.d_model, arch.mlp_hidden

    def kaiming(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int, gain: float) -> Array:
        return jax.random.normal(k, shape) * (gain / fan_in**0.5)

    def attn_default(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int) -> Array:
        bound = 1.0 / fan_in**0.5
        return jax.random.uniform(k, shape, minval=-bound, maxval=bound)

    def block(bkey: PRNGKeyArray) -> CIBlock:
        kq, kk, kv, ko, k1, k2 = jax.random.split(bkey, 6)
        return CIBlock(
            wq=attn_default(kq, (d, d), d), wk=attn_default(kk, (d, d), d),
            wv=attn_default(kv, (d, d), d), wo=attn_default(ko, (d, d), d),
            w1=kaiming(k1, (d, mlp), d, relu_gain), b1=jnp.zeros((mlp,)),
            w2=kaiming(k2, (mlp, d), mlp, 1.0), b2=jnp.zeros((d,)),
            n_head=arch.n_heads, eps=CI_FN_RMS_EPS,
        )  # fmt: skip

    in_key, out_key, *block_keys = jax.random.split(key, arch.n_blocks + 2)
    c_chunk = sum(slot_cs)
    glued_w = kaiming(out_key, (d, c_chunk), d, 1.0)
    glued_b = jnp.zeros((c_chunk,))
    offsets = [0]
    for c in slot_cs:
        offsets.append(offsets[-1] + c)
    return ChunkTransformer(
        in_proj_w=kaiming(in_key, (total_d_in, d), total_d_in, relu_gain),
        in_proj_b=jnp.zeros((d,)),
        blocks=[block(bk) for bk in block_keys],
        out_ws=tuple(glued_w[:, offsets[j] : offsets[j + 1]] for j in range(len(slot_cs))),
        out_bs=tuple(glued_b[offsets[j] : offsets[j + 1]] for j in range(len(slot_cs))),
    )


def init_chunkwise_transformer_ci_fn(
    arch: ChunkwiseTransformerCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> ChunkwiseTransformerCIFn:
    """Validate the output partition + chunk homogeneity, then build STACKED chunk params.

    - partition: the chunks' output sites are disjoint and cover every model site.
    - homogeneity: equal tap count (→ equal total input width) and an identical per-SLOT C
      tuple (same C-per-output-site in the same ORDER) across every chunk, so the per-chunk
      params — including the per-slot output heads — stack rectangularly along the scanned
      `n_chunks` axis. The per-slot heads stack slot-by-slot, so a mismatched C ORDER would
      silently misalign sites across chunks: fail fast.
    """
    site_c = {s.name: s.C for s in sites}
    covered = [name for ch in arch.chunks for name in ch.output_sites]
    assert sorted(covered) == sorted(s.name for s in sites), "chunks must partition sites"
    assert len(covered) == len(set(covered)), "chunks overlap on an output site"
    slot_cs_per_chunk = {tuple(site_c[n] for n in ch.output_sites) for ch in arch.chunks}
    assert len(slot_cs_per_chunk) == 1, (
        f"chunks not homogeneous in per-slot C tuple (the per-slot heads stack slot-by-slot "
        f"across chunks — equal C-per-site ORDER required): {slot_cs_per_chunk}"
    )
    (slot_cs,) = slot_cs_per_chunk
    assert all(ch.input_taps for ch in arch.chunks), "each chunk needs at least one input tap"
    # Per-chunk cat width must equal `arch.input_dim` (lab guarantees it; the runtime
    # `jnp.stack` / in_proj einsum fails loud if a chunk's taps don't sum to it).

    hd = arch.d_model // arch.n_heads
    assert arch.d_model % arch.n_heads == 0 and hd % 2 == 0, (arch.d_model, arch.n_heads)
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))

    per_chunk = [
        _init_chunk_transformer(arch, arch.input_dim, slot_cs, jax.random.fold_in(key, i))
        for i in range(len(arch.chunks))
    ]
    stacked: ChunkTransformer = jax.tree.map(lambda *xs: jnp.stack(xs), *per_chunk)

    return ChunkwiseTransformerCIFn(
        chunks=stacked,
        inv_freq=inv_freq,
        input_names=tuple(sorted({tap for ch in arch.chunks for tap in ch.input_taps})),
        output_names=tuple(name for ch in arch.chunks for name in ch.output_sites),
        chunk_meta=tuple(_ChunkMeta(ch.input_taps, ch.output_sites) for ch in arch.chunks),
        eps=CI_FN_RMS_EPS,
        expects_axes=("sequence",),
    )


# ------------------- per-site / global MLPs (positionless `expects_axes=()`) -------------------


# The MLP arches below mirror their pydantic configs (`LayerwiseMlpCiConfig` /
# `GlobalMlpCiConfig`) field-for-field by design: the lab's `ci_arch` converter is a trivial
# `type`-strip + list→tuple. The duplication is deliberate — it keeps a uniform `CIFnArch`
# union for `build_ci_fn` (vs the chunkwise arch, which genuinely resolves against a target).
@dataclass(frozen=True)
class MLPCIArch:
    """Hidden widths shared by every per-site MLP."""

    hidden_dims: tuple[int, ...]


class SiteMLP(eqx.Module):
    """`hidden_dims` Linear+GELU layers then a linear head: Kaiming-`relu` (`gain √2`)
    hidden layers with zero bias, linear-gain (`1`) final head."""

    weights: list[Float[Array, "d_in d_out"]]
    biases: list[Float[Array, " d_out"]]

    def shardings(self, mesh: Mesh) -> "SiteMLP":
        """Each `[d_in, d_out]` weight shards its OUTPUT axis (axis 1) ÷N over the FULL mesh
        (`("replicate","fsdp")`) — the master + Adam state shard ÷N. 1-D biases replicate.
        The toy MLP is single-shot (no scan), so there is no compute reconstruction; GSPMD
        gathers as needed (trivial at the toy's small device count). Asserts every output dim
        tiles the device count."""
        shard_out = NamedSharding(mesh, P(None, ("replicate", "fsdp")))
        repl = NamedSharding(mesh, P())
        n = mesh.devices.size
        for layer_idx, w in enumerate(self.weights):
            assert w.shape[1] % n == 0, (
                f"SiteMLP.weights[{layer_idx}].d_out {w.shape[1]} not ÷ N={n}"
            )
        return eqx.tree_at(
            lambda m: (m.weights, m.biases),
            self,
            ([shard_out] * len(self.weights), [repl] * len(self.biases)),
        )

    def __call__(self, x: Float[Array, "*leading d_in"]) -> Float[Array, "*leading C"]:
        n_hidden = len(self.weights) - 1
        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = einops.einsum(x, w, "... i, i o -> ... o") + b
            if layer_idx < n_hidden:
                x = jax.nn.gelu(x, approximate=False)
        return x


class LayerwiseMLPCIFn(eqx.Module):
    """One MLP per site behind the `CIFn` protocol. Each site reads its own tap, so
    `input_names == output_names`."""

    site_mlps: dict[str, SiteMLP]
    input_names: tuple[str, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    expects_axes: tuple[str, ...] = eqx.field(static=True)

    def shardings(self, mesh: Mesh) -> "LayerwiseMLPCIFn":
        return eqx.tree_at(
            lambda f: f.site_mlps,
            self,
            {name: mlp.shardings(mesh) for name, mlp in self.site_mlps.items()},
        )

    def site_logits(self, taps: dict[str, Array]) -> dict[str, Array]:
        assert set(taps) == set(self.input_names), (
            f"tap keys {sorted(taps)} != CI fn inputs {sorted(self.input_names)}"
        )
        return {name: self.site_mlps[name](taps[name]) for name in self.output_names}

    def __call__(self, taps: dict[str, Array], *, remat: bool) -> CI:
        del remat  # single-shot (no scan to bound) -> remat is a no-op for the MLP CI fns
        return CI.from_logits(self.site_logits(taps))


def _init_mlp_stack(dims: tuple[int, ...], key: PRNGKeyArray) -> SiteMLP:
    """One `Linear+GELU` stack `dims[0] -> ... -> dims[-1]`: Kaiming `relu`-gain (`√2`) on
    the hidden layers, linear gain (`1`) on the final head, zero biases."""
    relu_gain = 2.0**0.5
    layer_keys = jax.random.split(key, len(dims) - 1)
    weights: list[Array] = []
    biases: list[Array] = []
    for layer_idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        gain = relu_gain if layer_idx < len(dims) - 2 else 1.0
        weights.append(jax.random.normal(layer_keys[layer_idx], (d_in, d_out)) * (gain / d_in**0.5))
        biases.append(jnp.zeros((d_out,)))
    return SiteMLP(weights=weights, biases=biases)


def init_layerwise_mlp_ci_fn(
    arch: MLPCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> LayerwiseMLPCIFn:
    """Per-site MLP init: each site's MLP maps `d_in -> hidden_dims... -> C`."""
    assert arch.hidden_dims, "MLP CI fn needs at least one hidden layer"
    site_mlps = {
        spec.name: _init_mlp_stack(
            (spec.d_in, *arch.hidden_dims, spec.C), jax.random.fold_in(key, site_idx)
        )
        for site_idx, spec in enumerate(sites)
    }
    names = tuple(s.name for s in sites)
    return LayerwiseMLPCIFn(
        site_mlps=site_mlps, input_names=names, output_names=names, expects_axes=()
    )


@dataclass(frozen=True)
class GlobalMLPCIArch:
    """Hidden widths of the single global MLP shared across ALL sites."""

    hidden_dims: tuple[int, ...]


class GlobalMLPCIFn(eqx.Module):
    """ONE shared MLP over all sites behind the `CIFn` protocol. Per-site inputs are
    concatenated in `input_names` order into `[*leading, Σ d_in]`, mapped to `[*leading,
    Σ C]`, and split back per output site by `c_sizes` in `output_names` order — so a
    site's logits depend on every site's input."""

    mlp: SiteMLP
    input_names: tuple[str, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    in_sizes: tuple[int, ...] = eqx.field(static=True)
    c_sizes: tuple[int, ...] = eqx.field(static=True)
    expects_axes: tuple[str, ...] = eqx.field(static=True)

    def shardings(self, mesh: Mesh) -> "GlobalMLPCIFn":
        return eqx.tree_at(lambda f: f.mlp, self, self.mlp.shardings(mesh))

    def site_logits(self, taps: dict[str, Array]) -> dict[str, Array]:
        assert set(taps) == set(self.input_names), (
            f"tap keys {sorted(taps)} != CI fn inputs {sorted(self.input_names)}"
        )
        for name, in_size in zip(self.input_names, self.in_sizes, strict=True):
            assert taps[name].shape[-1] == in_size, (
                f"tap {name} d_in {taps[name].shape[-1]} != expected {in_size}"
            )
        concatenated = jnp.concatenate([taps[n] for n in self.input_names], axis=-1)
        logits = self.mlp(concatenated)
        offsets = [0]
        for c in self.c_sizes:
            offsets.append(offsets[-1] + c)
        return {
            name: logits[..., offsets[i] : offsets[i + 1]]
            for i, name in enumerate(self.output_names)
        }

    def __call__(self, taps: dict[str, Array], *, remat: bool) -> CI:
        del remat  # single-shot (no scan to bound) -> remat is a no-op for the MLP CI fns
        return CI.from_logits(self.site_logits(taps))


def init_global_mlp_ci_fn(
    arch: GlobalMLPCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> GlobalMLPCIFn:
    """Global MLP init: one stack `Σ d_in -> hidden_dims... -> Σ C`, same Kaiming scheme
    as the per-site MLP."""
    assert arch.hidden_dims, "global MLP CI fn needs at least one hidden layer"
    in_sizes = tuple(s.d_in for s in sites)
    c_sizes = tuple(s.C for s in sites)
    names = tuple(s.name for s in sites)
    dims = (sum(in_sizes), *arch.hidden_dims, sum(c_sizes))
    return GlobalMLPCIFn(
        mlp=_init_mlp_stack(dims, key),
        input_names=names,
        output_names=names,
        in_sizes=in_sizes,
        c_sizes=c_sizes,
        expects_axes=(),
    )


# ----------------------------- construction (placement-agnostic) -----------------------------


CIFnArch = ChunkwiseTransformerCIArch | MLPCIArch | GlobalMLPCIArch
"""Every CI-fn architecture. Construction goes through `build_ci_fn`; sharding/placement is
a separate, scale-driven concern (see `llama8b_sharding`), never coupled to arch type."""


def build_ci_fn(arch: CIFnArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray) -> CIFn:
    """Construct the CI fn for `arch`, host-side and unsharded. Placement is applied by the
    caller by SCALE (mesh × C-divisibility), never by which arch this is."""
    match arch:
        case ChunkwiseTransformerCIArch():
            return init_chunkwise_transformer_ci_fn(arch, sites, key)
        case MLPCIArch():
            return init_layerwise_mlp_ci_fn(arch, sites, key)
        case GlobalMLPCIArch():
            return init_global_mlp_ci_fn(arch, sites, key)
