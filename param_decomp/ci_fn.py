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
from param_decomp.sharding import assert_divisible
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

    def __call__(self, taps: dict[str, Array]) -> CI: ...

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
        """Uniform FSDP(`dp`)×TP(`tp`); the leading `n_chunks` axis (axis 0) is UNSHARDED — a
        plain vmap axis, not a sharding dim (the mesh-simplification: every param shards the
        same way regardless of the chunk structure). Each weight keeps its Megatron dim on
        `tp` and FSDP's the other matmul dim on `dp`:

        - qkv (`[nc, head_out, d_model_in]`): head on `tp` (col-parallel), d_model FSDP on `dp`.
        - out-proj (`[nc, d_model_out, head_in]`): head on `tp` (row-parallel), d_model FSDP on `dp`.
        - w1 up-proj (`[nc, d_model_in, mlp_hidden_out]`): mlp_hidden on `tp`, d_model FSDP on `dp`.
        - w2 down-proj (`[nc, mlp_hidden_in, d_model_out]`): mlp_hidden on `tp`, d_model FSDP on `dp`.

        Biases replicate (GSPMD broadcasts over the tp-sharded activation). The FSDP dim is
        all-gathered per matmul, bounded by `/tp` (TP keeps its half)."""
        col = NamedSharding(mesh, P(None, "tp", "dp"))  # axis1 (out) on tp, axis2 (in) FSDP on dp
        row = NamedSharding(mesh, P(None, "dp", "tp"))  # axis1 (out) FSDP on dp, axis2 (in) on tp
        repl = NamedSharding(mesh, P())
        for w in (self.wq, self.wk, self.wv):
            assert_divisible(w.shape[1], mesh, "tp", "CIBlock qkv out (head dim)")
            assert_divisible(w.shape[2], mesh, "dp", "CIBlock qkv in (d_model)")
        assert_divisible(self.wo.shape[1], mesh, "dp", "CIBlock out-proj out (d_model)")
        assert_divisible(self.wo.shape[2], mesh, "tp", "CIBlock out-proj in (head dim)")
        assert_divisible(self.w1.shape[1], mesh, "dp", "CIBlock w1 in (d_model)")
        assert_divisible(self.w1.shape[2], mesh, "tp", "CIBlock w1 out (mlp_hidden)")
        assert_divisible(self.w2.shape[1], mesh, "tp", "CIBlock w2 in (mlp_hidden)")
        assert_divisible(self.w2.shape[2], mesh, "dp", "CIBlock w2 out (d_model)")
        return eqx.tree_at(
            lambda b: (b.wq, b.wk, b.wv, b.wo, b.w1, b.b1, b.w2, b.b2),
            self,
            (col, col, col, row, row, repl, col, repl),
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
        # xla (not cuDNN flash): the CI fn's attention weights are Megatron-sharded (heads on
        # `tp`) under the 2-D mesh, and cuDNN's custom partitioner rejects that q/k/v layout
        # ("Query, key and value should have same sharding"). The CI fn is small (low d_model,
        # short seq) so the xla path's (B,H,T,T) score is tens of MB — no OOM, unlike the
        # target's attention which keeps cuDNN flash. Bidirectional.
        y = jax.nn.dot_product_attention(qt, kt, vt, is_causal=False, implementation="xla")
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
    output_sites: tuple[str, ...]  # output sites this chunk scores
    c: tuple[int, ...]  # C per output site (splits the head's c_chunk slab)


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
    `[*leading, c_chunk]` logits, via in_proj → RoPE blocks → head.

    In the bundle every array below carries a leading `n_chunks` axis and the module is
    run under `jax.lax.scan` over that axis, so this body is written for a single chunk."""

    in_proj_w: Float[Array, "total_d_in d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: list[CIBlock]
    out_w: Float[Array, "d_model c_chunk"]
    out_b: Float[Array, " c_chunk"]

    def shardings(self, mesh: Mesh) -> "ChunkTransformer":
        """Uniform FSDP(`dp`)×TP(`tp`); leading `n_chunks` axis (axis 0) UNSHARDED. `in_proj_w
        [nc, total_d_in, d_model]`: d_model on `tp` (col-parallel out), total_d_in FSDP on `dp`.
        `out_w [nc, d_model, c_chunk]`: c_chunk on `tp` (col-parallel out — the C-on-`tp` that
        aligns with V/U and the mask), d_model FSDP on `dp`. Blocks delegate to
        `CIBlock.shardings`; biases replicate (GSPMD broadcasts over the tp-sharded output)."""
        row = NamedSharding(mesh, P(None, "dp", "tp"))  # axis1 (in) FSDP on dp, axis2 (out) on tp
        repl = NamedSharding(mesh, P())
        assert_divisible(
            self.in_proj_w.shape[1], mesh, "dp", "ChunkTransformer in_proj_w total_d_in"
        )
        assert_divisible(self.in_proj_w.shape[2], mesh, "tp", "ChunkTransformer in_proj_w d_model")
        assert_divisible(self.out_w.shape[1], mesh, "dp", "ChunkTransformer out_w d_model")
        assert_divisible(self.out_w.shape[2], mesh, "tp", "ChunkTransformer out_w c_chunk")
        return eqx.tree_at(
            lambda ct: (ct.in_proj_w, ct.in_proj_b, ct.blocks, ct.out_w, ct.out_b),
            self,
            (row, repl, [b.shardings(mesh) for b in self.blocks], row, repl),
        )

    def __call__(self, x: Float[Array, "*leading total_d_in"], inv_freq: Array) -> Array:
        x = einops.einsum(x, self.in_proj_w, "... i, i o -> ... o") + self.in_proj_b
        for block in self.blocks:
            x = block(x, inv_freq)
        return einops.einsum(x, self.out_w, "... i, i o -> ... o") + self.out_b


class ChunkwiseTransformerCIFn(eqx.Module):
    """Per-chunk `ChunkTransformer`s stacked along a leading `n_chunks` axis, iterated by a
    `jax.lax.scan` over that axis (lowers as a loop so one chunk's FSDP weight gather is live
    at a time, not all `n_chunks` at once). Each chunk's input is its `chunk_input_taps`
    RMS-normed per tap and concatenated. Requires homogeneous chunks (equal total input width
    and `c_chunk`) so the stack is rectangular — asserted at init."""

    chunks: ChunkTransformer  # arrays stacked along leading n_chunks
    inv_freq: Array  # shared across chunks (RoPE buffer); NOT mapped

    input_names: tuple[str, ...] = eqx.field(static=True)  # dedup union (for read_activations)
    output_names: tuple[str, ...] = eqx.field(static=True)  # all sites, flat
    chunk_meta: tuple[_ChunkMeta, ...] = eqx.field(static=True)  # per-chunk routing
    eps: float = eqx.field(static=True)
    expects_axes: tuple[str, ...] = eqx.field(static=True)

    def shardings(self, mesh: Mesh) -> "ChunkwiseTransformerCIFn":
        """The stacked per-chunk transformer's Megatron layout (`ChunkTransformer.shardings`,
        leading `n_chunks` axis un-sharded); `inv_freq` (a 1-D RoPE buffer) replicates."""
        return eqx.tree_at(
            lambda f: (f.chunks, f.inv_freq),
            self,
            (self.chunks.shardings(mesh), NamedSharding(mesh, P())),
        )

    def __call__(self, taps: dict[str, Array]) -> CI:
        per_chunk_in = [
            jnp.concatenate(
                [_weightless_rms_norm(taps[k], self.eps) for k in m.input_taps], axis=-1
            )
            for m in self.chunk_meta
        ]
        stacked_in = jnp.stack(per_chunk_in, axis=0)  # [n_chunks, *leading, total_d_in]
        inv_freq = jax.lax.stop_gradient(self.inv_freq)
        # `lax.scan` (not `filter_vmap`) over the leading `n_chunks` axis so XLA lowers the
        # chunk iteration as a loop: one chunk's FSDP weight all-gather (∝ ΣC/tp) is live at
        # a time, then freed, instead of every chunk's gathered weights materialized at once
        # (the vmap unrolls, hoisting all n_chunks gathers into the flat entry computation).
        # Same math as the vmap — scan stacks per-iteration outputs exactly as vmap maps
        # them; results match up to fp32 reassociation (XLA picks different matmul layouts).
        chunk_arrays, chunk_static = eqx.partition(self.chunks, eqx.is_array)

        def run_chunk(_: None, scanned: tuple[ChunkTransformer, Array]) -> tuple[None, Array]:
            chunk_array, chunk_input = scanned
            chunk = eqx.combine(chunk_array, chunk_static)
            return None, chunk(chunk_input, inv_freq)

        _, stacked_logits = jax.lax.scan(
            run_chunk, None, (chunk_arrays, stacked_in)
        )  # [n_chunks, *leading, c_chunk]
        return CI.from_logits(self._split(stacked_logits))

    def _split(self, stacked: Array) -> SiteDict:
        """Static per-chunk, per-site split of the `[n_chunks, *leading, c_chunk]` slab."""
        out: SiteDict = {}
        for chunk_idx, m in enumerate(self.chunk_meta):
            offset = 0
            for site, c in zip(m.output_sites, m.c, strict=True):
                out[site] = stacked[chunk_idx, ..., offset : offset + c]
                offset += c
            assert offset == stacked.shape[-1], (offset, stacked.shape[-1])  # full slab consumed
        return out


def _init_chunk_transformer(
    arch: ChunkwiseTransformerCIArch,
    total_d_in: int,
    c_chunk: int,
    key: PRNGKeyArray,
) -> ChunkTransformer:
    """One chunk's params, same Kaiming scheme as the old global transformer: relu-gain
    (√2) on in_proj / MLP-in, linear gain (1) on out / MLP-out, PyTorch-default
    `U(±1/√fan_in)` on the attention projections, zero biases.

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
    return ChunkTransformer(
        in_proj_w=kaiming(in_key, (total_d_in, d), total_d_in, relu_gain),
        in_proj_b=jnp.zeros((d,)),
        blocks=[block(bk) for bk in block_keys],
        out_w=kaiming(out_key, (d, c_chunk), d, 1.0),
        out_b=jnp.zeros((c_chunk,)),
    )


def init_chunkwise_transformer_ci_fn(
    arch: ChunkwiseTransformerCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> ChunkwiseTransformerCIFn:
    """Validate the output partition + chunk homogeneity, then build STACKED chunk params.

    - partition: the chunks' output sites are disjoint and cover every model site.
    - homogeneity: equal tap count (→ equal total input width) and equal `Σ C` per chunk,
      so the per-chunk params stack rectangularly along the scanned `n_chunks` axis.
    """
    site_c = {s.name: s.C for s in sites}
    covered = [name for ch in arch.chunks for name in ch.output_sites]
    assert sorted(covered) == sorted(s.name for s in sites), "chunks must partition sites"
    assert len(covered) == len(set(covered)), "chunks overlap on an output site"
    c_per_chunk = {sum(site_c[n] for n in ch.output_sites) for ch in arch.chunks}
    assert len(c_per_chunk) == 1, f"chunks not homogeneous in Σ C (vmap needs equal): {c_per_chunk}"
    (c_chunk,) = c_per_chunk
    assert all(ch.input_taps for ch in arch.chunks), "each chunk needs at least one input tap"
    # Per-chunk cat width must equal `arch.input_dim` (lab guarantees it; the runtime
    # `jnp.stack` / in_proj einsum fails loud if a chunk's taps don't sum to it).

    hd = arch.d_model // arch.n_heads
    assert arch.d_model % arch.n_heads == 0 and hd % 2 == 0, (arch.d_model, arch.n_heads)
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))

    per_chunk = [
        _init_chunk_transformer(arch, arch.input_dim, c_chunk, jax.random.fold_in(key, i))
        for i in range(len(arch.chunks))
    ]
    stacked: ChunkTransformer = jax.tree.map(lambda *xs: jnp.stack(xs), *per_chunk)

    return ChunkwiseTransformerCIFn(
        chunks=stacked,
        inv_freq=inv_freq,
        input_names=tuple(sorted({tap for ch in arch.chunks for tap in ch.input_taps})),
        output_names=tuple(name for ch in arch.chunks for name in ch.output_sites),
        chunk_meta=tuple(
            _ChunkMeta(ch.input_taps, ch.output_sites, tuple(site_c[n] for n in ch.output_sites))
            for ch in arch.chunks
        ),
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
        """Each `[d_in, d_out]` weight shards its OUTPUT axis (axis 1) over `dp`; 1-D biases
        replicate. Asserts every output dim tiles the mesh."""
        shard_out = NamedSharding(mesh, P(None, "dp"))
        repl = NamedSharding(mesh, P())
        for layer_idx, w in enumerate(self.weights):
            assert_divisible(w.shape[1], mesh, "dp", f"SiteMLP.weights[{layer_idx}].d_out")
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

    def __call__(self, taps: dict[str, Array]) -> CI:
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

    def __call__(self, taps: dict[str, Array]) -> CI:
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
