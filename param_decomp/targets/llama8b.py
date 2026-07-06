"""Llama-3.1-8B vendored target — the first `DecomposedModel` implementation.

The decomposed sites are any per-layer weight matrices (SPEC §1/§3) named torch-style:
`layers.{i}.self_attn.{q,k,v,o}_proj` and `layers.{i}.mlp.{gate,up,down}_proj`, each
with its own C. `LlamaDecomposedModel` (an `eqx.Module`) carries the full frozen model —
embedding through every layer to the LM head — as array fields, threaded into the jitted
step as a pytree arg; layers without sites run the plain frozen block.

q/k/v sites are decomposed BEFORE RoPE/SDPA (the masked site output feeds the
attention math); the o site applies to the attention output. V/U masters are fp32
keyed per site (`DecompVU`); frozen weights are stored bf16 (SPEC N1) — the trainer
casts for compute.

Real HF weights load straight from the cached safetensors (no torch dep).
"""

import json
import re
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int
from safetensors import safe_open

from param_decomp.components import (
    DecompVU,
    SiteC,
    SiteSpec,
    dequantize_fp8,
    quantize_fp8,
    site_out,
)
from param_decomp.losses import kl_per_position
from param_decomp.sharding import assert_divisible
from vendored_jax.llama import (
    LlamaConfig,
    apply_rope,
    causal_sdpa,
    llama3_inv_freq,
    rms_norm,
    rope_cos_sin,
)

DT = jnp.bfloat16

KIND_ORDER = ("q", "k", "v", "o", "gate", "up", "down")
"""Within-layer canonical site order = computation order. The canonical site order
(`llama_site_specs`) is layer-ascending, then this."""
ATTN_KINDS = ("q", "k", "v", "o")
MLP_KINDS = ("gate", "up", "down")

SITE_NAME_PATTERN = re.compile(
    r"^layers\.(\d+)\.(?:self_attn\.(q|k|v|o)|mlp\.(gate|up|down))_proj$"
)


def llama31_8b_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128256,
        n_layer=32,
        n_head=32,
        n_kv_head=8,
        n_embd=4096,
        n_intermediate=14336,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=131072,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=8192,
    )


def site_name(layer: int, kind: str) -> str:
    assert kind in KIND_ORDER, kind
    submodule = "self_attn" if kind in ATTN_KINDS else "mlp"
    return f"layers.{layer}.{submodule}.{kind}_proj"


def parse_site_name(name: str) -> tuple[int, str]:
    """`layers.{i}.{self_attn,mlp}.{kind}_proj` -> (layer, kind); rejects anything else
    (including kind/submodule mismatches like `self_attn.gate_proj`)."""
    match = SITE_NAME_PATTERN.match(name)
    assert match is not None, f"unsupported site name {name!r}"
    layer, attn_kind, mlp_kind = match.groups()
    return int(layer), attn_kind if attn_kind is not None else mlp_kind


def site_dims(cfg: LlamaConfig, kind: str) -> tuple[int, int]:
    """(d_in, d_out) of one per-layer matrix, right-mult orientation."""
    d, di = cfg.n_embd, cfg.n_intermediate
    qd = cfg.n_head * cfg.head_dim
    kvd = cfg.n_kv_head * cfg.head_dim
    match kind:
        case "q":
            return d, qd
        case "k" | "v":
            return d, kvd
        case "o":
            return qd, d
        case "gate" | "up":
            return d, di
        case "down":
            return di, d
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    """Canonical site order: layer-ascending, `KIND_ORDER` within a layer. Names must
    parse and be unique."""
    names = [site.name for site in site_cs]
    assert len(set(names)) == len(names), f"duplicate sites in {names}"

    def order_key(site: SiteC) -> tuple[int, int]:
        layer, kind = parse_site_name(site.name)
        return layer, KIND_ORDER.index(kind)

    return tuple(sorted(site_cs, key=order_key))


def mlp_family_site_cs(first_layer: int, last_layer: int, C: int) -> tuple[SiteC, ...]:
    """The gate/up/down sites of a contiguous layer range at one C (the native-config
    target family), in canonical order."""
    assert first_layer <= last_layer, (first_layer, last_layer)
    return tuple(
        SiteC(site_name(layer, kind), C)
        for layer in range(first_layer, last_layer + 1)
        for kind in MLP_KINDS
    )


def llama_site_specs(cfg: LlamaConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    """Shape-resolved specs in canonical order (input must already be canonical)."""
    assert site_cs == canonical_site_cs(site_cs), f"sites not in canonical order: {site_cs}"
    specs = []
    for site in site_cs:
        layer, kind = parse_site_name(site.name)
        assert 0 <= layer < cfg.n_layer, (site.name, cfg.n_layer)
        assert site.C >= 1, site
        specs.append(SiteSpec(site.name, *site_dims(cfg, kind), site.C))
    return tuple(specs)


# ----------------------------- frozen layers -----------------------------


class FrozenAttn(eqx.Module):
    wq: Float[Array, "qd d"]
    wk: Float[Array, "kvd d"]
    wv: Float[Array, "kvd d"]
    wo: Float[Array, "d qd"]
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    n_rep: int = eqx.field(static=True)

    def shardings(self, mesh: "Mesh") -> "FrozenAttn":
        """Stacked (leading `n_layer`, UNSHARDED — the scan axis) FSDP on `fsdp`: the `d` dim
        shards on `fsdp` (gathered per layer in the scan, on NVLink); the HEAD dim stays
        REPLICATED. `core` runs batch-parallel attention (q/k/v constrained batch over the
        full mesh, heads replicated — that identical spec is what cuDNN's partitioner
        requires), so the projections must come out heads-replicated. Attention weights are
        small, so FSDP-on-`fsdp` sharding is plenty."""
        in_fsdp = NamedSharding(mesh, P(None, None, "fsdp"))  # qkv [nc, head(repl), d on fsdp]
        out_fsdp = NamedSharding(mesh, P(None, "fsdp", None))  # wo [nc, d on fsdp, head(repl)]
        for w in (self.wq, self.wk, self.wv):
            assert_divisible(w.shape[2], mesh, "fsdp", "FrozenAttn qkv in (d)")
        assert_divisible(self.wo.shape[1], mesh, "fsdp", "FrozenAttn out-proj out (d)")
        return eqx.tree_at(
            lambda a: (a.wq, a.wk, a.wv, a.wo), self, (in_fsdp, in_fsdp, in_fsdp, out_fsdp)
        )

    def core(
        self,
        q_flat: Float[Array, "b t qd"],
        k_flat: Float[Array, "b t kvd"],
        v_flat: Float[Array, "b t kvd"],
        inv_freq: Array,
    ) -> Float[Array, "b t qd"]:
        """RoPE + causal SDPA between the q/k/v projections and the o projection —
        the seam the decomposed q/k/v site outputs feed into."""
        b, t, _ = q_flat.shape
        assert q_flat.shape[-1] == self.n_head * self.head_dim, q_flat.shape
        assert k_flat.shape[-1] == self.n_kv_head * self.head_dim, k_flat.shape
        assert v_flat.shape[-1] == self.n_kv_head * self.head_dim, v_flat.shape
        q = q_flat.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k_flat.reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v_flat.reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, q_flat.dtype)
        q, k = apply_rope(q, k, cos, sin)
        # Native GQA: do NOT repeat_kv — `dot_product_attention` handles the q-heads:kv-heads
        # grouping internally. Repeating k/v to n_head and THEN sharding makes the SPMD
        # partitioner derive the repeated-k/v layout from the small n_kv_head source,
        # inconsistent with q -> cuDNN "Query, key and value should have same sharding" (forward
        # AND its rematerialized backward). Real GQA keeps q (n_head) and k/v (n_kv_head) as
        # independent head-parallel tensors with the SAME spec (different head COUNTS is fine).
        # cuDNN flash attention's custom partitioner requires q/k/v IDENTICALLY sharded.
        # Pure HSDP: pin all three batch-parallel over the FULL mesh, HEADS replicated
        # (`P(('replicate','fsdp'), None, None, None)` for `[b, heads, t, hd]`). The identical
        # q/k/v spec is exactly what cuDNN's flash partitioner demands; heads-replicated keeps
        # q (n_head) and k/v (n_kv_head) consistently sharded (no head TP that would split them
        # to different per-rank counts). The q/k/v projection outputs are `d_out`-replicated
        # (U FSDP's the C side, not d_out's head); this constraint pins the batch right here.
        # Guarded so it's a no-op off-mesh (CPU tests / single device); `run.py` sets the
        # global mesh.
        if not jax.sharding.get_abstract_mesh().empty:
            qkv_spec = jax.sharding.PartitionSpec(
                ("replicate", "fsdp"), None, None, None
            )  # batch-parallel: q/k/v IDENTICAL spec -> cuDNN happy; flash = no score materialization
            q, k, v = (jax.lax.with_sharding_constraint(a, qkv_spec) for a in (q, k, v))
        return causal_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(b, t, self.n_head * self.head_dim)

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Array) -> Array:
        return self.core(x @ self.wq.T, x @ self.wk.T, x @ self.wv.T, inv_freq) @ self.wo.T


class LlamaLayer(eqx.Module):
    """One layer's frozen weights — norms, attention, MLP. Decomposed sites read
    their frozen target W from here at forward time; layers without sites run the
    plain frozen block from the same fields. Weights pass as a runtime arg — never
    baked into the HLO as a multi-GB constant."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    Wg: Float[Array, "di d"]
    Wu: Float[Array, "di d"]
    Wd: Float[Array, "d di"]

    def shardings(self, mesh: "Mesh") -> "LlamaLayer":
        """Stacked FSDP on `fsdp` (no TP): every MLP weight shards its `d`-dim on `fsdp`,
        the intermediate dim stays replicated; gathered back per layer inside the scan.
        Norms replicate; attn delegates to `FrozenAttn.shardings`."""
        in_fsdp = NamedSharding(mesh, P(None, None, "fsdp"))  # Wg/Wu [nc, di(repl), d on fsdp]
        out_fsdp = NamedSharding(mesh, P(None, "fsdp", None))  # Wd [nc, d on fsdp, di(repl)]
        repl = NamedSharding(mesh, P())
        assert_divisible(self.Wg.shape[2], mesh, "fsdp", "Wg in (d)")
        assert_divisible(self.Wd.shape[1], mesh, "fsdp", "Wd out (d)")
        return eqx.tree_at(
            lambda layer: (layer.ln1, layer.ln2, layer.attn, layer.Wg, layer.Wu, layer.Wd),
            self,
            (repl, repl, self.attn.shardings(mesh), in_fsdp, in_fsdp, out_fsdp),
        )


def _frozen_site_weight(layer: LlamaLayer, kind: str) -> Array:
    match kind:
        case "q":
            return layer.attn.wq
        case "k":
            return layer.attn.wk
        case "v":
            return layer.attn.wv
        case "o":
            return layer.attn.wo
        case "gate":
            return layer.Wg
        case "up":
            return layer.Wu
        case "down":
            return layer.Wd
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


# ----------------------------- forwards -----------------------------


def _clean_mlp_out(layer: LlamaLayer, mlp_in: Array) -> Array:
    """Frozen target MLP — exactly `W` applied, not the `V@U + (W−V@U)` identity, so
    non-live sites carry no V/U gradient and no decomposition rounding (SPEC S2/S3)."""
    return (jax.nn.silu(mlp_in @ layer.Wg.T) * (mlp_in @ layer.Wu.T)) @ layer.Wd.T


def _stack_layers(layers: list[LlamaLayer]) -> LlamaLayer:
    """Stack a per-layer `LlamaLayer` list into one whose array leaves carry a leading
    layer axis — the `xs` for a `lax.scan` over the (homogeneous) block stack. Static
    fields (attn head counts) ride in the treedef, shared across iterations."""
    return jax.tree.map(lambda *per_layer: jnp.stack(per_layer), *layers)


def _tap_layer(key: str) -> int:
    """Global block index a `read_activations` key reads at: the block a `resid.{L}` tap
    enters, or the block a decomposed site lives in."""
    if key.startswith("resid."):
        return int(key.split(".")[1])
    return parse_site_name(key)[0]


def _per_kind_dims(components: DecompVU) -> dict[str, tuple[int, int, int]]:
    """Per decomposed KIND, the `(d_in, C, d_out)` shared across its layers — asserting
    uniformity, the precondition for the layer-`lax.scan` masked forward (it stacks each
    kind across layers, so every layer's matrix of that kind must be the same shape)."""
    kind_dims: dict[str, tuple[int, int, int]] = {}
    for name, (V, U) in components.vu.items():
        kind = parse_site_name(name)[1]
        dims = (V.shape[0], V.shape[1], U.shape[1])
        assert kind_dims.setdefault(kind, dims) == dims, (
            f"per-kind dims must be uniform across layers for the scan masked forward: "
            f"{kind} {dims} != {kind_dims[kind]}"
        )
    return kind_dims


def _stack_per_kind_vu(components: DecompVU, n_layers: int) -> dict[str, dict[str, Array]]:
    """Per decomposed KIND, the layer-stacked `(V, U)` arrays — the MASK-INDEPENDENT part of
    the scan inputs (a leading layer axis, one homogeneous body across layers). Mask/live/
    delta/route are attached per-forward by `_attach_per_kind_masks`; the V/U stack +
    `_reconstruct_compute_weights` (the ÷N→÷fsdp cross-node gather) are the same for EVERY
    forward in a step, so they are built ONCE via `prepare_compute_weights` and shared.
    Per-kind dims (d_in, C, d_out) must be uniform across layers (asserted in `_per_kind_dims`)."""
    kind_dims = _per_kind_dims(components)
    vu_dt = next(iter(components.vu.values()))[0].dtype
    per_kind: dict[str, dict[str, Array]] = {}
    for kind, (d_in, C, d_out) in kind_dims.items():
        names = [site_name(layer, kind) for layer in range(n_layers)]
        Vs = jnp.stack(
            [
                components.vu[n][0] if n in components.vu else jnp.zeros((d_in, C), vu_dt)
                for n in names
            ]
        )
        Us = jnp.stack(
            [
                components.vu[n][1] if n in components.vu else jnp.zeros((C, d_out), vu_dt)
                for n in names
            ]
        )
        per_kind[kind] = {"V": Vs, "U": Us}
    return per_kind


def _attach_per_kind_masks(
    prepared: dict[str, dict[str, Array]],
    n_layers: int,
    leading: tuple[int, ...],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live_set: frozenset[str],
    has_delta: bool,
) -> dict[str, dict[str, Array]]:
    """Attach the per-forward `(live, mask[, delta][, route])` stacks to the shared, already
    stacked + ÷fsdp-reconstructed `prepared` per-kind `(V, U)` weights. Sites absent from
    `live` get dummy mask/delta/route (the `cond` frozen branch ignores them); `masks`/
    `delta_masks`/`routes` exist only for live sites (recon builds them per-chunk)."""
    # Dummy mask/delta/route shapes match the REAL entries (the source scope sets the leading
    # shape: `sc` broadcasts over batch as `(1, T)`, not the full `(B, T)`).
    a_mask = next(iter(masks.values())) if masks else None
    mask_lead = a_mask.shape[:-1] if a_mask is not None else leading
    a_delta = next(iter(delta_masks.values())) if (has_delta and delta_masks) else None
    a_route = next(iter(routes.values())) if (routes and len(routes)) else None

    per_kind: dict[str, dict[str, Array]] = {}
    for kind, vu_entry in prepared.items():
        C = vu_entry["V"].shape[-1]
        mask_dt = a_mask.dtype if a_mask is not None else vu_entry["V"].dtype
        names = [site_name(layer, kind) for layer in range(n_layers)]
        live_flags = jnp.array([n in live_set for n in names])
        masks_k = jnp.stack(
            [masks[n] if n in live_set else jnp.ones((*mask_lead, C), mask_dt) for n in names]
        )
        entry: dict[str, Array] = {**vu_entry, "live": live_flags, "mask": masks_k}
        if has_delta:
            d_shape = a_delta.shape if a_delta is not None else leading
            d_dt = a_delta.dtype if a_delta is not None else mask_dt
            entry["delta"] = jnp.stack(
                [delta_masks[n] if n in live_set else jnp.zeros(d_shape, d_dt) for n in names]
            )
        if routes is not None:
            r_shape = a_route.shape if a_route is not None else leading
            r_dt = a_route.dtype if a_route is not None else jnp.bool_
            entry["route"] = jnp.stack(
                [routes[n] if n in live_set else jnp.zeros(r_shape, r_dt) for n in names]
            )
        per_kind[kind] = entry
    return per_kind


def _stack_ci_per_kind(ci_lower: dict[str, Array], n_layers: int) -> dict[str, Array]:
    """Stack the per-site CI envelope into per-kind `[n_layer, *leading, C]` — built ONCE per
    step and SHARED across every stochastic recon forward (the CI envelope is identical for
    all). Mirrors `_stack_per_kind_vu`: the shared stack replaces N per-forward mask stacks."""
    kinds: dict[str, Array] = {}
    sample_by_kind: dict[str, Array] = {}
    for name, v in ci_lower.items():
        sample_by_kind.setdefault(parse_site_name(name)[1], v)
    for kind, sample in sample_by_kind.items():
        names = [site_name(layer, kind) for layer in range(n_layers)]
        kinds[kind] = jnp.stack(
            [ci_lower[n] if n in ci_lower else jnp.zeros_like(sample) for n in names]
        )
    return kinds


def _attach_per_kind_stochastic(
    prepared: dict[str, dict[str, Array]],
    n_layers: int,
    leading: tuple[int, ...],
    ci_stacked: dict[str, Array],
    draw_key: Array,
    routes: dict[str, Array] | None,
    live_set: frozenset[str],
) -> dict[str, dict[str, Array]]:
    """Stochastic recon: attach the SHARED per-kind `ci` stack + per-(layer,kind) RNG keys
    instead of pre-built mask/delta stacks. `masked_site` draws `source = uniform(key)` and
    builds `mask = ci + (1−ci)·source` INSIDE the checkpointed block, so the per-forward mask
    is recomputed in the backward (faithful by checkpoint determinism — same key fwd+bwd) and
    never held. Only the (tiny) keys + live-flags are per-forward; the `[n_layer,*,C]` ci stack
    is shared, so N forwards' mask stacks collapse to one ci stack (the memory win)."""
    src_base, delta_base = jax.random.split(draw_key)
    a_route = next(iter(routes.values())) if (routes and len(routes)) else None
    per_kind: dict[str, dict[str, Array]] = {}
    for kind, vu_entry in prepared.items():
        names = [site_name(layer, kind) for layer in range(n_layers)]
        live_flags = jnp.array([n in live_set for n in names])
        kind_idx = KIND_ORDER.index(kind)
        src_keys = jnp.stack(
            [
                jax.random.fold_in(jax.random.fold_in(src_base, kind_idx), layer)
                for layer in range(n_layers)
            ]
        )
        delta_keys = jnp.stack(
            [
                jax.random.fold_in(jax.random.fold_in(delta_base, kind_idx), layer)
                for layer in range(n_layers)
            ]
        )
        entry: dict[str, Array] = {
            **vu_entry, "live": live_flags, "ci": ci_stacked[kind],
            "src_key": src_keys, "delta_key": delta_keys,
        }  # fmt: skip
        if routes is not None:
            r_shape = a_route.shape if a_route is not None else leading
            r_dt = a_route.dtype if a_route is not None else jnp.bool_
            entry["route"] = jnp.stack(
                [routes[n] if n in live_set else jnp.zeros(r_shape, r_dt) for n in names]
            )
        per_kind[kind] = entry
    return per_kind


def _reconstruct_compute_weights(
    per_kind: dict[str, dict[str, Array]],
    fp8: bool,
) -> dict[str, dict[str, Array]]:
    """The ZeRO-1 weight reconstruction (pure-HSDP backup layout). The stacked
    `[n_layer, d_in, C]` / `[n_layer, C, d_out]` compute weights arrive with their FSDP dim
    sharded ÷N over the FULL mesh (the master is `P(("replicate","fsdp"), ...)`). Reconstruct
    them to the `fsdp`-sharded (÷fsdp) COMPUTE layout here — BEFORE the layer scan — so:

      * the cross-`replicate` gather runs ONCE per step in ENTRY (off the hot path),
        landing a SMALL ÷fsdp-resident weight stack (`[n_layer, d_in/fsdp, C]`), NOT the full
        `[n_layer, d_in, C]` model;
      * the per-layer scan body then gathers ONE layer's `fsdp` shard to full d_in
        transiently (intra-node NVLink), freed each iteration — never a full-model resident.

    Cast to bf16 HERE (not f32) so the ÷fsdp-resident compute stack is half-size and XLA
    can't keep an f32 full copy alive. The fp32 masters + Adam stay ÷N (untouched — this is
    a separate read-only compute view). The leading `n_layer` axis (the scan `xs`) stays
    unsharded. No-op off-mesh (CPU / single device); `run.py` sets the global mesh."""
    if jax.sharding.get_abstract_mesh().empty:
        return per_kind
    v_spec = P(None, "fsdp", "tp")  # [n_layer, d_in ÷fsdp, C ÷tp] — d gathered/step, C stays ÷tp
    u_spec = P(None, "tp", "fsdp")  # [n_layer, C ÷tp, d_out ÷fsdp]
    out: dict[str, dict[str, Array]] = {}
    for kind, entry in per_kind.items():
        pinned = dict(entry)
        # optimization_barrier forces the cast/quant to materialize BEFORE the ÷N→÷fsdp gather,
        # so the collective moves the compute dtype — XLA otherwise sinks the convert past the
        # all-gather and gathers the f32 master (2x the comm; the convert gathers in the HLO).
        if fp8:
            # Quantized All-Gather: the ÷fsdp compute weights are fp8; the per-layer ÷fsdp→full
            # gather then moves fp8 (½ the bf16 bytes), dequantized to bf16 in `masked_site`.
            # Per-tensor scalar scale rides alongside (replicated, survives the gather).
            vq, vs = quantize_fp8(entry["V"])
            uq, us = quantize_fp8(entry["U"])
            pinned["V"] = jax.lax.with_sharding_constraint(jax.lax.optimization_barrier(vq), v_spec)
            pinned["U"] = jax.lax.with_sharding_constraint(jax.lax.optimization_barrier(uq), u_spec)
            pinned["V_scale"], pinned["U_scale"] = vs, us
        else:
            pinned["V"] = jax.lax.with_sharding_constraint(
                jax.lax.optimization_barrier(entry["V"].astype(jnp.bfloat16)), v_spec
            )
            pinned["U"] = jax.lax.with_sharding_constraint(
                jax.lax.optimization_barrier(entry["U"].astype(jnp.bfloat16)), u_spec
            )
        out[kind] = pinned
    return out


class LlamaDecomposedModel(eqx.Module):
    """The Llama-8B `DecomposedModel` (the `lm.py` contract; SPEC §1).

    Carries the FROZEN full model (embedding, all blocks, final norm, lm_head) as array
    fields — so it threads into the jitted step as a pytree arg, its weights traced not
    baked. The TRAINABLE V/U (`vu: DecompVU`) is passed to the forward methods explicitly:
    separate lifecycle (own optimizer + checkpoint, C-sharded while these weights
    replicate), so it is NOT a field here.

    Forward methods take token `inputs` and embed internally. Blocks with no decomposed
    site run the plain frozen path — so a subset decomposition just leaves the rest
    frozen.

    `sites` / `leading_axes` are static config."""

    embed: Float[Array, "vocab d"]
    stacked: LlamaLayer  # the per-layer weights stacked on a leading layer axis (the scan
    # `xs`), stored pre-stacked: a saved jit input, never re-stacked inside a forward.
    n_layer: int = eqx.field(static=True)
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    leading_axes: tuple[str, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    scan_unroll: int = eqx.field(static=True, default=1)
    """`lax.scan(unroll=)` factor over the block stack (`RuntimeConfig.scan_unroll`); 1 =
    plain per-layer scan."""
    gather_fp8: bool = eqx.field(static=True, default=False)
    """Quantized all-gather of the ÷fsdp compute V/U (`RuntimeConfig.gather_fp8`)."""

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    @property
    def layers(self) -> list[LlamaLayer]:
        """Per-layer view of `stacked` (slices the leading layer axis). For non-hot
        consumers (attn-patterns recipe, equivalence harness); the forwards use `stacked`."""
        return [jax.tree.map(lambda a, idx=i: a[idx], self.stacked) for i in range(self.n_layer)]

    def shardings(self, mesh: "Mesh") -> "LlamaDecomposedModel":
        """FSDP-on-`fsdp` the per-layer weights (`stacked.shardings` — `d` on `fsdp`,
        head/intermediate replicated; the ~14 GB layer bulk shards `/fsdp`, gathered per layer
        inside the scan, on NVLink). embed / lm_head / norm / inv_freq REPLICATE — the ~2 GB
        embed+head is small and vocab-parallel logits/lookup aren't worth the complexity. (The
        old all-replicate justification — "the target is small vs activations" — is stale: at
        the full 32-layer model the replicated target + its backward/remat copies dominate the
        step's peak, which is what this shards away.)"""
        repl = NamedSharding(mesh, P())
        return eqx.tree_at(
            lambda m: (m.embed, m.norm, m.lm_head, m.inv_freq, m.stacked),
            self,
            (repl, repl, repl, repl, self.stacked.shardings(mesh)),
        )

    @staticmethod
    def recon_loss_fn(masked_output: Array, clean_output: Array) -> Array:
        return kl_per_position(masked_output, clean_output)

    def embed_tokens(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t d"]:
        return self.embed[tokens]

    def clean_output(self, inputs: Int[Array, "b t"]) -> Array:
        """The all-frozen forward — the recon target (SPEC S3). A `lax.scan` over the block
        stack so XLA compiles one block body instead of unrolling all 32 layers (the compile
        fix for the full model; the scan reassociates float ops vs an unrolled loop, within
        fp32 tolerance)."""

        def block(x: Array, layer: LlamaLayer) -> tuple[Array, None]:
            x = x + layer.attn(rms_norm(x, layer.ln1, self.eps), self.inv_freq)
            x = x + _clean_mlp_out(layer, rms_norm(x, layer.ln2, self.eps))
            return x, None

        x, _ = jax.lax.scan(block, self.embed_tokens(inputs), self.stacked)
        x = rms_norm(x, self.norm, self.eps)
        return x @ self.lm_head.T

    def read_activations(
        self, inputs: Int[Array, "b t"], wanted: tuple[str, ...]
    ) -> dict[str, Array]:
        """Frozen-path activation accessor (CI input side, SPEC S4; harvest's per-site
        matrix inputs).

        `wanted` keys are either `resid.{layer}` (residual stream ENTERING that block — the
        chunkwise CI fn's `input_names`) or a decomposed SITE NAME (the activation entering
        that site's weight on the frozen path: `q/k/v_proj` ← post-LN1 residual, `o_proj` ←
        the attention output, `gate/up_proj` ← post-LN2 residual, `down_proj` ←
        `silu(gate)·up`). The residual is threaded identically to `clean_output`; the
        per-site intermediates come from the same RMSNorm/attn/MLP math. Stops once the last
        requested key's block is fully covered (no wasted block compute past it)."""
        wanted_set = frozenset(wanted)
        last = max(_tap_layer(key) for key in wanted)
        taps: dict[str, Array] = {}
        x = self.embed_tokens(inputs)
        for layer in range(self.n_layer):
            block = jax.tree.map(lambda a, li=layer: a[li], self.stacked)
            if f"resid.{layer}" in wanted_set:
                taps[f"resid.{layer}"] = x
            attn = block.attn
            h1 = rms_norm(x, block.ln1, self.eps)
            attn_y = attn.core(h1 @ attn.wq.T, h1 @ attn.wk.T, h1 @ attn.wv.T, self.inv_freq)
            post_attn = x + attn_y @ attn.wo.T
            mlp_in = rms_norm(post_attn, block.ln2, self.eps)
            down_in = jax.nn.silu(mlp_in @ block.Wg.T) * (mlp_in @ block.Wu.T)
            for kind, site_input in (
                ("q", h1), ("k", h1), ("v", h1), ("o", attn_y),
                ("gate", mlp_in), ("up", mlp_in), ("down", down_in),
            ):  # fmt: skip
                name = site_name(layer, kind)
                if name in wanted_set:
                    taps[name] = site_input
            x = post_attn + down_in @ block.Wd.T
            if layer == last:
                break
        assert set(taps) == wanted_set, (sorted(taps), sorted(wanted))
        return taps

    def _run_masked_forward(
        self,
        prepared: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
        remat: bool,
        collect: dict[str, Array] | None,
        stochastic: tuple[dict[str, Array], Array] | None = None,
    ) -> Array:
        """The masked decomposed forward shared by `masked_output` and `masked_site_outputs`
        (SPEC §1.3, S2), as a `lax.scan` over the block stack with a per-site `lax.cond`: a
        site in `live` runs its decomposed forward (`masks[s]`/`delta_masks[s]`/`routes[s]`),
        every other site — and every site absent from the decomposition — runs the frozen
        `x @ W`. One block body compiles regardless of depth / chunk count; `cond` runs only
        the taken branch so frozen sites do no V@U. `live`/`has_delta` are static; a non-None
        `collect` gathers per-live-site decomposed outputs (SPEC S31). Requires per-kind dims
        uniform across layers (asserted) — the layer stack must be homogeneous to scan.

        `prepared` is the shared, stacked + ÷fsdp-reconstructed per-kind `(V, U)` from
        `prepare_compute_weights` (built ONCE per step) — this fn only ATTACHES the per-forward
        masks, so the ÷N→÷fsdp cross-node gather is not re-run here (SPEC unchanged; numerics
        identical — the reconstruction is mask-independent and the same for every forward)."""
        live_set = frozenset(live)
        resid = self.embed_tokens(inputs)
        leading = resid.shape[:-1]
        if stochastic is not None:
            ci_stacked, draw_key = stochastic
            per_kind = _attach_per_kind_stochastic(
                prepared, self.n_layer, leading, ci_stacked, draw_key, routes, live_set
            )
        else:
            per_kind = _attach_per_kind_masks(
                prepared, self.n_layer, leading, masks, delta_masks, routes, live_set, has_delta
            )
        decomposed_kinds = frozenset(per_kind)
        want_collect = collect is not None

        # STATIC liveness. `live_set` is known at trace, so the live/frozen choice per site needs
        # NO runtime `lax.cond` — and removing it lets XLA pack + prefetch the V/U gathers (the
        # cond was a scheduling/packing barrier). We assume LAYER-ALIGNED, CONTIGUOUS chunks
        # (every production plan: `into_groups` with sites_per_chunk % n_decomposed_kinds == 0,
        # and `one_chunk`); both are asserted below. The forward is then
        # [frozen prefix] → [live block] → [frozen suffix], each a static sub-scan; only the live
        # block carries V/U and gathers them.
        def layer_is_live(layer: int) -> bool:
            flags = {site_name(layer, kind) in live_set for kind in decomposed_kinds}
            assert len(flags) == 1, (
                f"layer {layer} is partially live ({flags}); the segmented masked forward assumes "
                f"layer-aligned chunks (sites_per_chunk % {len(decomposed_kinds)} == 0)"
            )
            return flags.pop()

        live_layers = [layer for layer in range(self.n_layer) if layer_is_live(layer)]
        if live_layers:
            first_live, last_live = live_layers[0], live_layers[-1] + 1
            assert live_layers == list(range(first_live, last_live)), (
                f"live layers must be contiguous, got {live_layers}"
            )
        else:
            first_live = last_live = 0

        def decomp_site(x_in: Array, W: Array, e: dict[str, Array]) -> Array:
            v, u = e["V"], e["U"]
            if "V_scale" in e:  # fp8 QAG: gather the fp8 ÷fsdp weight to full d (½ bytes on the
                # wire), THEN dequant to bf16 — the barrier keeps the convert after the gather so
                # the collective moves fp8, not bf16.
                v = dequantize_fp8(
                    jax.lax.optimization_barrier(
                        jax.lax.with_sharding_constraint(v, P(None, "tp"))
                    ),
                    e["V_scale"],
                )
                u = dequantize_fp8(
                    jax.lax.optimization_barrier(
                        jax.lax.with_sharding_constraint(u, P("tp", None))
                    ),
                    e["U_scale"],
                )
            if "ci" in e:  # stochastic recompute: draw source from the per-layer key and build the
                # mask INLINE (recomputed in the backward, not held — the shared `ci` stack + tiny
                # key replace the per-forward mask stack).
                ci = e["ci"]
                source = jax.random.uniform(e["src_key"], ci.shape, dtype=ci.dtype)
                mask = ci + (1.0 - ci) * source
                delta = (
                    jax.random.uniform(e["delta_key"], ci.shape[:-1], dtype=ci.dtype)
                    if has_delta
                    else None
                )
                return site_out(x_in, v, u, W, mask, delta, e.get("route"))
            return site_out(x_in, v, u, W, e["mask"], e.get("delta"), e.get("route"))

        def masked_site(
            x_in: Array, kind: str, W: Array, pk: dict[str, dict[str, Array]]
        ) -> tuple[Array, Array | None]:
            # LIVE block only: every decomposed kind decomps (static — no cond); a kind absent
            # from the decomposition stays frozen.
            if kind not in decomposed_kinds:
                return x_in @ W.T, None
            out = decomp_site(x_in, W, pk[kind])
            return out, (out if want_collect else None)

        def live_block(
            x: Array, layer_in: tuple[LlamaLayer, dict[str, dict[str, Array]]]
        ) -> tuple[Array, dict[str, Array] | None]:
            sl, pk = layer_in
            attn = sl.attn
            h1 = rms_norm(x, sl.ln1, self.eps)
            q, qc = masked_site(h1, "q", attn.wq, pk)
            k, kc = masked_site(h1, "k", attn.wk, pk)
            v, vc = masked_site(h1, "v", attn.wv, pk)
            attn_y = attn.core(q, k, v, self.inv_freq)
            o, oc = masked_site(attn_y, "o", attn.wo, pk)
            post_attn = x + o
            h2 = rms_norm(post_attn, sl.ln2, self.eps)
            g, gc = masked_site(h2, "gate", sl.Wg, pk)
            u, uc = masked_site(h2, "up", sl.Wu, pk)
            d, dc = masked_site(jax.nn.silu(g) * u, "down", sl.Wd, pk)
            x = post_attn + d
            collected = (
                {
                    name: c
                    for name, c in (("q", qc), ("k", kc), ("v", vc), ("o", oc),
                                    ("gate", gc), ("up", uc), ("down", dc))
                    if c is not None
                }
                if want_collect
                else None
            )  # fmt: skip
            return x, collected

        def frozen_block(x: Array, sl: LlamaLayer) -> tuple[Array, None]:
            # Bit-identical to a frozen `masked_site` branch (`x @ Wᵀ` per site), shared with
            # `clean_output`. Carries NO V/U → a frozen segment gathers nothing.
            x = x + sl.attn(rms_norm(x, sl.ln1, self.eps), self.inv_freq)
            x = x + _clean_mlp_out(sl, rms_norm(x, sl.ln2, self.eps))
            return x, None

        # Per-LAYER checkpoint of the scan BODY in BOTH modes — `remat` controls ONLY whether the
        # layer ACTIVATIONS are recomputed; it NEVER controls the ÷fsdp→full V/U gather. That
        # gather is a NON-dot collective, so it is never a saved residual under either policy — it
        # re-gathers in the backward, transient one layer at a time (without checkpoint XLA keeps
        # every layer's full gathered V/U live across the scan → OOM).
        #   remat=True  → nothing_saveable: recompute activations AND the gather (min memory).
        #   remat=False → dots_saveable: SAVE the activation matmuls (no batch dims here → they
        #     qualify) but still recompute the gather + cheap elementwise. Pure recompute either
        #     way; zero numerics change.
        # `scan_unroll` (native `lax.scan(unroll=k)`) emits k iterations straight-line so XLA can
        # prefetch gather(L+1) under matmul(L) — the overlap a 1-layer while-body denies.
        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )

        def run_scan(body: Any, carry: Array, xs: Any) -> tuple[Array, Any]:
            return jax.lax.scan(
                jax.checkpoint(body, policy=policy), carry, xs, unroll=self.scan_unroll
            )

        def slice_layers(lo: int, hi: int) -> LlamaLayer:
            return jax.tree.map(lambda a: a[lo:hi], self.stacked)

        x = resid
        ys: dict[str, Array] | None = None
        if first_live > 0:
            x, _ = run_scan(frozen_block, x, slice_layers(0, first_live))
        if last_live > first_live:
            pk_live = {
                kind: {k: v[first_live:last_live] for k, v in e.items()}
                for kind, e in per_kind.items()
            }
            x, ys = run_scan(live_block, x, (slice_layers(first_live, last_live), pk_live))
        if last_live < self.n_layer:
            x, _ = run_scan(frozen_block, x, slice_layers(last_live, self.n_layer))

        x = rms_norm(x, self.norm, self.eps)
        logits = x @ self.lm_head.T
        if collect is not None:
            assert ys is not None  # collect requested -> the live block emitted per-kind outputs
            for site in live:
                layer, kind = parse_site_name(site)
                collect[site] = ys[kind][layer - first_live]
        return logits

    def prepare_compute_weights(self, vu: DecompVU) -> dict[str, dict[str, Array]]:
        """Build the shared per-kind compute weights ONCE per step (SPEC unchanged): stack the
        per-site V/U into the layer-stacked `[n_layer, …]` form and run the ÷N→÷fsdp cross-node
        reconstruction + bf16 cast. The result is mask-independent and identical for every
        forward in the step, so the engine builds it once and threads it into all
        `masked_output` / `masked_site_outputs` calls — the cross-node gather then runs ONCE per
        step (ENTRY) instead of once per forward (the per-forward re-gather was ~10 co-resident
        copies of the ÷fsdp stack at peak)."""
        return _reconstruct_compute_weights(_stack_per_kind_vu(vu, self.n_layer), self.gather_fp8)

    def masked_output(
        self,
        prepared: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
        *,
        remat: bool,
    ) -> Array:
        return self._run_masked_forward(
            prepared, inputs, masks, delta_masks, routes, live, has_delta, remat, None
        )

    def stack_ci(self, ci_lower: dict[str, Array]) -> dict[str, Array]:
        """Per-kind `[n_layer, *leading, C]` stack of the CI envelope, built ONCE per step and
        shared across all stochastic recon forwards (`masked_output_stochastic`). The
        StochasticReconCapable capability (SPEC unchanged — pure recompute restructuring)."""
        return _stack_ci_per_kind(ci_lower, self.n_layer)

    def masked_output_stochastic(
        self,
        prepared: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        ci_stacked: dict[str, Array],
        draw_key: Array,
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
        *,
        remat: bool,
    ) -> Array:
        """Stochastic recon forward that RECOMPUTES masks in-block (memory win): the shared
        `ci_stacked` + per-layer keys from `draw_key` replace the per-forward mask stack; each
        live site draws `source = uniform(key)` and forms `mask = ci + (1−ci)·source` inside the
        checkpointed block (faithful by checkpoint determinism). Same forward semantics as
        `masked_output` with stochastic sources — only the masks' liverange changes."""
        return self._run_masked_forward(
            prepared, inputs, {}, {}, routes, live, has_delta, remat, None, (ci_stacked, draw_key)
        )

    def masked_site_outputs(
        self,
        prepared: dict[str, dict[str, Array]],
        inputs: Int[Array, "b t"],
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, Array]:
        """Per-`live`-site decomposed output of the masked forward (SPEC S31). Runs the
        exact `masked_output` forward, discards the logits, returns the collected outputs."""
        collect: dict[str, Array] = {}
        self._run_masked_forward(
            prepared, inputs, masks, delta_masks, routes, live, has_delta, False, collect
        )
        assert set(collect) == set(live), (sorted(collect), sorted(live))
        return collect

    def weight_deltas(self, vu: DecompVU) -> dict[str, Array]:
        """fp32 `W − V@U` per site from fp32 masters (SPEC N2; faithfulness input)."""
        out: dict[str, Array] = {}
        for spec in self.sites:
            layer, kind = parse_site_name(spec.name)
            W = _frozen_site_weight(jax.tree.map(lambda a, li=layer: a[li], self.stacked), kind)
            V, U = vu.site(spec.name)
            out[spec.name] = (
                W.astype(jnp.float32) - (V.astype(jnp.float32) @ U.astype(jnp.float32)).T
            )
        return out


# ----------------------------- HF weight loading -----------------------------


def _hf_snapshot_dir(model_name: str) -> Path:
    """Newest local snapshot of `model_name`. `HF_HUB_CACHE` overrides; otherwise on
    cluster (`DATA_MOUNT` set) the shared world-readable cache is the source — a home
    `~/.cache` hub is silently mutable, and a wiped entry strands running jobs that
    reload weights on requeue."""
    import os

    default_cache = (
        f"{os.environ['DATA_MOUNT']}/artifacts/hf_cache/hub"
        if "DATA_MOUNT" in os.environ
        else str(Path.home() / ".cache/huggingface/hub")
    )
    cache = Path(os.environ.get("HF_HUB_CACHE", default_cache))
    repo = "models--" + model_name.replace("/", "--")
    snaps = sorted((cache / repo / "snapshots").iterdir())
    assert snaps, f"no snapshot for {model_name} under {cache}"
    return snaps[-1]


class _HFWeights:
    """Lazy keyed access to the sharded safetensors of an HF Llama checkpoint."""

    def __init__(self, snapshot: Path):
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        self._key_to_file = index["weight_map"]
        self._snapshot = snapshot
        self._open: dict[str, Any] = {}

    def get(self, key: str) -> Array:
        fname = self._key_to_file[key]
        if fname not in self._open:
            self._open[fname] = safe_open(str(self._snapshot / fname), framework="numpy")
        return jnp.asarray(np.array(self._open[fname].get_tensor(key)), dtype=DT)


def _load_attn(w: _HFWeights, i: int, cfg: LlamaConfig) -> FrozenAttn:
    pre = "model.layers"
    return FrozenAttn(
        wq=w.get(f"{pre}.{i}.self_attn.q_proj.weight"),
        wk=w.get(f"{pre}.{i}.self_attn.k_proj.weight"),
        wv=w.get(f"{pre}.{i}.self_attn.v_proj.weight"),
        wo=w.get(f"{pre}.{i}.self_attn.o_proj.weight"),
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        head_dim=cfg.head_dim,
        n_rep=cfg.n_rep,
    )


def _load_blocks(w: "_HFWeights", cfg: LlamaConfig) -> list[LlamaLayer]:
    pre = "model.layers"
    return [
        LlamaLayer(
            ln1=w.get(f"{pre}.{i}.input_layernorm.weight"),
            ln2=w.get(f"{pre}.{i}.post_attention_layernorm.weight"),
            attn=_load_attn(w, i, cfg),
            Wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
            Wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
            Wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
        )
        for i in range(cfg.n_layer)
    ]


def build_decomposed_lm(
    embed: Array,
    layers: list[LlamaLayer],
    norm: Array,
    lm_head: Array,
    inv_freq: Array,
    cfg: LlamaConfig,
    sites: tuple[SiteSpec, ...],
    scan_unroll: int = 1,
    gather_fp8: bool = False,
) -> LlamaDecomposedModel:
    """Assemble a `LlamaDecomposedModel` from the frozen full-model arrays + decomposition
    config. `sites` must be canonical-ordered with dims matching `cfg`. `scan_unroll` /
    `gather_fp8` are the `RuntimeConfig` compute knobs (1 / off = the default forward)."""
    site_cs = tuple(SiteC(s.name, s.C) for s in sites)
    assert sites == llama_site_specs(cfg, canonical_site_cs(site_cs)), (
        f"sites are not the canonical specs for this config: {sites}"
    )
    return LlamaDecomposedModel(
        embed=embed,
        stacked=_stack_layers(layers),
        n_layer=len(layers),
        norm=norm,
        lm_head=lm_head,
        inv_freq=inv_freq,
        sites=sites,
        leading_axes=("sequence",),
        eps=cfg.rms_norm_eps,
        scan_unroll=scan_unroll,
        gather_fp8=gather_fp8,
    )


def load_decomposed_lm_from_hf(
    model_name: str,
    cfg: LlamaConfig,
    sites: tuple[SiteSpec, ...],
    scan_unroll: int = 1,
    gather_fp8: bool = False,
) -> LlamaDecomposedModel:
    """Load the Llama-8B `DecomposedModel`: the full frozen model (embedding, all blocks,
    final norm, lm_head) as fields plus the static decomposition config (`sites`). Blocks
    without a decomposed site run the plain frozen path. `scan_unroll` / `gather_fp8` are the
    `RuntimeConfig` compute knobs."""
    w = _HFWeights(_hf_snapshot_dir(model_name))
    return build_decomposed_lm(
        embed=w.get("model.embed_tokens.weight"),
        layers=_load_blocks(w, cfg),
        norm=w.get("model.norm.weight"),
        lm_head=w.get("lm_head.weight"),
        inv_freq=llama3_inv_freq(cfg),
        cfg=cfg,
        sites=sites,
        scan_unroll=scan_unroll,
        gather_fp8=gather_fp8,
    )
