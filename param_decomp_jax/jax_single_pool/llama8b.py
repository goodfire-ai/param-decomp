"""Llama-3.1-8B vendored target — the first `DecomposedModel` implementation.

The decomposed sites are any per-layer weight matrices (SPEC §1/§3) named torch-style:
`layers.{i}.self_attn.{q,k,v,o}_proj` and `layers.{i}.mlp.{gate,up,down}_proj`, each
with its own C. The frozen residual-start suffix runs from the lowest decomposed layer
(`first_decomposed_layer`) to the LM head as a `Target` pytree threaded as a runtime
arg; suffix layers without sites run the plain frozen block.

q/k/v sites are decomposed BEFORE RoPE/SDPA (the masked site output feeds the
attention math); the o site applies to the attention output. V/U masters are fp32
keyed per site (`DecompVU`); frozen weights are stored bf16 (SPEC N1) — the trainer
casts for compute.

Real HF weights load straight from the cached safetensors (no torch dep).
"""

import json
import re
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from safetensors import safe_open
from vendored_jax.llama import (
    LlamaConfig,
    apply_rope,
    causal_sdpa,
    llama3_inv_freq,
    repeat_kv,
    rms_norm,
    rope_cos_sin,
)

from jax_single_pool.lm import DecomposedModel, SiteC, SiteSpec

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


def first_decomposed_layer(site_names: tuple[str, ...]) -> int:
    """The residual-start boundary: the suffix runs from the lowest decomposed layer."""
    assert site_names
    return min(parse_site_name(name)[0] for name in site_names)


# ----------------------------- frozen suffix -----------------------------


class FrozenAttn(eqx.Module):
    wq: Float[Array, "qd d"]
    wk: Float[Array, "kvd d"]
    wv: Float[Array, "kvd d"]
    wo: Float[Array, "d qd"]
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    n_rep: int = eqx.field(static=True)

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
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        return causal_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(b, t, self.n_head * self.head_dim)

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Array) -> Array:
        return self.core(x @ self.wq.T, x @ self.wk.T, x @ self.wv.T, inv_freq) @ self.wo.T


class FrozenMLP(eqx.Module):
    wg: Float[Array, "di d"]
    wu: Float[Array, "di d"]
    wd: Float[Array, "d di"]

    def __call__(self, x: Array) -> Array:
        return (jax.nn.silu(x @ self.wg.T) * (x @ self.wu.T)) @ self.wd.T


class FrozenBlock(eqx.Module):
    """A fully-frozen transformer block — the `Prefix` building block."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    mlp: FrozenMLP
    eps: float = eqx.field(static=True)

    def __call__(self, x: Array, inv_freq: Array) -> Array:
        x = x + self.attn(rms_norm(x, self.ln1, self.eps), inv_freq)
        x = x + self.mlp(rms_norm(x, self.ln2, self.eps))
        return x


class SuffixLayer(eqx.Module):
    """One suffix layer's frozen weights — norms, attention, MLP. Decomposed sites read
    their frozen target W from here at forward time; layers without sites run the
    plain frozen block from the same fields. Weights pass as a runtime arg — never
    baked into the HLO as a multi-GB constant."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    Wg: Float[Array, "di d"]
    Wu: Float[Array, "di d"]
    Wd: Float[Array, "d di"]


class Target(eqx.Module):
    """Frozen residual-start suffix: layers `first_decomposed_layer..n_layer-1`, final
    norm, lm_head."""

    layers: list[SuffixLayer]
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]
    eps: float = eqx.field(static=True)


def _frozen_site_weight(suffix_layer: SuffixLayer, kind: str) -> Array:
    match kind:
        case "q":
            return suffix_layer.attn.wq
        case "k":
            return suffix_layer.attn.wk
        case "v":
            return suffix_layer.attn.wv
        case "o":
            return suffix_layer.attn.wo
        case "gate":
            return suffix_layer.Wg
        case "up":
            return suffix_layer.Wu
        case "down":
            return suffix_layer.Wd
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


# ----------------------------- decomposed V/U (per site) -----------------------------


class DecompVU(eqx.Module):
    """fp32 master V `(d_in, C_s)` / U `(C_s, d_out)` per decomposed site, keyed by
    site name."""

    vu: dict[str, tuple[Float[Array, "d_in C"], Float[Array, "C d_out"]]]

    def site(self, name: str) -> tuple[Array, Array]:
        return self.vu[name]


def init_decomp_vu(sites: tuple[SiteSpec, ...], key: Array) -> DecompVU:
    """Small random fp32 V ~ N(0, d_in^-0.5), U ~ N(0, C^-0.5) per site; the
    weight-delta channel carries the faithfulness residual at init (before
    faithfulness warmup)."""
    keys = jax.random.split(key, 2 * len(sites))
    vu: dict[str, tuple[Array, Array]] = {}
    for site_idx, spec in enumerate(sites):
        V = jax.random.normal(keys[2 * site_idx], (spec.d_in, spec.C)) * spec.d_in**-0.5
        U = jax.random.normal(keys[2 * site_idx + 1], (spec.C, spec.d_out)) * spec.C**-0.5
        vu[spec.name] = (V, U)
    return DecompVU(vu=vu)


# ----------------------------- forwards -----------------------------


def _site_out(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
) -> Array:
    """One decomposed linear (SPEC §1.3): `((x@V)*m)@U + (x@Δ)*d`, routed per position
    against the frozen `x @ W.T`. `mask` may be None (fully on); `route` None routes
    everywhere. `delta_mask` None drops the delta path entirely (constant-source entries
    carry no delta, LOSS_PARITY_DESIGN §4b). `delta_mask`/`route` broadcast over batch;
    trailing dim added here."""
    acts = x @ V
    if mask is not None:
        acts = acts * mask
    out = acts @ U
    if delta_mask is not None:
        # DIVERGENCE from torch-under-autocast (documented, accepted): this delta is
        # computed in bf16 from the cast components; torch computes W − V@U in fp32 then
        # casts at the einsum. bf16-rounding-level difference on the delta PATH only —
        # the faithfulness loss uses the fp32 `weight_deltas` (SPEC N2), not this.
        delta = W - (V @ U).T  # (d_out, d_in)
        out = out + delta_mask[..., None] * (x @ delta.T)
    if route is not None:
        out = jnp.where(route[..., None], out, x @ W.T)
    return out


def _clean_mlp_out(suffix_layer: SuffixLayer, mlp_in: Array) -> Array:
    """Frozen target MLP — exactly `W` applied, not the `V@U + (W−V@U)` identity, so
    non-live sites carry no V/U gradient and no decomposition rounding (SPEC S2/S3)."""
    return (
        jax.nn.silu(mlp_in @ suffix_layer.Wg.T) * (mlp_in @ suffix_layer.Wu.T)
    ) @ suffix_layer.Wd.T


def clean_suffix_logits(target: Target, resid: Float[Array, "b t d"]) -> Array:
    """The all-frozen suffix forward — the recon target (SPEC S3)."""
    x = resid
    for suffix_layer in target.layers:
        x = x + suffix_layer.attn(rms_norm(x, suffix_layer.ln1, target.eps), target.inv_freq)
        x = x + _clean_mlp_out(suffix_layer, rms_norm(x, suffix_layer.ln2, target.eps))
    x = rms_norm(x, target.norm, target.eps)
    return x @ target.lm_head.T


def clean_site_inputs(
    target: Target, first_layer: int, site_names: tuple[str, ...], resid: Float[Array, "b t d"]
) -> dict[str, Array]:
    """Clean CI inputs per site (SPEC S4), all on the frozen path, threaded layer to
    layer: q/k/v ← the post-LN1 residual, o ← the pre-o_proj attention output,
    gate/up ← the post-LN2 residual, down ← silu(gate)·up."""
    wanted = frozenset(site_names)
    last_site_layer = max(parse_site_name(name)[0] for name in site_names)
    inputs: dict[str, Array] = {}
    x = resid
    for layer_offset, suffix_layer in enumerate(target.layers):
        layer = first_layer + layer_offset
        if layer > last_site_layer:
            break
        attn = suffix_layer.attn
        h1 = rms_norm(x, suffix_layer.ln1, target.eps)
        attn_y = attn.core(h1 @ attn.wq.T, h1 @ attn.wk.T, h1 @ attn.wv.T, target.inv_freq)
        post_attn = x + attn_y @ attn.wo.T
        mlp_in = rms_norm(post_attn, suffix_layer.ln2, target.eps)
        gate = mlp_in @ suffix_layer.Wg.T
        up = mlp_in @ suffix_layer.Wu.T
        down_in = jax.nn.silu(gate) * up
        for kind, site_input in (
            ("q", h1), ("k", h1), ("v", h1), ("o", attn_y),
            ("gate", mlp_in), ("up", mlp_in), ("down", down_in),
        ):  # fmt: skip
            name = site_name(layer, kind)
            if name in wanted:
                inputs[name] = site_input
        x = post_attn + down_in @ suffix_layer.Wd.T
    assert set(inputs) == wanted, (sorted(inputs), sorted(wanted))
    return inputs


def _masked_site_out(
    components: DecompVU,
    site: str,
    W: Array,
    x_in: Array,
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live_set: frozenset[str],
    has_delta: bool,
    collect: dict[str, Array] | None,
) -> Array:
    """One site's output in the masked forward; if `collect` is given, the per-`live`-site
    decomposed output is recorded there (the hidden-acts recon material, SPEC S31).
    Non-live sites take the frozen `x @ W` path and are NOT collected."""
    if site not in live_set:
        return x_in @ W.T
    V, U = components.site(site)
    out = _site_out(
        x_in, V, U, W, masks[site], delta_masks[site] if has_delta else None,
        None if routes is None else routes[site],
    )  # fmt: skip
    if collect is not None:
        collect[site] = out
    return out


def _run_masked_suffix(
    target: Target,
    components: DecompVU,
    first_layer: int,
    resid: Float[Array, "b t d"],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live: tuple[str, ...],
    has_delta: bool,
    collect: dict[str, Array] | None,
) -> Array:
    """The masked decomposed suffix forward shared by `masked_suffix_logits` and
    `masked_suffix_site_outputs` (SPEC §1.3, S2): sites in `live` run their decomposed
    forward with `masks[s]` / `delta_masks[s]` / `routes[s]`; every other site — and every
    site absent from the decomposition entirely — runs the frozen `x @ W` path. `live` and
    `has_delta` are static under jit; `has_delta` False skips the `x @ Δ` matmul
    (LOSS_PARITY_DESIGN §4b). A non-None `collect` gathers per-site decomposed outputs."""
    live_set = frozenset(live)
    x = resid
    for layer_offset, suffix_layer in enumerate(target.layers):
        layer = first_layer + layer_offset
        live_kinds = {kind for kind in KIND_ORDER if site_name(layer, kind) in live_set}
        attn = suffix_layer.attn
        site_args = (masks, delta_masks, routes, live_set, has_delta, collect)
        h1 = rms_norm(x, suffix_layer.ln1, target.eps)
        if not live_kinds & set(ATTN_KINDS):
            attn_out = attn(h1, target.inv_freq)
        else:
            q = _masked_site_out(components, site_name(layer, "q"), attn.wq, h1, *site_args)
            k = _masked_site_out(components, site_name(layer, "k"), attn.wk, h1, *site_args)
            v = _masked_site_out(components, site_name(layer, "v"), attn.wv, h1, *site_args)
            attn_y = attn.core(q, k, v, target.inv_freq)
            attn_out = _masked_site_out(
                components, site_name(layer, "o"), attn.wo, attn_y, *site_args
            )
        post_attn = x + attn_out
        mlp_in = rms_norm(post_attn, suffix_layer.ln2, target.eps)
        if not live_kinds & set(MLP_KINDS):
            mlp_out = _clean_mlp_out(suffix_layer, mlp_in)
        else:
            gate = _masked_site_out(
                components, site_name(layer, "gate"), suffix_layer.Wg, mlp_in, *site_args
            )
            up = _masked_site_out(
                components, site_name(layer, "up"), suffix_layer.Wu, mlp_in, *site_args
            )
            down_in = jax.nn.silu(gate) * up
            mlp_out = _masked_site_out(
                components, site_name(layer, "down"), suffix_layer.Wd, down_in, *site_args
            )
        x = post_attn + mlp_out
    x = rms_norm(x, target.norm, target.eps)
    return x @ target.lm_head.T


def masked_suffix_logits(
    target: Target,
    components: DecompVU,
    first_layer: int,
    resid: Float[Array, "b t d"],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live: tuple[str, ...],
    has_delta: bool,
) -> Array:
    return _run_masked_suffix(
        target, components, first_layer, resid, masks, delta_masks, routes, live, has_delta, None
    )


def masked_suffix_site_outputs(
    target: Target,
    components: DecompVU,
    first_layer: int,
    resid: Float[Array, "b t d"],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live: tuple[str, ...],
    has_delta: bool,
) -> dict[str, Array]:
    """Per-`live`-site decomposed output of the masked forward (SPEC S31). Runs the exact
    `masked_suffix_logits` forward, discards the logits, returns the collected outputs."""
    collect: dict[str, Array] = {}
    _run_masked_suffix(
        target, components, first_layer, resid, masks, delta_masks, routes, live, has_delta, collect
    )
    assert set(collect) == set(live), (sorted(collect), sorted(live))
    return collect


def weight_deltas_fp32(
    target: Target, components: DecompVU, first_layer: int, sites: tuple[SiteSpec, ...]
) -> dict[str, Array]:
    """fp32 `W − V@U` per site from fp32 masters (SPEC N2; faithfulness input)."""
    out: dict[str, Array] = {}
    for spec in sites:
        layer, kind = parse_site_name(spec.name)
        W = _frozen_site_weight(target.layers[layer - first_layer], kind)
        V, U = components.site(spec.name)
        out[spec.name] = W.astype(jnp.float32) - (V.astype(jnp.float32) @ U.astype(jnp.float32)).T
    return out


def llama_decomposed_lm(cfg: LlamaConfig, sites: tuple[SiteSpec, ...]) -> DecomposedModel:
    """The `DecomposedModel` boundary for this target (SPEC §1; `lm.py` contract).
    `sites` must be canonical-ordered with dims matching `cfg` (`llama_site_specs`)."""
    site_cs = tuple(SiteC(s.name, s.C) for s in sites)
    assert sites == llama_site_specs(cfg, canonical_site_cs(site_cs)), (
        f"sites are not the canonical specs for this config: {sites}"
    )
    site_names = tuple(s.name for s in sites)
    first_layer = first_decomposed_layer(site_names)
    return DecomposedModel(
        sites=sites,
        leading_axes=("sequence",),
        clean_output=lambda frozen, resid: clean_suffix_logits(frozen, resid),
        site_inputs=lambda frozen, resid: clean_site_inputs(frozen, first_layer, site_names, resid),
        masked_output=lambda frozen,
        components,
        resid,
        masks,
        delta_masks,
        routes,
        live,
        has_delta: (
            masked_suffix_logits(
                frozen, components, first_layer, resid, masks, delta_masks, routes, live, has_delta
            )
        ),
        masked_site_outputs=lambda frozen,
        components,
        resid,
        masks,
        delta_masks,
        routes,
        live,
        has_delta: (
            masked_suffix_site_outputs(
                frozen, components, first_layer, resid, masks, delta_masks, routes, live, has_delta
            )
        ),
        weight_deltas=lambda frozen, components: weight_deltas_fp32(
            frozen, components, first_layer, sites
        ),
    )


# ----------------------------- HF weight loading -----------------------------


def _hf_snapshot_dir(model_name: str) -> Path:
    import os

    cache = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
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
        self._open: dict[str, object] = {}

    def get(self, key: str) -> Array:
        fname = self._key_to_file[key]
        if fname not in self._open:
            self._open[fname] = safe_open(str(self._snapshot / fname), framework="numpy")
        return jnp.asarray(np.array(self._open[fname].get_tensor(key)), dtype=DT)  # type: ignore[union-attr]


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


def _load_block(w: _HFWeights, i: int, cfg: LlamaConfig) -> FrozenBlock:
    pre = "model.layers"
    return FrozenBlock(
        ln1=w.get(f"{pre}.{i}.input_layernorm.weight"),
        ln2=w.get(f"{pre}.{i}.post_attention_layernorm.weight"),
        attn=_load_attn(w, i, cfg),
        mlp=FrozenMLP(
            wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
            wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
            wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
        ),
        eps=cfg.rms_norm_eps,
    )


def load_target_from_hf(model_name: str, cfg: LlamaConfig, first_layer: int) -> Target:
    """Load the frozen residual-start suffix. Only `first_layer..n_layer-1` is
    materialized — the prefix is consumed via `make_real_target_residual`."""
    w = _HFWeights(_hf_snapshot_dir(model_name))
    pre = "model.layers"
    layers = [
        SuffixLayer(
            ln1=w.get(f"{pre}.{i}.input_layernorm.weight"),
            ln2=w.get(f"{pre}.{i}.post_attention_layernorm.weight"),
            attn=_load_attn(w, i, cfg),
            Wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
            Wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
            Wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
        )
        for i in range(first_layer, cfg.n_layer)
    ]
    return Target(
        layers=layers,
        norm=w.get("model.norm.weight"),
        lm_head=w.get("lm_head.weight"),
        inv_freq=llama3_inv_freq(cfg),
        eps=cfg.rms_norm_eps,
    )


class Prefix(eqx.Module):
    """The frozen L0..first-1 prefix: embedding + blocks. Used only to harvest the
    residual entering the suffix (SPEC §1.1) — never in any gradient graph."""

    embed: Float[Array, "vocab d"]
    blocks: list[FrozenBlock]
    inv_freq: Float[Array, " hd2"]


def load_prefix_from_hf(model_name: str, cfg: LlamaConfig, first_layer: int) -> Prefix:
    w = _HFWeights(_hf_snapshot_dir(model_name))
    return Prefix(
        embed=w.get("model.embed_tokens.weight"),
        blocks=[_load_block(w, i, cfg) for i in range(first_layer)],
        inv_freq=llama3_inv_freq(cfg),
    )


def prefix_residual(prefix: Prefix, idx: Array) -> Array:
    """Pure prefix forward: token ids `(b, t)` -> residual entering `first` (b, t, d).
    The trainer jits this with `prefix` as a runtime arg and the batch dp-sharded."""
    x = prefix.embed[idx]
    for blk in prefix.blocks:
        x = blk(x, prefix.inv_freq)
    return x


def make_real_target_residual(
    model_name: str, cfg: LlamaConfig, first_layer: int, idx: Array, chunk: int
) -> Array:
    """One-shot eager harvest for the bench: loads the prefix, runs it in micro-batch
    chunks (`chunk`) so peak activation is one chunk's forward, discards the weights.
    The trainer instead keeps a `Prefix` resident and jits `prefix_residual`."""
    prefix = load_prefix_from_hf(model_name, cfg, first_layer)
    b = idx.shape[0]
    outs = [
        jax.block_until_ready(prefix_residual(prefix, idx[i : i + chunk]))
        for i in range(0, b, chunk)
    ]
    return jnp.concatenate(outs, axis=0)
