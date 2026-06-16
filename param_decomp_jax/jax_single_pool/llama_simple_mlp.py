"""`LlamaSimpleMLP` pile-pretrained target — the second `DecomposedModel` implementation.

Torch reference (read-only, ground truth):
`param_decomp_lab/experiments/lm/pretrain/models/llama_simple_mlp.py`, weights from
pretrain run `goodfire/spd/runs/t-9d2b8f02`. Llama-style pre-RMSNorm blocks under a
GPT2-style module tree `h.{i}.`: rotary GQA attention (`rotary_dim == head_dim`,
plain base-`rotary_base` rotate-half RoPE — NOT llama3-rescaled) and a GELU(tanh) MLP
`c_fc -> gelu -> down_proj`. `wte` and `lm_head` are tied; no biases anywhere.

The torch RoPE construction (`freq = base**(i/(rd/2))` tiled `.repeat(2)`,
`rotate_every_two` with `rotary_adjacent_pairs=False`) is exactly the rotate-half RoPE
of `vendored_jax.llama.rope_cos_sin`/`apply_rope` with `inv_freq = base**(-2i/hd)` —
pinned by the torch-fixture equivalence test (`tests/simple_mlp_equivalence/`).

Decomposed sites are torch-module-path named: `h.{i}.attn.{q,k,v,o}_proj`,
`h.{i}.mlp.c_fc`, `h.{i}.mlp.down_proj`, each with its own C. q/k/v sites are
decomposed BEFORE RoPE/SDPA; o after; c_fc before the GELU; down_proj after. The
frozen residual-start suffix runs from the lowest decomposed layer; the prefix
(embedding + earlier blocks) only harvests the residual. With sites at layer 0 the
"residual" is the embedding output.

Weights load from the torch pretrain cache
(`$PARAM_DECOMP_OUT_DIR/pretrain_cache/<project>-<run_id>/`), converted once to
safetensors by `tools/convert_llama_simple_mlp_checkpoint.py` (torch venv).
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int
from safetensors import safe_open
from vendored_jax.llama import rms_norm

from jax_single_pool.llama8b import DecompVU, FrozenAttn
from jax_single_pool.lm import DecomposedModel, SiteC, SiteSpec

KIND_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")
"""Within-layer canonical site order = computation order. The canonical site order
(`site_specs`) is layer-ascending, then this."""
ATTN_KINDS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_KINDS = ("c_fc", "down_proj")

SITE_NAME_PATTERN = re.compile(
    r"^h\.(\d+)\.(?:attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(c_fc|down_proj))$"
)
WILDCARD_NAME_PATTERN = re.compile(
    r"^h\.\*\.(?:attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(c_fc|down_proj))$"
)


@dataclass(frozen=True)
class LlamaSimpleMLPConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    n_intermediate: int
    rotary_base: float
    rms_norm_eps: float
    n_ctx: int

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_rep(self) -> int:
        return self.n_head // self.n_kv_head


def config_from_model_config_dict(raw: dict[str, object]) -> LlamaSimpleMLPConfig:
    """Parse the pretrain run's `model_config.yaml` dict, refusing every torch-config
    variant this port does not implement (biases, adjacent-pair rotary, merged-qkv
    non-GQA attention). `flash_attention` is ignored: both torch paths compute the
    same causal SDPA math."""
    assert raw["model_type"] == "LlamaSimpleMLP", raw["model_type"]
    assert raw["use_grouped_query_attention"] is True, "merged-qkv (c_attn) unsupported"
    assert raw["attn_bias"] is False and raw["mlp_bias"] is False, "biases unsupported"
    assert raw["rotary_adjacent_pairs"] is False, "adjacent-pair rotary unsupported"
    cfg = LlamaSimpleMLPConfig(
        vocab_size=int(raw["vocab_size"]),  # pyright: ignore[reportArgumentType]
        n_layer=int(raw["n_layer"]),  # pyright: ignore[reportArgumentType]
        n_head=int(raw["n_head"]),  # pyright: ignore[reportArgumentType]
        n_kv_head=int(raw["n_key_value_heads"]),  # pyright: ignore[reportArgumentType]
        n_embd=int(raw["n_embd"]),  # pyright: ignore[reportArgumentType]
        n_intermediate=int(raw["n_intermediate"]),  # pyright: ignore[reportArgumentType]
        rotary_base=float(raw["rotary_base"]),  # pyright: ignore[reportArgumentType]
        rms_norm_eps=float(raw["rms_norm_eps"]),  # pyright: ignore[reportArgumentType]
        n_ctx=int(raw["n_ctx"]),  # pyright: ignore[reportArgumentType]
    )
    assert cfg.n_embd % cfg.n_head == 0 and cfg.n_head % cfg.n_kv_head == 0, cfg
    # the torch class forces rotary_dim = head_dim regardless of config; insist the
    # config agrees so nothing is silently overridden
    assert raw["rotary_dim"] == cfg.head_dim, (raw["rotary_dim"], cfg.head_dim)
    assert raw["block_size"] == cfg.n_ctx, (raw["block_size"], cfg.n_ctx)
    return cfg


def plain_rope_inv_freq(cfg: LlamaSimpleMLPConfig) -> Float[Array, " hd2"]:
    """`base**(-2i/head_dim)` fp32 — the torch `calculate_sin_cos_rotary` frequencies
    (`1 / base**(i/(rd/2))`), no llama3 rescaling."""
    exponents = jnp.arange(0, cfg.head_dim, 2, dtype=jnp.float32) / cfg.head_dim
    return 1.0 / (cfg.rotary_base**exponents)


def site_name(layer: int, kind: str) -> str:
    assert kind in KIND_ORDER, kind
    submodule = "attn" if kind in ATTN_KINDS else "mlp"
    return f"h.{layer}.{submodule}.{kind}"


def parse_site_name(name: str) -> tuple[int, str]:
    """`h.{i}.{attn,mlp}.{kind}` -> (layer, kind); rejects anything else (including
    kind/submodule mismatches like `attn.c_fc`)."""
    match = SITE_NAME_PATTERN.match(name)
    assert match is not None, f"unsupported site name {name!r}"
    layer, attn_kind, mlp_kind = match.groups()
    return int(layer), attn_kind if attn_kind is not None else mlp_kind


def site_dims(cfg: LlamaSimpleMLPConfig, kind: str) -> tuple[int, int]:
    """(d_in, d_out) of one per-layer matrix, right-mult orientation."""
    d, di = cfg.n_embd, cfg.n_intermediate
    qd = cfg.n_head * cfg.head_dim
    kvd = cfg.n_kv_head * cfg.head_dim
    match kind:
        case "q_proj":
            return d, qd
        case "k_proj" | "v_proj":
            return d, kvd
        case "o_proj":
            return qd, d
        case "c_fc":
            return d, di
        case "down_proj":
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


def expand_wildcard_site_cs(entries: tuple[SiteC, ...], n_layer: int) -> tuple[SiteC, ...]:
    """`h.*.<submodule>.<kind>` entries expand to every layer at that entry's C
    (the torch `module_pattern` wildcard convention); explicit `h.{i}.` entries pass
    through. Result is canonical-ordered; duplicates raise."""
    expanded: list[SiteC] = []
    for entry in entries:
        wildcard = WILDCARD_NAME_PATTERN.match(entry.name)
        if wildcard is None:
            parse_site_name(entry.name)
            expanded.append(entry)
        else:
            attn_kind, mlp_kind = wildcard.groups()
            kind = attn_kind if attn_kind is not None else mlp_kind
            expanded.extend(SiteC(site_name(layer, kind), entry.C) for layer in range(n_layer))
    return canonical_site_cs(tuple(expanded))


def site_specs(cfg: LlamaSimpleMLPConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
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


# ----------------------------- frozen suffix / prefix -----------------------------


class SimpleMLPSuffixLayer(eqx.Module):
    """One layer's frozen weights — norms, rotary GQA attention (`FrozenAttn` is
    target-agnostic: weights + RoPE/SDPA core), GELU MLP. Decomposed sites read their
    frozen target W from here at forward time; weights pass as a runtime arg."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    Wfc: Float[Array, "di d"]
    Wdown: Float[Array, "d di"]


class SimpleMLPTarget(eqx.Module):
    """Frozen residual-start suffix: layers `first_decomposed_layer..n_layer-1`,
    final norm, tied lm_head."""

    layers: list[SimpleMLPSuffixLayer]
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]
    eps: float = eqx.field(static=True)
    n_ctx: int = eqx.field(static=True)


class SimpleMLPPrefix(eqx.Module):
    """The frozen 0..first-1 prefix: embedding + blocks. Used only to harvest the
    residual entering the suffix — never in any gradient graph. With sites at layer 0
    `blocks` is empty and the residual is the raw embedding output."""

    embed: Float[Array, "vocab d"]
    blocks: list[SimpleMLPSuffixLayer]
    inv_freq: Float[Array, " hd2"]
    eps: float = eqx.field(static=True)
    n_ctx: int = eqx.field(static=True)


def _frozen_site_weight(layer: SimpleMLPSuffixLayer, kind: str) -> Array:
    match kind:
        case "q_proj":
            return layer.attn.wq
        case "k_proj":
            return layer.attn.wk
        case "v_proj":
            return layer.attn.wv
        case "o_proj":
            return layer.attn.wo
        case "c_fc":
            return layer.Wfc
        case "down_proj":
            return layer.Wdown
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


# ----------------------------- forwards -----------------------------


def _gelu_tanh(x: Array) -> Array:
    """The torch reference's `NewGELU` (tanh approximation), exactly
    `jax.nn.gelu(approximate=True)` — pinned by the torch-fixture equivalence test."""
    return jax.nn.gelu(x, approximate=True)


def _site_out(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
) -> Array:
    """One decomposed linear (SPEC §4.1), same as `llama8b._site_out`: `((x@V)*m)@U +
    (x@Δ)*d`, routed per position against the frozen `x @ W.T`. `mask` may be None
    (fully on); `route` None routes everywhere. `delta_mask` None drops the delta path
    entirely (constant-source entries carry no delta, LOSS_PARITY_DESIGN §4b). The delta
    on this PATH is bf16-computed from the cast components (documented divergence; the
    faithfulness loss uses the fp32 `weight_deltas`, SPEC N2)."""
    acts = x @ V
    if mask is not None:
        acts = acts * mask
    out = acts @ U
    if delta_mask is not None:
        delta = W - (V @ U).T  # (d_out, d_in)
        out = out + delta_mask[..., None] * (x @ delta.T)
    if route is not None:
        out = jnp.where(route[..., None], out, x @ W.T)
    return out


def _clean_mlp_out(layer: SimpleMLPSuffixLayer, mlp_in: Array) -> Array:
    """Frozen target MLP — exactly `W` applied, not the `V@U + (W−V@U)` identity, so
    non-live sites carry no V/U gradient and no decomposition rounding (SPEC S2/S3)."""
    return _gelu_tanh(mlp_in @ layer.Wfc.T) @ layer.Wdown.T


def _clean_block(layer: SimpleMLPSuffixLayer, x: Array, inv_freq: Array, eps: float) -> Array:
    x = x + layer.attn(rms_norm(x, layer.ln1, eps), inv_freq)
    return x + _clean_mlp_out(layer, rms_norm(x, layer.ln2, eps))


def clean_suffix_logits(target: SimpleMLPTarget, resid: Float[Array, "b t d"]) -> Array:
    """The all-frozen suffix forward — the recon target (SPEC S3)."""
    assert resid.shape[1] <= target.n_ctx, (resid.shape, target.n_ctx)
    x = resid
    for layer in target.layers:
        x = _clean_block(layer, x, target.inv_freq, target.eps)
    x = rms_norm(x, target.norm, target.eps)
    return x @ target.lm_head.T


def clean_site_inputs(
    target: SimpleMLPTarget,
    first_layer: int,
    site_names: tuple[str, ...],
    resid: Float[Array, "b t d"],
) -> dict[str, Array]:
    """Clean CI inputs per site (SPEC S4), all on the frozen path, threaded layer to
    layer: q/k/v ← the post-LN1 residual, o ← the pre-o_proj attention output,
    c_fc ← the post-LN2 residual, down_proj ← gelu(c_fc out)."""
    assert resid.shape[1] <= target.n_ctx, (resid.shape, target.n_ctx)
    wanted = frozenset(site_names)
    last_site_layer = max(parse_site_name(name)[0] for name in site_names)
    inputs: dict[str, Array] = {}
    x = resid
    for layer_offset, layer in enumerate(target.layers):
        layer_idx = first_layer + layer_offset
        if layer_idx > last_site_layer:
            break
        attn = layer.attn
        h1 = rms_norm(x, layer.ln1, target.eps)
        attn_y = attn.core(h1 @ attn.wq.T, h1 @ attn.wk.T, h1 @ attn.wv.T, target.inv_freq)
        post_attn = x + attn_y @ attn.wo.T
        mlp_in = rms_norm(post_attn, layer.ln2, target.eps)
        down_in = _gelu_tanh(mlp_in @ layer.Wfc.T)
        for kind, site_input in (
            ("q_proj", h1), ("k_proj", h1), ("v_proj", h1), ("o_proj", attn_y),
            ("c_fc", mlp_in), ("down_proj", down_in),
        ):  # fmt: skip
            name = site_name(layer_idx, kind)
            if name in wanted:
                inputs[name] = site_input
        x = post_attn + down_in @ layer.Wdown.T
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
) -> Array:
    if site not in live_set:
        return x_in @ W.T
    V, U = components.site(site)
    return _site_out(
        x_in, V, U, W, masks[site], delta_masks[site] if has_delta else None,
        None if routes is None else routes[site],
    )  # fmt: skip


def masked_suffix_logits(
    target: SimpleMLPTarget,
    components: DecompVU,
    first_layer: int,
    resid: Float[Array, "b t d"],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live: tuple[str, ...],
    has_delta: bool,
) -> Array:
    """Masked decomposed suffix forward (SPEC §4.1, S2): sites in `live` run their
    decomposed forward with `masks[s]` / `delta_masks[s]` / `routes[s]`; every other
    site — and every site absent from the decomposition entirely — runs the frozen
    `x @ W` path. `live` and `has_delta` are static under jit; `has_delta` False skips
    the `x @ Δ` matmul (LOSS_PARITY_DESIGN §4b)."""
    assert resid.shape[1] <= target.n_ctx, (resid.shape, target.n_ctx)
    live_set = frozenset(live)
    x = resid
    for layer_offset, layer in enumerate(target.layers):
        layer_idx = first_layer + layer_offset
        live_kinds = {kind for kind in KIND_ORDER if site_name(layer_idx, kind) in live_set}
        attn = layer.attn
        site_args = (masks, delta_masks, routes, live_set, has_delta)
        h1 = rms_norm(x, layer.ln1, target.eps)
        if not live_kinds & set(ATTN_KINDS):
            attn_out = attn(h1, target.inv_freq)
        else:
            q = _masked_site_out(
                components, site_name(layer_idx, "q_proj"), attn.wq, h1, *site_args
            )
            k = _masked_site_out(
                components, site_name(layer_idx, "k_proj"), attn.wk, h1, *site_args
            )
            v = _masked_site_out(
                components, site_name(layer_idx, "v_proj"), attn.wv, h1, *site_args
            )
            attn_y = attn.core(q, k, v, target.inv_freq)
            attn_out = _masked_site_out(
                components, site_name(layer_idx, "o_proj"), attn.wo, attn_y, *site_args
            )
        post_attn = x + attn_out
        mlp_in = rms_norm(post_attn, layer.ln2, target.eps)
        if not live_kinds & set(MLP_KINDS):
            mlp_out = _clean_mlp_out(layer, mlp_in)
        else:
            fc = _masked_site_out(
                components, site_name(layer_idx, "c_fc"), layer.Wfc, mlp_in, *site_args
            )
            mlp_out = _masked_site_out(
                components, site_name(layer_idx, "down_proj"), layer.Wdown, _gelu_tanh(fc),
                *site_args,
            )  # fmt: skip
        x = post_attn + mlp_out
    x = rms_norm(x, target.norm, target.eps)
    return x @ target.lm_head.T


def weight_deltas_fp32(
    target: SimpleMLPTarget,
    components: DecompVU,
    first_layer: int,
    sites: tuple[SiteSpec, ...],
) -> dict[str, Array]:
    """fp32 `W − V@U` per site from fp32 masters (SPEC N2; faithfulness input)."""
    out: dict[str, Array] = {}
    for spec in sites:
        layer_idx, kind = parse_site_name(spec.name)
        W = _frozen_site_weight(target.layers[layer_idx - first_layer], kind)
        V, U = components.site(spec.name)
        out[spec.name] = W.astype(jnp.float32) - (V.astype(jnp.float32) @ U.astype(jnp.float32)).T
    return out


def llama_simple_mlp_decomposed_lm(
    cfg: LlamaSimpleMLPConfig, sites: tuple[SiteSpec, ...]
) -> DecomposedModel:
    """The `DecomposedModel` boundary for this target (`lm.py` contract). `sites` must be
    canonical-ordered with dims matching `cfg` (`site_specs`)."""
    site_cs = tuple(SiteC(s.name, s.C) for s in sites)
    assert sites == site_specs(cfg, canonical_site_cs(site_cs)), (
        f"sites are not the canonical specs for this config: {sites}"
    )
    site_names = tuple(s.name for s in sites)
    first_layer = first_decomposed_layer(site_names)
    return DecomposedModel(
        sites=sites,
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
        weight_deltas=lambda frozen, components: weight_deltas_fp32(
            frozen, components, first_layer, sites
        ),
    )


def prefix_residual(prefix: SimpleMLPPrefix, idx: Int[Array, "b t"]) -> Array:
    """Pure prefix forward: token ids `(b, t)` -> residual entering the suffix
    `(b, t, d)`. The trainer jits this with `prefix` as a runtime arg."""
    assert idx.shape[1] <= prefix.n_ctx, (idx.shape, prefix.n_ctx)
    x = prefix.embed[idx]
    for layer in prefix.blocks:
        x = _clean_block(layer, x, prefix.inv_freq, prefix.eps)
    return x


# ----------------------------- weight loading -----------------------------


def pretrain_cache_dir(run_path: str) -> Path:
    """Resolve a torch `PretrainRunInfo` wandb run path (`entity/project[/runs]/run_id`)
    to its download cache dir. The cache must already exist (populated by the torch
    repo's `PretrainRunInfo.from_path`); this trainer never talks to wandb."""
    match run_path.strip("/").split("/"):
        case [_entity, project, "runs", run_id] | [_entity, project, run_id]:
            pass
        case parts:
            raise AssertionError(f"unsupported pretrain run path {run_path!r} ({parts})")
    out_root = os.environ.get("PARAM_DECOMP_OUT_DIR")
    if out_root is None:
        data_mount = os.environ.get("DATA_MOUNT")
        assert data_mount is not None, (
            "set PARAM_DECOMP_OUT_DIR (or DATA_MOUNT) to locate the pretrain cache"
        )
        out_root = f"{data_mount}/artifacts/mechanisms/param-decomp"
    cache_dir = Path(out_root) / "pretrain_cache" / f"{project}-{run_id}"
    assert cache_dir.exists(), (
        f"pretrain cache missing: {cache_dir} — download it once via the torch repo "
        f"(`PretrainRunInfo.from_path({run_path!r})`)"
    )
    return cache_dir


def load_model_config(cache_dir: Path) -> LlamaSimpleMLPConfig:
    return config_from_model_config_dict(
        yaml.safe_load((cache_dir / "model_config.yaml").read_text())
    )


def checkpoint_safetensors_path(cache_dir: Path) -> Path:
    candidates = sorted(cache_dir.glob("model_step_*.safetensors"))
    assert len(candidates) == 1, (
        f"expected exactly one model_step_*.safetensors under {cache_dir}, found "
        f"{candidates or 'none'} — convert the torch checkpoint once (torch venv): "
        f"python jax_single_pool/tools/convert_llama_simple_mlp_checkpoint.py {cache_dir}"
    )
    return candidates[0]


WeightGetter = Callable[[str], Array]
"""Checkpoint key -> array, e.g. `h.0.attn.q_proj.weight`. `lm_head.weight` is NOT a
key — the head is tied to `wte.weight`."""


def _checkpoint_weight_getter(cache_dir: Path, dtype: DTypeLike) -> WeightGetter:
    handle = safe_open(str(checkpoint_safetensors_path(cache_dir)), framework="numpy")
    return lambda key: jnp.asarray(np.array(handle.get_tensor(key)), dtype=dtype)  # type: ignore[attr-defined]


def _layer_from_weights(
    get: WeightGetter, layer_idx: int, cfg: LlamaSimpleMLPConfig
) -> SimpleMLPSuffixLayer:
    return SimpleMLPSuffixLayer(
        ln1=get(f"h.{layer_idx}.rms_1.weight"),
        ln2=get(f"h.{layer_idx}.rms_2.weight"),
        attn=FrozenAttn(
            wq=get(f"h.{layer_idx}.attn.q_proj.weight"),
            wk=get(f"h.{layer_idx}.attn.k_proj.weight"),
            wv=get(f"h.{layer_idx}.attn.v_proj.weight"),
            wo=get(f"h.{layer_idx}.attn.o_proj.weight"),
            n_head=cfg.n_head,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            n_rep=cfg.n_rep,
        ),
        Wfc=get(f"h.{layer_idx}.mlp.c_fc.weight"),
        Wdown=get(f"h.{layer_idx}.mlp.down_proj.weight"),
    )


def target_from_weights(
    get: WeightGetter, cfg: LlamaSimpleMLPConfig, first_layer: int
) -> SimpleMLPTarget:
    """Build the frozen residual-start suffix (`first_layer..n_layer-1`, final norm,
    tied lm_head) from checkpoint-keyed weights."""
    return SimpleMLPTarget(
        layers=[_layer_from_weights(get, i, cfg) for i in range(first_layer, cfg.n_layer)],
        norm=get("ln_f.weight"),
        lm_head=get("wte.weight"),
        inv_freq=plain_rope_inv_freq(cfg),
        eps=cfg.rms_norm_eps,
        n_ctx=cfg.n_ctx,
    )


def prefix_from_weights(
    get: WeightGetter, cfg: LlamaSimpleMLPConfig, first_layer: int
) -> SimpleMLPPrefix:
    return SimpleMLPPrefix(
        embed=get("wte.weight"),
        blocks=[_layer_from_weights(get, i, cfg) for i in range(first_layer)],
        inv_freq=plain_rope_inv_freq(cfg),
        eps=cfg.rms_norm_eps,
        n_ctx=cfg.n_ctx,
    )


def load_target_from_pretrain_cache(
    cache_dir: Path, cfg: LlamaSimpleMLPConfig, first_layer: int, dtype: DTypeLike
) -> SimpleMLPTarget:
    return target_from_weights(_checkpoint_weight_getter(cache_dir, dtype), cfg, first_layer)


def load_prefix_from_pretrain_cache(
    cache_dir: Path, cfg: LlamaSimpleMLPConfig, first_layer: int, dtype: DTypeLike
) -> SimpleMLPPrefix:
    return prefix_from_weights(_checkpoint_weight_getter(cache_dir, dtype), cfg, first_layer)


def replicate_frozen[FrozenTree: (SimpleMLPTarget, SimpleMLPPrefix)](
    tree: FrozenTree, mesh: Mesh
) -> FrozenTree:
    """This target is small (~67M params); replicating the frozen weights on every
    device is the whole sharding story (the `llama8b_sharding.replicate_target`
    analog). V/U / CI / source placement reuses the generic per-site plan."""
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, tree)
