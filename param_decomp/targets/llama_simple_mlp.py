"""`LlamaSimpleMLP` pile-pretrained target, hosted on the shared GLU-transformer engine.

Torch reference (read-only, ground truth):
`param_decomp/experiments/lm/pretrain/models/llama_simple_mlp.py`, weights from
pretrain run `goodfire/spd/runs/t-9d2b8f02`. Llama-style pre-RMSNorm blocks under a
GPT2-style module tree `h.{i}.`: rotary GQA attention (`rotary_dim == head_dim`,
plain base-`rotary_base` rotate-half RoPE — NOT llama3-rescaled) and a GELU(tanh) MLP
`c_fc -> gelu -> down_proj`. `wte` and `lm_head` are tied; no biases anywhere.

The torch RoPE construction (`freq = base**(i/(rd/2))` tiled `.repeat(2)`,
`rotate_every_two` with `rotary_adjacent_pairs=False`) is exactly the rotate-half RoPE
of `param_decomp.vendored_jax.llama.rope_cos_sin`/`apply_rope` with `inv_freq = base**(-2i/hd)` —
pinned by the torch-fixture equivalence test (`param_decomp/tests/targets/simple_mlp_equivalence/`).

This module is the family DECLARATION: the site vocabulary (`SIMPLE_MLP_ANATOMY` binds
`q_proj`/…/`c_fc`/`down_proj` to the engine's structural roles, `PlainMLP` + `TiedHead`
select its anatomical arms), the torch config parser, and the checkpoint loaders. All
forwards run on `glu_transformer.GLUDecomposedModel`.

Decomposed sites are torch-module-path named: `h.{i}.attn.{q,k,v,o}_proj`,
`h.{i}.mlp.c_fc`, `h.{i}.mlp.down_proj`, each with its own C.

Weights load from the torch pretrain cache
(`<data_root>/pretrain_cache/<project>-<run_id>/`), converted once to
safetensors by `tools/convert_llama_simple_mlp_checkpoint.py` (torch venv).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

import jax.numpy as jnp
import numpy as np
import yaml
from jax.typing import DTypeLike
from jaxtyping import Array, Float
from safetensors import safe_open

from param_decomp.core import family
from param_decomp.core.components import SiteC, SiteDims, SiteSpec
from param_decomp.core.family import ArchFamily
from param_decomp.core.nonlinearity import NonlinearityPartition
from param_decomp.targets.glu_transformer import (
    Anatomy,
    FrozenAttn,
    GLUDecomposedModel,
    GLULayer,
    PlainMLP,
    PlainMLPKinds,
    TiedHead,
    anatomy_nonlinearity_partition,
    anatomy_site_dims,
    build_engine_model,
)

# Plain-GELU MLP (LlamaSimpleMLP). The family's matrix vocabulary — the authored c-spec
# keys (lab-side) are typed by it, so a c-spec key and a target matrix cannot drift.
SimpleMlpMatrix = Literal["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj"]

KIND_ORDER: tuple[str, ...] = get_args(SimpleMlpMatrix)
"""Within-layer canonical site order = computation order, DERIVED from the `SimpleMlpMatrix`
vocabulary. The canonical site order (`site_specs`) is layer-ascending, then this."""
ATTN_KINDS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_KINDS = ("c_fc", "down_proj")
assert KIND_ORDER == ATTN_KINDS + MLP_KINDS, KIND_ORDER


SITE_NAME_PATTERN = re.compile(
    r"^h\.(\d+)\.(?:attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(c_fc|down_proj))$"
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
    assert match is not None, (
        f"not a simple_mlp site: {name!r} (sites are h.{{i}}.attn.{{q|k|v|o}}_proj"
        f" / h.{{i}}.mlp.{{c_fc|down_proj}})"
    )
    layer, attn_kind, mlp_kind = match.groups()
    return int(layer), attn_kind if attn_kind is not None else mlp_kind


FAMILY = ArchFamily("simple_mlp", KIND_ORDER, site_name, parse_site_name)
"""This target's matrix grammar as data — the vocabulary + name renderer the tiled
`simple_mlp` c-specs resolve against."""

SIMPLE_MLP_ANATOMY = Anatomy(
    family=FAMILY,
    q="q_proj",
    k="k_proj",
    v="v_proj",
    o="o_proj",
    mlp=PlainMLPKinds(fc="c_fc", down="down_proj"),
)
"""This target's vocabulary bound to the shared engine's structural roles."""


def site_dims(cfg: LlamaSimpleMLPConfig, kind: str) -> SiteDims:
    """Dimensions of one per-layer matrix in right-mult orientation."""
    return anatomy_site_dims(SIMPLE_MLP_ANATOMY, cfg, kind)


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    return family.canonical_site_cs(FAMILY, site_cs)


def nonlinearity_partition(cfg: LlamaSimpleMLPConfig, kind: str) -> NonlinearityPartition | None:
    return anatomy_nonlinearity_partition(SIMPLE_MLP_ANATOMY, cfg, kind)


def site_specs(cfg: LlamaSimpleMLPConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    return family.site_specs(
        FAMILY,
        site_cs,
        lambda kind: site_dims(cfg, kind),
        lambda kind: nonlinearity_partition(cfg, kind),
        cfg.n_layer,
    )


# ----------------------------- weight loading -----------------------------


def load_model_config(cache_dir: Path) -> LlamaSimpleMLPConfig:
    return config_from_model_config_dict(
        yaml.safe_load((cache_dir / "model_config.yaml").read_text())
    )


def checkpoint_safetensors_path(cache_dir: Path) -> Path:
    candidates = sorted(cache_dir.glob("model_step_*.safetensors"))
    assert len(candidates) == 1, (
        f"expected exactly one model_step_*.safetensors under {cache_dir}, found "
        f"{candidates or 'none'} — convert the torch checkpoint once (torch venv; "
        f"converter at git tag `torch-oracle`): {cache_dir}"
    )
    return candidates[0]


WeightGetter = Callable[[str], Array]
"""Checkpoint key -> array, e.g. `h.0.attn.q_proj.weight`. `lm_head.weight` is NOT a
key — the head is tied to `wte.weight`."""


def _checkpoint_weight_getter(cache_dir: Path, dtype: DTypeLike) -> WeightGetter:
    handle = safe_open(str(checkpoint_safetensors_path(cache_dir)), framework="numpy")

    def get(key: str) -> Array:
        host_array = np.asarray(handle.get_tensor(key), dtype=dtype)
        return cast(Array, cast(object, host_array))

    return get


def _layer_from_weights(get: WeightGetter, layer_idx: int, cfg: LlamaSimpleMLPConfig) -> GLULayer:
    return GLULayer(
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
            implementation="auto",
        ),
        mlp=PlainMLP(
            Wfc=get(f"h.{layer_idx}.mlp.c_fc.weight"),
            Wdown=get(f"h.{layer_idx}.mlp.down_proj.weight"),
        ),
    )


def build_decomposed_simple_mlp(
    embed: Array,
    layers: list[GLULayer],
    norm: Array,
    cfg: LlamaSimpleMLPConfig,
    sites: tuple[SiteSpec, ...],
) -> GLUDecomposedModel:
    """`build_engine_model` at the SimpleMLP anatomy — plain-GELU MLP arm, TIED output
    head. `sites` must be canonical-ordered with dims matching `cfg`."""
    return build_engine_model(
        embed=embed,
        layers=layers,
        norm=norm,
        lm_head=TiedHead(),
        inv_freq=plain_rope_inv_freq(cfg),
        cfg=cfg,
        sites=sites,
        anatomy=SIMPLE_MLP_ANATOMY,
    )


def all_site_specs(cfg: LlamaSimpleMLPConfig) -> tuple[SiteSpec, ...]:
    """The full decomposable site set (every kind at every layer, C=1) — for a
    forward-only model (e.g. the torch-parity equivalence test) that decomposes nothing."""
    site_cs = tuple(
        SiteC(site_name(layer, kind), 1) for layer in range(cfg.n_layer) for kind in KIND_ORDER
    )
    return site_specs(cfg, canonical_site_cs(site_cs))


def target_from_weights(get: WeightGetter, cfg: LlamaSimpleMLPConfig) -> GLUDecomposedModel:
    """Build a forward-only decomposed model from checkpoint-keyed weights, with the full
    decomposable site set. Used by the torch-parity equivalence test (it calls the no-capture
    clean forward, which is independent of the site config)."""
    return build_decomposed_simple_mlp(
        embed=get("wte.weight"),
        layers=[_layer_from_weights(get, i, cfg) for i in range(cfg.n_layer)],
        norm=get("ln_f.weight"),
        cfg=cfg,
        sites=all_site_specs(cfg),
    )


def load_target_from_pretrain_cache(
    cache_dir: Path, cfg: LlamaSimpleMLPConfig, dtype: DTypeLike
) -> GLUDecomposedModel:
    """Forward-only decomposed model from the pretrain cache (full site set). For the
    torch-parity equivalence test; the real PD path uses
    `load_decomposed_lm_from_pretrain_cache`."""
    return target_from_weights(_checkpoint_weight_getter(cache_dir, dtype), cfg)


def load_decomposed_lm_from_pretrain_cache(
    cache_dir: Path, cfg: LlamaSimpleMLPConfig, sites: tuple[SiteSpec, ...], dtype: DTypeLike
) -> GLUDecomposedModel:
    """Load the `SimpleMLP` `DecomposedModel`: the full frozen model (tied embedding, all
    blocks, final norm) as fields plus the static decomposition `sites`."""
    get = _checkpoint_weight_getter(cache_dir, dtype)
    return build_decomposed_simple_mlp(
        embed=get("wte.weight"),
        layers=[_layer_from_weights(get, i, cfg) for i in range(cfg.n_layer)],
        norm=get("ln_f.weight"),
        cfg=cfg,
        sites=sites,
    )
