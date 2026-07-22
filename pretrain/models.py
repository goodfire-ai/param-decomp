"""Trainable JAX/equinox definitions for the three in-house target archs.

These pretrain the FROZEN targets the decomposition trainer then decomposes. The torch
reference is `torch-oracle:param_decomp_lab/experiments/lm/pretrain/models/`; this is a
capability reimplementation (next-token CE, AdamW, cosine LR) — NOT a bit-exact port.

The `LlamaSimpleMLP` checkpoint format is load-bearing: `param_decomp.llama_simple_mlp`
(`load_target_from_pretrain_cache`) reads safetensors keyed
`h.{i}.attn.{q,k,v,o}_proj.weight`, `h.{i}.mlp.{c_fc,down_proj}.weight`,
`h.{i}.rms_{1,2}.weight`, `wte.weight`, `ln_f.weight` (NO `lm_head.weight` — tied to
`wte`), every weight in torch `nn.Linear` orientation `(d_out, d_in)`. `state_dict`
(below) emits exactly those keys, so a freshly-pretrained target is decomposable with no
conversion. The other two archs follow the same key convention for symmetry.

All three are pre-norm decoder blocks under a flat `h.{i}.` module tree, `wte` tied to
`lm_head`, no biases on the Llama variants. RoPE is plain rotate-half
(`vendored_jax.llama.{rope_cos_sin,apply_rope}`); the GELU is the tanh approximation
(torch `NewGELU`), matching `llama_simple_mlp._gelu_tanh`.
"""

import math
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from param_decomp.base_config import BaseConfig
from vendored_jax.llama import apply_rope, causal_sdpa, repeat_kv, rms_norm, rope_cos_sin

# ----------------------------- configs -----------------------------


class GPT2SimpleConfig(BaseConfig):
    model_type: Literal["GPT2Simple"]
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    layer_norm_eps: float = 1e-5
    flash_attention: bool = True

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_intermediate(self) -> int:
        return 4 * self.n_embd


class LlamaSimpleConfig(BaseConfig):
    model_type: Literal["LlamaSimple"]
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_intermediate: int = 768 * 4 * 2 // 3
    mlp_bias: bool = False
    attn_bias: bool = False
    rotary_adjacent_pairs: bool = False
    rotary_dim: int = 768 // 12
    rotary_base: int = 10000
    n_ctx: int = 1024
    n_key_value_heads: int = 12 // 4
    use_grouped_query_attention: bool = True
    flash_attention: bool = True
    rms_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_rep(self) -> int:
        return self.n_head // self.n_key_value_heads


class LlamaSimpleMLPConfig(BaseConfig):
    model_type: Literal["LlamaSimpleMLP"]
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_intermediate: int = 768 * 4
    mlp_bias: bool = False
    attn_bias: bool = False
    rotary_adjacent_pairs: bool = False
    rotary_dim: int = 768 // 12
    rotary_base: int = 10000
    n_ctx: int = 1024
    n_key_value_heads: int = 12 // 4
    use_grouped_query_attention: bool = True
    flash_attention: bool = True
    rms_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_rep(self) -> int:
        return self.n_head // self.n_key_value_heads


ModelConfig = GPT2SimpleConfig | LlamaSimpleConfig | LlamaSimpleMLPConfig


def parse_model_config(raw: dict[str, object]) -> ModelConfig:
    match raw["model_type"]:
        case "GPT2Simple":
            return GPT2SimpleConfig(**raw)  # pyright: ignore[reportArgumentType]
        case "LlamaSimple":
            return LlamaSimpleConfig(**raw)  # pyright: ignore[reportArgumentType]
        case "LlamaSimpleMLP":
            return LlamaSimpleMLPConfig(**raw)  # pyright: ignore[reportArgumentType]
        case other:
            raise AssertionError(f"unknown model_type {other!r}")


# ----------------------------- init -----------------------------


def _linear_std(cfg: ModelConfig, residual_scaled: bool) -> float:
    """Torch `_init_weights`: 0.02, halved by `1/sqrt(2*n_layer)` on residual-output
    projections (`o_proj` / `down_proj`)."""
    return 0.02 / math.sqrt(2 * cfg.n_layer) if residual_scaled else 0.02


def _normal(key: Array, shape: tuple[int, ...], std: float) -> Array:
    return jax.random.normal(key, shape, dtype=jnp.float32) * std


# ----------------------------- GPT2Simple -----------------------------


def _gelu_tanh(x: Array) -> Array:
    """Torch `NewGELU` (tanh approximation)."""
    return jax.nn.gelu(x, approximate=True)


class GPT2SimpleAttention(eqx.Module):
    wq: Float[Array, "d d"]
    wk: Float[Array, "d d"]
    wv: Float[Array, "d d"]
    wo: Float[Array, "d d"]
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __call__(self, x: Float[Array, "b t d"]) -> Float[Array, "b t d"]:
        b, t, _ = x.shape
        q = (x @ self.wq.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = (x @ self.wk.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = (x @ self.wv.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        y = causal_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(b, t, self.n_head * self.head_dim)
        return y @ self.wo.T


class GPT2SimpleBlock(eqx.Module):
    ln1_w: Float[Array, " d"]
    ln1_b: Float[Array, " d"]
    ln2_w: Float[Array, " d"]
    ln2_b: Float[Array, " d"]
    attn: GPT2SimpleAttention
    Wfc: Float[Array, "di d"]
    Wdown: Float[Array, "d di"]
    eps: float = eqx.field(static=True)

    def __call__(self, x: Float[Array, "b t d"]) -> Float[Array, "b t d"]:
        x = x + self.attn(_layer_norm(x, self.ln1_w, self.ln1_b, self.eps))
        h = _layer_norm(x, self.ln2_w, self.ln2_b, self.eps)
        return x + _gelu_tanh(h @ self.Wfc.T) @ self.Wdown.T


def _layer_norm(x: Array, w: Array, b: Array, eps: float) -> Array:
    in_dtype = x.dtype
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(var + eps)
    return (w * x.astype(in_dtype)) + b


class GPT2Simple(eqx.Module):
    wte: Float[Array, "vocab d"]
    wpe: Float[Array, "block d"]
    blocks: list[GPT2SimpleBlock]
    lnf_w: Float[Array, " d"]
    lnf_b: Float[Array, " d"]
    n_ctx: int = eqx.field(static=True)

    def __call__(self, idx: Int[Array, "b t"]) -> Float[Array, "b t vocab"]:
        _, t = idx.shape
        assert t <= self.n_ctx, (t, self.n_ctx)
        x = self.wte[idx] + self.wpe[jnp.arange(t)][None]
        for block in self.blocks:
            x = block(x)
        x = _layer_norm(x, self.lnf_w, self.lnf_b, self.blocks[0].eps)
        return x @ self.wte.T

    def state_dict(self) -> dict[str, Array]:
        sd: dict[str, Array] = {"wte.weight": self.wte, "wpe.weight": self.wpe}
        sd["ln_f.weight"] = self.lnf_w
        sd["ln_f.bias"] = self.lnf_b
        for i, block in enumerate(self.blocks):
            sd[f"h.{i}.ln_1.weight"] = block.ln1_w
            sd[f"h.{i}.ln_1.bias"] = block.ln1_b
            sd[f"h.{i}.ln_2.weight"] = block.ln2_w
            sd[f"h.{i}.ln_2.bias"] = block.ln2_b
            sd[f"h.{i}.attn.q_proj.weight"] = block.attn.wq
            sd[f"h.{i}.attn.k_proj.weight"] = block.attn.wk
            sd[f"h.{i}.attn.v_proj.weight"] = block.attn.wv
            sd[f"h.{i}.attn.o_proj.weight"] = block.attn.wo
            sd[f"h.{i}.mlp.c_fc.weight"] = block.Wfc
            sd[f"h.{i}.mlp.down_proj.weight"] = block.Wdown
        return sd


def init_gpt2_simple(cfg: GPT2SimpleConfig, key: Array) -> GPT2Simple:
    keys = iter(jax.random.split(key, cfg.n_layer * 6 + 3))
    d, di = cfg.n_embd, cfg.n_intermediate
    blocks = [
        GPT2SimpleBlock(
            ln1_w=jnp.ones((d,)),
            ln1_b=jnp.zeros((d,)),
            ln2_w=jnp.ones((d,)),
            ln2_b=jnp.zeros((d,)),
            attn=GPT2SimpleAttention(
                wq=_normal(next(keys), (d, d), _linear_std(cfg, False)),
                wk=_normal(next(keys), (d, d), _linear_std(cfg, False)),
                wv=_normal(next(keys), (d, d), _linear_std(cfg, False)),
                wo=_normal(next(keys), (d, d), _linear_std(cfg, True)),
                n_head=cfg.n_head,
                head_dim=cfg.head_dim,
            ),
            Wfc=_normal(next(keys), (di, d), _linear_std(cfg, False)),
            Wdown=_normal(next(keys), (d, di), _linear_std(cfg, True)),
            eps=cfg.layer_norm_eps,
        )
        for _ in range(cfg.n_layer)
    ]
    return GPT2Simple(
        wte=_normal(next(keys), (cfg.vocab_size, d), 0.02),
        wpe=_normal(next(keys), (cfg.block_size, d), 0.02),
        blocks=blocks,
        lnf_w=jnp.ones((d,)),
        lnf_b=jnp.zeros((d,)),
        n_ctx=cfg.block_size,
    )


# ----------------------------- Llama variants (shared rotary GQA attention) -----------------------------


class LlamaAttention(eqx.Module):
    wq: Float[Array, "qd d"]
    wk: Float[Array, "kvd d"]
    wv: Float[Array, "kvd d"]
    wo: Float[Array, "d qd"]
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    n_rep: int = eqx.field(static=True)

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Float[Array, " hd2"]) -> Array:
        b, t, _ = x.shape
        q = (x @ self.wq.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = (x @ self.wk.T).reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = (x @ self.wv.T).reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        y = causal_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(b, t, self.n_head * self.head_dim)
        return y @ self.wo.T


def _plain_rope_inv_freq(cfg: "LlamaSimpleConfig | LlamaSimpleMLPConfig") -> Float[Array, " hd2"]:
    exponents = jnp.arange(0, cfg.head_dim, 2, dtype=jnp.float32) / cfg.head_dim
    return 1.0 / (cfg.rotary_base**exponents)


def _init_llama_attention(
    cfg: "LlamaSimpleConfig | LlamaSimpleMLPConfig", keys: "Callable[[], Array]"
) -> LlamaAttention:
    d = cfg.n_embd
    qd = cfg.n_head * cfg.head_dim
    kvd = cfg.n_key_value_heads * cfg.head_dim
    assert cfg.use_grouped_query_attention, "merged-qkv (c_attn) unsupported"
    assert not cfg.attn_bias, "attn bias unsupported"
    return LlamaAttention(
        wq=_normal(keys(), (qd, d), _linear_std(cfg, False)),
        wk=_normal(keys(), (kvd, d), _linear_std(cfg, False)),
        wv=_normal(keys(), (kvd, d), _linear_std(cfg, False)),
        wo=_normal(keys(), (d, qd), _linear_std(cfg, True)),
        n_head=cfg.n_head,
        n_kv_head=cfg.n_key_value_heads,
        head_dim=cfg.head_dim,
        n_rep=cfg.n_rep,
    )


class LlamaSimpleBlock(eqx.Module):
    """SwiGLU MLP block (`LlamaSimple`)."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: LlamaAttention
    Wgate: Float[Array, "di d"]
    Wup: Float[Array, "di d"]
    Wdown: Float[Array, "d di"]
    eps: float = eqx.field(static=True)

    def __call__(self, x: Array, inv_freq: Array) -> Array:
        x = x + self.attn(rms_norm(x, self.ln1, self.eps), inv_freq)
        h = rms_norm(x, self.ln2, self.eps)
        return x + (jax.nn.silu(h @ self.Wgate.T) * (h @ self.Wup.T)) @ self.Wdown.T


class LlamaSimple(eqx.Module):
    wte: Float[Array, "vocab d"]
    blocks: list[LlamaSimpleBlock]
    norm: Float[Array, " d"]
    inv_freq: Float[Array, " hd2"]
    n_ctx: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, idx: Int[Array, "b t"]) -> Float[Array, "b t vocab"]:
        t = idx.shape[1]
        assert t <= self.n_ctx, (t, self.n_ctx)
        x = self.wte[idx]
        for block in self.blocks:
            x = block(x, self.inv_freq)
        x = rms_norm(x, self.norm, self.eps)
        return x @ self.wte.T

    def state_dict(self) -> dict[str, Array]:
        sd: dict[str, Array] = {"wte.weight": self.wte, "ln_f.weight": self.norm}
        for i, block in enumerate(self.blocks):
            sd[f"h.{i}.rms_1.weight"] = block.ln1
            sd[f"h.{i}.rms_2.weight"] = block.ln2
            sd[f"h.{i}.attn.q_proj.weight"] = block.attn.wq
            sd[f"h.{i}.attn.k_proj.weight"] = block.attn.wk
            sd[f"h.{i}.attn.v_proj.weight"] = block.attn.wv
            sd[f"h.{i}.attn.o_proj.weight"] = block.attn.wo
            sd[f"h.{i}.mlp.gate_proj.weight"] = block.Wgate
            sd[f"h.{i}.mlp.up_proj.weight"] = block.Wup
            sd[f"h.{i}.mlp.down_proj.weight"] = block.Wdown
        return sd


def init_llama_simple(cfg: LlamaSimpleConfig, key: Array) -> LlamaSimple:
    keys_it = iter(jax.random.split(key, cfg.n_layer * 7 + 2))
    nxt = lambda: next(keys_it)
    d, di = cfg.n_embd, cfg.n_intermediate
    assert not cfg.mlp_bias, "mlp bias unsupported"
    blocks = [
        LlamaSimpleBlock(
            ln1=jnp.ones((d,)),
            ln2=jnp.ones((d,)),
            attn=_init_llama_attention(cfg, nxt),
            Wgate=_normal(nxt(), (di, d), _linear_std(cfg, False)),
            Wup=_normal(nxt(), (di, d), _linear_std(cfg, False)),
            Wdown=_normal(nxt(), (d, di), _linear_std(cfg, True)),
            eps=cfg.rms_norm_eps,
        )
        for _ in range(cfg.n_layer)
    ]
    return LlamaSimple(
        wte=_normal(nxt(), (cfg.vocab_size, d), 0.02),
        blocks=blocks,
        norm=jnp.ones((d,)),
        inv_freq=_plain_rope_inv_freq(cfg),
        n_ctx=cfg.n_ctx,
        eps=cfg.rms_norm_eps,
    )


class LlamaSimpleMLPBlock(eqx.Module):
    """GELU MLP block (`LlamaSimpleMLP`) — the primary decomposition target arch."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: LlamaAttention
    Wfc: Float[Array, "di d"]
    Wdown: Float[Array, "d di"]
    eps: float = eqx.field(static=True)

    def __call__(self, x: Array, inv_freq: Array) -> Array:
        x = x + self.attn(rms_norm(x, self.ln1, self.eps), inv_freq)
        h = rms_norm(x, self.ln2, self.eps)
        return x + _gelu_tanh(h @ self.Wfc.T) @ self.Wdown.T


class LlamaSimpleMLP(eqx.Module):
    wte: Float[Array, "vocab d"]
    blocks: list[LlamaSimpleMLPBlock]
    norm: Float[Array, " d"]
    inv_freq: Float[Array, " hd2"]
    n_ctx: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, idx: Int[Array, "b t"]) -> Float[Array, "b t vocab"]:
        t = idx.shape[1]
        assert t <= self.n_ctx, (t, self.n_ctx)
        x = self.wte[idx]
        for block in self.blocks:
            x = block(x, self.inv_freq)
        x = rms_norm(x, self.norm, self.eps)
        return x @ self.wte.T

    def state_dict(self) -> dict[str, Array]:
        """The exact keys `param_decomp.llama_simple_mlp` loads — no `lm_head` (tied)."""
        sd: dict[str, Array] = {"wte.weight": self.wte, "ln_f.weight": self.norm}
        for i, block in enumerate(self.blocks):
            sd[f"h.{i}.rms_1.weight"] = block.ln1
            sd[f"h.{i}.rms_2.weight"] = block.ln2
            sd[f"h.{i}.attn.q_proj.weight"] = block.attn.wq
            sd[f"h.{i}.attn.k_proj.weight"] = block.attn.wk
            sd[f"h.{i}.attn.v_proj.weight"] = block.attn.wv
            sd[f"h.{i}.attn.o_proj.weight"] = block.attn.wo
            sd[f"h.{i}.mlp.c_fc.weight"] = block.Wfc
            sd[f"h.{i}.mlp.down_proj.weight"] = block.Wdown
        return sd


def init_llama_simple_mlp(cfg: LlamaSimpleMLPConfig, key: Array) -> LlamaSimpleMLP:
    keys_it = iter(jax.random.split(key, cfg.n_layer * 6 + 2))
    nxt = lambda: next(keys_it)
    d, di = cfg.n_embd, cfg.n_intermediate
    assert not cfg.mlp_bias, "mlp bias unsupported"
    blocks = [
        LlamaSimpleMLPBlock(
            ln1=jnp.ones((d,)),
            ln2=jnp.ones((d,)),
            attn=_init_llama_attention(cfg, nxt),
            Wfc=_normal(nxt(), (di, d), _linear_std(cfg, False)),
            Wdown=_normal(nxt(), (d, di), _linear_std(cfg, True)),
            eps=cfg.rms_norm_eps,
        )
        for _ in range(cfg.n_layer)
    ]
    return LlamaSimpleMLP(
        wte=_normal(nxt(), (cfg.vocab_size, d), 0.02),
        blocks=blocks,
        norm=jnp.ones((d,)),
        inv_freq=_plain_rope_inv_freq(cfg),
        n_ctx=cfg.n_ctx,
        eps=cfg.rms_norm_eps,
    )


PretrainModel = GPT2Simple | LlamaSimple | LlamaSimpleMLP


def init_model(cfg: ModelConfig, key: Array) -> PretrainModel:
    match cfg:
        case GPT2SimpleConfig():
            return init_gpt2_simple(cfg, key)
        case LlamaSimpleConfig():
            return init_llama_simple(cfg, key)
        case LlamaSimpleMLPConfig():
            return init_llama_simple_mlp(cfg, key)


def model_logits(model: PretrainModel, idx: Int[Array, "b t"]) -> Float[Array, "b t vocab"]:
    return model(idx)
