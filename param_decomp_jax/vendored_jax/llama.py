"""JAX/Equinox port of the vendored Llama-3.1 decomposition target.

Faithful translation of `param_decomp_lab/experiments/lm/vendored/llama_3_1/{model,components}.py`
(itself verbatim-from-HF-transformers numeric kernels). NOT written from memory: every
computational line mirrors the torch vendored module, which cites transformers v4.57.3.

The torch design's whole point -- thread a path-keyed `mask_infos` dict through the forward
(no hooks) so the masked forward is pure -- IS the JAX-native style, so this port is a direct
structural mirror. ComponentLinear routes `(x, mask_info)` exactly like the torch one:
  mask_info None        -> frozen target:  x @ target_weight.T + bias
  mask_info given       -> components:     ((x @ V) * component_mask) @ U  (+ target where routed)

V/U math matches `param_decomp.components.LinearComponents`: weight == (V @ U).T,
component_acts == x @ V, out == (component_acts * mask) @ U.

v1 scope for the parity spike: routing == "all", no weight-delta term (both optional/None on
the clean + stochastic paths). RoPE uses the llama3 frequency rescaling.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    n_intermediate: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    # llama3 rope scaling (None => plain rope)
    rope_factor: float | None = None
    rope_low_freq_factor: float | None = None
    rope_high_freq_factor: float | None = None
    rope_original_max_position_embeddings: int | None = None

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_rep(self) -> int:
        return self.n_head // self.n_kv_head


# ----------------------------- numeric kernels (mirror torch verbatim) -----------------------------


def llama3_inv_freq(cfg: LlamaConfig) -> Float[Array, " hd2"]:
    dim = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    if cfg.rope_factor is None:
        return inv_freq
    factor = cfg.rope_factor
    low = cfg.rope_low_freq_factor
    high = cfg.rope_high_freq_factor
    old_ctx = cfg.rope_original_max_position_embeddings
    low_freq_wavelen = old_ctx / low
    high_freq_wavelen = old_ctx / high
    wavelen = 2 * jnp.pi / inv_freq
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return jnp.where(is_medium, smoothed, inv_freq_llama)


def rms_norm(
    x: Float[Array, "... d"], weight: Float[Array, " d"], eps: float
) -> Float[Array, "... d"]:
    in_dtype = x.dtype
    x = x.astype(jnp.float32)
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(var + eps)
    return weight * x.astype(in_dtype)


def rope_cos_sin(inv_freq: Float[Array, " hd2"], seq_len: int, dtype) -> tuple[Array, Array]:
    pos = jnp.arange(seq_len, dtype=jnp.float32)  # (T,)
    freqs = jnp.einsum("f,t->tf", inv_freq, pos)  # (T, hd2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (T, hd)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def rotate_half(x: Array) -> Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(q: Array, k: Array, cos: Array, sin: Array) -> tuple[Array, Array]:
    # q,k: (B, n_head, T, hd); cos,sin: (T, hd) -> broadcast over (B, head)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x: Float[Array, "b kvh t hd"], n_rep: int) -> Float[Array, "b h t hd"]:
    b, kvh, t, hd = x.shape
    if n_rep == 1:
        return x
    x = jnp.broadcast_to(x[:, :, None, :, :], (b, kvh, n_rep, t, hd))
    return x.reshape(b, kvh * n_rep, t, hd)


def attn_implementation() -> Literal["cudnn", "xla"]:
    """cuDNN flash attention on GPU; the XLA composite elsewhere (CPU tests). The
    composite MATERIALIZES (B, H, T, T) score matrices — at seq 2048 that is ~2 GiB
    per suffix forward per layer and was the dominant term in the trainer's OOM."""
    return "cudnn" if jax.default_backend() == "gpu" else "xla"


def causal_sdpa(q: Array, k: Array, v: Array) -> Array:
    # q,k,v: (B, H, T, hd); jax.nn.dot_product_attention takes (B, T, H, D).
    qt, kt, vt = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
    out = jax.nn.dot_product_attention(
        qt, kt, vt, is_causal=True, implementation=attn_implementation()
    )
    return out.transpose(0, 2, 1, 3)


# ----------------------------- component leaf (mirror ComponentLinear) -----------------------------


class MaskInfo(NamedTuple):
    component_mask: Float[Array, "... C"]  # routing == "all", no weight-delta in v1


class ComponentLinear(eqx.Module):
    V: Float[Array, "d_in C"]  # trainable
    U: Float[Array, "C d_out"]  # trainable
    target_weight: Float[Array, "d_out d_in"]  # frozen (eqx leaf; filtered out for grads)
    bias: Float[Array, " d_out"] | None

    def target_forward(self, x: Array) -> Array:
        y = x @ self.target_weight.T
        return y if self.bias is None else y + self.bias

    def __call__(self, x: Array, mask_info: MaskInfo | None) -> Array:
        if mask_info is None:
            return self.target_forward(x)
        comp_acts = x @ self.V  # (... C)
        comp_acts = comp_acts * mask_info.component_mask
        out = comp_acts @ self.U  # (... d_out)
        return out if self.bias is None else out + self.bias


class Attention(eqx.Module):
    q_proj: ComponentLinear
    k_proj: ComponentLinear
    v_proj: ComponentLinear
    o_proj: ComponentLinear
    inv_freq: Float[Array, " hd2"] = eqx.field(static=False)
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    n_rep: int = eqx.field(static=True)
    paths: tuple[str, str, str, str] = eqx.field(static=True)  # q,k,v,o mask keys

    def __call__(self, x: Float[Array, "b t d"], masks: "dict | None") -> Array:
        b, t, _ = x.shape
        mi = lambda p: None if masks is None else masks.get(p)
        q = self.q_proj(x, mi(self.paths[0]))
        k = self.k_proj(x, mi(self.paths[1]))
        v = self.v_proj(x, mi(self.paths[2]))
        q = q.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(self.inv_freq, t, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        y = causal_sdpa(q, k, v)  # (b, h, t, hd)
        y = y.transpose(0, 2, 1, 3).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(y, mi(self.paths[3]))


class MLP(eqx.Module):
    gate_proj: ComponentLinear
    up_proj: ComponentLinear
    down_proj: ComponentLinear
    paths: tuple[str, str, str] = eqx.field(static=True)

    def __call__(self, x: Array, masks: "dict | None") -> Array:
        mi = lambda p: None if masks is None else masks.get(p)
        gate = self.gate_proj(x, mi(self.paths[0]))
        up = self.up_proj(x, mi(self.paths[1]))
        return self.down_proj(jax.nn.silu(gate) * up, mi(self.paths[2]))


class Block(eqx.Module):
    input_layernorm: Float[Array, " d"]
    post_attention_layernorm: Float[Array, " d"]
    self_attn: Attention
    mlp: MLP
    eps: float = eqx.field(static=True)

    def __call__(self, x: Array, masks: "dict | None") -> Array:
        x = x + self.self_attn(rms_norm(x, self.input_layernorm, self.eps), masks)
        x = x + self.mlp(rms_norm(x, self.post_attention_layernorm, self.eps), masks)
        return x


class ComponentLlama(eqx.Module):
    embed_tokens: Float[Array, "vocab d"]
    blocks: list[Block]
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    eps: float = eqx.field(static=True)

    def __call__(
        self, idx: Int[Array, "b t"], masks: "dict | None" = None
    ) -> Float[Array, "b t vocab"]:
        x = self.embed_tokens[idx]  # (b, t, d)
        for block in self.blocks:
            x = block(x, masks)
        x = rms_norm(x, self.norm, self.eps)
        return x @ self.lm_head.T


# ----------------------------- build from HF state dict -----------------------------

# decomposition-target leaves (per block) and their HF-stripped key suffixes
_ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
_MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def _clin(sd: dict, path: str) -> ComponentLinear:
    return ComponentLinear(
        V=sd[f"{path}.components.V"],
        U=sd[f"{path}.components.U"],
        target_weight=sd[f"{path}.target_weight"],
        bias=sd.get(f"{path}.bias"),
    )


def build_from_torch_state(cfg: LlamaConfig, sd: dict[str, Array]) -> ComponentLlama:
    """sd: torch ComponentLlama state keyed HF-style (model. prefix already stripped),
    with `layers.{i}.<leaf>.target_weight` / `.components.V` / `.components.U`."""
    inv_freq = llama3_inv_freq(cfg)
    blocks = []
    for i in range(cfg.n_layer):
        p = f"layers.{i}"
        attn = Attention(
            q_proj=_clin(sd, f"{p}.self_attn.q_proj"),
            k_proj=_clin(sd, f"{p}.self_attn.k_proj"),
            v_proj=_clin(sd, f"{p}.self_attn.v_proj"),
            o_proj=_clin(sd, f"{p}.self_attn.o_proj"),
            inv_freq=inv_freq,
            n_head=cfg.n_head,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            n_rep=cfg.n_rep,
            paths=tuple(f"{p}.{s}" for s in _ATTN),
        )
        mlp = MLP(
            gate_proj=_clin(sd, f"{p}.mlp.gate_proj"),
            up_proj=_clin(sd, f"{p}.mlp.up_proj"),
            down_proj=_clin(sd, f"{p}.mlp.down_proj"),
            paths=tuple(f"{p}.{s}" for s in _MLP),
        )
        blocks.append(
            Block(
                input_layernorm=sd[f"{p}.input_layernorm.weight"],
                post_attention_layernorm=sd[f"{p}.post_attention_layernorm.weight"],
                self_attn=attn,
                mlp=mlp,
                eps=cfg.rms_norm_eps,
            )
        )
    return ComponentLlama(
        embed_tokens=sd["embed_tokens.weight"],
        blocks=blocks,
        norm=sd["norm.weight"],
        lm_head=sd["lm_head.weight"],
        eps=cfg.rms_norm_eps,
    )


def all_target_paths(cfg: LlamaConfig) -> list[str]:
    return [f"layers.{i}.{s}" for i in range(cfg.n_layer) for s in (_ATTN + _MLP)]


def site_shapes(cfg: LlamaConfig) -> dict[str, tuple[int, int]]:
    """(d_in, d_out) per decomposition-target leaf."""
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    per_layer = {
        "self_attn.q_proj": (d, qd),
        "self_attn.k_proj": (d, kvd),
        "self_attn.v_proj": (d, kvd),
        "self_attn.o_proj": (qd, d),
        "mlp.gate_proj": (d, di),
        "mlp.up_proj": (d, di),
        "mlp.down_proj": (di, d),
    }
    return {f"layers.{i}.{k}": v for i in range(cfg.n_layer) for k, v in per_layer.items()}


def random_init(cfg: LlamaConfig, C: int, key) -> ComponentLlama:
    """Random ComponentLlama with C components/site — for benchmarking at scale (no weights)."""
    shapes = site_shapes(cfg)
    ks = iter(jax.random.split(key, len(shapes) * 3 + cfg.n_layer * 2 + 4))

    def clin(d_in, d_out):
        sc = 1.0 / (d_in**0.5)
        return ComponentLinear(
            V=jax.random.normal(next(ks), (d_in, C)) * sc,
            U=jax.random.normal(next(ks), (C, d_out)) * (1.0 / C**0.5),
            target_weight=jax.random.normal(next(ks), (d_out, d_in)) * sc,
            bias=None,
        )

    inv_freq = llama3_inv_freq(cfg)
    blocks = []
    for i in range(cfg.n_layer):
        p = f"layers.{i}"
        attn = Attention(
            q_proj=clin(*shapes[f"{p}.self_attn.q_proj"]),
            k_proj=clin(*shapes[f"{p}.self_attn.k_proj"]),
            v_proj=clin(*shapes[f"{p}.self_attn.v_proj"]),
            o_proj=clin(*shapes[f"{p}.self_attn.o_proj"]),
            inv_freq=inv_freq,
            n_head=cfg.n_head,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            n_rep=cfg.n_rep,
            paths=tuple(f"{p}.{s}" for s in _ATTN),
        )
        mlp = MLP(
            gate_proj=clin(*shapes[f"{p}.mlp.gate_proj"]),
            up_proj=clin(*shapes[f"{p}.mlp.up_proj"]),
            down_proj=clin(*shapes[f"{p}.mlp.down_proj"]),
            paths=tuple(f"{p}.{s}" for s in _MLP),
        )
        blocks.append(
            Block(
                input_layernorm=jnp.ones((cfg.n_embd,)),
                post_attention_layernorm=jnp.ones((cfg.n_embd,)),
                self_attn=attn,
                mlp=mlp,
                eps=cfg.rms_norm_eps,
            )
        )
    return ComponentLlama(
        embed_tokens=jax.random.normal(next(ks), (cfg.vocab_size, cfg.n_embd)) * 0.02,
        blocks=blocks,
        norm=jnp.ones((cfg.n_embd,)),
        lm_head=jax.random.normal(next(ks), (cfg.vocab_size, cfg.n_embd)) * 0.02,
        eps=cfg.rms_norm_eps,
    )
