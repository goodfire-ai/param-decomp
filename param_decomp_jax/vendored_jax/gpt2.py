"""JAX/Equinox port of the vendored GPT-2 (GPT2Simple / ComponentGPT2) decomposition target.

Faithful translation of `pretrain/models/gpt2_simple.py` + `vendored/gpt2.py`. GPT-2 differs
from Llama: LayerNorm (weight+bias) not RMSNorm, tanh-GELU, learned positional embeddings (no
RoPE), split q/k/v/o `nn.Linear` WITH bias, and `wte` tied to `lm_head`.

Reuses the generic `ComponentLinear` / `MaskInfo` / `causal_sdpa` from the Llama port — the V/U
routing math is identical; only the surrounding architecture changes.
"""

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from vendored_jax.llama import ComponentLinear, causal_sdpa


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    ln_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def layer_norm(x: Float[Array, "... d"], w: Array, b: Array, eps: float) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)  # ddof=0 == torch var(unbiased=False)
    return (x - mean) * jax.lax.rsqrt(var + eps) * w + b


def new_gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class Attention(eqx.Module):
    q_proj: ComponentLinear
    k_proj: ComponentLinear
    v_proj: ComponentLinear
    o_proj: ComponentLinear
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    paths: tuple[str, str, str, str] = eqx.field(static=True)

    def __call__(self, x: Float[Array, "b t d"], masks: "dict | None") -> Array:
        b, t, c = x.shape
        mi = lambda p: None if masks is None else masks.get(p)
        q = self.q_proj(x, mi(self.paths[0]))
        k = self.k_proj(x, mi(self.paths[1]))
        v = self.v_proj(x, mi(self.paths[2]))
        # (b, t, n_head, hd) -> (b, n_head, t, hd); GPT-2 has no GQA and no RoPE
        q = q.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        y = causal_sdpa(q, k, v)
        y = y.transpose(0, 2, 1, 3).reshape(b, t, c)
        return self.o_proj(y, mi(self.paths[3]))


class MLP(eqx.Module):
    c_fc: ComponentLinear
    down_proj: ComponentLinear
    paths: tuple[str, str] = eqx.field(static=True)

    def __call__(self, x: Array, masks: "dict | None") -> Array:
        mi = lambda p: None if masks is None else masks.get(p)
        x = self.c_fc(x, mi(self.paths[0]))
        x = new_gelu(x)
        return self.down_proj(x, mi(self.paths[1]))


class Block(eqx.Module):
    ln_1_w: Array
    ln_1_b: Array
    ln_2_w: Array
    ln_2_b: Array
    attn: Attention
    mlp: MLP
    eps: float = eqx.field(static=True)

    def __call__(self, x: Array, masks: "dict | None") -> Array:
        x = x + self.attn(layer_norm(x, self.ln_1_w, self.ln_1_b, self.eps), masks)
        x = x + self.mlp(layer_norm(x, self.ln_2_w, self.ln_2_b, self.eps), masks)
        return x


class ComponentGPT2(eqx.Module):
    wte: Float[Array, "vocab d"]  # tied to lm_head
    wpe: Float[Array, "block d"]
    blocks: list[Block]
    ln_f_w: Array
    ln_f_b: Array
    eps: float = eqx.field(static=True)

    def __call__(
        self, idx: Int[Array, "b t"], masks: "dict | None" = None
    ) -> Float[Array, "b t vocab"]:
        t = idx.shape[1]
        x = self.wte[idx] + self.wpe[jnp.arange(t)]
        for block in self.blocks:
            x = block(x, masks)
        x = layer_norm(x, self.ln_f_w, self.ln_f_b, self.eps)
        return x @ self.wte.T  # tied head


_ATTN = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"]
_MLP = ["mlp.c_fc", "mlp.down_proj"]


def _clin(sd: dict, path: str) -> ComponentLinear:
    return ComponentLinear(
        V=sd[f"{path}.components.V"],
        U=sd[f"{path}.components.U"],
        target_weight=sd[f"{path}.target_weight"],
        bias=sd.get(f"{path}.bias"),
    )


def build_from_torch_state(cfg: GPT2Config, sd: dict[str, Array]) -> ComponentGPT2:
    blocks = []
    for i in range(cfg.n_layer):
        p = f"h.{i}"
        attn = Attention(
            q_proj=_clin(sd, f"{p}.attn.q_proj"),
            k_proj=_clin(sd, f"{p}.attn.k_proj"),
            v_proj=_clin(sd, f"{p}.attn.v_proj"),
            o_proj=_clin(sd, f"{p}.attn.o_proj"),
            n_head=cfg.n_head,
            head_dim=cfg.head_dim,
            paths=tuple(f"{p}.{s}" for s in _ATTN),
        )
        mlp = MLP(
            c_fc=_clin(sd, f"{p}.mlp.c_fc"),
            down_proj=_clin(sd, f"{p}.mlp.down_proj"),
            paths=tuple(f"{p}.{s}" for s in _MLP),
        )
        blocks.append(
            Block(
                ln_1_w=sd[f"{p}.ln_1.weight"],
                ln_1_b=sd[f"{p}.ln_1.bias"],
                ln_2_w=sd[f"{p}.ln_2.weight"],
                ln_2_b=sd[f"{p}.ln_2.bias"],
                attn=attn,
                mlp=mlp,
                eps=cfg.ln_eps,
            )
        )
    return ComponentGPT2(
        wte=sd["wte.weight"],
        wpe=sd["wpe.weight"],
        blocks=blocks,
        ln_f_w=sd["ln_f.weight"],
        ln_f_b=sd["ln_f.bias"],
        eps=cfg.ln_eps,
    )


def all_target_paths(cfg: GPT2Config) -> list[str]:
    return [f"h.{i}.{s}" for i in range(cfg.n_layer) for s in (_ATTN + _MLP)]
