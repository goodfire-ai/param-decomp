"""The Qwen3-8B-Base family target over the shared GLU-transformer machinery
(`targets/glu_transformer.py`). Qwen3's one structural delta vs Llama is QK-norm:
per-head RMSNorm on q/k between the projection and RoPE (HF `Qwen3Attention.q_norm` /
`k_norm`) — `Qwen3FrozenAttn` carries the norm weights as REQUIRED fields (never an
optional flag) and applies them in the `_prep_qk` hook. Decomposition semantics are
unchanged: a masked q/k site output feeds q_norm → RoPE → SDPA.

JAX↔HF parity is pinned directly by `param_decomp/tests/qwen3_hf_parity/`."""

from typing import override

import equinox as eqx
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from param_decomp.components import SiteSpec
from param_decomp.targets.glu_transformer import (
    FrozenAttn,
    GLUConfig,
    GLUDecomposedModel,
    HFWeights,
    default_inv_freq,
    load_decomposed_glu_from_hf,
)
from vendored_jax.llama import rms_norm


def qwen3_8b_config() -> GLUConfig:
    """Qwen3-8B(-Base) per HF `Qwen/Qwen3-8B-Base` config.json. HF's explicit `head_dim`
    (128) coincides with `n_embd // n_head` for 8B, so the derived `GLUConfig.head_dim`
    is exact (NOT true of every Qwen3 size — check before adding one)."""
    return GLUConfig(
        vocab_size=151936,
        n_layer=36,
        n_head=32,
        n_kv_head=8,
        n_embd=4096,
        n_intermediate=12288,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
    )


class Qwen3FrozenAttn(FrozenAttn):
    """GQA attention + Qwen3 QK-norm (per-head RMSNorm over head_dim, before RoPE)."""

    q_norm: Float[Array, " hd"]
    k_norm: Float[Array, " hd"]
    eps: float = eqx.field(static=True)
    """RMSNorm eps for the QK-norm only (block norms use the model-level eps)."""

    def __check_init__(self):
        assert self.eps > 0.0, "QK-norm needs a real eps"

    @override
    def _prep_qk(
        self, q: Float[Array, "b t h hd"], k: Float[Array, "b t kvh hd"]
    ) -> tuple[Array, Array]:
        return rms_norm(q, self.q_norm, self.eps), rms_norm(k, self.k_norm, self.eps)

    @override
    def shardings(self, mesh: Mesh) -> "Qwen3FrozenAttn":
        """The shared projection layout plus replicated norm vectors."""
        repl = NamedSharding(mesh, P())
        placed = super().shardings(mesh)
        assert isinstance(placed, Qwen3FrozenAttn)
        return eqx.tree_at(lambda a: (a.q_norm, a.k_norm), placed, (repl, repl))


def _load_attn(w: HFWeights, i: int, cfg: GLUConfig) -> Qwen3FrozenAttn:
    pre = "model.layers"
    return Qwen3FrozenAttn(
        wq=w.get(f"{pre}.{i}.self_attn.q_proj.weight"),
        wk=w.get(f"{pre}.{i}.self_attn.k_proj.weight"),
        wv=w.get(f"{pre}.{i}.self_attn.v_proj.weight"),
        wo=w.get(f"{pre}.{i}.self_attn.o_proj.weight"),
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        head_dim=cfg.head_dim,
        n_rep=cfg.n_rep,
        q_norm=w.get(f"{pre}.{i}.self_attn.q_norm.weight"),
        k_norm=w.get(f"{pre}.{i}.self_attn.k_norm.weight"),
        eps=cfg.rms_norm_eps,
    )


def load_decomposed_qwen3_from_hf(
    model_name: str,
    cfg: GLUConfig,
    sites: tuple[SiteSpec, ...],
    scan_unroll: int = 1,
    gather_fp8: bool = False,
) -> GLUDecomposedModel:
    """The Qwen3 family HF load: QK-norm attention + plain RoPE frequencies."""
    return load_decomposed_glu_from_hf(
        model_name,
        cfg,
        sites,
        load_attn=lambda w, i: _load_attn(w, i, cfg),
        weights_dtype=jnp.bfloat16,  # the family is bf16-only (TargetConfig.supported_weights_dtypes)
        inv_freq=default_inv_freq(cfg.head_dim, cfg.rope_theta),
        scan_unroll=scan_unroll,
        gather_fp8=gather_fp8,
    )
