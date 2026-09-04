"""The Qwen3 architecture target over the shared GLU-transformer machinery.

Concrete configs cover every dense Qwen3 Base/post-trained variant through 14B.
Qwen3's one structural delta vs Llama is QK-norm:
per-head RMSNorm on q/k between the projection and RoPE (HF `Qwen3Attention.q_norm` /
`k_norm`) — `Qwen3FrozenAttn` carries the norm weights as REQUIRED fields (never an
optional flag) and applies them in the `_prep_qk` hook. Decomposition semantics are
unchanged: a masked q/k site output feeds q_norm → RoPE → SDPA.

JAX↔HF parity is pinned directly by `param_decomp/tests/targets/qwen3_hf_parity/`."""

from dataclasses import replace
from typing import override

import equinox as eqx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from param_decomp.core.axes import Axes
from param_decomp.core.components import SiteSpec
from param_decomp.core.placement import PlacementRules
from param_decomp.targets.glu_transformer import (
    FrozenAttn,
    GLUConfig,
    GLUDecomposedModel,
    HFWeights,
    default_inv_freq,
    load_decomposed_glu_from_hf,
)
from param_decomp.vendored_jax.llama import rms_norm


def _qwen3_config(
    *,
    n_layer: int,
    n_head: int,
    n_embd: int,
    n_intermediate: int,
    max_position_embeddings: int,
    tie_word_embeddings: bool,
) -> GLUConfig:
    return GLUConfig(
        vocab_size=151936,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=8,
        n_embd=n_embd,
        n_intermediate=n_intermediate,
        head_dim=128,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie_word_embeddings,
    )


def qwen3_0_6b_base_config() -> GLUConfig:
    """Architecture of `Qwen/Qwen3-0.6B-Base`. Its explicit 128-wide heads make
    the query projection twice the residual width; `n_embd // n_head` is not the
    head width for this model."""
    return _qwen3_config(
        n_layer=28,
        n_head=16,
        n_embd=1024,
        n_intermediate=3072,
        max_position_embeddings=32768,
        tie_word_embeddings=True,
    )


def qwen3_0_6b_config() -> GLUConfig:
    """Architecture of the post-trained `Qwen/Qwen3-0.6B`."""
    return replace(qwen3_0_6b_base_config(), max_position_embeddings=40960)


def qwen3_1_7b_base_config() -> GLUConfig:
    """Architecture of `Qwen/Qwen3-1.7B-Base`."""
    return _qwen3_config(
        n_layer=28,
        n_head=16,
        n_embd=2048,
        n_intermediate=6144,
        max_position_embeddings=32768,
        tie_word_embeddings=True,
    )


def qwen3_1_7b_config() -> GLUConfig:
    """Architecture of the post-trained `Qwen/Qwen3-1.7B`."""
    return replace(qwen3_1_7b_base_config(), max_position_embeddings=40960)


def qwen3_4b_base_config() -> GLUConfig:
    """Architecture of `Qwen/Qwen3-4B-Base`."""
    return _qwen3_config(
        n_layer=36,
        n_head=32,
        n_embd=2560,
        n_intermediate=9728,
        max_position_embeddings=32768,
        tie_word_embeddings=True,
    )


def qwen3_4b_config() -> GLUConfig:
    """Architecture of the post-trained `Qwen/Qwen3-4B`."""
    return replace(qwen3_4b_base_config(), max_position_embeddings=40960)


def qwen3_8b_base_config() -> GLUConfig:
    """Architecture of `Qwen/Qwen3-8B-Base`."""
    return _qwen3_config(
        n_layer=36,
        n_head=32,
        n_embd=4096,
        n_intermediate=12288,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
    )


def qwen3_8b_config() -> GLUConfig:
    """Architecture of the post-trained `Qwen/Qwen3-8B`."""
    return replace(qwen3_8b_base_config(), max_position_embeddings=40960)


def qwen3_14b_base_config() -> GLUConfig:
    """Architecture of `Qwen/Qwen3-14B-Base`."""
    return _qwen3_config(
        n_layer=40,
        n_head=40,
        n_embd=5120,
        n_intermediate=17408,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
    )


def qwen3_14b_config() -> GLUConfig:
    """Architecture of the post-trained `Qwen/Qwen3-14B`."""
    return replace(qwen3_14b_base_config(), max_position_embeddings=40960)


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
    def shardings(self, placement: PlacementRules, axes: Axes) -> "Qwen3FrozenAttn":
        """The shared projection layout plus declared norm-vector placement."""
        placed = super().shardings(placement, axes)
        assert isinstance(placed, Qwen3FrozenAttn)
        norm_axes: Axes = (*axes[:-2], "head_dim")
        placement.target.normalization.validate_shape(norm_axes, self.q_norm.shape)
        placement.target.normalization.validate_shape(norm_axes, self.k_norm.shape)
        norm = placement.target.normalization.sharding_for(norm_axes)
        return eqx.tree_at(lambda a: (a.q_norm, a.k_norm), placed, (norm, norm))


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
        implementation="auto",
        q_norm=w.get(f"{pre}.{i}.self_attn.q_norm.weight"),
        k_norm=w.get(f"{pre}.{i}.self_attn.k_norm.weight"),
        eps=cfg.rms_norm_eps,
    )


def load_decomposed_qwen3_from_hf(
    model_name: str,
    cfg: GLUConfig,
    sites: tuple[SiteSpec, ...],
    weights_dtype: DTypeLike,
) -> GLUDecomposedModel:
    """The Qwen3 family HF load: QK-norm attention + plain RoPE frequencies."""
    return load_decomposed_glu_from_hf(
        model_name,
        cfg,
        sites,
        load_attn=lambda w, i: _load_attn(w, i, cfg),
        weights_dtype=weights_dtype,
        inv_freq=default_inv_freq(cfg.head_dim, cfg.rope_theta),
    )
