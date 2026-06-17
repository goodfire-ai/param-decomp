"""The shared-transformer CI fn, generic over an ordered set of decomposed sites.

Mirrors torch `GlobalSharedTransformerCiFn` (SPEC §4.6) exactly: per-site clean inputs
are RMS-normed (weightless), concatenated in the canonical site order, linearly
projected to `d_model` (bias, NO nonlinearity), run through pre-norm bidirectional-RoPE
transformer blocks (weightless norms; bias-free q/k/v/out; biased GELU MLP), and a
final biased head emits `Σ_s C_s` logits split back per site.

The SAME logits are squashed two ways (SPEC S5/S6): `lower_leaky_hard` feeds the recon
/ PPGD masks; `upper_leaky_hard` feeds importance-minimality. Params are fp32 masters
(SPEC N1); the trainer casts for bf16 compute.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from vendored_jax.llama import apply_rope, attn_implementation, rms_norm, rope_cos_sin

from jax_single_pool.lm import SiteSpec

CI_FN_RMS_EPS = float(jnp.finfo(jnp.float32).eps)
"""Matches torch's `F.rms_norm` default eps (`finfo(fp32).eps` ~1.19e-7); RMS upcasts to
fp32 internally, so this is the dtype that governs (SPEC S4)."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CIValues:
    """The two squashed views of the CI-fn logits, per site (`{site: (B, T, C)}`)."""

    lower: dict[str, Array]
    upper: dict[str, Array]


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
    head_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Array) -> Array:
        b, t, d = x.shape
        h = _weightless_rms_norm(x, self.eps)
        q = (h @ self.wq.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = (h @ self.wk.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = (h @ self.wv.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        qt, kt, vt = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        y = jax.nn.dot_product_attention(
            qt, kt, vt, is_causal=False, implementation=attn_implementation()
        )  # bidirectional
        y = y.reshape(b, t, d)
        x = x + y @ self.wo.T
        h = _weightless_rms_norm(x, self.eps)
        return x + (jax.nn.gelu(h @ self.w1 + self.b1, approximate=False) @ self.w2 + self.b2)


class CIFn(eqx.Module):
    in_proj_w: Float[Array, "total_in d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: list[CIBlock]
    out_w: Float[Array, "d_model total_c"]
    out_b: Float[Array, " total_c"]
    inv_freq: Array
    site_names: tuple[str, ...] = eqx.field(static=True)
    split_sizes: tuple[int, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    expects_axes: tuple[str, ...] = eqx.field(static=True)
    """The leading position-axis names this CI fn expects; must equal the
    `DecomposedModel.leading_axes` it pairs with (asserted at trainer construction). This
    LM CI fn applies RoPE over a single `sequence` axis, so it expects `("sequence",)`."""

    def _site_slices(self) -> dict[str, slice]:
        offsets = [0]
        for c in self.split_sizes:
            offsets.append(offsets[-1] + c)
        return {name: slice(offsets[i], offsets[i + 1]) for i, name in enumerate(self.site_names)}

    def site_logits(self, site_inputs: dict[str, Array]) -> dict[str, Array]:
        """Pre-squash CI-fn logits, per site (`{site: (B, T, C)}`). This is torch's
        `pre_sigmoid` view (`CIHistograms` plots it alongside `lower_leaky`)."""
        assert set(site_inputs) == set(self.site_names), (
            f"site_inputs keys {sorted(site_inputs)} != CI fn sites {sorted(self.site_names)}"
        )
        normed = [_weightless_rms_norm(site_inputs[n], self.eps) for n in self.site_names]
        x = jnp.concatenate(normed, axis=-1) @ self.in_proj_w + self.in_proj_b
        inv_freq = jax.lax.stop_gradient(self.inv_freq)  # RoPE buffer, never trained
        for block in self.blocks:
            x = block(x, inv_freq)
        logits = x @ self.out_w + self.out_b  # (b, t, Σ_s C_s)
        site_slices = self._site_slices()
        return {name: logits[..., site_slices[name]] for name in self.site_names}

    def __call__(self, site_inputs: dict[str, Array]) -> CIValues:
        """`site_inputs`: clean per-site inputs, keyed by site name (canonical order is
        `self.site_names`). Returns the two squashings of the same logits, per site."""
        site_logits = self.site_logits(site_inputs)
        return CIValues(
            lower={name: lower_leaky_hard_sigmoid(site_logits[name]) for name in self.site_names},
            upper={name: upper_leaky_hard_sigmoid(site_logits[name]) for name in self.site_names},
        )


@dataclass(frozen=True)
class CIArch:
    d_model: int
    n_blocks: int
    n_heads: int
    mlp_hidden: int


def init_ci_fn(arch: CIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray) -> CIFn:
    """fp32 init matching torch: Kaiming-normal `N(0, gain/√fan_in)` on the custom
    linears (relu gain √2 on in_proj / MLP-in, linear gain 1 elsewhere), PyTorch-default
    `U(±1/√fan_in)` on the attention projections, zero biases."""
    key_iter = iter(jax.random.split(key, len(sites) + arch.n_blocks * 8 + 4))
    hd = arch.d_model // arch.n_heads
    assert arch.d_model % arch.n_heads == 0 and hd % 2 == 0, (arch.d_model, arch.n_heads)

    def kaiming(shape: tuple[int, ...], fan_in: int, gain: float) -> Array:
        return jax.random.normal(next(key_iter), shape) * (gain / fan_in**0.5)

    def attn_default(shape: tuple[int, ...], fan_in: int) -> Array:
        bound = 1.0 / fan_in**0.5
        return jax.random.uniform(next(key_iter), shape, minval=-bound, maxval=bound)

    relu_gain = 2.0**0.5

    def block() -> CIBlock:
        return CIBlock(
            wq=attn_default((arch.d_model, arch.d_model), arch.d_model),
            wk=attn_default((arch.d_model, arch.d_model), arch.d_model),
            wv=attn_default((arch.d_model, arch.d_model), arch.d_model),
            wo=attn_default((arch.d_model, arch.d_model), arch.d_model),
            w1=kaiming((arch.d_model, arch.mlp_hidden), arch.d_model, relu_gain),
            b1=jnp.zeros((arch.mlp_hidden,)),
            w2=kaiming((arch.mlp_hidden, arch.d_model), arch.mlp_hidden, 1.0),
            b2=jnp.zeros((arch.d_model,)),
            n_head=arch.n_heads,
            head_dim=hd,
            eps=CI_FN_RMS_EPS,
        )

    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))
    total_in = sum(s.d_in for s in sites)
    total_c = sum(s.C for s in sites)
    return CIFn(
        in_proj_w=kaiming((total_in, arch.d_model), total_in, relu_gain),
        in_proj_b=jnp.zeros((arch.d_model,)),
        blocks=[block() for _ in range(arch.n_blocks)],
        out_w=kaiming((arch.d_model, total_c), arch.d_model, 1.0),
        out_b=jnp.zeros((total_c,)),
        inv_freq=inv_freq,
        site_names=tuple(s.name for s in sites),
        split_sizes=tuple(s.C for s in sites),
        eps=CI_FN_RMS_EPS,
        expects_axes=("sequence",),
    )
