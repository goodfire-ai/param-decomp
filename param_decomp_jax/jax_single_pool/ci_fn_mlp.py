"""The layerwise per-site MLP CI fn — the sibling of `ci_fn.py`'s transformer for
positionless (`leading_axes=()`) targets like TMS.

One independent MLP per decomposed site maps that site's clean input `[*leading, d_in]`
to `[*leading, C]` pre-squash logits; the SAME logits feed the shared
`lower_leaky_hard` (recon/PPGD masks) and `upper_leaky_hard` (importance-minimality)
squashings (SPEC S5/S6), exactly as the transformer CI fn. Params are fp32 masters
(SPEC N1); the trainer casts for bf16 compute.

`expects_axes = ()` (no position axes): the MLP is applied independently over every
leading cell, so it places no structural constraint on the leading prefix — it works for
any `leading_axes` that the paired model declares empty. (The transformer CI fn, by
contrast, applies RoPE over a `sequence` axis and so declares `expects_axes=("sequence",)`.)

This is the vector-input per-site MLP: each site's MLP consumes the full `[*leading, d_in]`
site input and emits `C` logits (torch `VectorMLPCiFn` shape; see the JAX TMS CLAUDE
note on why this is the chosen `fn_type=mlp` realization rather than torch's scalar
`get_component_acts(x)=x@V` coupling, which would change the generic `ci_fn(site_inputs)`
contract)."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jax_single_pool.ci_fn import CIValues, lower_leaky_hard_sigmoid, upper_leaky_hard_sigmoid
from jax_single_pool.lm import SiteSpec


@dataclass(frozen=True)
class MLPCIArch:
    """Hidden widths shared by every per-site MLP (torch `LayerwiseCiConfig.hidden_dims`)."""

    hidden_dims: tuple[int, ...]


class SiteMLP(eqx.Module):
    """One site's MLP: `hidden_dims` ReLU-init Linear+GELU layers then a linear head to C.

    Matches torch `VectorMLPCiFn` layer structure: each hidden layer is Kaiming-`relu`
    (`gain √2`) initialized with zero bias, the final head is linear-gain (`1`)."""

    weights: list[Float[Array, "d_in d_out"]]
    biases: list[Float[Array, " d_out"]]

    def __call__(self, x: Float[Array, "*leading d_in"]) -> Float[Array, "*leading C"]:
        n_hidden = len(self.weights) - 1
        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ w + b
            if layer_idx < n_hidden:
                x = jax.nn.gelu(x, approximate=False)
        return x


class LayerwiseMLPCIFn(eqx.Module):
    """A per-site MLP bundle behind the same dict-in / `CIValues`-out interface as the
    transformer `CIFn` (`__call__(site_inputs) -> CIValues`)."""

    site_mlps: dict[str, SiteMLP]
    site_names: tuple[str, ...] = eqx.field(static=True)
    expects_axes: tuple[str, ...] = eqx.field(static=True)

    def site_logits(self, site_inputs: dict[str, Array]) -> dict[str, Array]:
        assert set(site_inputs) == set(self.site_names), (
            f"site_inputs keys {sorted(site_inputs)} != CI fn sites {sorted(self.site_names)}"
        )
        return {name: self.site_mlps[name](site_inputs[name]) for name in self.site_names}

    def __call__(self, site_inputs: dict[str, Array]) -> CIValues:
        logits = self.site_logits(site_inputs)
        return CIValues(
            lower={name: lower_leaky_hard_sigmoid(logits[name]) for name in self.site_names},
            upper={name: upper_leaky_hard_sigmoid(logits[name]) for name in self.site_names},
        )


def init_layerwise_mlp_ci_fn(
    arch: MLPCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> LayerwiseMLPCIFn:
    """Per-site MLP init: each site's MLP maps `d_in -> hidden_dims... -> C`, Kaiming
    `relu`-gain on the hidden layers (matching torch `init_param_` fan-in init) and
    linear gain on the C head, zero biases."""
    assert arch.hidden_dims, "MLP CI fn needs at least one hidden layer"
    relu_gain = 2.0**0.5
    site_mlps: dict[str, SiteMLP] = {}
    for site_idx, spec in enumerate(sites):
        dims = (spec.d_in, *arch.hidden_dims, spec.C)
        layer_keys = jax.random.split(jax.random.fold_in(key, site_idx), len(dims) - 1)
        weights: list[Array] = []
        biases: list[Array] = []
        for layer_idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            gain = relu_gain if layer_idx < len(dims) - 2 else 1.0
            weights.append(
                jax.random.normal(layer_keys[layer_idx], (d_in, d_out)) * (gain / d_in**0.5)
            )
            biases.append(jnp.zeros((d_out,)))
        site_mlps[spec.name] = SiteMLP(weights=weights, biases=biases)
    return LayerwiseMLPCIFn(
        site_mlps=site_mlps, site_names=tuple(s.name for s in sites), expects_axes=()
    )
