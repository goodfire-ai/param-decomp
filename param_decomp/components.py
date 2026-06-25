"""The decomposition representation, shared by every target (LM and toy alike).

`SiteC` / `SiteSpec` are the per-site shape primitives (config-level name+C, and the
shape-carrying spec); `DecompVU` is the trainable per-site V/U master pytree;
`init_decomp_vu` seeds it; `site_out` is the one decomposed-linear primitive (SPEC §4.1,
`((x@V)*m)@U + (x@Δ)*d`). These are domain-neutral — they depend only on the site shapes
and the V/U/W arrays — so they live here rather than inside `lm.py` (whose `DecomposedModel`
Protocol references `DecompVU`/`SiteSpec`) or any one target.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from param_decomp.sharding import assert_divisible


@dataclass(frozen=True)
class SiteC:
    """A decomposed site as configured: its torch-module-path name and its C.

    The shape-carrying `SiteSpec` is derived from this plus the target's config."""

    name: str
    C: int


@dataclass(frozen=True)
class SiteSpec:
    name: str
    d_in: int
    d_out: int
    C: int


class DecompVU(eqx.Module):
    """fp32 master V `(d_in, C_s)` / U `(C_s, d_out)` per decomposed site, keyed by
    site name."""

    vu: dict[str, tuple[Float[Array, "d_in C"], Float[Array, "C d_out"]]]

    def site(self, name: str) -> tuple[Array, Array]:
        return self.vu[name]

    def shardings(self, mesh: "Mesh", replicate_u_dout: frozenset[str]) -> "DecompVU":
        """Uniform FSDP(`dp`)×TP(`tp`): V `(d_in, C)` shards d_in on `dp` + C on `tp`; U
        `(C, d_out)` shards C on `tp` + d_out on `dp`. C-on-`tp` aligns with the CI mask
        (no mask reshard); the weight delta is never formed (`site_out` is activation-space),
        so FSDP'ing d_in/d_out no longer reshards. EXCEPTION: sites in `replicate_u_dout`
        (the attention q/k/v projections — d_out is the head dim, and `tp` is already taken
        by C, so it can't live there) keep d_out REPLICATED and re-shard to head-on-`tp` at
        the attention seam. Asserts each sharded axis tiles its mesh axis."""
        shard_V = NamedSharding(mesh, P("dp", "tp"))
        shard_U_fsdp = NamedSharding(mesh, P("tp", "dp"))
        shard_U_repl = NamedSharding(mesh, P("tp", None))
        placed: dict[str, tuple[NamedSharding, NamedSharding]] = {}
        for name, (V, U) in self.vu.items():
            assert_divisible(V.shape[0], mesh, "dp", f"DecompVU[{name}].V.d_in")
            assert_divisible(V.shape[1], mesh, "tp", f"DecompVU[{name}].V.C")
            if name in replicate_u_dout:
                placed[name] = (shard_V, shard_U_repl)
            else:
                assert_divisible(U.shape[1], mesh, "dp", f"DecompVU[{name}].U.d_out")
                placed[name] = (shard_V, shard_U_fsdp)
        return DecompVU(vu=placed)  # pyright: ignore[reportArgumentType]


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


def site_out(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
) -> Array:
    """One decomposed linear (SPEC §4.1): `((x@V)*m)@U + (x@Δ)*d`, routed per position
    against the frozen `x @ W.T`. `mask` may be None (fully on); `route` None routes
    everywhere. `delta_mask` None drops the delta path entirely (constant-source entries
    carry no delta, LOSS_PARITY_DESIGN §4b). `delta_mask`/`route` broadcast over batch;
    trailing dim added here."""
    xV = x @ V
    acts = xV * mask if mask is not None else xV
    out = acts @ U
    if delta_mask is not None:
        # `(x @ Δ.T)` for `Δ = W − (V@U).T`, expanded to activation space as
        # `x@W.T − (x@V)@U` so the `[d_out, d_in]` weight delta is NEVER formed. Under
        # FSDP that delta would mix V's dp-sharded d_in with U's dp-sharded d_out (two dims
        # demanding the dp axis) and force a replicate-then-repartition reshard; the
        # activation-space form is all activation×weight matmuls that shard cleanly.
        # (Still a bf16-rounding DIVERGENCE vs the fp32 oracle delta — accepted; the
        # faithfulness loss uses the fp32 `weight_deltas`, SPEC N2, not this path.)
        out = out + delta_mask[..., None] * (x @ W.T - xV @ U)
    if route is not None:
        out = jnp.where(route[..., None], out, x @ W.T)
    return out
