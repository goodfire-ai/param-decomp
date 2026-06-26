"""The decomposition representation, shared by every target (LM and toy alike).

`SiteC` / `SiteSpec` are the per-site shape primitives (config-level name+C, and the
shape-carrying spec); `DecompVU` is the trainable per-site V/U master pytree;
`init_decomp_vu` seeds it; `site_out` is the one decomposed-linear primitive (SPEC §4.1,
`((x@V)*m)@U + (x@Δ)*d`). These are domain-neutral — they depend only on the site shapes
and the V/U/W arrays — so they live here rather than inside `lm.py` (whose `DecomposedModel`
Protocol references `DecompVU`/`SiteSpec`) or any one target.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.sharding import DATA_AXES, assert_divisible


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


# The V/U leaf type: `Array` for the real fp32 masters (the default — so bare `DecompVU`
# means `DecompVU[Array]` and no call site needs the parameter), or `NamedSharding` for the
# same-structure placement tree `.shardings` returns for `jax.jit(out_shardings=...)`.
VULeaf = TypeVar("VULeaf", default=Array)


class DecompVU(eqx.Module, Generic[VULeaf]):
    """Per-decomposed-site V `(d_in, C_s)` / U `(C_s, d_out)`, keyed by site name. The leaves
    are fp32 master Arrays (`DecompVU[Array]`), or `NamedSharding`s in the placement tree
    returned by `.shardings` (`DecompVU[NamedSharding]`) — same pytree structure, sharding
    leaves, for `jax.jit(out_shardings=...)`."""

    vu: dict[str, tuple[VULeaf, VULeaf]]

    def site(self, name: str) -> tuple[VULeaf, VULeaf]:
        return self.vu[name]

    def shardings(
        self: "DecompVU[Array]", mesh: "Mesh", replicate_u_dout: frozenset[str]
    ) -> "DecompVU[NamedSharding]":
        """ZeRO-1 ÷N FSDP(`replicate × dp`)×TP(`tp`) for the STORED masters: V `(d_in, C)`
        shards d_in over BOTH data-parallel axes (`DATA_AXES = replicate × dp`) + C on `tp`;
        U `(C, d_out)` shards C on `tp` + d_out over `DATA_AXES`. This is the PERSISTENCE
        layout — master + Adam mu/nu memory ÷ (replicate·dp·tp) = ÷N over the FULL mesh (the
        612 GB → ~5 GB/GPU win). `_hoist_vu_gather` gathers the `DATA_AXES`-sharded d_in/d_out
        ONCE per step in ENTRY (the ZeRO-1 reconstruction, incl. the cross-node `replicate`
        leg — off the hot path); the per-layer scan body then runs batch-DP + C-on-`tp` with
        NO weight gather. C-on-`tp` aligns with the CI mask (no mask reshard). EXCEPTION:
        sites in `replicate_u_dout` (the attention q/k/v projections — d_out is the head dim,
        and `tp` is already taken by C, so it can't live there) keep d_out REPLICATED and
        re-shard to head-on-`tp` at the attention seam. Asserts each sharded axis tiles its
        mesh — d_in/d_out must tile `replicate·dp` (the product of the two FSDP axes)."""
        shard_V = NamedSharding(mesh, P(DATA_AXES, "tp"))
        shard_U_fsdp = NamedSharding(mesh, P("tp", DATA_AXES))
        shard_U_repl = NamedSharding(mesh, P("tp", None))
        n_fsdp = mesh.shape["replicate"] * mesh.shape["dp"]
        placed: dict[str, tuple[NamedSharding, NamedSharding]] = {}
        for name, (V, U) in self.vu.items():
            assert V.shape[0] % n_fsdp == 0, (
                f"DecompVU[{name}].V.d_in {V.shape[0]} not divisible by replicate·dp {n_fsdp}"
            )
            assert_divisible(V.shape[1], mesh, "tp", f"DecompVU[{name}].V.C")
            if name in replicate_u_dout:
                placed[name] = (shard_V, shard_U_repl)
            else:
                assert U.shape[1] % n_fsdp == 0, (
                    f"DecompVU[{name}].U.d_out {U.shape[1]} not divisible by replicate·dp {n_fsdp}"
                )
                placed[name] = (shard_V, shard_U_fsdp)
        return DecompVU(vu=placed)


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
    # Pin the decomposed matmuls DATA-PARALLEL over both DP axes (`replicate × dp`): the
    # d_in/d_out-space activation `x` stays batch-on-`DATA_AXES`, feature-replicated, and the
    # component-space activation `x@V` stays batch-on-`DATA_AXES`, C-on-`tp`. The V/U masters
    # are sharded over `DATA_AXES` (ZeRO-1 ÷N); `_reconstruct_zero1_replicate` gathers only the
    # cross-node `replicate` leg once/step in ENTRY (the ÷fsdp weight stays resident, small),
    # so inside this matmul the compute weight is still `dp`(fsdp)-SHARDED — and this activation
    # pin forces the per-layer `dp` gather to happen HERE, one layer at a time over NVLink
    # (transient, freed each scan iteration; never the full V/U stack). The weight-grad then
    # reduce-scatters back over `DATA_AXES`. WITHOUT pinning `x`, the weight-grad backward is free to instead
    # shard `x`'s feature dim and REPLICATE the global batch (the forward gathers V, the
    # backward does not), which GSPMD can't reshard cheaply -> involuntary full
    # rematerialization -> OOM at tp>1. Pinning the activations (not V/U) keeps the weights as
    # plain matmul args. Guarded so it's a no-op off-mesh (CPU tests / single device);
    # `run.py` sets the global mesh. waist is `[*leading, d]`, leading = (batch, *position):
    # pin batch->`DATA_AXES` (positions + feature replicated for `x`; C-on-`tp` for `x@V`).
    on_mesh = not jax.sharding.get_abstract_mesh().empty
    if on_mesh:
        x = jax.lax.with_sharding_constraint(x, P(DATA_AXES, *(None,) * (x.ndim - 1)))
    xV = x @ V
    if on_mesh:
        xV = jax.lax.with_sharding_constraint(xV, P(DATA_AXES, *(None,) * (xV.ndim - 2), "tp"))
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
