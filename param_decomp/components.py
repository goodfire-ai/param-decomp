"""The decomposition representation, shared by every target (LM and toy alike).

`SiteC` / `SiteSpec` are the per-site shape primitives (config-level name+C, and the
shape-carrying spec); `ComponentStacks` is the trainable master pytree, persisted as same-shape
STACKS (owner-partitioned layout); `init_component_stacks` seeds it; `site_out` is the one
decomposed-linear primitive (`((x@V)*m)@U + (x@Δ)*d`). These are domain-neutral
— they depend only on the site shapes and the V/U/W arrays — so they live here rather than
inside `model.py` (whose `DecomposedModel` Protocol references `ComponentStacks`/`SiteSpec`) or any
one target.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from typing import ClassVar, Generic, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

# typing_extensions for `TypeVar(default=...)` (PEP 696): stdlib-supported only from 3.13,
# and this package supports 3.12.
from typing_extensions import TypeVar


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


_FP8_E4M3_MAX = 448.0  # largest finite magnitude of float8_e4m3fn


def quantize_fp8(w: Array) -> tuple[Array, Array]:
    """Per-LEADING-ROW symmetric fp8 (e4m3fn) quantization for the Quantized All-Gather (QAG):
    the weight is gathered as fp8 (½ the bf16 bytes), then dequantized for the bf16 matmul.
    The scale reduces over every axis EXCEPT axis 0 (the stacked `n_layer` / `n_chunk` scan
    axis), keepdims — so it (a) carries that scan axis like the weight does (a per-tensor
    scalar would have no leading axis to `scan`-slice → IndexError) and (b) gives per-row
    quantization (tighter than one global scale). Returns `(qvalue: float8_e4m3fn,
    scale: f32 [L,1,…])` with `w ≈ qvalue.astype(bf16) * scale`. Computed BEFORE the gather
    (scale rides along, survives it). amax==0 rows floored so the divide is finite."""
    axes = tuple(range(1, w.ndim))
    scale = (
        jnp.maximum(jnp.max(jnp.abs(w), axis=axes, keepdims=True), jnp.float32(1e-12))
        / _FP8_E4M3_MAX
    )
    q = (w / scale).astype(jnp.float8_e4m3fn)
    return q, scale.astype(jnp.float32)


def dequantize_fp8(q: Array, scale: Array) -> Array:
    """Inverse of `quantize_fp8` to bf16 (the compute dtype). Done AFTER the all-gather so the
    gather moves fp8 bytes."""
    return q.astype(jnp.bfloat16) * scale.astype(jnp.bfloat16)


VUShape = tuple[int, int, int]  # (d_in, d_out, C)

# How `init_component_stacks` seeds V/U; the config schema (`PDConfig.weight_init`) reuses it.
WeightInit = Literal["kaiming", "coupled"]

# site name -> (shape group, slot on the group's stack axis); static, canonical site order
SiteSlots = tuple[tuple[str, VUShape, int], ...]

# The V/U leaf type: `Array` for the real fp32 masters (the default — so bare `ComponentStacks`
# means `ComponentStacks[Array]` and no call site needs the parameter), or `NamedSharding` for
# the same-structure placement tree `placement.component_stacks_shardings` returns for
# `jax.jit(out_shardings=...)`.
VULeaf = TypeVar("VULeaf", default=Array)


def vu_shape_groups(sites: tuple[SiteSpec, ...]) -> dict[VUShape, tuple[SiteSpec, ...]]:
    """Sites grouped by V/U shape, site order preserved within a group."""
    groups: dict[VUShape, list[SiteSpec]] = {}
    for spec in sites:
        groups.setdefault((spec.d_in, spec.d_out, spec.C), []).append(spec)
    return {shape: tuple(specs) for shape, specs in groups.items()}


def site_slots_for(sites: tuple[SiteSpec, ...]) -> SiteSlots:
    """The canonical site→(shape, slot) mapping: slots follow site order within each shape
    group (`vu_shape_groups` preserves it), entries follow overall site order."""
    by_name: dict[str, tuple[VUShape, int]] = {}
    for shape, specs in vu_shape_groups(sites).items():
        for slot, spec in enumerate(specs):
            by_name[spec.name] = (shape, slot)
    return tuple((spec.name, *by_name[spec.name]) for spec in sites)


@cache
def _slot_index(site_slots: SiteSlots) -> dict[str, tuple[VUShape, int]]:
    return {name: (shape, slot) for name, shape, slot in site_slots}


class ComponentStacks(eqx.Module, Generic[VULeaf]):
    """The trainable V/U masters, persisted as same-shape STACKS — the owner-partitioned
    layout: one `(Vs [g, d_in, C], Us [g, C, d_out])` pair per `(d_in, d_out, C)` shape
    group, with `site_slots` (static) mapping each site name to its slot on the stack axis.

    Why stacks: per-matrix ownership is only expressible under SPMD by sharding a STACK
    axis (a per-site leaf cannot live wholly on one rank), the llama8b scan consumes
    per-kind stacks natively, and the checkpoint/init trees shrink from `2·n_sites` leaves
    to `2·n_shapes`. Per-site access is `site(name)` — a static slice of the stack.

    Leaves are fp32 master Arrays (`ComponentStacks[Array]`) or `NamedSharding`s in the
    same-structure placement tree `placement.component_stacks_shardings` returns
    (`ComponentStacks[NamedSharding]`). This module is placement-FREE: the per-group row
    lookup and its boundary validation live in `placement.py`, above."""

    stacks: dict[VUShape, tuple[VULeaf, VULeaf]]
    site_slots: SiteSlots = eqx.field(static=True)

    def site(self: "ComponentStacks[Array]", name: str) -> tuple[Array, Array]:
        shape, slot = _slot_index(self.site_slots)[name]
        Vs, Us = self.stacks[shape]
        return Vs[slot], Us[slot]

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.site_slots)

    def sites_items(self: "ComponentStacks[Array]") -> Iterator[tuple[str, tuple[Array, Array]]]:
        """`(name, (V, U))` in canonical site order — the per-site view of the stacks."""
        for name, _, _ in self.site_slots:
            yield name, self.site(name)

    V_AXES: "ClassVar[tuple[str, ...]]" = ("stack", "d_in", "C")
    U_AXES: "ClassVar[tuple[str, ...]]" = ("stack", "C", "d_out")

    def group_lengths(self) -> dict[VUShape, int]:
        """Stack length per shape group, from the STATIC index (works on eval-shape
        trees) — what the placement boundary validates the received row assignment
        against (`placement.component_stacks_shardings` / `_audit`)."""
        lengths: dict[VUShape, int] = {}
        for _name, shape, slot in self.site_slots:
            lengths[shape] = max(lengths.get(shape, 0), slot + 1)
        return lengths


def _kaiming_stack_arrays(
    sites: tuple[SiteSpec, ...], key: Array
) -> dict[VUShape, tuple[Array, Array]]:
    """Seeded stack init: `{(d_in, d_out, C): (V [g, d_in, C], U [g, C, d_out])}`, vmapped
    over per-site keys drawn in site order — each site's slice is BIT-IDENTICAL to the
    retired per-site init's draw (pinned by `test_sharding`). Under jit the graph has
    2×n_shapes sharded outputs instead of 2×n_sites, which keeps the init compile from
    scaling with site count."""
    keys = jax.random.split(key, 2 * len(sites))
    site_index = {spec.name: idx for idx, spec in enumerate(sites)}
    stacked: dict[VUShape, tuple[Array, Array]] = {}
    for (d_in, d_out, c), specs in vu_shape_groups(sites).items():
        idxs = jnp.array([site_index[spec.name] for spec in specs])
        Vs = jax.vmap(lambda k, s=(d_in, c): jax.random.normal(k, s))(keys[2 * idxs])
        Us = jax.vmap(lambda k, s=(c, d_out): jax.random.normal(k, s))(keys[2 * idxs + 1])
        stacked[(d_in, d_out, c)] = (Vs * d_in**-0.5, Us * c**-0.5)
    return stacked


def _coupled_stack_arrays(
    sites: tuple[SiteSpec, ...], target_weights: dict[str, Array], key: Array
) -> dict[VUShape, tuple[Array, Array]]:
    """Coupled stack init: unit-norm seed on the narrow side, wide side its raw W-image.

    `d_in <= d_out`: `v_c ~ unit norm`, `U_c <- (W v_c)^T`; else `u_c ~ unit norm`,
    `V_c <- W^T u_c`. No C-dependent rescale — components sit at W's natural scale and
    the component sum equals W restricted to the seed span (`W V V^T` when V is seeded),
    the delta carrying the complement. One key per site, split in site order; vmapped
    per shape group like `_kaiming_stack_arrays` so the compiled init doesn't scale with
    site count."""
    keys = jax.random.split(key, len(sites))
    site_index = {spec.name: idx for idx, spec in enumerate(sites)}
    stacked: dict[VUShape, tuple[Array, Array]] = {}
    for (d_in, d_out, c), specs in vu_shape_groups(sites).items():
        ws = jnp.stack([target_weights[spec.name] for spec in specs]).astype(jnp.float32)
        assert ws.shape == (len(specs), d_out, d_in), (ws.shape, (d_in, d_out, c))
        ks = keys[jnp.array([site_index[spec.name] for spec in specs])]
        if d_in <= d_out:

            def seed_v(k: Array, w: Array, s: tuple[int, int] = (d_in, c)) -> tuple[Array, Array]:
                v = jax.random.normal(k, s)
                v = v / jnp.linalg.norm(v, axis=0, keepdims=True)
                return v, (w @ v).T
        else:

            def seed_v(k: Array, w: Array, s: tuple[int, int] = (c, d_out)) -> tuple[Array, Array]:
                u = jax.random.normal(k, s)
                u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
                return w.T @ u.T, u

        stacked[(d_in, d_out, c)] = jax.vmap(seed_v)(ks, ws)
    return stacked


def component_stacks_from_sites(vu: dict[str, tuple[Array, Array]]) -> ComponentStacks:
    """Build the stacked `ComponentStacks` from a per-site `{name: (V, U)}` dict (site order =
    dict order). The explicit-arrays constructor for toys and tests; the trainer inits
    directly in the stacked layout (`init_component_stacks`)."""
    sites = tuple(
        SiteSpec(name=name, d_in=V.shape[0], d_out=U.shape[1], C=V.shape[1])
        for name, (V, U) in vu.items()
    )
    stacks = {
        shape: (
            jnp.stack([vu[s.name][0] for s in specs]),
            jnp.stack([vu[s.name][1] for s in specs]),
        )
        for shape, specs in vu_shape_groups(sites).items()
    }
    return ComponentStacks(stacks=stacks, site_slots=site_slots_for(sites))


def init_component_stacks(
    sites: tuple[SiteSpec, ...],
    key: Array,
    weight_init: WeightInit = "kaiming",
    target_weights: dict[str, Array] | None = None,
) -> ComponentStacks:
    """Seeded V/U masters in the stacked persistence layout, per `weight_init`:
    'kaiming' — small random fp32 V ~ N(0, d_in^-0.5), U ~ N(0, C^-0.5), ignores W
    (the weight-delta channel carries the faithfulness residual at init); 'coupled' —
    `_coupled_stack_arrays`, seeded from `target_weights` (required there, refused
    otherwise)."""
    match weight_init:
        case "kaiming":
            assert target_weights is None, "kaiming init draws blind — no target_weights"
            stacks = _kaiming_stack_arrays(sites, key)
        case "coupled":
            assert target_weights is not None, "coupled init seeds from W — pass target_weights"
            stacks = _coupled_stack_arrays(sites, target_weights, key)
    return ComponentStacks(stacks=stacks, site_slots=site_slots_for(sites))


def site_out(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
) -> Array:
    """One decomposed linear: `((x@V)*m)@U + (x@Δ)*d`, routed per position
    against the frozen `x @ W.T`. `mask` may be None (fully on); `route` None routes
    everywhere. `delta_mask` None drops the delta path entirely (constant-source entries
    carry no delta). `delta_mask`/`route` broadcast over batch;
    trailing dim added here."""
    # Pin the decomposed matmuls DATA-PARALLEL over the FULL mesh: the d_in/d_out-space
    # activation `x` stays batch-sharded over `('replicate', 'fsdp')`, feature-replicated, and
    # the component-space activation `x@V` stays batch-sharded, C-REPLICATED (no TP axis).
    # This forces the sharded V/U masters to be GATHERED for compute and their grads
    # reduced back to the master layout (symmetric — the intended layout). WITHOUT pinning
    # `x`, the weight-grad backward is free to instead shard `x`'s feature dim and REPLICATE
    # the batch (the forward gathers V, the backward does not), which GSPMD can't reshard
    # cheaply -> involuntary full rematerialization -> OOM. Pinning the activations (not
    # V/U) keeps the weights as plain matmul args. Guarded so it's a no-op off-mesh (CPU
    # tests / single device); `run.py` sets the global mesh. waist is `[*leading, d]`,
    # leading = (batch, *position): pin batch over the full mesh (positions + feature
    # replicated for `x`; C replicated for `x@V`).
    on_mesh = not jax.sharding.get_abstract_mesh().empty
    batch_axes = ("replicate", "fsdp")
    if on_mesh:
        x = jax.lax.with_sharding_constraint(x, P(batch_axes, *(None,) * (x.ndim - 1)))
    xV = x @ V
    if on_mesh:
        # batch over the data axes; the component C axis (last) on `tp` (Megatron-C) — so the
        # `x@V` weight is ÷tp and the `xV * mask` stays local (mask is C-on-tp too).
        xV = jax.lax.with_sharding_constraint(xV, P(batch_axes, *(None,) * (xV.ndim - 2), "tp"))
    acts = xV * mask if mask is not None else xV
    out = acts @ U
    if on_mesh:
        # `acts @ U` contracts the tp-sharded C → reduce over `tp`; pin the output d-full /
        # batch-sharded so the tp-reduce + fsdp-gather are symmetric and the weight-grad
        # backward reshards the same way (avoids the involuntary-remat OOM).
        out = jax.lax.with_sharding_constraint(out, P(batch_axes, *(None,) * (out.ndim - 1)))
    if delta_mask is not None:
        # `(x @ Δ.T)` for `Δ = W − (V@U).T`, expanded to activation space as
        # `x@W.T − (x@V)@U` so the `[d_out, d_in]` weight delta is NEVER formed. Under
        # FSDP that delta would mix V's sharded d_in with U's sharded d_out (two dims
        # demanding the data axis) and force a replicate-then-repartition reshard; the
        # activation-space form is all activation×weight matmuls that shard cleanly.
        # (Still a bf16-rounding DIVERGENCE vs the fp32 oracle delta — accepted; the
        # faithfulness loss uses the fp32 `weight_deltas`, not this path.)
        out = out + delta_mask[..., None] * (x @ W.T - xV @ U)
    if route is not None:
        out = jnp.where(route[..., None], out, x @ W.T)
    return out
