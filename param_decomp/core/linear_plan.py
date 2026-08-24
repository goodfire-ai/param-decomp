"""Typed execution plans for placed linears under the Explicit mesh.

A `LinearPlan` declares one linear's layout contract: the public input/output
activation specs, the weight's resident (stored) spec, and the operand specs both
sides contract at. `placed_linear` executes it as typed reshards + one einsum with a
typed output — the collectives are whatever those transitions require, derived by
JAX's explicit-sharding rules rather than spelled as manual collectives.

Mesh axes the operand DROPS from the weight's resident coverage are typed
`reduced`: the gather's backward reduction is thereby DEFERRED — cotangents flow
unreduced through the layer scan and reduce ONCE where the resident itself was
materialized from the masters (the chained-reduced spelling; one reduce-scatter at
exit, zero cross-replicate collectives inside the loop).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array


def _axes(entry: object) -> tuple[str, ...]:
    match entry:
        case None:
            return ()
        case str():
            return (entry,)
        case tuple():
            assert all(isinstance(axis, str) for axis in entry), entry
            return entry
        case _:
            raise AssertionError(entry)


def spec_axes(spec: P) -> frozenset[str]:
    return frozenset(axis for entry in spec for axis in _axes(entry))


@dataclass(frozen=True)
class LinearPlan:
    """`weight_reduced` is the weight's MASTER PROVENANCE: the mesh axes its resident
    was gathered over when materialized from the persistent masters (empty for frozen
    weights, which have no master to defer a reduction to). The reduced typing does not
    survive a jit boundary on concrete arrays, so the plan carries it and
    `placed_linear` re-tags — a pure typing move when the value is already replicated
    over those axes."""

    mesh: Mesh
    input: P
    operand_input: P
    resident_weight: P
    operand: P
    output: P
    weight_reduced: frozenset[str]

    def __post_init__(self) -> None:
        assert len(self.resident_weight) == len(self.operand) == 2
        assert len(self.input) == len(self.operand_input) == len(self.output)
        for source, required in zip(self.input, self.operand_input, strict=True):
            assert set(_axes(required)) <= set(_axes(source)), (source, required)
        for resident, operand in zip(self.resident_weight, self.operand, strict=True):
            assert set(_axes(operand)) <= set(_axes(resident)), (resident, operand)
        assert not self.weight_reduced & spec_axes(self.resident_weight), (
            self.weight_reduced,
            self.resident_weight,
        )


def value_mesh(value: Array) -> "jax.sharding.AbstractMesh":
    """The mesh the value's TYPE is committed to (empty = untyped/off-mesh). The right
    guard for explicit-sharding arms: the ambient mesh context can be unset while
    values entering a jit still carry axis-typed avals."""
    return jax.typeof(value).sharding.mesh


def uniform_like(
    key: Array,
    reference: Array,
    *,
    drop_last_axis: bool = False,
    dtype: jnp.dtype | None = None,
) -> Array:
    """`uniform(0,1)` in `reference`'s shape (optionally minus the trailing axis), dtype,
    and — when the reference is mesh-typed — its sharding. An untyped draw lowers
    REPLICATED under the Explicit mesh: every rank computes the full-batch threefry bits
    and holds global-shape tensors (measured +22.5 GB/rank at the 32L production shape).
    Threefry is counter-based, so the sharded draw is value-identical (SPEC D4)."""
    shape = reference.shape[:-1] if drop_last_axis else reference.shape
    draw_dtype = reference.dtype if dtype is None else dtype
    if value_mesh(reference).empty:
        return jax.random.uniform(key, shape, dtype=draw_dtype)
    spec = jax.typeof(reference).sharding.spec
    out_spec = P(*(spec[:-1] if drop_last_axis else spec))
    return jax.random.uniform(
        key,
        shape,
        dtype=draw_dtype,
        out_sharding=NamedSharding(value_mesh(reference), out_spec),
    )


def unreduce(value: Array) -> Array:
    """Drop a `reduced` typing (the value is already materialized; this only moves the
    deferred backward reduction HERE). Needed before ops jax has no reduced-rule for —
    manual slicing of a resident stack (scan xs slicing has one, `x[i]` does not)."""
    spec = jax.typeof(value).sharding.spec
    if not spec.reduced:
        return value
    return jax.sharding.reshard(value, P(*spec))


def slice_leading(value: Array, lo: int, hi: int) -> Array:
    """`value[lo:hi]` that survives a `reduced` typing: untag, slice, re-tag (typing
    moves only — the deferred reduction stays at the boundary that materialized the
    tag, not here). The full range is the identity — no reshard pair to land in the
    remat policy's saved-residual set."""
    if lo == 0 and hi == value.shape[0]:
        return value
    tag = frozenset(jax.typeof(value).sharding.spec.reduced)
    if not tag:
        return value[lo:hi]
    sliced = unreduce(value)[lo:hi]
    spec = jax.typeof(sliced).sharding.spec
    return jax.sharding.reshard(sliced, P(*spec, reduced=tag))


def placed_linear(x: Array, weight: Array, plan: LinearPlan) -> Array:
    assert weight.ndim == 2, weight.shape
    carried = frozenset(jax.typeof(weight).sharding.spec.reduced)
    assert carried <= plan.weight_reduced, (carried, plan.weight_reduced)
    # A provenance-carrying weight gets the chained-reduced typing: provenance plus the
    # axes this gather drops — together exactly the batch contraction's axis set, which
    # is what the dW transpose demands. A provenance-FREE (frozen) weight stays untagged:
    # a partial tag would refuse its (rare) dW, and there is no master boundary to defer
    # its reduction to anyway.
    if plan.weight_reduced:
        reduced = plan.weight_reduced | (spec_axes(plan.resident_weight) - spec_axes(plan.operand))
        operand_sharding = NamedSharding(plan.mesh, P(*plan.operand, reduced=reduced))
    else:
        operand_sharding = NamedSharding(plan.mesh, plan.operand)
    x_operand = jax.sharding.reshard(x, NamedSharding(plan.mesh, plan.operand_input))
    weight_operand = jax.sharding.reshard(weight, operand_sharding)
    return jnp.einsum(
        "...i,ij->...j",
        x_operand,
        weight_operand,
        out_sharding=NamedSharding(plan.mesh, plan.output),
    )
