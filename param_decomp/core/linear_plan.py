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
from typing import Literal

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
    Threefry is counter-based, so the sharded draw is value-identical. The draw
    is NOT dtype-invariant: bf16 and fp32 draws from one key are unrelated samples (16 vs
    32 random bits consumed), so a cross-dtype parity check must pass `dtype` explicitly
    and cast."""
    shape = reference.shape[:-1] if drop_last_axis else reference.shape
    draw_dtype = reference.dtype if dtype is None else dtype
    if value_mesh(reference).empty:
        return jax.random.uniform(key, shape, dtype=draw_dtype)
    spec = jax.typeof(reference).sharding.spec.partitions
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
    return jax.sharding.reshard(value, P(*spec.partitions))


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


BlockContraction = Literal["fused_output", "fused_input"]
"""How a block-factored site (`components.BlockedFactorization`) meets its boundary
activations. The target declares this value where it builds the site's linears, because
only the target knows which of its sites is which; the factorization deliberately does
not carry it.

- `"fused_output"`: the site input is one shared vector that every block's V reads in
  full, and the site output concatenates the per-block outputs. This is qwen36_moe's
  gate/up orientation. V computes `...i,eij->...ej` and U computes `...ej,ejk->...ek`.
- `"fused_input"`: the site input concatenates the per-block inputs, and the site
  output is one shared vector that sums the blocks' outputs. This is qwen36_moe's down
  orientation. V computes `...ei,eij->...ej` and U computes `...ej,ejk->...k`.
"""


def block_einsum(contraction: BlockContraction, factor: Literal["V", "U"]) -> str:
    """The einsum of one block-factored linear. Every (contraction, factor) combination
    names a real linear; the four strings are the ones `BlockContraction` spells out."""
    match contraction, factor:
        case ("fused_output", "V"):
            return "...i,eij->...ej"
        case ("fused_output", "U"):
            return "...ej,ejk->...ek"
        case ("fused_input", "V"):
            return "...ei,eij->...ej"
        case ("fused_input", "U"):
            return "...ej,ejk->...k"


@dataclass(frozen=True)
class BlockedLinearPlan:
    """The block-factored sibling of `LinearPlan`. It plans one linear whose weight is a
    rank-3 table of blocks `[block, a, b]` — one site's slice of the 4-D persistence
    stack. Like `LinearPlan`, the plan is a layout contract: it says where the
    activations and the weight live. The einsum itself follows from `contraction` and
    `factor` together (`block_einsum`); every combination of the two names a real
    linear. A selected execution, which gathers `[jobs, ...]` work items instead of
    computing densely over every block, consumes the same weights through its own
    executor and plan. `weight_reduced` means what it means on `LinearPlan`: the mesh
    axes the resident weight was gathered over when it was materialized from the
    persistent masters."""

    mesh: Mesh
    contraction: BlockContraction
    factor: Literal["V", "U"]
    input: P
    operand_input: P
    resident_weight: P
    operand: P
    output: P
    weight_reduced: frozenset[str]

    def __post_init__(self) -> None:
        assert len(self.resident_weight) == len(self.operand) == 3
        assert len(self.input) == len(self.operand_input)
        for source, required in zip(self.input, self.operand_input, strict=True):
            assert set(_axes(required)) <= set(_axes(source)), (source, required)
        for resident, operand in zip(self.resident_weight, self.operand, strict=True):
            assert set(_axes(operand)) <= set(_axes(resident)), (resident, operand)
        assert not self.weight_reduced & spec_axes(self.resident_weight), (
            self.weight_reduced,
            self.resident_weight,
        )


def _planned_contraction(
    x: Array,
    weight: Array,
    plan: "LinearPlan | BlockedLinearPlan",
    einsum: str,
    stack_prefix: tuple[None, ...],
) -> Array:
    """Execute one planned linear: reshard both operands to their declared rows, then
    contract with the output typed to the public output row. `stack_prefix` leads the
    weight and output specs with the unsharded axes of a weight stack applied whole.

    A provenance-carrying weight gets the chained-reduced typing: provenance plus the
    axes this gather drops — together exactly the batch contraction's axis set, which
    is what the dW transpose demands. A provenance-FREE (frozen) weight stays untagged:
    a partial tag would refuse its (rare) dW, and there is no master boundary to defer
    its reduction to anyway."""
    operand_sharding = NamedSharding(
        plan.mesh, P(*stack_prefix, *plan.operand, reduced=_operand_reduced(weight, plan))
    )
    x_operand = jax.sharding.reshard(x, NamedSharding(plan.mesh, plan.operand_input))
    weight_operand = jax.sharding.reshard(weight, operand_sharding)
    return jnp.einsum(
        einsum,
        x_operand,
        weight_operand,
        out_sharding=NamedSharding(plan.mesh, P(*stack_prefix, *plan.output)),
    )


def _operand_reduced(weight: Array, plan: "LinearPlan | BlockedLinearPlan") -> frozenset[str]:
    carried = frozenset(jax.typeof(weight).sharding.spec.reduced)
    assert carried <= plan.weight_reduced, (carried, plan.weight_reduced)
    if not plan.weight_reduced:
        return frozenset()
    return plan.weight_reduced | (spec_axes(plan.resident_weight) - spec_axes(plan.operand))


def placed_linear(x: Array, weight: Array, plan: LinearPlan) -> Array:
    assert weight.ndim == 2, weight.shape
    return _planned_contraction(x, weight, plan, "...i,ij->...j", ())


def stacked_placed_linear(x: Array, weights: Array, plan: LinearPlan) -> Array:
    """Apply every matrix of a stack, whole on its leading axis, to one input; the outputs
    lead with the stack axis. `plan` places one matrix of the stack."""
    assert weights.ndim == 3, weights.shape
    return _planned_contraction(x, weights, plan, "...i,sij->s...j", (None,))


def blocked_placed_linear(x: Array, weight: Array, plan: BlockedLinearPlan) -> Array:
    assert weight.ndim == 3, weight.shape
    return _planned_contraction(x, weight, plan, block_einsum(plan.contraction, plan.factor), ())
