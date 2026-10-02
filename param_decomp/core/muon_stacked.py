"""Muon: optax.contrib.muon semantics with the Newton-Schulz orthogonalization batched per
semantic kind and sharded on the stack axis.

Why: GSPMD lowers per-leaf NS on ÷N-sharded fp32 masters into per-iteration full-Gram
all-reduces with the largest matmul replicated on every device — serialized collectives
that dominate the optimizer step.
Each muon leaf is one kind's `[g, rows, cols]` stack; sharding the STACK axis makes each
NS device-local: a short hop chain in, its reverse out, zero per-iteration collectives
(the Kimi "Muon is Scalable" parameter-partitioned recipe, expressed GSPMD-natively).

Why the entry/exit comms are hand-pinned rather than left to declarative lowering: under
the Explicit mesh the naive per-leaf spelling does not even trace — NS's Gram matmul puts
the sharded matrix axis on BOTH result dims (`ShardingTypeError`); and in Auto mode the
partitioner re-gathers the sharded axis and all-reduces Gram partials INSIDE the NS
while-loop every iteration — it never hoists to reshard-once/compute-local/reshard-back.

The staging is DECLARED, not derived: each muon leaf's `waypoints` sharding is its
placement table's `ns_compute` row verbatim (`placement.ns_staging_sharding`; a kind
whose stack does not tile the declared split refuses at the stacked-muon consumer's
claim — `placement.assert_stacked_muon_component_staging` for V/U,
`ci_fn.optimizer.assert_ci_muon_staging_tiles` for CI matrices — nothing is ever padded or gathered
whole). NS executes AT the waypoint, reached by the
`staging_hops` chain — one mesh axis moved per reshard, because a single reshard that
both moves axes between dims and gathers others trips the SPMD
involuntary-full-rematerialization fallback (which replicates the whole tensor). The
row language shards the stack axis only: a matrix-axis-carrying waypoint — the persist
row verbatim included — is refused at table build, for that same fallback (pinned by
the fd-2-warning tests in param_decomp/tests/core/test_optim_torch_parity.py) and because a
matrix-sharded NS operand is an explicit-mode type error on the Gram contraction.

Structure mirrors `optax.contrib.muon` exactly — same `MuonState(count, mu, ns_coeffs)`,
same muon/adam partition, same chain (NS -> consistent-rms shape scale -> weight decay ->
lr). Only the NS call differs: each leaf is canonicalized to `[g, rows<=cols]`, cast to
`ns_dtype`, staged at its waypoint, orthogonalized, and landed back on its own layout. The
vmap batching reorders float ops, so trajectories match per-leaf `optax.contrib.muon` (the
reference the parity tests build directly) only up to reassociation — same tolerance
class as device-count invariance.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

# Not re-exported from `optax.contrib` in the pinned 0.2.8; the venv is pinned, so the
# private import is stable. `muon()` itself builds from these same pieces.
from optax.contrib._muon import orthogonalize_via_newton_schulz, scale_by_shape

from param_decomp.core.axes import MeshAxis

_NS_DIMS = optax.contrib.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1)
_2D_DIMS = optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=1)


@dataclass(frozen=True)
class StageInPlace:
    """Newton–Schulz runs where the matrix already rests; no reshard."""


NSWaypoint = NamedSharding | StageInPlace

NSWaypoints = Callable[[optax.Updates], Any]
"""Maps the muon update tree to a same-structure tree of per-leaf `NSWaypoint`s: a
staging sharding (an `ns_compute` placement row at the canonical `[g, rows<=cols]` rank,
`placement.ns_staging_sharding`) or `StageInPlace`."""


def _canonicalize(leaf: Array) -> tuple[Array, bool]:
    """View a muon leaf as a `[g, a, b]` stack with `a <= b`: 2D leaves get `g=1`, and a
    4D `[stack, expert, rows, cols]` leaf folds its two leading axes into one batch axis
    (a pure reshape, entirely within one semantic kind — never a cross-kind grouping).
    NS orthogonalization and the `consistent_rms`/`sqrt(max(fan_in, fan_out))` scalings
    are transpose-symmetric, so orientation only affects the canonical view, not
    semantics. `_decanonicalize` inverts the view."""
    match leaf.ndim:
        case 2:
            stacked = leaf[None]
        case 3:
            stacked = leaf
        case 4:
            stacked = leaf.reshape(leaf.shape[0] * leaf.shape[1], *leaf.shape[2:])
        case _:
            raise AssertionError(
                f"stacked NS handles 2D matrices, 3D [stack, rows, cols] stacks, and 4D"
                f" [stack, expert, rows, cols] stacks, got shape {leaf.shape} — extend"
                f" `_canonicalize` for this layout"
            )
    transposed = stacked.shape[-2] > stacked.shape[-1]
    return (jnp.swapaxes(stacked, -2, -1) if transposed else stacked), transposed


def _decanonicalize(canonical: Array, leaf: Array, transposed: bool) -> Array:
    """Land a canonical `[g, a, b]` result back on `leaf`'s own rank and orientation."""
    out = jnp.swapaxes(canonical, -2, -1) if transposed else canonical
    match leaf.ndim:
        case 2:
            return out[0]
        case 3:
            return out
        case 4:
            return out.reshape(leaf.shape)
        case _:
            raise AssertionError(leaf.shape)


def _spec_entry_axes(entry: object) -> tuple[MeshAxis, ...]:
    if entry is None:
        return ()
    if isinstance(entry, str):
        return (entry,)  # pyright: ignore[reportReturnType]
    return tuple(entry)  # pyright: ignore[reportArgumentType]


def staging_hops(source: P, target_stack: tuple[MeshAxis, ...]) -> list[P]:
    """The staging path from a canonical `[g, rows, cols]` master layout to its NS-legal
    waypoint (`P(target_stack, None, None)`), one mesh axis moved per reshard.

    A single reshard that simultaneously moves axes between dims and gathers others is
    exactly what trips GSPMD's involuntary-full-rematerialization fallback (it replicates
    the whole tensor as its last resort — the fd-2-warning tests pin absence); every
    single-axis move and the one trailing matrix-dim gather lower shard-to-shard. Hops:
    per target stack axis in declared order, move it from whichever matrix dim holds it
    (an axis absent from the source is a comms-free split); then one final reshard drops
    every remaining matrix-dim (and surplus stack) axis."""
    stack = list(_spec_entry_axes(source[0]))
    rows = list(_spec_entry_axes(source[1]))
    cols = list(_spec_entry_axes(source[2]))
    hops: list[P] = []
    for axis in target_stack:
        if axis in stack:
            continue
        if axis in rows:
            rows.remove(axis)
        elif axis in cols:
            cols.remove(axis)
        stack.append(axis)
        hops.append(P(tuple(stack), tuple(rows) or None, tuple(cols) or None))
    final = P(target_stack or None, None, None)
    if not hops or hops[-1] != final:
        hops.append(final)
    return hops


def _declares_executed_convention(spec: optax.contrib.MuonDimensionNumbers, ndim: int) -> bool:
    """True iff the declared axes normalize (optax's `ax % ndim`) to exactly what the
    kernel hardcodes: reduction=ndim-2, output=ndim-1, on a 2D matrix or 3D [stack, a, b]
    stack. Orientation is checked too: with `consistent_rms=None` optax honors the
    declared orientation through the directional width scaling `sqrt(max(1, fan_out /
    fan_in))`, which the `scale_by_shape` wiring here does not."""
    if ndim not in (2, 3, 4):
        return False

    def normalized_single_axis(axis: Sequence[int] | int) -> int | None:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        return axes[0] % ndim if len(axes) == 1 else None

    declared = (
        normalized_single_axis(spec.reduction_axis),
        normalized_single_axis(spec.output_axis),
    )
    return declared == (ndim - 2, ndim - 1)


def _muon_mask_from_validated_dim_numbers(
    resolved_dim_numbers: optax.Params, params: optax.Params
) -> optax.Params:
    """Muon/adam labels for `optax.partition`, refusing any muon-labeled leaf whose
    declared axes differ from the convention the kernel executes. `_canonicalize` and
    the `scale_by_shape` wiring DISCARD the declaration (hardcoded trailing-two axes),
    while `optax.contrib.muon` — the reference semantics — honors it, so a
    nonconforming declaration would silently orthogonalize different axes than the
    reference; it dies here at optimizer build instead."""

    def label(
        spec_path_for_err: tuple[Any, ...], spec: object, subtree: optax.Params
    ) -> optax.Params:
        assert spec is None or isinstance(spec, optax.contrib.MuonDimensionNumbers), spec
        if spec is None:
            return jax.tree.map(lambda _: "adam", subtree)
        for leaf_path, leaf in jax.tree_util.tree_flatten_with_path(subtree)[0]:
            assert _declares_executed_convention(spec, leaf.ndim), (
                f"muon leaf {jax.tree_util.keystr(spec_path_for_err + leaf_path)} with shape"
                f" {leaf.shape} declares MuonDimensionNumbers(reduction_axis="
                f"{spec.reduction_axis}, output_axis={spec.output_axis}), but the"
                f" stacked NS executes hardcoded trailing-two matrix axes"
                f" (reduction=-2, output=-1; 3D = [stack, rows, cols], 4D ="
                f" [stack, expert, rows, cols]) and discards the"
                f" declaration — extend `_canonicalize` and the `scale_by_shape` wiring"
                f" to honor declared axes"
            )
        return jax.tree.map(lambda _: "muon", subtree)

    return jax.tree_util.tree_map_with_path(
        label,
        resolved_dim_numbers,
        params,
        is_leaf=lambda x: x is None or isinstance(x, optax.contrib.MuonDimensionNumbers),
    )


def _orthogonalize_at_waypoint(
    leaf: Array,
    waypoint: NamedSharding,
    ns_dtype: jnp.dtype,
    orthogonalize: Callable[[Array], Array],
) -> Array:
    staged_leaf = leaf.astype(ns_dtype)
    if staged_leaf.ndim == 4:
        # A 4D [stack, expert, rows, cols] leaf merges its leading two axes in
        # `_canonicalize`, and a merge whose SECOND axis is sharded (the
        # `*-replicated-resident-moe` presets' `expert->tp`) has no shard-preserving
        # view — an explicit-sharding refusal. One single-axis-move reshard (the hop
        # discipline, in ns_dtype bytes) clears the block axis (`expert`) first, by
        # where its mesh axes belong at the waypoint: staging axes park on the leading
        # stack axis (the merge is then a pure view and they ride the canonical
        # stack); non-staging axes (owner's co-located `expert: tp`) GATHER — the
        # intra-node whole-block move owner staging accepts — leaving the stack axes
        # in place, so the canonical leaf typically sits at the waypoint already.
        entry0, entry1, *matrix = (
            *jax.typeof(staged_leaf).sharding.spec,
            *(None,) * (4 - len(jax.typeof(staged_leaf).sharding.spec)),
        )
        if _spec_entry_axes(entry1):
            if set(_spec_entry_axes(entry1)) <= set(_spec_entry_axes(waypoint.spec[0])):
                merged_stack = (*_spec_entry_axes(entry0), *_spec_entry_axes(entry1))
                staged_leaf = jax.sharding.reshard(
                    staged_leaf,
                    NamedSharding(waypoint.mesh, P(merged_stack, None, *matrix)),
                )
            else:
                staged_leaf = jax.sharding.reshard(
                    staged_leaf,
                    NamedSharding(waypoint.mesh, P(entry0, None, *matrix)),
                )
    stacked, transposed = _canonicalize(staged_leaf)
    entries = tuple(jax.typeof(stacked).sharding.spec)
    source = P(*entries, *(None,) * (3 - len(entries)))
    hops = staging_hops(source, _spec_entry_axes(waypoint.spec[0]))
    for spec in hops:
        stacked = jax.sharding.reshard(stacked, NamedSharding(waypoint.mesh, spec))
    orthogonalized = orthogonalize(stacked)
    # The reversed hop chain lands back on the master canonical layout with the
    # collectives still moving ns_dtype bytes; the cast to the leaf dtype is then
    # layout-local.
    for spec in reversed([source, *hops[:-1]]):
        orthogonalized = jax.sharding.reshard(orthogonalized, NamedSharding(waypoint.mesh, spec))
    orthogonalized = _decanonicalize(orthogonalized.astype(leaf.dtype), leaf, transposed)
    return jax.sharding.reshard(
        orthogonalized, NamedSharding(waypoint.mesh, jax.typeof(leaf).sharding.spec)
    )


def _staged_newton_schulz(
    mu_hat: optax.Updates,
    ns_coeffs: Array,
    ns_steps: int,
    ns_dtype: jnp.dtype,
    waypoints: NSWaypoints | None,
) -> optax.Updates:
    """Orthogonalize every leaf of `mu_hat` (all muon-labeled by the partition) via ONE
    batched NS per leaf — each leaf IS one semantic kind's stack, so no cross-kind
    grouping, concatenation, or padding exists.

    Placed execution (`waypoints` given) is one generic choreography per leaf: cast to
    `ns_dtype` and stage to the leaf's `ns_compute` waypoint via the single-axis-move
    hop chain (`staging_hops`; the cast sits on the master layout, so the staging
    collectives move ns_dtype bytes and never ride an unpinned layout transition — task
    570); NS executes AT the waypoint — whole matrices per device, which is what keeps
    the NS loop collective-free (a matrix-sharded operand is an explicit-mode type error
    on the Gram contraction); egress reverses the hop chain and lands the update on the
    leaf's own layout (the optimizer add demands exact sharding agreement under the
    Explicit mesh). A `StageInPlace` leaf, and every leaf when `waypoints=None` (no mesh:
    the CPU trajectory tests), is the same math with no reshards."""
    leaves, treedef = jax.tree.flatten(mu_hat)
    per_leaf_waypoints: list[NSWaypoint] = (
        [StageInPlace()] * len(leaves)
        if waypoints is None
        else jax.tree.flatten(waypoints(mu_hat))[0]
    )
    assert len(per_leaf_waypoints) == len(leaves), (
        f"ns_compute waypoints tree yields {len(per_leaf_waypoints)} leaves for"
        f" {len(leaves)} muon leaves — the waypoints callable must mirror the update"
        f" tree's structure"
    )

    def orthogonalize(stacked: Array) -> Array:
        return orthogonalize_via_newton_schulz(
            stacked,
            jnp.asarray(ns_coeffs, ns_dtype),
            ns_steps=ns_steps,
            dimension_numbers=_NS_DIMS,
        )

    out: list[Array] = []
    for leaf, waypoint in zip(leaves, per_leaf_waypoints, strict=True):
        match waypoint:
            case StageInPlace():
                stacked, transposed = _canonicalize(leaf.astype(ns_dtype))
                orthogonalized = orthogonalize(stacked).astype(leaf.dtype)
                out.append(_decanonicalize(orthogonalized, leaf, transposed))
            case NamedSharding():
                out.append(_orthogonalize_at_waypoint(leaf, waypoint, ns_dtype, orthogonalize))
    return jax.tree.unflatten(treedef, out)


def scale_by_stacked_muon(
    *,
    beta: float,
    ns_steps: int,
    ns_dtype: jnp.dtype,
    waypoints: NSWaypoints | None,
) -> optax.GradientTransformation:
    """`optax.contrib.scale_by_muon` (nesterov, non-adaptive, frobenius) with the per-leaf
    NS replaced by `_staged_newton_schulz`. State is optax's `MuonState` verbatim."""
    reference = optax.contrib.scale_by_muon(beta=beta, ns_steps=ns_steps, nesterov=True)

    def update_fn(
        updates: optax.Updates, state: optax.OptState, params: optax.Params | None = None
    ) -> tuple[optax.Updates, optax.OptState]:
        del params
        assert isinstance(state, optax.contrib.MuonState)
        mu = optax.tree.update_moment(updates, state.mu, beta, 1)
        count_inc = optax.safe_increment(state.count)
        mu_hat = jax.tree.map(
            lambda m, g: beta * m + (1 - beta) * g,
            optax.tree.bias_correction(mu, beta, optax.safe_increment(count_inc)),
            optax.tree.bias_correction(updates, beta, count_inc),
        )
        orthogonalized = _staged_newton_schulz(
            mu_hat, jnp.asarray(state.ns_coeffs), ns_steps, ns_dtype, waypoints
        )
        return orthogonalized, optax.contrib.MuonState(
            count=count_inc, mu=mu, ns_coeffs=state.ns_coeffs
        )

    return optax.GradientTransformation(reference.init, update_fn)


def stacked_muon(
    learning_rate: optax.ScalarOrSchedule,
    *,
    beta: float,
    weight_decay: float,
    consistent_rms: float | None,
    muon_weight_dimension_numbers: Callable[[optax.Params], optax.Params] | None,
    ns_steps: int,
    ns_dtype: jnp.dtype,
    waypoints: NSWaypoints | None,
) -> optax.GradientTransformation:
    """`optax.contrib.muon(...)`'s semantics at our call site (`run_state`): same muon/adam
    leaf partition, same post-NS chain, same state pytree — NS runs stacked per kind, its
    staging declared by `waypoints` (None => no mesh, no reshards — the CPU trajectory
    tests; every run stages at its placement rows). Muon-labeled leaves must declare
    exactly the trailing-two convention the kernel executes; anything else dies at
    optimizer build (`_muon_mask_from_validated_dim_numbers`)."""
    dim_nums = muon_weight_dimension_numbers
    if dim_nums is None:
        dim_nums = lambda params: jax.tree.map(lambda x: _NS_DIMS if x.ndim == 2 else None, params)

    return optax.partition(
        transforms={
            "muon": optax.chain(
                scale_by_stacked_muon(
                    beta=beta,
                    ns_steps=ns_steps,
                    ns_dtype=ns_dtype,
                    waypoints=waypoints,
                ),
                scale_by_shape(
                    weight_dimension_numbers=lambda updates: jax.tree.map(
                        lambda x: _2D_DIMS if x.ndim == 2 else _NS_DIMS, updates
                    ),
                    consistent_rms=consistent_rms,
                ),
                optax.add_decayed_weights(weight_decay),
                optax.scale_by_learning_rate(learning_rate),
            ),
            # Matches optax muon's fallback exactly: adamw at the muon lr, muon's
            # nesterov=True threaded through (optax.adamw defaults nesterov=False).
            "adam": optax.adamw(
                learning_rate=learning_rate,
                b1=0.9,
                b2=0.999,
                eps=1e-8,
                weight_decay=0.0,
                nesterov=True,
            ),
        },
        param_labels=lambda params: _muon_mask_from_validated_dim_numbers(dim_nums(params), params),
    )
