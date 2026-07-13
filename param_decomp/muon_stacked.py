"""Stacked-NS Muon: optax.contrib.muon semantics with the Newton-Schulz orthogonalization
batched across same-shape leaves and sharded on the stack axis (SPEC S20, `impl: stacked`).

Why: GSPMD lowers per-leaf NS on ÷N-sharded fp32 masters into per-iteration full-Gram
all-reduces with the largest matmul replicated on every device (~24.6 GB of serialized
collectives per step for the 4L ci-fn group — the measured 3.3x muon-ci step-time hit).
Stacking same-shape matrices and sharding the STACK axis makes each NS device-local:
one reshard in, one out, zero per-iteration collectives (the Kimi "Muon is Scalable"
parameter-partitioned recipe, expressed GSPMD-natively).

Structure mirrors `optax.contrib.muon` exactly — same `MuonState(count, mu, ns_coeffs)`,
same muon/adam partition, same chain (NS -> consistent-rms shape scale -> weight decay ->
lr) — so checkpoints round-trip across `impl: optax|stacked` unchanged. Only the NS call
differs: leaves are canonicalized to `[g, rows<=cols]`, grouped by 2D shape, concatenated,
optionally cast to `ns_dtype`, orthogonalized once per group, and unstacked. Stacking
reorders float ops (concat + shared vmap), so trajectories match `impl: optax` only up to
reassociation — same tolerance class as device-count invariance (SPEC D4).
"""

from collections import defaultdict
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

# Not re-exported from `optax.contrib` in the pinned 0.2.8; the venv is pinned, so the
# private import is stable. `muon()` itself builds from these same pieces.
from optax.contrib._muon import orthogonalize_via_newton_schulz, scale_by_shape

_NS_DIMS = optax.contrib.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1)
_2D_DIMS = optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=1)


def _canonicalize(leaf: Array) -> tuple[Array, bool]:
    """View a muon leaf as a `[g, a, b]` stack with `a <= b` (2D leaves get `g=1`).
    NS orthogonalization and the `consistent_rms`/`sqrt(max(fan_in, fan_out))` scalings
    are transpose-symmetric, so orientation only affects grouping, not semantics."""
    assert leaf.ndim in (2, 3), f"muon leaf must be 2D or a 3D stack, got {leaf.shape}"
    stacked = leaf[None] if leaf.ndim == 2 else leaf
    transposed = stacked.shape[-2] > stacked.shape[-1]
    return (jnp.swapaxes(stacked, -2, -1) if transposed else stacked), transposed


def _pad_to_multiple(stack: Array, multiple: int) -> Array:
    remainder = stack.shape[0] % multiple
    if remainder == 0:
        return stack
    pad = multiple - remainder
    return jnp.concatenate([stack, jnp.zeros((pad, *stack.shape[1:]), stack.dtype)], axis=0)


def _grouped_newton_schulz(
    mu_hat: optax.Updates,
    ns_coeffs: Array,
    ns_steps: int,
    ns_dtype: jnp.dtype,
    stack_sharding: NamedSharding | None,
) -> optax.Updates:
    """Orthogonalize every leaf of `mu_hat` (all muon-labeled by the partition) via ONE
    batched NS per distinct canonical 2D shape, the batch axis sharded per
    `stack_sharding` so each 2D orthogonalization runs device-local."""
    leaves, treedef = jax.tree.flatten(mu_hat)
    canon = [_canonicalize(leaf) for leaf in leaves]
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (stacked, _) in enumerate(canon):
        rows, cols = stacked.shape[-2:]
        groups[(rows, cols)].append(idx)

    n_shards = 1 if stack_sharding is None else stack_sharding.num_devices
    out: dict[int, Array] = {}
    for shape, indices in groups.items():
        stacks = [canon[i][0] for i in indices]
        sizes = [s.shape[0] for s in stacks]
        out_dtype = stacks[0].dtype
        # Cast BEFORE the sharding constraint so the ingress reshard moves ns_dtype
        # bytes (half, for bf16 NS), not fp32.
        grouped = jnp.concatenate(stacks, axis=0).astype(ns_dtype)
        grouped = _pad_to_multiple(grouped, n_shards)
        if stack_sharding is not None:
            grouped = jax.lax.with_sharding_constraint(grouped, stack_sharding)
        orthogonalized = orthogonalize_via_newton_schulz(
            grouped,
            jnp.asarray(ns_coeffs, ns_dtype),
            ns_steps=ns_steps,
            dimension_numbers=_NS_DIMS,
        ).astype(out_dtype)
        offset = 0
        for i, size in zip(indices, sizes, strict=True):
            piece = orthogonalized[offset : offset + size]
            offset += size
            _, transposed = canon[i]
            piece = jnp.swapaxes(piece, -2, -1) if transposed else piece
            out[i] = piece[0] if leaves[i].ndim == 2 else piece
        del shape
    return jax.tree.unflatten(treedef, [out[i] for i in range(len(leaves))])


def scale_by_stacked_muon(
    *,
    beta: float,
    ns_steps: int,
    ns_dtype: jnp.dtype,
    stack_sharding: NamedSharding | None,
) -> optax.GradientTransformation:
    """`optax.contrib.scale_by_muon` (nesterov, non-adaptive, frobenius) with the per-leaf
    NS replaced by `_grouped_newton_schulz`. State is optax's `MuonState` verbatim."""
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
        orthogonalized = _grouped_newton_schulz(
            mu_hat, jnp.asarray(state.ns_coeffs), ns_steps, ns_dtype, stack_sharding
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
    mesh: Mesh | None,
) -> optax.GradientTransformation:
    """Drop-in for `optax.contrib.muon(...)` at our call site (`run_state`): same
    muon/adam leaf partition, same post-NS chain, same state pytree — NS runs stacked,
    sharded over `(replicate, fsdp)` when a mesh is given (None => single-device, e.g.
    CPU tests and the toys)."""
    stack_sharding = (
        NamedSharding(mesh, P(("replicate", "fsdp"), None, None)) if mesh is not None else None
    )
    dim_nums = muon_weight_dimension_numbers
    if dim_nums is None:
        dim_nums = lambda params: jax.tree.map(lambda x: _NS_DIMS if x.ndim == 2 else None, params)

    def param_labels(params: optax.Params) -> optax.Params:
        resolved = dim_nums(params)
        return jax.tree.map(
            lambda spec, x: jax.tree.map(lambda _: "muon" if spec is not None else "adam", x),
            resolved,
            params,
            is_leaf=lambda x: x is None or isinstance(x, optax.contrib.MuonDimensionNumbers),
        )

    return optax.partition(
        transforms={
            "muon": optax.chain(
                scale_by_stacked_muon(
                    beta=beta,
                    ns_steps=ns_steps,
                    ns_dtype=ns_dtype,
                    stack_sharding=stack_sharding,
                ),
                scale_by_shape(
                    weight_dimension_numbers=lambda updates: jax.tree.map(
                        lambda x: _NS_DIMS if x.ndim == 3 else _2D_DIMS, updates
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
        param_labels=param_labels,
    )
