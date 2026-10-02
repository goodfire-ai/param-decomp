"""Routed local sparse expert compute: sort → grouped matmul → unsort.

Instead of a dense matmul over all E experts, each token's top-k expert assignments
become k JOBS. Jobs are stably argsorted by expert so each expert's jobs form one
contiguous row block, a grouped matmul contracts each block against its expert's matrix,
and the inverse permutation restores token order for the fp32 routing-weighted combine.
Shapes are STATIC throughout: the jobs axis is exactly T·k, and data-dependence lives
only in integer VALUES (the permutations and `group_sizes`), never in shapes.

The sort/unsort gathers carry custom VJPs (after MaxText's `_sort_activations_custom`):
each direction's backward is the gather by the opposite permutation, so the transpose
never lowers to an XLA scatter-add. `gather_tokens` additionally fuses Levanter's
gather-instead-of-repeat (`x[sort_idx // k]`); its backward is the inverse gather plus a
per-token sum over the k job slots.

One `RoutedJobs` schedule per MoE layer serves every matrix of that layer — frozen AND
decomposed: the expert-blocked component factors `V_e [d_in, c]` / `U_e [c, d_out]` are
stored expert-major, exactly the grouped-matmul rhs layout, so a routed decomposed site
is the same gather → grouped matmuls → unsort with two extra job-space tensors. Those
ride their own primitives here: `sort_jobs` (per-(token, slot) values — routing weights —
into sorted job order), `gather_job_blocks` (each job's row of a per-(token, expert)
table — masks/CI narrowed to the k selected blocks), and `sum_jobs` (the weightless
combine, for consumers that fold routing weights upstream of the down matmul). The block
gather is the one PARTIAL gather in the module — only T·k of the T·E rows are read — so
its autodiff transpose is honestly a scatter-add (row indices distinct: top-k experts
are distinct within a token), not an inverse gather.

`ExpertShardedJobs` is the EXPERT-PARALLEL sibling for the explicit `(data, tp)` mesh:
the global argsort (whose sort axis would span data shards — sort refuses a sharded
axis, correctly) becomes a per-(batch-row, expert-shard) sentinel sort, each shard's
local jobs sorted first by local expert and everyone else's masked to the tail. Every
schedule integer is computed replicated over the expert-shard axis — recomputing all S
tiny sorts on every rank costs nothing and keeps the schedule build collective-free —
while the gathered activations and matmuls shard over it. The grouped matmul is one
batched `ragged_dot_general` (batch dims = (batch row, expert shard), per-cell
`group_sizes`) under a custom VJP: jax's derivative rule for the ragged primitive does
not thread `out_sharding`, so the two transpose contractions (mode-1 d_lhs, mode-2
d_rhs) are spelled here with theirs. Partial outputs zero outside a shard's local jobs,
so the combine's sum over shards — the one forward collective, an all-reduce over the
expert-shard axis — is exact.

`ep_grouped_matmul` additionally serves TRAINED expert stacks (decomposed V/U blocks):
a weight carrying master provenance (`reduced` typing — its compute resident was
gathered over those mesh axes once per step, at entry) is untagged inside the custom
VJP (a typing move the VJP owns), and its weight cotangent comes back typed `unreduced`
over that same provenance — each rank's local batch contribution, deferred to the entry
boundary's transpose, so no weight-gradient collective ever runs inside a loop. This is
the spelling `linear_plan._planned_contraction` gets from jax's einsum reduced-typing
rule; the ragged primitive has no such rule, so the custom VJP carries it. The
`tokamax_split_vjp` backend runs the same contract on kernels: tokamax exposes no
`out_sharding`, so each product becomes per-(batch row) 2-D kernel calls under
`jax.shard_map` over the same axes — identical custom-VJP structure, identical
`unreduced` weight-cotangent typing. `ep_sort_jobs` / `ep_gather_job_blocks` /
`ep_sum_jobs` are the EP siblings of the decomposed-seam primitives above.
"""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.extend.backend
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

GroupedMatmulBackend = Literal["ragged_dot", "tokamax", "tokamax_split_vjp"]
"""The closed grouped-matmul arms: `ragged_dot` is the everywhere-correct oracle
(`jax.lax.ragged_dot`; its GPU lowering is dense-masked — correct, not sparse-fast);
`tokamax` is openxla's kernel arm as shipped (Mosaic-GPU sm90/sm100 kernels, XLA
fallback elsewhere, custom VJPs of its own — but no sm100 kernel exists for the
backward's ragged-contracting d_weights product, openxla/tokamax#628, so its VJP is
unusable on B200); `tokamax_split_vjp` is the sm100 TRAINING arm: tokamax triton
forward, backward split per product — d_input is forward-shaped (a second tokamax
triton call against transposed weights), d_weights runs the in-repo
`transposed_grouped_matmul` kernel (tokamax's triton spelling of that product
miscomputes empty groups, openxla/tokamax#887). Both tokamax calls pin
`implementation=("triton", "xla")`: 0.0.13's sm100 Mosaic kernel mis-schedules at
large expert counts — NaN/garbage values or a spin that never returns, shape-dependent
(the forward's comment carries the evidence)."""

ExpertImplementation = Literal["dense_masked"] | GroupedMatmulBackend
"""Expert execution strategy, independent of component/CI emission layout."""

_TRANS_RHS_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [2]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)
"""`ragged_dot` against `[E, d_out, d_in]`-transposed weights — the d_input product's
shape. Matches tokamax's `TRANS_RHS` constant, which its Mosaic arms fold into the
default forward kernel."""


@dataclass(frozen=True)
class RoutedJobs:
    """One MoE layer's job schedule. Job j = (token j // k, slot j % k) in token-major
    ids; `sort_idx` lists job ids grouped by expert (stable sort, so token order survives
    within an expert), `inv_sort_idx` is its inverse permutation, and `group_sizes[e]`
    counts expert e's jobs. Integers only — the routing WEIGHTS stay with the caller."""

    top_idx: Int[Array, "T k"]
    sort_idx: Int[Array, " J"]
    inv_sort_idx: Int[Array, " J"]
    group_sizes: Int[Array, " E"]

    @property
    def experts_per_token(self) -> int:
        return self.top_idx.shape[-1]


def routed_jobs(top_idx: Int[Array, "T k"], n_experts: int) -> RoutedJobs:
    flat_expert_of_job = top_idx.reshape(-1)
    sort_idx = jnp.argsort(flat_expert_of_job, stable=True)
    return RoutedJobs(
        top_idx=top_idx,
        sort_idx=sort_idx,
        inv_sort_idx=jnp.argsort(sort_idx),
        group_sizes=jnp.bincount(flat_expert_of_job, length=n_experts),
    )


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _gather_rows(k: int, x: Array, sort_idx: Array, inv_sort_idx: Array) -> Array:
    del inv_sort_idx
    return jnp.take(x, sort_idx // k, axis=0)


def _gather_rows_fwd(k: int, x: Array, sort_idx: Array, inv_sort_idx: Array) -> tuple[Array, Array]:
    return _gather_rows(k, x, sort_idx, inv_sort_idx), inv_sort_idx


def _gather_rows_bwd(k: int, inv_sort_idx: Array, grad: Array) -> tuple[Array, None, None]:
    token_major = jnp.take(grad, inv_sort_idx, axis=0)
    return token_major.reshape(-1, k, grad.shape[-1]).sum(axis=1), None, None


_gather_rows.defvjp(_gather_rows_fwd, _gather_rows_bwd)


@jax.custom_vjp
def _permute_rows(y: Array, perm: Array, inverse_perm: Array) -> Array:
    """`y[perm]` with the permutation-gather transpose: backward is `grad[inverse_perm]`,
    never an XLA scatter-add."""
    del inverse_perm
    return jnp.take(y, perm, axis=0)


def _permute_rows_fwd(y: Array, perm: Array, inverse_perm: Array) -> tuple[Array, Array]:
    return _permute_rows(y, perm, inverse_perm), inverse_perm


def _permute_rows_bwd(inverse_perm: Array, grad: Array) -> tuple[Array, None, None]:
    return jnp.take(grad, inverse_perm, axis=0), None, None


_permute_rows.defvjp(_permute_rows_fwd, _permute_rows_bwd)


def gather_tokens(x: Float[Array, "T d"], jobs: RoutedJobs) -> Float[Array, "J d"]:
    """Each job's token row in expert-sorted job order — the grouped-matmul lhs."""
    assert x.shape[0] == jobs.top_idx.shape[0], (x.shape, jobs.top_idx.shape)
    return _gather_rows(jobs.experts_per_token, x, jobs.sort_idx, jobs.inv_sort_idx)


def unsort_jobs(y: Float[Array, "J n"], jobs: RoutedJobs) -> Float[Array, "T k n"]:
    """Expert-sorted job rows back to token-major (token, slot) layout."""
    n_tokens, k = jobs.top_idx.shape
    return _permute_rows(y, jobs.inv_sort_idx, jobs.sort_idx).reshape(n_tokens, k, y.shape[-1])


def sort_jobs(values: Float[Array, "T k"], jobs: RoutedJobs) -> Float[Array, " J"]:
    """Per-(token, slot) job values — the routing weights — in expert-sorted job order."""
    assert values.shape == jobs.top_idx.shape, (values.shape, jobs.top_idx.shape)
    return _permute_rows(values.reshape(-1), jobs.sort_idx, jobs.inv_sort_idx)


def gather_job_blocks(table: Float[Array, "T E n"], jobs: RoutedJobs) -> Float[Array, "J n"]:
    """Each job's (token, expert) row of a per-(token, expert) table — a mask/CI tensor
    viewed `[T, E, block]` — in expert-sorted job order. A PARTIAL gather (T·k of the
    T·E rows), so its transpose is left to autodiff as the scatter-add it honestly is;
    the read rows are distinct (top-k experts are distinct within a token)."""
    assert table.shape[0] == jobs.top_idx.shape[0], (table.shape, jobs.top_idx.shape)
    token_of_job = jobs.sort_idx // jobs.experts_per_token
    expert_of_job = jobs.top_idx.reshape(-1)[jobs.sort_idx]
    return table[token_of_job, expert_of_job]


def sort_job_values(values: Float[Array, "T k n"], jobs: RoutedJobs) -> Float[Array, "J n"]:
    """Per-(token, slot) job ROWS — a narrow mask/CI tensor whose slot m already means
    the token's m-th routed expert — in expert-sorted job order: `sort_jobs` with a
    payload axis. Job j's row is `values[token(j), slot(j)]`, a pure permutation (its
    transpose is the inverse gather, never a scatter-add)."""
    n_tokens, k = jobs.top_idx.shape
    assert values.shape[:2] == (n_tokens, k), (values.shape, jobs.top_idx.shape)
    return _permute_rows(values.reshape(n_tokens * k, -1), jobs.sort_idx, jobs.inv_sort_idx)


def sum_jobs(y: Float[Array, "J d"], jobs: RoutedJobs) -> Float[Array, "T d"]:
    """fp32 sum of each token's k job outputs, cast back to `y.dtype` — the weightless
    sibling of `combine_jobs`, for consumers that fold routing weights upstream."""
    per_token = unsort_jobs(y, jobs)
    return jnp.sum(per_token.astype(jnp.float32), axis=1).astype(y.dtype)


def _transposed_grouped_matmul_kernel(
    cum_ref, lhs_ref, dout_ref, out_ref, *, block_j: int, block_m: int, block_n: int
) -> None:
    expert = pl.program_id(0)
    lo = cum_ref[expert]
    hi = cum_ref[expert + 1]

    def accumulate(step: Array, acc: Array) -> Array:
        start = (lo // block_j + step) * block_j
        rows = start + jnp.arange(block_j)
        live = ((rows >= lo) & (rows < hi))[:, None]
        lhs = plgpu.load(lhs_ref.at[pl.ds(start, block_j)], mask=live, other=0.0)
        dout = plgpu.load(dout_ref.at[pl.ds(start, block_j)], mask=live, other=0.0)
        return acc + plgpu.dot(lhs.T, dout, precision=jax.lax.Precision.HIGHEST)

    # Block starts are block_j-ALIGNED with rows masked to [lo, hi): a jobs axis padded
    # to a block_j multiple then never yields an out-of-bounds slice, which interpret
    # mode would clamp (silently shifting the window) where triton's masked pointer
    # loads would not. An EMPTY expert's blocks mask to nothing, so it comes back
    # exactly zero.
    n_steps = pl.cdiv(hi, block_j) - lo // block_j
    acc = jax.lax.fori_loop(0, n_steps, accumulate, jnp.zeros((block_m, block_n), jnp.float32))
    out_ref[...] = acc.astype(out_ref.dtype)


def transposed_grouped_matmul(
    lhs: Float[Array, "J d_in"],
    dout: Float[Array, "J d_out"],
    group_sizes: Int[Array, " E"],
    out_dtype: jnp.dtype,
) -> Float[Array, "E d_in d_out"]:
    """`out[e] = lhs[block_e].T @ dout[block_e]` over the contiguous per-expert row
    blocks — the grouped matmul's WEIGHT cotangent, ragged over the contracted jobs
    axis. Neither tokamax nor jax ships this product as an sm100 kernel (tokamax's
    Mosaic arm has it for sm90 only, its triton arm miscomputes empty groups
    openxla/tokamax#887, and jax's in-tree `transposed_ragged_dot_mgpu` is wgmma —
    Hopper-only), so it lives here as a Pallas-Triton kernel: fp32 accumulation,
    one (expert, d_in-tile, d_out-tile) grid cell per output tile, each walking only
    its expert's job rows."""
    n_jobs, d_in = lhs.shape
    assert dout.shape[0] == n_jobs, (lhs.shape, dout.shape)
    d_out = dout.shape[1]
    block_j, block_m, block_n = 32, 64, 64
    assert d_in % block_m == 0 and d_out % block_n == 0, (lhs.shape, dout.shape)
    # Pad the jobs axis to a block_j multiple so aligned block slices stay in bounds
    # (kernel comment); padded rows sit beyond every group's `hi` and mask to nothing.
    padded = pl.cdiv(n_jobs, block_j) * block_j
    lhs = jnp.pad(lhs, ((0, padded - n_jobs), (0, 0)))
    dout = jnp.pad(dout, ((0, padded - n_jobs), (0, 0)))
    cum = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(group_sizes, dtype=jnp.int32)])
    return pl.pallas_call(
        partial(
            _transposed_grouped_matmul_kernel, block_j=block_j, block_m=block_m, block_n=block_n
        ),
        out_shape=jax.ShapeDtypeStruct((group_sizes.shape[0], d_in, d_out), out_dtype),
        in_specs=[
            pl.no_block_spec,
            pl.BlockSpec((padded, block_m), lambda e, i, j: (0, i)),
            pl.BlockSpec((padded, block_n), lambda e, i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((None, block_m, block_n), lambda e, i, j: (e, i, j)),
        grid=(group_sizes.shape[0], d_in // block_m, d_out // block_n),
        # Triton CompilerParams select the TRITON lowering — pallas_call's GPU default
        # is Mosaic-GPU, whose warpgroup semantics do not lower masked loads.
        compiler_params=plgpu.CompilerParams(),
        # The interpret gate keys on the default DEVICE, not the process backend: a
        # deviceless compile under a `jax.default_device(<topology GPU>)` context
        # lowers the real Triton kernel, while the CPU institutions (unit parity, the
        # trace gate) carry no such context and get interpret mode — the same program,
        # lowered as plain XLA ops.
        interpret=jax.extend.backend.get_default_device().platform != "gpu",
    )(cum, lhs, dout)


@jax.custom_vjp
def _tokamax_split_grouped_matmul(lhs: Array, rhs: Array, group_sizes: Array) -> Array:
    import tokamax

    # Pinned off the sm100 Mosaic kernel: 0.0.13's scheduling is broken at large expert
    # counts — garbage values at [16384,512]x[256,512,512], and a spin that never
    # returns at [32768,64]x[256,64,512] (probes 264649/264871) — suspected
    # openxla/tokamax#1276, fixed upstream after 0.0.13. Triton serves the forward on
    # GPU; XLA everywhere else.
    return tokamax.ragged_dot(lhs, rhs, group_sizes, implementation=("triton", "xla"))


def _tokamax_split_fwd(
    lhs: Array, rhs: Array, group_sizes: Array
) -> tuple[Array, tuple[Array, Array, Array]]:
    return _tokamax_split_grouped_matmul(lhs, rhs, group_sizes), (lhs, rhs, group_sizes)


def _tokamax_split_bwd(
    residuals: tuple[Array, Array, Array], grad: Array
) -> tuple[Array, Array, None]:
    lhs, rhs, group_sizes = residuals
    import tokamax

    # d_input is FORWARD-shaped — a ragged dot against transposed weights — on the same
    # triton/xla pin as the forward (the Mosaic exclusion there applies with the same
    # evidence: NaN at [16384,512]x[256,512,512] TRANS_RHS). Only d_weights needs the
    # ragged-contracting product no sm100 kernel ships; it runs the in-repo kernel.
    d_lhs = tokamax.ragged_dot_general(
        grad,
        rhs,
        group_sizes,
        _TRANS_RHS_DIM_NUMS,
        preferred_element_type=lhs.dtype,
        implementation=("triton", "xla"),
    )
    d_rhs = transposed_grouped_matmul(lhs, grad, group_sizes, out_dtype=rhs.dtype)
    return d_lhs, d_rhs, None


_tokamax_split_grouped_matmul.defvjp(_tokamax_split_fwd, _tokamax_split_bwd)


def grouped_matmul(
    lhs: Float[Array, "J d_in"],
    rhs: Float[Array, "E d_in d_out"],
    group_sizes: Int[Array, " E"],
    backend: GroupedMatmulBackend,
) -> Float[Array, "J d_out"]:
    """`lhs`'s e-th contiguous `group_sizes[e]`-row block times `rhs[e]`."""
    match backend:
        case "ragged_dot":
            return jax.lax.ragged_dot(lhs, rhs, group_sizes)
        case "tokamax":
            # Deferred import (both tokamax arms): tokamax pulls a heavy dep chain
            # (flax/qwix/xprof) that only these arms need. Order hazard: the chain's
            # xprof C++ static init self-deadlocks if pyarrow's bundled protobuf loaded
            # first, so a process that reads parquet (the LM data path, the test suite)
            # must import tokamax before pyarrow — the root conftest does; a trainer
            # flipping to these arms must import it at its composition root before data
            # loading.
            import tokamax

            return tokamax.ragged_dot(lhs, rhs, group_sizes)
        case "tokamax_split_vjp":
            return _tokamax_split_grouped_matmul(lhs, rhs, group_sizes)


def combine_jobs(
    y: Float[Array, "J d"], jobs: RoutedJobs, weights: Float[Array, "T k"]
) -> Float[Array, "T d"]:
    """fp32 routing-weighted sum of each token's k job outputs, cast back to `y.dtype`.
    An elementwise product under a reduction, never a dot: the fp32 promotion fuses
    into the reduce, so no fp32 copy of job space materializes."""
    per_token = unsort_jobs(y, jobs)
    weighted = per_token.astype(jnp.float32) * weights.astype(jnp.float32)[..., None]
    return jnp.sum(weighted, axis=1).astype(y.dtype)


def scatter_jobs(y: Float[Array, "J n"], jobs: RoutedJobs, n_experts: int) -> Float[Array, "T E n"]:
    """Job outputs placed at their (token, expert) slots; unassigned slots are ZERO.
    Full-width materialization — call only when a consumer demands the dense layout."""
    per_token = unsort_jobs(y, jobs)
    n_tokens = jobs.top_idx.shape[0]
    zeros = jnp.zeros((n_tokens, n_experts, y.shape[-1]), y.dtype)
    # top-k expert ids are distinct within a token, so the scatter has unique indices.
    return zeros.at[jnp.arange(n_tokens)[:, None], jobs.top_idx].set(per_token, unique_indices=True)


# ── expert-parallel jobs (the explicit-mesh arm) ──────────────────────────────


@dataclass(frozen=True)
class ExpertShardedJobs:
    """Per-(batch row, expert shard) job schedules for expert-parallel routed compute.

    Row (b, s) permutes ALL of batch row b's `J = T·k` jobs: jobs for shard s's local
    experts first (stably sorted by local expert id), everyone else's sentinel-sorted to
    the tail; `group_sizes[b, s]` counts shard s's local experts' jobs, so its sum is
    the row's live-job count and the tail beyond it is dead. Integers only, replicated
    over the expert-shard axis by construction (the consumers reshard where a sharded
    copy is cheaper); batch rows ride whatever sharding `top_idx` carries."""

    top_idx: Int[Array, "B T k"]
    sort_idx: Int[Array, "B S J"]
    inv_sort_idx: Int[Array, "B S J"]
    group_sizes: Int[Array, "B S E_local"]

    @property
    def experts_per_token(self) -> int:
        return self.top_idx.shape[-1]

    @property
    def n_shards(self) -> int:
        return self.sort_idx.shape[1]


def expert_sharded_jobs(
    top_idx: Int[Array, "B T k"], n_experts: int, n_shards: int
) -> ExpertShardedJobs:
    assert n_experts % n_shards == 0, (n_experts, n_shards)
    experts_per_shard = n_experts // n_shards
    expert_of_job = top_idx.reshape(top_idx.shape[0], -1)  # [B, J]
    shard_of_job = expert_of_job // experts_per_shard
    local_expert_of_job = expert_of_job % experts_per_shard
    # sentinel = experts_per_shard: non-local jobs sort after every local expert.
    local = jnp.where(
        shard_of_job[:, None, :] == jnp.arange(n_shards)[None, :, None],
        local_expert_of_job[:, None, :],
        experts_per_shard,
    )  # [B, S, J]
    sort_idx = jnp.argsort(local, axis=-1, stable=True)
    return ExpertShardedJobs(
        top_idx=top_idx,
        sort_idx=sort_idx,
        inv_sort_idx=jnp.argsort(sort_idx, axis=-1),
        group_sizes=jnp.sum(jax.nn.one_hot(local, experts_per_shard, dtype=jnp.int32), axis=-2),
    )


def _mesh_of(value: Array) -> Mesh | AbstractMesh:
    mesh = jax.typeof(value).sharding.mesh
    assert not mesh.empty, "the expert-parallel arm runs only on an explicit mesh"
    return mesh


def _batch_entry(value: Array) -> str | tuple[str, ...] | None:
    """The batch-row axis assignment carried by `value`'s type (its leading spec entry)."""
    return jax.typeof(value).sharding.spec[0]


def _shard_jobs_ints(ints: Int[Array, "B S J"], shard_axis: str) -> Array:
    """Slice a replicated schedule tensor to its expert-shard rows (a local move)."""
    mesh = _mesh_of(ints)
    return jax.sharding.reshard(ints, NamedSharding(mesh, P(_batch_entry(ints), shard_axis, None)))


def _ep_gather_replicated_rows(
    rows: Float[Array, "B R n"], row_of_job: Int[Array, "B S J"], shard_axis: str
) -> Float[Array, "B S J n"]:
    """Each (batch row, expert shard)'s rows of a shard-replicated table, in that shard's
    job order. The schedule is sliced to this shard BEFORE the gather — the table's shard
    axis is a typed broadcast, a local view — so the gather reads only this shard's `J`
    rows; XLA does not push a reshard through a gather, so gathering all `S` shards' rows
    and slicing afterwards materializes the whole `[B, S, J, n]` first."""
    b, s, _j = row_of_job.shape
    per_shard = jnp.broadcast_to(
        rows[:, None],
        (b, s, *rows.shape[1:]),
        out_sharding=NamedSharding(_mesh_of(rows), P(_batch_entry(rows), shard_axis, None, None)),
    )
    return jnp.take_along_axis(
        per_shard, _shard_jobs_ints(row_of_job, shard_axis)[..., None], axis=2
    )


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ep_gather_rows(shard_axis: str, x: Array, sort_idx: Array, inv_sort_idx: Array) -> Array:
    del inv_sort_idx
    k = sort_idx.shape[-1] // x.shape[1]
    return _ep_gather_replicated_rows(x, sort_idx // k, shard_axis)


def _ep_gather_rows_fwd(
    shard_axis: str, x: Array, sort_idx: Array, inv_sort_idx: Array
) -> tuple[Array, tuple[Array, int]]:
    return _ep_gather_rows(shard_axis, x, sort_idx, inv_sort_idx), (
        inv_sort_idx,
        sort_idx.shape[-1] // x.shape[1],
    )


def _ep_gather_rows_bwd(
    shard_axis: str, residuals: tuple[Array, int], grad: Array
) -> tuple[Array, None, None]:
    inv_sort_idx, k = residuals
    token_major = jnp.take_along_axis(
        grad, _shard_jobs_ints(inv_sort_idx, shard_axis)[..., None], axis=2
    )
    b, s, j, d = token_major.shape
    per_token = token_major.reshape(b, s, j // k, k, d)
    # the s-sum is the transpose of replicating x to every expert shard: an all-reduce
    # over the expert-shard axis, the backward twin of the combine's.
    return (
        jnp.einsum(
            "bstkd->btd",
            per_token,
            out_sharding=NamedSharding(_mesh_of(grad), P(_batch_entry(grad), None, None)),
        ),
        None,
        None,
    )


_ep_gather_rows.defvjp(_ep_gather_rows_fwd, _ep_gather_rows_bwd)


def ep_gather_tokens(
    x: Float[Array, "B T d"], jobs: ExpertShardedJobs, shard_axis: str
) -> Float[Array, "B S J d"]:
    """Each (batch row, expert shard)'s token rows in its sentinel-sorted job order."""
    assert x.shape[:2] == jobs.top_idx.shape[:2], (x.shape, jobs.top_idx.shape)
    return _ep_gather_rows(shard_axis, x, jobs.sort_idx, jobs.inv_sort_idx)


# lhs [B, S, J, d_in] × rhs [G, B, S, d_in, d_out] with per-(B, S) groups on the ragged
# J → [B, S, J, d_out]; MODE2 instead contracts the ragged J into per-group outer
# products — exactly the weight-gradient transpose — and runs on a shard-major FUSED
# batch [S*B, J, x] → [G, S*B, d_in, d_out]. The fusing is load-bearing: jaxlib 0.11's
# GPU ragged_dot_rewriter miscompiles a two-batch-dim mode-2 ragged dot (the dense dot
# it emits orders the batch dims opposite to the shape rule's expectation and the HLO
# verifier rejects the module), and one fused batch dim leaves no order to disagree on.
_EP_MODE1 = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((3,), (3,)), ((0, 1), (1, 2))),
    lhs_ragged_dimensions=[2],
    rhs_group_dimensions=[0],
)
_EP_MODE2_FUSED = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((0,), (0,))),
    lhs_ragged_dimensions=[1],
    rhs_group_dimensions=[],
)


def _live_job_mask(group_sizes: Int[Array, "B S G"], n_jobs: int) -> Array:
    """True on rows the per-cell groups cover; the sentinel tail beyond them is dead."""
    live = jnp.sum(group_sizes, axis=-1)  # [B, S]
    row = jax.lax.broadcasted_iota(jnp.int32, (*group_sizes.shape[:-1], n_jobs), 2)
    return row < live[..., None]


def _ep_expert_rhs(
    shard_axis: str, experts: Array, n_batch_rows: int, batch: str | tuple[str, ...] | None
) -> Array:
    """`[E@shard, a, b]` → the batched ragged rhs `[G=E_local, B, S@shard, a, b]`: untag
    any master provenance (a typing move — the enclosing custom VJP owns the weight
    cotangent), split shard-major, broadcast over batch rows; every move is local to
    the rank's own expert blocks."""
    mesh = _mesh_of(experts)
    n_experts, a, b = experts.shape
    n_shards = mesh.shape[shard_axis]
    plain = jax.sharding.reshard(experts, NamedSharding(mesh, P(shard_axis, None, None)))
    blocks = plain.reshape(n_shards, n_experts // n_shards, a, b)
    rhs = jnp.broadcast_to(
        blocks.transpose(1, 0, 2, 3)[:, None],
        (n_experts // n_shards, n_batch_rows, n_shards, a, b),
    )
    return jax.sharding.reshard(rhs, NamedSharding(mesh, P(None, batch, shard_axis, None, None)))


def _ep_weight_cotangent_spec(experts: Array) -> P:
    """The weight cotangent's spec: the primal's partitions with its `reduced`
    provenance flipped to `unreduced` — each rank's local-batch partial, deferred to
    the entry boundary's transpose (`_ep_expert_matmul_bwd`'s comment carries the full
    account). A provenance-free (frozen) weight's spec passes through unchanged."""
    spec = jax.typeof(experts).sharding.spec
    provenance = frozenset(spec.reduced)
    entries = tuple(spec.partitions)
    return P(*entries, unreduced=provenance) if provenance else P(*entries)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ep_expert_matmul(shard_axis: str, x_jobs: Array, experts: Array, group_sizes: Array) -> Array:
    """x_jobs [B,S,J,a] × experts [E,a,b] → [B,S,J,b], sentinel tail zeroed explicitly
    (ragged_dot leaves rows past the groups unspecified)."""
    mesh = _mesh_of(x_jobs)
    rhs = _ep_expert_rhs(shard_axis, experts, x_jobs.shape[0], _batch_entry(x_jobs))
    out = jax.lax.ragged_dot_general(
        x_jobs,
        rhs,
        group_sizes,
        _EP_MODE1,
        out_sharding=NamedSharding(mesh, P(_batch_entry(x_jobs), shard_axis, None, None)),
    )
    return jnp.where(_live_job_mask(group_sizes, x_jobs.shape[2])[..., None], out, 0.0)


def _ep_expert_matmul_fwd(
    shard_axis: str, x_jobs: Array, experts: Array, group_sizes: Array
) -> tuple[Array, tuple[Array, Array, Array]]:
    return _ep_expert_matmul(shard_axis, x_jobs, experts, group_sizes), (
        x_jobs,
        experts,
        group_sizes,
    )


def _ep_expert_matmul_bwd(
    shard_axis: str, residuals: tuple[Array, Array, Array], grad: Array
) -> tuple[Array, Array, None]:
    x_jobs, experts, group_sizes = residuals
    mesh = _mesh_of(x_jobs)
    batch = _batch_entry(x_jobs)
    n_batch_rows, _, _, _ = x_jobs.shape
    live = _live_job_mask(group_sizes, x_jobs.shape[2])[..., None]
    grad = jnp.where(live, grad, 0.0)
    rhs = _ep_expert_rhs(shard_axis, experts, n_batch_rows, batch)
    d_lhs = jax.lax.ragged_dot_general(
        grad,
        rhs.transpose(0, 1, 2, 4, 3),
        group_sizes,
        _EP_MODE1,
        out_sharding=NamedSharding(mesh, P(batch, shard_axis, None, None)),
    )
    # The weight cotangent: per-(batch row, shard) block outer products (mode-2), the
    # batch rows summed by ONE einsum whose output is typed `unreduced` over the
    # weight's provenance — each data rank keeps its partial, and the reduction fires
    # at the entry boundary that applied the `reduced` tag. A provenance-free (frozen)
    # weight's cotangent would be a genuine cross-batch all-reduce; frozen callers never
    # pull it, so DCE removes this path (the census pins that).
    #
    # The mode-2 call runs on the shard-major fused batch (see _EP_MODE2_FUSED): the
    # shard dim leads so the (S, B) merge and its inverse split are layout-preserving.
    match batch:
        case None:
            fused_entry: str | tuple[str, ...] = shard_axis
        case str():
            fused_entry = (shard_axis, batch)
        case tuple():
            fused_entry = (shard_axis, *batch)
    n_shards = x_jobs.shape[1]

    def fuse(value: Array) -> Array:
        swapped = value.transpose(1, 0, 2, 3)
        return jnp.reshape(
            swapped,
            (n_shards * n_batch_rows, *value.shape[2:]),
            out_sharding=NamedSharding(mesh, P(fused_entry, None, None)),
        )

    fused_group_sizes = jnp.reshape(
        group_sizes.transpose(1, 0, 2),
        (n_shards * n_batch_rows, group_sizes.shape[2]),
        out_sharding=NamedSharding(mesh, P(fused_entry, None)),
    )
    d_fused = jax.lax.ragged_dot_general(
        fuse(x_jobs),
        fuse(grad),
        fused_group_sizes,
        _EP_MODE2_FUSED,
        out_sharding=NamedSharding(mesh, P(None, fused_entry, None, None)),
    )
    d_blocks = jnp.reshape(
        d_fused,
        (d_fused.shape[0], n_shards, n_batch_rows, *d_fused.shape[2:]),
        out_sharding=NamedSharding(mesh, P(None, shard_axis, batch, None, None)),
    )
    stacked = d_blocks.transpose(2, 1, 0, 3, 4).reshape(n_batch_rows, *experts.shape)
    d_experts = jnp.einsum(
        "neab->eab", stacked, out_sharding=NamedSharding(mesh, _ep_weight_cotangent_spec(experts))
    )
    # cotangent dtypes must match the primals': mixed-precision callers (fp32 masks on
    # bf16 weights) promote the lhs, and the ragged products come back at the promoted
    # dtype — the cast jax's convert transpose would insert for a native einsum.
    return (
        jnp.where(live, d_lhs, 0.0).astype(x_jobs.dtype),
        d_experts.astype(experts.dtype),
        None,
    )


_ep_expert_matmul.defvjp(_ep_expert_matmul_fwd, _ep_expert_matmul_bwd)


# ── the split-VJP arm under expert parallelism ────────────────────────────────
#
# `_ep_expert_matmul`'s custom-VJP structure with each product's typed
# `ragged_dot_general` swapped for its `tokamax_split_vjp` kernel: tokamax exposes no
# `out_sharding`, so under the explicit mesh its calls (and the in-repo transposed
# kernel) run in manual axes via `jax.shard_map` — one 2-D grouped matmul per local
# (batch row, expert shard) cell against the rank's own expert blocks, exactly the
# per-rank shape the arm was priced at. `check_vma=False` on every map: neither tokamax
# 0.0.13 nor `transposed_grouped_matmul` declares output vma (`manual_axis_type`) on
# its pallas out_shapes, so the vma checker refuses them under a manual map; the
# out_specs are this module's contract, pinned by the EP parity tests.


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ep_tokamax_split_matmul(
    shard_axis: str, x_jobs: Array, experts: Array, group_sizes: Array
) -> Array:
    mesh = _mesh_of(x_jobs)
    batch = _batch_entry(x_jobs)
    # untag master provenance — the typing move the VJP owns (`_ep_expert_rhs` docs);
    # shard_map takes the plain shard-resident blocks.
    plain = jax.sharding.reshard(experts, NamedSharding(mesh, P(shard_axis, None, None)))

    def forward_cells(x_cells: Array, blocks: Array, sizes: Array) -> Array:
        # local views: x_cells [B_local, 1, J, a], blocks [E_local, a, b],
        # sizes [B_local, 1, E_local] — the schedule's shard axis matches the mesh axis.
        import tokamax

        assert x_cells.shape[1] == 1, (x_cells.shape, shard_axis)
        return jnp.stack(
            [
                tokamax.ragged_dot(
                    x_cells[row, 0], blocks, sizes[row, 0], implementation=("triton", "xla")
                )
                for row in range(x_cells.shape[0])
            ]
        )[:, None]

    out = jax.shard_map(
        forward_cells,
        mesh=mesh,
        in_specs=(
            P(batch, shard_axis, None, None),
            P(shard_axis, None, None),
            P(batch, shard_axis, None),
        ),
        out_specs=P(batch, shard_axis, None, None),
        check_vma=False,
    )(x_jobs, plain, group_sizes)
    return jnp.where(_live_job_mask(group_sizes, x_jobs.shape[2])[..., None], out, 0.0)


def _ep_tokamax_split_matmul_fwd(
    shard_axis: str, x_jobs: Array, experts: Array, group_sizes: Array
) -> tuple[Array, tuple[Array, Array, Array]]:
    return _ep_tokamax_split_matmul(shard_axis, x_jobs, experts, group_sizes), (
        x_jobs,
        experts,
        group_sizes,
    )


def _ep_tokamax_split_matmul_bwd(
    shard_axis: str, residuals: tuple[Array, Array, Array], grad: Array
) -> tuple[Array, Array, None]:
    x_jobs, experts, group_sizes = residuals
    mesh = _mesh_of(x_jobs)
    batch = _batch_entry(x_jobs)
    live = _live_job_mask(group_sizes, x_jobs.shape[2])[..., None]
    grad = jnp.where(live, grad, 0.0)
    plain = jax.sharding.reshard(experts, NamedSharding(mesh, P(shard_axis, None, None)))

    def d_lhs_cells(grad_cells: Array, blocks: Array, sizes: Array) -> Array:
        import tokamax

        # d_input is forward-shaped on the same triton/xla pin; the Mosaic exclusion
        # and its evidence live on `_tokamax_split_bwd`.
        return jnp.stack(
            [
                tokamax.ragged_dot_general(
                    grad_cells[row, 0],
                    blocks,
                    sizes[row, 0],
                    _TRANS_RHS_DIM_NUMS,
                    preferred_element_type=x_jobs.dtype,
                    implementation=("triton", "xla"),
                )
                for row in range(grad_cells.shape[0])
            ]
        )[:, None]

    d_lhs = jax.shard_map(
        d_lhs_cells,
        mesh=mesh,
        in_specs=(
            P(batch, shard_axis, None, None),
            P(shard_axis, None, None),
            P(batch, shard_axis, None),
        ),
        out_specs=P(batch, shard_axis, None, None),
        check_vma=False,
    )(grad, plain, group_sizes)

    def d_experts_cells(x_cells: Array, grad_cells: Array, sizes: Array) -> Array:
        # The local batch rows accumulate into ONE fp32 [E_local, a, b] carry (the
        # kernel's own accumulator dtype, cast to the weight dtype once, below): each
        # row's kernel output is live only until it is added, never all rows at once.
        def add_row(row: Array, acc: Array) -> Array:
            return acc + transposed_grouped_matmul(
                x_cells[row, 0], grad_cells[row, 0], sizes[row, 0], out_dtype=jnp.float32
            )

        acc = jnp.zeros((sizes.shape[-1], x_cells.shape[-1], grad_cells.shape[-1]), jnp.float32)
        return jax.lax.fori_loop(0, x_cells.shape[0], add_row, acc)[None]

    # [n_batch_shards, E, a, b]: one summed cell per data shard; the einsum's typed
    # `unreduced` output keeps each shard's partial for the entry boundary's transpose.
    local_sums = jax.shard_map(
        d_experts_cells,
        mesh=mesh,
        in_specs=(
            P(batch, shard_axis, None, None),
            P(batch, shard_axis, None, None),
            P(batch, shard_axis, None),
        ),
        out_specs=P(batch, shard_axis, None, None),
        check_vma=False,
    )(x_jobs, grad, group_sizes)
    d_experts = jnp.einsum(
        "neab->eab",
        local_sums,
        out_sharding=NamedSharding(mesh, _ep_weight_cotangent_spec(experts)),
    )
    return (
        jnp.where(live, d_lhs, 0.0).astype(x_jobs.dtype),
        d_experts.astype(experts.dtype),
        None,
    )


_ep_tokamax_split_matmul.defvjp(_ep_tokamax_split_matmul_fwd, _ep_tokamax_split_matmul_bwd)


def ep_grouped_matmul(
    x_jobs: Float[Array, "B S J d_in"],
    experts: Float[Array, "E d_in d_out"],
    jobs: ExpertShardedJobs,
    shard_axis: str,
    backend: GroupedMatmulBackend,
) -> Float[Array, "B S J d_out"]:
    """Each (batch row, expert shard)'s live jobs times their local expert matrices;
    the dead sentinel tail comes back exactly zero. `experts` is expert-major and must
    rest split over `shard_axis` — this arm exists so no rank ever touches another
    shard's expert weights. Serves frozen stacks and provenance-carrying trained stacks
    alike (module docstring)."""
    assert experts.shape[0] % jobs.n_shards == 0, (experts.shape, jobs.n_shards)
    mesh = _mesh_of(x_jobs)
    group_sizes = jax.sharding.reshard(
        jobs.group_sizes, NamedSharding(mesh, P(_batch_entry(x_jobs), shard_axis, None))
    )
    match backend:
        case "ragged_dot":
            return _ep_expert_matmul(shard_axis, x_jobs, experts, group_sizes)
        case "tokamax_split_vjp":
            return _ep_tokamax_split_matmul(shard_axis, x_jobs, experts, group_sizes)
        case "tokamax":
            raise NotImplementedError(
                "the tokamax arm (tokamax's own VJP) is enumerated but not wired for "
                "expert-parallel compute: no caller needs it — sm100 training requires "
                "the backward split per product (tokamax#628), which tokamax_split_vjp "
                "wires under manual axes here"
            )


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ep_unsort_rows(shard_axis: str, y: Array, sort_idx: Array, inv_sort_idx: Array) -> Array:
    del sort_idx
    return jnp.take_along_axis(y, _shard_jobs_ints(inv_sort_idx, shard_axis)[..., None], axis=2)


def _ep_unsort_rows_fwd(
    shard_axis: str, y: Array, sort_idx: Array, inv_sort_idx: Array
) -> tuple[Array, Array]:
    return _ep_unsort_rows(shard_axis, y, sort_idx, inv_sort_idx), sort_idx


def _ep_unsort_rows_bwd(shard_axis: str, sort_idx: Array, grad: Array) -> tuple[Array, None, None]:
    return (
        jnp.take_along_axis(grad, _shard_jobs_ints(sort_idx, shard_axis)[..., None], axis=2),
        None,
        None,
    )


_ep_unsort_rows.defvjp(_ep_unsort_rows_fwd, _ep_unsort_rows_bwd)


def ep_combine_jobs(
    y: Float[Array, "B S J d"],
    jobs: ExpertShardedJobs,
    weights: Float[Array, "B T k"],
    shard_axis: str,
    out_spec: P,
) -> Float[Array, "B T d"]:
    """fp32 routing-weighted sum of each token's k job outputs across expert shards —
    every job is live on exactly one shard and zero elsewhere, so the shard sum is
    exact. `out_spec` is the caller's `[B, T, d]` output placement: token axis
    replicated lowers the sum as an all-reduce over `shard_axis`; token axis sharded
    over it, as a reduce-scatter (the sequence-parallel waist). An elementwise product
    under a reduction, never a dot: the fp32 promotion fuses into the reduce, so no fp32
    copy of job space (7/8 of it the dead sentinel tail) materializes."""
    b, _s, j, d = y.shape
    k = jobs.experts_per_token
    per_token = _ep_unsort_rows(shard_axis, y, jobs.sort_idx, jobs.inv_sort_idx).reshape(
        b, jobs.n_shards, j // k, k, d
    )
    weighted = per_token.astype(jnp.float32) * weights.astype(jnp.float32)[:, None, :, :, None]
    combined = jnp.einsum("bstkd->btd", weighted, out_sharding=NamedSharding(_mesh_of(y), out_spec))
    return combined.astype(y.dtype)


def ep_sum_jobs(
    y: Float[Array, "B S J d"], jobs: ExpertShardedJobs, shard_axis: str, out_spec: P
) -> Float[Array, "B T d"]:
    """fp32 sum of each token's k job outputs across expert shards — `ep_combine_jobs`
    without the weights, for consumers that fold routing weights upstream. The shard sum
    is exact (every job is live on exactly one shard and zero elsewhere); `out_spec`
    picks its collective exactly as on `ep_combine_jobs`."""
    b, _s, j, d = y.shape
    k = jobs.experts_per_token
    per_token = _ep_unsort_rows(shard_axis, y, jobs.sort_idx, jobs.inv_sort_idx).reshape(
        b, jobs.n_shards, j // k, k, d
    )
    summed = jnp.einsum(
        "bstkd->btd",
        per_token.astype(jnp.float32),
        out_sharding=NamedSharding(_mesh_of(y), out_spec),
    )
    return summed.astype(y.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ep_sort_slots(shard_axis: str, values: Array, sort_idx: Array, inv_sort_idx: Array) -> Array:
    del inv_sort_idx
    return _ep_gather_replicated_rows(values[..., None], sort_idx, shard_axis)[..., 0]


def _ep_sort_slots_fwd(
    shard_axis: str, values: Array, sort_idx: Array, inv_sort_idx: Array
) -> tuple[Array, Array]:
    return _ep_sort_slots(shard_axis, values, sort_idx, inv_sort_idx), inv_sort_idx


def _ep_sort_slots_bwd(
    shard_axis: str, inv_sort_idx: Array, grad: Array
) -> tuple[Array, None, None]:
    job_major = jnp.take_along_axis(grad, _shard_jobs_ints(inv_sort_idx, shard_axis), axis=-1)
    # the s-sum is the transpose of replicating the values to every expert shard: an
    # all-reduce over the expert-shard axis (dead-tail grads are zero — every job is
    # live on exactly one shard).
    return (
        jnp.einsum(
            "bsj->bj",
            job_major,
            out_sharding=NamedSharding(_mesh_of(grad), P(_batch_entry(grad), None)),
        ),
        None,
        None,
    )


_ep_sort_slots.defvjp(_ep_sort_slots_fwd, _ep_sort_slots_bwd)


def ep_sort_jobs(
    values: Float[Array, "B T k"], jobs: ExpertShardedJobs, shard_axis: str
) -> Float[Array, "B S J"]:
    """Per-(token, slot) job values — the routing weights — in each (batch row, expert
    shard)'s sentinel-sorted job order."""
    assert values.shape == jobs.top_idx.shape, (values.shape, jobs.top_idx.shape)
    flat = values.reshape(values.shape[0], -1)
    return _ep_sort_slots(shard_axis, flat, jobs.sort_idx, jobs.inv_sort_idx)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ep_sort_rows(shard_axis: str, values: Array, sort_idx: Array, inv_sort_idx: Array) -> Array:
    del inv_sort_idx
    return _ep_gather_replicated_rows(values, sort_idx, shard_axis)


def _ep_sort_rows_fwd(
    shard_axis: str, values: Array, sort_idx: Array, inv_sort_idx: Array
) -> tuple[Array, Array]:
    return _ep_sort_rows(shard_axis, values, sort_idx, inv_sort_idx), inv_sort_idx


def _ep_sort_rows_bwd(
    shard_axis: str, inv_sort_idx: Array, grad: Array
) -> tuple[Array, None, None]:
    per_shard = jnp.take_along_axis(
        grad, _shard_jobs_ints(inv_sort_idx, shard_axis)[..., None], axis=2
    )
    # the s-sum is the transpose of replicating the rows to every expert shard: an
    # all-reduce over the expert-shard axis at the cotangent's own dtype (dead-tail
    # grads are zero — every job is live on exactly one shard, so the reduction sums
    # exact zeros against one live partial).
    return (
        jnp.einsum(
            "bsjn->bjn",
            per_shard,
            out_sharding=NamedSharding(_mesh_of(grad), P(_batch_entry(grad), None, None)),
        ),
        None,
        None,
    )


_ep_sort_rows.defvjp(_ep_sort_rows_fwd, _ep_sort_rows_bwd)


def ep_sort_job_values(
    values: Float[Array, "B T k n"], jobs: ExpertShardedJobs, shard_axis: str
) -> Float[Array, "B S J n"]:
    """Per-(token, slot) job ROWS — a narrow mask/CI tensor whose slot m already means
    the token's m-th routed expert — in each (batch row, expert shard)'s sentinel-sorted
    job order: `ep_sort_jobs` with a payload axis. Dead sentinel-tail rows carry real
    (foreign-job) values, not zeros — consumers multiply them into `ep_grouped_matmul`
    outputs, whose dead tail is exactly zero (`ep_gather_tokens` has the same
    contract)."""
    b, t, k, n = values.shape
    assert (b, t, k) == jobs.top_idx.shape, (values.shape, jobs.top_idx.shape)
    return _ep_sort_rows(shard_axis, values.reshape(b, t * k, n), jobs.sort_idx, jobs.inv_sort_idx)


def ep_unsort_jobs(
    y: Float[Array, "B S J d"], jobs: ExpertShardedJobs, shard_axis: str
) -> Float[Array, "B T k d"]:
    """Job-space rows back to token-major (token, slot) layout, fp32-summed across
    expert shards — `ep_sum_jobs` keeping the slot axis: the narrow-emission combine.
    Every job is live on exactly one shard and zero elsewhere, so the shard sum (an
    all-reduce over `shard_axis`) is exact."""
    b, _s, j, d = y.shape
    k = jobs.experts_per_token
    per_token = _ep_unsort_rows(shard_axis, y, jobs.sort_idx, jobs.inv_sort_idx).reshape(
        b, jobs.n_shards, j // k, k, d
    )
    summed = jnp.einsum(
        "bstkd->btkd",
        per_token.astype(jnp.float32),
        out_sharding=NamedSharding(_mesh_of(y), P(_batch_entry(y), None, None, None)),
    )
    return summed.astype(y.dtype)


def ep_gather_job_blocks(
    table: Float[Array, "source_batch source_position E n"],
    jobs: ExpertShardedJobs,
    shard_axis: str,
) -> Float[Array, "B S J n"]:
    """Read each job's source row in expert-owned order, with dead rows zeroed.

    Each source leading axis spans either its token axis or one broadcast entry.
    Position broadcasting changes only gather indices; no full-token table is built. The
    expert axis reshards to `shard_axis` before gathering. Storage may already use
    that layout, replicate experts, or partition within each expert; this boundary
    performs any required redistribution, so each gather reads its own expert rows. A partial gather: the transpose is left to autodiff as the
    scatter-add it honestly is (`gather_job_blocks`)."""
    source_batch, source_position, n_experts, n = table.shape
    b, t, _ = jobs.top_idx.shape
    assert source_batch in (1, b), (table.shape, jobs.top_idx.shape)
    assert source_position in (1, t), (table.shape, jobs.top_idx.shape)
    n_shards = jobs.n_shards
    assert n_experts % n_shards == 0, (n_experts, n_shards)
    e_local = n_experts // n_shards
    assert jobs.group_sizes.shape[-1] == e_local, (jobs.group_sizes.shape, e_local)
    mesh = _mesh_of(table)
    batch = _batch_entry(jobs.top_idx)
    view = jax.sharding.reshard(
        table.reshape(source_batch, source_position, n_shards, e_local, n),
        NamedSharding(mesh, P(_batch_entry(table), None, shard_axis, None, None)),
    )
    rows = view.transpose(0, 2, 1, 3, 4).reshape(
        source_batch, n_shards, source_position * e_local, n
    )
    rows = jnp.broadcast_to(
        rows,
        (b, n_shards, source_position * e_local, n),
        out_sharding=NamedSharding(mesh, P(batch, shard_axis, None, None)),
    )
    sort_idx = _shard_jobs_ints(jobs.sort_idx, shard_axis)
    token_of_job = (sort_idx // jobs.experts_per_token) % source_position
    block_indices = jnp.broadcast_to(
        jobs.top_idx.reshape(b, 1, -1),
        sort_idx.shape,
        out_sharding=NamedSharding(mesh, P(batch, shard_axis, None)),
    )
    expert_of_job = jnp.take_along_axis(block_indices, sort_idx, axis=-1)
    # non-local (sentinel-tail) jobs index another shard's expert mod e_local — an
    # arbitrary in-range row, zeroed below with the rest of the dead tail.
    idx = _shard_jobs_ints(token_of_job * e_local + expert_of_job % e_local, shard_axis)
    gathered = jnp.take_along_axis(rows, idx[..., None], axis=2)
    live = _live_job_mask(jobs.group_sizes, jobs.sort_idx.shape[-1])
    return jnp.where(live[..., None], gathered, 0.0)
