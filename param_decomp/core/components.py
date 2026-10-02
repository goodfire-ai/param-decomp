"""The decomposition representation, shared by every target (LM and toy alike).

`SiteC` / `SiteDims` / `SiteSpec` are the per-site shape primitives (configured name+C,
matrix dimensions, and the combined shape-carrying spec); `Factorization` (`DenseFactorization` |
`BlockedFactorization`) says how a site's V/U factor its matrix, and every consumer whose
behavior depends on the kind matches on it; `ComponentStacks` is the trainable
master pytree, grouped by target-declared semantic role; `init_component_stacks` seeds it.
These are domain-neutral — they depend only on the site shapes and the V/U arrays — so they
live here rather than inside `model.py` (whose `DecomposedModel` Protocol references
`ComponentStacks`/`SiteSpec`) or any one target. Executing a decomposed site is placement's
business, above: `decomposed_linear.site_forward`.
"""

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Generic, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float
from typing_extensions import TypeVar

from param_decomp.core.axes import Axes, SemanticAxis
from param_decomp.core.flops.types import MatrixParameters, ParameterCensus
from param_decomp.core.nonlinearity import (
    ComponentSide,
    DeltaNetHeads,
    KVHeads,
    Neurons,
    NonlinearityAlignment,
    NonlinearityPartition,
    QueryHeads,
)


def activation_axes(ndim: int, feature: SemanticAxis) -> Axes:
    """THE semantic axis names of a waist activation `[batch, *positions, feature]`.
    Placement lookups are exact-name; every consumer derives the tuple here so a
    misspelled feature axis (silent replication) has no second spelling to hide in.
    The waist comes in exactly TWO shapes (`model.py`) — positionless or one position
    axis — so the position vocabulary is the enumeration below, not an open family."""
    match ndim:
        case 2:
            return ("batch", feature)
        case 3:
            return ("batch", "position", feature)
        case _:
            raise AssertionError(ndim)


@dataclass(frozen=True)
class SiteC:
    """A decomposed site as configured: its torch-module-path name and its C.

    The shape-carrying `SiteSpec` is derived from this plus the target's config."""

    name: str
    C: int


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BlockSelection:
    """A target's per-token selection from a table of weight blocks, per layer, as its
    CLEAN forward made it: `indices[l, .., m]` is the m-th block token `..` picked at
    layer `l` (int32, `[n_layer, *leading, k]`) and `weights[l, .., m]` the scalar that
    pick's block output is scaled by (fp32, same shape). It is such a target's `Conditioning`
    (`model.Conditioning`): a masked forward reproduces `indices` at every layer — a
    `SelectedCI` pick m means block `indices[l, .., m]` — and recomputes its own
    `weights` from its own activations; a CI fn scoring on the selection reads both
    halves."""

    indices: Array
    weights: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SelectedCI:
    """One block-factored site's CI (or mask) over the SELECTED blocks only: the picks'
    scores and the block indices that give them meaning, travelling as ONE value.
    `values[.., m·c + j]` scores global component `block_indices[.., m]·c + j`
    (`c = values.shape[-1] // block_indices.shape[-1]`; the site's flat C = `n_blocks·c`).
    `block_indices` is the pinned `BlockSelection`'s slice for the site's layer — the CI
    fn and the target build every bundle from it, and the masked forward keys its blocks
    by the selection itself, never by a bundle — carried here so core's per-SITE
    reductions (imp-min frequencies, source gathers, harvest) stay layer-blind. An
    unselected component's CI is zero by definition — structurally absent, never
    computed — and selected values separated from their block indices are
    unrepresentable: consumers reduce, gather, and shard on this bundle. Dense expert
    computation may expand the selected values to an internal expert table; this does
    not change the selected emission interface."""

    values: Array
    block_indices: Array
    n_blocks: int = field(metadata=dict(static=True))

    @property
    def c_per_block(self) -> int:
        k_c, k = self.values.shape[-1], self.block_indices.shape[-1]
        assert k_c % k == 0, (k_c, k)
        return k_c // k

    @property
    def C(self) -> int:
        return self.n_blocks * self.c_per_block

    def map_values(self, f: "Callable[[Array], Array]") -> "SelectedCI":
        """The same picks under a pointwise map — block indices carried through."""
        return SelectedCI(f(self.values), self.block_indices, self.n_blocks)


SiteCI = Array | SelectedCI
"""One site's CI value at the CI/mask boundary: full emission is the bare `[*leading, C]`
array. Selected emission carries token-ordered picks and their block indices
(`SelectedCI`) on both placed and unplaced paths. Pointwise operations preserve those indices."""


def map_site_ci(f: "Callable[[Array], Array]", value: SiteCI) -> SiteCI:
    """Apply a pointwise map to one site's CI values, whichever emission it carries."""
    match value:
        case SelectedCI():
            return value.map_values(f)
        case jax.Array():
            return f(value)


def site_ci_values(value: SiteCI) -> Array:
    """The values in logical token order, with either full or selected component width."""
    match value:
        case SelectedCI():
            return value.values
        case jax.Array():
            return value


def site_ci_leading(value: SiteCI) -> tuple[int, ...]:
    """One site's waist leading shape, whichever emission it carries."""
    return site_ci_values(value).shape[:-1]


def selected_component_sums(bundle: SelectedCI, data: Array) -> Array:
    """fp32 scatter-sum of per-(token, pick) `data` (shaped like `bundle.values`) onto
    the site's FULL component axis — `out[e·c + j] = Σ_{selected (n, m): ids[n,m]=e}
    data[n, m·c + j]`. Spelled as a one-hot contraction over the (possibly sharded)
    leading axes, with NO leading collapse: the ellipsis contraction is unambiguous
    where a flattening reshape is not (only the unsharded minor pick axis splits, as in
    `block_selection_counts`). Partial sums stay shard-local and reduce globally once, and
    the `[C]` vector exists only as this reduction's output, never as a per-token
    tensor. The contraction is an fp32 scatter-sum spelled as a matmul, so it pins
    HIGHEST precision: the default would run it as a reduced-precision (TF32) dot and
    round the fp32 `data` it exists to sum exactly."""
    assert data.shape == bundle.values.shape, (data.shape, bundle.values.shape)
    k = bundle.block_indices.shape[-1]
    picks = data.astype(jnp.float32).reshape(*data.shape[:-1], k, bundle.c_per_block)
    one_hot = jax.nn.one_hot(bundle.block_indices, bundle.n_blocks, dtype=jnp.float32)
    if jax.sharding.get_abstract_mesh().empty:
        per_block = jnp.einsum(
            "...kc,...ke->ec", picks, one_hot, precision=jax.lax.Precision.HIGHEST
        )
    else:
        # The token contraction spans the dp-sharded lead, so the output's sharding is
        # the contraction's to declare: shard-local partials, one global reduction.
        per_block = jnp.einsum(
            "...kc,...ke->ec",
            picks,
            one_hot,
            precision=jax.lax.Precision.HIGHEST,
            out_sharding=P(None, None),
        )
    return per_block.reshape(bundle.C)


def block_selection_counts(bundle: SelectedCI) -> Array:
    """fp32 per-BLOCK selected-token counts `[n_blocks]` — the complement against the
    leading-axis extent prices the unselected (token, component) pairs, which share one
    count across a block. Ellipsis reduction, no leading collapse: the full sum over
    (possibly sharded) leading axes is unambiguous where a flattening reshape is not."""
    one_hot = jax.nn.one_hot(bundle.block_indices, bundle.n_blocks, dtype=jnp.float32)
    return jnp.einsum("...ke->e", one_hot)


def selected_component_maxes(bundle: SelectedCI, data: Array) -> Array:
    """fp32 segment-max of per-(token, pick) `data` (shaped like `bundle.values`) onto
    the site's FULL component axis — exactly the max over the full-width view, where an
    unselected (token, component) entry is zero: a block some token left unselected takes
    `max(selected, 0)` across its components, and a never-selected block is exactly 0.
    The `[C]` vector exists only as this reduction's output, never as a per-token
    tensor. Under a placed (explicit-sharding) mesh the token flattening keeps the
    leading axis's own spec and the scatter-max declares its replicated output — each
    shard maxes its tokens locally, one cross-shard max combines them."""
    assert data.shape == bundle.values.shape, (data.shape, bundle.values.shape)
    n = math.prod(bundle.values.shape[:-1])
    k = bundle.block_indices.shape[-1]
    data = data.astype(jnp.float32)
    selected_init = jnp.full((bundle.n_blocks, bundle.c_per_block), -jnp.inf, jnp.float32)
    if jax.sharding.get_abstract_mesh().empty:
        flat = data.reshape(n * k, bundle.c_per_block)
        ids = bundle.block_indices.reshape(n * k)
        selected = selected_init.at[ids].max(flat)
    else:
        lead = jax.typeof(data).sharding.spec[0]
        flat = jax.lax.reshape(data, (n * k, bundle.c_per_block), out_sharding=P(lead, None))
        ids = jax.lax.reshape(bundle.block_indices, (n * k,), out_sharding=P(lead))
        # jax's basearray stub lags the runtime signature: `out_sharding` exists on the
        # scatter ops but not in the shipped .pyi.
        selected = selected_init.at[ids].max(flat, out_sharding=P(None, None))  # pyright: ignore[reportCallIssue]
    unselected_somewhere = block_selection_counts(bundle) < n
    full = jnp.where(unselected_somewhere[:, None], jnp.maximum(selected, 0.0), selected)
    return full.reshape(bundle.C)


def require_full_emission(value: SiteCI) -> Array:
    """The full-emission arm of one site's CI. A consumer calling this has no selected
    arm: each call site is an enumerated gap — a selected-emitting site reaching it
    needs dispatch on the `SelectedCI` bundle, never a scatter to `[.., C]`."""
    match value:
        case SelectedCI():
            raise NotImplementedError(
                "this consumer has no selected-emission arm; it must dispatch on the "
                "SelectedCI bundle, not materialize the full component axis"
            )
        case jax.Array():
            return value


COMPONENT_STACK_AXES: Axes = ("stack",)
"""The axis indexing a semantic group's independent V/U matrices."""

DENSE_V_AXES: Axes = ("stack", "d_in", "C")
DENSE_U_AXES: Axes = ("stack", "C", "d_out")
DENSE_DELTA_AXES: Axes = ("stack", "d_out", "d_in")

BLOCKED_V_AXES: Axes = ("stack", "expert", "d_in", "C_block")
BLOCKED_U_AXES: Axes = ("stack", "expert", "C_block", "d_out")
BLOCKED_DELTA_AXES: Axes = ("stack", "expert", "d_out", "d_in")


@dataclass(frozen=True, kw_only=True)
class DenseFactorization:
    """The whole-matrix factorization: the site's `W [d_out, d_in]` is factored as
    `V [d_in, C] @ U [C, d_out]`, and a group of such sites persists as one 3-D stack
    per factor."""

    d_in: int
    d_out: int
    C: int

    @property
    def v_axes(self) -> Axes:
        return DENSE_V_AXES

    @property
    def u_axes(self) -> Axes:
        return DENSE_U_AXES

    @property
    def delta_axes(self) -> Axes:
        return DENSE_DELTA_AXES

    def v_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.d_in, self.C)

    def u_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.C, self.d_out)

    def delta_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.d_out, self.d_in)


@dataclass(frozen=True, kw_only=True)
class BlockedFactorization:
    """The block-local factorization, for a site whose weight matrix is a table of
    `n_blocks` blocks picked per token (`BlockSelection`). Component `(e, j)` reads and
    writes only block `e`: each block gets its own factors `V_e [d_in, c_per_block]` and
    `U_e [c_per_block, d_out]`, and a group of such sites persists as one 4-D stack per
    factor (`[stack, block, ...]`; the placement rows spell the block axis `expert`).
    `d_in` and `d_out` here are the dimensions of ONE block, not of the whole site.
    Which side of the site concatenates the blocks (the output for qwen36_moe's gate/up
    matrices, the input for its down matrices) is deliberately absent: only the forward
    computation needs it, so it is declared by the target where the site's linears are
    built (`linear_plan.BlockContraction`). At the engine's mask/CI boundary the site
    still has one flat component axis of size `C = n_blocks * c_per_block`, ordered
    block-major."""

    n_blocks: int
    d_in: int
    d_out: int
    c_per_block: int

    @property
    def C(self) -> int:
        return self.n_blocks * self.c_per_block

    @property
    def v_axes(self) -> Axes:
        return BLOCKED_V_AXES

    @property
    def u_axes(self) -> Axes:
        return BLOCKED_U_AXES

    @property
    def delta_axes(self) -> Axes:
        return BLOCKED_DELTA_AXES

    def v_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.n_blocks, self.d_in, self.c_per_block)

    def u_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.n_blocks, self.c_per_block, self.d_out)

    def delta_leaf_shape(self, stack_len: int) -> tuple[int, ...]:
        return (stack_len, self.n_blocks, self.d_out, self.d_in)


type Factorization = DenseFactorization | BlockedFactorization
"""How one site's V/U factor its matrix. Every consumer whose behavior depends on the
kind (init, placement transitions, muon labeling, the linear primitive, faithfulness
deltas) matches on this union, so a new kind fails loudly wherever it lacks an arm.
All sites of one semantic group share one factorization (`vu_groups`)."""


@dataclass(frozen=True, kw_only=True)
class SiteDims:
    d_in: int
    d_out: int

    def dense(self, C: int) -> DenseFactorization:
        return DenseFactorization(d_in=self.d_in, d_out=self.d_out, C=C)


@dataclass(frozen=True)
class SiteSpec:
    """One decomposed site: its name, how its V/U factor its matrix, and its
    target-declared persistence group. Alignment describes architectural geometry;
    `None` means neither matrix side directly faces a nonlinearity. The factorization
    is the stored shape truth; `C` (the flat per-site component count) is derived from it, and the dense-only
    `d_in`/`d_out` views assert the site is dense — a block-factored site has no
    single fused view here, because the factorization carries no orientation."""

    name: str
    factorization: Factorization
    group: str
    alignment: NonlinearityAlignment | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        match self.alignment:
            case NonlinearityAlignment(side=side, partition=partition):
                match side:
                    case "input":
                        width = self.factorization.d_in
                    case "output":
                        width = self.factorization.d_out
                match partition:
                    case (
                        QueryHeads(head_count=head_count)
                        | KVHeads(head_count=head_count)
                        | DeltaNetHeads(head_count=head_count)
                    ):
                        assert width % head_count == 0, self
                    case Neurons():
                        pass
            case None:
                pass

    @property
    def C(self) -> int:
        return self.factorization.C

    @property
    def d_in(self) -> int:
        assert isinstance(self.factorization, DenseFactorization), self
        return self.factorization.d_in

    @property
    def d_out(self) -> int:
        assert isinstance(self.factorization, DenseFactorization), self
        return self.factorization.d_out


def component_parameters(sites: tuple[SiteSpec, ...]) -> ParameterCensus:
    """Logical V/U parameter shapes, before placement padding or replication."""
    matrices: list[MatrixParameters] = []
    for site in sites:
        match site.factorization:
            case DenseFactorization(d_in=d_in, d_out=d_out, C=n_components):
                n_copies = 1
            case BlockedFactorization(
                d_in=d_in, d_out=d_out, c_per_block=n_components, n_blocks=n_copies
            ):
                pass
        matrices.extend(
            (
                MatrixParameters(d_in, n_components, n_copies),
                MatrixParameters(n_components, d_out, n_copies),
            )
        )
    return ParameterCensus(tuple(matrices), 0)


def nonlinearity_partitions(sites: tuple[SiteSpec, ...]) -> dict[str, NonlinearityPartition]:
    return {name: alignment.partition for name, alignment in nonlinearity_alignments(sites).items()}


def nonlinearity_alignments(sites: tuple[SiteSpec, ...]) -> dict[str, NonlinearityAlignment]:
    return {s.name: s.alignment for s in sites if s.alignment is not None}


def aligned_component_vectors(factors: tuple[Array, Array], side: ComponentSide) -> Array:
    """Read or write vectors in the common `[..., C, coordinate]` orientation."""
    v, u = factors
    match side:
        case "input":
            return jnp.swapaxes(v, -1, -2)
        case "output":
            return u


@dataclass(frozen=True)
class SiteComponents:
    """The two rank-one factor matrices for one decomposed site."""

    V: Array
    U: Array


# site name -> (target-declared group, index on the group's stack axis)
SiteStackIndices = tuple[tuple[str, str, int], ...]

# The V/U leaf type: `Array` for the real fp32 masters (the default — so bare `ComponentStacks`
# means `ComponentStacks[Array]` and no call site needs the parameter), or `NamedSharding` for
# the same-structure placement tree `placement.component_stacks_shardings` returns for
# `jax.jit(out_shardings=...)`.
VULeaf = TypeVar("VULeaf", default=Array)


@dataclass(frozen=True)
class VUGroup:
    """One semantic persistence group: its sites in canonical order, and the one
    factorization they all share. The factorization lives here — established once, when
    the grouping is built — so no consumer ever has to re-derive it from a member."""

    factorization: Factorization
    specs: tuple[SiteSpec, ...]


def vu_groups(sites: tuple[SiteSpec, ...]) -> dict[str, VUGroup]:
    """Sites grouped by the target's semantic persistence group. A group's sites must
    all share one factorization, because they persist along the stack axis of one
    homogeneous stack."""
    grouped: dict[str, list[SiteSpec]] = {}
    for spec in sites:
        grouped.setdefault(spec.group, []).append(spec)
    groups: dict[str, VUGroup] = {}
    for name, specs in grouped.items():
        factorizations = {spec.factorization for spec in specs}
        assert len(factorizations) == 1, (
            f"component group {name!r} mixes factorizations: {factorizations}"
        )
        groups[name] = VUGroup(factorization=factorizations.pop(), specs=tuple(specs))
    return groups


def group_factorizations(sites: tuple[SiteSpec, ...]) -> dict[str, Factorization]:
    """Each semantic group's factorization, keyed by group name."""
    return {name: group.factorization for name, group in vu_groups(sites).items()}


def site_stack_indices_for(sites: tuple[SiteSpec, ...]) -> SiteStackIndices:
    """The canonical site→(group, stack index) mapping in site order."""
    by_name: dict[str, tuple[str, int]] = {}
    for name, group in vu_groups(sites).items():
        for stack_index, spec in enumerate(group.specs):
            by_name[spec.name] = (name, stack_index)
    return tuple((spec.name, *by_name[spec.name]) for spec in sites)


@cache
def stack_index_by_site(site_stack_indices: SiteStackIndices) -> dict[str, tuple[str, int]]:
    """site name -> (group, stack index), cached per `SiteStackIndices` value."""
    return {name: (group, index) for name, group, index in site_stack_indices}


class ComponentStacks(eqx.Module, Generic[VULeaf]):
    """The trainable V/U masters: one homogeneous stack per target-declared semantic group.

    A group holds `(Vs [g, d_in, C], Us [g, C, d_out])`; `site_stack_indices` maps each
    site to its index on the stack axis. LM targets declare matrix kind as the group,
    making each scan input a leaf.
    Toy targets may declare independent per-site groups. Placement is separate: a rule may
    shard the stack axis for ownership or shard matrix dimensions instead.

    `stack_pads` enumerates the persist-layer PADS: a stack-sharding placement whose
    extent the real stack length does not tile pads each stack with trailing all-zero
    matrices (`pad_component_stacks`), and this field carries that fact as data — one
    count per group with a nonzero pad, `()` = unpadded — so no consumer ever infers a
    pad from a shape. Pads exist only between the persist layer and the entry boundaries
    that strip them; `site_stack_indices` never indexes them.

    Leaves are fp32 master Arrays (`ComponentStacks[Array]`) or `NamedSharding`s in the
    same-structure placement tree `placement.component_stacks_shardings` returns
    (`ComponentStacks[NamedSharding]`). This module is placement-FREE: the per-group row
    lookup and its boundary validation live in `placement.py`, above."""

    stacks: dict[str, tuple[VULeaf, VULeaf]]
    site_stack_indices: SiteStackIndices = eqx.field(static=True)
    stack_pads: tuple[tuple[str, int], ...] = eqx.field(static=True, default=())

    def __check_init__(self) -> None:
        groups = {group for _, group, _ in self.site_stack_indices}
        assert all(group in groups and pad > 0 for group, pad in self.stack_pads), (
            self.stack_pads,
            sorted(groups),
        )

    def pad_of(self, group: str) -> int:
        return dict(self.stack_pads).get(group, 0)

    def stack_index_of(self, name: str) -> tuple[str, int]:
        return stack_index_by_site(self.site_stack_indices)[name]

    def site(self: "ComponentStacks[Array]", name: str) -> SiteComponents:
        group, index = self.stack_index_of(name)
        Vs, Us = self.stacks[group]
        return SiteComponents(V=Vs[index], U=Us[index])

    def component_activations(
        self: "ComponentStacks[Array]", site: str, x: Float[Array, "*leading d_in"]
    ) -> Float[Array, "*leading C"]:
        """`x @ V` for a dense site, as a target whose prepared weights are these stacks
        computes it."""
        V = self.site(site).V
        assert V.ndim == 2, f"{site} is block-factored: its V {V.shape} has a block axis"
        return x @ V

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.site_stack_indices)

    def sites_items(self: "ComponentStacks[Array]") -> Iterator[tuple[str, SiteComponents]]:
        """Named site components in canonical site order."""
        for name, _, _ in self.site_stack_indices:
            yield name, self.site(name)

    def group_lengths(self) -> dict[str, int]:
        """Stack length per semantic group, available from eval-shape trees."""
        lengths: dict[str, int] = {}
        for _name, group, index in self.site_stack_indices:
            lengths[group] = max(lengths.get(group, 0), index + 1)
        return lengths


def pad_component_stacks(
    stacks: "ComponentStacks[Array]", pads: "Mapping[str, int]"
) -> "ComponentStacks[Array]":
    """Append each group's declared pad count as trailing ALL-ZERO matrices — the one
    constructor of a padded persist tree. Pads are enumerated on `stack_pads`, never
    inferred; entries of 0 are dropped so `()` stays the single spelling of unpadded."""
    assert stacks.stack_pads == (), f"already padded: {stacks.stack_pads}"
    assert pads.keys() <= stacks.stacks.keys(), (sorted(pads), sorted(stacks.stacks))
    padded: dict[str, tuple[Array, Array]] = {}
    for group, (Vs, Us) in stacks.stacks.items():
        pad = pads.get(group, 0)
        if pad == 0:
            padded[group] = (Vs, Us)
            continue
        padded[group] = (
            jnp.concatenate(
                [
                    Vs,
                    jnp.zeros_like(
                        Vs, shape=(pad, *Vs.shape[1:]), out_sharding=jax.typeof(Vs).sharding
                    ),
                ]
            ),
            jnp.concatenate(
                [
                    Us,
                    jnp.zeros_like(
                        Us, shape=(pad, *Us.shape[1:]), out_sharding=jax.typeof(Us).sharding
                    ),
                ]
            ),
        )
    return ComponentStacks(
        stacks=padded,
        site_stack_indices=stacks.site_stack_indices,
        stack_pads=tuple((group, pad) for group, pad in pads.items() if pad > 0),
    )


BLOCKED_U_INIT_FAN_IN: Literal["site", "block"] = "site"
"""Which fan-in sets a block-factored U's init scale. `"site"` draws
`U_e ~ N(0, C^-1/2)` with `C = n_blocks * c_per_block`: a site whose output SUMS the
blocks (the down orientation) then starts with the same output variance as a dense
site. `"block"` is the live alternative — the dense rule applied to each block's own
fan-in, `U_e ~ N(0, c_per_block^-1/2)`, which instead variance-matches a site whose
output CONCATENATES the blocks (the gate/up orientation). The factorization
deliberately carries no orientation, so one choice applies to every block's U; flipping
this constant is the whole change."""


def init_stack_arrays(sites: tuple[SiteSpec, ...], key: Array) -> dict[str, tuple[Array, Array]]:
    """Seed each semantic group's V/U stacks, drawing one (V, U) key pair per site in
    site order. V scales by its fan-in (`d_in^-1/2`, the per-block `d_in` for
    block-factored groups); U scales by `C^-1/2` for dense groups and by
    `BLOCKED_U_INIT_FAN_IN`'s choice for block-factored groups."""
    site_keys = jax.random.split(key, (len(sites), 2))
    site_index = {spec.name: idx for idx, spec in enumerate(sites)}
    stacked: dict[str, tuple[Array, Array]] = {}
    for name, group in vu_groups(sites).items():
        idxs = jnp.array([site_index[spec.name] for spec in group.specs])
        v_keys, u_keys = site_keys[idxs, 0], site_keys[idxs, 1]
        match group.factorization:
            case DenseFactorization(d_in=d_in, d_out=d_out, C=c):
                Vs = jax.vmap(lambda k, s=(d_in, c): jax.random.normal(k, s))(v_keys)
                Us = jax.vmap(lambda k, s=(c, d_out): jax.random.normal(k, s))(u_keys)
                stacked[name] = (Vs * d_in**-0.5, Us * c**-0.5)
            case (
                BlockedFactorization(n_blocks=n_blocks, d_in=d_in, d_out=d_out, c_per_block=c) as f
            ):
                Vs = jax.vmap(lambda k, s=(n_blocks, d_in, c): jax.random.normal(k, s))(v_keys)
                Us = jax.vmap(lambda k, s=(n_blocks, c, d_out): jax.random.normal(k, s))(u_keys)
                match BLOCKED_U_INIT_FAN_IN:
                    case "site":
                        u_fan_in = f.C
                    case "block":
                        u_fan_in = c
                stacked[name] = (Vs * d_in**-0.5, Us * u_fan_in**-0.5)
    return stacked


def component_stacks_from_site_arrays(
    sites: tuple[SiteSpec, ...], vu: dict[str, tuple[Array, Array]]
) -> ComponentStacks:
    assert tuple(vu) == tuple(spec.name for spec in sites), (tuple(vu), sites)
    stacks = {
        name: (
            jnp.stack([vu[spec.name][0] for spec in group.specs]),
            jnp.stack([vu[spec.name][1] for spec in group.specs]),
        )
        for name, group in vu_groups(sites).items()
    }
    return ComponentStacks(stacks=stacks, site_stack_indices=site_stack_indices_for(sites))


def component_stacks_from_sites(vu: dict[str, tuple[Array, Array]]) -> ComponentStacks:
    """Build independently grouped component leaves from explicit per-site arrays."""
    sites = tuple(
        SiteSpec(
            name=name,
            factorization=DenseFactorization(d_in=V.shape[0], d_out=U.shape[1], C=V.shape[1]),
            group=name,
        )
        for name, (V, U) in vu.items()
    )
    return component_stacks_from_site_arrays(sites, vu)


def init_component_stacks(sites: tuple[SiteSpec, ...], key: Array) -> ComponentStacks:
    """Small random fp32 V/U per site at the scales `init_stack_arrays` documents, built
    directly in the stacked persistence layout; the weight-delta channel carries the
    faithfulness residual at init (before faithfulness warmup)."""
    return ComponentStacks(
        stacks=init_stack_arrays(sites, key), site_stack_indices=site_stack_indices_for(sites)
    )
