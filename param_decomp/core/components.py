"""The decomposition representation, shared by every target (LM and toy alike).

`SiteC` / `SiteDims` / `SiteSpec` are the per-site shape primitives (configured name+C,
matrix dimensions, and the combined shape-carrying spec); `ComponentStacks` is the trainable
master pytree, grouped by target-declared semantic role; `init_component_stacks` seeds it.
These are domain-neutral — they depend only on the site shapes and the V/U arrays — so they
live here rather than inside `model.py` (whose `DecomposedModel` Protocol references
`ComponentStacks`/`SiteSpec`) or any one target. Executing a decomposed site is placement's
business, above: `decomposed_linear.site_forward`.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from typing import ClassVar, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import TypeVar

from param_decomp.core.axes import Axes, SemanticAxis
from param_decomp.core.nonlinearity import (
    KVHeads,
    Neurons,
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


@dataclass(frozen=True, kw_only=True)
class SiteDims:
    d_in: int
    d_out: int


@dataclass(frozen=True)
class SiteSpec:
    name: str
    d_in: int
    d_out: int
    C: int
    group: str
    nonlinearity_partition: NonlinearityPartition | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        match self.nonlinearity_partition:
            case QueryHeads(head_count=head_count) | KVHeads(head_count=head_count):
                assert self.d_out % head_count == 0, self
            case Neurons() | None:
                pass


def nonlinearity_partitions(sites: tuple[SiteSpec, ...]) -> dict[str, NonlinearityPartition]:
    return {s.name: s.nonlinearity_partition for s in sites if s.nonlinearity_partition is not None}


@dataclass(frozen=True)
class SiteComponents:
    """The two rank-one factor matrices for one decomposed site."""

    V: Array
    U: Array


VUShape = tuple[int, int, int]  # (d_in, d_out, C)

# site name -> (target-declared group, slot on the group's stack axis)
SiteSlots = tuple[tuple[str, str, int], ...]

# The V/U leaf type: `Array` for the real fp32 masters (the default — so bare `ComponentStacks`
# means `ComponentStacks[Array]` and no call site needs the parameter), or `NamedSharding` for
# the same-structure placement tree `placement.component_stacks_shardings` returns for
# `jax.jit(out_shardings=...)`.
VULeaf = TypeVar("VULeaf", default=Array)


def vu_groups(sites: tuple[SiteSpec, ...]) -> dict[str, tuple[SiteSpec, ...]]:
    """Sites grouped by the target's semantic persistence group."""
    groups: dict[str, list[SiteSpec]] = {}
    for spec in sites:
        groups.setdefault(spec.group, []).append(spec)
    for group, specs in groups.items():
        shapes = {(spec.d_in, spec.d_out, spec.C) for spec in specs}
        assert len(shapes) == 1, f"component group {group!r} mixes V/U shapes: {sorted(shapes)}"
    return {group: tuple(specs) for group, specs in groups.items()}


def group_shape(specs: tuple[SiteSpec, ...]) -> VUShape:
    assert specs
    return (specs[0].d_in, specs[0].d_out, specs[0].C)


def site_slots_for(sites: tuple[SiteSpec, ...]) -> SiteSlots:
    """The canonical site→(group, slot) mapping in site order."""
    by_name: dict[str, tuple[str, int]] = {}
    for group, specs in vu_groups(sites).items():
        for slot, spec in enumerate(specs):
            by_name[spec.name] = (group, slot)
    return tuple((spec.name, *by_name[spec.name]) for spec in sites)


@cache
def _slot_index(site_slots: SiteSlots) -> dict[str, tuple[str, int]]:
    return {name: (group, slot) for name, group, slot in site_slots}


class ComponentStacks(eqx.Module, Generic[VULeaf]):
    """The trainable V/U masters: one homogeneous stack per target-declared semantic group.

    A group holds `(Vs [g, d_in, C], Us [g, C, d_out])`; `site_slots` maps each site to
    its slot. LM targets declare matrix kind as the group, making each scan input a leaf.
    Toy targets may declare independent per-site groups. Placement is separate: a rule may
    shard the stack axis for ownership or shard matrix dimensions instead.

    Leaves are fp32 master Arrays (`ComponentStacks[Array]`) or `NamedSharding`s in the
    same-structure placement tree `placement.component_stacks_shardings` returns
    (`ComponentStacks[NamedSharding]`). This module is placement-FREE: the per-group row
    lookup and its boundary validation live in `placement.py`, above."""

    stacks: dict[str, tuple[VULeaf, VULeaf]]
    site_slots: SiteSlots = eqx.field(static=True)

    def slot_of(self, name: str) -> tuple[str, int]:
        return _slot_index(self.site_slots)[name]

    def site(self: "ComponentStacks[Array]", name: str) -> SiteComponents:
        group, slot = self.slot_of(name)
        Vs, Us = self.stacks[group]
        return SiteComponents(V=Vs[slot], U=Us[slot])

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.site_slots)

    def sites_items(self: "ComponentStacks[Array]") -> Iterator[tuple[str, SiteComponents]]:
        """Named site components in canonical site order."""
        for name, _, _ in self.site_slots:
            yield name, self.site(name)

    V_AXES: ClassVar[Axes] = ("stack", "d_in", "C")
    U_AXES: ClassVar[Axes] = ("stack", "C", "d_out")

    def group_lengths(self) -> dict[str, int]:
        """Stack length per semantic group, available from eval-shape trees."""
        lengths: dict[str, int] = {}
        for _name, group, slot in self.site_slots:
            lengths[group] = max(lengths.get(group, 0), slot + 1)
        return lengths


def init_stack_arrays(sites: tuple[SiteSpec, ...], key: Array) -> dict[str, tuple[Array, Array]]:
    """Seed each semantic group's V/U stacks over per-site keys drawn in site order."""
    keys = jax.random.split(key, 2 * len(sites))
    site_index = {spec.name: idx for idx, spec in enumerate(sites)}
    stacked: dict[str, tuple[Array, Array]] = {}
    for group, specs in vu_groups(sites).items():
        d_in, d_out, c = group_shape(specs)
        idxs = jnp.array([site_index[spec.name] for spec in specs])
        Vs = jax.vmap(lambda k, s=(d_in, c): jax.random.normal(k, s))(keys[2 * idxs])
        Us = jax.vmap(lambda k, s=(c, d_out): jax.random.normal(k, s))(keys[2 * idxs + 1])
        stacked[group] = (Vs * d_in**-0.5, Us * c**-0.5)
    return stacked


def component_stacks_from_site_arrays(
    sites: tuple[SiteSpec, ...], vu: dict[str, tuple[Array, Array]]
) -> ComponentStacks:
    assert tuple(vu) == tuple(spec.name for spec in sites), (tuple(vu), sites)
    stacks = {
        group: (
            jnp.stack([vu[spec.name][0] for spec in specs]),
            jnp.stack([vu[spec.name][1] for spec in specs]),
        )
        for group, specs in vu_groups(sites).items()
    }
    return ComponentStacks(stacks=stacks, site_slots=site_slots_for(sites))


def component_stacks_from_sites(vu: dict[str, tuple[Array, Array]]) -> ComponentStacks:
    """Build independently grouped component leaves from explicit per-site arrays."""
    sites = tuple(
        SiteSpec(name=name, d_in=V.shape[0], d_out=U.shape[1], C=V.shape[1], group=name)
        for name, (V, U) in vu.items()
    )
    return component_stacks_from_site_arrays(sites, vu)


def init_component_stacks(sites: tuple[SiteSpec, ...], key: Array) -> ComponentStacks:
    """Small random fp32 V ~ N(0, d_in^-0.5), U ~ N(0, C^-0.5) per site, built directly in
    the stacked persistence layout; the weight-delta channel carries the faithfulness
    residual at init (before faithfulness warmup)."""
    return ComponentStacks(stacks=init_stack_arrays(sites, key), site_slots=site_slots_for(sites))
