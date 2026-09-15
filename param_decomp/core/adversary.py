"""Adversarial source state, initialization, and optimization.

Two semantically distinct adversaries share source initialization and optimization but
nothing else:

- **Persistent PGD (PPGD)** — `PersistentPGDReconLossConfig`. The sources + their
  SRC_STEP optimizer state live in `TrainState` across steps, persisted as
  target-declared semantic stacks (`SourceStacks`, the grouping `ComponentStacks` uses)
  and shaped per `source_shape` (`configs.SourceShape`); every consumer reads the
  site-keyed `per_site()` view. SRC_STEP is a closed enumeration: `adam` carries
  coordinate moments (`SourcesAdamState`); `sgd` is the stateless plain ascent
  (`SourcesSgdState`, no leaves) — the arm whose persistent state is the sources alone;
  `momentum_sgd` carries one float velocity buffer (`SourcesMomentumState`).
  Source VALUES are stored per `source_dtype` — a float dtype, or the uint16
  unit-interval fixed-point representation (`UINT16_UNIT_SCALE`): ascents differentiate
  and update the float view and store back with stochastic rounding.
  Each step runs `n_warmup_steps` supplemental ascents plus one final ascent from
  the main backward, projecting to [0,1] after every update.
- **Fresh PGD** — `PGDReconLossConfig` (torch `PGDReconLoss` as a TRAINING loss).
  Sources are re-initialized every step, ascended `n_steps` times by
  `step_size * sign(grad)` with clamp to [0,1], and carry NO state across steps —
  `TrainState.adversaries` stays empty for this variant.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PRNGKeyArray
from typing_extensions import TypeVar

from param_decomp.core.components import (
    Dense,
    ExpertBlocked,
    Factorization,
    SiteSlots,
    SiteSpec,
    site_slots_for,
    slot_index,
    vu_groups,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    MomentumSgdPGDConfig,
    PGDInitStrategy,
    SgdPGDConfig,
    SourceShape,
)
from param_decomp.core.linear_plan import uniform_like
from param_decomp.core.losses import scheduled_value_at


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ExpertBlockedSource:
    """An expert-blocked site's component sources in BLOCK dim: `values[.., e, j]`
    sources component `(e, j)` directly — no flat-offset arithmetic anywhere. Mask
    formation reads either the flat expert-major view (`flat` — the FULL-emission arm,
    bit-identical to the dense spelling) or a take along the block axis by the token's
    router indices (the narrow-emission arm, landing with the routed-decomposed masked
    forward). Shards `expert: tp` like every other expert-shaped tensor — the same
    bytes-per-rank as the flat expert-major `C: tp` split whenever tp | E."""

    values: Float[Array, "*leading E c"]

    @property
    def flat(self) -> Float[Array, "*leading C"]:
        return self.values.reshape(*self.values.shape[:-2], -1)


SourceComponents = Float[Array, "*leading C"] | ExpertBlockedSource
"""One site's component sources: dense sites carry the bare flat array, expert-blocked
sites the block-dim bundle — discriminated the same way CI values are (`SiteCI`)."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SiteSource:
    """ONE site's adversarial sources, site-shaped (`[*leading, ..]`, no slot axis): the
    component sources plus the EXPLICIT weight-delta source — its own tensor, never a
    column inside the components. The stacked persistence shape is `SourceStack`."""

    components: SourceComponents
    delta: Array


type Sources = dict[str, SiteSource]
"""The site-keyed CONSUMER view of adversarial sources — what mask formation reads.
Persistent sources are stored as `SourceStacks` and viewed through `per_site()`; fresh
PGD sources are drawn directly in this form."""


# The stored-leaf type: `Array` for the resident sources (the default — bare `SourceStacks`
# means `SourceStacks[Array]`), or `NamedSharding` for the same-structure placement tree
# `init_placed.persistent_sources_shardings` returns.
SourceLeaf = TypeVar("SourceLeaf", default=Array)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourceStack(Generic[SourceLeaf]):
    """One semantic GROUP's sources, slot-major over its member sites — the stacked
    counterpart of `SiteSource`, distinct in type so a stack can never pass for a site
    view: component sources `[slot, *leading, C]` (dense) or
    `ExpertBlockedSource([slot, *leading, E, c])`, delta sources `[slot, *leading]`.
    The name says what `ComponentStacks.stacks[group]` is — one group's stack; unlike
    that bare `(Vs, Us)` tuple the fields are named, because they differ in rank."""

    components: SourceLeaf | ExpertBlockedSource
    delta: SourceLeaf


class SourceStacks(eqx.Module, Generic[SourceLeaf]):
    """Persistent sources stacked per target-declared semantic group — the SAME grouping
    `ComponentStacks` persists under (`SiteSpec.group`; slot = the site's order within
    its group, so for the LM families group = matrix kind and slot = layer). One group
    rides one homogeneous `SourceStack`. Slot-major keeps a stage-blocked reshape of the
    slot axis a view.

    `site()` is the ONE read boundary (`per_site()` is the whole-dict spelling of it):
    its slot slices are views, and slicing commutes with the elementwise dequant, so
    site views of the float view are bit-equal to per-site float sources. The SRC_STEP
    state (velocity / moments) and the checkpoint tree mirror this layout by `tree.map`.
    Leaves are stored Arrays or the same-structure `NamedSharding` tree
    `init_placed.persistent_sources_shardings` returns (`SourceStacks[NamedSharding]`).

    No `stack_pads` counterpart: the slot axis is never sharded for sources (the batch
    axis rides `data`, C / the expert axis rides `tp`, the slot axis replicates), so the
    persist-layer padding `ComponentStacks` enumerates never arises here."""

    stacks: dict[str, SourceStack[SourceLeaf]]
    site_slots: SiteSlots = eqx.field(static=True)

    def slot_of(self, name: str) -> tuple[str, int]:
        return slot_index(self.site_slots)[name]

    def site(self: "SourceStacks[Array]", name: str) -> SiteSource:
        group, slot = self.slot_of(name)
        stack = self.stacks[group]
        match stack.components:
            case ExpertBlockedSource(values=values):
                components: SourceComponents = ExpertBlockedSource(values=values[slot])
            case jax.Array():
                components = stack.components[slot]
        return SiteSource(components=components, delta=stack.delta[slot])

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.site_slots)

    def per_site(self: "SourceStacks[Array]") -> Sources:
        return {name: self.site(name) for name in self.site_names}


UINT16_UNIT_SCALE = 65535.0
"""The uint16 fixed-point representation of a unit-interval value: `value = u / 65535`.
Sources live in [0,1] by construction (projection), so the representation is
exact at the interval's ends and uniformly 1/65535 everywhere — bf16 resolves only
~2^-8 near 1.0. The stored integer is NOT differentiable; every consumer reads the
float view (`source_values_to_float`) and every store quantizes back."""


def store_unit_float(value: Array, storage_dtype: DTypeLike) -> Array:
    """A unit-interval float landing in its storage representation: float dtypes cast,
    uint16 quantizes round-to-nearest (init-time only — ascents store through
    `_quantize_unit_stochastic`, whose rounding must be unbiased)."""
    dtype = jnp.dtype(storage_dtype)
    if dtype == jnp.uint16:
        return jnp.round(value.astype(jnp.float32) * UINT16_UNIT_SCALE).astype(jnp.uint16)
    return value.astype(dtype)


def source_values_to_float[T](sources: T) -> T:
    """The differentiable float view of stored sources (any source tree): uint16
    fixed-point dequantizes to fp32 (`u / 65535`); float storage passes through as the
    SAME arrays (bit-identical leaves, so float-stored trajectories are untouched by the
    representation seam)."""

    def view(leaf: Array) -> Array:
        if leaf.dtype == jnp.uint16:
            return leaf.astype(jnp.float32) / UINT16_UNIT_SCALE
        return leaf

    return jax.tree.map(view, sources)


def _quantize_unit_stochastic(value: Array, key: PRNGKeyArray) -> Array:
    """[0,1] float -> uint16 fixed-point with STOCHASTIC rounding (unbiased: an update
    smaller than half a step still lands in expectation — round-to-nearest would stall
    every |Δ| < 1/(2·65535) forever) and saturating ends (composes exactly with the
    projection). The rounding draw is typed to the value's sharding: a bare draw
    lowers REPLICATED under the Explicit mesh, every rank forming the global-shape bits."""
    scaled = jnp.clip(value.astype(jnp.float32), 0.0, 1.0) * UINT16_UNIT_SCALE
    low = jnp.floor(scaled)
    rounded_up = uniform_like(key, scaled) < (scaled - low)
    return jnp.clip(low + rounded_up, 0.0, UINT16_UNIT_SCALE).astype(jnp.uint16)


def store_sources[T](stored: T, new_values: T, key: PRNGKeyArray) -> T:
    """Projected float source values landing back in `stored`'s representation — the
    single write-back seam: float leaves pass through (the arms already produced them at
    storage dtype), uint16 leaves quantize with stochastic rounding under per-leaf folds
    of `key` (leaf index in `tree_leaves` order, so the stream is a function of the
    container's leaf enumeration)."""
    leaves, treedef = jax.tree_util.tree_flatten(stored)
    new_leaves = jax.tree_util.tree_leaves(new_values)
    out = [
        _quantize_unit_stochastic(new, random.fold_in(key, idx)) if old.dtype == jnp.uint16 else new
        for idx, (old, new) in enumerate(zip(leaves, new_leaves, strict=True))
    ]
    return jax.tree_util.tree_unflatten(treedef, out)


def full_source_components(components: SourceComponents) -> Float[Array, "*leading C"]:
    """The flat expert-major `[.., C]` view — the FULL-emission mask arm's read (the
    narrow arm takes along the block axis by router indices instead,
    `masking._narrow_source_values`)."""
    match components:
        case ExpertBlockedSource():
            return components.flat
        case jax.Array():
            return components


def _draw_site_source(
    site_key: PRNGKeyArray, factorization: Factorization, leading_shape: tuple[int, ...]
) -> SiteSource:
    """One site's U[0,1] sources drawn directly in their honest shapes — block-dim
    `[.., E, c]` values for expert-blocked sites, flat `[.., C]` for dense, the delta
    as its own tensor. Fp32 draws; callers cast to the storage dtype per field.
    `init_persistent_sources` vmaps THIS function over per-site keys, so each slot of a
    stack is bit-identical to the standalone per-site draw (the RNG-chain pin)."""
    components_key, delta_key = random.split(site_key)
    delta = random.uniform(delta_key, leading_shape, jnp.float32)
    match factorization:
        case ExpertBlocked(n_experts=n_experts, c_per_expert=c_per_expert):
            values = random.uniform(
                components_key, (*leading_shape, n_experts, c_per_expert), jnp.float32
            )
            return SiteSource(components=ExpertBlockedSource(values=values), delta=delta)
        case Dense(C=c):
            return SiteSource(
                components=random.uniform(components_key, (*leading_shape, c), jnp.float32),
                delta=delta,
            )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcesAdamState:
    m: SourceStacks
    v: SourceStacks
    step_count: Float[Array, ""]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcesSgdState:
    """The stateless SRC_STEP's optimizer state: a typed marker with no leaves, so the
    persistent bundle checkpoints and shards as the sources alone."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcesMomentumState:
    """SRC_STEP `momentum_sgd`'s one buffer: the velocity, a float tree mirroring the
    source stacks (same shapes, same shardings) at `velocity_dtype` of the storage —
    SIGNED, so it never takes the uint16 unit-interval representation."""

    velocity: SourceStacks


def velocity_dtype(source_storage_dtype: DTypeLike) -> jnp.dtype:
    """The momentum buffer's float dtype for a source storage dtype: fp32 storage keeps
    fp32 velocity; both 16-bit storages (bf16, uint16 fixed-point) ride bf16 velocity."""
    dtype = jnp.dtype(source_storage_dtype)
    return jnp.dtype(jnp.float32) if dtype == jnp.float32 else jnp.dtype(jnp.bfloat16)


SourcesOptState = SourcesAdamState | SourcesSgdState | SourcesMomentumState

SourceOptimizerConfig = AdamPGDConfig | SgdPGDConfig | MomentumSgdPGDConfig
"""The SRC_STEP enumeration, paired with its state by `init_sources_opt_state`
and re-matched at every ascent — a config/state mismatch dies in `sources_ascend_project`."""


def init_persistent_sources(
    sites: tuple[SiteSpec, ...],
    leading_shape: tuple[int, ...],
    source_dtype: DTypeLike,
    key: PRNGKeyArray,
) -> SourceStacks:
    """PPGD component and weight-delta sources, initialized U[0,1] into their semantic
    stacks. Keys split per SITE (in site order) and each group vmaps `_draw_site_source`
    over its members' keys, so slot `j` of a stack is bit-identical to the standalone
    per-site draw under `keys[site_index]` — and the jitted graph has n_groups sharded
    outputs, not n_sites (the per-site form was a ~55s XLA compile at 224 sites).

    `leading_shape` spells the `source_shape` over the model's leading axes,
    rank matching the waist with size-1 broadcast axes — e.g. an LM's `(1, T)` for `sc`
    (shared across batch, free per position), `(B, 1)` for `bc` (per batch element,
    shared over positions).

    `source_dtype` is the resident storage dtype (fp32 for oracle parity; bf16 to
    halve footprint). Drawing in fp32 then casting keeps the U[0,1] draw dtype-stable."""
    keys = random.split(key, len(sites))
    site_index = {site.name: idx for idx, site in enumerate(sites)}
    stacks: dict[str, SourceStack] = {}
    for group, members in vu_groups(sites).items():
        member_keys = keys[jnp.array([site_index[spec.name] for spec in members.specs])]
        # vmap leaves the stacked draws inside the site-shaped container; rewrap
        drawn = jax.vmap(lambda k, f=members.factorization: _draw_site_source(k, f, leading_shape))(
            member_keys
        )
        stored = jax.tree.map(lambda a: store_unit_float(a, source_dtype), drawn)
        stacks[group] = SourceStack(components=stored.components, delta=stored.delta)
    return SourceStacks(stacks=stacks, site_slots=site_slots_for(sites))


def init_fresh_pgd_sources(
    sites: tuple[SiteSpec, ...],
    init: PGDInitStrategy,
    source_shape: SourceShape,
    leading: tuple[int, ...],
    key: PRNGKeyArray,
) -> Sources:
    """Per-site fresh adversarial component and weight-delta sources.

    `leading = (B,) + position axes`. The source's leading
    shape spells `source_shape` (`configs.SourceShape`) over those axes: `bsc` keeps the
    full leading, `bc` collapses every position axis to 1 (`(B, 1)`), `c` collapses every
    axis to 1 (`(1, 1)`). The persistent counterpart enumerates its config-time shapes in
    `init_sources_sharded`."""
    batch, *positions = leading
    match source_shape:
        case "bsc":
            source_leading = leading
        case "bc":
            source_leading = (batch, *(1 for _ in positions))
        case "c":
            source_leading = tuple(1 for _ in leading)
        case "sc":
            raise AssertionError("unreachable: PGDConfig validation rejects `sc`")
    keys = random.split(key, len(sites))
    sources: Sources = {}
    for site, site_key in zip(sites, keys, strict=True):
        match site.factorization:
            case ExpertBlocked(n_experts=n_experts, c_per_expert=c_per_expert):
                component_shape = (*source_leading, n_experts, c_per_expert)
            case Dense():
                component_shape = (*source_leading, site.C)
        delta_shape = source_leading
        match init:
            case "random":
                sources[site.name] = _draw_site_source(site_key, site.factorization, source_leading)
            case "ones" | "zeroes":
                fill = jnp.ones if init == "ones" else jnp.zeros
                components: SourceComponents = fill(component_shape, jnp.float32)
                if isinstance(site.factorization, ExpertBlocked):
                    components = ExpertBlockedSource(values=components)
                sources[site.name] = SiteSource(
                    components=components, delta=fill(delta_shape, jnp.float32)
                )
    return sources


def init_sources_adam_state(sources: SourceStacks) -> SourcesAdamState:
    return SourcesAdamState(
        m=jax.tree.map(jnp.zeros_like, sources),
        v=jax.tree.map(jnp.zeros_like, sources),
        step_count=jnp.zeros(()),
    )


def init_sources_opt_state(
    optimizer: SourceOptimizerConfig, sources: SourceStacks
) -> SourcesOptState:
    storage_dtypes = {leaf.dtype for leaf in jax.tree.leaves(sources)}
    match optimizer:
        case AdamPGDConfig():
            assert jnp.dtype(jnp.uint16) not in storage_dtypes, (
                "SRC_STEP adam keeps moments AT the source storage dtype; uint16 "
                "fixed-point storage pairs with the float-buffered arms (momentum_sgd) "
                "or the stateless one (sgd)"
            )
            return init_sources_adam_state(sources)
        case SgdPGDConfig():
            return SourcesSgdState()
        case MomentumSgdPGDConfig():
            (storage_dtype,) = storage_dtypes
            return SourcesMomentumState(
                velocity=jax.tree.map(
                    lambda a: jnp.zeros_like(a, dtype=velocity_dtype(storage_dtype)), sources
                )
            )


def sources_sgd_ascend_project(
    sources: SourceStacks, sources_grad: SourceStacks, lr: Array
) -> SourceStacks:
    """One plain-SGD ASCENT on the persistent sources, then project to [0,1] (SRC_STEP `sgd`). The `lr·grad` product runs at fp32 (scalar-lr promotion)
    and casts ONCE into the source storage dtype — nothing wider persists."""
    return jax.tree.map(
        lambda source, g: jnp.clip(source + (lr * g).astype(source.dtype), 0.0, 1.0),
        sources,
        sources_grad,
    )


def sources_momentum_ascend_project(
    values: SourceStacks,
    sources_grad: SourceStacks,
    state: SourcesMomentumState,
    lr: Array,
    momentum: float,
) -> tuple[SourceStacks, SourcesMomentumState]:
    """One momentum-SGD ASCENT on the float source values, then project to [0,1] (SRC_STEP `momentum_sgd`): `v = momentum·v + grad; values += lr·v`. The EMA
    forms in fp32 and rounds ONCE into the velocity's own float dtype — a Python-float
    coefficient times a bf16 buffer multiplies IN bf16, rounding the coefficient itself;
    the store rounding is the 16-bit seam, the coefficient rounding is not."""
    velocity = jax.tree.map(
        lambda v, g: (momentum * v.astype(jnp.float32) + g.astype(jnp.float32)).astype(v.dtype),
        state.velocity,
        sources_grad,
    )
    new_values = jax.tree.map(
        lambda value, v: jnp.clip(value + (lr * v).astype(value.dtype), 0.0, 1.0),
        values,
        velocity,
    )
    return new_values, SourcesMomentumState(velocity=velocity)


def sources_ascend_project(
    sources: SourceStacks,
    sources_grad: SourceStacks,
    opt_state: SourcesOptState,
    lr: Array,
    optimizer: SourceOptimizerConfig,
    key: PRNGKeyArray,
) -> tuple[SourceStacks, SourcesOptState]:
    """One SRC_STEP ASCENT dispatched over the config/state pair — the
    enumerated arms share only the projection contract. `sources` is the STORED
    representation; the arms ascend its float view (`sources_grad` is d/d(float view))
    and `store_sources` lands the projected values back — `key` feeds only the uint16
    stochastic rounding and is dead in the graph for float storage."""
    values = source_values_to_float(sources)
    match optimizer, opt_state:
        case AdamPGDConfig(), SourcesAdamState():
            new_values, new_state = sources_adam_ascend_project(
                values, sources_grad, opt_state, lr, optimizer
            )
        case SgdPGDConfig(), SourcesSgdState():
            new_values, new_state = sources_sgd_ascend_project(values, sources_grad, lr), opt_state
        case MomentumSgdPGDConfig(momentum=momentum), SourcesMomentumState():
            new_values, new_state = sources_momentum_ascend_project(
                values, sources_grad, opt_state, lr, momentum
            )
        case _:
            raise AssertionError(
                f"SRC_STEP config/state mismatch: {type(optimizer).__name__} "
                f"with {type(opt_state).__name__}"
            )
    return store_sources(sources, new_values, key), new_state


def sources_adam_ascend_project(
    sources: SourceStacks,
    sources_grad: SourceStacks,
    adam_state: SourcesAdamState,
    lr: Array,
    adam: AdamPGDConfig,
) -> tuple[SourceStacks, SourcesAdamState]:
    """One Adam ASCENT on the persistent sources, then project to [0,1]."""
    step_count = adam_state.step_count + 1.0
    # `sources_grad` arrives in the masked-forward compute dtype (bf16); cast to the moment
    # dtype so the persistent `m`/`v` keep their declared storage dtype across steps.
    grad = jax.tree.map(lambda g, moment: g.astype(moment.dtype), sources_grad, adam_state.m)
    m = jax.tree.map(
        lambda moment, g: adam.beta1 * moment + (1 - adam.beta1) * g, adam_state.m, grad
    )
    v = jax.tree.map(
        lambda moment, g: adam.beta2 * moment + (1 - adam.beta2) * g * g, adam_state.v, grad
    )
    bias_correction1 = 1 - adam.beta1**step_count
    bias_correction2 = 1 - adam.beta2**step_count
    new_sources = jax.tree.map(
        lambda source, first, second: jnp.clip(
            source
            + (
                lr * (first / bias_correction1) / (jnp.sqrt(second / bias_correction2) + adam.eps)
            ).astype(source.dtype),
            0.0,
            1.0,
        ),
        sources,
        m,
        v,
    )
    return new_sources, SourcesAdamState(m=m, v=v, step_count=step_count)


class PersistentAdversary(eqx.Module):
    """One persistent-PGD adversary: the source stacks + their SRC_STEP state
    that persist across steps, plus the lifecycle the trainer drives around the shared
    backward. `sources` / `opt_state` are dynamic state; the rest is static config.

    Per step: `warmup_ascend` (n_warmup supplemental ascents vs a scoring forward, params
    + CI detached) → the warmed sources enter the main `value_and_grad` as leaves →
    `final_ascend` (one more ascent from the SAME backward's source-grad — which IS
    `dL_term/d(sources)`: the source path is never coeff-scaled).
    Ascents and grads are STACKED throughout; a scoring loss takes the stacked float view
    and reads sites through `per_site()`."""

    sources: SourceStacks
    opt_state: SourcesOptState
    state_key: str = eqx.field(static=True)
    optimizer: SourceOptimizerConfig = eqx.field(static=True)
    n_warmup: int = eqx.field(static=True)

    def source_lr(self, train_frac: Array) -> Array:
        return scheduled_value_at(train_frac, self.optimizer.lr_schedule)

    @property
    def float_sources(self) -> SourceStacks:
        """The differentiable float view of the stored sources — what mask formation
        consumes (through `per_site()`) and what every source gradient is taken with
        respect to."""
        return source_values_to_float(self.sources)

    def warmup_ascend(
        self,
        scoring_loss: Callable[[SourceStacks], Array],
        train_frac: Array,
        key: PRNGKeyArray,
    ) -> "PersistentAdversary":
        """`n_warmup` supplemental SRC_STEP ascents on the sources vs `scoring_loss` (the
        route-all all-sites recon forward over FLOAT source values, params/CI detached —
        provided by the step). The warmed sources are `stop_gradient`'d: they enter the
        main backward as leaves, so the main graph differentiates w.r.t. them, not back
        through this scan. `key` feeds only the uint16 stochastic store."""
        lr = self.source_lr(train_frac)

        def body(
            carry: tuple[SourceStacks, SourcesOptState], ascent_key: PRNGKeyArray
        ) -> tuple[tuple[SourceStacks, SourcesOptState], None]:
            sources, opt = carry
            grad = jax.grad(scoring_loss)(source_values_to_float(sources))
            return sources_ascend_project(sources, grad, opt, lr, self.optimizer, ascent_key), None

        (warmed, warmed_opt), _ = jax.lax.scan(
            body, (self.sources, self.opt_state), random.split(key, self.n_warmup)
        )
        return eqx.tree_at(
            lambda a: (a.sources, a.opt_state), self, (jax.lax.stop_gradient(warmed), warmed_opt)
        )

    def after_one_ascent(
        self, grad: SourceStacks, train_frac: Array, key: PRNGKeyArray
    ) -> "PersistentAdversary":
        """The adversary one SRC_STEP ascent-and-project further along
        `grad` (taken w.r.t. the float source view)."""
        lr = self.source_lr(train_frac)
        sources, opt_state = sources_ascend_project(
            self.sources, grad, self.opt_state, lr, self.optimizer, key
        )
        return eqx.tree_at(lambda a: (a.sources, a.opt_state), self, (sources, opt_state))

    def final_ascend(
        self, source_grad: SourceStacks, train_frac: Array, key: PRNGKeyArray
    ) -> "PersistentAdversary":
        """One final ascent recycled from the shared backward: the
        source path enters the backward UNSCALED — the term's coeff scales only the
        model-side cotangents (`train.model_cotangents_scaled`) — so `source_grad` IS
        `dL_term/d(sources)`, with nothing to unscale, at every step of any coeff
        schedule, activation gates included."""
        return self.after_one_ascent(source_grad, train_frac, key)
