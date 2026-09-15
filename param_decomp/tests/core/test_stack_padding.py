"""Persist-stack padding: census resolution, the padded value tree, pad-inert
optimizers, and the faithfulness lane's pad exit — for
the V/U semantic groups and for the chunkwise CI fn's chunk stack.

The placed integration (entry slice, grad transpose, full train step, padded-vs-unpadded
bit-identity) lives in `param_decomp/tests/targets/test_stack_padding_placed.py`."""

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from jax import random
from jax.sharding import PartitionSpec as P

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    MHACIAttention,
    build_ci_fn,
    pad_ci_fn,
    resolve_ci_placement,
)
from param_decomp.core.components import (
    DENSE_U_AXES,
    DENSE_V_AXES,
    ComponentStacks,
    Dense,
    ExpertBlocked,
    SiteSpec,
    init_component_stacks,
    pad_component_stacks,
    site_slots_for,
)
from param_decomp.core.configs import PlacementTableConfig
from param_decomp.core.faithfulness import make_faithfulness_loss
from param_decomp.core.muon_stacked import stacked_muon
from param_decomp.core.placement import (
    CIFnPlacement,
    StackCensus,
    assert_stacked_muon_ci_staging,
    assert_stacked_muon_component_staging,
    from_config,
    padded_entry_waypoint,
)
from param_decomp.core.run_state import component_muon_dimension_numbers
from param_decomp.tests.core.test_placement import (
    _ACTIVATION_ROWS,
    _OWNER_TABLE_ROWS,
    CI_ROWS,
    TARGET_ROWS,
    _sites,
)

MESH = jax.sharding.AbstractMesh((4, 2, 1), ("replicate", "fsdp", "tp"))
MOE_MESH = jax.sharding.AbstractMesh((4, 2), ("data", "tp"))
RESIDENT_D64 = jax.sharding.AbstractMesh((64, 1), ("data", "tp"))
RESIDENT_D8_TP8 = jax.sharding.AbstractMesh((8, 8), ("data", "tp"))


def _expert_sites(stack_len: int) -> tuple[SiteSpec, ...]:
    factorization = ExpertBlocked(n_experts=2, d_in=8, d_out=4, c_per_expert=4)
    return tuple(
        SiteSpec(f"expert.{i}", factorization, "experts") for i in range(stack_len)
    ) + tuple(
        SiteSpec(f"shared.{i}", Dense(d_in=8, d_out=8, C=16), "shared") for i in range(stack_len)
    )


# ── census resolution ─────────────────────────────────────────────────────────


def test_owner_census_pads_to_the_next_stack_cut_multiple():
    census = from_config("owner", MESH, _sites({(64, 32, 8): 6})).components.group_census
    [entry] = census.values()
    assert (entry.stack_len, entry.stack_pad, entry.padded_stack_len) == (6, 2, 8)
    assert entry.ns_stack_len == 8


def test_moe_owner_census_pads_expert_and_dense_groups():
    census = from_config(
        "owner-replicated-resident-moe", MOE_MESH, _expert_sites(6)
    ).components.group_census
    assert census["experts"].stack_pad == 2 and census["shared"].stack_pad == 2
    # the expert axis folds into muon's canonical NS stack — over the PADDED length
    assert census["experts"].ns_stack_len == 8 * 2
    assert census["shared"].ns_stack_len == 8


def test_moe_zero1_census_needs_no_pads():
    census = from_config(
        "zero1-replicated-resident-moe", MOE_MESH, _expert_sites(6)
    ).components.group_census
    assert all(entry.stack_pad == 0 for entry in census.values())


def test_pad_resolves_over_the_lcm_of_all_stack_sharding_rows():
    # optimizer/faithfulness rows cut ÷replicate=4, the delta row ÷fsdp=2: the persist
    # pad must satisfy BOTH, i.e. the lcm — here lcm(4, 4, 2) = 4, so 6 pads to 8.
    table = PlacementTableConfig.model_validate(
        {
            "components": {
                "optimizer_state": {"stack": "replicate", "d_in": "fsdp", "C": "tp"},
                "compute_weights": {"d_in": "fsdp", "C": "tp"},
                "faithfulness_weights": {"stack": "replicate", "d_in": "fsdp", "C": "tp"},
                "faithfulness_deltas": {"stack": "fsdp"},
                "operands": {"C": "tp"},
                "ns_compute": {"stack": "replicate"},
            },
            "ci_fn": CI_ROWS,
            "activations": {
                "external": {"batch": ["replicate", "fsdp"]},
                "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
            },
            "target": TARGET_ROWS,
        }
    )
    census = from_config(table, MESH, _sites({(64, 32, 8): 6})).components.group_census
    [entry] = census.values()
    assert (entry.stack_pad, entry.padded_stack_len) == (2, 8)


def test_muon_staging_claim_holds_on_the_padded_census():
    rules = from_config("owner", MESH, _sites({(64, 32, 8): 6}))
    assert_stacked_muon_component_staging(rules)  # 8 tiles ÷replicate=4


# ── the padded entry's waypoint ───────────────────────────────────────────────


def test_padded_entry_waypoint_parks_the_stack_cut_minor():
    """The padded entry's waypoint is the compute row with the persist stack cut nested
    MINOR on the leaf's last axis: under the resident owner masters `{stack: data, C: tp}`
    a V leaf's C carries `(tp, data)` and a U leaf's d_out `(data,)`; under three-axis
    owner the cut is `replicate`, nested after the compute row's fsdp / tp."""
    resident = from_config(
        "owner-replicated-resident", RESIDENT_D64, _sites({(64, 64, 64): 36})
    ).components
    waypoint_v = padded_entry_waypoint(
        resident.optimizer_state, resident.compute_weights, DENSE_V_AXES
    )
    waypoint_u = padded_entry_waypoint(
        resident.optimizer_state, resident.compute_weights, DENSE_U_AXES
    )
    assert waypoint_v.spec_for(DENSE_V_AXES) == P(None, None, ("tp", "data"))
    assert waypoint_u.spec_for(DENSE_U_AXES) == P(None, "tp", "data")
    assert waypoint_v.label == "components/compute_weights (padded-entry waypoint)"
    owner = from_config("owner", MESH, _sites({(64, 32, 8): 6})).components
    assert padded_entry_waypoint(
        owner.optimizer_state, owner.compute_weights, DENSE_V_AXES
    ).spec_for(DENSE_V_AXES) == P(None, "fsdp", ("tp", "replicate"))
    assert padded_entry_waypoint(
        owner.optimizer_state, owner.compute_weights, DENSE_U_AXES
    ).spec_for(DENSE_U_AXES) == P(None, "tp", ("fsdp", "replicate"))


def test_padded_world_refuses_a_minor_axis_the_stack_cut_cannot_tile():
    """A padded stack's entry demands the leaf's last dim tile its compute assignment with
    the stack cut nested: C=8 cannot carry `(tp, data)` = ÷64 — refused where the rows
    are bound, naming the waypoint. The same sites at a pad-free world make no such
    demand."""
    with pytest.raises(
        AssertionError, match=r"padded-entry waypoint\): semantic axis 'C' \(dim 8\) does not tile"
    ):
        from_config("owner-replicated-resident", RESIDENT_D64, _sites({(64, 64, 8): 36}))
    from_config("owner-replicated-resident", RESIDENT_D8_TP8, _sites({(64, 64, 8): 8}))


def test_padded_entry_refuses_a_gather_beyond_the_stack_cut():
    """The padded entry is spelled for the stack-only gather the owner layouts declare: a
    table whose compute row also gathers a matrix axis of a PADDED stack refuses at
    construction; the same table places a pad-free stack through the plain gather."""
    table = PlacementTableConfig.model_validate(
        {
            "components": {
                "optimizer_state": {"stack": "replicate", "d_in": "fsdp", "C": "tp"},
                "compute_weights": {"C": "tp"},
                "faithfulness_weights": {"stack": "replicate", "d_in": "fsdp", "C": "tp"},
                "faithfulness_deltas": {"stack": "replicate", "d_out": "fsdp"},
                "operands": {"C": "tp"},
                "ns_compute": {"stack": "replicate"},
            },
            "ci_fn": CI_ROWS,
            "activations": {
                "external": {"batch": ["replicate", "fsdp"]},
                "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
            },
            "target": TARGET_ROWS,
        }
    )
    with pytest.raises(AssertionError, match="spelled for a stack-only gather"):
        from_config(table, MESH, _sites({(64, 32, 8): 6}))
    from_config(table, MESH, _sites({(64, 32, 8): 8}))


# ── the padded value tree ─────────────────────────────────────────────────────


def _padded_stacks() -> tuple[tuple[SiteSpec, ...], ComponentStacks, ComponentStacks]:
    sites = _sites({(8, 4, 4): 3})
    unpadded = init_component_stacks(sites, random.PRNGKey(0))
    return sites, unpadded, pad_component_stacks(unpadded, {"8x4x4": 2})


def test_pad_component_stacks_appends_zero_slots_and_enumerates_them():
    _, unpadded, padded = _padded_stacks()
    assert padded.stack_pads == (("8x4x4", 2),) and padded.pad_of("8x4x4") == 2
    assert padded.site_slots == unpadded.site_slots
    for (Vs, Us), (pVs, pUs) in zip(unpadded.stacks.values(), padded.stacks.values(), strict=True):
        assert pVs.shape == (5, *Vs.shape[1:]) and pUs.shape == (5, *Us.shape[1:])
        assert (pVs[:3] == Vs).all() and (pUs[:3] == Us).all()
        assert (pVs[3:] == 0).all() and (pUs[3:] == 0).all()
    with pytest.raises(AssertionError, match="already padded"):
        pad_component_stacks(padded, {"8x4x4": 1})


def test_zero_pad_counts_collapse_to_the_unpadded_spelling():
    sites = _sites({(8, 4, 4): 3})
    stacks = pad_component_stacks(init_component_stacks(sites, random.PRNGKey(0)), {"8x4x4": 0})
    assert stacks.stack_pads == ()


# ── pad-inert optimizers ──────────────────────────────────────────────────────


def _pad_row_grads(padded: ComponentStacks) -> ComponentStacks:
    """Synthetic grads: nonzero on real slots, exactly zero on pad slots — what the
    entry-slice transpose and the zero-delta faithfulness lane deliver."""
    stacks = {}
    for group, (Vs, Us) in padded.stacks.items():
        pad = padded.pad_of(group)
        real = Vs.shape[0] - pad
        mask = jnp.arange(Vs.shape[0]) < real
        key = random.PRNGKey(hash(group) % (2**31))
        stacks[group] = (
            random.normal(key, Vs.shape) * mask.reshape(-1, *([1] * (Vs.ndim - 1))),
            random.normal(random.fold_in(key, 1), Us.shape)
            * mask.reshape(-1, *([1] * (Us.ndim - 1))),
        )
    return ComponentStacks(
        stacks=stacks, site_slots=padded.site_slots, stack_pads=padded.stack_pads
    )


@pytest.mark.parametrize("which", ("adamw", "muon"))
def test_pads_stay_exactly_zero_through_optimizer_steps(which: str):
    _, _, padded = _padded_stacks()
    opt: optax.GradientTransformation
    match which:
        case "adamw":
            opt = optax.adamw(1e-3, weight_decay=0.0)
        case _:
            opt = stacked_muon(
                1e-3,
                beta=0.95,
                weight_decay=0.0,
                consistent_rms=None,
                muon_weight_dimension_numbers=component_muon_dimension_numbers(
                    {"8x4x4": Dense(d_in=8, d_out=4, C=4)}
                ),
                ns_steps=5,
                ns_dtype=jnp.float32,
                waypoints=None,
            )
    grads = _pad_row_grads(padded)
    state = opt.init(eqx.filter(padded, eqx.is_array))
    params = padded
    for _ in range(3):
        updates, state = opt.update(
            eqx.filter(grads, eqx.is_array), state, eqx.filter(params, eqx.is_array)
        )
        params = eqx.apply_updates(params, updates)
    for group, (Vs, Us) in params.stacks.items():
        real = Vs.shape[0] - params.pad_of(group)
        assert (Vs[real:] == 0.0).all() and (Us[real:] == 0.0).all(), group
        assert not (Vs[:real] == 0.0).all(), "real slots must actually have moved"
    for leaf in jax.tree.leaves(eqx.filter(state, eqx.is_array)):
        if leaf.ndim >= 1 and leaf.shape[:1] == (5,):
            assert (leaf[3:] == 0.0).all(), "optimizer moments grew pad mass"


# ── the faithfulness lane's pad exit ──────────────────────────────────────────


def test_faithfulness_loss_ignores_zero_pad_slots_exactly():
    sites = _sites({(8, 4, 4): 3})
    site_slots = site_slots_for(sites)
    norms = {"8x4x4": (2.0, 3.0, 4.0)}
    deltas = {"8x4x4": random.normal(random.PRNGKey(0), (3, 4, 8))}
    padded_deltas = {"8x4x4": jnp.concatenate([deltas["8x4x4"], jnp.zeros((2, 4, 8))])}
    unpadded_loss = make_faithfulness_loss(site_slots, norms, {})(deltas)
    padded_loss = make_faithfulness_loss(site_slots, norms, {"8x4x4": 2})(padded_deltas)
    assert padded_loss == unpadded_loss


def test_faithfulness_loss_refuses_a_delta_stack_of_the_wrong_extent():
    sites = _sites({(8, 4, 4): 3})
    loss = make_faithfulness_loss(site_slots_for(sites), {"8x4x4": (1.0, 1.0, 1.0)}, {"8x4x4": 2})
    with pytest.raises(AssertionError):
        loss({"8x4x4": jnp.zeros((3, 4, 8))})  # real-length deltas at a padded binding


# ── the chunkwise CI fn's chunk-stack census ──────────────────────────────────


def _chunkwise_arch(
    sites: tuple[SiteSpec, ...], sites_per_chunk: int
) -> ChunkwiseTransformerCIArch:
    """One chunk per `sites_per_chunk` consecutive sites (equal C, so the heads stack)."""
    names = tuple(spec.name for spec in sites)
    assert len(names) % sites_per_chunk == 0, (len(names), sites_per_chunk)
    return ChunkwiseTransformerCIArch(
        chunks=tuple(
            Chunk(input_taps=("tap",), output_sites=names[i : i + sites_per_chunk])
            for i in range(0, len(names), sites_per_chunk)
        ),
        input_dim=64,
        d_model=64,
        n_blocks=1,
        attention=MHACIAttention(n_heads=8),
        ffn_hidden=128,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def _resolved(
    spec: str, mesh: jax.sharding.AbstractMesh, sites: tuple[SiteSpec, ...]
) -> CIFnPlacement:
    placement = resolve_ci_placement(
        _chunkwise_arch(sites, 1),
        from_config(spec, mesh, sites),  # pyright: ignore[reportArgumentType]
    )
    assert placement is not None
    return placement


def test_owner_resident_ci_census_pads_the_chunk_stack_to_the_data_cut():
    # the 36-layer GLU seat at one block per chunk: 36 chunks pad to 64 at data=64, to 40
    # at data=8 — the same cut its 36-slot component stacks take (every leaf's last dim
    # tiles the padded entry's ÷64 waypoint)
    sites = _sites({(64, 64, 64): 36})
    for mesh, pad in ((RESIDENT_D64, 28), (RESIDENT_D8_TP8, 4)):
        placement = _resolved("owner-replicated-resident", mesh, sites)
        assert placement.chunks == StackCensus(stack_len=36, stack_pad=pad)
        [entry] = set(
            from_config("owner-replicated-resident", mesh, sites).components.group_census.values()
        )
        assert entry.stack_pad == pad


def test_zero1_ci_rows_cut_no_stack_so_no_chunk_pad_resolves():
    # 3 chunks do not tile MOE_MESH's data=4, but zero1 (and ddp) cut no CI stack axis
    sites = _sites({(64, 32, 8): 3})
    for spec, mesh in (("zero1-replicated-resident", MOE_MESH), ("ddp", MESH)):
        assert _resolved(spec, mesh, sites).chunks == StackCensus(stack_len=3, stack_pad=0)


def test_ci_muon_staging_claim_holds_on_the_padded_chunk_census():
    sites = _sites({(64, 64, 64): 36})
    rules = from_config("owner-replicated-resident", RESIDENT_D64, sites)
    placement = resolve_ci_placement(_chunkwise_arch(sites, 1), rules)
    assert placement is not None
    assert_stacked_muon_ci_staging(placement)  # 64 padded slots tile the ÷64 split
    with pytest.raises(AssertionError, match=r"ci_fn/attention \(stacks 36\)"):
        assert_stacked_muon_ci_staging(
            CIFnPlacement.resolved(rules.ci_fn, StackCensus(stack_len=36, stack_pad=0))
        )


def test_resolution_refuses_a_ci_master_leaf_the_rows_cannot_tile():
    # the build-time twin of the group census' shape validation: every arch-known master
    # leaf tiles at the padded extent, or the config refuses before any device sees it
    sites = _sites({(64, 32, 8): 8})
    arch = dataclasses.replace(_chunkwise_arch(sites, 1), ffn_hidden=12)
    rules = from_config("owner-replicated-resident", RESIDENT_D8_TP8, sites)
    with pytest.raises(AssertionError, match=r"'ffn_hidden' \(dim 12\) does not tile"):
        resolve_ci_placement(arch, rules)


def test_pad_ci_fn_appends_zero_chunk_slots_and_enumerates_them():
    sites = _sites({(64, 32, 8): 3})
    arch = _chunkwise_arch(sites, 1)
    placement = resolve_ci_placement(
        arch, from_config("owner-replicated-resident", MOE_MESH, sites)
    )
    assert placement is not None and placement.chunks == StackCensus(stack_len=3, stack_pad=1)
    fn = build_ci_fn(arch, sites, random.PRNGKey(0))
    assert isinstance(fn, ChunkwiseTransformerCIFn) and fn.stack_pad == 0
    padded = pad_ci_fn(fn, placement)
    assert isinstance(padded, ChunkwiseTransformerCIFn)
    assert padded.stack_pad == 1 and padded.chunk_meta == fn.chunk_meta
    assert padded.inv_freq is fn.inv_freq
    for real, wide in zip(jax.tree.leaves(fn.chunks), jax.tree.leaves(padded.chunks), strict=True):
        assert wide.shape == (4, *real.shape[1:])
        assert (wide[:3] == real).all() and (wide[3:] == 0).all()
    with pytest.raises(AssertionError, match="already padded"):
        pad_ci_fn(padded, placement)
    # the scan consumes the compute residents only; a padded persist tree never reaches it
    with pytest.raises(AssertionError, match="compute residents"):
        padded({"tap": jnp.zeros((2, 4, 64))}, remat=False, placement=None)


def test_zero_chunk_pad_keeps_the_fn_itself():
    sites = _sites({(64, 32, 8): 4})
    arch = _chunkwise_arch(sites, 1)
    placement = resolve_ci_placement(
        arch, from_config("owner-replicated-resident", MOE_MESH, sites)
    )
    assert placement is not None and placement.chunks.stack_pad == 0
    fn = build_ci_fn(arch, sites, random.PRNGKey(0))
    assert pad_ci_fn(fn, placement) is fn


@pytest.mark.parametrize(
    "row", ("components/compute_weights", "ci_fn/ffn.compute_weights", "ci_fn/vectors")
)
def test_the_scanned_rows_refuse_a_stack_assignment(row: str):
    # the arrays at these rows are iterated slot by slot by the scans and are what the pad
    # exit slices, so their stack axis rests whole — a `stack` key refuses at bind
    components: dict[str, Any] = dict(_OWNER_TABLE_ROWS)
    ci_fn: dict[str, Any] = {family: dict(rows) for family, rows in CI_ROWS.items()}
    match row:
        case "components/compute_weights":
            components["compute_weights"] = {**components["compute_weights"], "stack": "replicate"}
        case "ci_fn/ffn.compute_weights":
            ci_fn["ffn"] = {
                **ci_fn["ffn"],
                "compute_weights": {**ci_fn["ffn"]["compute_weights"], "stack": "replicate"},
            }
        case "ci_fn/vectors":
            ci_fn["vectors"] = {**ci_fn["vectors"], "stack": "replicate"}
        case _:
            raise AssertionError(row)
    table = PlacementTableConfig.model_validate(
        {
            "components": components,
            "ci_fn": ci_fn,
            "activations": _ACTIVATION_ROWS,
            "target": TARGET_ROWS,
        }
    )
    with pytest.raises(AssertionError, match=rf"row '{row}' assigns `stack`"):
        from_config(table, MESH, _sites({(64, 32, 8): 4}))
