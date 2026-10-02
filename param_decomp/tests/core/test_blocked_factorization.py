"""Block-factored V/U through the real engine.

A synthetic block-factored target (one gate-style site fusing blocks on its output, one
down-style site fusing them on its input) drives the actual `make_train_step`, so the
whole block-factored path is exercised where it will live: 4-D stacks through init,
masking with routes and weight deltas, blocked faithfulness deltas, and an optimizer
step under both adamw and stacked muon. The site-forward parity tests pin the block
einsums against a per-block loop reference; the placement tests bind an explicit table
over the `expert`/`C_block` axes and check the fail-closed refusals."""

import dataclasses
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.components import (
    BLOCKED_U_INIT_FAN_IN,
    BLOCKED_V_AXES,
    BlockedFactorization,
    ComponentStacks,
    DenseFactorization,
    SiteCI,
    SiteSpec,
    group_factorizations,
    init_component_stacks,
    require_full_emission,
    site_stack_indices_for,
    vu_groups,
)
from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MuonOptimizerConfig,
    PlacementTableConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.decomposed_linear import (
    BlockedPlannedComponentLinear,
    BlockedSiteWeights,
    blocked_site_forward,
)
from param_decomp.core.dict_utils import FrozenMapping
from param_decomp.core.faithfulness import faithfulness_loss_for, make_faithfulness_loss
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.linear_plan import BlockContraction, BlockedLinearPlan
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.masking import materialize_masking
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ForwardResult,
    Masking,
    MaterializedMasking,
    PlacedModel,
    SiteRoutes,
    StochasticMasking,
)
from param_decomp.core.objective import build_objective
from param_decomp.core.placement import PlacedRule, PlacementRules, from_config
from param_decomp.core.run_state import (
    _adamw_optimizer,
    _muon_optimizer,
    component_muon_dimension_numbers,
)
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    PDTrainingState,
    TrainState,
    make_train_step,
)
from param_decomp.sequence import SequenceLayout

B, T, D = 2, 3, 8
E, DI, CPB = 4, 4, 2
C = E * CPB
K_OUT = 4
GATE, DOWN = "block.0.gate", "block.0.down"

GATE_FACTORIZATION = BlockedFactorization(n_blocks=E, d_in=D, d_out=DI, c_per_block=CPB)
DOWN_FACTORIZATION = BlockedFactorization(n_blocks=E, d_in=DI, d_out=D, c_per_block=CPB)


def _sites() -> tuple[SiteSpec, ...]:
    return (
        SiteSpec(GATE, GATE_FACTORIZATION, "gate"),
        SiteSpec(DOWN, DOWN_FACTORIZATION, "down"),
    )


def _untype(value: Array) -> Array:
    """Drop axis typing (identity off-mesh): this fixture's linears are bare einsums,
    which cannot resolve weight grads against an axis-typed operand under Explicit."""
    sharding = jax.typeof(value).sharding
    if sharding.mesh.empty:
        return value
    return jax.sharding.reshard(value, NamedSharding(sharding.mesh, P(*([None] * value.ndim))))


class SyntheticBlockedModel(eqx.Module):
    """A `DecomposedModel` whose two sites are block-factored: `w_gate [E*DI, D]` fuses
    blocks on its output, `w_down [D, E*DI]` on its input, with a silu between them —
    the minimal block-factored pair covering both `BlockContraction` values."""

    feat_proj: Float[Array, "D D"]
    w_gate: Float[Array, "ED D"]
    w_down: Float[Array, "D ED"]
    read: Float[Array, "K D"]
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, placement: PlacementRules) -> "SyntheticBlockedModel":
        repl = NamedSharding(placement.mesh, P())
        return jax.tree.map(lambda _a: repl, self)

    @staticmethod
    def recon_loss_fn(masked_output: Array, clean_output: Array) -> Float[Array, ""]:
        err = (masked_output.astype(jnp.float32) - clean_output.astype(jnp.float32)) ** 2
        return jnp.sum(err) / (B * T)

    @staticmethod
    def pin_output_batch(output: Array, mesh: Mesh | None) -> Array:
        return batch_shard_leading(output, mesh)

    def _residual(self, inputs: dict[str, Array]) -> Float[Array, "B T D"]:
        return _untype(inputs["feat"] @ self.feat_proj.T)

    def _ordered_capture_keys(self, keys: CaptureKeys) -> tuple[str, ...]:
        allowed = {GATE, DOWN, f"{GATE}.out", f"{DOWN}.out"}
        assert keys <= allowed, (keys, allowed)
        return tuple(sorted(keys))

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(f"{site}.out" for site in sites)

    def _result(
        self,
        inputs: dict[str, Array],
        output: Array,
        values: dict[str, Array],
        ordered_capture_keys: tuple[str, ...],
    ) -> ForwardResult[Array, dict[str, Array]]:
        return ForwardResult.from_producer(
            leading_shape=output.shape[:-1],
            output=output,
            capture_keys=ordered_capture_keys,
            capture_values=tuple(values[key] for key in ordered_capture_keys),
            conditioning=inputs,
            sequence=SequenceLayout(jnp.zeros_like(output[..., 0], dtype=jnp.int32)),
        )

    def clean_forward(
        self,
        inputs: dict[str, Array],
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult[Array, dict[str, Array]]:
        del placement
        ordered = self._ordered_capture_keys(capture_keys)
        resid = self._residual(inputs)
        g = resid @ self.w_gate.T
        h = jax.nn.silu(g)
        down = h @ self.w_down.T
        output = _untype(resid + down) @ self.read.T
        values = {GATE: resid, DOWN: h, f"{GATE}.out": g, f"{DOWN}.out": down}
        return self._result(inputs, output, values, ordered)

    def prepare_compute_weights(
        self, vu: ComponentStacks, placement: PlacementRules | None
    ) -> ComponentStacks:
        del placement
        return vu

    def component_activation_forward(
        self,
        prepared_weights: ComponentStacks,
        inputs: dict[str, Array],
        /,
        *,
        sites: tuple[str, ...],
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult[Array, dict[str, Array]], dict[str, SiteCI]]:
        del prepared_weights, inputs, sites, capture_keys, placement
        raise NotImplementedError

    @staticmethod
    def prepare_masking(masking: Masking) -> MaterializedMasking:
        return materialize_masking(masking)

    @staticmethod
    def prepare_stochastic_masking(
        ci: Mapping[str, SiteCI],
    ) -> Callable[[Array], MaterializedMasking]:
        def draw(draw_key: Array) -> MaterializedMasking:
            return materialize_masking(StochasticMasking(ci=ci, draw_key=draw_key))

        return draw

    def masked_forward(
        self,
        vu: ComponentStacks,
        inputs: dict[str, Array],
        /,
        *,
        masking: MaterializedMasking,
        routes: SiteRoutes | None,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult[Array, dict[str, Array]]:
        del placement, remat
        ordered = self._ordered_capture_keys(capture_keys)
        assert set(masking.component_masks) == {GATE, DOWN}
        resid = self._residual(inputs)

        def site(name: str, x: Array, frozen: Array, contraction: BlockContraction) -> Array:
            comps = vu.site(name)
            deltas = masking.weight_delta_masks
            return blocked_site_forward(
                x,
                BlockedSiteWeights(frozen, _untype(comps.V), _untype(comps.U), None, None),
                _untype(require_full_emission(masking.component_masks[name])),
                None if deltas is None else _untype(deltas[name]),
                None if routes is None else _untype(routes[name]),
                contraction,
            ).output

        g = site(GATE, resid, self.w_gate, "fused_output")
        h = jax.nn.silu(g)
        down = site(DOWN, h, self.w_down, "fused_input")
        output = _untype(resid + down) @ self.read.T
        values = {GATE: resid, DOWN: h, f"{GATE}.out": g, f"{DOWN}.out": down}
        return self._result(inputs, output, values, ordered)

    def target_weight_sq_norms(self) -> dict[str, Array]:
        return {
            "gate": jnp.sum(self.w_gate.astype(jnp.float32) ** 2)[None],
            "down": jnp.sum(self.w_down.astype(jnp.float32) ** 2)[None],
        }

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Array]:
        def blocked_delta(name: str, frozen_blocks: Array) -> Array:
            comps = vu.site(name)
            tilde = jnp.einsum(
                "eij,ejk->eki", comps.V.astype(jnp.float32), comps.U.astype(jnp.float32)
            )
            return (frozen_blocks.astype(jnp.float32) - tilde)[None]

        gate_blocks = self.w_gate.reshape(E, DI, D)
        down_blocks = self.w_down.reshape(D, E, DI).transpose(1, 0, 2)
        return {
            "gate": blocked_delta(GATE, gate_blocks),
            "down": blocked_delta(DOWN, down_blocks),
        }


def _model(key: Array) -> SyntheticBlockedModel:
    return SyntheticBlockedModel(
        feat_proj=random.normal(random.fold_in(key, 0), (D, D)),
        w_gate=random.normal(random.fold_in(key, 1), (E * DI, D)),
        w_down=random.normal(random.fold_in(key, 2), (D, E * DI)),
        read=random.normal(random.fold_in(key, 3), (K_OUT, D)),
        sites=_sites(),
        has_position_axis=True,
    )


def _inputs(key: Array, batch: int = B) -> dict[str, Array]:
    return {"feat": random.normal(key, (batch, T, D))}


# ── factorization + init ──────────────────────────────────────────────────────


def test_vu_groups_refuses_mixed_factorizations():
    mixed = (
        SiteSpec("a.0", GATE_FACTORIZATION, "g"),
        SiteSpec("a.1", DenseFactorization(d_in=D, d_out=E * DI, C=C), "g"),
    )
    with pytest.raises(AssertionError, match="mixes factorizations"):
        vu_groups(mixed)


def test_site_spec_has_no_fused_dims_for_block_factored_sites():
    spec = _sites()[0]
    assert spec.C == C
    with pytest.raises(AssertionError):
        _ = spec.d_in
    with pytest.raises(AssertionError):
        _ = spec.d_out


def test_init_block_factored_shapes_and_scales():
    assert BLOCKED_U_INIT_FAN_IN == "site", "test pins the current init-scale decision"
    n_blocks, d_in, d_out, c = 8, 64, 48, 16
    factorization = BlockedFactorization(n_blocks=n_blocks, d_in=d_in, d_out=d_out, c_per_block=c)
    stacks = init_component_stacks((SiteSpec("s", factorization, "s"),), random.PRNGKey(0)).stacks[
        "s"
    ]
    Vs, Us = stacks
    assert Vs.shape == (1, n_blocks, d_in, c)
    assert Us.shape == (1, n_blocks, c, d_out)
    np.testing.assert_allclose(jnp.std(Vs), d_in**-0.5, rtol=0.05)
    np.testing.assert_allclose(jnp.std(Us), (n_blocks * c) ** -0.5, rtol=0.05)


# ── the site forward against a per-block loop reference ──────────────────────


def _reference_site_forward(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
    contraction: BlockContraction,
) -> tuple[Array, Array]:
    lead = x.shape[:-1]
    n_blocks, d_in_b, c = V.shape
    if contraction == "fused_output":
        xV = jnp.stack([x @ V[e] for e in range(n_blocks)], axis=-2)
    else:
        xe = x.reshape(*lead, n_blocks, d_in_b)
        xV = jnp.stack([xe[..., e, :] @ V[e] for e in range(n_blocks)], axis=-2)
    xV_flat = xV.reshape(*lead, n_blocks * c)
    coeff = mask
    delta = None
    if delta_mask is not None:
        delta = delta_mask[..., None]
        coeff = 1.0 - delta if coeff is None else coeff - delta
    acts = (xV_flat * coeff if coeff is not None else xV_flat).reshape(*lead, n_blocks, c)
    if contraction == "fused_output":
        out = jnp.concatenate([acts[..., e, :] @ U[e] for e in range(n_blocks)], axis=-1)
    else:
        out = jnp.sum(jnp.stack([acts[..., e, :] @ U[e] for e in range(n_blocks)]), axis=0)
    frozen = x @ W.T
    if delta_mask is not None:
        assert delta is not None
        out = out + delta * frozen
    if route is not None:
        out = jnp.where(route[..., None], out, frozen)
    return out, xV_flat


@pytest.mark.parametrize("contraction", ["fused_output", "fused_input"])
@pytest.mark.parametrize("with_delta_and_route", [False, True])
def test_blocked_site_forward_matches_per_block_reference(
    contraction: BlockContraction, with_delta_and_route: bool
):
    key = random.PRNGKey(5)
    d_in_b, d_out_b = (D, DI) if contraction == "fused_output" else (DI, D)
    site_d_in = d_in_b * (E if contraction == "fused_input" else 1)
    site_d_out = d_out_b * (E if contraction == "fused_output" else 1)
    x = random.normal(random.fold_in(key, 0), (B, T, site_d_in))
    V = random.normal(random.fold_in(key, 1), (E, d_in_b, CPB)) * 0.3
    U = random.normal(random.fold_in(key, 2), (E, CPB, d_out_b)) * 0.3
    W = random.normal(random.fold_in(key, 3), (site_d_out, site_d_in))
    mask = random.uniform(random.fold_in(key, 4), (B, T, C))
    delta_mask = random.uniform(random.fold_in(key, 5), (B, T)) if with_delta_and_route else None
    route = random.bernoulli(random.fold_in(key, 6), 0.5, (B, T)) if with_delta_and_route else None

    got = blocked_site_forward(
        x, BlockedSiteWeights(W, V, U, None, None), mask, delta_mask, route, contraction
    )
    want_out, want_xv = _reference_site_forward(x, V, U, W, mask, delta_mask, route, contraction)
    np.testing.assert_allclose(got.output, want_out, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got.component_activation, want_xv, rtol=1e-5, atol=1e-6)


def test_faithfulness_reduces_block_factored_deltas_per_slot():
    key = random.PRNGKey(6)
    sites = _sites()
    model = _model(key)
    vu = init_component_stacks(sites, random.fold_in(key, 1))
    deltas = model.weight_deltas(vu)
    assert deltas["gate"].shape == (1, E, DI, D)
    assert deltas["down"].shape == (1, E, D, DI)
    norms = {g: tuple(float(v) for v in vals) for g, vals in model.target_weight_sq_norms().items()}
    loss = make_faithfulness_loss(site_stack_indices_for(sites), norms, {})(deltas)
    want = 0.5 * sum(float(jnp.sum(deltas[g] ** 2)) / norms[g][0] for g in ("gate", "down"))
    np.testing.assert_allclose(loss, want, rtol=1e-6)


# ── the real train step ───────────────────────────────────────────────────────


def _ci_fn_arch() -> ChunkwiseTransformerCIFnArch:
    return ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(input_taps=(GATE,), output_sites=(GATE, DOWN)),),
        input_dim=D,
        d_model=8,
        n_blocks=1,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        ffn_hidden=16,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def _objective(sites: tuple[SiteSpec, ...]):
    return build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-4,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.5))
                ),
            ),
            StochasticReconLossConfig(coeff=1.0),
        ),
        sites,
    )


def _optimizers(kind: str, sites: tuple[SiteSpec, ...]):
    match kind:
        case "adamw":
            return _adamw_optimizer(
                AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-2)), 1
            ), _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-2)), 1)
        case "stacked_muon":
            opt_vu = _muon_optimizer(
                MuonOptimizerConfig(
                    type="muon", lr_schedule=ScheduleConfig.constant(1e-2), consistent_rms=None
                ),
                1,
                component_muon_dimension_numbers(group_factorizations(sites)),
                None,
            )
            return opt_vu, _adamw_optimizer(
                AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-2)), 1
            )
        case _:
            raise AssertionError(kind)


def _run_steps(kind: str, mesh: Mesh | None, n_steps: int, batch: int = B) -> ComponentStacks:
    # Placed runs execute under their mesh so bare-PartitionSpec reshards resolve.
    with nullcontext() if mesh is None else jax.set_mesh(mesh):
        key = random.PRNGKey(7)
        model = _model(key)
        sites = model.sites
        components = init_component_stacks(sites, random.fold_in(key, 1))
        rules = None if mesh is None else from_config("ddp", mesh, sites)
        ci_fn = _ci_fn_arch().initialize(sites, rules, random.fold_in(key, 2))
        opt_vu, opt_ci = _optimizers(kind, sites)
        objective = _objective(model.sites)
        state = TrainState(
            decomposition=Decomposition(components=components, ci_fn=ci_fn),
            training=PDTrainingState(
                frequency=BatchFrequency(),
                objective=objective,
                components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
                ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
                adversaries={},
                step=jnp.zeros((), jnp.int32),
            ),
        )
        placed = PlacedModel(model=model, placement=rules)
        step_fn = jax.jit(
            make_train_step(
                model_static=placed,
                substrate=ForwardSubstrate.of(
                    placed,
                    remat_recon_forwards=False,
                    remat_ci_fn=False,
                    ci_capture_keys=frozenset({GATE}),
                ),
                components_optimizer=opt_vu,
                ci_fn_optimizer=opt_ci,
                total_steps=10,
                faithfulness=faithfulness_loss_for(placed),
            )
        )
        inputs = _inputs(random.fold_in(key, 3), batch)
        run_key = random.PRNGKey(8)
        for step_idx in range(n_steps):
            state, metrics = step_fn(placed, state, inputs, random.fold_in(run_key, step_idx))
            assert jnp.isfinite(metrics["total"]), (kind, step_idx, metrics["total"])
        return state.decomposition.components


def _mesh(n_devices: int) -> Mesh:
    devices = np.asarray(jax.devices()[:n_devices]).reshape(n_devices, 1, 1)
    return Mesh(devices, ("replicate", "fsdp", "tp"), axis_types=(AxisType.Explicit,) * 3)


@pytest.mark.parametrize("kind", ["adamw", "stacked_muon"])
def test_train_step_runs_through_block_factored_target(kind: str):
    """Two real train steps over 4-D block stacks: finite loss, V moves, both sites'
    orientations exercised (routes and weight-delta masks arrive via the stochastic
    recon term's draws)."""
    key = random.PRNGKey(7)
    before = jax.device_get(init_component_stacks(_sites(), random.fold_in(key, 1)).site(GATE).V)
    components = _run_steps(kind, None, n_steps=2)
    assert components.site(GATE).V.shape == (E, D, CPB)
    assert components.site(DOWN).U.shape == (E, CPB, D)
    assert not jnp.allclose(components.site(GATE).V, before), "V did not move — step is a no-op"


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_device_count_invariance_block_factored():
    """For the 4-D leaves: the trajectory at one device matches the trajectory
    at four (ddp rules), up to float reassociation."""
    one = _run_steps("adamw", _mesh(1), n_steps=3, batch=4)
    four = _run_steps("adamw", _mesh(4), n_steps=3, batch=4)
    for group in ("gate", "down"):
        for got, want in zip(four.stacks[group], one.stacks[group], strict=True):
            # Reassociation drift passes through Adam's sqrt(v)-normalized update three
            # times; a wrong cross-device reduction would diverge by orders of magnitude.
            np.testing.assert_allclose(
                jax.device_get(got), jax.device_get(want), rtol=1e-2, atol=1e-4
            )


# ── placed execution + explicit-table placement ───────────────────────────────


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_placed_blocked_site_forward_matches_unplaced():
    """The block plans' reshard-and-contract choreography is value-identical to the
    unplaced einsums: the table axis (`expert`) split over tp, each block's matrix dims
    over fsdp."""
    devices = np.asarray(jax.devices()[:4]).reshape(1, 2, 2)
    mesh = Mesh(devices, ("replicate", "fsdp", "tp"), axis_types=(AxisType.Explicit,) * 3)
    key = random.PRNGKey(9)
    x = random.normal(random.fold_in(key, 0), (B, T, D))
    V = random.normal(random.fold_in(key, 1), (E, D, CPB)) * 0.3
    U = random.normal(random.fold_in(key, 2), (E, CPB, DI)) * 0.3
    mask = random.uniform(random.fold_in(key, 4), (B, T, C))
    W = random.normal(random.fold_in(key, 3), (E * DI, D))
    unplaced = blocked_site_forward(
        x, BlockedSiteWeights(W, V, U, None, None), mask, None, None, "fused_output"
    )

    def plan(factor: Literal["V", "U"], operand_input: P, output: P) -> BlockedLinearPlan:
        return BlockedLinearPlan(
            mesh=mesh,
            contraction="fused_output",
            factor=factor,
            input=operand_input,
            operand_input=operand_input,
            resident_weight=P("tp", "fsdp", None) if factor == "V" else P("tp", None, None),
            operand=P("tp", "fsdp", None) if factor == "V" else P("tp", None, None),
            output=output,
            weight_reduced=frozenset(),
        )

    planned = BlockedPlannedComponentLinear(
        v=plan("V", P(None, None, "fsdp"), P(None, None, "tp", None)),
        u=plan("U", P(None, None, "tp", None), P(None, None, "tp", None)),
        component=PlacedRule(
            mesh=mesh, label_for_log="component", rule=FrozenMapping({"C": ("tp",)})
        ),
        output=PlacedRule(mesh=mesh, label_for_log="output", rule=FrozenMapping({})),
    )
    put = lambda value, spec: jax.device_put(value, NamedSharding(mesh, spec))
    placed = blocked_site_forward(
        put(x, P()),
        BlockedSiteWeights(
            W, put(V, P("tp", "fsdp", None)), put(U, P("tp", None, None)), None, planned
        ),
        put(mask, P(None, None, "tp")),
        None,
        None,
        "fused_output",
    )
    np.testing.assert_allclose(
        jax.device_get(placed.output), jax.device_get(unplaced.output), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        jax.device_get(placed.component_activation),
        jax.device_get(unplaced.component_activation),
        rtol=1e-5,
        atol=1e-6,
    )


_TARGET_ROWS = {
    "embedding": {"persist": {}, "operand": {}},
    "normalization": {},
    "position_encoding": {},
    "column": {"persist": {}, "operand": {}, "input": "external", "output": "intermediate"},
    "row": {"persist": {}, "operand": {}, "input": "intermediate", "output": "external"},
    "output": {"persist": {}, "operand": {}},
    "intermediate": {"batch": ["replicate", "fsdp"]},
    "component": {"input": "external", "output": "external"},
}


def _blocked_table(component_rows: dict[str, dict[str, object]]) -> PlacementTableConfig:
    return PlacementTableConfig.model_validate(
        {
            "components": component_rows,
            "ci_fn": "zero1",
            "activations": {
                "external": {"batch": ["replicate", "fsdp"]},
                "component": {"batch": ["replicate", "fsdp"], "C": "tp"},
            },
            "target": _TARGET_ROWS,
        }
    )


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_within_block_tp_can_differ_from_public_component_ownership():
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    table = _blocked_table(
        {
            "optimizer_state": {"C_block": "tp"},
            "compute_weights": {"C_block": "tp"},
            "faithfulness_weights": {"C_block": "tp"},
            "faithfulness_deltas": {},
            "operands": {"C_block": "tp"},
            "ns_compute": {},
        }
    )
    table.activations.component["expert"] = None
    table.activations.component["C_block"] = "tp"
    rules = from_config(table, mesh, _sites())
    assert rules.activations.component.assignment("C") == ("tp",)
    assert rules.activations.component.assignment("expert") == ()
    assert rules.activations.component.assignment("C_block") == ("tp",)
    planned = BlockedPlannedComponentLinear(
        v=rules.blocked_component_linear_plan(rules.target.column, "fused_output", "V", 3),
        u=rules.blocked_component_linear_plan(rules.target.column, "fused_output", "U", 3),
        component=rules.activations.component,
        output=rules.activations.external,
    )
    key = random.PRNGKey(81)
    x = random.normal(random.fold_in(key, 0), (B, T, D))
    v = random.normal(random.fold_in(key, 1), (E, D, CPB)) * 0.3
    u = random.normal(random.fold_in(key, 2), (E, CPB, DI)) * 0.3
    mask = random.uniform(random.fold_in(key, 4), (B, T, C))
    W = random.normal(random.fold_in(key, 3), (E * DI, D))

    def evaluate(
        v: Array,
        x: Array,
        u: Array,
        mask: Array,
        plan: BlockedPlannedComponentLinear | None,
    ) -> tuple[Array, Array]:
        result = blocked_site_forward(
            x, BlockedSiteWeights(W, v, u, None, plan), mask, None, None, "fused_output"
        )
        return result.output, result.component_activation

    expected, expected_activation = evaluate(v, x, u, mask, None)
    expected_grad = jax.grad(lambda vv: jnp.sum(evaluate(vv, x, u, mask, None)[0] ** 2))(v)
    put = lambda value, spec: jax.device_put(value, NamedSharding(mesh, spec))
    with jax.set_mesh(mesh):
        args = (
            put(v, P(None, None, "tp")),
            put(x, P(("replicate", "fsdp"), None, None)),
            put(u, P(None, "tp", None)),
            put(mask, P(("replicate", "fsdp"), None, "tp")),
        )

        def placed(v: Array, x: Array, u: Array, mask: Array):
            return evaluate(v, x, u, mask, planned)

        actual, actual_activation = jax.jit(placed)(*args)
        grad = jax.jit(jax.grad(lambda v, x, u, mask: jnp.sum(placed(v, x, u, mask)[0] ** 2)))(
            *args
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual_activation, expected_activation, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-6)
    assert actual_activation.sharding.is_equivalent_to(
        rules.frequency_sharding.update(spec=P(("replicate", "fsdp"), None, "tp")),
        ndim=3,
    )


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_explicit_table_places_blocked_rows():
    """An explicit table may shard the `expert` axis (component ownership by block,
    the analog of dense `C: tp`); the census carries each group's factorization and the
    placed init lands 4-D leaves on the declared rows."""
    devices = np.asarray(jax.devices()[:4]).reshape(1, 2, 2)
    mesh = Mesh(devices, ("replicate", "fsdp", "tp"), axis_types=(AxisType.Explicit,) * 3)
    table = _blocked_table(
        {
            "optimizer_state": {"expert": "tp", "d_in": "fsdp"},
            "compute_weights": {"expert": "tp", "d_in": "fsdp"},
            "faithfulness_weights": {"expert": "tp", "d_in": "fsdp"},
            "faithfulness_deltas": {"expert": "tp"},
            "operands": {"expert": "tp"},
            "ns_compute": {},
        }
    )
    sites = _sites()
    rules = from_config(table, mesh, sites)
    census = rules.components.group_census
    assert census["gate"] == dataclasses.replace(census["gate"], factorization=GATE_FACTORIZATION)
    assert census["gate"].stack_len == 1 and census["gate"].ns_stack_len == E
    assert rules.components.optimizer_state.spec_for(BLOCKED_V_AXES) == P(None, "tp", "fsdp", None)

    stacks = init_component_stacks_placed(sites, random.PRNGKey(0), rules)
    Vs, Us = stacks.stacks["gate"]
    assert Vs.shape == (1, E, D, CPB) and Us.shape == (1, E, CPB, DI)
    row = rules.components.optimizer_state
    assert Vs.sharding.is_equivalent_to(NamedSharding(mesh, row.spec_for(BLOCKED_V_AXES)), Vs.ndim)


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_blocked_placement_refusals():
    """Fail-closed at construction: a preset keyed on `C` refuses a block-factored-only
    site set (no tensor consumes `C` at the component rows), and an `expert` split the
    block count does not tile refuses with the divisibility message."""
    devices = np.asarray(jax.devices()[:4]).reshape(1, 2, 2)
    mesh = Mesh(devices, ("replicate", "fsdp", "tp"), axis_types=(AxisType.Explicit,) * 3)
    with pytest.raises(AssertionError, match="name no semantic axis"):
        from_config("owner", mesh, _sites())
    three_blocks = BlockedFactorization(n_blocks=3, d_in=D, d_out=DI, c_per_block=CPB)
    table = _blocked_table(
        {
            "optimizer_state": {"expert": "tp"},
            "compute_weights": {"expert": "tp"},
            "faithfulness_weights": {"expert": "tp"},
            "faithfulness_deltas": {"expert": "tp"},
            "operands": {"expert": "tp"},
            "ns_compute": {},
        }
    )
    with pytest.raises(AssertionError, match="does not tile"):
        from_config(table, mesh, (SiteSpec(GATE, three_blocks, "gate"),))
