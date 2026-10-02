"""Forcing function for the generic model-I/O seam.

The trainer's `[B,T,d]` residual is the fixed waist; only three EDGES are generic — the
model INPUT (the opaque batch `clean_forward` / `masked_forward` consume), the model
OUTPUT (`ForwardResult[Out]`, `Out` the target's declared type — bound here to a tuple of
heads; the trainer never inspects an output, it acts on one only through the target's own
`recon_loss_fn` and `pin_output_batch`), and the recon comparison
(`DecomposedModel.recon_loss_fn`).
The trainable components are NOT a generic edge: every target carries the universal
`ComponentStacks` V/U pytree, so this synthetic target uses it too. This builds a tiny non-LM
target that bends the three real edges at once:

  * INPUT  — a dict `{"feat": [B,T,d], "gain": [B,T]}` rather than token ids.
  * OUTPUT — a tuple `(coords [B,T,k], aux [B,T,m])` rather than `[B,T,vocab]` logits,
    its two output operations written head-by-head.
  * LOSS   — a geometric MSE over the tuple rather than `kl_per_position`.

The site machinery is genuine: a real `[B,T,d]` residual, one decomposed site with V/U,
`[B,T,C]` masks, the frozen `x @ W` path for absent sites, real `weight_deltas`. It then
drives the actual `make_train_step` through a couple of steps and asserts the loss is
finite and the trainable state moves — locking the genericity against silent regression
to LM-only. The LM neutrality of these edges is proved separately by the stacked-parity /
equivalence goldens passing unchanged.

The same target is POSITIONED, so it also pins the shipped fast-tier eval kernels over the
positioned non-categorical combination — the one neither the LM binder (KL over logits)
nor the toy binder (positionless) covers.
"""

from collections.abc import Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.ci_l0_eval import make_ci_l0_eval_step
from param_decomp.core.components import (
    ComponentStacks,
    DenseFactorization,
    SiteCI,
    SiteSpec,
    component_stacks_from_sites,
    require_full_emission,
    site_stack_indices_for,
)
from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.masking import all_live_masking_no_delta, materialize_masking
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    DecomposedModel,
    ForwardResult,
    Masking,
    MaterializedMasking,
    PlacedModel,
    SiteRoutes,
    StochasticMasking,
)
from param_decomp.core.objective import PDObjective, build_objective
from param_decomp.core.placement import PlacementRules, from_config
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.recon_eval import FreshPGDReconEval, make_fresh_pgd_eval_step
from param_decomp.core.run_state import _adamw_optimizer
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

B, T, D, C = 2, 3, 8, 5
K_COORDS, M_AUX = 4, 2
SITE = "block.0.proj"

type SyntheticOutput = tuple[Array, Array]
"""The synthetic target's `Out`: `(coords [.., k], aux [.., m])`, two heads sharing one
leading prefix."""


def _untype(value: Array) -> Array:
    """Drop axis typing on this fixture's one-device mesh arm (identity there): bare
    matmuls cannot resolve weight grads against an axis-typed batch under Explicit."""
    sharding = jax.typeof(value).sharding
    if sharding.mesh.empty:
        return value
    return jax.sharding.reshard(
        value, jax.sharding.NamedSharding(sharding.mesh, P(*([None] * value.ndim)))
    )


class SyntheticDecomposedModel(eqx.Module):
    """A non-LM `DecomposedModel`: dict input, tuple `(coords, aux)` output, geometric-MSE
    recon. Carries its frozen target weights (`feat_proj` + `W` + two readout heads) as
    array fields; the trainable V/U (the universal `ComponentStacks`) stays an explicit method arg."""

    feat_proj: Float[Array, "D D"]
    W: Float[Array, "D D"]
    read_coords: Float[Array, "K D"]
    read_aux: Float[Array, "M D"]
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, placement: PlacementRules) -> "SyntheticDecomposedModel":
        repl = jax.sharding.NamedSharding(placement.mesh, jax.sharding.PartitionSpec())
        return jax.tree.map(lambda _a: repl, self)

    @staticmethod
    def recon_loss_fn(
        masked_output: SyntheticOutput, clean_output: SyntheticOutput
    ) -> Float[Array, ""]:
        """Non-KL recon: mean squared error over both tuple heads, fp32, per position."""
        coords_err = (
            masked_output[0].astype(jnp.float32) - clean_output[0].astype(jnp.float32)
        ) ** 2
        aux_err = (masked_output[1].astype(jnp.float32) - clean_output[1].astype(jnp.float32)) ** 2
        return (jnp.sum(coords_err) + jnp.sum(aux_err)) / (B * T)

    @staticmethod
    def pin_output_batch(output: SyntheticOutput, mesh: Mesh | None) -> SyntheticOutput:
        coords, aux = output
        return batch_shard_leading(coords, mesh), batch_shard_leading(aux, mesh)

    def _heads(self, hidden: Array) -> SyntheticOutput:
        # Untyped output edge: cotangents flowing back must be axis-free too.
        hidden = _untype(hidden)
        return hidden @ self.read_coords.T, hidden @ self.read_aux.T

    def _residual(self, inputs: dict[str, Array]) -> Float[Array, "B T D"]:
        """Input edge: the loader's native DICT batch -> the `[B,T,D]` residual (not token ids).

        This fixture's linears are bare matmuls, which cannot resolve a weight-grad
        against an axis-typed batch under the Explicit mesh; its mesh arm is pinned to
        one device (partitioning is tested elsewhere), where dropping the vacuous
        typing is the identity."""
        return _untype((inputs["feat"] @ self.feat_proj.T) * inputs["gain"][..., None])

    def _ordered_capture_keys(self, keys: CaptureKeys) -> tuple[str, ...]:
        allowed = {SITE, f"{SITE}.out"}
        assert keys <= allowed, (keys, allowed)
        return tuple(sorted(keys))

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        assert sites == (SITE,), sites
        return (f"{SITE}.out",)

    def clean_forward(
        self,
        inputs: dict[str, Array],
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult[SyntheticOutput, dict[str, Array]]:
        del placement
        ordered_capture_keys = self._ordered_capture_keys(capture_keys)
        residual = self._residual(inputs)
        if not ordered_capture_keys:
            return ForwardResult.from_producer(
                leading_shape=residual.shape[:-1],
                output=self._heads(residual @ self.W.T),
                capture_keys=ordered_capture_keys,
                capture_values=(),
                conditioning=inputs,
                sequence=SequenceLayout(jnp.zeros_like(residual[..., 0], dtype=jnp.int32)),
            )
        hidden = residual @ self.W.T
        values = {SITE: residual, f"{SITE}.out": hidden}
        return ForwardResult.from_producer(
            leading_shape=residual.shape[:-1],
            output=self._heads(hidden),
            capture_keys=ordered_capture_keys,
            capture_values=tuple(values[key] for key in ordered_capture_keys),
            conditioning=inputs,
            sequence=SequenceLayout(jnp.zeros_like(residual[..., 0], dtype=jnp.int32)),
        )

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
    ) -> tuple[ForwardResult[SyntheticOutput, dict[str, Array]], dict[str, SiteCI]]:
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
    ) -> ForwardResult[SyntheticOutput, dict[str, Array]]:
        del placement, remat
        ordered_capture_keys = self._ordered_capture_keys(capture_keys)
        assert tuple(masking.component_masks) == (SITE,)
        assert routes is None
        residual = self._residual(inputs)
        site_components = vu.site(SITE)
        mask = _untype(require_full_emission(masking.component_masks[SITE]))
        hidden = (residual @ site_components.V) * mask @ site_components.U
        if masking.weight_delta_masks is not None:
            delta = self.W - (site_components.V @ site_components.U).T
            hidden = hidden + _untype(masking.weight_delta_masks[SITE])[..., None] * (
                residual @ delta.T
            )
        if not ordered_capture_keys:
            return ForwardResult.from_producer(
                leading_shape=residual.shape[:-1],
                output=self._heads(hidden),
                capture_keys=ordered_capture_keys,
                capture_values=(),
                conditioning=inputs,
                sequence=SequenceLayout(jnp.zeros_like(residual[..., 0], dtype=jnp.int32)),
            )
        values = {SITE: residual, f"{SITE}.out": hidden}
        return ForwardResult.from_producer(
            leading_shape=residual.shape[:-1],
            output=self._heads(hidden),
            capture_keys=ordered_capture_keys,
            capture_values=tuple(values[key] for key in ordered_capture_keys),
            conditioning=inputs,
            sequence=SequenceLayout(jnp.zeros_like(residual[..., 0], dtype=jnp.int32)),
        )

    def target_weight_sq_norms(self) -> dict[str, Array]:
        ((_name, group, slot),) = site_stack_indices_for(self.sites)
        assert slot == 0
        return {group: jnp.sum(self.W.astype(jnp.float32) ** 2)[None]}

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Array]:
        site_components = vu.site(SITE)
        delta = (
            self.W.astype(jnp.float32)
            - (site_components.V.astype(jnp.float32) @ site_components.U.astype(jnp.float32)).T
        )
        shape, slot = vu.stack_index_of(SITE)
        assert slot == 0
        return {shape: delta[None]}


def test_all_live_masking_uses_each_site_component_count() -> None:
    sites = (
        SiteSpec("a", DenseFactorization(d_in=3, d_out=4, C=2), "a"),
        SiteSpec("b", DenseFactorization(d_in=4, d_out=5, C=7), "b"),
    )
    masking = all_live_masking_no_delta(sites, leading_shape=(2, 3), dtype=jnp.bfloat16)

    assert tuple(masking.component_masks) == ("a", "b")
    masks = {name: require_full_emission(m) for name, m in masking.component_masks.items()}
    assert masks["a"].shape == (2, 3, 2)
    assert masks["b"].shape == (2, 3, 7)
    assert all(mask.dtype == jnp.bfloat16 for mask in masks.values())
    assert all(bool(jnp.all(mask == 1)) for mask in masks.values())


def _synthetic_lm(key: jax.Array) -> SyntheticDecomposedModel:
    return SyntheticDecomposedModel(
        feat_proj=random.normal(random.fold_in(key, 7), (D, D)),
        W=random.normal(random.fold_in(key, 0), (D, D)),
        read_coords=random.normal(random.fold_in(key, 1), (K_COORDS, D)),
        read_aux=random.normal(random.fold_in(key, 2), (M_AUX, D)),
        sites=(
            SiteSpec(name=SITE, factorization=DenseFactorization(d_in=D, d_out=D, C=C), group=SITE),
        ),
        has_position_axis=True,
    )


def _synthetic_vu(key: jax.Array) -> ComponentStacks:
    V = random.normal(random.fold_in(key, 3), (D, C)) * 0.1
    U = random.normal(random.fold_in(key, 4), (C, D)) * 0.1
    return component_stacks_from_sites({SITE: (V, U)})


def test_prepare_compute_weights_owns_the_compute_dtype_boundary() -> None:
    components = _synthetic_vu(random.PRNGKey(0))
    prepared_weights = PlacedModel(
        model=_synthetic_lm(random.PRNGKey(1)), placement=None
    ).prepare_compute_weights(components)

    assert {leaf.dtype for leaf in jax.tree.leaves(prepared_weights)} == {jnp.dtype(COMPUTE_DT)}


def _synthetic_ci_fn_arch() -> ChunkwiseTransformerCIFnArch:
    return ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(input_taps=(SITE,), output_sites=(SITE,)),),
        input_dim=D,
        d_model=8,
        n_blocks=1,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        ffn_hidden=16,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def _synthetic_inputs(key: jax.Array) -> dict[str, Array]:
    return {
        "feat": random.normal(random.fold_in(key, 5), (B, T, D)),
        "gain": random.uniform(random.fold_in(key, 6), (B, T)),
    }


def _one_device_mesh() -> jax.sharding.Mesh:
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1, 1)
    return jax.sharding.Mesh(
        devices, ("replicate", "fsdp", "tp"), axis_types=(AxisType.Explicit,) * 3
    )


def test_dict_input_tuple_output_and_geometric_loss_flow():
    """The model consumes the loader's native DICT batch (not token ids);
    clean/masked `ForwardResult.output` is a tuple; `recon_loss_fn` contracts it."""
    key = random.PRNGKey(1)
    model = _synthetic_lm(key)
    components = _synthetic_vu(key)
    inputs = _synthetic_inputs(key)

    assert model.clean_forward(inputs, frozenset({SITE}), placement=None).captures[SITE].shape == (
        B,
        T,
        D,
    )

    clean = model.clean_forward(inputs, placement=None)
    clean_output = clean.output
    assert isinstance(clean_output, tuple) and len(clean_output) == 2
    assert clean_output[0].shape == (B, T, K_COORDS) and clean_output[1].shape == (B, T, M_AUX)

    masks = {SITE: jnp.ones((B, T, C))}
    masked_output = model.masked_forward(
        components,
        clean.conditioning,
        masking=model.prepare_masking(MaterializedMasking(component_masks=masks)),
        routes=None,
        placement=None,
        remat=False,
    ).output
    assert isinstance(masked_output, tuple) and len(masked_output) == 2

    loss = model.recon_loss_fn(masked_output, clean_output)
    assert loss.shape == () and jnp.isfinite(loss)


def _initial_state(
    model: DecomposedModel[
        dict[str, Array], SyntheticOutput, ComponentStacks, dict[str, Array], MaterializedMasking
    ],
    components: ComponentStacks,
    ci_fn_arch: ChunkwiseTransformerCIFnArch,
    objective: PDObjective,
    rules: PlacementRules | None,
):
    opt_vu = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-2)), 1)
    opt_ci = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-2)), 1)
    ci_fn = ci_fn_arch.initialize(model.sites, rules, random.PRNGKey(11))
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
    return state, opt_vu, opt_ci


@pytest.mark.parametrize("with_mesh", [False, True])
def test_train_step_runs_through_generic_target(with_mesh: bool):
    """End-to-end: the real `make_train_step` drives the synthetic dict-in/tuple-out/MSE
    target for two steps; the loss stays finite and the trainable V/U actually move.

    Run meshless and with an explicit one-device mesh: only the latter exercises
    `NamedSharding`, while pinning one device keeps this generic-I/O contract independent
    of the ambient device count. Multi-device partitioning is tested separately."""
    key = random.PRNGKey(2)
    model = _synthetic_lm(key)
    components = _synthetic_vu(key)
    inputs = _synthetic_inputs(key)

    loss_terms = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-4,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                ),
            ),
            StochasticReconLossConfig(coeff=1.0),
        ),
        model.sites,
    )
    mesh = _one_device_mesh() if with_mesh else None
    rules = None if mesh is None else from_config("ddp", mesh, model.sites)
    state, opt_vu, opt_ci = _initial_state(
        model, components, _synthetic_ci_fn_arch(), loss_terms, rules
    )
    placed = PlacedModel(model=model, placement=rules)
    step_fn = jax.jit(
        make_train_step(
            model_static=placed,
            substrate=ForwardSubstrate.of(
                placed,
                remat_recon_forwards=False,
                remat_ci_fn=False,
                ci_capture_keys=frozenset({SITE}),
            ),
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=10,
            faithfulness=faithfulness_loss_for(placed),
        )
    )

    V_before = jax.device_get(
        state.decomposition.components.site(SITE).V
    )  # host copy survives step donation
    run_key = random.PRNGKey(3)
    for step_idx in range(2):
        state, metrics = step_fn(placed, state, inputs, random.fold_in(run_key, step_idx))
        assert jnp.isfinite(metrics["total"]), (step_idx, metrics["total"])
        assert "loss/StochasticReconLoss" in metrics

    assert not jnp.allclose(state.decomposition.components.site(SITE).V, V_before), (
        "V did not move — step is a no-op"
    )


def test_fast_eval_metrics_bind_to_positioned_non_categorical_target():
    """The shipped fast tier serves a target that is POSITIONED and NON-categorical at once
    — the combination the LM kernels (KL over logits) and the toy kernels (positionless)
    each exclude. `PGDReconLoss` scores through the target's own MSE `recon_loss_fn`;
    `CI_L0` reads the CI envelope alone, so both drop in with no target-side authoring."""
    key = random.PRNGKey(4)
    model = _synthetic_lm(key)
    assert model.has_position_axis
    components = _synthetic_vu(key)
    inputs = _synthetic_inputs(key)
    ci_fn = _synthetic_ci_fn_arch().initialize(model.sites, None, random.PRNGKey(11))
    placed = PlacedModel(model=model, placement=None)

    pgd_step = jax.jit(
        make_fresh_pgd_eval_step(
            placed,
            FreshPGDReconEval(n_steps=2, step_size=0.1),
            ci_fn.capture_keys,
        )
    )
    pgd = pgd_step(placed, components, ci_fn, inputs, random.PRNGKey(5))
    assert pgd.shape == ()
    assert jnp.isfinite(pgd) and pgd >= 0.0

    l0_step = jax.jit(
        make_ci_l0_eval_step(placed, ci_fn.capture_keys, 0.5, {"block": ("block.*",)})
    )
    l0 = l0_step(placed, components, ci_fn, inputs, random.PRNGKey(6))
    assert set(l0) == {f"l0/0.5_{SITE}", "l0/0.5_block"}
    assert all(value.shape == () and 0.0 <= value <= C for value in l0.values())
    assert l0["l0/0.5_block"] == l0[f"l0/0.5_{SITE}"], "single-member group sums to its site"


@pytest.mark.parametrize("reconstruction_keys", [frozenset(), frozenset({f"{SITE}.out"})])
def test_stream_shape_does_not_require_ci_taps(reconstruction_keys: CaptureKeys):
    target = _synthetic_lm(random.PRNGKey(0))
    inputs = _synthetic_inputs(random.PRNGKey(1))
    model = PlacedModel(model=target, placement=None)
    substrate = ForwardSubstrate.of(
        model,
        remat_recon_forwards=False,
        remat_ci_fn=False,
        ci_capture_keys=frozenset(),
    )

    @jax.jit
    def prepare_stream(batch: dict[str, Array]) -> SyntheticOutput:
        stream = substrate.prep_stream(model, batch, reconstruction_keys)
        assert stream.taps == {}
        assert stream.leading == (B, T)
        assert set(stream.clean.captures) == reconstruction_keys
        return stream.clean.output

    output = prepare_stream(inputs)
    expected = model.clean_forward(inputs, reconstruction_keys).output
    for actual, wanted in zip(output, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
