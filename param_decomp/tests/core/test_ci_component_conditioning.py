"""CI conditioned on live component activations, around the global transformer: the
readout, its calibration to data, and V's shared gradient."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from param_decomp.core.ci_fn.implementations.global_transformer.arch import (
    GlobalTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.backbone import BackboneCIFn
from param_decomp.core.ci_fn.implementations.transformer.component_conditioning import (
    CALIBRATION_CHUNK_TOKENS,
    ClampedAffineArm,
    ComponentActivationReadout,
    ComponentConditioned,
    ConditionedCIFnArch,
    InputScaleCalibration,
    SiteInput,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.ci_fn.interface import CI, CIFn, TapSpec
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.ci_fn.squashing import symmetric_leaky_hard_sigmoid
from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    SiteSpec,
    init_component_stacks,
    require_full_emission,
)
from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.init_placed import init_component_stacks_placed
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.model import MaterializedMasking, PlacedModel, Positioned
from param_decomp.core.objective import build_objective
from param_decomp.core.placement import from_config
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import hsdp_mesh, place_target, shard_batch
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    PDTrainingState,
    StreamInputs,
    TrainState,
    _grad_norm_metrics,
    make_train_step,
)
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.metric_taxonomy import grouped_metric_key
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer import (
    KIND_ORDER,
    TransformerDecomposedModel,
    TransformerPreparedMasking,
    TransformerPreparedWeights,
    glu_site_specs,
    site_name,
)
from param_decomp.targets.transformer_taps import resid_tap_key
from param_decomp.tests.placed_ci_fn import placed_ci_fn

type PlacedGlu = PlacedModel[
    LMBatchWithDocuments,
    LMOutput,
    TransformerPreparedWeights,
    LMBatchWithDocuments,
    TransformerPreparedMasking,
]
LAYERS = (3, 4)


def _sites() -> tuple[SiteSpec, ...]:
    """Every kind, so the clean inputs span the attention input and output and the MLP
    input and hidden captures."""
    return glu_site_specs(
        tiny_glu_cfg(),
        tuple(SiteC(site_name(layer, kind), 8) for layer in LAYERS for kind in KIND_ORDER),
    )


def _model(sites: tuple[SiteSpec, ...]) -> TransformerDecomposedModel:
    return tiny_glu_decomposed_lm(tiny_glu_cfg(), sites, jax.random.PRNGKey(0))


def _conditioned_arch(
    model: TransformerDecomposedModel,
) -> ConditionedCIFnArch[GlobalTransformerCIFnArch]:
    return ConditionedCIFnArch(
        GlobalTransformerCIFnArch(
            input_taps=tuple(
                TapSpec(resid_tap_key(layer), tiny_glu_cfg().n_embd) for layer in (*LAYERS, 5)
            ),
            d_model=16,
            n_blocks=2,
            attention=MHACIFnAttention(n_heads=2, implementation="xla", mask="causal"),
            ffn_hidden=32,
            ffn_kind="gelu",
            learned_norm_scale=False,
        ),
        tuple(SiteInput(name, model.anatomy.site_input_key(name)) for name in model.site_names),
        output_scale_init=0.5,
        calibration=InputScaleCalibration(min_n_tokens=64),
    )


def _batch(batch_size: int) -> LMBatchWithDocuments:
    tokens = jax.random.randint(
        jax.random.PRNGKey(4), (batch_size, 16), 0, tiny_glu_cfg().vocab_size
    )
    return LMBatchWithDocuments.from_unsegmented_sequences(tokens)


def _conditioned(fn: object) -> ComponentConditioned:
    assert isinstance(fn, BackboneCIFn)
    backbone = fn.backbone
    assert isinstance(backbone, ComponentConditioned)
    return backbone


def _preactivation_cotangents(fn: CIFn[Any], leading: tuple[int, ...]) -> dict[str, Array]:
    readouts = _conditioned(fn).readouts
    return {
        site: jax.random.normal(
            jax.random.fold_in(jax.random.PRNGKey(5), index),
            (*leading, readout.bias.shape[0]),
        )
        for index, (site, readout) in enumerate(sorted(readouts.items()))
    }


def _inside_clamp(
    fn: BackboneCIFn, components: ComponentStacks, captures: dict[str, Array]
) -> BackboneCIFn:
    """Distinct readout vectors whose clamp inputs lie strictly inside (0, 1) on these
    captures, where each clamp is the identity and its gradient exact."""

    def rescaled(site: str, readout: ComponentActivationReadout) -> ComponentActivationReadout:
        activation = captures[readout.capture_key] @ components.site(site).V
        width = readout.bias.shape[0]
        input_scale = jnp.full((width,), 0.4) / jnp.max(jnp.abs(activation))
        return ComponentActivationReadout(
            positive=ClampedAffineArm(
                input_scale, jnp.full((width,), 0.5), jnp.linspace(0.5, 1.5, width)
            ),
            negative=ClampedAffineArm(
                0.8 * input_scale, jnp.full((width,), 0.45), jnp.linspace(1.2, -0.3, width)
            ),
            bias=jnp.linspace(-0.2, 0.2, width),
            capture_key=readout.capture_key,
        )

    readouts = _conditioned(fn).readouts
    return eqx.tree_at(
        lambda f: _conditioned(f).readouts,
        fn,
        {site: rescaled(site, readout) for site, readout in readouts.items()},
    )


def _pullback(
    substrate: ForwardSubstrate[Any, Any, Any, Any, Any],
    target: PlacedGlu,
    ci_fn: CIFn[Any],
    components: ComponentStacks,
    stream: StreamInputs[LMOutput, LMBatchWithDocuments],
    cotangents: dict[str, Array],
) -> tuple[dict[str, Array], CIFn[Any], ComponentStacks]:
    """The step's CI pullbacks for `sum(cotangent * preactivation)`: onto the CI masters,
    and through the target's prepared components onto the component masters."""
    prepared, prepared_vjp = substrate.component_weights_vjp(target, components)
    compute, weights_vjp = substrate.ci_fn_prepare_vjp(ci_fn)

    def objective(
        compute_ci_fn: CIFn[Any], prepared_weights: TransformerPreparedWeights
    ) -> tuple[Array, dict[str, Array]]:
        ci = evaluate_ci_from_captures(
            compute_ci_fn,
            stream.taps,
            stream.conditioning,
            prepared_weights,
            sequence=stream.sequence,
            remat=False,
        )
        preactivations = {
            site: jnp.asarray(value, jnp.float32) for site, value in ci.preactivations.items()
        }
        total = sum(jnp.sum(cotangents[site] * value) for site, value in preactivations.items())
        return jnp.asarray(total), preactivations

    (compute_grad, prepared_grad), preactivations = eqx.filter_grad(
        lambda both: objective(*both), has_aux=True
    )((compute, prepared))
    (ci_grad,) = weights_vjp(compute_grad)
    (components_grad,) = prepared_vjp(prepared_grad)
    return preactivations, ci_grad, components_grad


def _stream(
    target: PlacedGlu, ci_fn: CIFn[Any], batch: LMBatchWithDocuments
) -> StreamInputs[LMOutput, LMBatchWithDocuments]:
    substrate = ForwardSubstrate.of(
        target, remat_recon_forwards=False, remat_ci_fn=False, ci_capture_keys=ci_fn.capture_keys
    )
    return substrate.prep_stream(target, batch, frozenset())


def _reconstruction_and_minimality(
    target: PlacedGlu,
    stream: StreamInputs[LMOutput, LMBatchWithDocuments],
    prepared: TransformerPreparedWeights,
    ci: CI,
) -> Array:
    """A masked reconstruction whose masks are the lower CI, plus the upper CI's mean: V
    reaches the loss through the masked forward and through the CI readout."""
    masked = target.masked_forward(
        prepared,
        stream.conditioning,
        masking=target.model.prepare_masking(MaterializedMasking(component_masks=ci.lower)),
        routes=None,
        remat=False,
    )
    minimality = sum(jnp.mean(require_full_emission(value)) for value in ci.upper.values())
    return target.recon_loss_fn(masked.output, stream.clean.output) + minimality


def _engine_and_reference_v_gradients(
    target: PlacedGlu, ci_fn: CIFn[Any], components: ComponentStacks, batch: LMBatchWithDocuments
) -> tuple[ComponentStacks, ComponentStacks, ComponentStacks]:
    """V's gradient as the train step assembles it — the CI forward's cotangent into the
    prepared components joins the reconstruction's before the one pullback onto the
    masters — beside plain autodiff of the same loss, and the CI path's share alone."""
    substrate = ForwardSubstrate.of(
        target, remat_recon_forwards=False, remat_ci_fn=False, ci_capture_keys=ci_fn.capture_keys
    )
    stream = _stream(target, ci_fn, batch)
    prepared, prepared_vjp = substrate.component_weights_vjp(target, components)
    compute_ci_fn, _ = substrate.ci_fn_prepare_vjp(ci_fn)
    ci, ci_fn_forward_vjp = substrate.ci_fn_forward_vjp(compute_ci_fn, prepared, stream)
    recon_grad, ci_grad = eqx.filter_grad(
        lambda both: _reconstruction_and_minimality(target, stream, *both)
    )((prepared, ci))
    _, prepared_grad_from_ci_fn = ci_fn_forward_vjp(ci_grad)
    (engine,) = prepared_vjp(jax.tree.map(lambda a, b: a + b, recon_grad, prepared_grad_from_ci_fn))
    (ci_path,) = prepared_vjp(prepared_grad_from_ci_fn)

    def reference_loss(vu: ComponentStacks) -> Array:
        reference_prepared = target.prepare_compute_weights(vu)
        reference_ci = substrate.shard_ci(
            evaluate_ci_from_captures(
                ci_fn.prepare(),
                stream.taps,
                stream.conditioning,
                reference_prepared,
                sequence=stream.sequence,
                remat=False,
            )
        )
        return _reconstruction_and_minimality(target, stream, reference_prepared, reference_ci)

    return engine, jax.grad(reference_loss)(components), ci_path


def _assert_v_gradients_match_the_reference(
    sites: tuple[SiteSpec, ...],
    engine: ComponentStacks,
    reference: ComponentStacks,
    ci_path: ComponentStacks,
) -> None:
    for site in sites:
        got, expected = engine.site(site.name).V, reference.site(site.name).V
        relative_error = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert relative_error < 1e-2, (site.name, relative_error)
        assert jnp.linalg.norm(ci_path.site(site.name).V) > 0, site.name


def test_v_gradient_through_reconstruction_and_ci_matches_plain_autodiff():
    sites = _sites()
    model = _model(sites)
    arch = _conditioned_arch(model)
    target = PlacedModel(model=model, placement=None)
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = arch.initialize(sites, None, jax.random.PRNGKey(2))

    gradients = eqx.filter_jit(_engine_and_reference_v_gradients)(
        target, ci_fn, components, _batch(2)
    )

    _assert_v_gradients_match_the_reference(sites, *gradients)


type ReadoutVectors = tuple[Array, Array, Array, Array, Array, Array, Array]
"""`a₊, b₊, s₊, a₋, b₋, s₋, bias`."""


def _reference_readout(arms: ReadoutVectors, h: Array) -> Array:
    a_pos, b_pos, s_pos, a_neg, b_neg, s_neg, bias = arms
    positive = s_pos * symmetric_leaky_hard_sigmoid(a_pos * h + b_pos)
    negative = s_neg * symmetric_leaky_hard_sigmoid(a_neg * -h + b_neg)
    return positive + negative + bias


def _arms(readout: ComponentActivationReadout) -> ReadoutVectors:
    positive, negative = readout.positive, readout.negative
    return (
        positive.input_scale,
        positive.input_bias,
        positive.output_scale,
        negative.input_scale,
        negative.input_bias,
        negative.output_scale,
        readout.bias,
    )


def _reference_objective(V: Array, arms: ReadoutVectors, x: Array, cotangent: Array) -> Array:
    return jnp.sum(cotangent * _reference_readout(arms, x @ V))


def test_readout_values_and_gradients_match_a_plain_jax_reference():
    """Through the engine's CI pullbacks, the readout's value and its gradients onto V
    and every readout vector match the formula written in plain JAX."""
    sites = _sites()
    model = _model(sites)
    arch = _conditioned_arch(model)
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    placed = PlacedModel(model=model, placement=None)
    clean = model.clean_forward(_batch(2), arch.capture_keys, placement=None)
    fn = _inside_clamp(
        arch.initialize(sites, None, jax.random.PRNGKey(2)), components, clean.captures
    )
    conditioned = _conditioned(fn)
    cotangents = _preactivation_cotangents(fn, clean.leading_shape)
    substrate = ForwardSubstrate.of(
        placed, remat_recon_forwards=False, remat_ci_fn=False, ci_capture_keys=arch.capture_keys
    )
    backbone_preactivations = conditioned.backbone.preactivations(
        cast_floating(clean.captures, COMPUTE_DT),
        None,
        placed.prepare_compute_weights(components),
        sequence=clean.sequence,
        remat=False,
    )

    preactivations, ci_grad, components_grad = _pullback(
        substrate, placed, fn, components, _stream(placed, fn, _batch(2)), cotangents
    )

    readout_grads = _conditioned(ci_grad).readouts
    for site, readout in conditioned.readouts.items():
        x = clean.captures[readout.capture_key]
        V = components.site(site).V
        value = _reference_readout(_arms(readout), x @ V)
        V_grad, arms_grad = jax.grad(_reference_objective, argnums=(0, 1))(
            V, _arms(readout), x, cotangents[site]
        )
        got_value = preactivations[site] - require_full_emission(backbone_preactivations[site])
        for got, expected in (
            (got_value, value),
            (components_grad.site(site).V, V_grad),
            *zip(_arms(readout_grads[site]), arms_grad, strict=True),
        ):
            relative_error = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
            assert relative_error < 3e-2, (site, relative_error)
        np.testing.assert_array_equal(np.asarray(components_grad.site(site).U), 0)


def _reference_magnitude_quantile(h: Array) -> np.ndarray:
    """Each component's 99.9th percentile of `|h|` over every token, as numpy computes it."""
    magnitudes = np.abs(np.asarray(h, np.float32)).reshape(-1, h.shape[-1])
    return np.quantile(magnitudes, 0.999, axis=0)


def _assert_exact_quantile_at(topology: tuple[int, int, int]) -> None:
    """Two chunks of heavy-tailed activations: exponentiating the clean inputs makes them
    lognormal-like, so `|h|`'s tail sits far above its bulk."""
    sites = _sites()
    model = _model(sites)
    arch = _conditioned_arch(model)
    batch = _batch(2 * CALIBRATION_CHUNK_TOKENS // 16)
    mesh = hsdp_mesh(*topology)
    rules = from_config("owner", mesh, sites)

    def calibrated_and_activations(
        initial: ComponentConditioned,
        target: PlacedGlu,
        components: ComponentStacks,
        inputs: LMBatchWithDocuments,
    ) -> tuple[ComponentConditioned, dict[str, Array]]:
        captures = target.clean_forward(inputs, arch.capture_keys).captures
        taps = {key: jnp.exp(3 * value) for key, value in captures.items()}
        prepared = target.prepare_compute_weights(components)
        compute_taps = cast_floating(taps, COMPUTE_DT)
        activations = {
            site: prepared.component_activations(site, compute_taps[readout.capture_key])
            for site, readout in initial.readouts.items()
        }
        return initial.with_calibrated_input_scales(taps, prepared), activations

    with jax.set_mesh(mesh):
        target = place_target(model, rules)
        initial = _conditioned(placed_ci_fn(arch, sites, jax.random.PRNGKey(2), mesh, rules))
        components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
        calibrated, activations = eqx.filter_jit(calibrated_and_activations)(
            initial,
            target,
            components,
            jax.tree.map(lambda x: shard_batch(x, mesh, batch_axis=0), batch),
        )
    initial, calibrated, activations = jax.device_get((initial, calibrated, activations))

    for site in sites:
        readout, initial_readout = calibrated.readouts[site.name], initial.readouts[site.name]
        quantile = _reference_magnitude_quantile(activations[site.name])
        for arm, initial_arm in zip(
            (readout.positive, readout.negative),
            (initial_readout.positive, initial_readout.negative),
            strict=True,
        ):
            np.testing.assert_allclose(1 / arm.input_scale, quantile, rtol=1e-6)
            for got, expected in (
                (arm.input_bias, initial_arm.input_bias),
                (arm.output_scale, initial_arm.output_scale),
            ):
                np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
        np.testing.assert_array_equal(np.asarray(readout.bias), np.asarray(initial_readout.bias))


def test_every_arm_maps_the_exact_magnitude_quantile_to_one_on_one_device():
    _assert_exact_quantile_at((1, 1, 1))


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) != 8, reason="requires eight local devices")
def test_every_arm_maps_the_exact_magnitude_quantile_to_one_on_a_placed_mesh():
    _assert_exact_quantile_at((2, 2, 2))


def test_readouts_start_from_the_configured_output_scale():
    sites = _sites()
    conditioned = _conditioned_arch(_model(sites)).initialize_backbone(
        sites, None, jax.random.PRNGKey(2)
    )

    for readout in conditioned.readouts.values():
        for arm in (readout.positive, readout.negative):
            np.testing.assert_array_equal(np.asarray(arm.input_scale), 1)
            np.testing.assert_array_equal(np.asarray(arm.input_bias), 0)
            np.testing.assert_array_equal(np.asarray(arm.output_scale), 0.5)
        np.testing.assert_array_equal(np.asarray(readout.bias), 0)


def test_conditioning_adds_seven_readout_vectors_per_site_to_the_census():
    sites = _sites()
    arch = _conditioned_arch(_model(sites))

    conditioned = arch.parameter_census(sites)
    plain = arch.inner.parameter_census(sites)

    assert conditioned.matrices == plain.matrices
    assert conditioned.n_vector_parameters - plain.n_vector_parameters == 7 * sum(
        site.C for site in sites
    )


def test_conditioning_preserves_the_backbone_initialization():
    sites = _sites()
    arch = _conditioned_arch(_model(sites))
    key = jax.random.PRNGKey(2)

    conditioned = _conditioned(arch.initialize(sites, None, key))
    plain = arch.inner.initialize(sites, None, key)

    for got, expected in zip(
        jax.tree.leaves(conditioned.backbone), jax.tree.leaves(plain.backbone), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_conditioning_adds_each_site_input_projection_to_the_useful_flops():
    sites = _sites()
    arch = _conditioned_arch(_model(sites))
    batch_size, positions = 3, Positioned(n_positions=16)

    conditioned = arch.useful_flops(sites, batch_size, positions, n_selected_blocks_per_token=None)
    plain = arch.inner.useful_flops(sites, batch_size, positions, n_selected_blocks_per_token=None)

    projection = sum(2 * batch_size * 16 * site.d_in * site.C for site in sites)
    assert conditioned.forward - plain.forward == projection
    assert conditioned.backward - plain.backward == projection


def test_every_conditioned_parameter_has_a_grouped_gradient_metric_name():
    sites = _sites()
    fn = _conditioned_arch(_model(sites)).initialize(sites, None, jax.random.PRNGKey(2))

    metrics = _grad_norm_metrics(init_component_stacks(sites, jax.random.PRNGKey(1)), fn, None)

    readout_keys = [key for key in metrics if ".readouts[" in key]
    assert len(readout_keys) == 7 * len(sites)
    for key in metrics:
        if not key.startswith("grad_norms/summary/"):
            grouped_metric_key(f"train/{key}")


@pytest.mark.parametrize("conditioned", [True, False])
def test_minimality_alone_moves_v_exactly_when_ci_is_conditioned(conditioned: bool):
    """With faithfulness and reconstruction weighted zero, V's only gradient is the one
    returned through the CI forward's prepared components; U has none."""
    sites = _sites()
    model = _model(sites)
    conditioned_arch = _conditioned_arch(model)
    arch = conditioned_arch if conditioned else conditioned_arch.inner
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = arch.initialize(sites, None, jax.random.PRNGKey(2))
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=0.0),
            ImportanceMinimalityLossConfig(coeff=1.0, gamma=ScheduleConfig.constant(1.0)),
            StochasticReconLossConfig(coeff=0.0),
        ),
        sites,
    )
    optimizer = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-2)), 1)
    state = TrainState(
        decomposition=Decomposition(components=components, ci_fn=ci_fn),
        training=PDTrainingState(
            frequency=BatchFrequency(),
            objective=objective,
            components_opt_state=optimizer.init(eqx.filter(components, eqx.is_array)),
            ci_fn_opt_state=optimizer.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            step=jnp.zeros((), jnp.int32),
        ),
    )
    placed = PlacedModel(model=model, placement=None)
    step = jax.jit(
        make_train_step(
            model_static=placed,
            substrate=ForwardSubstrate.of(
                placed,
                remat_recon_forwards=False,
                remat_ci_fn=False,
                ci_capture_keys=ci_fn.capture_keys,
            ),
            components_optimizer=optimizer,
            ci_fn_optimizer=optimizer,
            total_steps=10,
            faithfulness=faithfulness_loss_for(placed),
        )
    )

    updated, _ = step(placed, state, _batch(2), jax.random.PRNGKey(100))

    for site in model.site_names:
        before = components.site(site)
        after = updated.decomposition.components.site(site)
        assert bool(jnp.any(after.V != before.V)) == conditioned, site
        np.testing.assert_array_equal(np.asarray(after.U), np.asarray(before.U))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_symmetric_clamp_leaks_only_inward_gradients(dtype: jnp.dtype):
    x = jnp.asarray([-0.5, 0.25, 1.5, -0.5, 0.25, 1.5], dtype)
    g = jnp.asarray([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype)

    value, pullback = jax.vjp(symmetric_leaky_hard_sigmoid, x)

    np.testing.assert_array_equal(np.asarray(value, np.float32), [0, 0.25, 1, 0, 0.25, 1])
    # A descent step moves x by -g: below 0 only g < 0 points inside, above 1 only g > 0.
    np.testing.assert_allclose(
        np.asarray(pullback(g)[0], np.float32), [-0.01, -1, 0, 0, 1, 0.01], rtol=1e-2
    )


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) != 8, reason="requires eight local devices")
def test_placed_conditioning_matches_unplaced_values_and_component_gradients():
    sites = _sites()
    model = _model(sites)
    arch = _conditioned_arch(model)
    batch = _batch(4)
    unplaced = PlacedModel(model=model, placement=None)
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    fn = arch.initialize(sites, None, jax.random.PRNGKey(2))
    clean = model.clean_forward(batch, arch.capture_keys, placement=None)
    fn = _inside_clamp(fn, components, clean.captures)
    cotangents = _preactivation_cotangents(fn, clean.leading_shape)

    def gradients(
        target: PlacedGlu, ci_fn: CIFn[Any], vu: ComponentStacks, inputs: LMBatchWithDocuments
    ) -> tuple[dict[str, Array], ComponentStacks]:
        preactivations, _, components_grad = _pullback(
            ForwardSubstrate.of(
                target,
                remat_recon_forwards=False,
                remat_ci_fn=True,
                ci_capture_keys=arch.capture_keys,
            ),
            target,
            ci_fn,
            vu,
            _stream(target, ci_fn, inputs),
            cotangents,
        )
        return preactivations, components_grad

    expected_values, expected_grads = eqx.filter_jit(gradients)(unplaced, fn, components, batch)
    mesh = hsdp_mesh(2, 2, 2)
    rules = from_config("owner", mesh, sites)
    with jax.set_mesh(mesh):
        placed = place_target(model, rules)
        placed_fn = placed_ci_fn(arch, sites, jax.random.PRNGKey(2), mesh, rules)
        readout_shardings = _conditioned(placed_fn.shardings(mesh)).readouts
        placed_fn = eqx.tree_at(
            lambda f: _conditioned(f).readouts,
            placed_fn,
            jax.device_put(_conditioned(fn).readouts, readout_shardings),
        )
        placed_components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
        values, grads = eqx.filter_jit(gradients)(
            placed,
            placed_fn,
            placed_components,
            jax.tree.map(lambda x: shard_batch(x, mesh, batch_axis=0), batch),
        )

    def relative_error(got: Array, expected: Array) -> Array:
        return jnp.linalg.norm(jnp.asarray(got) - expected) / jnp.linalg.norm(expected)

    host_grads: ComponentStacks = jax.device_get(grads)
    for site in model.site_names:
        assert relative_error(values[site], expected_values[site]) < 1e-2, site
        got, expected = host_grads.site(site).V, expected_grads.site(site).V
        assert relative_error(got, expected) < 3e-2, site


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) != 8, reason="requires eight local devices")
def test_placed_v_gradient_through_reconstruction_and_ci_matches_plain_autodiff():
    """Under placement the prepared V is typed `reduced`: the CI forward's cotangent into it
    must sum with the reconstruction's before the entry gather's transpose reduces once."""
    sites = _sites()
    model = _model(sites)
    arch = _conditioned_arch(model)
    mesh = hsdp_mesh(2, 2, 2)
    rules = from_config("owner", mesh, sites)
    with jax.set_mesh(mesh):
        target = place_target(model, rules)
        ci_fn = placed_ci_fn(arch, sites, jax.random.PRNGKey(2), mesh, rules)
        components = init_component_stacks_placed(sites, jax.random.PRNGKey(1), rules)
        prepared = jax.eval_shape(target.prepare_compute_weights, components)
        for stacks in prepared.per_kind.values():
            assert stacks["V"].sharding.spec.reduced, stacks["V"].sharding
        gradients = eqx.filter_jit(_engine_and_reference_v_gradients)(
            target,
            ci_fn,
            components,
            jax.tree.map(lambda x: shard_batch(x, mesh, batch_axis=0), _batch(4)),
        )

    _assert_v_gradients_match_the_reference(sites, *jax.device_get(gradients))
