"""Placed engine-hosted SimpleMLP forwards: trace coverage and parity vs unplaced.

The migration gate for hosting SimpleMLP on the shared scan engine: every preset
traces (forward AND grad), and the placed zero1 forward/gradient/step match the
unplaced run by value — the same obligations the deleted hand-rolled model's placed
suite pinned.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import random
from jax.sharding import AxisType, Mesh

from param_decomp.core.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    MHACIAttention,
)
from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    SiteSpec,
    init_component_stacks,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    StochasticReconSubsetLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.init_placed import init_ci_fn_placed, init_component_stacks_placed
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.objective import build_objective
from param_decomp.core.placement import PlacementRules, from_config
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.sharding import place_target, shard_batch
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_train_step,
)
from param_decomp.targets.glu_transformer import GLUDecomposedModel
from param_decomp.targets.llama_simple_mlp import (
    KIND_ORDER,
    canonical_site_cs,
    site_name,
    site_specs,
)
from param_decomp.targets.testing import (
    tiny_simple_mlp_cfg,
    tiny_simple_mlp_decomposed_model,
)

pytestmark = [
    pytest.mark.multidevice,
    pytest.mark.skipif(
        jax.default_backend() != "cpu" or jax.device_count() < 4,
        reason="requires a four-device CPU topology from make test-multidevice",
    ),
]

_B, _T, _C = 8, 16, 8


def _mesh_2_2_1() -> Mesh:
    """A (2,2,1) mesh over the first four devices — `hsdp_mesh` claims the whole
    allocation, and the multidevice suite provides eight."""
    return Mesh(
        np.array(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )


def _model_and_sites() -> tuple[GLUDecomposedModel, tuple[SiteSpec, ...]]:
    cfg = tiny_simple_mlp_cfg()
    site_cs = canonical_site_cs(
        tuple(
            SiteC(site_name(layer, kind), _C) for layer in range(cfg.n_layer) for kind in KIND_ORDER
        )
    )
    sites = site_specs(cfg, site_cs)
    return tiny_simple_mlp_decomposed_model(cfg, sites, random.PRNGKey(0)), sites


def _tokens() -> jax.Array:
    return random.randint(random.PRNGKey(4), (_B, _T), 0, tiny_simple_mlp_cfg().vocab_size)


def _ones_masking(model: GLUDecomposedModel) -> MaterializedMasking:
    return MaterializedMasking(
        component_masks={s.name: jnp.ones((_B, _T, s.C)) for s in model.sites}
    )


def _masked_loss(
    model: GLUDecomposedModel,
    vu: ComponentStacks,
    tokens: jax.Array,
    masking: MaterializedMasking,
    rules: PlacementRules | None,
) -> jax.Array:
    prepared = model.prepare_compute_weights(vu, rules)
    output = model.masked_forward(
        prepared, tokens, masking=masking, placement=rules, capture_keys=frozenset(), remat=False
    ).output
    return jnp.mean(jnp.square(output))


@pytest.mark.parametrize("preset", ("zero1", "owner", "ddp"))
def test_placed_forwards_and_grads_trace(preset: str):
    model, sites = _model_and_sites()
    mesh = _mesh_2_2_1()
    rules = from_config(preset, mesh, model.sites)
    placed_target = place_target(model, rules).model
    assert isinstance(placed_target, GLUDecomposedModel)
    tokens = shard_batch(_tokens(), mesh, batch_axis=0)
    masking = _ones_masking(model)
    with jax.set_mesh(mesh):
        vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
        jax.make_jaxpr(jax.grad(lambda c: _masked_loss(placed_target, c, tokens, masking, rules)))(
            vu
        )
        jax.make_jaxpr(
            lambda m, c, x: m.component_activation_forward(
                m.prepare_compute_weights(c, rules), x, capture_keys=frozenset(), placement=rules
            )[1]
        )(placed_target, vu, tokens)


def test_placed_zero1_forward_and_grad_match_unplaced():
    model, sites = _model_and_sites()
    tokens = _tokens()
    masking = _ones_masking(model)
    vu = init_component_stacks(sites, random.PRNGKey(1))

    reference_loss, reference_grads = jax.jit(
        jax.value_and_grad(lambda c: _masked_loss(model, c, tokens, masking, None))
    )(vu)
    _, reference_acts = jax.jit(
        lambda m, c, x: m.component_activation_forward(
            m.prepare_compute_weights(c, None), x, capture_keys=frozenset(), placement=None
        )
    )(model, vu, tokens)

    mesh = _mesh_2_2_1()
    rules = from_config("zero1", mesh, model.sites)
    placed_target = place_target(model, rules).model
    assert isinstance(placed_target, GLUDecomposedModel)
    placed_tokens = shard_batch(tokens, mesh, batch_axis=0)
    with jax.set_mesh(mesh):
        # bit-identical values to the unplaced init (same key), placed on the rules
        placed_vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
        placed_loss, placed_grads = jax.jit(
            jax.value_and_grad(
                lambda c: _masked_loss(placed_target, c, placed_tokens, masking, rules)
            )
        )(placed_vu)
        _, placed_acts = jax.jit(
            lambda m, c, x: m.component_activation_forward(
                m.prepare_compute_weights(c, rules), x, capture_keys=frozenset(), placement=rules
            )
        )(placed_target, placed_vu, placed_tokens)

    assert jnp.allclose(reference_loss, placed_loss, rtol=1e-5), (reference_loss, placed_loss)
    for group in vu.stacks:
        for reference, placed, which in zip(
            reference_grads.stacks[group], placed_grads.stacks[group], ("V", "U"), strict=True
        ):
            gathered = jax.device_get(placed)
            assert jnp.allclose(reference, gathered, atol=1e-6, rtol=1e-4), (group, which)
    for site in reference_acts:
        assert jnp.allclose(
            reference_acts[site], jax.device_get(placed_acts[site]), atol=1e-5, rtol=1e-4
        ), site


@pytest.mark.parametrize("preset", ("zero1", "owner"))
def test_placed_step_runs_the_full_objective(preset: str):
    """The real `make_train_step` (persistent PPGD + stochastic subset + faithfulness —
    the committed pile seat's term classes) executes placed at (2,2,1)."""
    cfg = tiny_simple_mlp_cfg()
    model, sites = _model_and_sites()
    mesh = _mesh_2_2_1()
    rules = from_config(preset, mesh, model.sites)
    model = place_target(model, rules)
    tokens = shard_batch(_tokens(), mesh, batch_axis=0)
    with jax.set_mesh(mesh):
        vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
        half = len(model.site_names) // 2
        arch = ChunkwiseTransformerCIArch(
            chunks=(
                Chunk(input_taps=("resid.0",), output_sites=model.site_names[:half]),
                Chunk(input_taps=("resid.0",), output_sites=model.site_names[half:]),
            ),
            input_dim=cfg.n_embd,
            d_model=16,
            n_blocks=2,
            attention=MHACIAttention(n_heads=2),
            ffn_hidden=32,
            ffn_kind="gelu",
            learned_norm_scale=False,
        )
        ci_fn = init_ci_fn_placed(arch, model.sites, random.PRNGKey(2), mesh, rules)
        assert isinstance(ci_fn, ChunkwiseTransformerCIFn)
        src = init_persistent_sources(
            model.site_names,
            tuple(s.C for s in model.sites),
            (1, _T),
            jnp.float32,
            random.PRNGKey(3),
        )
        opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
        opt_ci = optax.adamw(1e-3, weight_decay=0.0)
        ppgd_cfg = PersistentPGDReconLossConfig(
            coeff=0.5,
            source_shape="sc",
            optimizer=AdamPGDConfig(
                beta1=0.5,
                beta2=0.99,
                lr_schedule=ScheduleConfig(
                    max_val=0.01,
                    points=(
                        Knot(at=0.0, frac=0.0),
                        Knot(at=0.025, frac=1.0),
                        Knot(at=1.0, frac=1.0),
                    ),
                ),
            ),
            n_warmup_steps=2,
        )
        state = TrainState(
            decomposition=Decomposition(components=vu, ci_fn=ci_fn),
            training=TrainingItem(
                components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
                ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
                adversaries={
                    ppgd_cfg.type: PersistentAdversary(
                        sources=src,
                        opt_state=init_sources_adam_state(src),
                        state_key=ppgd_cfg.type,
                        adam=ppgd_cfg.optimizer,
                        n_warmup=ppgd_cfg.n_warmup_steps,
                    )
                },
                freq_ema=None,
                step=jnp.zeros((), jnp.int32),
            ),
        )
        loss_terms = build_objective(
            (
                FaithfulnessLossConfig(coeff=1e5),
                ImportanceMinimalityLossConfig(
                    coeff=5e-6,
                    gamma=ScheduleConfig(
                        max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                    ),
                ),
                StochasticReconSubsetLossConfig(
                    routing=UniformKSubsetRoutingConfig(), coeff=0.5, n_mask_samples=1
                ),
                ppgd_cfg,
            ),
            model.site_names,
        )
        step = make_train_step(
            model_static=model,
            substrate=ForwardSubstrate.of(
                model,
                remat_recon_forwards=True,
                remat_ci_fn=False,
                ci_capture_keys=ci_fn.capture_keys,
                ci_placement=rules.ci_fn,
            ),
            objective=loss_terms,
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=100,
            faithfulness=faithfulness_loss_for(model.model),
        )
        state, metrics = step(model, state, tokens, random.PRNGKey(100))
    assert jnp.isfinite(metrics["total"]), metrics
    assert cfg.n_layer > 1, "the defect only bites multi-layer stacks; keep this real"
