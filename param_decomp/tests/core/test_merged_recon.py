"""MergedStochasticSubsetPPGDReconLoss: the one-forward stoch+PPGD term through the jitted step."""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from param_decomp.core.adversary import (
    PersistentAdversary,
    SourcesAdamState,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.core.ci_fn.implementations.layerwise_mlp import (
    LayerwiseMLPCIFnArch,
    init_layerwise_mlp_ci_fn,
)
from param_decomp.core.components import (
    ComponentStacks,
    DenseFactorization,
    SiteC,
    SiteSpec,
    init_component_stacks,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    AuxiliaryReconstructionConfig,
    BatchSourceShape,
    CaptureReconstruction,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    PDConfig,
    SourcePoolConfig,
    TargetedPDConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.model import PlacedModel
from param_decomp.core.objective import build_objective
from param_decomp.core.recon import (
    MixedPersistentStochasticSources,
    PersistentSourcePool,
)
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    PDTrainingState,
    ReconGrid,
    TrainState,
    make_train_step,
)
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.targets import resid_mlp
from param_decomp.targets.llama_simple_mlp import site_specs
from param_decomp.targets.testing import (
    SIMPLE_MLP_MIXED_SITE_CS,
    tiny_simple_mlp_cfg,
    tiny_simple_mlp_chunkwise_ci_fn,
    tiny_simple_mlp_decomposed_model,
)


def _merged_cfg(
    n_warmup: int,
    adv_fraction: ScheduleConfig | None = None,
    source_shape: BatchSourceShape = "bsc",
    auxiliaries: tuple[AuxiliaryReconstructionConfig, ...] = (),
    adversary_objective: Literal["term", "e2e"] | None = None,
) -> MergedStochasticSubsetPPGDReconLossConfig:
    cfg = MergedStochasticSubsetPPGDReconLossConfig(
        coeff=1.0,
        adv_fraction=adv_fraction or ScheduleConfig.constant(0.5),
        routing=UniformKSubsetRoutingConfig(),
        source_shape=source_shape,
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(
                max_val=0.01,
                points=(Knot(at=0.0, frac=0.0), Knot(at=0.025, frac=1.0), Knot(at=1.0, frac=1.0)),
            ),
        ),
        n_warmup_steps=n_warmup,
        auxiliaries=auxiliaries,
    )
    if adversary_objective is not None:
        cfg = cfg.model_copy(update={"adversary_objective": adversary_objective})
    return cfg


def _pooled_cfg(
    n_warmup: int,
    pool: SourcePoolConfig,
    auxiliaries: tuple[AuxiliaryReconstructionConfig, ...] = (),
) -> MergedStochasticSubsetPooledPPGDReconLossConfig:
    return MergedStochasticSubsetPooledPPGDReconLossConfig(
        coeff=1.0,
        adv_fraction=ScheduleConfig.constant(0.5),
        routing=UniformKSubsetRoutingConfig(),
        pool=pool,
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(
                max_val=0.02,
                points=(
                    Knot(at=0.0, frac=0.0),
                    Knot(at=0.025, frac=1.0),
                    Knot(at=1.0, frac=1.0),
                ),
            ),
        ),
        n_warmup_steps=n_warmup,
        auxiliaries=auxiliaries,
    )


def test_pooled_config_builds_selected_strategy_without_a_dense_source_shape():
    cfg = _pooled_cfg(n_warmup=1, pool=SourcePoolConfig(size_per_batch_element=8))
    assert cfg.optimizer.lr_schedule.max_val == 0.02
    losses = build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(coeff=5e-6, gamma=ScheduleConfig.constant(1.0)),
            cfg,
        ),
        tuple(
            SiteSpec(name=name, factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g")
            for name in ("a", "b")
        ),
    )
    (term,) = losses.recon
    assert isinstance(term.sources, PersistentSourcePool)

    with pytest.raises(ValidationError, match="source_shape"):
        MergedStochasticSubsetPooledPPGDReconLossConfig.model_validate(
            cfg.model_dump() | {"source_shape": "bsc"}
        )


def test_pool_configuration_has_one_required_size():
    pool = SourcePoolConfig(size_per_batch_element=3)
    assert pool.model_dump() == {"size_per_batch_element": 3}
    assert SourcePoolConfig.model_validate(pool.model_dump()) == pool
    with pytest.raises(ValidationError, match="size_per_batch_element"):
        SourcePoolConfig.model_validate({})


@pytest.mark.parametrize("size", [0, -1])
def test_pool_rejects_nonpositive_sizes(size: int):
    with pytest.raises(ValidationError):
        SourcePoolConfig(size_per_batch_element=size)


@pytest.mark.parametrize(
    "obsolete", [{"size": 32}, {"group_size": 32}, {"kind": "global"}, {"kind": "grouped"}]
)
def test_pool_rejects_obsolete_configuration(obsolete: dict[str, int | str]):
    with pytest.raises(ValidationError, match="Extra inputs"):
        SourcePoolConfig.model_validate({"size_per_batch_element": 4} | obsolete)


@pytest.mark.parametrize("pd_type", [PDConfig, TargetedPDConfig])
def test_pool_size_is_independent_of_batch_size(pd_type: type[PDConfig | TargetedPDConfig]):
    pooled = _pooled_cfg(n_warmup=1, pool=SourcePoolConfig(size_per_batch_element=3))
    optimizer = AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3))
    losses = [
        ImportanceMinimalityLossConfig(coeff=5e-6, gamma=ScheduleConfig.constant(1.0)).model_dump(),
        pooled.model_dump(),
    ]
    if pd_type is PDConfig:
        losses.append(FaithfulnessLossConfig(coeff=1e5).model_dump())
    authored = dict(
        steps=100,
        components_optimizer=optimizer.model_dump(),
        ci_fn_optimizer=optimizer.model_dump(),
        loss_metrics=losses,
    )
    for batch in (1, 6, 8):
        assert pd_type.model_validate(authored | {"batch_size": batch}).batch_size == batch


def test_adv_fraction_ramp_accepted_and_bounded():
    ramp_to_one = ScheduleConfig(
        max_val=1.0, points=(Knot(at=0.0, frac=0.1), Knot(at=1.0, frac=1.0))
    )
    cfg = _merged_cfg(n_warmup=0, adv_fraction=ramp_to_one)
    assert cfg.adv_fraction.max_val == 1.0

    escapes_probability_range = ScheduleConfig(
        max_val=2.0, points=(Knot(at=0.0, frac=0.25), Knot(at=1.0, frac=1.0))
    )
    with pytest.raises(ValidationError, match="adv_fraction"):
        _merged_cfg(n_warmup=0, adv_fraction=escapes_probability_range)


def test_merged_config_builds_one_mixed_sources_term():
    cfg = _merged_cfg(n_warmup=1)
    assert cfg.adversary_objective == "e2e"
    losses = build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                ),
            ),
            cfg,
        ),
        tuple(
            SiteSpec(name=name, factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g")
            for name in ("a", "b")
        ),
    )
    (term,) = losses.recon
    assert isinstance(term.sources, MixedPersistentStochasticSources)
    assert not (ReconGrid(losses.recon, key_offset=1).e2e_terms_requiring_source_grad_retake_by_key)


@pytest.mark.slow
@pytest.mark.parametrize(
    "source_shape,src_leading,pool",
    [
        ("bc", (2, 1), None),
        ("bsc", (2, 16), None),
        ("bsc", (2, 8), SourcePoolConfig(size_per_batch_element=8)),
    ],
)
def test_merged_train_step_end_to_end(
    source_shape: BatchSourceShape,
    src_leading: tuple[int, ...],
    pool: SourcePoolConfig | None,
):
    """Full jitted steps cover every dense source shape and the selected vector pool."""
    cfg = tiny_simple_mlp_cfg()
    seq = 16
    n_warmup = 1
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = tiny_simple_mlp_chunkwise_ci_fn(model, jax.random.PRNGKey(2))
    opt_vu = _adamw_optimizer(
        AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3), grad_clip_norm=0.01), 1
    )
    opt_ci = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1)

    hidden_acts = AuxiliaryReconstructionConfig(
        name="hidden_acts_reconstruction",
        coeff=0.2,
        comparisons=tuple(
            CaptureReconstruction(capture=point, distance="relative_squared_error")
            for point in ("resid.3", "resid.4", "resid.5", "resid.6")
        ),
    )
    merged = (
        _merged_cfg(n_warmup, source_shape=source_shape, auxiliaries=(hidden_acts,))
        if pool is None
        else _pooled_cfg(n_warmup, pool, (hidden_acts,))
    )
    src = init_persistent_sources(
        model.sites,
        src_leading,
        jnp.float32,
        jax.random.PRNGKey(3),
    )
    losses = build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                ),
            ),
            merged,
        ),
        model.sites,
    )
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=PDTrainingState(
            frequency=BatchFrequency(),
            objective=losses,
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={
                merged.type: PersistentAdversary(
                    sources=src,
                    opt_state=init_sources_adam_state(src),
                    state_key=merged.type,
                    optimizer=merged.optimizer,
                    n_warmup=merged.n_warmup_steps,
                )
            },
            step=jnp.zeros((), jnp.int32),
        ),
    )
    placed = PlacedModel(model=model, placement=None)
    step = jax.jit(
        make_train_step(
            model_static=placed,
            substrate=ForwardSubstrate.of(
                placed,
                remat_recon_forwards=True,
                remat_ci_fn=False,
                ci_capture_keys=ci_fn.capture_keys,
            ),
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=100,
            faithfulness=faithfulness_loss_for(placed),
        )
    )

    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    n_steps = 3
    for i in range(n_steps):
        state, metrics = step(
            placed,
            state,
            LMBatchWithDocuments.from_unsegmented_sequences(tokens),
            jax.random.PRNGKey(100 + i),
        )
        assert all(bool(jnp.isfinite(v).all()) for v in metrics.values())
        assert f"loss/{merged.type}" in metrics
        assert f"loss/{merged.type}/hidden_acts_reconstruction" in metrics
    assert int(state.training.step) == n_steps

    adv = state.training.adversaries[merged.type]
    assert isinstance(adv.opt_state, SourcesAdamState)
    assert float(adv.opt_state.step_count) == n_steps * (n_warmup + 1)
    for v in jax.tree.leaves(adv.sources):
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    assert isinstance(state.decomposition.components, ComponentStacks)
    for _, site_components in state.decomposition.components.sites_items():
        assert site_components.V.dtype == jnp.float32
        assert site_components.U.dtype == jnp.float32


def _one_step_adversary_objective_probe(
    auxiliaries: tuple[AuxiliaryReconstructionConfig, ...],
    adversary_objective: Literal["term", "e2e"],
) -> tuple[PersistentAdversary, ComponentStacks]:
    cfg = resid_mlp.ResidMLPConfig(
        n_features=5,
        d_embed=5,
        d_mlp=8,
        n_layers=2,
        act_fn_name="relu",
        in_bias=False,
        out_bias=False,
    )
    sites = resid_mlp.site_specs(
        cfg,
        tuple(
            SiteC(f"layers.{layer}.{kind}", count)
            for layer in range(cfg.n_layers)
            for kind, count in (("mlp_in", 6), ("mlp_out", 7))
        ),
    )
    model = resid_mlp.resid_mlp_decomposed_model(
        cfg, resid_mlp.init_resid_mlp_target(cfg, jax.random.PRNGKey(0)), sites
    )
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = init_layerwise_mlp_ci_fn(
        LayerwiseMLPCIFnArch(
            hidden_dims=(8,), has_position_axis=False, input_names=model.site_names
        ),
        sites,
        jax.random.PRNGKey(2),
    )
    components_optimizer = _adamw_optimizer(
        AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1
    )
    ci_fn_optimizer = _adamw_optimizer(
        AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3)), 1
    )
    merged = _merged_cfg(
        n_warmup=1,
        source_shape="bc",
        auxiliaries=auxiliaries,
        adversary_objective=adversary_objective,
    ).model_copy(
        update={
            "optimizer": AdamPGDConfig(
                beta1=0.5,
                beta2=0.99,
                lr_schedule=ScheduleConfig.constant(0.05),
            )
        }
    )
    sources = init_persistent_sources(
        model.sites,
        (2,),
        jnp.float32,
        jax.random.PRNGKey(3),
    )
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-6,
                gamma=ScheduleConfig.constant(1.0),
            ),
            merged,
        ),
        model.sites,
    )
    state = TrainState(
        decomposition=Decomposition(components=components, ci_fn=ci_fn),
        training=PDTrainingState(
            frequency=BatchFrequency(),
            objective=objective,
            components_opt_state=components_optimizer.init(eqx.filter(components, eqx.is_array)),
            ci_fn_opt_state=ci_fn_optimizer.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={
                merged.type: PersistentAdversary(
                    sources=sources,
                    opt_state=init_sources_adam_state(sources),
                    state_key=merged.type,
                    optimizer=merged.optimizer,
                    n_warmup=merged.n_warmup_steps,
                )
            },
            step=jnp.zeros((), jnp.int32),
        ),
    )
    placed = PlacedModel(model=model, placement=None)
    step = jax.jit(
        make_train_step(
            model_static=placed,
            substrate=ForwardSubstrate.of(
                placed,
                remat_recon_forwards=True,
                remat_ci_fn=False,
                ci_capture_keys=ci_fn.capture_keys,
            ),
            components_optimizer=components_optimizer,
            ci_fn_optimizer=ci_fn_optimizer,
            total_steps=100,
            faithfulness=faithfulness_loss_for(placed),
        )
    )
    inputs = jax.random.normal(jax.random.PRNGKey(4), (2, cfg.d_embed))
    state, _ = step(
        placed,
        state,
        inputs,
        jax.random.PRNGKey(100),
    )
    updated_components = state.decomposition.components
    assert isinstance(updated_components, ComponentStacks)
    return state.training.adversaries[merged.type], updated_components


def test_e2e_adversary_excludes_hidden_acts_reconstruction_from_sources_only():
    """The objective split belongs to the trainer, not the target architecture.

    A two-layer residual MLP keeps genuine upstream/downstream captures and the real
    jitted warmup/update path. LM source shapes and hidden captures run end to end above.
    """
    hidden_acts_reconstruction = AuxiliaryReconstructionConfig(
        name="hidden_acts_reconstruction",
        coeff=0.2,
        comparisons=tuple(
            CaptureReconstruction(capture=point, distance="relative_squared_error")
            for point in ("layers.0.mlp_out.out", "layers.1.mlp_out.out")
        ),
    )
    e2e_adversary, e2e_components = _one_step_adversary_objective_probe(
        (hidden_acts_reconstruction,), "e2e"
    )
    term_adversary, _ = _one_step_adversary_objective_probe((hidden_acts_reconstruction,), "term")
    output_only_adversary, output_only_components = _one_step_adversary_objective_probe((), "term")

    assert all(
        bool(jnp.allclose(e2e, output_only, atol=1e-6))
        for e2e, output_only in zip(
            jax.tree.leaves(e2e_adversary.opt_state),
            jax.tree.leaves(output_only_adversary.opt_state),
            strict=True,
        )
    )
    assert any(
        not bool(jnp.allclose(e2e, term, atol=1e-6))
        for e2e, term in zip(
            jax.tree.leaves(e2e_adversary.opt_state),
            jax.tree.leaves(term_adversary.opt_state),
            strict=True,
        )
    )
    assert any(
        not bool(jnp.allclose(e2e, output_only, atol=1e-6))
        for e2e, output_only in zip(
            jax.tree.leaves(e2e_components),
            jax.tree.leaves(output_only_components),
            strict=True,
        )
    )
