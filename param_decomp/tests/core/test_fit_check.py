"""The deviceless fit check declares each algorithm's runtime state: every piece of state the
objective demands must be typed into the standin, or the receipt dies at trace instead
of reaching a verdict."""

import dataclasses
from collections.abc import Callable
from typing import Literal

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import TypeCheckError

from param_decomp.core.adversary import BlockedSourceComponents
from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.components import SiteC
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    CIMaskedReconLossConfig,
    FaithfulnessLossConfig,
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    NontargetConfig,
    PDConfig,
    SourcePoolConfig,
    TargetedLossMetricConfig,
    TargetedPDConfig,
)
from param_decomp.core.init_placed import seeded_ci_fn_initializer
from param_decomp.core.losses import EmaFrequency
from param_decomp.core.model import Positioned
from param_decomp.core.placement import from_config
from param_decomp.core.run_state import init_pd_state, init_targeted_pd_state
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import place_target
from param_decomp.core.tools.fit_check import (
    _on_mesh,
    abstract_placed_model,
    argument_audit,
    declared_decomposition,
    fit_report_of_compiled,
    lowered_targeted_train_step,
    lowered_train_step,
    standin_faithfulness_loss,
)
from param_decomp.experiments.lm.abstract_inputs import (
    abstract_lm_batch,
    abstract_lm_batch_with_documents,
)
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.llama_simple_mlp import (
    KIND_ORDER,
    canonical_site_cs,
    site_name,
    site_specs,
)
from param_decomp.targets.testing import tiny_simple_mlp_cfg, tiny_simple_mlp_decomposed_model

_B, _T, _C = 4, 8, 8


@pytest.mark.parametrize("algorithm", ["plain", "targeted"])
@pytest.mark.parametrize("tp", [1, pytest.param(2, marks=pytest.mark.multidevice)])
def test_ema_frequency_and_pooled_adversary_reach_a_receipt_verdict(
    tp: int, algorithm: Literal["plain", "targeted"]
):
    """The stand-in state must carry every configured persistent runtime leaf."""
    if jax.device_count() < tp:
        pytest.skip(f"requires {tp} local devices")
    particles = 3
    # Two layers retain non-singleton site stacks without a deeper training graph.
    cfg = dataclasses.replace(tiny_simple_mlp_cfg(), n_layer=2)
    sites = site_specs(
        cfg,
        canonical_site_cs(
            tuple(
                SiteC(site_name(layer, kind), _C)
                for layer in range(cfg.n_layer)
                for kind in KIND_ORDER
            )
        ),
    )
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    mesh = Mesh(
        np.array(jax.devices()[:tp]).reshape(1, 1, tp),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    rules = from_config("zero1", mesh, model.sites)
    placed = place_target(model, rules)
    abstract_model = abstract_placed_model(model, rules)
    redeclared_model = abstract_placed_model(abstract_model.model, rules)
    assert jax.tree.structure(abstract_model) == jax.tree.structure(placed)
    for actual, abstract, redeclared in zip(
        jax.tree.leaves(placed),
        jax.tree.leaves(abstract_model),
        jax.tree.leaves(redeclared_model),
        strict=True,
    ):
        assert isinstance(abstract, jax.ShapeDtypeStruct)
        assert abstract == redeclared
        assert abstract.shape == actual.shape
        assert abstract.dtype == actual.dtype
        assert abstract.sharding == actual.sharding
    ci_fn = ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(input_taps=("resid.0",), output_sites=placed.model.site_names),),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=1,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    optimizer = AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3))
    losses: list[TargetedLossMetricConfig] = [
        ImportanceMinimalityLossConfig(
            coeff=5e-6,
            gamma=ScheduleConfig.constant(1.0),
            frequency=FrequencyMinimalityConfig(
                coeff=1e-6, reference_datapoint_count=_B * _T, ema_halflife_steps=8.0
            ),
        ),
        MergedStochasticSubsetPooledPPGDReconLossConfig(
            coeff=1.0,
            pool=SourcePoolConfig(size_per_batch_element=particles),
            adv_fraction=ScheduleConfig.constant(0.5),
            optimizer=AdamPGDConfig(lr_schedule=ScheduleConfig.constant(0.02)),
            n_warmup_steps=1,
        ),
    ]
    batch = abstract_lm_batch_with_documents(_B, _T, mesh)
    with jax.set_mesh(mesh):
        match algorithm:
            case "plain":
                pd = PDConfig(
                    steps=100,
                    batch_size=_B,
                    components_optimizer=optimizer,
                    ci_fn_optimizer=optimizer,
                    loss_metrics=[FaithfulnessLossConfig(coeff=1e5), *losses],
                )
                lowered, declared = lowered_train_step(
                    pd,
                    ci_fn,
                    abstract_model,
                    Positioned(n_positions=_T),
                    batch,
                    standin_faithfulness_loss(placed),
                    remat_recon_forwards=True,
                    remat_ci_fn=False,
                    compiler_options=None,
                )
                initialized = init_pd_state(
                    pd,
                    placed,
                    seeded_ci_fn_initializer(ci_fn, model.sites, rules),
                    Positioned(n_positions=_T),
                    declared.opt_vu,
                    declared.opt_ci,
                    jax.random.PRNGKey(1),
                    jax.random.PRNGKey(2),
                )
                frequency = declared.state.training.frequency
                assert isinstance(frequency, EmaFrequency)
                histories = (frequency.estimate,)
            case "targeted":
                targeted_pd = TargetedPDConfig(
                    steps=100,
                    batch_size=_B,
                    components_optimizer=optimizer,
                    ci_fn_optimizer=optimizer,
                    loss_metrics=losses,
                )
                nontarget = NontargetConfig(
                    batch_size=_B // 2,
                    recon=[CIMaskedReconLossConfig(coeff=1.0)],
                    impmin_coeff=1e-6,
                )
                lowered, declared = lowered_targeted_train_step(
                    targeted_pd,
                    nontarget,
                    ci_fn,
                    placed,
                    Positioned(n_positions=_T),
                    batch,
                    abstract_lm_batch_with_documents(_B // 2, _T, mesh),
                    remat_recon_forwards=True,
                    remat_ci_fn=False,
                    compiler_options=None,
                )
                initialized = init_targeted_pd_state(
                    targeted_pd,
                    placed,
                    seeded_ci_fn_initializer(ci_fn, model.sites, rules),
                    Positioned(n_positions=_T),
                    declared.opt_vu,
                    declared.opt_ci,
                    jax.random.PRNGKey(1),
                    jax.random.PRNGKey(2),
                    nontarget,
                )
                target_frequency = declared.state.training.target_frequency
                nontarget_frequency = declared.state.training.nontarget_frequency
                assert isinstance(target_frequency, EmaFrequency)
                assert isinstance(nontarget_frequency, EmaFrequency)
                histories = (target_frequency.estimate, nontarget_frequency.estimate)
        decomposition = declared_decomposition(ci_fn, abstract_model)
    assert jax.tree.structure(decomposition) == jax.tree.structure(declared.state.decomposition)
    for actual, abstract in zip(
        jax.tree.leaves(initialized), jax.tree.leaves(declared.state), strict=True
    ):
        assert isinstance(abstract, jax.ShapeDtypeStruct)
        assert actual.shape == abstract.shape
        assert actual.dtype == abstract.dtype
        assert isinstance(abstract.sharding, NamedSharding)
        if isinstance(actual.sharding, NamedSharding):
            assert actual.sharding.is_equivalent_to(abstract.sharding, ndim=actual.ndim)
        else:
            assert abstract.sharding.is_fully_replicated
    with pytest.raises(TypeCheckError):
        dataclasses.replace(declared, state=initialized)
    with pytest.raises(TypeCheckError):
        _on_mesh(initialized, mesh)
    for history in histories:
        assert {name: leaf.shape for name, leaf in history.items()} == {
            spec.name: (spec.C,) for spec in placed.model.sites
        }
        for leaf in history.values():
            assert leaf.sharding == NamedSharding(mesh, P("tp"))
    (adversary,) = declared.state.training.adversaries.values()
    for stack in adversary.sources.stacks.values():
        assert stack.delta.shape[1:] == (_B, particles)
        assert not isinstance(stack.components, BlockedSourceComponents)
        assert stack.components.shape[1:] == (_B, particles, _C)
    argument_audit((placed, declared.state, batch), pool_gib=8.0)
    report = fit_report_of_compiled(lowered.compile(), pool_gib=8.0)
    assert "VERDICT" in report.render()


@pytest.mark.parametrize("batch_factory", [abstract_lm_batch, abstract_lm_batch_with_documents])
def test_abstract_inputs_keep_concrete_mesh_through_aot_audit(
    batch_factory: Callable[[int, int, Mesh], LMBatch | LMBatchWithDocuments],
):
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    batch = batch_factory(_B, _T, mesh)
    expected = NamedSharding(mesh, P("data", None))
    for leaf in jax.tree.leaves(batch):
        assert isinstance(leaf, jax.ShapeDtypeStruct)
        assert leaf.sharding == expected
        assert leaf.sharding.mesh is mesh
    argument_audit(batch, pool_gib=8.0)
    with jax.set_mesh(mesh):
        compiled = (
            jax.jit(lambda value: jax.tree.map(lambda x: x + 1, value)).lower(batch).compile()
        )
    for sharding in jax.tree.leaves(compiled.input_shardings):
        assert sharding == expected
