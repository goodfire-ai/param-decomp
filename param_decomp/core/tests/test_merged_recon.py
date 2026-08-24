"""MergedStochasticSubsetPPGDReconLoss: the one-forward stoch+PPGD term through the jitted step."""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from pydantic import ValidationError

from param_decomp.core.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.core.components import ComponentStacks, init_component_stacks
from param_decomp.core.configs import (
    AdamPGDConfig,
    FaithfulnessLossConfig,
    HiddenActsReconstruction,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    SourceShape,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.model import PlacedModel
from param_decomp.core.objective import build_objective
from param_decomp.core.recon import MixedPersistentStochasticSources
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    ReconGrid,
    TrainingItem,
    TrainState,
    make_train_step,
)
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
    source_shape: SourceShape = "sc",
    hidden_acts_reconstruction: HiddenActsReconstruction | None = None,
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
        hidden_acts_reconstruction=hidden_acts_reconstruction,
    )
    if adversary_objective is not None:
        cfg = cfg.model_copy(update={"adversary_objective": adversary_objective})
    return cfg


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
                pnorm=ScheduleConfig(
                    max_val=2.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.2))
                ),
            ),
            cfg,
        ),
        ("a", "b"),
    )
    (term,) = losses.recon
    assert isinstance(term.sources, MixedPersistentStochasticSources)
    assert term.uses_weight_deltas
    assert not (
        ReconGrid.of(losses.recon, key_offset=1).e2e_terms_requiring_source_grad_retake_by_key
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "source_shape,src_leading",
    [("c", (1, 1)), ("bc", (2, 1)), ("sc", (1, 16)), ("bsc", (2, 16))],
)
def test_merged_train_step_end_to_end(source_shape: SourceShape, src_leading: tuple[int, int]):
    """Full jitted step with ONE merged recon term, at every `source_shape`: finite
    losses, the persistent adversary updates (n_warmup + 1 per step through warmup + the
    S14' final ascent), sources stay projected."""
    cfg = tiny_simple_mlp_cfg()
    seq = 16
    n_warmup = 1
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = tiny_simple_mlp_chunkwise_ci_fn(model, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    merged = _merged_cfg(
        n_warmup,
        source_shape=source_shape,
        hidden_acts_reconstruction=HiddenActsReconstruction(
            coeff=0.2,
            points=("resid.3", "resid.4", "resid.5", "resid.6"),
        ),
    )
    src = init_persistent_sources(
        model.site_names,
        tuple(s.C for s in model.sites),
        src_leading,
        jnp.float32,
        jax.random.PRNGKey(3),
    )
    assert merged.coeff is not None
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={
                merged.type: PersistentAdversary(
                    sources=src,
                    opt_state=init_sources_adam_state(src),
                    state_key=merged.type,
                    adam=merged.optimizer,
                    n_warmup=merged.n_warmup_steps,
                )
            },
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    losses = build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(
                    max_val=2.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.2))
                ),
            ),
            merged,
        ),
        model.site_names,
    )
    placed = PlacedModel(model=model, placement=None)
    step = make_train_step(
        model_static=placed,
        substrate=ForwardSubstrate.of(
            placed,
            remat_recon_forwards=True,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=losses,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        faithfulness=faithfulness_loss_for(model),
    )

    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    n_steps = 3
    for i in range(n_steps):
        state, metrics = step(placed, state, tokens, jax.random.PRNGKey(100 + i))
        assert all(bool(jnp.isfinite(v).all()) for v in metrics.values())
        assert "loss/MergedStochasticSubsetPPGDReconLoss" in metrics
        assert "loss/MergedStochasticSubsetPPGDReconLoss/hidden_acts_reconstruction" in metrics
    assert int(state.training.step) == n_steps

    adv = state.training.adversaries[merged.type]
    assert float(adv.opt_state.step_count) == n_steps * (n_warmup + 1)
    for v in jax.tree.leaves(adv.sources):
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    assert isinstance(state.decomposition.components, ComponentStacks)
    for _, site_components in state.decomposition.components.sites_items():
        assert site_components.V.dtype == jnp.float32
        assert site_components.U.dtype == jnp.float32


def _one_step_adversary_objective_probe(
    hidden_acts_reconstruction: HiddenActsReconstruction | None,
    adversary_objective: Literal["term", "e2e"],
) -> tuple[PersistentAdversary, ComponentStacks]:
    cfg = tiny_simple_mlp_cfg()
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = tiny_simple_mlp_chunkwise_ci_fn(model, jax.random.PRNGKey(2))
    components_optimizer = optax.adamw(1e-3, weight_decay=0.0)
    ci_fn_optimizer = optax.adamw(1e-3, weight_decay=0.0)
    merged = _merged_cfg(
        n_warmup=1,
        hidden_acts_reconstruction=hidden_acts_reconstruction,
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
        model.site_names,
        tuple(site.C for site in model.sites),
        (1, 16),
        jnp.float32,
        jax.random.PRNGKey(3),
    )
    state = TrainState(
        decomposition=Decomposition(components=components, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=components_optimizer.init(eqx.filter(components, eqx.is_array)),
            ci_fn_opt_state=ci_fn_optimizer.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={
                merged.type: PersistentAdversary(
                    sources=sources,
                    opt_state=init_sources_adam_state(sources),
                    state_key=merged.type,
                    adam=merged.optimizer,
                    n_warmup=merged.n_warmup_steps,
                )
            },
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-6,
                pnorm=ScheduleConfig.constant(2.0),
            ),
            merged,
        ),
        model.site_names,
    )
    placed = PlacedModel(model=model, placement=None)
    step = make_train_step(
        model_static=placed,
        substrate=ForwardSubstrate.of(
            placed,
            remat_recon_forwards=True,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=objective,
        components_optimizer=components_optimizer,
        ci_fn_optimizer=ci_fn_optimizer,
        total_steps=100,
        faithfulness=faithfulness_loss_for(model),
    )
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    state, _ = step(placed, state, tokens, jax.random.PRNGKey(100))
    updated_components = state.decomposition.components
    assert isinstance(updated_components, ComponentStacks)
    return state.training.adversaries[merged.type], updated_components


def test_e2e_adversary_excludes_hidden_acts_reconstruction_from_sources_only():
    hidden_acts_reconstruction = HiddenActsReconstruction(coeff=0.2, points=("resid.3", "resid.4"))
    e2e_adversary, e2e_components = _one_step_adversary_objective_probe(
        hidden_acts_reconstruction, "e2e"
    )
    term_adversary, _ = _one_step_adversary_objective_probe(hidden_acts_reconstruction, "term")
    output_only_adversary, output_only_components = _one_step_adversary_objective_probe(
        None, "term"
    )

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
