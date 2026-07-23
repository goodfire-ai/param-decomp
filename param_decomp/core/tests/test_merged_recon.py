"""MergedStochasticSubsetPPGDReconLoss: the one-forward stoch+PPGD term through the jitted step."""

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
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    SourceShape,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.recon import MixedPersistentStochasticSources, build_loss_terms
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import Decomposition, TrainingItem, TrainState, make_train_step
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
) -> MergedStochasticSubsetPPGDReconLossConfig:
    return MergedStochasticSubsetPPGDReconLossConfig(
        coeff=1.0,
        adv_fraction=adv_fraction or ScheduleConfig(start_val=0.5),
        routing=UniformKSubsetRoutingConfig(),
        source_shape=source_shape,
        optimizer=AdamPGDConfig(
            beta1=0.5, beta2=0.99, lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025)
        ),
        n_warmup_steps=n_warmup,
    )


def test_adv_fraction_ramp_accepted_and_bounded():
    ramp_to_one = ScheduleConfig(start_val=0.1, fn_type="linear", final_val_frac=10.0)
    cfg = _merged_cfg(n_warmup=0, adv_fraction=ramp_to_one)
    assert cfg.adv_fraction.final_val_frac == 10.0

    escapes_probability_range = ScheduleConfig(start_val=0.5, fn_type="linear", final_val_frac=4.0)
    with pytest.raises(ValidationError, match="adv_fraction"):
        _merged_cfg(n_warmup=0, adv_fraction=escapes_probability_range)


def test_merged_term_builds_one_entry():
    cfg = _merged_cfg(n_warmup=1)
    losses = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
            ),
            cfg,
        ),
        ("a", "b"),
    )
    (term,) = losses.recon
    (entry,) = term.plan
    assert isinstance(entry.sources, MixedPersistentStochasticSources)
    assert entry.live_sites == ("a", "b")
    assert entry.has_delta


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

    merged = _merged_cfg(n_warmup, source_shape=source_shape)
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
                    coeff=merged.coeff,
                    adam=merged.optimizer,
                    n_warmup=merged.n_warmup_steps,
                )
            },
            step=jnp.zeros((), jnp.int32),
        ),
    )
    losses = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
            ),
            merged,
        ),
        model.site_names,
    )
    step = make_train_step(
        model_static=model,
        losses=losses,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=True,
        remat_ci_fn=False,
        mesh=None,
    )

    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    n_steps = 3
    for i in range(n_steps):
        state, metrics = step(model, state, tokens, jax.random.PRNGKey(100 + i))
        assert all(bool(jnp.isfinite(v).all()) for v in metrics.values())
        assert "loss/MergedStochasticSubsetPPGDReconLoss" in metrics
    assert int(state.training.step) == n_steps

    adv = state.training.adversaries[merged.type]
    assert float(adv.opt_state.step_count) == n_steps * (n_warmup + 1)
    for v in adv.sources.values():
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    assert isinstance(state.decomposition.components, ComponentStacks)
    for _, (V, U) in state.decomposition.components.sites_items():
        assert V.dtype == jnp.float32 and U.dtype == jnp.float32
