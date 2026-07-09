"""MergedStochasticPGDReconLoss: the one-forward stoch+PPGD term through the jitted step."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from param_decomp.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.components import DecompVU, init_decomp_vu
from param_decomp.configs import (
    AdamPGDConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticPGDReconLossConfig,
    SCScope,
    UniformKSubsetRoutingConfig,
)
from param_decomp.recon import MixedPersistentStochasticSources, build_loss_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.targets.llama_simple_mlp import site_specs
from param_decomp.tests.test_llama_simple_mlp import (
    _MIXED_SITE_CS,
    _build_chunkwise_ci_fn,
    _tiny_cfg,
    _tiny_decomposed_model,
)
from param_decomp.train import TrainState, make_train_step


def _merged_cfg(n_warmup: int) -> MergedStochasticPGDReconLossConfig:
    return MergedStochasticPGDReconLossConfig(
        coeff=1.0,
        adv_fraction=0.5,
        routing=UniformKSubsetRoutingConfig(),
        scope=SCScope(),
        optimizer=AdamPGDConfig(
            beta1=0.5, beta2=0.99, lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025)
        ),
        n_warmup_steps=n_warmup,
    )


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


def test_merged_train_step_end_to_end():
    """Full jitted step with ONE merged recon term: finite losses, the persistent
    adversary updates (n_warmup + 1 per step through warmup + the S14' final ascent),
    sources stay projected."""
    cfg = _tiny_cfg()
    seq = 16
    n_warmup = 1
    sites = site_specs(cfg, _MIXED_SITE_CS)
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_chunkwise_ci_fn(lm, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    merged = _merged_cfg(n_warmup)
    src = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), (1, seq), jnp.float32, jax.random.PRNGKey(3)
    )
    assert merged.coeff is not None
    state = TrainState(
        components=vu, ci_fn=ci_fn,
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
    )  # fmt: skip
    losses = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
            ),
            merged,
        ),
        lm.site_names,
    )
    step = make_train_step(
        lm=lm,
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
        state, metrics = step(lm, state, tokens, jax.random.PRNGKey(100 + i))
        assert all(bool(jnp.isfinite(v).all()) for v in metrics.values())
        assert "loss/MergedStochasticPGDReconLoss" in metrics
    assert int(state.step) == n_steps

    adv = state.adversaries[merged.type]
    assert float(adv.opt_state.step_count) == n_steps * (n_warmup + 1)
    for v in adv.sources.values():
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    assert isinstance(state.components, DecompVU)
    for V, U in state.components.vu.values():
        assert V.dtype == jnp.float32 and U.dtype == jnp.float32
