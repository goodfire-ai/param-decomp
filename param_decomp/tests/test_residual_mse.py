"""Block-residual relative-MSE auxiliary recon loss (SPEC S35)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from param_decomp.core.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.core.components import init_component_stacks
from param_decomp.core.configs import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.losses import residual_mse_loss
from param_decomp.core.recon import build_loss_terms
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import Decomposition, TrainingItem, TrainState, make_train_step
from param_decomp.targets.llama_simple_mlp import site_specs
from param_decomp.targets.testing import (
    SIMPLE_MLP_MIXED_SITE_CS,
    tiny_simple_mlp_cfg,
    tiny_simple_mlp_chunkwise_ci_fn,
    tiny_simple_mlp_decomposed_model,
)


def test_residual_mse_loss_hand_computed():
    # One block, batch=2, seq=1, d=2: clean = [[1,0]]x2, masked = [[1,1],[1,-1]].
    clean = jnp.array([[[[1.0, 0.0]], [[1.0, 0.0]]]])  # (n_block=1, batch=2, seq=1, d=2)
    masked = jnp.array([[[[1.0, 1.0]], [[1.0, -1.0]]]])
    # sq_diff summed over (batch,seq,d) = 1 + 1 = 2; sq_clean summed = 1 + 1 = 2 -> ratio 1.0.
    assert float(residual_mse_loss(masked, clean)) == 1.0


def test_residual_mse_loss_zero_when_identical():
    key = jax.random.PRNGKey(0)
    resids = jax.random.normal(key, (4, 2, 8, 16))
    assert float(residual_mse_loss(resids, resids)) == 0.0


def test_residual_mse_loss_averages_across_blocks():
    # Block-count-normalized (SPEC S35): a lone differing block's contribution shrinks
    # as 1/n_block, so the coefficient's effective strength stays depth-invariant rather
    # than scaling with how many blocks are measured.
    clean = jnp.ones((3, 2, 1, 4))
    masked = clean.at[0].set(clean[0] + 1.0)  # only block 0 differs
    per_block = float(residual_mse_loss(masked[0:1], clean[0:1]))
    total = float(residual_mse_loss(masked, clean))
    assert total == pytest.approx(per_block / 3)  # blocks 1, 2 are identical -> contribute 0


def test_build_loss_terms_threads_residual_mse_coeff():
    cfg = ChunkwiseSubsetReconLossConfig(
        routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=2,
        n_samples=1, residual_mse_coeff=0.1,
    )  # fmt: skip
    losses = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
            ),  # fmt: skip
            cfg,
        ),
        ("a", "b"),
    )
    (term,) = losses.recon
    assert term.residual_mse_coeff == 0.1


def _step_with_chunkwise_recon(residual_mse_coeff: float | None):
    cfg = tiny_simple_mlp_cfg()
    seq = 16
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = tiny_simple_mlp_chunkwise_ci_fn(model, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            step=jnp.zeros((), jnp.int32),
        ),
    )
    losses = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
            ),  # fmt: skip
            ChunkwiseSubsetReconLossConfig(
                routing=UniformKSubsetRoutingConfig(),
                coeff=0.5,
                sites_per_chunk=2,
                n_samples=1,
                residual_mse_coeff=residual_mse_coeff,
            ),  # fmt: skip
        ),
        model.site_names,
    )
    step = make_train_step(
        model_static=model, losses=losses, components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100, remat_recon_forwards=True, remat_ci_fn=False, mesh=None,
    )  # fmt: skip
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    return step(model, state, tokens, jax.random.PRNGKey(100))


def test_residual_mse_coeff_trains_finite():
    _, metrics = _step_with_chunkwise_recon(residual_mse_coeff=0.3)
    assert all(bool(jnp.isfinite(v).all()) for v in metrics.values())
    assert metrics["loss/ChunkwiseSubsetReconLoss"] > 0


def test_residual_mse_coeff_logs_e2e_and_per_block_breakdown_separately():
    cfg = tiny_simple_mlp_cfg()
    _, metrics = _step_with_chunkwise_recon(residual_mse_coeff=0.3)
    name = "loss/ChunkwiseSubsetReconLoss"
    assert name in metrics
    assert f"{name}/e2e" in metrics
    assert float(metrics[f"{name}/e2e"]) != float(metrics[name])  # combined != bare e2e
    for i in range(cfg.n_layer):
        assert f"{name}/residual_mse/block_{i}" in metrics
    assert f"{name}/residual_mse/block_{cfg.n_layer}" not in metrics


def test_residual_mse_coeff_disabled_has_no_per_block_breakdown():
    _, metrics = _step_with_chunkwise_recon(residual_mse_coeff=None)
    name = "loss/ChunkwiseSubsetReconLoss"
    assert f"{name}/e2e" in metrics  # e2e always logged
    assert not any(k.startswith(f"{name}/residual_mse/") for k in metrics)


def test_residual_mse_coeff_zero_matches_disabled():
    # coeff=0.0 (collects + reduces block resids, then multiplies by 0) must reproduce
    # coeff=None (skips collection) to float32 precision — mathematically a no-op, but
    # the extra zero-weighted ops still reshape the traced HLO, so XLA's fusion/op-order
    # elsewhere in the SAME graph (e.g. the shared CI-fn gradient) isn't bit-exact across
    # backends (float addition isn't associative; SPEC D4 makes the same allowance for
    # device-count reassociation).
    _, metrics_disabled = _step_with_chunkwise_recon(residual_mse_coeff=None)
    _, metrics_zero = _step_with_chunkwise_recon(residual_mse_coeff=0.0)
    for key in metrics_disabled:
        assert float(metrics_disabled[key]) == pytest.approx(float(metrics_zero[key])), key


def test_residual_mse_with_persistent_pgd_adversary():
    """The adversary's warmup + final ascent must run through the combined (e2e +
    residual-mse) objective without shape/finiteness errors (SPEC S35 x S13'/S14')."""
    cfg = tiny_simple_mlp_cfg()
    seq = 16
    n_warmup = 1
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = tiny_simple_mlp_chunkwise_ci_fn(model, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    src = init_persistent_sources(
        model.site_names,
        tuple(s.C for s in model.sites),
        (1, seq),
        jnp.float32,
        jax.random.PRNGKey(3),
    )
    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=0.5,
        source_shape="sc",
        optimizer=AdamPGDConfig(
            beta1=0.5, beta2=0.99, lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025)
        ),
        n_warmup_steps=n_warmup,
        residual_mse_coeff=0.2,
    )
    assert ppgd_cfg.coeff is not None
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
                    coeff=ppgd_cfg.coeff,
                    adam=ppgd_cfg.optimizer,
                    n_warmup=ppgd_cfg.n_warmup_steps,
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
            ),  # fmt: skip
            ppgd_cfg,
        ),
        model.site_names,
    )
    step = make_train_step(
        model_static=model, losses=losses, components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100, remat_recon_forwards=True, remat_ci_fn=False, mesh=None,
    )  # fmt: skip
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    n_steps = 3
    for i in range(n_steps):
        state, metrics = step(model, state, tokens, jax.random.PRNGKey(100 + i))
        assert all(bool(jnp.isfinite(v).all()) for v in metrics.values())
    assert int(state.training.step) == n_steps
    adv = state.training.adversaries[ppgd_cfg.type]
    assert float(adv.opt_state.step_count) == n_steps * (n_warmup + 1)
    for v in adv.sources.values():
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
