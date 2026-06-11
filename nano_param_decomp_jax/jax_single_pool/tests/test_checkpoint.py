"""Round-trip + resume-continuation tests for `checkpoint.py` (orbax) on the generic
trainer state (SPEC S22): a restored `TrainState` must continue the EXACT trajectory —
including the persistent adversary's sources and Adam moments."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from jax_single_pool.adversary import init_persistent_sources, init_sources_adam_state
from jax_single_pool.checkpoint import make_checkpoint_manager, restore_latest, save_state
from jax_single_pool.ci_fn import CIArch, init_ci_fn
from jax_single_pool.llama8b import (
    init_decomp_vu,
    llama_decomposed_lm,
    llama_site_specs,
    mlp_family_site_cs,
)
from jax_single_pool.recon import subset_chunk_plan
from jax_single_pool.tests.test_llama8b import _tiny_cfg, _tiny_target
from jax_single_pool.train import TrainState, make_train_step
from param_decomp_config.losses import (
    AdamPGDConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
)
from param_decomp_config.schedule import ScheduleConfig


def _build(seed: int):
    cfg = _tiny_cfg()
    tgt = _tiny_target(cfg, 3, jax.random.PRNGKey(0))
    C, seq = 8, 16
    sites = llama_site_specs(cfg, mlp_family_site_cs(3, 4, C))
    lm = llama_decomposed_lm(cfg, sites)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(seed))
    ci_fn = init_ci_fn(CIArch(16, 2, 2, 32), lm.sites, jax.random.PRNGKey(seed + 1))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    src = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), seq, jax.random.PRNGKey(seed + 2)
    )
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources=src, sources_adam_state=init_sources_adam_state(src), step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    step = make_train_step(
        lm=lm,
        faith_coeff=1e5,
        stoch_coeff=0.5,
        imp_min=ImportanceMinimalityLossConfig(
            coeff=5e-6, pnorm=2.0, beta=0.2,
            p_anneal_start_frac=0.0, p_anneal_final_p=0.4, p_anneal_end_frac=1.0,
        ),
        adversary=PersistentPGDReconLossConfig(
            coeff=0.5,
            scope=SCScope(),
            optimizer=AdamPGDConfig(
                beta1=0.5, beta2=0.99,
                lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
            ),
            n_warmup_steps=1,
        ),
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100, recon_plan=subset_chunk_plan(lm.site_names, 3, 1),
        remat_recon_forwards=True, mesh=None,
    )  # fmt: skip
    resid = jax.random.normal(jax.random.PRNGKey(9), (2, seq, cfg.n_embd)) * 0.5
    return tgt, state, step, resid


def test_roundtrip_and_exact_resume(tmp_path: Path):
    tgt, state, step, resid = _build(seed=1)
    for i in range(2):
        state, _ = step(state, tgt, resid, jax.random.PRNGKey(i))

    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 2, state)

    # Restore onto a DIFFERENTLY-seeded reference: every leaf must come from disk.
    _, fresh, _, _ = _build(seed=7)
    restored = restore_latest(mgr, fresh)
    assert restored is not None
    loaded, ckpt_step = restored
    assert ckpt_step == 2
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(loaded), strict=True):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))

    # SPEC S22: the restored state continues the exact trajectory.
    state_cont, m_cont = step(state, tgt, resid, jax.random.PRNGKey(100))
    loaded_cont, m_load = step(loaded, tgt, resid, jax.random.PRNGKey(100))
    for k in m_cont:
        assert float(m_cont[k]) == float(m_load[k]), k
    for a, b in zip(jax.tree.leaves(state_cont), jax.tree.leaves(loaded_cont), strict=True):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))


def test_no_checkpoint_returns_none(tmp_path: Path):
    _, fresh, _, _ = _build(seed=7)
    mgr = make_checkpoint_manager(tmp_path / "empty", keep_last=2)
    assert restore_latest(mgr, fresh) is None
