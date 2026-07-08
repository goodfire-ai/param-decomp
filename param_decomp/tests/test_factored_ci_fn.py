"""FactoredCIFn: shapes/partition, stop-grad-V, ctx zero-init, gate-only equivalence,
and the end-to-end train-step smoke (the jitted VPD step with a factored CI fn)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import PRNGKeyArray

from param_decomp.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.ci_fn import CI, FactoredCIArch, FactoredCIFn, FactoredCtxArch, build_ci_fn
from param_decomp.components import DecompVU, SiteSpec, init_decomp_vu
from param_decomp.configs import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
    UniformKSubsetRoutingConfig,
)
from param_decomp.recon import build_loss_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.tests.test_llama_simple_mlp import (
    _MIXED_SITE_CS,
    _tiny_cfg,
    _tiny_decomposed_model,
)
from param_decomp.train import TrainState, make_train_step

SITES = (SiteSpec("h.0.attn.q_proj", 16, 16, 8), SiteSpec("h.0.mlp.c_fc", 16, 32, 12))
CTX = FactoredCtxArch(
    taps=("resid.0",), input_dim=16, d_model=8, n_blocks=1, n_heads=2, mlp_hidden=16, rank=4,
    summary_k=None, modulate_slopes=False,
)  # fmt: skip
CTX_FULL = FactoredCtxArch(
    taps=("resid.0",), input_dim=16 + 2 * 3, d_model=8, n_blocks=1, n_heads=2, mlp_hidden=16,
    rank=4, summary_k=3, modulate_slopes=True,
)  # fmt: skip


def _taps(key: PRNGKeyArray):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "h.0.attn.q_proj": jax.random.normal(k1, (2, 6, 16)),
        "h.0.mlp.c_fc": jax.random.normal(k2, (2, 6, 16)),
        "resid.0": jax.random.normal(k3, (2, 6, 16)),
    }


def test_factored_shapes_and_partition():
    vu = init_decomp_vu(SITES, jax.random.key(0))
    taps = _taps(jax.random.key(1))
    for ctx in (None, CTX, CTX_FULL):
        fn = build_ci_fn(FactoredCIArch(ctx=ctx), SITES, jax.random.key(2))
        assert fn.output_names == tuple(s.name for s in SITES)
        assert fn.expects_axes == ("sequence",)
        ci = fn(taps, vu=vu, remat=False)
        assert isinstance(ci, CI)
        for spec in SITES:
            assert ci.logits[spec.name].shape == (2, 6, spec.C)
            assert jnp.all(ci.lower[spec.name] >= 0) and jnp.all(ci.lower[spec.name] <= 1)
    gate_only = build_ci_fn(FactoredCIArch(ctx=None), SITES, jax.random.key(2))
    assert gate_only.input_names == tuple(s.name for s in SITES)
    with_ctx = build_ci_fn(FactoredCIArch(ctx=CTX), SITES, jax.random.key(2))
    assert set(with_ctx.input_names) == {*(s.name for s in SITES), "resid.0"}


def test_factored_stop_grad_v():
    vu = init_decomp_vu(SITES, jax.random.key(0))
    taps = _taps(jax.random.key(1))
    fn = build_ci_fn(FactoredCIArch(ctx=CTX), SITES, jax.random.key(2))

    def loss(v: DecompVU):
        return sum(jnp.sum(x) for x in fn(taps, vu=v, remat=False).logits.values())

    grad = jax.grad(loss)(vu)
    for leaf in jax.tree_util.tree_leaves(grad):
        assert float(jnp.abs(leaf).max()) == 0.0, "CI gradient reached V/U through the gate"


def test_factored_ctx_zero_init_matches_gate_only():
    """The zero-init ctx out_proj kills EVERY z-dependent term at init — additive readout,
    slope modulation, and (transitively) the summary inputs — so all variants start at the
    same `0.1·â + β` logits."""
    vu = init_decomp_vu(SITES, jax.random.key(0))
    taps = _taps(jax.random.key(1))
    gate_only = build_ci_fn(FactoredCIArch(ctx=None), SITES, jax.random.key(2))
    lg_gate = gate_only(taps, vu=vu, remat=False).logits
    for ctx in (CTX, CTX_FULL):
        with_ctx = build_ci_fn(FactoredCIArch(ctx=ctx), SITES, jax.random.key(2))
        lg_ctx = with_ctx(taps, vu=vu, remat=False).logits
        for site in lg_gate:
            assert jnp.allclose(lg_ctx[site], lg_gate[site]), (ctx.summary_k, site)


def test_factored_train_step_end_to_end():
    """The full jitted VPD step (stoch + PPGD + faith + imp-min) with a factored CI fn:
    finite losses, both param sets actually train, sources stay projected."""
    from param_decomp.targets.llama_simple_mlp import site_specs

    cfg = _tiny_cfg()
    seq = 16
    sites = site_specs(cfg, _MIXED_SITE_CS)
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ctx = FactoredCtxArch(
        taps=("resid.2", "resid.3"), input_dim=2 * cfg.n_embd + 4 * 3,
        d_model=16, n_blocks=1, n_heads=2, mlp_hidden=32, rank=4,
        summary_k=3, modulate_slopes=True,
    )  # fmt: skip
    ci_fn = build_ci_fn(FactoredCIArch(ctx=ctx), lm.sites, jax.random.PRNGKey(2))
    assert isinstance(ci_fn, FactoredCIFn)
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    src = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), (1, seq), jnp.float32, jax.random.PRNGKey(3)
    )
    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=0.5,
        scope=SCScope(),
        optimizer=AdamPGDConfig(
            beta1=0.5, beta2=0.99, lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025)
        ),
        n_warmup_steps=2,
    )
    assert ppgd_cfg.coeff is not None
    state = TrainState(
        components=vu, ci_fn=ci_fn,
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
    )  # fmt: skip
    loss_terms = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
            ),
            ChunkwiseSubsetReconLossConfig(
                routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=2, n_samples=1
            ),
            ppgd_cfg,
        ),
        lm.site_names,
    )
    step = make_train_step(
        lm=lm,
        losses=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=True,
        remat_ci_fn=False,
        mesh=None,
    )

    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    site0 = lm.site_names[0]
    # The jitted step donates its state buffers — snapshot pre-training values to host.
    beta0 = np.asarray(ci_fn.beta[site0])
    v0 = np.asarray(vu.site(site0)[0])
    n_steps = 3
    for i in range(n_steps):
        state, metrics = step(lm, state, tokens, jax.random.PRNGKey(100 + i))
        assert all(bool(jnp.isfinite(v).all()) for v in metrics.values())
    assert int(state.step) == n_steps

    new_ci = state.ci_fn
    assert isinstance(new_ci, FactoredCIFn)
    assert not jnp.allclose(new_ci.beta[site0], beta0), "ci_fn gate did not train"
    assert new_ci.alpha[site0].dtype == jnp.float32
    v_new, _ = state.components.site(site0)
    assert not jnp.allclose(v_new, v0), "components did not train"
    assert v_new.dtype == jnp.float32
    for s in state.adversaries[ppgd_cfg.type].sources.values():
        assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0
