"""CPU tests for the Llama target + generic trainer at a tiny config.

Validates the `DecomposedModel` contract (clean == all-frozen masked forward, shapes) and
the full SPEC step (trains, VPD loss signature, adversary state advances) — for the
MLP site family AND for attention (q/k/v/o) sites with heterogeneous per-site C —
without real weights or a GPU.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from vendored_jax.llama import LlamaConfig, llama3_inv_freq

from jax_single_pool.adversary import init_persistent_sources, init_sources_adam_state
from jax_single_pool.ci_fn import CIArch, init_ci_fn
from jax_single_pool.llama8b import (
    DecompVU,
    FrozenAttn,
    SuffixLayer,
    Target,
    canonical_site_cs,
    first_decomposed_layer,
    init_decomp_vu,
    llama_decomposed_lm,
    llama_site_specs,
    mlp_family_site_cs,
    parse_site_name,
    site_name,
)
from jax_single_pool.lm import SiteC, SiteSpec
from jax_single_pool.recon import build_recon_terms
from jax_single_pool.train import TrainState, make_faith_warmup_step, make_train_step
from param_decomp_config.losses import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    SCScope,
)
from param_decomp_config.routing import UniformKSubsetRoutingConfig
from param_decomp_config.schedule import ScheduleConfig


def _tiny_cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        n_layer=8,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=128,
    )


def _tiny_target(cfg: LlamaConfig, first_layer: int, key: jax.Array) -> Target:
    ks = iter(jax.random.split(key, 1024))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def n(shape: tuple[int, ...], s: float | None = None) -> jax.Array:
        return jax.random.normal(next(ks), shape) * (s or d**-0.5)

    def fattn():
        return FrozenAttn(
            n((qd, d)), n((kvd, d)), n((kvd, d)), n((d, qd)),
            cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_rep,
        )  # fmt: skip

    def suffix_layer():
        return SuffixLayer(
            jnp.ones((d,)), jnp.ones((d,)), fattn(), n((di, d)), n((di, d)), n((d, di))
        )

    return Target(
        layers=[suffix_layer() for _ in range(cfg.n_layer - first_layer)],
        norm=jnp.ones((d,)), lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=llama3_inv_freq(cfg), eps=cfg.rms_norm_eps,
    )  # fmt: skip


def _mlp_sites(cfg: LlamaConfig, first: int, last: int, C: int) -> tuple[SiteSpec, ...]:
    return llama_site_specs(cfg, mlp_family_site_cs(first, last, C))


_QVDOWN_SITE_CS = (
    SiteC("layers.4.self_attn.q_proj", 8),
    SiteC("layers.4.self_attn.v_proj", 12),
    SiteC("layers.4.mlp.down_proj", 8),
)
"""Attention + MLP sites on one layer with heterogeneous per-site C."""


def test_site_name_helpers():
    assert site_name(18, "q") == "layers.18.self_attn.q_proj"
    assert site_name(18, "gate") == "layers.18.mlp.gate_proj"
    assert parse_site_name("layers.18.self_attn.o_proj") == (18, "o")
    assert parse_site_name("layers.2.mlp.up_proj") == (2, "up")
    with pytest.raises(AssertionError):
        parse_site_name("layers.18.self_attn.gate_proj")
    with pytest.raises(AssertionError):
        parse_site_name("model.layers.18.mlp.gate_proj")
    with pytest.raises(AssertionError):
        parse_site_name("embed_tokens")

    shuffled = (
        SiteC("layers.4.mlp.down_proj", 8),
        SiteC("layers.3.self_attn.v_proj", 4),
        SiteC("layers.4.self_attn.q_proj", 8),
    )
    assert canonical_site_cs(shuffled) == (
        SiteC("layers.3.self_attn.v_proj", 4),
        SiteC("layers.4.self_attn.q_proj", 8),
        SiteC("layers.4.mlp.down_proj", 8),
    )
    with pytest.raises(AssertionError):
        canonical_site_cs((SiteC("layers.3.mlp.up_proj", 4), SiteC("layers.3.mlp.up_proj", 8)))
    assert first_decomposed_layer(("layers.5.mlp.up_proj", "layers.3.self_attn.k_proj")) == 3


def test_llama_site_specs_dims():
    cfg = _tiny_cfg()
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    specs = llama_site_specs(
        cfg,
        canonical_site_cs(
            tuple(SiteC(site_name(2, kind), 4) for kind in ("q", "k", "v", "o", "gate", "down"))
        ),
    )

    def dims(s: SiteSpec) -> tuple[int, int, int]:
        return (s.d_in, s.d_out, s.C)

    by_name = {s.name: s for s in specs}
    assert dims(by_name["layers.2.self_attn.q_proj"]) == (cfg.n_embd, qd, 4)
    assert dims(by_name["layers.2.self_attn.k_proj"]) == (cfg.n_embd, kvd, 4)
    assert dims(by_name["layers.2.self_attn.o_proj"]) == (qd, cfg.n_embd, 4)
    assert dims(by_name["layers.2.mlp.gate_proj"]) == (cfg.n_embd, cfg.n_intermediate, 4)
    assert dims(by_name["layers.2.mlp.down_proj"]) == (cfg.n_intermediate, cfg.n_embd, 4)
    with pytest.raises(AssertionError, match="canonical"):
        llama_site_specs(cfg, tuple(reversed(mlp_family_site_cs(2, 2, 4))))


@pytest.mark.parametrize("first,last", [(4, 4), (3, 6)])
def test_clean_path_and_masked_identity(first: int, last: int):
    cfg = _tiny_cfg()
    tgt = _tiny_target(cfg, first, jax.random.PRNGKey(0))
    C = 8
    sites = _mlp_sites(cfg, first, last, C)
    lm = llama_decomposed_lm(cfg, sites)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    resid = jax.random.normal(jax.random.PRNGKey(2), (b, t, cfg.n_embd)) * 0.5

    clean = lm.clean_output(tgt, resid)
    assert clean.shape == (b, t, cfg.vocab_size)

    # SPEC S2: a masked forward with NO live sites is the frozen path — bit-identical
    # to the clean target.
    none_masked = lm.masked_output(tgt, vu, resid, {}, {}, None, (), True)
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    # All-live, masks=1, delta=1, route-everywhere reconstructs the frozen path up to
    # decomposition rounding (the V@U + (W − V@U) identity; exact only in exact math).
    names = lm.site_names
    ones_masks = {s: jnp.ones((b, t, C)) for s in names}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    full = lm.masked_output(tgt, vu, resid, ones_masks, ones_delta, None, names, True)
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    site_in = lm.site_inputs(tgt, resid)
    assert set(site_in) == set(names)
    deltas = lm.weight_deltas(tgt, vu)
    d, di = cfg.n_embd, cfg.n_intermediate
    assert deltas[names[0]].shape == (di, d)  # gate: (d_out, d_in)
    assert deltas[names[2]].shape == (d, di)  # down
    assert all(v.dtype == jnp.float32 for v in deltas.values())


def test_attention_sites_clean_and_masked_identity():
    cfg = _tiny_cfg()
    first = 4
    tgt = _tiny_target(cfg, first, jax.random.PRNGKey(0))
    sites = llama_site_specs(cfg, _QVDOWN_SITE_CS)
    lm = llama_decomposed_lm(cfg, sites)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    resid = jax.random.normal(jax.random.PRNGKey(2), (b, t, cfg.n_embd)) * 0.5

    # per-site heterogeneous C is preserved end to end
    assert {s.name: s.C for s in lm.sites} == {sc.name: sc.C for sc in _QVDOWN_SITE_CS}
    for spec in lm.sites:
        V, U = vu.site(spec.name)
        assert V.shape == (spec.d_in, spec.C) and U.shape == (spec.C, spec.d_out)

    clean = lm.clean_output(tgt, resid)
    none_masked = lm.masked_output(tgt, vu, resid, {}, {}, None, (), True)
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    names = lm.site_names
    ones_masks = {s.name: jnp.ones((b, t, s.C)) for s in lm.sites}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    full = lm.masked_output(tgt, vu, resid, ones_masks, ones_delta, None, names, True)
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted (attention sites)"

    # zero-mask + zero-delta on q alone must CHANGE the logits (the site is live on
    # the attention path, ahead of RoPE/SDPA)
    q_site = "layers.4.self_attn.q_proj"
    zero_mask = {q_site: jnp.zeros((b, t, 8))}
    zero_delta = {q_site: jnp.zeros((b, t))}
    ablated = lm.masked_output(tgt, vu, resid, zero_mask, zero_delta, None, (q_site,), True)
    assert not jnp.allclose(clean, ablated, atol=1e-4), "ablating q did nothing"

    site_in = lm.site_inputs(tgt, resid)
    assert set(site_in) == set(names)
    # q and v read the same post-LN1 residual; down reads the (di,) MLP inner acts
    assert jnp.array_equal(site_in[q_site], site_in["layers.4.self_attn.v_proj"])
    assert site_in["layers.4.mlp.down_proj"].shape == (b, t, cfg.n_intermediate)

    deltas = lm.weight_deltas(tgt, vu)
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    assert deltas[q_site].shape == (qd, cfg.n_embd)
    assert deltas["layers.4.self_attn.v_proj"].shape == (kvd, cfg.n_embd)


def test_o_site_masks_attention_output():
    cfg = _tiny_cfg()
    first = 4
    tgt = _tiny_target(cfg, first, jax.random.PRNGKey(0))
    o_site = "layers.4.self_attn.o_proj"
    sites = llama_site_specs(cfg, (SiteC(o_site, 8),))
    lm = llama_decomposed_lm(cfg, sites)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    resid = jax.random.normal(jax.random.PRNGKey(2), (b, t, cfg.n_embd)) * 0.5

    clean = lm.clean_output(tgt, resid)
    ones = lm.masked_output(
        tgt,
        vu,
        resid,
        {o_site: jnp.ones((b, t, 8))},
        {o_site: jnp.ones((b, t))},
        None,
        (o_site,),
        True,
    )
    assert jnp.allclose(clean, ones, atol=1e-4)
    # o's clean site input is the pre-o_proj attention output, shape (b, t, qd)
    site_in = lm.site_inputs(tgt, resid)
    assert site_in[o_site].shape == (b, t, cfg.n_head * cfg.head_dim)


@pytest.mark.parametrize(
    "site_cs",
    [mlp_family_site_cs(4, 4, 8), mlp_family_site_cs(3, 6, 8), _QVDOWN_SITE_CS],
    ids=["mlp_l4", "mlp_l3_6", "qv_down_l4"],
)
def test_step_trains_and_has_vpd_signature(site_cs: tuple[SiteC, ...]):
    cfg = _tiny_cfg()
    first = first_decomposed_layer(tuple(s.name for s in site_cs))
    tgt = _tiny_target(cfg, first, jax.random.PRNGKey(0))
    seq = 16
    n_warmup = 2
    sites = llama_site_specs(cfg, site_cs)
    lm = llama_decomposed_lm(cfg, sites)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = init_ci_fn(CIArch(d_model=16, n_blocks=2, n_heads=2, mlp_hidden=32),
                       lm.sites, jax.random.PRNGKey(2))  # fmt: skip
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    src = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), (1, seq), jax.random.PRNGKey(3)
    )
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources={"PersistentPGDReconLoss": src},
        sources_opt_state={"PersistentPGDReconLoss": init_sources_adam_state(src)},
        step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    loss_spec = build_recon_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=2.0,
                beta=0.2,
                p_anneal_start_frac=0.0,
                p_anneal_final_p=0.4,
                p_anneal_end_frac=1.0,
            ),
            ChunkwiseSubsetReconLossConfig(
                routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=3, n_samples=1
            ),
            PersistentPGDReconLossConfig(
                coeff=0.5,
                scope=SCScope(),
                optimizer=AdamPGDConfig(
                    beta1=0.5,
                    beta2=0.99,
                    lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
                ),
                n_warmup_steps=n_warmup,
            ),
        ),
        lm.site_names,
        n_mask_samples=1,
        sampling="continuous",
    )
    step = make_train_step(
        lm=lm,
        loss_spec=loss_spec,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=True,
        mesh=None,
    )

    resid = jax.random.normal(jax.random.PRNGKey(4), (2, seq, cfg.n_embd)) * 0.5
    n_steps = 4
    losses = []
    for i in range(n_steps):
        state, m = step(state, tgt, resid, jax.random.PRNGKey(100 + i))
        losses.append({k: float(v) for k, v in m.items()})

    assert all(jnp.isfinite(jnp.array(list(m.values()))).all() for m in losses)
    assert int(state.step) == n_steps
    # SPEC S13: n_warmup + 1 source-Adam updates per training step, moments persist.
    ppgd_opt_state = state.sources_opt_state["PersistentPGDReconLoss"]
    assert float(ppgd_opt_state.step_count) == n_steps * (n_warmup + 1)
    # SPEC S15: sources stay projected to [0,1].
    for v in state.sources["PersistentPGDReconLoss"].values():
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    # SPEC S9: p annealed below its 2.0 start by step 4 of 100.
    assert losses[-1]["p_imp"] < 2.0
    # fp32 masters preserved through updates (SPEC N1).
    assert isinstance(state.components, DecompVU)
    for V, U in state.components.vu.values():
        assert V.dtype == jnp.float32 and U.dtype == jnp.float32
    assert state.ci_fn.in_proj_w.dtype == jnp.float32


def test_faith_warmup_decreases_faith():
    cfg = _tiny_cfg()
    first = 3
    tgt = _tiny_target(cfg, first, jax.random.PRNGKey(0))
    sites = _mlp_sites(cfg, 3, 4, 8)
    lm = llama_decomposed_lm(cfg, sites)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    opt = optax.adamw(1e-2, weight_decay=0.0)
    wstep = make_faith_warmup_step(lm, opt)
    ostate = opt.init(eqx.filter(vu, eqx.is_array))
    first_loss: float | None = None
    loss = None
    for _ in range(30):
        vu, ostate, loss = wstep(vu, ostate, tgt)
        first_loss = float(loss) if first_loss is None else first_loss
    assert first_loss is not None and loss is not None
    assert float(loss) < first_loss * 0.9, (first_loss, float(loss))


def test_decomp_vu_shapes_fp32():
    cfg = _tiny_cfg()
    sites = llama_site_specs(cfg, _QVDOWN_SITE_CS)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    V_q, U_q = vu.site("layers.4.self_attn.q_proj")
    V_v, U_v = vu.site("layers.4.self_attn.v_proj")
    V_d, U_d = vu.site("layers.4.mlp.down_proj")
    assert V_q.shape == (d, 8) and U_q.shape == (8, qd)
    assert V_v.shape == (d, 12) and U_v.shape == (12, kvd)
    assert V_d.shape == (di, 8) and U_d.shape == (8, d)
    assert isinstance(vu, DecompVU)
    assert all(a.dtype == jnp.float32 for pair in vu.vu.values() for a in pair)


def test_fresh_pgd_adversary_step():
    """Fresh per-batch sign-PGD (torch PGDReconLoss as the TRAINING adversary):
    no persistent source state, metrics keyed `loss/PGDReconLoss`, sources
    sampled+ascended inside the step, and the ascent strength responds to n_steps."""
    cfg = _tiny_cfg()
    site_cs = (
        SiteC("layers.4.self_attn.q_proj", 8),
        SiteC("layers.4.mlp.gate_proj", 8),
        SiteC("layers.4.mlp.up_proj", 8),
        SiteC("layers.4.mlp.down_proj", 12),
    )
    first = first_decomposed_layer(tuple(s.name for s in site_cs))
    tgt = _tiny_target(cfg, first, jax.random.PRNGKey(0))
    seq = 16
    sites = llama_site_specs(cfg, site_cs)
    lm = llama_decomposed_lm(cfg, sites)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = init_ci_fn(CIArch(d_model=16, n_blocks=1, n_heads=2, mlp_hidden=32),
                       lm.sites, jax.random.PRNGKey(2))  # fmt: skip
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    def make_state() -> TrainState:
        return TrainState(
            components=vu, ci_fn=ci_fn,
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            sources={}, sources_opt_state={},
            step=jnp.zeros((), jnp.int32),
        )  # fmt: skip

    def run_step(n_ascent_steps: int) -> tuple[TrainState, dict[str, jax.Array]]:
        loss_spec = build_recon_terms(
            (
                FaithfulnessLossConfig(coeff=1e7),
                ImportanceMinimalityLossConfig(
                    coeff=2e-4,
                    pnorm=2.0,
                    beta=0.5,
                    p_anneal_start_frac=0.0,
                    p_anneal_final_p=0.4,
                    p_anneal_end_frac=1.0,
                ),
                ChunkwiseSubsetReconLossConfig(
                    routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=4, n_samples=1
                ),
                PGDReconLossConfig(
                    coeff=0.5,
                    init="random",
                    step_size=1.0,
                    n_steps=n_ascent_steps,
                    mask_scope="bsc",
                ),
            ),
            lm.site_names,
            n_mask_samples=1,
            sampling="continuous",
        )
        step = make_train_step(
            lm=lm,
            loss_spec=loss_spec,
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=100,
            remat_recon_forwards=False,
            mesh=None,
        )
        resid = jax.random.normal(jax.random.PRNGKey(4), (2, seq, cfg.n_embd)) * 0.5
        return step(make_state(), tgt, resid, jax.random.PRNGKey(100))

    state, metrics = run_step(n_ascent_steps=1)
    assert "loss/PGDReconLoss" in metrics
    assert "loss/PersistentPGDReconLoss" not in metrics and "src_lr" not in metrics
    assert jnp.isfinite(
        jnp.array(
            [
                float(metrics[k])
                for k in ("total", "loss/PGDReconLoss", "loss/ChunkwiseSubsetReconLoss")
            ]
        )
    ).all()
    assert state.sources == {}, "fresh adversary carries no persistent sources"
    assert state.sources_opt_state == {}
    assert int(state.step) == 1

    _, metrics_unascended = run_step(n_ascent_steps=0)
    assert float(metrics["loss/PGDReconLoss"]) >= float(metrics_unascended["loss/PGDReconLoss"]), (
        "one sign step from the same init must not weaken the adversary"
    )
