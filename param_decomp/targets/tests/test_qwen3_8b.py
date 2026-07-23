"""CPU tests for the Qwen3 family target at a tiny config.

Mirrors `test_llama8b.py` over the shared GLU-transformer machinery: the
`DecomposedModel` contract (clean == all-frozen masked forward, mask=1 identity, ablation
changes logits) and one full SPEC step — plus the family-specific pin that the QK-norm is
actually load-bearing in the forward. Direct HF parity lives in `tests/qwen3_hf_parity/`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from param_decomp.core.components import SiteC, SiteSpec, init_component_stacks
from param_decomp.targets.glu_transformer import (
    GLUConfig,
    GLUDecomposedModel,
    GLULayer,
    build_decomposed_lm,
    default_inv_freq,
    glu_site_specs,
    parse_site_name,
)
from param_decomp.targets.qwen3_8b import Qwen3FrozenAttn


def _tiny_qwen_cfg() -> GLUConfig:
    """Qwen3-shaped tiny config: QK-norm attention, plain RoPE."""
    return GLUConfig(
        vocab_size=64,
        n_layer=8,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
    )


def _tiny_decomposed_qwen(
    cfg: GLUConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> GLUDecomposedModel:
    """A tiny random Qwen3-family model — the CPU-test analog of
    `load_decomposed_qwen3_from_hf` (`testing.tiny_glu_decomposed_lm`'s sibling)."""
    ks = iter(jax.random.split(key, 1024))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def n(shape: tuple[int, ...], s: float | None = None) -> jax.Array:
        return jax.random.normal(next(ks), shape) * (s or d**-0.5)

    def fattn():
        return Qwen3FrozenAttn(
            n((qd, d)), n((kvd, d)), n((kvd, d)), n((d, qd)),
            cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_rep,
            # non-trivial norm weights (≈1) so a wrong/missing norm application shows
            q_norm=1.0 + 0.1 * jax.random.normal(next(ks), (cfg.head_dim,)),
            k_norm=1.0 + 0.1 * jax.random.normal(next(ks), (cfg.head_dim,)),
            eps=cfg.rms_norm_eps,
        )  # fmt: skip

    def layer():
        return GLULayer(jnp.ones((d,)), jnp.ones((d,)), fattn(), n((di, d)), n((di, d)), n((d, di)))

    return build_decomposed_lm(
        embed=n((cfg.vocab_size, d), 0.02),
        layers=[layer() for _ in range(cfg.n_layer)],
        norm=jnp.ones((d,)),
        lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=default_inv_freq(cfg.head_dim, cfg.rope_theta),
        cfg=cfg,
        sites=sites,
    )


_QVDOWN_SITE_CS = (
    SiteC("layers.4.self_attn.q_proj", 8),
    SiteC("layers.4.self_attn.v_proj", 12),
    SiteC("layers.4.mlp.down_proj", 8),
)
"""Attention + MLP sites on one layer with heterogeneous per-site C."""


def test_clean_path_and_masked_identity():
    cfg = _tiny_qwen_cfg()
    sites = glu_site_specs(cfg, _QVDOWN_SITE_CS)
    model = _tiny_decomposed_qwen(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    clean = model.clean_output(tokens)
    assert clean.shape == (b, t, cfg.vocab_size)

    # SPEC S2: a masked forward with NO live sites is the frozen path — bit-identical.
    none_masked = model.masked_output(
        model.prepare_compute_weights(vu), tokens, {}, {}, None, (), True, remat=False
    )
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    # mask=1 identity through the QK-norm (V@U + (W − V@U); exact only in exact math).
    names = model.site_names
    ones_masks = {s.name: jnp.ones((b, t, s.C)) for s in model.sites}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    full = model.masked_output(
        model.prepare_compute_weights(vu), tokens, ones_masks, ones_delta, None, names, True,
        remat=False,
    )  # fmt: skip
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    # zero-mask + zero-delta on layer 4's decomposed sites must CHANGE the logits (q is
    # live on the attention path ahead of QK-norm/RoPE/SDPA).
    zero_mask = {s.name: jnp.zeros((b, t, s.C)) for s in model.sites}
    zero_delta = {s: jnp.zeros((b, t)) for s in names}
    ablated = model.masked_output(
        model.prepare_compute_weights(vu), tokens, zero_mask, zero_delta, None, names, True,
        remat=False,
    )  # fmt: skip
    assert not jnp.allclose(clean, ablated, atol=1e-4), "ablating layer 4 did nothing"


def test_qk_norm_is_load_bearing():
    """The QK-norm actually enters the forward: scaling ONE layer's `q_norm` changes the
    logits; the pre-projection q/k/v site inputs stay untouched (q/k sites decompose
    BEFORE the norm — the masked site output feeds q_norm → RoPE → SDPA); o's site input
    (the attention output) responds."""
    cfg = _tiny_qwen_cfg()
    sites = glu_site_specs(cfg, _QVDOWN_SITE_CS)
    model = _tiny_decomposed_qwen(cfg, sites, jax.random.PRNGKey(0))
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, 16), 0, cfg.vocab_size)

    attn = model.stacked.attn
    assert isinstance(attn, Qwen3FrozenAttn)
    assert attn.q_norm.shape == (cfg.n_layer, cfg.head_dim)
    # scale ONLY layer 4's q_norm so the residual ENTERING layer 4 stays untouched
    scaled = eqx.tree_at(lambda m: m.stacked.attn.q_norm, model, attn.q_norm.at[4].mul(2.0))
    assert not jnp.allclose(model.clean_output(tokens), scaled.clean_output(tokens), atol=1e-4)

    q_site = "layers.4.self_attn.q_proj"
    taps = model.read_activations(tokens, model.site_names)
    scaled_taps = scaled.read_activations(tokens, model.site_names)
    assert jnp.array_equal(taps[q_site], scaled_taps[q_site])
    o_site = "layers.4.self_attn.o_proj"
    o_tap = model.read_activations(tokens, (o_site,))[o_site]
    o_tap_scaled = scaled.read_activations(tokens, (o_site,))[o_site]
    assert not jnp.allclose(o_tap, o_tap_scaled)


def test_attn_pattern_applies_that_layers_qk_norm():
    """The target-owned attn-pattern recipe uses the site's LAYER norms: the same q/k
    flats produce different patterns for two layers whose q_norm weights differ."""
    cfg = _tiny_qwen_cfg()
    site_cs = (
        SiteC("layers.4.self_attn.q_proj", 8),
        SiteC("layers.4.self_attn.k_proj", 8),
        SiteC("layers.5.self_attn.q_proj", 8),
        SiteC("layers.5.self_attn.k_proj", 8),
    )
    sites = glu_site_specs(cfg, site_cs)
    model = _tiny_decomposed_qwen(cfg, sites, jax.random.PRNGKey(0))
    attn = model.stacked.attn
    assert isinstance(attn, Qwen3FrozenAttn)
    model = eqx.tree_at(lambda m: m.stacked.attn.q_norm, model, attn.q_norm.at[5].mul(3.0))

    b, t = 2, 9
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    q = jax.random.normal(jax.random.PRNGKey(1), (b, t, qd))
    k = jax.random.normal(jax.random.PRNGKey(2), (b, t, kvd))
    p4 = model.attn_pattern("layers.4.self_attn.q_proj", q, k)
    p5 = model.attn_pattern("layers.5.self_attn.q_proj", q, k)
    assert p4.shape == (b, cfg.n_head, t, t)
    assert not jnp.allclose(p4, p5)
    assert parse_site_name("layers.5.self_attn.q_proj") == (5, "q")


def test_step_trains():
    """One full generic train step over the qwen tiny target — pins that the family plugs
    into the engine (the step machinery itself is pinned in `test_llama8b.py`)."""
    import optax

    from param_decomp.core.configs import (
        ChunkwiseSubsetReconLossConfig,
        FaithfulnessLossConfig,
        ImportanceMinimalityLossConfig,
        UniformKSubsetRoutingConfig,
    )
    from param_decomp.core.recon import build_loss_terms
    from param_decomp.core.schedule import ScheduleConfig
    from param_decomp.core.train import Decomposition, TrainingItem, TrainState, make_train_step
    from param_decomp.targets.testing import tiny_glu_chunkwise_ci_fn

    cfg = _tiny_qwen_cfg()
    sites = glu_site_specs(cfg, _QVDOWN_SITE_CS)
    model = _tiny_decomposed_qwen(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = tiny_glu_chunkwise_ci_fn(model, jax.random.PRNGKey(2), n_blocks=1)
    opt_vu = optax.adamw(1e-3, weight_decay=0.0)
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
    loss_terms = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
            ),
            ChunkwiseSubsetReconLossConfig(
                routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=3, n_samples=1
            ),
        ),
        model.site_names,
    )
    step = make_train_step(
        model_static=model,
        losses=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=True,
        remat_ci_fn=False,
        mesh=None,
    )
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    state, metrics = step(model, state, tokens, jax.random.PRNGKey(100))
    assert all(jnp.isfinite(jnp.asarray(v)).all() for v in metrics.values())
    assert int(state.training.step) == 1
