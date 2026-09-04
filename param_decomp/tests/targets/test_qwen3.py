"""CPU tests for the Qwen3 family target at a tiny config.

Mirrors `test_llama31.py` over the shared GLU-transformer machinery: the
`DecomposedModel` contract (mask=1 identity reconstructs the clean forward, ablation
changes logits) and one full SPEC step — plus the family-specific pin that the QK-norm is
actually load-bearing in the forward. Direct HF parity lives in `param_decomp/tests/targets/qwen3_hf_parity/`.
"""

from dataclasses import replace
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp

from param_decomp.core.components import SiteC, SiteDims, SiteSpec, init_component_stacks
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.init_placed import (
    ComponentInitializer,
    init_model_component_stacks_placed,
)
from param_decomp.core.model import PlacedModel
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import hsdp_mesh, place_target
from param_decomp.targets.glu_transformer import (
    GLU_ANATOMY,
    GatedMLP,
    GLUConfig,
    GLUDecomposedModel,
    GLULayer,
    build_decomposed_lm,
    default_inv_freq,
    glu_site_specs,
    neuron_aligned_component_initializer,
    parse_site_name,
    site_dims,
    site_name,
    validate_neuron_aligned_capacity,
)
from param_decomp.targets.qwen3 import (
    Qwen3FrozenAttn,
    qwen3_0_6b_base_config,
    qwen3_0_6b_config,
    qwen3_1_7b_base_config,
    qwen3_1_7b_config,
    qwen3_4b_base_config,
    qwen3_4b_config,
    qwen3_8b_base_config,
    qwen3_8b_config,
    qwen3_14b_base_config,
    qwen3_14b_config,
)
from param_decomp.targets.testing import capture_clean, run_clean, run_masked
from param_decomp.targets.transformer_taps import (
    attention_input_tap_key,
    attention_output_tap_key,
)


def test_released_qwen3_architectures():
    pairs = (
        (qwen3_0_6b_base_config(), qwen3_0_6b_config(), (28, 16, 1024, 3072), True),
        (qwen3_1_7b_base_config(), qwen3_1_7b_config(), (28, 16, 2048, 6144), True),
        (qwen3_4b_base_config(), qwen3_4b_config(), (36, 32, 2560, 9728), True),
        (qwen3_8b_base_config(), qwen3_8b_config(), (36, 32, 4096, 12288), False),
        (qwen3_14b_base_config(), qwen3_14b_config(), (40, 40, 5120, 17408), False),
    )
    for base, posttrained, expected_shape, tied in pairs:
        assert (base.n_layer, base.n_head, base.n_embd, base.n_intermediate) == expected_shape
        assert base.n_kv_head == 8
        assert base.head_dim == 128
        assert base.max_position_embeddings == 32768
        assert posttrained == replace(base, max_position_embeddings=40960)
        assert base.tie_word_embeddings is posttrained.tie_word_embeddings is tied

    # Qwen3-0.6B's explicit 128-wide heads make q/o rectangular in the unusual
    # direction: deriving head_dim as hidden_size / n_heads would silently halve them.
    small = qwen3_0_6b_base_config()
    assert site_dims(small, "q") == SiteDims(d_in=1024, d_out=2048)
    assert site_dims(small, "o") == SiteDims(d_in=2048, d_out=1024)


def test_neuron_aligned_init_handles_rectangular_q_and_gqa_exactly():
    cfg = replace(_tiny_qwen_cfg(), n_embd=16)
    capacities = {
        "q": cfg.n_head * cfg.head_dim,
        "k": cfg.n_kv_head * cfg.head_dim,
        "v": cfg.n_kv_head * cfg.head_dim,
        "o": cfg.n_head * cfg.head_dim,
        "gate": cfg.n_intermediate,
        "up": cfg.n_intermediate,
        "down": cfg.n_intermediate,
    }
    sites = glu_site_specs(
        cfg,
        tuple(SiteC(site_name(4, kind), c) for kind, c in capacities.items()),
    )
    model = _tiny_decomposed_qwen(cfg, sites, jax.random.PRNGKey(0))
    components = neuron_aligned_component_initializer(model, jax.random.PRNGKey(1))

    assert capacities["q"] > cfg.n_embd
    assert capacities["k"] < capacities["q"]
    for delta in model.weight_deltas(components).values():
        assert jnp.array_equal(delta, jnp.zeros_like(delta))
    for _name, site_components in components.sites_items():
        assert jnp.all(jnp.linalg.norm(site_components.V, axis=0) > 0)
        assert jnp.all(jnp.linalg.norm(site_components.U, axis=1) > 0)


def test_neuron_aligned_init_runs_directly_into_placed_stacks():
    cfg = replace(_tiny_qwen_cfg(), n_embd=16)
    capacities = {"q": 32, "k": 16, "v": 16, "o": 32, "gate": 64, "up": 64, "down": 64}
    sites = glu_site_specs(
        cfg, tuple(SiteC(site_name(4, kind), c) for kind, c in capacities.items())
    )
    model = _tiny_decomposed_qwen(cfg, sites, jax.random.PRNGKey(0))
    mesh = hsdp_mesh(1, 1, 1)
    rules = from_config("ddp", mesh, sites)
    placed = place_target(model, rules)

    with jax.set_mesh(mesh):
        components = init_model_component_stacks_placed(
            placed,
            jax.random.PRNGKey(1),
            rules,
            cast(ComponentInitializer, neuron_aligned_component_initializer),
        )

    for delta in placed.model.weight_deltas(components).values():
        assert jnp.array_equal(delta, jnp.zeros_like(delta))


def test_neuron_aligned_init_accepts_surplus_gqa_components():
    cfg = replace(_tiny_qwen_cfg(), n_embd=16)
    [k_spec] = glu_site_specs(
        cfg,
        (SiteC("layers.4.self_attn.k_proj", cfg.n_head * cfg.head_dim),),
    )

    validate_neuron_aligned_capacity(GLU_ANATOMY, k_spec)


def _tiny_qwen_cfg() -> GLUConfig:
    """Qwen3-shaped tiny config: QK-norm attention, plain RoPE."""
    return GLUConfig(
        vocab_size=64,
        n_layer=8,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        head_dim=8,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        tie_word_embeddings=False,
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
            n((qd, d)),
            n((kvd, d)),
            n((kvd, d)),
            n((d, qd)),
            cfg.n_head,
            cfg.n_kv_head,
            cfg.head_dim,
            cfg.n_rep,
            "auto",
            # non-trivial norm weights (≈1) so a wrong/missing norm application shows
            q_norm=1.0 + 0.1 * jax.random.normal(next(ks), (cfg.head_dim,)),
            k_norm=1.0 + 0.1 * jax.random.normal(next(ks), (cfg.head_dim,)),
            eps=cfg.rms_norm_eps,
        )

    def layer():
        attn = fattn()
        mlp = GatedMLP(Wg=n((di, d)), Wu=n((di, d)), Wd=n((d, di)))
        return GLULayer(jnp.ones((d,)), jnp.ones((d,)), attn, mlp)

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

    clean = run_clean(model, tokens)
    assert clean.shape == (b, t, cfg.vocab_size)

    # mask=1 identity through the QK-norm (V@U + (W − V@U); exact only in exact math).
    names = model.site_names
    ones_masks = {s.name: jnp.ones((b, t, s.C)) for s in model.sites}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    full = run_masked(
        model,
        model.prepare_compute_weights(vu, None),
        tokens,
        ones_masks,
        ones_delta,
        None,
        True,
        remat=False,
    )
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    # zero-mask + zero-delta on layer 4's decomposed sites must CHANGE the logits (q is
    # live on the attention path ahead of QK-norm/RoPE/SDPA).
    zero_mask = {s.name: jnp.zeros((b, t, s.C)) for s in model.sites}
    zero_delta = {s: jnp.zeros((b, t)) for s in names}
    ablated = run_masked(
        model,
        model.prepare_compute_weights(vu, None),
        tokens,
        zero_mask,
        zero_delta,
        None,
        True,
        remat=False,
    )
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
    assert not jnp.allclose(run_clean(model, tokens), run_clean(scaled, tokens), atol=1e-4)

    qkv_input = attention_input_tap_key(4)
    taps = capture_clean(model, tokens, (qkv_input,))
    scaled_taps = capture_clean(scaled, tokens, (qkv_input,))
    assert jnp.array_equal(taps[qkv_input], scaled_taps[qkv_input])
    attention_output = attention_output_tap_key(4)
    o_tap = capture_clean(model, tokens, (attention_output,))[attention_output]
    o_tap_scaled = capture_clean(scaled, tokens, (attention_output,))[attention_output]
    assert not jnp.allclose(o_tap, o_tap_scaled)


def test_attention_pattern_from_qk_applies_that_layers_qk_norm():
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
    p4 = model.attention_pattern_from_qk("layers.4.self_attn.q_proj", q, k)
    p5 = model.attention_pattern_from_qk("layers.5.self_attn.q_proj", q, k)
    assert p4.shape == (b, cfg.n_head, t, t)
    assert not jnp.allclose(p4, p5)
    assert parse_site_name("layers.5.self_attn.q_proj") == (5, "q")


def test_step_trains():
    """One full generic train step over the qwen tiny target — pins that the family plugs
    into the engine (the step machinery itself is pinned in `test_llama31.py`)."""
    import optax

    from param_decomp.core.configs import (
        FaithfulnessLossConfig,
        ImportanceMinimalityLossConfig,
        StochasticReconSubsetLossConfig,
        UniformKSubsetRoutingConfig,
    )
    from param_decomp.core.objective import build_objective
    from param_decomp.core.schedule import Knot, ScheduleConfig
    from param_decomp.core.train import (
        Decomposition,
        ForwardSubstrate,
        TrainingItem,
        TrainState,
        make_train_step,
    )
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
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    loss_terms = build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                ),
            ),
            StochasticReconSubsetLossConfig(
                routing=UniformKSubsetRoutingConfig(),
                coeff=0.5,
                n_mask_samples=1,
            ),
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
        objective=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        faithfulness=faithfulness_loss_for(model),
    )
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    state, metrics = step(placed, state, tokens, jax.random.PRNGKey(100))
    assert all(jnp.isfinite(jnp.asarray(v)).all() for v in metrics.values())
    assert int(state.training.step) == 1
