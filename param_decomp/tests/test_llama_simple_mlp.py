"""CPU tests for the LlamaSimpleMLP target + generic trainer at a tiny config.

Mirrors `test_llama8b.py`: validates the `DecomposedModel` contract (clean == all-frozen
masked forward, shapes, site seams) and the full SPEC step — for mixed attention + MLP
sites with heterogeneous per-site C — without real weights or a GPU.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from param_decomp.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    build_ci_fn,
)
from param_decomp.components import DecompVU, SiteC, SiteSpec, init_decomp_vu
from param_decomp.configs import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
    UniformKSubsetRoutingConfig,
)
from param_decomp.lm import DecomposedModel
from param_decomp.recon import build_loss_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.targets.llama8b import FrozenAttn
from param_decomp.targets.llama_simple_mlp import (
    LlamaSimpleMLPConfig,
    SimpleMLPDecomposedModel,
    SimpleMLPLayer,
    build_decomposed_simple_mlp,
    canonical_site_cs,
    expand_wildcard_site_cs,
    parse_site_name,
    site_name,
    site_specs,
)
from param_decomp.train import (
    TrainState,
    make_faith_warmup_step,
    make_stale_ci_train_steps,
    make_train_step,
)


def _tiny_cfg() -> LlamaSimpleMLPConfig:
    return LlamaSimpleMLPConfig(
        vocab_size=64,
        n_layer=6,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rotary_base=10000.0,
        rms_norm_eps=1e-6,
        n_ctx=64,
    )


def _tiny_layers(cfg: LlamaSimpleMLPConfig, n: int, key: jax.Array) -> list[SimpleMLPLayer]:
    ks = iter(jax.random.split(key, 1024))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def rand(shape: tuple[int, ...]) -> jax.Array:
        return jax.random.normal(next(ks), shape) * d**-0.5

    return [
        SimpleMLPLayer(
            ln1=jnp.ones((d,)),
            ln2=jnp.ones((d,)),
            attn=FrozenAttn(
                rand((qd, d)),
                rand((kvd, d)),
                rand((kvd, d)),
                rand((d, qd)),
                cfg.n_head,
                cfg.n_kv_head,
                cfg.head_dim,
                cfg.n_rep,
            ),  # fmt: skip
            Wfc=rand((di, d)),
            Wdown=rand((d, di)),
        )
        for _ in range(n)
    ]


def _tiny_decomposed_model(
    cfg: LlamaSimpleMLPConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> SimpleMLPDecomposedModel:
    """A tiny random `SimpleMLPDecomposedModel` carrying a random embedding + full frozen
    layer stack plus the decomposition `sites`."""
    layers_key, embed_key = jax.random.split(key)
    layers = _tiny_layers(cfg, cfg.n_layer, layers_key)
    embed = jax.random.normal(embed_key, (cfg.vocab_size, cfg.n_embd)) * 0.02
    return build_decomposed_simple_mlp(
        embed=embed, layers=layers, norm=jnp.ones((cfg.n_embd,)), lm_head=embed,
        cfg=cfg, sites=sites,
    )  # fmt: skip


_MIXED_SITE_CS = (
    SiteC("h.2.attn.q_proj", 8),
    SiteC("h.2.attn.v_proj", 12),
    SiteC("h.2.mlp.c_fc", 8),
    SiteC("h.3.mlp.down_proj", 16),
)
"""Attention + MLP sites across two layers with heterogeneous per-site C."""


def _build_chunkwise_ci_fn(lm: DecomposedModel, key: jax.Array) -> ChunkwiseTransformerCIFn:
    """Old `init_ci_fn(CIArch(16, 2, 2, 32), lm.sites, key)` → the new chunkwise builder:
    one chunk reading the residual entering the first decomposed block, emitting CI for every
    site. `input_dim` is the target residual width (`n_embd`)."""
    site_names = lm.site_names
    first_block = min(parse_site_name(n)[0] for n in site_names)
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=_tiny_cfg().n_embd,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        mlp_hidden=32,
    )
    ci_fn = build_ci_fn(arch, lm.sites, key)
    assert isinstance(ci_fn, ChunkwiseTransformerCIFn)
    return ci_fn


def test_site_name_helpers():
    assert site_name(0, "q_proj") == "h.0.attn.q_proj"
    assert site_name(3, "c_fc") == "h.3.mlp.c_fc"
    assert parse_site_name("h.0.attn.o_proj") == (0, "o_proj")
    assert parse_site_name("h.2.mlp.down_proj") == (2, "down_proj")
    with pytest.raises(AssertionError):
        parse_site_name("h.0.attn.c_fc")
    with pytest.raises(AssertionError):
        parse_site_name("h.0.mlp.gate_proj")
    with pytest.raises(AssertionError):
        parse_site_name("layers.0.mlp.down_proj")

    shuffled = (
        SiteC("h.1.mlp.c_fc", 8),
        SiteC("h.0.mlp.down_proj", 4),
        SiteC("h.0.attn.k_proj", 8),
    )
    assert canonical_site_cs(shuffled) == (
        SiteC("h.0.attn.k_proj", 8),
        SiteC("h.0.mlp.down_proj", 4),
        SiteC("h.1.mlp.c_fc", 8),
    )
    with pytest.raises(AssertionError):
        canonical_site_cs((SiteC("h.0.mlp.c_fc", 4), SiteC("h.0.mlp.c_fc", 8)))


def test_expand_wildcard_site_cs():
    expanded = expand_wildcard_site_cs(
        (SiteC("h.*.mlp.c_fc", 8), SiteC("h.*.attn.q_proj", 4), SiteC("h.1.mlp.down_proj", 16)),
        n_layer=2,
    )
    assert expanded == (
        SiteC("h.0.attn.q_proj", 4),
        SiteC("h.0.mlp.c_fc", 8),
        SiteC("h.1.attn.q_proj", 4),
        SiteC("h.1.mlp.c_fc", 8),
        SiteC("h.1.mlp.down_proj", 16),
    )
    with pytest.raises(AssertionError, match="duplicate"):
        expand_wildcard_site_cs((SiteC("h.*.mlp.c_fc", 8), SiteC("h.0.mlp.c_fc", 4)), n_layer=2)
    with pytest.raises(AssertionError, match="unsupported site name"):
        expand_wildcard_site_cs((SiteC("h.*.mlp.gate_proj", 8),), n_layer=2)


def test_site_specs_dims():
    cfg = _tiny_cfg()
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    specs = site_specs(cfg, canonical_site_cs(tuple(SiteC(site_name(2, k), 4) for k in (
        "q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj",
    ))))  # fmt: skip
    dims = {s.name: (s.d_in, s.d_out, s.C) for s in specs}
    assert dims["h.2.attn.q_proj"] == (cfg.n_embd, qd, 4)
    assert dims["h.2.attn.k_proj"] == (cfg.n_embd, kvd, 4)
    assert dims["h.2.attn.o_proj"] == (qd, cfg.n_embd, 4)
    assert dims["h.2.mlp.c_fc"] == (cfg.n_embd, cfg.n_intermediate, 4)
    assert dims["h.2.mlp.down_proj"] == (cfg.n_intermediate, cfg.n_embd, 4)
    with pytest.raises(AssertionError, match="canonical"):
        site_specs(cfg, (SiteC("h.2.mlp.c_fc", 4), SiteC("h.2.attn.q_proj", 4)))


def test_clean_path_and_masked_identity():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _MIXED_SITE_CS)
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    # per-site heterogeneous C is preserved end to end
    assert {s.name: s.C for s in lm.sites} == {s.name: s.C for s in _MIXED_SITE_CS}
    for spec in lm.sites:
        V, U = vu.site(spec.name)
        assert V.shape == (spec.d_in, spec.C) and U.shape == (spec.C, spec.d_out)

    clean = lm.clean_output(tokens)
    assert clean.shape == (b, t, cfg.vocab_size)

    # SPEC S2: a masked forward with NO live sites is the frozen path — bit-identical.
    none_masked = lm.masked_output(vu, tokens, {}, {}, None, (), True, remat=False)
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    # All-live, masks=1, delta=1, route-everywhere reconstructs the frozen path up to
    # decomposition rounding (the V@U + (W − V@U) identity; exact only in exact math).
    names = lm.site_names
    ones_masks = {s.name: jnp.ones((b, t, s.C)) for s in lm.sites}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    full = lm.masked_output(vu, tokens, ones_masks, ones_delta, None, names, True, remat=False)
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    site_in = lm.read_activations(tokens, lm.site_names)
    assert set(site_in) == set(names)
    # q and v read the same post-LN1 residual; down_proj reads the post-GELU acts
    assert jnp.array_equal(site_in["h.2.attn.q_proj"], site_in["h.2.attn.v_proj"])
    assert site_in["h.3.mlp.down_proj"].shape == (b, t, cfg.n_intermediate)
    assert site_in["h.2.mlp.c_fc"].shape == (b, t, cfg.n_embd)

    deltas = lm.weight_deltas(vu)
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    assert deltas["h.2.attn.q_proj"].shape == (qd, cfg.n_embd)
    assert deltas["h.2.attn.v_proj"].shape == (kvd, cfg.n_embd)
    assert deltas["h.2.mlp.c_fc"].shape == (cfg.n_intermediate, cfg.n_embd)
    assert deltas["h.3.mlp.down_proj"].shape == (cfg.n_embd, cfg.n_intermediate)
    assert all(v.dtype == jnp.float32 for v in deltas.values())


@pytest.mark.parametrize("ablated_site", ["h.2.attn.q_proj", "h.2.mlp.c_fc"])
def test_zero_masking_one_site_changes_logits(ablated_site: str):
    """q is live ahead of RoPE/SDPA; c_fc ahead of the GELU — zero-mask + zero-delta on
    either must change the logits."""
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _MIXED_SITE_CS)
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    clean = lm.clean_output(tokens)
    C = {s.name: s.C for s in _MIXED_SITE_CS}[ablated_site]
    ablated = lm.masked_output(
        vu, tokens,
        {ablated_site: jnp.zeros((b, t, C))}, {ablated_site: jnp.zeros((b, t))},
        None, (ablated_site,), True, remat=False,
    )  # fmt: skip
    assert not jnp.allclose(clean, ablated, atol=1e-4), f"ablating {ablated_site} did nothing"


def test_masked_site_outputs_frozen_when_routed_false_or_unmasked():
    """Clean per-site output: routing FALSE everywhere falls onto `site_out`'s frozen
    `x @ W` branch — exactly the target site output. With a single-site decomposition the
    frozen W per site is `site_input @ W.T`, recovered from `weight_deltas` + `V@U`."""
    cfg = _tiny_cfg()
    sites_cs = (SiteC("h.2.attn.q_proj", 8), SiteC("h.2.mlp.c_fc", 12))
    sites = site_specs(cfg, sites_cs)
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    names = lm.site_names
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    site_in = lm.read_activations(tokens, lm.site_names)
    ones_masks = {s.name: jnp.ones((b, t, s.C)) for s in lm.sites}
    zeros_delta = {s: jnp.zeros((b, t)) for s in names}
    false_routes = {s: jnp.zeros((b, t), bool) for s in names}

    clean_outs = lm.masked_site_outputs(
        vu, tokens, ones_masks, zeros_delta, false_routes, names, False
    )
    assert set(clean_outs) == set(names)
    # frozen `x @ W` per site, reconstructed independently from weight_deltas + V@U.
    deltas = lm.weight_deltas(vu)
    for s in names:
        V, U = vu.site(s)
        W = (V.astype(jnp.float32) @ U.astype(jnp.float32)).T + deltas[s]  # (d_out, d_in)
        expected = site_in[s].astype(jnp.float32) @ W.T
        assert jnp.allclose(clean_outs[s].astype(jnp.float32), expected, atol=1e-3), s


@pytest.mark.parametrize("site_name_str", ["h.2.attn.q_proj", "h.2.mlp.c_fc"])
def test_masked_site_outputs_match_hand_computed_masked_linear(site_name_str: str):
    """Masked per-site output equals the hand-computed `((x@V)*m)@U` (+ delta path). One
    site at a time so the masked site input equals the clean `site_inputs` (no upstream
    masked site contaminating the threaded forward)."""
    cfg = _tiny_cfg()
    sites_cs = (SiteC(site_name_str, 8),)
    sites = site_specs(cfg, sites_cs)
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    names = lm.site_names
    s = site_name_str
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    x_in = lm.read_activations(tokens, (s,))[s]
    V, U = vu.site(s)
    mask = jax.random.uniform(jax.random.PRNGKey(7), (b, t, sites_cs[0].C))

    no_delta = lm.masked_site_outputs(
        vu, tokens, {s: mask}, {s: jnp.zeros((b, t))}, None, names, False
    )
    hand = ((x_in @ V) * mask) @ U
    assert jnp.allclose(no_delta[s], hand, atol=1e-4), s

    # delta path: + delta_mask · (x @ Δ), Δ = W − V@U == lm.weight_deltas (fp32 oracle)
    delta_in = lm.weight_deltas(vu)[s]
    delta_mask = jax.random.uniform(jax.random.PRNGKey(9), (b, t))
    with_delta = lm.masked_site_outputs(vu, tokens, {s: mask}, {s: delta_mask}, None, names, True)
    hand_delta = delta_mask[..., None] * (x_in.astype(jnp.float32) @ delta_in.T)
    expected = hand.astype(jnp.float32) + hand_delta
    assert jnp.allclose(with_delta[s].astype(jnp.float32), expected, atol=1e-3), s


def test_o_site_masks_attention_output():
    cfg = _tiny_cfg()
    o_site = "h.2.attn.o_proj"
    sites = site_specs(cfg, (SiteC(o_site, 8),))
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    clean = lm.clean_output(tokens)
    ones = lm.masked_output(
        vu, tokens, {o_site: jnp.ones((b, t, 8))}, {o_site: jnp.ones((b, t))}, None,
        (o_site,), True, remat=False,
    )  # fmt: skip
    assert jnp.allclose(clean, ones, atol=1e-4)
    # o's clean site input is the pre-o_proj attention output, shape (b, t, qd)
    site_in = lm.read_activations(tokens, lm.site_names)
    assert site_in[o_site].shape == (b, t, cfg.n_head * cfg.head_dim)


def test_step_trains_and_has_vpd_signature():
    cfg = _tiny_cfg()
    site_cs = _MIXED_SITE_CS
    seq = 16
    n_warmup = 2
    sites = site_specs(cfg, site_cs)
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_chunkwise_ci_fn(lm, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    src = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), (1, seq), jnp.float32, jax.random.PRNGKey(3)
    )
    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=0.5,
        scope=SCScope(),
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
        ),
        n_warmup_steps=n_warmup,
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
    n_steps = 4
    losses = []
    for i in range(n_steps):
        state, m = step(lm, state, tokens, jax.random.PRNGKey(100 + i))
        losses.append({k: float(v) for k, v in m.items()})

    assert all(jnp.isfinite(jnp.array(list(m.values()))).all() for m in losses)
    assert int(state.step) == n_steps
    # SPEC S13: n_warmup + 1 source-Adam updates per training step, moments persist.
    ppgd_adv = state.adversaries["PersistentPGDReconLoss"]
    assert float(ppgd_adv.opt_state.step_count) == n_steps * (n_warmup + 1)
    # SPEC S15: sources stay projected to [0,1].
    for v in ppgd_adv.sources.values():
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    # SPEC S9: p annealed below its 2.0 start by step 4 of 100.
    assert losses[-1]["p_imp"] < 2.0
    # fp32 masters preserved through updates (SPEC N1).
    assert isinstance(state.components, DecompVU)
    for V, U in state.components.vu.values():
        assert V.dtype == jnp.float32 and U.dtype == jnp.float32
    assert isinstance(state.ci_fn, ChunkwiseTransformerCIFn)
    assert state.ci_fn.chunks.in_proj_w.dtype == jnp.float32


def _copied[T](tree: T) -> T:
    """Fresh buffers per step call — the jitted steps donate their state/batch args."""
    return jax.tree.map(lambda x: jnp.copy(x) if eqx.is_array(x) else x, tree)


def _stale_ci_harness(opt_ci: optax.GradientTransformation, ci_fn_lr: float):
    """Shared tiny-model setup for the stale-CI mode tests (SPEC S34).

    `opt_ci` MUST carry unit LR: the stale-CI factory's ci_fn-updating steps apply the
    constant `ci_fn_lr` schedule in-step at the global step. Returns
    `(lm, ci_fn, tokens, make_state, make_stale_steps, make_plain_step)` — `make_state`
    mints a donation-safe fresh `TrainState`; the two factories share every other knob so
    a plain step built with `plain_opt_ci` (LR baked into the optimizer) is the stale
    bundle's semantic twin."""
    cfg = _tiny_cfg()
    seq = 16
    sites = site_specs(cfg, _MIXED_SITE_CS)
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_chunkwise_ci_fn(lm, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))

    src = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), (1, seq), jnp.float32, jax.random.PRNGKey(3)
    )
    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=0.5,
        scope=SCScope(),
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
        ),
        n_warmup_steps=0,
    )
    ppgd_coeff = ppgd_cfg.coeff
    assert ppgd_coeff is not None

    def make_state() -> TrainState:
        vu_c, ci_fn_c, src_c = _copied(vu), _copied(ci_fn), _copied(src)
        return TrainState(
            components=vu_c, ci_fn=ci_fn_c,
            components_opt_state=opt_vu.init(eqx.filter(vu_c, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn_c, eqx.is_array)),
            adversaries={
                ppgd_cfg.type: PersistentAdversary(
                    sources=src_c,
                    opt_state=init_sources_adam_state(src_c),
                    state_key=ppgd_cfg.type,
                    coeff=ppgd_coeff,
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

    def make_stale_steps():
        return make_stale_ci_train_steps(
            lm=lm,
            losses=loss_terms,
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            ci_fn_lr_schedule=ScheduleConfig(start_val=ci_fn_lr),
            total_steps=100,
            remat_recon_forwards=True,
            remat_ci_fn=False,
            mesh=None,
        )

    def make_plain_step(plain_opt_ci: optax.GradientTransformation):
        return make_train_step(
            lm=lm,
            losses=loss_terms,
            components_optimizer=opt_vu,
            ci_fn_optimizer=plain_opt_ci,
            total_steps=100,
            remat_recon_forwards=True,
            remat_ci_fn=False,
            mesh=None,
        )

    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    return lm, ci_fn, tokens, make_state, make_stale_steps, make_plain_step


def _assert_ci_fn_untouched(state: TrainState, ci_fn: ChunkwiseTransformerCIFn) -> None:
    for got, want in zip(
        jax.tree.leaves(eqx.filter(state.ci_fn, eqx.is_array)),
        jax.tree.leaves(eqx.filter(ci_fn, eqx.is_array)),
        strict=True,
    ):
        assert (got == want).all()


def test_stale_ci_repeat_step_semantics():
    """SPEC S34: a repeat step given the fresh step's envelope trains the components and
    ascends the adversary like the fresh step (CI live vs constant doesn't change their
    gradients), while leaving the ci_fn and its optimizer state untouched."""
    # Unit-lr ci_fn optimizer: the stale-CI fresh step applies the schedule at the global step.
    lm, ci_fn, tokens, make_state, make_stale_steps, _ = _stale_ci_harness(
        optax.adamw(1.0, weight_decay=0.0), ci_fn_lr=1e-3
    )
    steps = make_stale_steps()
    key = jax.random.PRNGKey(100)

    envelope = steps.compute_ci(lm, ci_fn, tokens)
    state_fresh, m_fresh, env_fresh = steps.fresh(lm, make_state(), jnp.copy(tokens), jnp.copy(key))
    state_rep, m_rep = steps.repeat((lm, envelope), make_state(), jnp.copy(tokens), jnp.copy(key))

    # compute_ci reproduces the fresh step's own envelope (same ci_fn, same batch).
    for site in envelope.lower:
        assert jnp.allclose(envelope.lower[site], env_fresh.lower[site], atol=1e-6)
        assert jnp.allclose(envelope.upper[site], env_fresh.upper[site], atol=1e-6)

    # The repeat step leaves the ci_fn and its optimizer state bit-untouched...
    _assert_ci_fn_untouched(state_rep, ci_fn)
    # ...while the fresh step trained it.
    fresh_ci_leaves = jax.tree.leaves(eqx.filter(state_fresh.ci_fn, eqx.is_array))
    init_ci_leaves = jax.tree.leaves(eqx.filter(ci_fn, eqx.is_array))
    assert any((a != b).any() for a, b in zip(fresh_ci_leaves, init_ci_leaves, strict=True))

    # Components + adversary see the same gradients either way (CI enters their backward
    # as the same values, live or constant) — updates match to numeric tolerance.
    for (fv, fu), (rv, ru) in zip(
        state_fresh.components.vu.values(), state_rep.components.vu.values(), strict=True
    ):
        assert jnp.allclose(fv, rv, rtol=1e-4, atol=1e-6)
        assert jnp.allclose(fu, ru, rtol=1e-4, atol=1e-6)
    adv_f = state_fresh.adversaries["PersistentPGDReconLoss"]
    adv_r = state_rep.adversaries["PersistentPGDReconLoss"]
    for site in adv_f.sources:
        assert jnp.allclose(adv_f.sources[site], adv_r.sources[site], rtol=1e-4, atol=1e-6)

    # Metric families: the repeat step carries no ci_fn grad norms; totals stay finite.
    assert "grad_norms/summary/ci_fns" in m_fresh and "grad_norms/summary/ci_fns" not in m_rep
    assert jnp.isfinite(m_rep["total"]) and jnp.isfinite(m_fresh["total"])
    assert int(state_rep.step) == 1


def test_stale_ci_last_mode_semantics():
    """SPEC S34 mode `last` over a full window: the components + adversary trajectory
    matches mode `first`'s (both windows run on the same envelope — taps and ci_fn params
    are window-constant), the ci_fn is untouched before the window-last step, and the
    window-last ci_fn update equals what a plain `make_train_step` step applies at the
    same pre-step state."""
    lm, ci_fn, tokens, make_state, make_stale_steps, make_plain_step = _stale_ci_harness(
        optax.adamw(1.0, weight_decay=0.0), ci_fn_lr=1e-3
    )
    steps = make_stale_steps()
    window = 3
    keys = [jax.random.PRNGKey(100 + i) for i in range(window)]

    # Mode "first": fresh at t=0, repeats after (the old replay_stale_ci: true).
    first_traj: list[TrainState] = []
    state = make_state()
    state, _, envelope_first = steps.fresh(lm, state, jnp.copy(tokens), jnp.copy(keys[0]))
    first_traj.append(state)
    for t in range(1, window):
        state, _ = steps.repeat(
            (lm, envelope_first), _copied(first_traj[-1]), jnp.copy(tokens), jnp.copy(keys[t])
        )
        first_traj.append(state)

    # Mode "last": compute_ci at t=0, repeats through t=k-2, fresh at t=k-1.
    envelope = steps.compute_ci(lm, ci_fn, tokens)
    for site in envelope.lower:
        assert jnp.allclose(envelope.lower[site], envelope_first.lower[site], atol=1e-6)
        assert jnp.allclose(envelope.upper[site], envelope_first.upper[site], atol=1e-6)
    last_traj: list[TrainState] = []
    state = make_state()
    for t in range(window - 1):
        state, metrics = steps.repeat(
            (lm, envelope), _copied(state) if t else state, jnp.copy(tokens), jnp.copy(keys[t])
        )
        last_traj.append(state)
        assert "grad_norms/summary/ci_fns" not in metrics
        _assert_ci_fn_untouched(state, ci_fn)  # ci_fn + opt state ride through untouched
    pre_last = last_traj[-1]
    state, metrics, _ = steps.fresh(lm, _copied(pre_last), jnp.copy(tokens), jnp.copy(keys[-1]))
    last_traj.append(state)
    assert "grad_norms/summary/ci_fns" in metrics

    # V/U + adversary see identical gradients under both modes at every window step (the
    # envelope is the same constant; the ci_fn divergence never enters their backward).
    for s_first, s_last in zip(first_traj, last_traj, strict=True):
        for (fv, fu), (lv, lu) in zip(
            s_first.components.vu.values(), s_last.components.vu.values(), strict=True
        ):
            assert jnp.allclose(fv, lv, rtol=1e-4, atol=1e-6)
            assert jnp.allclose(fu, lu, rtol=1e-4, atol=1e-6)
        adv_f = s_first.adversaries["PersistentPGDReconLoss"]
        adv_l = s_last.adversaries["PersistentPGDReconLoss"]
        for site in adv_f.sources:
            assert jnp.allclose(adv_f.sources[site], adv_l.sources[site], rtol=1e-4, atol=1e-6)

    # The window-last ci_fn update == a plain fresh step's at the same pre-step state
    # (unit-LR Adam + in-step constant 1e-3 schedule vs LR-1e-3 Adam are the same update).
    plain_step = make_plain_step(optax.adamw(1e-3, weight_decay=0.0))
    plain_state, _ = plain_step(lm, _copied(pre_last), jnp.copy(tokens), jnp.copy(keys[-1]))
    for got, want in zip(
        jax.tree.leaves(eqx.filter(last_traj[-1].ci_fn, eqx.is_array)),
        jax.tree.leaves(eqx.filter(plain_state.ci_fn, eqx.is_array)),
        strict=True,
    ):
        assert jnp.allclose(got, want, rtol=1e-4, atol=1e-7)
    # And it actually trained the ci_fn.
    assert any(
        (a != b).any()
        for a, b in zip(
            jax.tree.leaves(eqx.filter(last_traj[-1].ci_fn, eqx.is_array)),
            jax.tree.leaves(eqx.filter(ci_fn, eqx.is_array)),
            strict=True,
        )
    )


def test_stale_ci_mean_mode_semantics():
    """SPEC S34 mode `mean`: the window-last pulled-back ci_fn gradient equals the mean of
    the k per-step ci_fn gradients computed independently (the fresh step's grad at each
    window state, ci_fn params held at window-start — which they are: mean-mode steps never
    touch the ci_fn mid-window). Uses unit-LR SGD so ci_fn deltas ARE (scheduled-LR-scaled)
    gradients and the mean commutes with the update. The large in-step LR keeps the
    SGD deltas well above the fp32 parameter ulp, so delta extraction is meaningful."""
    lm, ci_fn, tokens, make_state, make_stale_steps, _ = _stale_ci_harness(
        optax.sgd(1.0), ci_fn_lr=10.0
    )
    steps = make_stale_steps()
    window = 3
    keys = [jax.random.PRNGKey(100 + i) for i in range(window)]

    def ci_leaves(state: TrainState) -> list[jax.Array]:
        return jax.tree.leaves(eqx.filter(state.ci_fn, eqx.is_array))

    envelope = steps.compute_ci(lm, ci_fn, tokens)
    traj: list[TrainState] = [make_state()]
    cotangents = []
    for t in range(window - 1):
        state, metrics, cotangent = steps.mean_accum(
            (lm, envelope), _copied(traj[-1]), jnp.copy(tokens), jnp.copy(keys[t])
        )
        traj.append(state)
        cotangents.append(cotangent)
        assert "grad_norms/summary/ci_fns" not in metrics
        _assert_ci_fn_untouched(state, ci_fn)  # untouched mid-window
        assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(cotangent))
    acc = jax.tree.map(jnp.add, *cotangents)
    final_state, metrics = steps.mean_update(
        (lm, envelope),
        _copied(traj[-1]),
        jnp.copy(tokens),
        jnp.copy(keys[-1]),
        acc,
        jnp.asarray(1.0 / window, jnp.float32),
    )
    assert "grad_norms/summary/ci_fns" in metrics

    # Reference: the fresh step's ci_fn grad at each window state. With unit-LR SGD scaled
    # in-step by the constant schedule, new_ci_fn - ci_fn = -lr * grad, so the mean of the
    # per-step deltas is the delta the mean gradient applies. Compared per leaf in max-norm
    # (diff bounded relative to the leaf's LARGEST entries): the mean cotangent is rounded
    # to bf16 before the one pullback, while each reference delta pulls back its own bf16
    # cotangent, so small entries carry rounding noise from the leaf's big ones.
    per_step_deltas = []
    for t in range(window):
        fresh_state, _, _ = steps.fresh(lm, _copied(traj[t]), jnp.copy(tokens), jnp.copy(keys[t]))
        per_step_deltas.append(
            [f - i for f, i in zip(ci_leaves(fresh_state), ci_leaves(traj[t]), strict=True)]
        )
    mean_deltas = [
        sum(step_deltas[i] for step_deltas in per_step_deltas) / window
        for i in range(len(per_step_deltas[0]))
    ]
    got_deltas = [f - i for f, i in zip(ci_leaves(final_state), ci_leaves(traj[0]), strict=True)]
    for got, want in zip(got_deltas, mean_deltas, strict=True):
        max_diff = float(jnp.abs(got - want).max())
        assert max_diff <= 1e-2 * float(jnp.abs(want).max()) + 1e-8, (
            max_diff,
            float(jnp.abs(want).max()),
        )
    # The tolerance has teeth: any SINGLE step's gradient deviates from the mean beyond it
    # (the components/adversary move each step), so only the true mean passes.
    for single_deltas in (per_step_deltas[0], per_step_deltas[-1]):
        assert any(
            float(jnp.abs(single - want).max()) > 1e-2 * float(jnp.abs(want).max()) + 1e-8
            for single, want in zip(single_deltas, mean_deltas, strict=True)
        )

    # The mean-accum steps train components/adversary exactly like constant-envelope
    # repeats (tracing the envelope only adds a cotangent output).
    repeat_state, _ = steps.repeat(
        (lm, envelope), make_state(), jnp.copy(tokens), jnp.copy(keys[0])
    )
    for (mv, mu), (rv, ru) in zip(
        traj[1].components.vu.values(), repeat_state.components.vu.values(), strict=True
    ):
        assert jnp.allclose(mv, rv, rtol=1e-4, atol=1e-6)
        assert jnp.allclose(mu, ru, rtol=1e-4, atol=1e-6)


def test_faith_warmup_decreases_faith():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, canonical_site_cs(_MIXED_SITE_CS))
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    opt = optax.adamw(1e-2, weight_decay=0.0)
    wstep = make_faith_warmup_step(opt)
    ostate = opt.init(eqx.filter(vu, eqx.is_array))
    first_loss: float | None = None
    loss = None
    for _ in range(30):
        vu, ostate, loss = wstep(lm, vu, ostate)
        first_loss = float(loss) if first_loss is None else first_loss
    assert first_loss is not None and loss is not None
    assert float(loss) < first_loss * 0.9, (first_loss, float(loss))


def test_decomp_vu_shapes_fp32():
    cfg = _tiny_cfg()
    sites = site_specs(cfg, _MIXED_SITE_CS)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    V_q, U_q = vu.site("h.2.attn.q_proj")
    V_v, U_v = vu.site("h.2.attn.v_proj")
    V_fc, U_fc = vu.site("h.2.mlp.c_fc")
    V_dn, U_dn = vu.site("h.3.mlp.down_proj")
    assert V_q.shape == (d, 8) and U_q.shape == (8, qd)
    assert V_v.shape == (d, 12) and U_v.shape == (12, kvd)
    assert V_fc.shape == (d, 8) and U_fc.shape == (8, di)
    assert V_dn.shape == (di, 16) and U_dn.shape == (16, d)
    assert all(a.dtype == jnp.float32 for pair in vu.vu.values() for a in pair)


_REAL_CACHE_DIR = Path("/mnt/data/artifacts/mechanisms/param-decomp/pretrain_cache/spd-t-9d2b8f02")
_PRODUCTION_PATTERN_CS = {
    "h.*.mlp.c_fc": 3072,
    "h.*.mlp.down_proj": 3584,
    "h.*.attn.q_proj": 512,
    "h.*.attn.k_proj": 512,
    "h.*.attn.v_proj": 1024,
    "h.*.attn.o_proj": 1024,
}
"""The pile production decomposition (torch `pile_llama_simple_mlp-4L.yaml`)."""


@pytest.mark.skipif(not _REAL_CACHE_DIR.exists(), reason="t-9d2b8f02 pretrain cache not mounted")
def test_pretrained_target_converts_with_wildcards():
    """`kind: pretrained` LlamaSimpleMLP target specs convert, expanding `h.*`
    wildcard decomposition patterns over the checkpoint's n_layer (4)."""
    import yaml

    from param_decomp.built_run import DataConfig
    from param_decomp_lab.experiments.lm.config import (
        LlamaSimpleMLPTargetConfig,
        LMExperimentConfig,
        build_experiment_config,
    )

    reference_yaml = Path(__file__).parent.parent / "configs" / "llama8b_l18_b128_cmp32.yaml"
    raw = yaml.safe_load(reference_yaml.read_text())
    raw["target"]["spec"] = {
        "kind": "pretrained",
        "model_class": (
            "param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP"
        ),
        "run_path": "goodfire/spd/runs/t-9d2b8f02",
    }
    raw["pd"]["decomposition_targets"] = [
        {"module_pattern": pattern, "C": C} for pattern, C in _PRODUCTION_PATTERN_CS.items()
    ]
    raw["data"]["max_seq_len"] = 512  # the model's n_ctx

    cfg = build_experiment_config(LMExperimentConfig(**raw), "p-00000000")
    target = cfg.target
    assert isinstance(target, LlamaSimpleMLPTargetConfig)
    assert target.pretrain_run_path == "goodfire/spd/runs/t-9d2b8f02"
    assert len(target.sites) == 4 * 6
    assert target.sites == canonical_site_cs(target.sites)
    by_name = {sc.name: sc.C for sc in target.sites}
    for layer in range(4):
        assert by_name[f"h.{layer}.mlp.c_fc"] == 3072
        assert by_name[f"h.{layer}.mlp.down_proj"] == 3584
        assert by_name[f"h.{layer}.attn.q_proj"] == 512
        assert by_name[f"h.{layer}.attn.v_proj"] == 1024
    assert target.sites[0] == SiteC("h.0.attn.q_proj", 512)
    # StochasticReconSubsetLoss = one all-sites entry
    loss_terms = build_loss_terms(
        cfg.pd.loss_metrics,
        tuple(sc.name for sc in target.sites),
    )
    (stoch_term,) = [t for t in loss_terms.recon if t.name == "StochasticReconSubsetLoss"]
    (stoch_entry,) = stoch_term.plan
    assert len(stoch_entry.live_sites) == 24
    assert isinstance(cfg.data, DataConfig)
    assert cfg.data.seq_len == 512
