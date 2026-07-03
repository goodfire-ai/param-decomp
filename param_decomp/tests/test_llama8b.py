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
    CIMaskedReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    SCScope,
    UniformKSubsetRoutingConfig,
)
from param_decomp.lm import DecomposedModel
from param_decomp.recon import build_loss_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.targets.llama8b import (
    FrozenAttn,
    LlamaDecomposedModel,
    LlamaLayer,
    build_decomposed_lm,
    canonical_site_cs,
    llama_site_specs,
    mlp_family_site_cs,
    parse_site_name,
    site_name,
)
from param_decomp.train import TrainState, make_faith_warmup_step, make_train_step
from vendored_jax.llama import LlamaConfig, llama3_inv_freq


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


def _tiny_decomposed_lm(
    cfg: LlamaConfig,
    sites: tuple[SiteSpec, ...],
    key: jax.Array,
    save_gathered_weights: bool = False,
) -> LlamaDecomposedModel:
    """A tiny random `LlamaDecomposedModel` (random embedding + full frozen layer stack
    plus the decomposition `sites`) — the CPU-test analog of `load_decomposed_lm_from_hf`."""
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

    def layer():
        return LlamaLayer(
            jnp.ones((d,)), jnp.ones((d,)), fattn(), n((di, d)), n((di, d)), n((d, di))
        )

    return build_decomposed_lm(
        embed=n((cfg.vocab_size, d), 0.02),
        layers=[layer() for _ in range(cfg.n_layer)],
        norm=jnp.ones((d,)),
        lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=llama3_inv_freq(cfg),
        cfg=cfg,
        sites=sites,
        save_gathered_weights=save_gathered_weights,
    )


def _mlp_sites(cfg: LlamaConfig, first: int, last: int, C: int) -> tuple[SiteSpec, ...]:
    return llama_site_specs(cfg, mlp_family_site_cs(first, last, C))


_QVDOWN_SITE_CS = (
    SiteC("layers.4.self_attn.q_proj", 8),
    SiteC("layers.4.self_attn.v_proj", 12),
    SiteC("layers.4.mlp.down_proj", 8),
)
"""Attention + MLP sites on one layer with heterogeneous per-site C."""


def _build_chunkwise_ci_fn(
    lm: DecomposedModel, key: jax.Array, n_blocks: int
) -> ChunkwiseTransformerCIFn:
    """Old `init_ci_fn(CIArch(16, n_blocks, 2, 32), lm.sites, key)` → the new chunkwise
    builder: a single chunk reading the residual entering the first decomposed block and
    emitting CI for every site. `input_dim` is the target residual width (`n_embd`)."""
    site_names = lm.site_names
    first_block = min(parse_site_name(n)[0] for n in site_names)
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=_tiny_cfg().n_embd,
        d_model=16,
        n_blocks=n_blocks,
        n_heads=2,
        mlp_hidden=32,
    )
    ci_fn = build_ci_fn(arch, lm.sites, key)
    assert isinstance(ci_fn, ChunkwiseTransformerCIFn)
    return ci_fn


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


@pytest.mark.parametrize("remat", [True, False])
def test_save_gathered_weights_is_numerics_identical(remat: bool):
    """`save_gathered_weights` is a pure store-vs-recompute trade: outputs AND grads (wrt
    V/U and masks, through the checkpointed live-block scan) must match the default path."""
    cfg = _tiny_cfg()
    C = 8
    sites = _mlp_sites(cfg, 3, 5, C)
    key = jax.random.PRNGKey(0)
    lm_off = _tiny_decomposed_lm(cfg, sites, key)
    lm_on = _tiny_decomposed_lm(cfg, sites, key, save_gathered_weights=True)
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)
    names = lm_off.site_names
    masks = {
        s: jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(3), i), (b, t, C))
        for i, s in enumerate(names)
    }

    def loss(lm: LlamaDecomposedModel, vu_: DecompVU, masks_: dict[str, jax.Array]) -> jax.Array:
        out = lm.masked_output(
            lm.prepare_compute_weights(vu_), tokens, masks_, {}, None, names, False, remat=remat
        )
        return jnp.sum(out.astype(jnp.float32) ** 2)

    (l_off, g_off) = jax.value_and_grad(lambda v, m: loss(lm_off, v, m), argnums=(0, 1))(vu, masks)
    (l_on, g_on) = jax.value_and_grad(lambda v, m: loss(lm_on, v, m), argnums=(0, 1))(vu, masks)
    assert jnp.array_equal(l_off, l_on), (l_off, l_on)
    for leaf_off, leaf_on in zip(jax.tree.leaves(g_off), jax.tree.leaves(g_on), strict=True):
        assert jnp.array_equal(leaf_off, leaf_on)


@pytest.mark.parametrize("first,last", [(4, 4), (3, 6)])
def test_clean_path_and_masked_identity(first: int, last: int):
    cfg = _tiny_cfg()
    C = 8
    sites = _mlp_sites(cfg, first, last, C)
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    clean = lm.clean_output(tokens)
    assert clean.shape == (b, t, cfg.vocab_size)

    # SPEC S2: a masked forward with NO live sites is the frozen path — bit-identical
    # to the clean target.
    none_masked = lm.masked_output(
        lm.prepare_compute_weights(vu), tokens, {}, {}, None, (), True, remat=False
    )
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    # All-live, masks=1, delta=1, route-everywhere reconstructs the frozen path up to
    # decomposition rounding (the V@U + (W − V@U) identity; exact only in exact math).
    names = lm.site_names
    ones_masks = {s: jnp.ones((b, t, C)) for s in names}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    full = lm.masked_output(
        lm.prepare_compute_weights(vu),
        tokens,
        ones_masks,
        ones_delta,
        None,
        names,
        True,
        remat=False,
    )
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    site_in = lm.read_activations(tokens, lm.site_names)
    assert set(site_in) == set(names)
    deltas = lm.weight_deltas(vu)
    d, di = cfg.n_embd, cfg.n_intermediate
    assert deltas[names[0]].shape == (di, d)  # gate: (d_out, d_in)
    assert deltas[names[2]].shape == (d, di)  # down
    assert all(v.dtype == jnp.float32 for v in deltas.values())


def test_attention_sites_clean_and_masked_identity():
    cfg = _tiny_cfg()
    sites = llama_site_specs(cfg, _QVDOWN_SITE_CS)
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    # per-site heterogeneous C is preserved end to end
    assert {s.name: s.C for s in lm.sites} == {sc.name: sc.C for sc in _QVDOWN_SITE_CS}
    for spec in lm.sites:
        V, U = vu.site(spec.name)
        assert V.shape == (spec.d_in, spec.C) and U.shape == (spec.C, spec.d_out)

    clean = lm.clean_output(tokens)
    none_masked = lm.masked_output(
        lm.prepare_compute_weights(vu), tokens, {}, {}, None, (), True, remat=False
    )
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    names = lm.site_names
    ones_masks = {s.name: jnp.ones((b, t, s.C)) for s in lm.sites}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    full = lm.masked_output(
        lm.prepare_compute_weights(vu),
        tokens,
        ones_masks,
        ones_delta,
        None,
        names,
        True,
        remat=False,
    )
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted (attention sites)"

    # zero-mask + zero-delta on layer 4's decomposed sites must CHANGE the logits (q is live on
    # the attention path ahead of RoPE/SDPA). The segmented masked forward masks WHOLE layers, so
    # we ablate layer 4's full decomposed set rather than q alone.
    q_site = "layers.4.self_attn.q_proj"
    site_c = {s.name: s.C for s in lm.sites}
    layer4 = tuple(n for n in names if parse_site_name(n)[0] == 4)
    assert q_site in layer4
    zero_mask = {n: jnp.zeros((b, t, site_c[n])) for n in layer4}
    zero_delta = {n: jnp.zeros((b, t)) for n in layer4}
    ablated = lm.masked_output(
        lm.prepare_compute_weights(vu),
        tokens,
        zero_mask,
        zero_delta,
        None,
        layer4,
        True,
        remat=False,
    )
    assert not jnp.allclose(clean, ablated, atol=1e-4), "ablating layer 4 did nothing"

    site_in = lm.read_activations(tokens, lm.site_names)
    assert set(site_in) == set(names)
    # q and v read the same post-LN1 residual; down reads the (di,) MLP inner acts
    assert jnp.array_equal(site_in[q_site], site_in["layers.4.self_attn.v_proj"])
    assert site_in["layers.4.mlp.down_proj"].shape == (b, t, cfg.n_intermediate)

    deltas = lm.weight_deltas(vu)
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    assert deltas[q_site].shape == (qd, cfg.n_embd)
    assert deltas["layers.4.self_attn.v_proj"].shape == (kvd, cfg.n_embd)


def test_o_site_masks_attention_output():
    cfg = _tiny_cfg()
    o_site = "layers.4.self_attn.o_proj"
    sites = llama_site_specs(cfg, (SiteC(o_site, 8),))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    clean = lm.clean_output(tokens)
    ones = lm.masked_output(
        lm.prepare_compute_weights(vu),
        tokens,
        {o_site: jnp.ones((b, t, 8))},
        {o_site: jnp.ones((b, t))},
        None,
        (o_site,),
        True,
        remat=False,
    )
    assert jnp.allclose(clean, ones, atol=1e-4)
    # o's clean site input is the pre-o_proj attention output, shape (b, t, qd)
    site_in = lm.read_activations(tokens, lm.site_names)
    assert site_in[o_site].shape == (b, t, cfg.n_head * cfg.head_dim)


@pytest.mark.parametrize(
    "site_cs",
    [mlp_family_site_cs(4, 4, 8), mlp_family_site_cs(3, 6, 8), _QVDOWN_SITE_CS],
    ids=["mlp_l4", "mlp_l3_6", "qv_down_l4"],
)
def test_step_trains_and_has_vpd_signature(site_cs: tuple[SiteC, ...]):
    cfg = _tiny_cfg()
    seq = 16
    n_warmup = 2
    sites = llama_site_specs(cfg, site_cs)
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_chunkwise_ci_fn(lm, jax.random.PRNGKey(2), n_blocks=2)
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
                routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=3, n_samples=1
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


def test_faith_warmup_decreases_faith():
    cfg = _tiny_cfg()
    sites = _mlp_sites(cfg, 3, 4, 8)
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
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


@pytest.mark.parametrize("remat", [True, False])
def test_sequence_recon_entries_matches_fused_backward(remat: bool):
    """`sequence_recon_entries` only reorders scheduling: same forwards, same RNG, same
    losses; grads differ only by float reassociation in the shared-leaf accumulation.
    One full step per arm from bit-identical states, over the production term shapes
    (chunkwise-stochastic multi-entry + PPGD all-sites) plus a constant-sources term."""
    cfg = _tiny_cfg()
    seq = 16
    site_cs = mlp_family_site_cs(2, 5, 8)
    sites = llama_site_specs(cfg, site_cs)
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=0.5,
        scope=SCScope(),
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
        ),
        n_warmup_steps=2,
    )
    assert ppgd_cfg.coeff is not None
    loss_terms = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
            ),
            ChunkwiseSubsetReconLossConfig(
                routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=3, n_samples=2
            ),
            CIMaskedReconLossConfig(coeff=0.25),
            ppgd_cfg,
        ),
        lm.site_names,
    )

    def make_state() -> TrainState:
        # Fresh buffers per arm: `step` donates the state. Deterministic keys keep the two
        # arms' inits bit-identical.
        vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
        ci_fn = _build_chunkwise_ci_fn(lm, jax.random.PRNGKey(2), n_blocks=2)
        src = init_persistent_sources(
            lm.site_names,
            tuple(s.C for s in lm.sites),
            (1, seq),
            jnp.float32,
            jax.random.PRNGKey(3),
        )
        assert ppgd_cfg.coeff is not None
        return TrainState(
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

    def run_arm(sequence_recon_entries: bool) -> tuple[TrainState, dict[str, jax.Array]]:
        step = make_train_step(
            lm=lm,
            losses=loss_terms,
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=100,
            remat_recon_forwards=remat,
            remat_ci_fn=False,
            mesh=None,
            sequence_recon_entries=sequence_recon_entries,
        )
        tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
        return step(lm, make_state(), tokens, jax.random.PRNGKey(100))

    state_fused, metrics_fused = run_arm(False)
    state_seq, metrics_seq = run_arm(True)

    # Same forwards + RNG → the loss VALUES are identical (only the backward is restructured).
    assert set(metrics_fused) == set(metrics_seq)
    for k in ("total", "faith", "imp", "freq") + tuple(
        k for k in metrics_fused if k.startswith("loss/")
    ):
        assert jnp.array_equal(metrics_fused[k], metrics_seq[k]), (
            k, metrics_fused[k], metrics_seq[k],
        )  # fmt: skip

    # Grads (→ updated state) match up to reassociation of the per-forward accumulation.
    def leaves(tree: object) -> list[jax.Array]:
        return [leaf for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf)]

    for name, fused, seq_ in (
        ("components", state_fused.components, state_seq.components),
        ("ci_fn", state_fused.ci_fn, state_seq.ci_fn),
        ("adversaries", state_fused.adversaries, state_seq.adversaries),
    ):
        for leaf_fused, leaf_seq in zip(leaves(fused), leaves(seq_), strict=True):
            assert jnp.allclose(
                leaf_fused.astype(jnp.float32), leaf_seq.astype(jnp.float32), atol=1e-6, rtol=1e-5
            ), (name, jnp.abs(leaf_fused - leaf_seq).max())


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
    seq = 16
    sites = llama_site_specs(cfg, site_cs)
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    def make_state() -> TrainState:
        # Fresh buffers per call: `step` donates the state, so a shared vu/ci_fn would be
        # deleted after the first run_step and crash the second. Deterministic keys keep
        # the two states' inits bit-identical (the "same init" the comparison below needs).
        vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
        ci_fn = _build_chunkwise_ci_fn(lm, jax.random.PRNGKey(2), n_blocks=1)
        return TrainState(
            components=vu, ci_fn=ci_fn,
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            step=jnp.zeros((), jnp.int32),
        )  # fmt: skip

    def run_step(n_ascent_steps: int) -> tuple[TrainState, dict[str, jax.Array]]:
        loss_terms = build_loss_terms(
            (
                FaithfulnessLossConfig(coeff=1e7),
                ImportanceMinimalityLossConfig(
                    coeff=2e-4,
                    pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
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
        )
        step = make_train_step(
            lm=lm,
            losses=loss_terms,
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=100,
            remat_recon_forwards=False,
            remat_ci_fn=False,
            mesh=None,
        )
        tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
        return step(lm, make_state(), tokens, jax.random.PRNGKey(100))

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
    assert state.adversaries == {}, "fresh adversary carries no persistent sources"
    assert int(state.step) == 1

    _, metrics_unascended = run_step(n_ascent_steps=0)
    assert float(metrics["loss/PGDReconLoss"]) >= float(metrics_unascended["loss/PGDReconLoss"]), (
        "one sign step from the same init must not weaken the adversary"
    )
