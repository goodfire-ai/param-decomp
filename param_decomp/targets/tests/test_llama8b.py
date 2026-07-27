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

from param_decomp.core.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    MHACIAttention,
)
from param_decomp.core.components import ComponentStacks, SiteC, SiteSpec, init_component_stacks
from param_decomp.core.configs import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.recon import build_loss_terms
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    TrainingItem,
    TrainState,
    make_faith_warmup_step,
    make_train_step,
)
from param_decomp.targets.glu_transformer import (
    canonical_site_cs,
    glu_site_specs,
    mlp_family_site_cs,
    parse_site_name,
    site_name,
)
from param_decomp.targets.testing import (
    tiny_glu_cfg,
    tiny_glu_chunkwise_ci_fn,
    tiny_glu_decomposed_lm,
)
from param_decomp.vendored_jax.llama import LlamaConfig


def _mlp_sites(cfg: LlamaConfig, first: int, last: int, C: int) -> tuple[SiteSpec, ...]:
    return glu_site_specs(cfg, mlp_family_site_cs(first, last, C))


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


def test_llama_site_specs_dims():
    cfg = tiny_glu_cfg()
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    specs = glu_site_specs(
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
        glu_site_specs(cfg, tuple(reversed(mlp_family_site_cs(2, 2, 4))))


def test_masked_component_activations_pre_mask_and_matches_outputs():
    cfg = tiny_glu_cfg()
    C = 8
    sites = _mlp_sites(cfg, 4, 5, C)
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)
    names = model.site_names
    prepared = model.prepare_compute_weights(vu)
    zeros_delta = {s: jnp.zeros((b, t)) for s in names}
    ones = {s: jnp.ones((b, t, C)) for s in names}

    acts = model.masked_component_activations(
        prepared, tokens, ones, zeros_delta, None, names, False
    )
    # `acts[s]` is `x@V` BEFORE the per-component `*mask` (shape (b, t, C)). With all-ones
    # masks and no delta the site OUTPUT is exactly `(x@V) @ U`, so projecting the collected
    # activation through U reproduces `masked_site_outputs` — pinning that we collected the
    # pre-mask coefficient, not the post-mask/post-U output.
    outputs = model.masked_site_outputs(prepared, tokens, ones, zeros_delta, None, names, False)
    for s in names:
        assert acts[s].shape == (b, t, C)
        assert jnp.all(jnp.isfinite(acts[s]))
        _, u = vu.site(s)
        expected = acts[s].astype(jnp.float32) @ u.astype(jnp.float32)
        assert jnp.allclose(outputs[s].astype(jnp.float32), expected, atol=1e-2), s


@pytest.mark.parametrize("first,last", [(4, 4), (3, 6)])
def test_clean_path_and_masked_identity(first: int, last: int):
    cfg = tiny_glu_cfg()
    C = 8
    sites = _mlp_sites(cfg, first, last, C)
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    clean = model.clean_output(tokens)
    assert clean.shape == (b, t, cfg.vocab_size)

    # SPEC S2: a masked forward with NO live sites is the frozen path — bit-identical
    # to the clean target.
    none_masked = model.masked_output(
        model.prepare_compute_weights(vu), tokens, {}, {}, None, (), True, remat=False
    )
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    # All-live, masks=1, delta=1, route-everywhere reconstructs the frozen path up to
    # decomposition rounding (the V@U + (W − V@U) identity; exact only in exact math).
    names = model.site_names
    ones_masks = {s: jnp.ones((b, t, C)) for s in names}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    full = model.masked_output(
        model.prepare_compute_weights(vu),
        tokens,
        ones_masks,
        ones_delta,
        None,
        names,
        True,
        remat=False,
    )
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    site_in = model.read_activations(tokens, model.site_names)
    assert set(site_in) == set(names)
    deltas = model.weight_deltas(vu)
    d, di = cfg.n_embd, cfg.n_intermediate
    assert deltas[names[0]].shape == (di, d)  # gate: (d_out, d_in)
    assert deltas[names[2]].shape == (d, di)  # down
    assert all(v.dtype == jnp.float32 for v in deltas.values())


def test_attention_sites_clean_and_masked_identity():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, _QVDOWN_SITE_CS)
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    # per-site heterogeneous C is preserved end to end
    assert {s.name: s.C for s in model.sites} == {sc.name: sc.C for sc in _QVDOWN_SITE_CS}
    for spec in model.sites:
        V, U = vu.site(spec.name)
        assert V.shape == (spec.d_in, spec.C) and U.shape == (spec.C, spec.d_out)

    clean = model.clean_output(tokens)
    none_masked = model.masked_output(
        model.prepare_compute_weights(vu), tokens, {}, {}, None, (), True, remat=False
    )
    assert jnp.array_equal(clean, none_masked), "live=() must be the exact frozen path"

    names = model.site_names
    ones_masks = {s.name: jnp.ones((b, t, s.C)) for s in model.sites}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    full = model.masked_output(
        model.prepare_compute_weights(vu),
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
    site_c = {s.name: s.C for s in model.sites}
    layer4 = tuple(n for n in names if parse_site_name(n)[0] == 4)
    assert q_site in layer4
    zero_mask = {n: jnp.zeros((b, t, site_c[n])) for n in layer4}
    zero_delta = {n: jnp.zeros((b, t)) for n in layer4}
    ablated = model.masked_output(
        model.prepare_compute_weights(vu),
        tokens,
        zero_mask,
        zero_delta,
        None,
        layer4,
        True,
        remat=False,
    )
    assert not jnp.allclose(clean, ablated, atol=1e-4), "ablating layer 4 did nothing"

    site_in = model.read_activations(tokens, model.site_names)
    assert set(site_in) == set(names)
    # q and v read the same post-LN1 residual; down reads the (di,) MLP inner acts
    assert jnp.array_equal(site_in[q_site], site_in["layers.4.self_attn.v_proj"])
    assert site_in["layers.4.mlp.down_proj"].shape == (b, t, cfg.n_intermediate)

    deltas = model.weight_deltas(vu)
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    assert deltas[q_site].shape == (qd, cfg.n_embd)
    assert deltas["layers.4.self_attn.v_proj"].shape == (kvd, cfg.n_embd)


def test_clean_output_and_activations_shares_the_forward():
    """The fused accessor must be exactly the two separate calls (SPEC S3+S4): taps
    bit-equal to `read_activations`, and — when the taps reach the last block, every
    production config — clean logits bit-equal to `clean_output` (one full-depth scan,
    no tail; a mid-stack tap cutoff may recompile the tail within fp32 tolerance).
    Also pins tap invariance to the `wanted` set (a tap can't depend on its neighbors)."""
    cfg = tiny_glu_cfg()
    sites = _mlp_sites(cfg, 3, 6, 8)
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, 16), 0, cfg.vocab_size)

    resid_taps = tuple(f"resid.{i}" for i in range(cfg.n_layer))
    wanted = resid_taps + model.site_names
    logits, taps = model.clean_output_and_activations(tokens, wanted)
    assert jnp.array_equal(logits, model.clean_output(tokens)), "fused clean logits drifted"
    separate = model.read_activations(tokens, wanted)
    assert set(taps) == set(wanted)
    for key in wanted:
        assert jnp.array_equal(taps[key], separate[key]), key

    subset = ("resid.0", "resid.5", site_name(4, "gate"))
    for key, tap in model.read_activations(tokens, subset).items():
        assert jnp.array_equal(tap, separate[key]), key


def test_o_site_masks_attention_output():
    cfg = tiny_glu_cfg()
    o_site = "layers.4.self_attn.o_proj"
    sites = glu_site_specs(cfg, (SiteC(o_site, 8),))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    clean = model.clean_output(tokens)
    ones = model.masked_output(
        model.prepare_compute_weights(vu),
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
    site_in = model.read_activations(tokens, model.site_names)
    assert site_in[o_site].shape == (b, t, cfg.n_head * cfg.head_dim)


@pytest.mark.parametrize(
    "site_cs",
    [mlp_family_site_cs(4, 4, 8), mlp_family_site_cs(3, 6, 8), _QVDOWN_SITE_CS],
    ids=["mlp_l4", "mlp_l3_6", "qv_down_l4"],
)
def test_step_trains_and_has_vpd_signature(site_cs: tuple[SiteC, ...]):
    cfg = tiny_glu_cfg()
    seq = 16
    n_warmup = 2
    sites = glu_site_specs(cfg, site_cs)
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = tiny_glu_chunkwise_ci_fn(model, jax.random.PRNGKey(2), n_blocks=2)
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
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
        ),
        n_warmup_steps=n_warmup,
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

    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    n_steps = 4
    losses = []
    for i in range(n_steps):
        state, m = step(model, state, tokens, jax.random.PRNGKey(100 + i))
        losses.append({k: float(v) for k, v in m.items()})

    assert all(jnp.isfinite(jnp.array(list(m.values()))).all() for m in losses)
    assert int(state.training.step) == n_steps
    # SPEC S13: n_warmup + 1 source-Adam updates per training step, moments persist.
    ppgd_adv = state.training.adversaries["PersistentPGDReconLoss"]
    assert float(ppgd_adv.opt_state.step_count) == n_steps * (n_warmup + 1)
    # SPEC S15: sources stay projected to [0,1].
    for v in ppgd_adv.sources.values():
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    # SPEC S9: p annealed below its 2.0 start by step 4 of 100.
    assert losses[-1]["p_imp"] < 2.0
    # fp32 masters preserved through updates (SPEC N1).
    assert isinstance(state.decomposition.components, ComponentStacks)
    for _, (V, U) in state.decomposition.components.sites_items():
        assert V.dtype == jnp.float32 and U.dtype == jnp.float32
    assert isinstance(state.decomposition.ci_fn, ChunkwiseTransformerCIFn)
    assert state.decomposition.ci_fn.chunks.in_proj_w.dtype == jnp.float32


def test_faith_warmup_decreases_faith():
    cfg = tiny_glu_cfg()
    sites = _mlp_sites(cfg, 3, 4, 8)
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    opt = optax.adamw(1e-2, weight_decay=0.0)
    wstep = make_faith_warmup_step(opt)
    ostate = opt.init(eqx.filter(vu, eqx.is_array))
    first_loss: float | None = None
    loss = None
    for _ in range(30):
        vu, ostate, loss = wstep(model, vu, ostate)
        first_loss = float(loss) if first_loss is None else first_loss
    assert first_loss is not None and loss is not None
    assert float(loss) < first_loss * 0.9, (first_loss, float(loss))


def test_component_stacks_shapes_fp32():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, _QVDOWN_SITE_CS)
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    V_q, U_q = vu.site("layers.4.self_attn.q_proj")
    V_v, U_v = vu.site("layers.4.self_attn.v_proj")
    V_d, U_d = vu.site("layers.4.mlp.down_proj")
    assert V_q.shape == (d, 8) and U_q.shape == (8, qd)
    assert V_v.shape == (d, 12) and U_v.shape == (12, kvd)
    assert V_d.shape == (di, 8) and U_d.shape == (8, d)
    assert isinstance(vu, ComponentStacks)
    assert all(a.dtype == jnp.float32 for pair in vu.stacks.values() for a in pair)


def test_fresh_pgd_adversary_step():
    """Fresh per-batch sign-PGD (torch PGDReconLoss as the TRAINING adversary):
    no persistent source state, metrics keyed `loss/PGDReconLoss`, sources
    sampled+ascended inside the step, and the ascent strength responds to n_steps."""
    cfg = tiny_glu_cfg()
    site_cs = (
        SiteC("layers.4.self_attn.q_proj", 8),
        SiteC("layers.4.mlp.gate_proj", 8),
        SiteC("layers.4.mlp.up_proj", 8),
        SiteC("layers.4.mlp.down_proj", 12),
    )
    seq = 16
    sites = glu_site_specs(cfg, site_cs)
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    def make_state() -> TrainState:
        # Fresh buffers per call: `step` donates the state, so a shared vu/ci_fn would be
        # deleted after the first run_step and crash the second. Deterministic keys keep
        # the two states' inits bit-identical (the "same init" the comparison below needs).
        vu = init_component_stacks(sites, jax.random.PRNGKey(1))
        ci_fn = tiny_glu_chunkwise_ci_fn(model, jax.random.PRNGKey(2), n_blocks=1)
        return TrainState(
            decomposition=Decomposition(components=vu, ci_fn=ci_fn),
            training=TrainingItem(
                components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
                ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
                adversaries={},
                step=jnp.zeros((), jnp.int32),
            ),
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
                    source_shape="bsc",
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
            remat_recon_forwards=False,
            remat_ci_fn=False,
            mesh=None,
        )
        tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
        return step(model, make_state(), tokens, jax.random.PRNGKey(100))

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
    assert state.training.adversaries == {}, "fresh adversary carries no persistent sources"
    assert int(state.training.step) == 1

    _, metrics_unascended = run_step(n_ascent_steps=0)
    assert float(metrics["loss/PGDReconLoss"]) >= float(metrics_unascended["loss/PGDReconLoss"]), (
        "one sign step from the same init must not weaken the adversary"
    )


def test_chunkwise_ci_init_vmap_matches_unrolled_reference():
    """The vmapped multi-chunk CI init must be BIT-identical to the unrolled+stacked
    per-chunk form it replaced (`eqx.filter_vmap` over the SAME `fold_in` keys) — with
    n_chunks > 1 so the chunk axis is real, and heterogeneous per-slot C."""
    from param_decomp.core.ci_fn import _init_chunk_transformer, init_chunkwise_transformer_ci_fn

    kinds = (("self_attn.q_proj", 8), ("self_attn.v_proj", 12), ("mlp.down_proj", 16))
    n_chunks = 3
    sites = tuple(SiteSpec(f"layers.{i}.{k}", 24, 24, c) for i in range(n_chunks) for k, c in kinds)
    arch = ChunkwiseTransformerCIArch(
        chunks=tuple(
            Chunk(
                input_taps=(f"resid.{i}",),
                output_sites=tuple(f"layers.{i}.{k}" for k, _ in kinds),
            )
            for i in range(n_chunks)
        ),
        input_dim=24, d_model=16, n_blocks=2, attention=MHACIAttention(n_heads=2), ffn_hidden=32,
        ffn_kind="gelu", learned_norm_scale=False,
    )  # fmt: skip
    key = jax.random.PRNGKey(7)

    new = init_chunkwise_transformer_ci_fn(arch, sites, key)
    slot_cs = tuple(c for _, c in kinds)
    ref_stacked = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *[
            _init_chunk_transformer(arch, arch.input_dim, slot_cs, jax.random.fold_in(key, i))
            for i in range(n_chunks)
        ],
    )
    for got, want in zip(
        jax.tree.leaves(eqx.filter(new.chunks, eqx.is_array)),
        jax.tree.leaves(eqx.filter(ref_stacked, eqx.is_array)),
        strict=True,
    ):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert jnp.array_equal(got, want)
