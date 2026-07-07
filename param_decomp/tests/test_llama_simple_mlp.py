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
from param_decomp.train import TrainState, make_faith_warmup_step, make_train_step


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
