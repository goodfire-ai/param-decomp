"""CPU tests for the LlamaSimpleMLP target + generic trainer at a tiny config.

Mirrors `test_llama31.py`: validates the `DecomposedModel` contract (mask=1 identity
reconstructs the clean forward, shapes, site seams) and the full SPEC step — for mixed
attention + MLP sites with heterogeneous per-site C — without real weights or a GPU.
"""

import os
from pathlib import Path

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
    ChunkwiseTransformerCIFn,
)
from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    init_component_stacks,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    StochasticReconSubsetLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.model import MaterializedMasking, PlacedModel, site_weight_delta
from param_decomp.core.nonlinearity import (
    KVHeads,
    Neurons,
    QueryHeads,
)
from param_decomp.core.objective import build_objective
from param_decomp.core.recon import StochasticSources
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_faith_warmup_step,
    make_train_step,
)
from param_decomp.targets.glu_transformer import (
    GLUDecomposedModel,
    neuron_aligned_component_initializer,
)
from param_decomp.targets.llama_simple_mlp import (
    KIND_ORDER,
    SIMPLE_MLP_ANATOMY,
    canonical_site_cs,
    parse_site_name,
    site_dims,
    site_name,
    site_specs,
)
from param_decomp.targets.testing import (
    SIMPLE_MLP_MIXED_SITE_CS,
    capture_clean,
    capture_site_outputs,
    run_clean,
    run_masked,
    tiny_simple_mlp_cfg,
    tiny_simple_mlp_chunkwise_ci_fn,
    tiny_simple_mlp_decomposed_model,
)
from param_decomp.targets.transformer_taps import (
    attention_input_tap_key,
    attention_output_tap_key,
    mlp_hidden_tap_key,
    mlp_input_tap_key,
)


def _site_input_key(site: str) -> str:
    layer, kind = parse_site_name(site)
    match kind:
        case "q_proj" | "k_proj" | "v_proj":
            return attention_input_tap_key(layer)
        case "o_proj":
            return attention_output_tap_key(layer)
        case "c_fc":
            return mlp_input_tap_key(layer)
        case "down_proj":
            return mlp_hidden_tap_key(layer)
        case _:
            raise AssertionError(kind)


def _capture_site_inputs(
    model: GLUDecomposedModel, tokens: jax.Array, sites: tuple[str, ...]
) -> dict[str, jax.Array]:
    input_keys = tuple(_site_input_key(site) for site in sites)
    captures = capture_clean(model, tokens, tuple(dict.fromkeys(input_keys)))
    return dict(zip(sites, (captures[key] for key in input_keys), strict=True))


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


def test_site_specs_dims():
    cfg = tiny_simple_mlp_cfg()
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    specs = site_specs(
        cfg,
        canonical_site_cs(
            tuple(
                SiteC(site_name(2, k), 4)
                for k in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "c_fc",
                    "down_proj",
                )
            )
        ),
    )
    dims = {s.name: (s.d_in, s.d_out, s.C) for s in specs}
    assert dims["h.2.attn.q_proj"] == (cfg.n_embd, qd, 4)
    assert dims["h.2.attn.k_proj"] == (cfg.n_embd, kvd, 4)
    assert dims["h.2.attn.o_proj"] == (qd, cfg.n_embd, 4)
    assert dims["h.2.mlp.c_fc"] == (cfg.n_embd, cfg.n_intermediate, 4)
    assert dims["h.2.mlp.down_proj"] == (cfg.n_intermediate, cfg.n_embd, 4)
    partitions = {s.name: s.nonlinearity_partition for s in specs}
    assert partitions["h.2.mlp.c_fc"] == Neurons()
    assert partitions["h.2.attn.q_proj"] == QueryHeads(cfg.n_head)
    kv_heads = KVHeads(cfg.n_kv_head, cfg.n_head // cfg.n_kv_head)
    assert partitions["h.2.attn.k_proj"] == kv_heads
    assert partitions["h.2.attn.v_proj"] == kv_heads
    assert partitions["h.2.mlp.down_proj"] is None and partitions["h.2.attn.o_proj"] is None
    with pytest.raises(AssertionError, match="canonical"):
        site_specs(cfg, (SiteC("h.2.mlp.c_fc", 4), SiteC("h.2.attn.q_proj", 4)))


def test_simple_mlp_target_selects_neuron_aligned_initializer():
    from param_decomp.experiments.lm.load_run import component_initializer_for
    from param_decomp.experiments.lm.resolved import LlamaSimpleMLPTargetConfig

    target = LlamaSimpleMLPTargetConfig(
        pretrain_run_path="goodfire/spd/runs/t-9d2b8f02",
        sites=(),
        weights_dtype="float32",
        attention_implementation="auto",
        component_initialization="neuron_aligned",
    )

    assert component_initializer_for(target) is neuron_aligned_component_initializer


def test_neuron_aligned_init_exactly_reconstructs_simple_mlp():
    cfg = tiny_simple_mlp_cfg()
    capacities = {
        kind: (
            site_dims(cfg, kind).d_in
            if kind in SIMPLE_MLP_ANATOMY.row_kinds
            else site_dims(cfg, kind).d_out
        )
        for kind in KIND_ORDER
    }
    sites = site_specs(
        cfg,
        tuple(SiteC(site_name(2, kind), capacities[kind]) for kind in KIND_ORDER),
    )
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))

    components = neuron_aligned_component_initializer(model, jax.random.PRNGKey(1))

    for delta in model.weight_deltas(components).values():
        assert jnp.array_equal(delta, jnp.zeros_like(delta))
    for _name, site_components in components.sites_items():
        assert jnp.all(jnp.linalg.norm(site_components.V, axis=0) > 0)
        assert jnp.all(jnp.linalg.norm(site_components.U, axis=1) > 0)


def test_clean_path_and_masked_identity():
    cfg = tiny_simple_mlp_cfg()
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    # per-site heterogeneous C is preserved end to end
    assert {s.name: s.C for s in model.sites} == {s.name: s.C for s in SIMPLE_MLP_MIXED_SITE_CS}
    for spec in model.sites:
        site_components = vu.site(spec.name)
        assert site_components.V.shape == (spec.d_in, spec.C)
        assert site_components.U.shape == (spec.C, spec.d_out)

    clean = run_clean(model, tokens)
    assert clean.shape == (b, t, cfg.vocab_size)

    # Masks=1, delta=1, route-everywhere reconstructs the frozen path up to
    # decomposition rounding (the V@U + (W − V@U) identity; exact only in exact math).
    names = model.site_names
    ones_masks = {s.name: jnp.ones((b, t, s.C)) for s in model.sites}
    ones_delta = {s: jnp.ones((b, t)) for s in names}
    prepared = model.prepare_compute_weights(vu, None)
    full = run_masked(model, prepared, tokens, ones_masks, ones_delta, None, True, remat=False)
    assert jnp.allclose(clean, full, atol=1e-4), "mask=1 identity drifted"

    input_keys = model._capture_grammar().block_tap_keys((2, 3))
    site_in = capture_clean(model, tokens, input_keys)
    assert set(site_in) == set(input_keys)
    assert attention_input_tap_key(2) in site_in
    assert site_in[mlp_hidden_tap_key(3)].shape == (b, t, cfg.n_intermediate)
    assert site_in[mlp_input_tap_key(2)].shape == (b, t, cfg.n_embd)

    deltas = model.weight_deltas(vu)
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    assert site_weight_delta(deltas, vu, "h.2.attn.q_proj").shape == (qd, cfg.n_embd)
    assert site_weight_delta(deltas, vu, "h.2.attn.v_proj").shape == (kvd, cfg.n_embd)
    assert site_weight_delta(deltas, vu, "h.2.mlp.c_fc").shape == (cfg.n_intermediate, cfg.n_embd)
    assert site_weight_delta(deltas, vu, "h.3.mlp.down_proj").shape == (
        cfg.n_embd,
        cfg.n_intermediate,
    )
    assert all(v.dtype == jnp.float32 for v in deltas.values())
    target_sq_norms = model.target_weight_sq_norms()
    for name, group, slot in vu.site_slots:
        site = vu.site(name)
        delta = site_weight_delta(deltas, vu, name)
        target_weight = delta + (site.V.astype(jnp.float32) @ site.U.astype(jnp.float32)).T
        assert jnp.allclose(target_sq_norms[group][slot], jnp.sum(target_weight**2))


@pytest.mark.parametrize("ablated_site", ["h.2.attn.q_proj", "h.2.mlp.c_fc"])
def test_zero_masking_one_site_changes_logits(ablated_site: str):
    """q is live ahead of RoPE/SDPA; c_fc ahead of the GELU — zero-mask + zero-delta on
    either must change the logits. The other sites take mask=1 + delta=1 (exactly the
    frozen W), so the change is attributable to the ablated site alone."""
    cfg = tiny_simple_mlp_cfg()
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    clean = run_clean(model, tokens)
    fill = {s.name: 0.0 if s.name == ablated_site else 1.0 for s in model.sites}
    masks = {s.name: jnp.full((b, t, s.C), fill[s.name]) for s in model.sites}
    delta_masks = {s.name: jnp.full((b, t), fill[s.name]) for s in model.sites}
    prepared = model.prepare_compute_weights(vu, None)
    ablated = run_masked(model, prepared, tokens, masks, delta_masks, None, True, remat=False)
    assert not jnp.allclose(clean, ablated, atol=1e-4), f"ablating {ablated_site} did nothing"


def test_masked_site_outputs_frozen_when_routed_false_or_unmasked():
    """Clean per-site output: routing FALSE everywhere falls onto `site_out`'s frozen
    `x @ W` branch — exactly the target site output. With a single-site decomposition the
    frozen W per site is `site_input @ W.T`, recovered from `weight_deltas` + `V@U`."""
    cfg = tiny_simple_mlp_cfg()
    sites_cs = (SiteC("h.2.attn.q_proj", 8), SiteC("h.2.mlp.c_fc", 12))
    sites = site_specs(cfg, sites_cs)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    names = model.site_names
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    site_in = _capture_site_inputs(model, tokens, model.site_names)
    ones_masks = {s.name: jnp.ones((b, t, s.C)) for s in model.sites}
    false_routes = {s: jnp.zeros((b, t), bool) for s in names}

    clean_outs = capture_site_outputs(
        model,
        model.prepare_compute_weights(vu, None),
        tokens,
        MaterializedMasking(component_masks=ones_masks, routes=false_routes),
    )
    assert set(clean_outs) == set(names)
    # frozen `x @ W` per site, reconstructed independently from weight_deltas + V@U.
    deltas = model.weight_deltas(vu)
    for s in names:
        site_components = vu.site(s)
        W = (
            site_components.V.astype(jnp.float32) @ site_components.U.astype(jnp.float32)
        ).T + site_weight_delta(deltas, vu, s)  # (d_out, d_in)
        expected = site_in[s].astype(jnp.float32) @ W.T
        assert jnp.allclose(clean_outs[s].astype(jnp.float32), expected, atol=1e-3), s


@pytest.mark.parametrize("site_name_str", ["h.2.attn.q_proj", "h.2.mlp.c_fc"])
def test_masked_site_outputs_match_hand_computed_masked_linear(site_name_str: str):
    """Masked per-site output equals the hand-computed `((x@V)*m)@U` (+ delta path). One
    site at a time so the masked site input equals the clean `site_inputs` (no upstream
    masked site contaminating the threaded forward)."""
    cfg = tiny_simple_mlp_cfg()
    sites_cs = (SiteC(site_name_str, 8),)
    sites = site_specs(cfg, sites_cs)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    site = site_name_str
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    input_key = _site_input_key(site)
    x_in = capture_clean(model, tokens, (input_key,))[input_key]
    site_components = vu.site(site)
    mask = jax.random.uniform(jax.random.PRNGKey(7), (b, t, sites_cs[0].C))

    prepared = model.prepare_compute_weights(vu, None)
    no_delta = capture_site_outputs(
        model,
        prepared,
        tokens,
        MaterializedMasking(component_masks={site: mask}),
    )
    hand = ((x_in @ site_components.V) * mask) @ site_components.U
    assert jnp.allclose(no_delta[site], hand, atol=1e-4), site

    # delta path: + delta_mask · (x @ Δ), Δ = W − V@U == model.weight_deltas (fp32 oracle)
    delta_in = site_weight_delta(model.weight_deltas(vu), vu, site)
    delta_mask = jax.random.uniform(jax.random.PRNGKey(9), (b, t))
    with_delta = capture_site_outputs(
        model,
        prepared,
        tokens,
        MaterializedMasking(
            component_masks={site: mask},
            weight_delta_masks={site: delta_mask},
        ),
    )
    hand_delta = delta_mask[..., None] * (x_in.astype(jnp.float32) @ delta_in.T)
    expected = hand.astype(jnp.float32) + hand_delta
    assert jnp.allclose(with_delta[site].astype(jnp.float32), expected, atol=1e-3), site


def test_o_site_masks_attention_output():
    cfg = tiny_simple_mlp_cfg()
    o_site = "h.2.attn.o_proj"
    sites = site_specs(cfg, (SiteC(o_site, 8),))
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)

    clean = run_clean(model, tokens)
    ones = run_masked(
        model,
        model.prepare_compute_weights(vu, None),
        tokens,
        {o_site: jnp.ones((b, t, 8))},
        {o_site: jnp.ones((b, t))},
        None,
        True,
        remat=False,
    )
    assert jnp.allclose(clean, ones, atol=1e-4)
    # o's clean site input is the pre-o_proj attention output, shape (b, t, qd)
    site_in = _capture_site_inputs(model, tokens, model.site_names)
    assert site_in[o_site].shape == (b, t, cfg.n_head * cfg.head_dim)


def test_step_trains_and_has_vpd_signature():
    cfg = tiny_simple_mlp_cfg()
    site_cs = SIMPLE_MLP_MIXED_SITE_CS
    seq = 16
    n_warmup = 2
    sites = site_specs(cfg, site_cs)
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
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(
                max_val=0.01,
                points=(Knot(at=0.0, frac=0.0), Knot(at=0.025, frac=1.0), Knot(at=1.0, frac=1.0)),
            ),
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
                    adam=ppgd_cfg.optimizer,
                    n_warmup=ppgd_cfg.n_warmup_steps,
                )
            },
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
            ppgd_cfg,
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

    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    n_steps = 4
    losses = []
    for i in range(n_steps):
        state, m = step(placed, state, tokens, jax.random.PRNGKey(100 + i))
        losses.append({k: float(v) for k, v in m.items()})

    assert all(jnp.isfinite(jnp.array(list(m.values()))).all() for m in losses)
    assert int(state.training.step) == n_steps
    # SPEC S13: n_warmup + 1 source-Adam updates per training step, moments persist.
    ppgd_adv = state.training.adversaries["PersistentPGDReconLoss"]
    assert float(ppgd_adv.opt_state.step_count) == n_steps * (n_warmup + 1)
    # SPEC S15: sources stay projected to [0,1].
    for v in jax.tree.leaves(ppgd_adv.sources):
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0
    # SPEC S9: gamma annealed below its 1.0 start by step 4 of 100.
    assert losses[-1]["gamma_imp"] < 1.0
    # fp32 masters preserved through updates (SPEC N1).
    assert isinstance(state.decomposition.components, ComponentStacks)
    for _, site_components in state.decomposition.components.sites_items():
        assert site_components.V.dtype == jnp.float32
        assert site_components.U.dtype == jnp.float32
    assert isinstance(state.decomposition.ci_fn, ChunkwiseTransformerCIFn)
    assert state.decomposition.ci_fn.chunks.in_proj_w.dtype == jnp.float32


def test_faith_warmup_decreases_faith():
    cfg = tiny_simple_mlp_cfg()
    sites = site_specs(cfg, canonical_site_cs(SIMPLE_MLP_MIXED_SITE_CS))
    placed = PlacedModel(
        model=tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    opt = optax.adamw(1e-2, weight_decay=0.0)
    wstep = make_faith_warmup_step(opt, faithfulness_loss_for(placed.model))
    ostate = opt.init(eqx.filter(vu, eqx.is_array))
    first_loss: float | None = None
    loss = None
    for _ in range(30):
        vu, ostate, loss = wstep(placed, vu, ostate)
        first_loss = float(loss) if first_loss is None else first_loss
    assert first_loss is not None and loss is not None
    assert float(loss) < first_loss * 0.9, (first_loss, float(loss))


def test_component_stacks_shapes_fp32():
    cfg = tiny_simple_mlp_cfg()
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    q = vu.site("h.2.attn.q_proj")
    v = vu.site("h.2.attn.v_proj")
    fc = vu.site("h.2.mlp.c_fc")
    down = vu.site("h.3.mlp.down_proj")
    assert q.V.shape == (d, 8) and q.U.shape == (8, qd)
    assert v.V.shape == (d, 12) and v.U.shape == (12, kvd)
    assert fc.V.shape == (d, 8) and fc.U.shape == (8, di)
    assert down.V.shape == (di, 16) and down.U.shape == (16, d)
    assert all(a.dtype == jnp.float32 for pair in vu.stacks.values() for a in pair)


def test_engine_rejects_ragged_site_sets():
    """The engine's segmented masked forward demands kind-uniform decomposed layers —
    the deleted hand-rolled model's ragged freedom does not carry over."""
    cfg = tiny_simple_mlp_cfg()
    ragged = canonical_site_cs((SiteC("h.2.attn.q_proj", 8), SiteC("h.3.mlp.down_proj", 16)))
    model = tiny_simple_mlp_decomposed_model(cfg, site_specs(cfg, ragged), jax.random.PRNGKey(0))
    vu = init_component_stacks(model.sites, jax.random.PRNGKey(1))
    b, t = 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(2), (b, t), 0, cfg.vocab_size)
    masks = {s.name: jnp.ones((b, t, s.C)) for s in model.sites}
    deltas = {s.name: jnp.ones((b, t)) for s in model.sites}
    prepared = model.prepare_compute_weights(vu, None)
    with pytest.raises(AssertionError, match="partially decomposed"):
        run_masked(model, prepared, tokens, masks, deltas, None, True, remat=False)


_DATA_ROOT = Path(env) if (env := os.environ.get("PD_TEST_DATA_ROOT")) else None
_REAL_CACHE_DIR = _DATA_ROOT / "pretrain_cache" / "spd-t-9d2b8f02" if _DATA_ROOT else None
_PRODUCTION_CS = {
    "c_fc": 3072,
    "down_proj": 3584,
    "q_proj": 768,
    "k_proj": 768,
    "v_proj": 1024,
    "o_proj": 1024,
}
"""The current JAX 4-layer Pile reference (`pile_llama_simple_mlp-4L.yaml`)."""


@pytest.mark.skipif(_REAL_CACHE_DIR is None, reason="PD_TEST_DATA_ROOT not set")
def test_pretrained_target_converts_with_all_layers():
    """`kind: pretrained` LlamaSimpleMLP target specs convert, tiling the simple_mlp
    c-spec over the checkpoint's n_layer (4)."""
    import yaml

    assert _REAL_CACHE_DIR is not None and _DATA_ROOT is not None

    from param_decomp.experiments.lm.config import (
        LMExperimentConfig,
        build_experiment_config,
    )
    from param_decomp.experiments.lm.resolved import LlamaSimpleMLPTargetConfig, ResolvedLMData

    reference_yaml = (
        Path(__file__).parents[2] / "experiments" / "lm" / "configs" / "llama8b_l18_b128_cmp32.yaml"
    )
    raw = yaml.safe_load(reference_yaml.read_text())
    raw["target"]["spec"] = {
        "kind": "pretrained",
        "model_class": (
            "param_decomp.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP"
        ),
        "run_path": "goodfire/spd/runs/t-9d2b8f02",
    }
    raw["decomposition"]["sites"] = {
        "kind": "simple_mlp",
        "layers": {"kind": "all"},
        "cs": dict(_PRODUCTION_CS),
        "initialization": "neuron_aligned",
    }

    cfg = build_experiment_config(LMExperimentConfig(**raw), "p-00000000", _DATA_ROOT)
    target = cfg.target
    assert isinstance(target, LlamaSimpleMLPTargetConfig)
    assert target.pretrain_run_path == "goodfire/spd/runs/t-9d2b8f02"
    assert target.component_initialization == "neuron_aligned"
    assert len(target.sites) == 4 * 6
    assert target.sites == canonical_site_cs(target.sites)
    by_name = {sc.name: sc.C for sc in target.sites}
    for layer in range(4):
        assert by_name[f"h.{layer}.mlp.c_fc"] == 3072
        assert by_name[f"h.{layer}.mlp.down_proj"] == 3584
        assert by_name[f"h.{layer}.attn.q_proj"] == 768
        assert by_name[f"h.{layer}.attn.v_proj"] == 1024
    assert target.sites[0] == SiteC("h.0.attn.q_proj", 768)
    loss_terms = build_objective(
        cfg.pd.loss_metrics,
        tuple(sc.name for sc in target.sites),
    )
    (stoch_term,) = [t for t in loss_terms.recon if t.name == "StochasticReconSubsetLoss"]
    assert isinstance(stoch_term.sources, StochasticSources)
    assert stoch_term.uses_weight_deltas
    assert isinstance(cfg.data, ResolvedLMData)
    assert cfg.data.dir.name == "fineweb_llama_tok_2048"
