"""CPU tests for the qwen36_moe target at a tiny config.

The `DecomposedModel` contract over the hybrid DeltaNet/attention + MoE engine:
mask=1 + delta=1 reconstructs the clean forward, ablation changes logits, route=False
takes the frozen path, the stochastic masked forward runs and differentiates, the fused
gate/up taps carry the scattered routed semantics, and the whole-grid coverage contract
fails closed. The heart is the ROUTED-vs-DENSE decomposed parity block: the production
`"routed"` execution (job-space compute over selected experts) against the `"dense"`
oracle (`ExpertsExecution`) — outputs, captures, and gradients (dV/dU, d-masks/CI,
d-delta, d-input), materialized and stochastic, deltas/routes on and off. Stochastic
draws are full-width in both arms, so the same draw key yields the same realization and
the arms differ by fp32 reassociation only. Direct HF parity lives in
`tests/qwen36_moe_hf_parity/`.
"""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    SiteDims,
    init_component_stacks,
)
from param_decomp.core.model import (
    MaterializedMasking,
    StochasticMasking,
    site_weight_delta,
)
from param_decomp.targets.qwen36_moe import (
    KIND_ORDER,
    Qwen36MoeConfig,
    Qwen36MoeDecomposedModel,
    _fold_routing_weights,
    canonical_site_cs,
    full_site_cs,
    layer_is_full_attention,
    parse_site_name,
    qwen36_35b_a3b_config,
    qwen36_moe_site_specs,
    site_contraction,
    site_dims,
    site_name,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    capture_clean,
    materialized_logits,
    run_clean,
    run_masked,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
)
from param_decomp.targets.transformer_taps import resid_tap_key, site_output_tap_key


def _assert_routed_matches_dense(actual: jax.Array, desired: jax.Array) -> None:
    """Clean (ROUTED experts) vs identity-masked (DENSE experts) forwards reassociate
    differently — grouped per-expert blocks with the fp32 post-down combine vs one dense
    contraction with routing folded pre-down — and the difference compounds through the
    residual stream, so exact equality is impossible; the tolerance bounds the observed
    fp32 divergence with headroom."""
    np.testing.assert_allclose(actual, desired, rtol=2e-4, atol=2e-4)


def _full_model_and_vu(key: jax.Array) -> tuple[Qwen36MoeDecomposedModel, ComponentStacks]:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_CS))
    model_key, vu_key = jax.random.split(key)
    model = tiny_qwen36_decomposed_model(cfg, sites, model_key)
    return model, init_component_stacks(sites, vu_key)


def _tokens(cfg: Qwen36MoeConfig) -> jax.Array:
    return jax.random.randint(jax.random.PRNGKey(7), (2, 12), 0, cfg.vocab_size)


def _ones_masks(model: Qwen36MoeDecomposedModel, leading: tuple[int, ...]):
    masks = {spec.name: jnp.ones((*leading, spec.C)) for spec in model.sites}
    deltas = {spec.name: jnp.ones(leading) for spec in model.sites}
    return masks, deltas


def test_mask_one_delta_one_reconstructs_clean_forward():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(0))
    tokens = _tokens(model.cfg)
    prepared = model.prepare_compute_weights(vu, None)
    masks, deltas = _ones_masks(model, tokens.shape)
    clean = materialized_logits(run_clean(model, tokens))
    masked = materialized_logits(
        run_masked(
            model, prepared, tokens, masks, deltas, None, uses_weight_deltas=True, remat=False
        )
    )
    _assert_routed_matches_dense(masked, clean)


def test_zero_masks_change_logits():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(1))
    tokens = _tokens(model.cfg)
    prepared = model.prepare_compute_weights(vu, None)
    masks = {spec.name: jnp.zeros((*tokens.shape, spec.C)) for spec in model.sites}
    deltas = {spec.name: jnp.zeros(tokens.shape) for spec in model.sites}
    clean = materialized_logits(run_clean(model, tokens))
    ablated = materialized_logits(
        run_masked(
            model, prepared, tokens, masks, deltas, None, uses_weight_deltas=False, remat=False
        )
    )
    assert np.abs(np.asarray(ablated - clean)).max() > 1e-3


def test_route_false_takes_frozen_path():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(2))
    tokens = _tokens(model.cfg)
    prepared = model.prepare_compute_weights(vu, None)
    masks = {
        spec.name: jax.random.uniform(jax.random.PRNGKey(3), (*tokens.shape, spec.C))
        for spec in model.sites
    }
    deltas = {spec.name: jnp.zeros(tokens.shape) for spec in model.sites}
    routes = {spec.name: jnp.zeros(tokens.shape, bool) for spec in model.sites}
    clean = materialized_logits(run_clean(model, tokens))
    routed_off = materialized_logits(
        run_masked(
            model, prepared, tokens, masks, deltas, routes, uses_weight_deltas=False, remat=False
        )
    )
    _assert_routed_matches_dense(routed_off, clean)


def test_stochastic_masked_forward_runs_and_differentiates():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(4))
    tokens = _tokens(model.cfg)
    ci = {spec.name: jnp.full((*tokens.shape, spec.C), 0.5) for spec in model.sites}
    masking = StochasticMasking(
        ci_stacked=model.stack_ci(ci), draw_key=jax.random.PRNGKey(5), routes=None
    )
    clean = materialized_logits(run_clean(model, tokens))

    def loss(components: ComponentStacks) -> jax.Array:
        prepared = model.prepare_compute_weights(components, None)
        masked = model.masked_forward(prepared, tokens, masking=masking, placement=None, remat=True)
        return model.recon_loss_fn(masked.output, clean)

    value, grads = eqx.filter_value_and_grad(loss)(vu)
    assert jnp.isfinite(value)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert leaves and all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0) for leaf in leaves)


def _selected_expert_mask(cfg: Qwen36MoeConfig, router: jax.Array, h2: jax.Array) -> jax.Array:
    """Which experts the router selects per position (`[*lead, E]` bool) — the target's
    top-k routing re-derived from the captured MoE input."""
    probs = jax.nn.softmax((h2 @ router.T).astype(jnp.float32), axis=-1)
    _values, top_idx = jax.lax.top_k(probs, cfg.n_experts_per_token)
    return jnp.any(jax.nn.one_hot(top_idx, cfg.n_experts, dtype=bool), axis=-2)


def test_captures_cover_the_closed_vocabulary():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(6))
    cfg = model.cfg
    tokens = _tokens(cfg)
    second, mid, last = 1, cfg.n_layer // 2, cfg.n_layer - 1
    gate_key = site_output_tap_key(site_name(mid, "experts_gate"))
    keys = (
        resid_tap_key(0),
        resid_tap_key(second),
        resid_tap_key(cfg.n_layer),
        f"mlp_in.{mid}",
        f"mlp_in.{last}",
        gate_key,
        site_output_tap_key(site_name(last, "experts_down")),
        site_output_tap_key(site_name(second, "shared_down")),
    )
    captures = capture_clean(model, tokens, keys)
    assert set(captures) == set(keys)
    np.testing.assert_array_equal(captures[resid_tap_key(0)], model.embed[tokens])
    fused = cfg.n_experts * cfg.moe_intermediate
    assert captures[gate_key].shape == (*tokens.shape, fused)
    assert captures[f"mlp_in.{last}"].shape == (*tokens.shape, cfg.n_embd)

    # The masked forward under the exact identity reproduces the clean captures — except
    # the fused gate tap's unselected experts, which the dense decomposed arm materializes
    # and the routed clean arm zeroes (`_LayerActs`): those compare under the selection.
    prepared = model.prepare_compute_weights(vu, None)
    masks, deltas = _ones_masks(model, tokens.shape)
    masked = model.masked_forward(
        prepared,
        tokens,
        masking=MaterializedMasking(component_masks=masks, weight_delta_masks=deltas),
        placement=None,
        capture_keys=frozenset(keys),
        remat=False,
    )
    selected = _selected_expert_mask(cfg, model.moe.router[mid], captures[f"mlp_in.{mid}"])
    selected_wide = jnp.repeat(selected, cfg.moe_intermediate, axis=-1)
    for key in keys:
        masked_tap = masked.captures[key]
        if key == gate_key:
            masked_tap = jnp.where(selected_wide, masked_tap, 0.0)
        _assert_routed_matches_dense(masked_tap, captures[key])

    with pytest.raises(AssertionError):
        capture_clean(model, tokens, ("attn_in.0",))


def test_routed_frozen_fused_taps_scatter_selected_experts():
    """The clean (routed-frozen) fused gate/up taps hold the scattered job results:
    selected experts match the dense matmul to reduction-order tolerance, unselected
    experts are EXACTLY zero (the routed arm never computes them)."""
    model, _vu = _full_model_and_vu(jax.random.PRNGKey(12))
    cfg = model.cfg
    tokens = _tokens(cfg)
    layer = cfg.n_layer - 1
    keys = (
        f"mlp_in.{layer}",
        site_output_tap_key(site_name(layer, "experts_gate")),
        site_output_tap_key(site_name(layer, "experts_up")),
    )
    captures = capture_clean(model, tokens, keys)
    h2 = captures[f"mlp_in.{layer}"]
    selected = _selected_expert_mask(cfg, model.moe.router[layer], h2)
    selected_wide = np.asarray(jnp.repeat(selected, cfg.moe_intermediate, axis=-1))
    for kind, frozen_fused in (
        ("experts_gate", model.moe.experts_gate[layer]),
        ("experts_up", model.moe.experts_up[layer]),
    ):
        tap = np.asarray(captures[site_output_tap_key(site_name(layer, kind))])
        dense = np.asarray(h2 @ frozen_fused.T)
        np.testing.assert_allclose(tap[selected_wide], dense[selected_wide], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(tap[~selected_wide], 0.0)


def test_masked_grads_flow_through_routed_frozen_experts():
    """Decomposing only the shared kinds leaves the expert arm routed-frozen inside the
    MASKED forward: gradients must flow through the routed compute (the custom-VJP
    gathers, under remat) to the shared V/U."""
    cfg = tiny_qwen36_cfg()
    shared_cs = {kind: c for kind, c in TINY_QWEN36_CS.items() if kind.startswith("shared_")}
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, shared_cs))
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(13))
    vu = init_component_stacks(sites, jax.random.PRNGKey(14))
    tokens = _tokens(cfg)
    ci = {spec.name: jnp.full((*tokens.shape, spec.C), 0.5) for spec in model.sites}
    masking = StochasticMasking(
        ci_stacked=model.stack_ci(ci), draw_key=jax.random.PRNGKey(15), routes=None
    )
    clean = materialized_logits(run_clean(model, tokens))

    def loss(components: ComponentStacks) -> jax.Array:
        prepared = model.prepare_compute_weights(components, None)
        masked = model.masked_forward(prepared, tokens, masking=masking, placement=None, remat=True)
        return model.recon_loss_fn(masked.output, clean)

    value, grads = eqx.filter_value_and_grad(loss)(vu)
    assert jnp.isfinite(value)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert leaves and all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0) for leaf in leaves)


def _dense_twin(model: Qwen36MoeDecomposedModel) -> Qwen36MoeDecomposedModel:
    return dataclasses.replace(model, experts_execution="dense")


def _assert_values_close(actual: np.ndarray, desired: np.ndarray, err_msg: str = "") -> None:
    """fp32 CPU: the two executions compute identical math (identical draws included)
    and differ only by reassociation — grouped per-expert blocks + fp32 job sums vs one
    dense contraction — compounding through the 8-layer residual stream."""
    np.testing.assert_allclose(actual, desired, rtol=2e-4, atol=2e-4, err_msg=err_msg)


def _assert_grad_leaves_close(routed_grads: object, dense_grads: object) -> None:
    """Per-leaf relative Frobenius error: reassociation noise concentrates on near-zero
    ELEMENTS (verified down to the vendored kernels' internal fp32 floor in f64), so an
    elementwise bound punishes exactly the entries that carry no signal; a wrong or
    missing term lands O(1) on this metric, not 1e-3."""
    for routed_leaf, dense_leaf in zip(
        jax.tree.leaves(routed_grads), jax.tree.leaves(dense_grads), strict=True
    ):
        routed_np, dense_np = np.asarray(routed_leaf), np.asarray(dense_leaf)
        error = np.linalg.norm(routed_np - dense_np) / np.linalg.norm(dense_np)
        assert error < 2e-3, error


def _routed_and_dense_grads(
    loss_of_model: Callable[..., jax.Array], model: Qwen36MoeDecomposedModel, *args: object
) -> tuple[object, object]:
    """Gradients of `loss_of_model(model, *args)` wrt every arg, under both executions.
    The model rides as a traced jit arg (the HLO-baking rule); the static execution
    field selects the arm at trace time."""
    grad_fn = jax.jit(jax.grad(loss_of_model, argnums=tuple(range(1, 1 + len(args)))))
    return grad_fn(model, *args), grad_fn(_dense_twin(model), *args)


@pytest.mark.parametrize(
    ("with_deltas", "with_routes"), [(True, True), (True, False), (False, False)]
)
def test_routed_decomposed_matches_dense_materialized(with_deltas: bool, with_routes: bool):
    """The heart of the training math: the routed decomposed arm against the dense
    oracle through the full masked forward — outputs, captures (the fused gate tap
    under the selection mask: the routed arm zeroes unselected experts the oracle
    materializes), and gradients (dV/dU, d-masks, d-deltas, d-input via the token
    embedding) — with the delta channel and routes on and off."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(20))
    cfg = model.cfg
    tokens = _tokens(cfg)
    masks = {
        spec.name: jax.random.uniform(
            jax.random.fold_in(jax.random.PRNGKey(21), i), (*tokens.shape, spec.C)
        )
        for i, spec in enumerate(model.sites)
    }
    deltas = (
        {spec.name: jnp.full(tokens.shape, 0.25) for spec in model.sites} if with_deltas else None
    )
    routes = (
        {spec.name: (jnp.arange(tokens.size).reshape(tokens.shape) % 3 > 0) for spec in model.sites}
        if with_routes
        else None
    )
    masking = MaterializedMasking(component_masks=masks, weight_delta_masks=deltas, routes=routes)
    gate_key = site_output_tap_key(site_name(2, "experts_gate"))
    keys = (
        resid_tap_key(cfg.n_layer),
        "mlp_in.2",
        gate_key,
        site_output_tap_key(site_name(cfg.n_layer - 1, "experts_down")),
        site_output_tap_key(site_name(cfg.n_layer // 2, "shared_down")),
    )

    def forward(arm: Qwen36MoeDecomposedModel):
        prepared = arm.prepare_compute_weights(vu, None)
        return arm.masked_forward(
            prepared,
            tokens,
            masking=masking,
            placement=None,
            capture_keys=frozenset(keys),
            remat=False,
        )

    routed, dense = forward(model), forward(_dense_twin(model))
    _assert_values_close(np.asarray(routed.output), np.asarray(dense.output))
    selected = _selected_expert_mask(cfg, model.moe.router[2], routed.captures["mlp_in.2"])
    selected_wide = np.asarray(jnp.repeat(selected, cfg.moe_intermediate, axis=-1))
    for key in keys:
        routed_tap = np.asarray(routed.captures[key])
        dense_tap = np.asarray(dense.captures[key])
        if key == gate_key:
            np.testing.assert_array_equal(routed_tap[~selected_wide], 0.0)
            routed_tap, dense_tap = routed_tap[selected_wide], dense_tap[selected_wide]
        _assert_values_close(routed_tap, dense_tap, err_msg=key)

    def masked_loss(
        arm: Qwen36MoeDecomposedModel,
        value: ComponentStacks,
        embed: jax.Array,
        component: dict[str, jax.Array],
        delta: dict[str, jax.Array] | None,
    ) -> jax.Array:
        arm = eqx.tree_at(lambda m: m.embed, arm, embed)
        prepared = arm.prepare_compute_weights(value, None)
        out = materialized_logits(
            arm.masked_forward(
                prepared,
                tokens,
                masking=MaterializedMasking(
                    component_masks=component, weight_delta_masks=delta, routes=routes
                ),
                placement=None,
                remat=True,
            ).output
        )
        return jnp.sum(jnp.cos(out))

    if deltas is None:

        def loss_without_delta(
            arm: Qwen36MoeDecomposedModel,
            value: ComponentStacks,
            embed: jax.Array,
            component: dict[str, jax.Array],
        ) -> jax.Array:
            return masked_loss(arm, value, embed, component, None)

        routed_grads, dense_grads = _routed_and_dense_grads(
            loss_without_delta, model, vu, model.embed, masks
        )
    else:
        routed_grads, dense_grads = _routed_and_dense_grads(
            masked_loss, model, vu, model.embed, masks, deltas
        )
    _assert_grad_leaves_close(routed_grads, dense_grads)


def test_routed_decomposed_matches_dense_stochastic():
    """Stochastic masking parity: the draws are full-width in BOTH executions, so one
    draw key yields one realization and routed-vs-dense stays a pure reassociation
    delta — outputs and gradients (dV/dU, d-CI, d-input) compare directly."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(22))
    tokens = _tokens(model.cfg)
    ci = {
        spec.name: jax.random.uniform(
            jax.random.fold_in(jax.random.PRNGKey(23), i), (*tokens.shape, spec.C)
        )
        for i, spec in enumerate(model.sites)
    }

    def loss(
        arm: Qwen36MoeDecomposedModel,
        value: ComponentStacks,
        embed: jax.Array,
        ci_lower: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        arm = eqx.tree_at(lambda m: m.embed, arm, embed)
        masking = StochasticMasking(
            ci_stacked=arm.stack_ci(ci_lower), draw_key=jax.random.PRNGKey(24), routes=None
        )
        prepared = arm.prepare_compute_weights(value, None)
        out = materialized_logits(
            arm.masked_forward(prepared, tokens, masking=masking, placement=None, remat=True).output
        )
        return jnp.sum(jnp.cos(out)), out

    grad_fn = jax.jit(jax.grad(loss, argnums=(1, 2, 3), has_aux=True))
    routed_grads, routed_out = grad_fn(model, vu, model.embed, ci)
    dense_grads, dense_out = grad_fn(_dense_twin(model), vu, model.embed, ci)
    _assert_values_close(np.asarray(routed_out), np.asarray(dense_out))
    _assert_grad_leaves_close(routed_grads, dense_grads)


def test_routed_decomposed_matches_dense_with_broadcast_masking():
    """Delta masks and routes may carry size-1 broadcast lead axes (batch-shared
    persistent sources): the routed arm must broadcast them to the full
    lead before its job gathers, with the same gradient (the cross-lead sum) the dense
    arm's elementwise broadcasting yields — the shape batch-shared `sc` sources
    trace."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(28))
    tokens = _tokens(model.cfg)
    masks = {
        spec.name: jax.random.uniform(jax.random.PRNGKey(29), (*tokens.shape, spec.C))
        for spec in model.sites
    }
    deltas = {
        spec.name: jax.random.uniform(jax.random.PRNGKey(30), (1, tokens.shape[1]))
        for spec in model.sites
    }
    routes = {spec.name: (jnp.arange(tokens.shape[1])[None, :] % 2 > 0) for spec in model.sites}

    def loss(
        arm: Qwen36MoeDecomposedModel,
        value: ComponentStacks,
        component: dict[str, jax.Array],
        delta: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        prepared = arm.prepare_compute_weights(value, None)
        out = materialized_logits(
            arm.masked_forward(
                prepared,
                tokens,
                masking=MaterializedMasking(
                    component_masks=component, weight_delta_masks=delta, routes=routes
                ),
                placement=None,
                remat=False,
            ).output
        )
        return jnp.sum(jnp.cos(out)), out

    grad_fn = jax.jit(jax.grad(loss, argnums=(1, 2, 3), has_aux=True))
    routed_grads, routed_out = grad_fn(model, vu, masks, deltas)
    dense_grads, dense_out = grad_fn(_dense_twin(model), vu, masks, deltas)
    for name, leaf in routed_grads[2].items():
        assert leaf.shape == deltas[name].shape, (name, leaf.shape)
    _assert_values_close(np.asarray(routed_out), np.asarray(dense_out))
    _assert_grad_leaves_close(routed_grads, dense_grads)


def test_routed_decomposed_matches_dense_with_undecomposed_expert_kinds():
    """Decomposing a strict subset of the expert kinds (gate + down; up stays frozen)
    runs the frozen grouped matmuls inside the routed decomposed arm — parity against
    the dense oracle's mixed execution."""
    cfg = tiny_qwen36_cfg()
    partial_cs = {k: c for k, c in TINY_QWEN36_CS.items() if k != "experts_up"}
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, partial_cs))
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(25))
    vu = init_component_stacks(sites, jax.random.PRNGKey(26))
    tokens = _tokens(cfg)
    masks = {
        spec.name: jax.random.uniform(jax.random.PRNGKey(27), (*tokens.shape, spec.C))
        for spec in sites
    }
    deltas = {spec.name: jnp.full(tokens.shape, 0.5) for spec in sites}

    def loss(
        arm: Qwen36MoeDecomposedModel,
        value: ComponentStacks,
        component: dict[str, jax.Array],
        delta: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        prepared = arm.prepare_compute_weights(value, None)
        out = materialized_logits(
            arm.masked_forward(
                prepared,
                tokens,
                masking=MaterializedMasking(component_masks=component, weight_delta_masks=delta),
                placement=None,
                remat=False,
            ).output
        )
        return jnp.sum(jnp.cos(out)), out

    grad_fn = jax.jit(jax.grad(loss, argnums=(1, 2, 3), has_aux=True))
    routed_grads, routed_out = grad_fn(model, vu, masks, deltas)
    dense_grads, dense_out = grad_fn(_dense_twin(model), vu, masks, deltas)
    _assert_values_close(np.asarray(routed_out), np.asarray(dense_out))
    _assert_grad_leaves_close(routed_grads, dense_grads)


def test_hidden_acts_reconstruction_points_fail_closed():
    model, _vu = _full_model_and_vu(jax.random.PRNGKey(8))
    model.assert_hidden_acts_reconstruction_points(
        (resid_tap_key(1), site_output_tap_key(site_name(0, "experts_gate")))
    )
    with pytest.raises(AssertionError):
        model.assert_hidden_acts_reconstruction_points((resid_tap_key(0),))


def test_weight_deltas_and_norms_are_slot_aligned():
    model, vu = _full_model_and_vu(jax.random.PRNGKey(9))
    cfg = model.cfg
    deltas = model.weight_deltas(vu)
    norms = model.target_weight_sq_norms()
    assert set(deltas) == set(norms) == set(TINY_QWEN36_CS)
    for kind in TINY_QWEN36_CS:
        match site_contraction(kind):
            case None:
                dims = site_dims(cfg, kind)
                assert deltas[kind].shape == (cfg.n_layer, dims.d_out, dims.d_in)
            case "fused_output":
                assert deltas[kind].shape == (
                    cfg.n_layer,
                    cfg.n_experts,
                    cfg.moe_intermediate,
                    cfg.n_embd,
                )
            case "fused_input":
                assert deltas[kind].shape == (
                    cfg.n_layer,
                    cfg.n_experts,
                    cfg.n_embd,
                    cfg.moe_intermediate,
                )
        assert norms[kind].shape == (cfg.n_layer,)
    name = site_name(3, "shared_gate")
    site = vu.site(name)
    expected = model.moe.shared_gate[3].astype(jnp.float32) - (site.V @ site.U).T
    np.testing.assert_allclose(site_weight_delta(deltas, vu, name), expected, rtol=1e-6)
    gate_name = site_name(2, "experts_gate")
    gate = vu.site(gate_name)  # V [E, d, c], U [E, c, di]
    frozen_blocks = model.moe.experts_gate[2].reshape(
        cfg.n_experts, cfg.moe_intermediate, cfg.n_embd
    )
    expected_blocks = frozen_blocks.astype(jnp.float32) - jnp.einsum(
        "eic,eco->eoi", gate.V.astype(jnp.float32), gate.U.astype(jnp.float32)
    )
    np.testing.assert_allclose(site_weight_delta(deltas, vu, gate_name), expected_blocks, rtol=1e-6)


def test_weight_deltas_convert_after_relayout_bit_for_bit():
    """The frozen stack converts to fp32 only once it holds the blocked layout, so the
    resident bf16 bytes are never doubled whole. The reference here converts the whole
    resident stack FIRST and relayouts the fp32 copy: the convert commutes exactly with
    reshape and transpose, so the two orders agree bit for bit."""
    model, vu = _full_model_and_vu(jax.random.PRNGKey(12))
    cfg = model.cfg
    model = eqx.tree_at(
        lambda m: m.moe, model, jax.tree.map(lambda a: a.astype(jnp.bfloat16), model.moe)
    )
    deltas = model.weight_deltas(vu)
    for kind, (Vs, Us) in vu.stacks.items():
        frozen32 = getattr(model.moe, kind).astype(jnp.float32)
        v32, u32 = Vs.astype(jnp.float32), Us.astype(jnp.float32)
        match site_contraction(kind):
            case None:
                expected = frozen32 - jnp.einsum("gic,gco->goi", v32, u32)
            case "fused_output":
                blocks = frozen32.reshape(
                    cfg.n_layer, cfg.n_experts, cfg.moe_intermediate, cfg.n_embd
                )
                expected = blocks - jnp.einsum("geic,geco->geoi", v32, u32)
            case "fused_input":
                blocks = frozen32.reshape(
                    cfg.n_layer, cfg.n_embd, cfg.n_experts, cfg.moe_intermediate
                ).transpose(0, 2, 1, 3)
                expected = blocks - jnp.einsum("geic,geco->geoi", v32, u32)
        np.testing.assert_array_equal(deltas[kind], expected, err_msg=kind)


def test_partial_layer_coverage_is_refused():
    cfg = tiny_qwen36_cfg()
    partial = tuple(
        SiteC(site_name(layer, kind), TINY_QWEN36_CS[kind])
        for layer in range(cfg.n_layer // 2)
        for kind in KIND_ORDER
    )
    sites = qwen36_moe_site_specs(cfg, partial)
    model = tiny_qwen36_decomposed_model(cfg, sites, jax.random.PRNGKey(10))
    vu = init_component_stacks(sites, jax.random.PRNGKey(11))
    with pytest.raises(AssertionError, match="EVERY layer"):
        model.prepare_compute_weights(vu, None)


def test_family_grammar_round_trips_and_fails_closed():
    for layer in (0, 7, 39):
        for kind in KIND_ORDER:
            assert parse_site_name(site_name(layer, kind)) == (layer, kind)
    for bad in (
        "layers.0.self_attn.q_proj",
        "layers.0.mlp.experts.q_proj",
        "layers.0.mlp.shared_expert_gate",
        "layers.0.mlp.gate_proj",
    ):
        with pytest.raises(AssertionError):
            parse_site_name(bad)
    shuffled = full_site_cs(tiny_qwen36_cfg(), TINY_QWEN36_CS)[::-1]
    assert canonical_site_cs(shuffled) == full_site_cs(tiny_qwen36_cfg(), TINY_QWEN36_CS)


def test_released_qwen36_architecture():
    cfg = qwen36_35b_a3b_config()
    assert (cfg.n_layer, cfg.n_stages, cfg.rotary_dim) == (40, 10, 64)
    assert site_dims(cfg, "experts_gate") == SiteDims(d_in=2048, d_out=131072)
    assert site_dims(cfg, "experts_down") == SiteDims(d_in=131072, d_out=2048)
    assert site_dims(cfg, "shared_up") == SiteDims(d_in=2048, d_out=512)
    assert layer_is_full_attention(cfg, 39) and not layer_is_full_attention(cfg, 38)
    assert sum(layer_is_full_attention(cfg, i) for i in range(cfg.n_layer)) == 10


def test_routing_fold_rounds_the_fused_down_input_once():
    """bf16 gate/up with fp32 routing weights: the fold is the fp32 product rounded to
    bf16 exactly once (weights rounded to bf16 first, or a bf16 product chain, land off
    that value on some rows)."""
    key_gate, key_up, key_weights = jax.random.split(jax.random.PRNGKey(0), 3)
    gate = jax.random.normal(key_gate, (256, 32)).astype(jnp.bfloat16)
    up = jax.random.normal(key_up, (256, 32)).astype(jnp.bfloat16)
    weights = jax.nn.softmax(jax.random.normal(key_weights, (256, 1)), axis=0) * 256

    folded = _fold_routing_weights(gate, up, weights)
    assert folded.dtype == jnp.bfloat16
    exact = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32) * weights
    np.testing.assert_array_equal(np.asarray(folded), np.asarray(exact.astype(jnp.bfloat16)))
