"""CPU tests for the JAX-native slow (plot-type) eval pass.

Pins the reduction semantics against hand-rolled numpy (component activation density and
mean-CI per component are exact under micro-batching), the `pre_sigmoid`-vs-`lower`
distinction, the `n_batches_accum` cap on the histogram sample, and that the renderer
emits valid PNGs under the exact torch `slow_eval/figures/*` keys. Also covers the in-loop
slow tier (SPEC S28/S29): the `slow_every` / `slow_on_first_step` cadence and the rank-0
background `SlowEvalRenderer` logging figures on the live `_step` axis.
"""

import sys
import types
from typing import Any

import jax
import numpy as np
import pytest

from param_decomp.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    CIFn,
    build_ci_fn,
    lower_leaky_hard_sigmoid,
)
from param_decomp.configs import (
    IdentityCIErrorConfig,
    IdentityCITargetSpec,
    PermutedCIPlotsConfig,
    UVPlotsConfig,
)
from param_decomp.lm import DecomposedModel
from param_decomp.run import SlowEvalRenderer, slow_eval_due
from param_decomp.slow_eval import (
    PermutationMetricSpec,
    accumulate_position_ci,
    accumulate_site_reductions,
    compute_identity_ci_errors,
    dense_ci_error,
    identity_ci_error,
    make_position_ci_step,
    make_slow_eval_step,
    permute_to_dense,
    permute_to_identity,
    render_permutation_figures,
    render_slow_eval_figures,
    resolve_permutation_metrics,
)
from param_decomp.targets.llama8b import (
    llama_site_specs,
    mlp_family_site_cs,
)
from param_decomp.tests.test_llama8b import (
    _tiny_cfg,
    _tiny_decomposed_lm,
)


def _build_ci_fn(lm: DecomposedModel, n_embd: int, key: jax.Array) -> CIFn:
    """One transformer chunk over all sites, reading the residual entering the first
    decomposed block. The old `CIArch(16, 1, 2, 32)` dims map onto the chunk arch."""
    site_names = lm.site_names
    first_block = min(int(name.split(".")[1]) for name in site_names)
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=n_embd,
        d_model=16,
        n_blocks=1,
        n_heads=2,
        mlp_hidden=32,
    )
    return build_ci_fn(arch, lm.sites, key)


def _tiny_setup(threshold: float):
    cfg = _tiny_cfg()
    C = 8
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 5, C))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))
    step = make_slow_eval_step(lm, threshold)
    return cfg, lm, ci_fn, step, C


def test_reductions_match_hand_rolled_per_component():
    cfg, lm, ci_fn, step, C = _tiny_setup(threshold=0.0)
    b, t = 3, 16
    residual = jax.random.randint(jax.random.PRNGKey(4), (b, t), 0, cfg.vocab_size)

    reductions = accumulate_site_reductions(step, lm, ci_fn, [residual], n_batches_accum=None)

    taps = lm.read_activations(residual, ci_fn.input_names)
    logits = ci_fn(taps, remat=False).logits
    lower = {s: lower_leaky_hard_sigmoid(logits[s]) for s in lm.site_names}
    for site in lm.site_names:
        flat = np.asarray(lower[site]).reshape(-1, C).astype(np.float32)
        r = reductions[site]
        assert r.n_positions == b * t
        np.testing.assert_allclose(r.density_counts, (flat > 0.0).sum(0), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(r.ci_sums, flat.sum(0), rtol=1e-4, atol=1e-4)


def test_density_threshold_caps_counts_at_n_positions():
    cfg, lm, ci_fn, step, _ = _tiny_setup(threshold=-1.0)  # everything "alive"
    residual = jax.random.randint(jax.random.PRNGKey(7), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, lm, ci_fn, [residual], n_batches_accum=None)
    for r in reductions.values():
        np.testing.assert_array_equal(r.density_counts, np.full_like(r.density_counts, 2 * 16))


def test_cross_batch_sum_accumulates_linearly():
    cfg, lm, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    res_a = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    res_b = jax.random.randint(jax.random.PRNGKey(5), (2, 16), 0, cfg.vocab_size)

    one = accumulate_site_reductions(step, lm, ci_fn, [res_a], None)
    two = accumulate_site_reductions(step, lm, ci_fn, [res_a, res_b], None)
    other = accumulate_site_reductions(step, lm, ci_fn, [res_b], None)
    for site in lm.site_names:
        assert two[site].n_positions == one[site].n_positions + other[site].n_positions
        np.testing.assert_allclose(
            two[site].ci_sums, one[site].ci_sums + other[site].ci_sums, rtol=1e-4, atol=1e-4
        )


def test_n_batches_accum_caps_histogram_sample_only():
    cfg, lm, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    batches = [
        jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(9), i), (2, 16), 0, cfg.vocab_size)
        for i in range(3)
    ]
    capped = accumulate_site_reductions(step, lm, ci_fn, batches, n_batches_accum=1)
    full = accumulate_site_reductions(step, lm, ci_fn, batches, n_batches_accum=None)
    for site in lm.site_names:
        # the cap only limits the histogram raw-value sample; counts/sums span all batches
        assert capped[site].n_positions == full[site].n_positions == 3 * 2 * 16
        assert capped[site].lower_sample.size == 2 * 16 * 8  # one batch
        assert full[site].lower_sample.size == 3 * 2 * 16 * 8


def test_pre_sigmoid_differs_from_lower():
    cfg, lm, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, lm, ci_fn, [residual], None)
    for r in reductions.values():
        # lower is clamped to [0, 1]; logits are unbounded — they cannot be identical
        assert r.lower_sample.min() >= 0.0 and r.lower_sample.max() <= 1.0
        assert not np.allclose(r.lower_sample, r.logits_sample)


def test_render_emits_torch_keyed_pngs():
    cfg, lm, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, lm, ci_fn, [residual], None)
    figures = render_slow_eval_figures(reductions)
    assert set(figures) == {
        "figures/causal_importance_values",
        "figures/causal_importance_values_pre_sigmoid",
        "figures/component_activation_density",
        "figures/ci_mean_per_component",
        "figures/ci_mean_per_component_log",
    }
    for png in figures.values():
        assert png[:4] == b"\x89PNG", "renderer must emit valid PNG bytes"


def test_finite_reductions():
    cfg, lm, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, lm, ci_fn, [residual], None)
    for r in reductions.values():
        assert np.all(np.isfinite(r.density_counts))
        assert np.all(np.isfinite(r.ci_sums))
        assert np.all(np.isfinite(r.lower_sample))
        assert np.all(np.isfinite(r.logits_sample))


# ----------------------------- permutation / identity-error metrics -----------------------------


def test_permute_to_identity_recovers_a_shuffled_identity():
    eye = np.eye(4)
    perm = np.array([2, 0, 3, 1])
    shuffled = eye[:, perm]
    recovered, found = permute_to_identity(shuffled)
    np.testing.assert_allclose(recovered, eye)
    # found[i] is the source column placed at position i; applying it undoes the shuffle
    np.testing.assert_array_equal(shuffled[:, found], eye)


def test_permute_to_identity_handles_wide_matrix():
    # 3 features, 5 components: the extra columns append in order after the assigned ones
    ci = np.zeros((3, 5))
    ci[0, 4] = ci[1, 2] = ci[2, 0] = 1.0
    permuted, perm = permute_to_identity(ci)
    assert perm.shape == (5,)
    np.testing.assert_allclose(np.diagonal(permuted[:3, :3]), 1.0)


def test_permute_to_dense_orders_columns_by_mass():
    ci = np.array([[0.1, 0.9, 0.5], [0.0, 0.8, 0.4]])
    permuted, perm = permute_to_dense(ci)
    assert perm.tolist() == [1, 2, 0]
    assert (permuted.sum(0)[:-1] >= permuted.sum(0)[1:]).all()


def test_identity_ci_error_perfect_and_imperfect():
    perfect = np.eye(5)
    assert identity_ci_error(perfect, tolerance=0.1) == 0
    permuted = perfect[:, np.array([3, 1, 4, 0, 2])]
    assert identity_ci_error(permuted, tolerance=0.1) == 0
    assert identity_ci_error(np.zeros((5, 5)), tolerance=0.1) == 5  # all diagonals missing
    wide = np.concatenate([np.eye(5), np.zeros((5, 3))], axis=1)
    assert identity_ci_error(wide, tolerance=0.1) == 0


def test_identity_ci_error_counts_off_diagonal_leak():
    ci = np.eye(4)
    ci[0, 1] = 0.9  # one off-diagonal leak beyond tolerance
    assert identity_ci_error(ci, tolerance=0.1) == 1


def test_dense_ci_error_perfect_and_missing():
    ci = np.concatenate([np.ones((3, 2)), np.zeros((3, 2))], axis=1)
    assert dense_ci_error(ci, k=2, tolerance=0.1) == 0
    sparse = np.concatenate([np.ones((3, 1)), np.zeros((3, 3))], axis=1)
    assert dense_ci_error(sparse, k=2, tolerance=0.1) == 1  # one of the k columns is dead


def test_resolve_permutation_metrics_dispatches_patterns():
    sites = ("layers.4.mlp.gate_proj", "layers.4.mlp.down_proj", "layers.5.mlp.gate_proj")
    metrics = [
        PermutedCIPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"]),
        UVPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=None),
        IdentityCIErrorConfig(
            identity_ci=[IdentityCITargetSpec(layer_pattern="*gate_proj", n_features=4)],
            dense_ci=None,
        ),
    ]
    spec = resolve_permutation_metrics(sites, metrics)
    assert spec.permutation == {
        "layers.4.mlp.gate_proj": "identity",
        "layers.4.mlp.down_proj": "dense",
        "layers.5.mlp.gate_proj": "identity",
    }
    assert spec.want_uv_plots
    assert spec.any_plots and spec.any_identity_error
    assert set(spec.identity_targets) == {"layers.4.mlp.gate_proj", "layers.5.mlp.gate_proj"}
    assert spec.dense_targets == {}


def test_resolve_permutation_metrics_empty_when_unconfigured():
    spec = resolve_permutation_metrics(("a", "b"), [])
    assert not spec.any_plots and not spec.any_identity_error and not spec.want_uv_plots


def _tiny_position_ci():
    cfg = _tiny_cfg()
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 5, 8))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))
    residual = jax.random.randint(jax.random.PRNGKey(4), (3, 12), 0, cfg.vocab_size)
    position_ci = accumulate_position_ci(make_position_ci_step(lm), lm, ci_fn, [residual])
    return lm, position_ci


def test_position_ci_keeps_position_axis_and_batch_means():
    _, position_ci = _tiny_position_ci()
    for pci in position_ci.values():
        assert pci.lower.shape == (12, 8)  # (T, C), batch axis reduced away
        assert pci.upper.shape == (12, 8)
        assert np.all(np.isfinite(pci.lower)) and np.all(np.isfinite(pci.upper))
        assert pci.lower.min() >= 0.0 and pci.lower.max() <= 1.0  # lower-leaky clamps to [0, 1]


def test_render_permutation_figures_emits_pngs():
    lm, position_ci = _tiny_position_ci()
    metrics = [
        PermutedCIPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"]),
        UVPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"]),
    ]
    spec = resolve_permutation_metrics(lm.site_names, metrics)
    components = {name: (np.zeros((4, 8)), np.zeros((8, 4))) for name in lm.site_names}
    figures = render_permutation_figures(spec, position_ci, components)
    assert set(figures) == {
        "figures/causal_importances",
        "figures/causal_importances_upper_leaky",
        "figures/uv_matrices",
    }
    for png in figures.values():
        assert png[:4] == b"\x89PNG"


def test_render_permutation_figures_empty_without_plot_metrics():
    lm, position_ci = _tiny_position_ci()
    spec = resolve_permutation_metrics(lm.site_names, [])
    assert render_permutation_figures(spec, position_ci, {}) == {}


def test_compute_identity_ci_errors_end_to_end():
    lm, position_ci = _tiny_position_ci()
    spec = resolve_permutation_metrics(
        lm.site_names,
        [
            IdentityCIErrorConfig(
                identity_ci=[IdentityCITargetSpec(layer_pattern="*gate_proj", n_features=2)],
                dense_ci=None,
            )
        ],
    )
    errors = compute_identity_ci_errors(spec, position_ci, tolerance=0.1)
    assert "IdentityCIError" in errors
    gate_keys = [k for k in errors if k.startswith("IdentityCIError/")]
    assert gate_keys and all("gate_proj" in k for k in gate_keys)
    assert errors["IdentityCIError"] == sum(errors[k] for k in gate_keys)
    assert all(v >= 0 for v in errors.values())


def test_compute_identity_ci_errors_empty_when_unconfigured():
    _, position_ci = _tiny_position_ci()
    spec = PermutationMetricSpec({}, {}, {}, want_uv_plots=False)
    assert compute_identity_ci_errors(spec, position_ci, tolerance=0.1) == {}


def test_slow_eval_due_fires_on_cadence_and_first_step():
    # multiples of slow_every fire; non-multiples don't
    assert slow_eval_due(now_step=10000, every=1000, slow_every=10000, slow_on_first_step=False)
    assert not slow_eval_due(now_step=2000, every=1000, slow_every=10000, slow_on_first_step=False)
    assert slow_eval_due(now_step=20000, every=1000, slow_every=10000, slow_on_first_step=False)
    # slow_on_first_step additionally fires at the first eval step (now_step == every)
    assert slow_eval_due(now_step=1000, every=1000, slow_every=10000, slow_on_first_step=True)
    assert not slow_eval_due(now_step=1000, every=1000, slow_every=10000, slow_on_first_step=False)
    # the first eval step is the ONLY extra one slow_on_first_step adds
    assert not slow_eval_due(now_step=2000, every=1000, slow_every=10000, slow_on_first_step=True)


class _FakeWandb(types.ModuleType):
    """Minimal stand-in for the `wandb` module the background renderer imports."""

    class errors(types.ModuleType):  # noqa: N801 — mirrors the real `wandb.errors` submodule
        class CommError(Exception):
            pass

    def __init__(self):
        super().__init__("wandb")
        self.logged: list[tuple[dict[str, Any], int]] = []

    def Image(self, img: Any) -> Any:  # noqa: N802 — mirrors `wandb.Image`
        return img

    def log(self, payload: dict[str, Any], step: int) -> None:
        self.logged.append((payload, step))


def test_renderer_logs_figures_on_live_step_axis(monkeypatch: pytest.MonkeyPatch):
    cfg, lm, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, lm, ci_fn, [residual], None)

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    spec = resolve_permutation_metrics(lm.site_names, [])
    renderer = SlowEvalRenderer(is_main=True)
    renderer.submit(reductions, spec, position_ci=None, components=None, now_step=4242)
    renderer.join()  # flush the background render

    assert len(fake.logged) == 1
    payload, logged_step = fake.logged[0]
    assert logged_step == 4242  # on the live `_step` axis at the eval step
    assert set(payload) == {
        "slow_eval/figures/causal_importance_values",
        "slow_eval/figures/causal_importance_values_pre_sigmoid",
        "slow_eval/figures/component_activation_density",
        "slow_eval/figures/ci_mean_per_component",
        "slow_eval/figures/ci_mean_per_component_log",
    }


def test_in_loop_renderer_includes_permutation_heatmaps_and_uv_when_gathered(
    monkeypatch: pytest.MonkeyPatch,
):
    """The in-loop slow tier renders the CI heatmaps from the materialized position-CI and,
    when the config names UVPlots and the gathered V/U is passed, the UVPlots figure too
    (SPEC S28 amended: in-loop UVPlots is a naive gather, small-scale-only). IdentityCIError
    is computed synchronously on the collective path, not on the background thread."""
    cfg, lm, ci_fn, step, C = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (3, 12), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, lm, ci_fn, [residual], None)
    position_ci = accumulate_position_ci(make_position_ci_step(lm), lm, ci_fn, [residual])

    metrics = [
        PermutedCIPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"]),
        UVPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"]),
        IdentityCIErrorConfig(
            identity_ci=[IdentityCITargetSpec(layer_pattern="*gate_proj", n_features=2)],
            dense_ci=None,
        ),
    ]
    spec = resolve_permutation_metrics(lm.site_names, metrics)
    assert spec.want_uv_plots

    # the IdentityCIError SCALARS are computed synchronously (the in-loop collective path),
    # NOT on the background thread
    errors = compute_identity_ci_errors(spec, position_ci, tolerance=0.1)
    assert "IdentityCIError" in errors and any(k.startswith("IdentityCIError/") for k in errors)

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    components = {name: (np.zeros((4, C)), np.zeros((C, 5))) for name in lm.site_names}
    renderer = SlowEvalRenderer(is_main=True)
    renderer.submit(reductions, spec, position_ci, components, now_step=7000)
    renderer.join()

    assert len(fake.logged) == 1
    payload, logged_step = fake.logged[0]
    assert logged_step == 7000  # figures on the live `_step` axis
    assert "slow_eval/figures/causal_importances" in payload
    assert "slow_eval/figures/causal_importances_upper_leaky" in payload
    assert "slow_eval/figures/uv_matrices" in payload  # gathered V/U -> UVPlots renders
    # no scalar leaks onto the figure (background) payload — scalars ride the sync path
    assert all(k.startswith("slow_eval/figures/") for k in payload)


def test_in_loop_renderer_skips_uv_when_components_not_gathered(
    monkeypatch: pytest.MonkeyPatch,
):
    """When the config does NOT name UVPlots, the trainer gathers no V/U (`components=None`)
    and the UVPlots figure is skipped, while the CI heatmaps still render."""
    cfg, lm, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (3, 12), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, lm, ci_fn, [residual], None)
    position_ci = accumulate_position_ci(make_position_ci_step(lm), lm, ci_fn, [residual])

    spec = resolve_permutation_metrics(
        lm.site_names,
        [PermutedCIPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"])],
    )
    assert not spec.want_uv_plots

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    renderer = SlowEvalRenderer(is_main=True)
    renderer.submit(reductions, spec, position_ci, components=None, now_step=7000)
    renderer.join()

    payload, _ = fake.logged[0]
    assert "slow_eval/figures/causal_importances" in payload
    assert "slow_eval/figures/uv_matrices" not in payload


def test_renderer_noop_off_main_rank(monkeypatch: pytest.MonkeyPatch):
    cfg, lm, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, lm, ci_fn, [residual], None)

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    spec = resolve_permutation_metrics(lm.site_names, [])
    renderer = SlowEvalRenderer(is_main=False)
    renderer.submit(reductions, spec, position_ci=None, components=None, now_step=4242)
    renderer.join()
    assert fake.logged == []  # non-main ranks do the collective pull but never render/log


def test_in_loop_slow_tier_fires_on_cadence_without_stalling(monkeypatch: pytest.MonkeyPatch):
    """Smoke: drive the in-loop slow-tier block (collective accumulate -> background
    render) over a sequence of eval steps and assert figures land ONLY on slow steps, on
    the live `_step` axis, and the main loop never blocks waiting on a render."""
    import time

    cfg, lm, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    spec = resolve_permutation_metrics(lm.site_names, [])
    every, slow_every = 1000, 3000
    renderer = SlowEvalRenderer(is_main=True)
    # Time only the dispatch the loop pays (accumulate + submit), not the off-thread render.
    # Joining between submits, outside the timed window, recreates the real loop's gap of
    # `slow_every` train steps where the render finishes before the next submit (so submit's
    # one-in-flight `join` is a no-op).
    dispatch_s = 0.0
    for now_step in range(every, 10 * every + 1, every):  # 1000, 2000, ..., 10000
        if slow_eval_due(now_step, every, slow_every, slow_on_first_step=True):
            t0 = time.time()
            reductions = accumulate_site_reductions(step, lm, ci_fn, [residual], None)
            renderer.submit(reductions, spec, position_ci=None, components=None, now_step=now_step)
            dispatch_s += time.time() - t0
            renderer.join()
    renderer.join()  # flush

    logged_steps = sorted(s for _, s in fake.logged)
    # slow_on_first_step adds 1000; multiples of 3000 add 3000, 6000, 9000
    assert logged_steps == [1000, 3000, 6000, 9000]
    for payload, _ in fake.logged:
        assert all(k.startswith("slow_eval/figures/") for k in payload)
    # the dispatch loop itself must not block on rendering — accumulate + submit are quick
    assert dispatch_s < 30.0, dispatch_s
