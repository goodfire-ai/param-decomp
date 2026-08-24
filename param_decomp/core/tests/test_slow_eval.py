"""CPU tests for the JAX-native slow (plot-type) eval pass.

Pins the reduction semantics against hand-rolled numpy (component activation density and
mean-CI per component are exact under micro-batching), the `pre_sigmoid`-vs-`lower`
distinction, the value histograms binned on device against what `ax.hist` would draw
(and their refusal of more than one batch), and that the renderer
emits valid PNGs under the exact torch `slow_eval/figures/*` keys. Also covers the in-loop
slow tier (SPEC S28/S29): the `slow_every` / `slow_on_first_step` cadence and the rank-0
background `BackgroundRenderer` logging figures on a deferred semantic step axis.
"""

import base64
import sys
import types
from functools import cache, partial
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    MHACIAttention,
    PlacedCIFn,
    build_ci_fn,
    ci_preactivations,
    lower_leaky_hard_sigmoid,
)
from param_decomp.core.components import SiteC
from param_decomp.core.configs import (
    IdentityCIErrorConfig,
    IdentityCITargetSpec,
    PermutedCIPlotsConfig,
    UVPlotsConfig,
)
from param_decomp.core.eval_schedule import FirstThenEvery, eval_due
from param_decomp.core.model import DecomposedModel, PlacedModel
from param_decomp.core.run import (
    BackgroundRenderer,
    DeferredMediaRecord,
    MetricsSink,
    _combine_step_records,
)
from param_decomp.core.slow_eval import (
    VALUE_HISTOGRAM_N_BINS,
    PermutationMetricSpec,
    SiteReduction,
    ValueHistogram,
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
from param_decomp.targets.glu_transformer import glu_site_specs
from param_decomp.targets.testing import (
    capture_clean,
    tiny_glu_cfg,
    tiny_glu_decomposed_lm,
)


def _slow_eval_media(
    reductions: dict[str, Any],
    perm_spec: PermutationMetricSpec,
    position_ci: dict[str, Any] | None,
    components: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    now_step: int,
) -> DeferredMediaRecord:
    """A slow-tier figure payload, assembled as the operations in
    `experiments/lm/diagnostic_eval_operations.py` assemble theirs — the whole figure set at
    once, so one render exercises every key the deferred axis has to carry."""
    figures = render_slow_eval_figures(reductions)
    if position_ci is not None:
        figures |= render_permutation_figures(perm_spec, position_ci, components)
    return DeferredMediaRecord(
        step_key="slow_eval/figure_step",
        step=now_step,
        media={f"slow_eval/{key}": value for key, value in figures.items()},
    )


def _build_ci_fn(model: DecomposedModel, n_embd: int, key: jax.Array) -> PlacedCIFn:
    """One transformer chunk over all sites, reading the residual entering the first
    decomposed block. The old `CIArch(16, 1, 2, 32)` dims map onto the chunk arch."""
    site_names = model.site_names
    first_block = min(int(name.split(".")[1]) for name in site_names)
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=n_embd,
        d_model=16,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    return PlacedCIFn(fn=build_ci_fn(arch, model.sites, key), placement=None)


_C = 4
_SITE_CS = (SiteC("layers.4.mlp.down_proj", _C), SiteC("layers.5.mlp.gate_proj", _C))
"""Two sites over two layers and two MLP kinds — the smallest set that still exercises
multi-site reduction dicts, cross-layer site names, and both the `*gate_proj` /
`*down_proj` metric patterns. The renderer lays out one subplot per SITE and matplotlib
dominates these tests, so growing this grows every render-bearing test in the file."""


def _histograms(reduction: SiteReduction) -> tuple[ValueHistogram, ValueHistogram]:
    assert reduction.value_histograms is not None
    return reduction.value_histograms


@cache
def _tiny_setup(
    threshold: float,
    density_heatmap_n_bins: int | None = None,
    value_histogram_n_bins: int | None = VALUE_HISTOGRAM_N_BINS,
):
    """Shared across tests: everything returned is frozen (a pydantic config, equinox
    modules, a `filter_jit` over them), so there is nothing for a test to mutate."""
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, _SITE_CS)
    placed = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )
    ci_fn = _build_ci_fn(placed.model, cfg.n_embd, jax.random.PRNGKey(2))
    step = make_slow_eval_step(
        placed,
        ci_fn.fn.capture_keys,
        threshold,
        density_heatmap_n_bins,
        value_histogram_n_bins,
    )
    return cfg, placed, ci_fn, step, _C


def test_reductions_match_hand_rolled_per_component():
    cfg, model, ci_fn, step, C = _tiny_setup(threshold=0.0)
    b, t = 3, 16
    residual = jax.random.randint(jax.random.PRNGKey(4), (b, t), 0, cfg.vocab_size)

    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])

    # Mirror slow_eval_step's training-precision (bf16) readout.
    preactivations = ci_preactivations(
        ci_fn, capture_clean(model.model, residual, ci_fn.fn.capture_keys), remat=False
    )
    lower = {s: lower_leaky_hard_sigmoid(preactivations[s]) for s in model.site_names}
    for site in model.site_names:
        flat = np.asarray(lower[site]).reshape(-1, C).astype(np.float32)
        r = reductions[site]
        assert r.n_positions == b * t
        np.testing.assert_allclose(r.density_counts, (flat > 0.0).sum(0), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(r.ci_sums, flat.sum(0), rtol=1e-4, atol=1e-4)


def test_density_threshold_caps_counts_at_n_positions():
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=-1.0)  # everything "alive"
    residual = jax.random.randint(jax.random.PRNGKey(7), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
    for r in reductions.values():
        np.testing.assert_array_equal(r.density_counts, np.full_like(r.density_counts, 2 * 16))


def test_cross_batch_sum_accumulates_linearly():
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0, value_histogram_n_bins=None)
    res_a = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    res_b = jax.random.randint(jax.random.PRNGKey(5), (2, 16), 0, cfg.vocab_size)

    one = accumulate_site_reductions(step, model, ci_fn, [res_a])
    two = accumulate_site_reductions(step, model, ci_fn, [res_a, res_b])
    other = accumulate_site_reductions(step, model, ci_fn, [res_b])
    for site in model.site_names:
        assert two[site].n_positions == one[site].n_positions + other[site].n_positions
        np.testing.assert_allclose(
            two[site].ci_sums, one[site].ci_sums + other[site].ci_sums, rtol=1e-4, atol=1e-4
        )


def test_value_histograms_refuse_more_than_one_batch():
    """Each batch bins against its own min/max, so counts across batches sit on different
    edges and cannot be summed. Silently keeping the first batch would report a histogram
    over a fraction of the eval data as though it covered all of it."""
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    batches = [
        jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(9), i), (2, 16), 0, cfg.vocab_size)
        for i in range(2)
    ]

    with pytest.raises(AssertionError, match="eval.n_steps=1"):
        accumulate_site_reductions(step, model, ci_fn, batches)


def test_the_device_histogram_is_what_ax_hist_would_have_drawn():
    """Only `(counts, lo, hi)` crosses to the host, so the binning has to be exactly the
    binning matplotlib does over every value — same edges from the data's own min/max, same
    counts. Anything approximate here silently reshapes a figure read for its tails."""
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)

    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])

    preactivations = ci_preactivations(
        ci_fn, capture_clean(model.model, residual, ci_fn.fn.capture_keys), remat=False
    )
    lower = {s: lower_leaky_hard_sigmoid(preactivations[s]) for s in model.site_names}
    for site in model.site_names:
        for values, rendered in zip(
            (lower[site], preactivations[site]), _histograms(reductions[site]), strict=True
        ):
            flat = np.asarray(values).reshape(-1).astype(np.float32)
            expected, edges = np.histogram(flat, bins=VALUE_HISTOGRAM_N_BINS)
            np.testing.assert_array_equal(rendered.counts, expected)
            # numpy lays its edges out in the input's own float32; ours are the float64 linspace
            np.testing.assert_allclose(rendered.edges, edges, rtol=1e-6, atol=1e-6)
            assert rendered.counts.sum() == flat.size


def test_a_metric_wanting_no_histogram_bins_nothing():
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0, value_histogram_n_bins=None)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)

    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])

    assert all(r.value_histograms is None for r in reductions.values())
    assert all(r.density_counts.size and r.ci_sums.size for r in reductions.values())
    figures = render_slow_eval_figures(reductions)
    assert "figures/causal_importance_values" not in figures
    assert "figures/causal_importance_values_pre_sigmoid" not in figures
    assert "figures/component_activation_density" in figures


def test_pre_sigmoid_differs_from_lower():
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
    for r in reductions.values():
        # lower is clamped to [0, 1]; preactivations are unbounded — they cannot be identical
        lower, preactivations = _histograms(r)
        assert lower.lo >= 0.0 and lower.hi <= 1.0
        assert (preactivations.lo, preactivations.hi) != (lower.lo, lower.hi)


def test_render_emits_torch_keyed_pngs():
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
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
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
    for r in reductions.values():
        assert np.all(np.isfinite(r.density_counts))
        assert np.all(np.isfinite(r.ci_sums))
        assert all(np.isfinite((h.lo, h.hi)).all() for h in _histograms(r))


def test_density_hist_disabled_by_default():
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
    for r in reductions.values():
        assert r.density_hist is None


def test_density_hist_shape_and_conservation():
    n_bins = 40
    cfg, model, ci_fn, step, C = _tiny_setup(threshold=0.0, density_heatmap_n_bins=n_bins)
    residual = jax.random.randint(jax.random.PRNGKey(4), (3, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
    for r in reductions.values():
        assert r.density_hist is not None
        assert r.density_hist.shape == (C, n_bins + 1)
        # every (token, component) pair lands in exactly one bin (underflow col + n_bins bands)
        np.testing.assert_array_equal(r.density_hist.sum(1), np.full(C, r.n_positions))
        # column 0 = underflow (CI < 1e-9), which contains at least every exact-0 inactive token
        assert (r.density_hist[:, 0] >= r.n_positions - r.density_counts).all()


def test_density_hist_accumulates_over_all_batches():
    n_bins = 40
    cfg, model, ci_fn, step, _ = _tiny_setup(
        threshold=0.0, density_heatmap_n_bins=n_bins, value_histogram_n_bins=None
    )
    batches = [
        jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(9), i), (2, 16), 0, cfg.vocab_size)
        for i in range(3)
    ]
    accumulated = accumulate_site_reductions(step, model, ci_fn, batches)
    for r in accumulated.values():
        assert r.density_hist is not None
        assert r.density_hist.sum(1)[0] == 3 * 2 * 16  # all three batches


def test_render_includes_density_heatmap_when_enabled():
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0, density_heatmap_n_bins=40)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
    figures = render_slow_eval_figures(reductions)
    assert "figures/ci_density_heatmap" in figures
    assert figures["figures/ci_density_heatmap"][:4] == b"\x89PNG"


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


@cache
def _position_ci_step():
    """One trace for the file: `_SITE_CS` fixes the only model these tests use."""
    _, model, ci_fn, _, _ = _tiny_setup(threshold=0.0)
    return make_position_ci_step(model, ci_fn.fn.capture_keys)


@cache
def _tiny_position_ci():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, _SITE_CS)
    placed = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )
    ci_fn = _build_ci_fn(placed.model, cfg.n_embd, jax.random.PRNGKey(2))
    residual = jax.random.randint(jax.random.PRNGKey(4), (3, 12), 0, cfg.vocab_size)
    position_ci = accumulate_position_ci(_position_ci_step(), placed, ci_fn, [residual])
    return placed, position_ci


def test_position_ci_keeps_position_axis_and_batch_means():
    _, position_ci = _tiny_position_ci()
    for pci in position_ci.values():
        assert pci.lower.shape == (12, _C)  # (T, C), batch axis reduced away
        assert pci.upper.shape == (12, _C)
        assert np.all(np.isfinite(pci.lower)) and np.all(np.isfinite(pci.upper))
        assert pci.lower.min() >= 0.0 and pci.lower.max() <= 1.0  # lower-leaky clamps to [0, 1]


def test_render_permutation_figures_emits_pngs():
    model, position_ci = _tiny_position_ci()
    metrics = [
        PermutedCIPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"]),
        UVPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"]),
    ]
    spec = resolve_permutation_metrics(model.site_names, metrics)
    components = {name: (np.zeros((4, _C)), np.zeros((_C, 4))) for name in model.site_names}
    figures = render_permutation_figures(spec, position_ci, components)
    assert set(figures) == {
        "figures/causal_importances",
        "figures/causal_importances_upper_leaky",
        "figures/uv_matrices",
    }
    for png in figures.values():
        assert png[:4] == b"\x89PNG"


def test_render_permutation_figures_empty_without_plot_metrics():
    model, position_ci = _tiny_position_ci()
    spec = resolve_permutation_metrics(model.site_names, [])
    assert render_permutation_figures(spec, position_ci, {}) == {}


def test_compute_identity_ci_errors_end_to_end():
    model, position_ci = _tiny_position_ci()
    spec = resolve_permutation_metrics(
        model.site_names,
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


def test_train_and_eval_share_one_transport_record_per_model_step():
    combined = _combine_step_records(
        {"train/loss/total": 1.0},
        {"eval/loss/PGDReconLoss_20step": 0.25},
    )
    assert combined == {
        "train/loss/total": 1.0,
        "eval/loss/PGDReconLoss_20step": 0.25,
    }


def test_train_and_eval_key_collision_fails_closed():
    with pytest.raises(AssertionError, match="colliding keys"):
        _combine_step_records({"same": 1.0}, {"same": 2.0})


class _FakeWandb(types.ModuleType):
    """Minimal stand-in for the `wandb` module the background renderer imports."""

    class errors(types.ModuleType):  # noqa: N801 — mirrors the real `wandb.errors` submodule
        class CommError(Exception):
            pass

    def __init__(self):
        super().__init__("wandb")
        self.logged: list[tuple[dict[str, Any], int | None, bool | None]] = []
        self.dropped: list[dict[str, Any]] = []
        self._step = 0
        self.defined_metrics: list[tuple[str, str | None]] = []

    def define_metric(self, name: str, step_metric: str | None = None) -> None:
        self.defined_metrics.append((name, step_metric))

    def Image(self, img: Any) -> Any:  # noqa: N802 — mirrors `wandb.Image`
        return img

    def log(
        self,
        payload: dict[str, Any],
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        if step is not None and step < self._step:
            self.dropped.append(payload)
            return
        self.logged.append((payload, step, commit))
        if commit:
            self._step = (step if step is not None else self._step) + 1


def test_renderer_logs_figures_on_deferred_semantic_step_axis(monkeypatch: pytest.MonkeyPatch):
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    spec = resolve_permutation_metrics(model.site_names, [])
    renderer = BackgroundRenderer(MetricsSink(None, fake))
    renderer.submit(partial(_slow_eval_media, reductions, spec, None, None, 4242))
    renderer.join()  # flush the background render

    assert fake.defined_metrics == [
        (key, "slow_eval/figure_step")
        for key in (
            "slow_eval/figures/causal_importance_values",
            "slow_eval/figures/causal_importance_values_pre_sigmoid",
            "slow_eval/figures/component_activation_density",
            "slow_eval/figures/ci_mean_per_component",
            "slow_eval/figures/ci_mean_per_component_log",
        )
    ]
    assert len(fake.logged) == 1
    payload, logged_step, commit = fake.logged[0]
    assert logged_step is None  # never attempts an out-of-order W&B `_step`
    assert commit is False  # deferred media must not advance W&B before same-step scalars
    assert payload["slow_eval/figure_step"] == 4242.0
    assert set(payload) == {
        "slow_eval/figure_step",
        "slow_eval/figures/causal_importance_values",
        "slow_eval/figures/causal_importance_values_pre_sigmoid",
        "slow_eval/figures/component_activation_density",
        "slow_eval/figures/ci_mean_per_component",
        "slow_eval/figures/ci_mean_per_component_log",
    }


def test_metrics_sink_rejects_a_second_committed_record_at_the_same_step(tmp_path: Path):
    sink = MetricsSink((tmp_path / "metrics.jsonl").open("a"), None)
    sink.log(100, {"train/loss/total": 1.0})

    with pytest.raises(AssertionError, match="metrics steps must be strictly increasing"):
        sink.log(100, {"eval/loss/PGDReconLoss_20step": 0.25})


def test_deferred_media_cannot_advance_wandb_past_same_step_scalars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A completed background render may land between eval operations and scalar logging.

    Deferred media stays in W&B's pending row (`commit=False`), so the synchronous eval
    record at the same explicit step remains monotonic and commits both records.
    """
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
    spec = resolve_permutation_metrics(model.site_names, [])

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)
    sink = MetricsSink((tmp_path / "metrics.jsonl").open("a"), fake)
    renderer = BackgroundRenderer(sink)

    renderer.submit(partial(_slow_eval_media, reductions, spec, None, None, 100))
    renderer.join()
    sink.log(100, {"eval/loss/PGDReconLoss_20step": 0.25})

    assert fake.dropped == []
    assert fake.logged[-1] == (
        {"eval/loss/PGDReconLoss_20step": 0.25},
        100,
        True,
    )


def test_in_loop_renderer_includes_permutation_heatmaps_and_uv_when_gathered(
    monkeypatch: pytest.MonkeyPatch,
):
    """The in-loop slow tier renders the CI heatmaps from the materialized position-CI and,
    when the config names UVPlots and the gathered V/U is passed, the UVPlots figure too
    (SPEC S28 amended: in-loop UVPlots is a naive gather, small-scale-only). IdentityCIError
    is computed synchronously on the collective path, not on the background thread."""
    cfg, model, ci_fn, step, C = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (3, 12), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
    position_ci = accumulate_position_ci(_position_ci_step(), model, ci_fn, [residual])

    metrics = [
        PermutedCIPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"]),
        UVPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"]),
        IdentityCIErrorConfig(
            identity_ci=[IdentityCITargetSpec(layer_pattern="*gate_proj", n_features=2)],
            dense_ci=None,
        ),
    ]
    spec = resolve_permutation_metrics(model.site_names, metrics)
    assert spec.want_uv_plots

    # the IdentityCIError SCALARS are computed synchronously (the in-loop collective path),
    # NOT on the background thread
    errors = compute_identity_ci_errors(spec, position_ci, tolerance=0.1)
    assert "IdentityCIError" in errors and any(k.startswith("IdentityCIError/") for k in errors)

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    components = {name: (np.zeros((4, C)), np.zeros((C, 5))) for name in model.site_names}
    renderer = BackgroundRenderer(MetricsSink(None, fake))
    renderer.submit(partial(_slow_eval_media, reductions, spec, position_ci, components, 7000))
    renderer.join()

    assert len(fake.logged) == 1
    payload, logged_step, commit = fake.logged[0]
    assert logged_step is None
    assert commit is False
    assert payload["slow_eval/figure_step"] == 7000.0
    assert "slow_eval/figures/causal_importances" in payload
    assert "slow_eval/figures/causal_importances_upper_leaky" in payload
    assert "slow_eval/figures/uv_matrices" in payload  # gathered V/U -> UVPlots renders
    # No scientific scalar leaks onto the background payload; only the semantic step axis
    # accompanies the figures. Scalar eval remains synchronous on W&B `_step`.
    assert all(k == "slow_eval/figure_step" or k.startswith("slow_eval/figures/") for k in payload)


def test_in_loop_renderer_skips_uv_when_components_not_gathered(
    monkeypatch: pytest.MonkeyPatch,
):
    """When the config does NOT name UVPlots, the trainer gathers no V/U (`components=None`)
    and the UVPlots figure is skipped, while the CI heatmaps still render."""
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (3, 12), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
    position_ci = accumulate_position_ci(_position_ci_step(), model, ci_fn, [residual])

    spec = resolve_permutation_metrics(
        model.site_names,
        [PermutedCIPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"])],
    )
    assert not spec.want_uv_plots

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    renderer = BackgroundRenderer(MetricsSink(None, fake))
    renderer.submit(partial(_slow_eval_media, reductions, spec, position_ci, None, 7000))
    renderer.join()

    payload, _, _ = fake.logged[0]
    assert "slow_eval/figures/causal_importances" in payload
    assert "slow_eval/figures/uv_matrices" not in payload


def test_renderer_noop_off_main_rank(monkeypatch: pytest.MonkeyPatch):
    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    spec = resolve_permutation_metrics(model.site_names, [])
    renderer = BackgroundRenderer(MetricsSink.silent())
    renderer.submit(partial(_slow_eval_media, reductions, spec, None, None, 4242))
    renderer.join()
    assert fake.logged == []  # non-main ranks do the collective pull but never render/log


def test_figure_rendering_never_enters_the_pyplot_registry():
    """Figures are rendered on `BackgroundRenderer`'s worker thread, where pyplot's global
    figure registry is unsynchronized across the concurrent figure tiers and an interactive
    backend cannot build a figure manager at all. An empty registry after a full render is
    the portable, backend-independent evidence that the renderers use the OO `Figure` API."""
    from matplotlib import pyplot as plt

    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
    position_ci = accumulate_position_ci(
        make_position_ci_step(model, ci_fn.fn.capture_keys),
        model,
        ci_fn,
        [residual],
    )
    spec = resolve_permutation_metrics(
        model.site_names,
        [PermutedCIPlotsConfig(identity_patterns=["*gate_proj"], dense_patterns=["*down_proj"])],
    )

    assert plt.get_fignums() == []
    _slow_eval_media(reductions, spec, position_ci, None, 4242)
    assert plt.get_fignums() == []


def test_renderer_surfaces_a_failed_render_on_join(monkeypatch: pytest.MonkeyPatch):
    """A render that dies must not leave the tier quietly figure-less for the rest of the run."""
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)
    renderer = BackgroundRenderer(MetricsSink(None, fake))

    def failing_render() -> DeferredMediaRecord:
        raise ValueError("render exploded")

    renderer.submit(failing_render)
    with pytest.raises(RuntimeError, match="background figure render failed") as raised:
        renderer.join()
    assert isinstance(raised.value.__cause__, ValueError)
    assert fake.logged == []
    renderer.join()  # the failure is reported once, not latched into every later join


def test_in_loop_slow_tier_fires_on_cadence_without_stalling(monkeypatch: pytest.MonkeyPatch):
    """Smoke: drive the in-loop slow-tier block (collective accumulate -> background
    render) over a sequence of eval steps and assert figures land ONLY on slow steps, carry
    their semantic eval step, and never block the main loop waiting on a render."""
    import time

    cfg, model, ci_fn, step, _ = _tiny_setup(threshold=0.0)
    residual = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)

    spec = resolve_permutation_metrics(model.site_names, [])
    every, slow_every = 1000, 3000
    renderer = BackgroundRenderer(MetricsSink(None, fake))
    # Time only the dispatch the loop pays (accumulate + submit), not the off-thread render.
    # Joining between submits, outside the timed window, recreates the real loop's gap of
    # `slow_every` train steps where the render finishes before the next submit (so submit's
    # one-in-flight `join` is a no-op).
    dispatch_s = 0.0
    for now_step in range(every, 10 * every + 1, every):  # 1000, 2000, ..., 10000
        if eval_due(FirstThenEvery(every, slow_every), now_step):
            t0 = time.time()
            reductions = accumulate_site_reductions(step, model, ci_fn, [residual])
            renderer.submit(partial(_slow_eval_media, reductions, spec, None, None, now_step))
            dispatch_s += time.time() - t0
            renderer.join()
    renderer.join()  # flush

    assert all(step is None and commit is False for _, step, commit in fake.logged)
    # the schedule's explicit `first` adds 1000; multiples of 3000 add 3000, 6000, 9000
    assert sorted(payload["slow_eval/figure_step"] for payload, _, _ in fake.logged) == [
        1000.0,
        3000.0,
        6000.0,
        9000.0,
    ]
    for payload, _, _ in fake.logged:
        assert all(
            k == "slow_eval/figure_step" or k.startswith("slow_eval/figures/") for k in payload
        )
    # the dispatch loop itself must not block on rendering — accumulate + submit are quick
    assert dispatch_s < 30.0, dispatch_s


def test_deferred_media_rejects_duplicate_semantic_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "wandb.errors", fake.errors)
    sink = MetricsSink((tmp_path / "metrics.jsonl").open("a"), fake)
    encoded = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    record = DeferredMediaRecord(
        step_key="slow_eval/figure_step",
        step=100,
        media={"slow_eval/figures/uv_matrices": encoded},
    )
    sink.log_deferred_media(record)

    with pytest.raises(AssertionError, match="colliding semantic keys"):
        sink.log_deferred_media(record)
