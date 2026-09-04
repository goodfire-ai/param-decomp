"""CPU tests for transport-independent toy figure rendering."""

from typing import Literal

import jax

from param_decomp.core.ci_fn import LayerwiseMLPCIArch, init_layerwise_mlp_ci_fn
from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.configs import (
    KeepAllCheckpoints,
    NoCheckpointing,
    PeriodicCheckpointing,
    UVPlotsConfig,
)
from param_decomp.core.metrics import PNGImage
from param_decomp.core.model import PlacedModel
from param_decomp.experiments import toy_uv_eval
from param_decomp.targets.testing import capture_clean
from param_decomp.targets.tms import (
    TMSConfig,
    init_tms_target,
    single_feature_probe,
    site_input_tap_keys,
    site_specs,
    tms_decomposed_model,
)


def _toy_setup():
    cfg = TMSConfig(n_features=5, n_hidden=2)
    sites = site_specs(cfg, (SiteC("linear1", 8), SiteC("linear2", 6)))
    target = init_tms_target(cfg, jax.random.PRNGKey(3))
    model = tms_decomposed_model(cfg, target, sites)
    ci_fn = init_layerwise_mlp_ci_fn(
        LayerwiseMLPCIArch(
            hidden_dims=(16,),
            has_position_axis=False,
            input_names=site_input_tap_keys(tuple(s.name for s in sites)),
        ),
        sites,
        jax.random.PRNGKey(0),
    )
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    probe = single_feature_probe(cfg.n_features)
    ci = ci_fn(capture_clean(model, probe, ci_fn.capture_keys), remat=False, placement=None)
    return PlacedModel(model=model, placement=None), vu, ci.lower, ci.upper


def test_toy_uv_spec_gates_on_uvplots_in_config():
    model, _, _, _ = _toy_setup()
    assert toy_uv_eval.toy_uv_spec(model, None).want_uv_plots is False
    with_uv = UVPlotsConfig(identity_patterns=None, dense_patterns=None)
    assert toy_uv_eval.toy_uv_spec(model, with_uv).want_uv_plots is True


def test_render_uv_metric_returns_transport_independent_png():
    model, vu, _, probe_upper = _toy_setup()
    spec = toy_uv_eval.toy_uv_spec(
        model, UVPlotsConfig(identity_patterns=None, dense_patterns=None)
    )

    record = toy_uv_eval.render_uv_metric(spec, dict(vu.sites_items()), probe_upper)

    assert set(record) == {"slow_eval/figures/uv_matrices"}
    assert isinstance(record["slow_eval/figures/uv_matrices"], PNGImage)


def test_permuted_ci_heatmap_due_fires_on_save_every_and_final_step():
    periodic = PeriodicCheckpointing(save_every=5000, retention=KeepAllCheckpoints())
    assert toy_uv_eval.permuted_ci_heatmap_due(5000, 20000, periodic) is True
    assert toy_uv_eval.permuted_ci_heatmap_due(5001, 20000, periodic) is False
    assert toy_uv_eval.permuted_ci_heatmap_due(20000, 20000, periodic) is True


def test_permuted_ci_heatmap_due_fires_only_at_the_final_step_without_checkpointing():
    assert toy_uv_eval.permuted_ci_heatmap_due(5000, 20000, NoCheckpointing()) is False
    assert toy_uv_eval.permuted_ci_heatmap_due(20000, 20000, NoCheckpointing()) is True


def test_render_permuted_ci_heatmap_returns_both_leaky_views():
    model, _, ci_lower, ci_upper = _toy_setup()
    permutation: dict[str, Literal["identity", "dense"]] = {
        name: "identity" for name in model.site_names
    }

    record = toy_uv_eval.render_permuted_ci_heatmap(ci_lower, ci_upper, permutation)

    assert set(record) == {
        "slow_eval/figures/causal_importances",
        "slow_eval/figures/causal_importances_upper_leaky",
    }
    assert all(isinstance(value, PNGImage) for value in record.values())


def test_permuted_ci_heatmap_dense_site_permutes_by_mass_not_hungarian():
    model, _, ci_lower, ci_upper = _toy_setup()
    permutation: dict[str, Literal["identity", "dense"]] = {
        name: "dense" for name in model.site_names
    }

    record = toy_uv_eval.render_permuted_ci_heatmap(ci_lower, ci_upper, permutation)

    assert len(record) == 2


def test_toy_figure_owners_emit_disjoint_keys():
    """Each toy figure has exactly one owner: the runner-native ground-truth op renders
    the permuted-CI heatmaps (save cadence) and the authored UVPlots op renders the UV
    matrices (eval cadence). When the two cadences coincide the engine merges both records
    into one step and ASSERTS on key collisions — overlap here would crash real runs."""
    model, vu, ci_lower, ci_upper = _toy_setup()
    permutation: dict[str, Literal["identity", "dense"]] = {
        name: "identity" for name in model.site_names
    }
    spec = toy_uv_eval.toy_uv_spec(
        model, UVPlotsConfig(identity_patterns=None, dense_patterns=None)
    )

    heatmaps = toy_uv_eval.render_permuted_ci_heatmap(ci_lower, ci_upper, permutation)
    uv = toy_uv_eval.render_uv_metric(spec, dict(vu.sites_items()), ci_upper)

    assert not heatmaps.keys() & uv.keys()
