"""CPU tests for the toy figure metrics (`toy_uv_eval`): the config-gated `UVPlots` V/U
heatmap and the unconditional identity-permuted CI heatmap.

The toys feed `UVPlots` their probe CI `(n_features, C)` as the permutation source and their
small on-host V/U — so a toy config that names `UVPlots` produces a V/U-heatmap figure
(logged to the live wandb run), and one that does not is a no-op. The plot code itself is the
shared `slow_eval.render_uv_figure`, so this only pins the toy-side wiring + the config gate.
"""

import sys
import types
from typing import Any, Literal

import jax
import pytest

from param_decomp.ci_fn import MLPCIArch, init_layerwise_mlp_ci_fn
from param_decomp.components import SiteC, init_component_stacks
from param_decomp_lab.experiments import toy_uv_eval
from param_decomp_lab.experiments.tms.model import (
    TMSConfig,
    init_tms_target,
    single_feature_probe,
    site_specs,
    tms_decomposed_model,
)


def _toy_setup():
    cfg = TMSConfig(n_features=5, n_hidden=2)
    sites = site_specs(cfg, (SiteC("linear1", 8), SiteC("linear2", 6)))
    target = init_tms_target(cfg, jax.random.PRNGKey(3))
    model = tms_decomposed_model(cfg, target, sites)
    ci_fn = init_layerwise_mlp_ci_fn(MLPCIArch(hidden_dims=(16,)), sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    probe = single_feature_probe(cfg.n_features)
    ci = ci_fn(model.read_activations(probe, ci_fn.input_names), remat=False)
    return model, vu, ci.lower, ci.upper


def _raw(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {"eval": {"metrics": metrics}}


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.logged: list[tuple[dict[str, Any], int]] = []

    def Image(self, img: Any) -> Any:  # noqa: N802 — mirrors `wandb.Image`
        return img

    def log(self, payload: dict[str, Any], step: int) -> None:
        self.logged.append((payload, step))


def test_toy_uv_spec_gates_on_uvplots_in_config():
    model, _, _, _ = _toy_setup()
    assert toy_uv_eval.toy_uv_spec(model, {}).want_uv_plots is False
    assert toy_uv_eval.toy_uv_spec(model, _raw([])).want_uv_plots is False
    no_uv = _raw([{"type": "PermutedCIPlots", "identity_patterns": None, "dense_patterns": None}])
    assert toy_uv_eval.toy_uv_spec(model, no_uv).want_uv_plots is False
    with_uv = _raw([{"type": "UVPlots", "identity_patterns": None, "dense_patterns": None}])
    assert toy_uv_eval.toy_uv_spec(model, with_uv).want_uv_plots is True


def test_log_uv_figure_renders_png_when_configured(monkeypatch: pytest.MonkeyPatch):
    model, vu, _, probe_upper = _toy_setup()
    spec = toy_uv_eval.toy_uv_spec(
        model, _raw([{"type": "UVPlots", "identity_patterns": None, "dense_patterns": None}])
    )
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    toy_uv_eval.log_uv_figure(
        spec, dict(vu.sites_items()), probe_upper, now_step=42, wandb_active=True
    )

    assert len(fake.logged) == 1
    payload, step = fake.logged[0]
    assert step == 42
    assert set(payload) == {"slow_eval/figures/uv_matrices"}


def test_log_uv_figure_noop_when_unconfigured_or_wandb_off(monkeypatch: pytest.MonkeyPatch):
    model, vu, _, probe_upper = _toy_setup()
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    # config does not name UVPlots -> no-op
    no_uv = toy_uv_eval.toy_uv_spec(model, _raw([]))
    toy_uv_eval.log_uv_figure(
        no_uv, dict(vu.sites_items()), probe_upper, now_step=42, wandb_active=True
    )
    assert fake.logged == []

    # configured but wandb off -> no-op
    with_uv = toy_uv_eval.toy_uv_spec(
        model, _raw([{"type": "UVPlots", "identity_patterns": None, "dense_patterns": None}])
    )
    toy_uv_eval.log_uv_figure(
        with_uv, dict(vu.sites_items()), probe_upper, now_step=42, wandb_active=False
    )
    assert fake.logged == []


def test_permuted_ci_heatmap_due_fires_on_save_every_and_final_step():
    assert toy_uv_eval.permuted_ci_heatmap_due(5000, 20000, save_every=5000) is True
    assert toy_uv_eval.permuted_ci_heatmap_due(5001, 20000, save_every=5000) is False
    assert toy_uv_eval.permuted_ci_heatmap_due(20000, 20000, save_every=5000) is True
    # save_every unset -> only the final step fires
    assert toy_uv_eval.permuted_ci_heatmap_due(20000, 20000, save_every=None) is True
    assert toy_uv_eval.permuted_ci_heatmap_due(10000, 20000, save_every=None) is False


def test_log_permuted_ci_heatmap_renders_both_leaky_views(monkeypatch: pytest.MonkeyPatch):
    model, _, ci_lower, ci_upper = _toy_setup()
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    permutation: dict[str, Literal["identity", "dense"]] = {
        name: "identity" for name in model.site_names
    }

    toy_uv_eval.log_permuted_ci_heatmap(
        ci_lower, ci_upper, permutation, now_step=5000, wandb_active=True
    )

    assert len(fake.logged) == 1
    payload, step = fake.logged[0]
    assert step == 5000
    assert set(payload) == {
        "slow_eval/figures/causal_importances",
        "slow_eval/figures/causal_importances_upper_leaky",
    }


def test_log_permuted_ci_heatmap_noop_when_wandb_off(monkeypatch: pytest.MonkeyPatch):
    model, _, ci_lower, ci_upper = _toy_setup()
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    permutation: dict[str, Literal["identity", "dense"]] = {
        name: "identity" for name in model.site_names
    }

    toy_uv_eval.log_permuted_ci_heatmap(
        ci_lower, ci_upper, permutation, now_step=5000, wandb_active=False
    )

    assert fake.logged == []


def test_log_permuted_ci_heatmap_dense_site_permutes_by_mass_not_hungarian(
    monkeypatch: pytest.MonkeyPatch,
):
    model, _, ci_lower, ci_upper = _toy_setup()
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    permutation: dict[str, Literal["identity", "dense"]] = {
        name: "dense" for name in model.site_names
    }

    # A dense-target site should render without error under the dense (column-mass) sort —
    # it must not silently fall back to the identity/Hungarian permutation.
    toy_uv_eval.log_permuted_ci_heatmap(
        ci_lower, ci_upper, permutation, now_step=5000, wandb_active=True
    )
    assert len(fake.logged) == 1
