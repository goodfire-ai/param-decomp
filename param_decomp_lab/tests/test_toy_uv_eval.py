"""CPU tests for the config-gated toy permutation figures (`toy_uv_eval`).

The toys feed their single-feature probe CI `(n_features, C)` + small on-host V/U to the shared
`slow_eval.render_permutation_figures`, so a toy config that names `PermutedCIPlots` produces
the CI identity heatmap and one that names `UVPlots` also produces the V/U heatmap (both logged
to the live wandb run); a config that names neither is a no-op. This pins the toy-side wiring +
the config gate, not the plot code.
"""

import sys
import types
from typing import Any

import jax
import pytest

from param_decomp.ci_fn import MLPCIArch, init_layerwise_mlp_ci_fn
from param_decomp.components import SiteC, init_decomp_vu
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
    lm = tms_decomposed_model(cfg, target, sites)
    ci_fn = init_layerwise_mlp_ci_fn(MLPCIArch(hidden_dims=(16,)), sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    probe = single_feature_probe(cfg.n_features)
    ci = ci_fn(lm.read_activations(probe, ci_fn.input_names), remat=False)
    return lm, vu, ci.lower, ci.upper


def _raw(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {"eval": {"metrics": metrics}}


def _permuted_ci(patterns: list[str] | None = None) -> dict[str, Any]:
    return {"type": "PermutedCIPlots", "identity_patterns": patterns, "dense_patterns": None}


def _uv_plots(patterns: list[str] | None = None) -> dict[str, Any]:
    return {"type": "UVPlots", "identity_patterns": patterns, "dense_patterns": None}


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.logged: list[tuple[dict[str, Any], int]] = []

    def Image(self, img: Any) -> Any:  # noqa: N802 — mirrors `wandb.Image`
        return img

    def log(self, payload: dict[str, Any], step: int) -> None:
        self.logged.append((payload, step))


def test_toy_permutation_spec_gates_on_config():
    lm, *_ = _toy_setup()
    assert toy_uv_eval.toy_permutation_spec(lm, {}).any_plots is False
    assert toy_uv_eval.toy_permutation_spec(lm, _raw([])).any_plots is False
    ci_only = toy_uv_eval.toy_permutation_spec(lm, _raw([_permuted_ci()]))
    assert ci_only.any_plots is True and ci_only.want_uv_plots is False
    with_uv = toy_uv_eval.toy_permutation_spec(lm, _raw([_uv_plots()]))
    assert with_uv.want_uv_plots is True


def test_permuted_ci_plots_renders_identity_heatmaps(monkeypatch: pytest.MonkeyPatch):
    lm, vu, lower, upper = _toy_setup()
    spec = toy_uv_eval.toy_permutation_spec(lm, _raw([_permuted_ci(["*"])]))
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    toy_uv_eval.log_permutation_figures(spec, vu.vu, lower, upper, now_step=42, wandb_active=True)

    assert len(fake.logged) == 1
    payload, step = fake.logged[0]
    assert step == 42
    assert set(payload) == {
        "slow_eval/figures/causal_importances",
        "slow_eval/figures/causal_importances_upper_leaky",
    }


def test_uvplots_adds_uv_matrices(monkeypatch: pytest.MonkeyPatch):
    lm, vu, lower, upper = _toy_setup()
    spec = toy_uv_eval.toy_permutation_spec(lm, _raw([_uv_plots(["*"])]))
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    toy_uv_eval.log_permutation_figures(spec, vu.vu, lower, upper, now_step=7, wandb_active=True)

    payload, _ = fake.logged[0]
    assert "slow_eval/figures/uv_matrices" in payload


def test_noop_when_unconfigured_or_wandb_off(monkeypatch: pytest.MonkeyPatch):
    lm, vu, lower, upper = _toy_setup()
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    no_plots = toy_uv_eval.toy_permutation_spec(lm, _raw([]))
    toy_uv_eval.log_permutation_figures(no_plots, vu.vu, lower, upper, 42, wandb_active=True)
    assert fake.logged == []

    configured = toy_uv_eval.toy_permutation_spec(lm, _raw([_permuted_ci(["*"])]))
    toy_uv_eval.log_permutation_figures(configured, vu.vu, lower, upper, 42, wandb_active=False)
    assert fake.logged == []
