"""CPU tests for the config-gated toy `UVPlots` figure (`toy_uv_eval`).

The toys feed `UVPlots` their probe CI `(n_features, C)` as the permutation source and their
small on-host V/U — so a toy config that names `UVPlots` produces a V/U-heatmap figure
(logged to the live wandb run), and one that does not is a no-op. The plot code itself is the
shared `slow_eval.render_uv_figure`, so this only pins the toy-side wiring + the config gate.
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
    probe_upper = ci_fn(lm.read_activations(probe, ci_fn.input_names)).upper
    return lm, vu, probe_upper


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
    lm, _, _ = _toy_setup()
    assert toy_uv_eval.toy_uv_spec(lm, {}).want_uv_plots is False
    assert toy_uv_eval.toy_uv_spec(lm, _raw([])).want_uv_plots is False
    no_uv = _raw([{"type": "PermutedCIPlots", "identity_patterns": None, "dense_patterns": None}])
    assert toy_uv_eval.toy_uv_spec(lm, no_uv).want_uv_plots is False
    with_uv = _raw([{"type": "UVPlots", "identity_patterns": None, "dense_patterns": None}])
    assert toy_uv_eval.toy_uv_spec(lm, with_uv).want_uv_plots is True


def test_log_uv_figure_renders_png_when_configured(monkeypatch: pytest.MonkeyPatch):
    lm, vu, probe_upper = _toy_setup()
    spec = toy_uv_eval.toy_uv_spec(
        lm, _raw([{"type": "UVPlots", "identity_patterns": None, "dense_patterns": None}])
    )
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    toy_uv_eval.log_uv_figure(spec, vu.vu, probe_upper, now_step=42, wandb_active=True)

    assert len(fake.logged) == 1
    payload, step = fake.logged[0]
    assert step == 42
    assert set(payload) == {"slow_eval/figures/uv_matrices"}


def test_log_uv_figure_noop_when_unconfigured_or_wandb_off(monkeypatch: pytest.MonkeyPatch):
    lm, vu, probe_upper = _toy_setup()
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    # config does not name UVPlots -> no-op
    no_uv = toy_uv_eval.toy_uv_spec(lm, _raw([]))
    toy_uv_eval.log_uv_figure(no_uv, vu.vu, probe_upper, now_step=42, wandb_active=True)
    assert fake.logged == []

    # configured but wandb off -> no-op
    with_uv = toy_uv_eval.toy_uv_spec(
        lm, _raw([{"type": "UVPlots", "identity_patterns": None, "dense_patterns": None}])
    )
    toy_uv_eval.log_uv_figure(with_uv, vu.vu, probe_upper, now_step=42, wandb_active=False)
    assert fake.logged == []
