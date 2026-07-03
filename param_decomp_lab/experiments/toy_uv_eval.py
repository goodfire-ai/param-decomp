"""Config-gated `UVPlots` figure for the positionless toys (TMS / ResidMLP).

`UVPlots` is the one slow figure metric usable for any decomposition (SPEC S28): it heatmaps
the per-site V/U matrices with columns reordered by the same identity/dense permutation the
LM CI plots use. The toys have no `(T, C)` position axis, so they drive the column order off
their single-feature probe CI `(n_features, C)` instead. Toy V/U is small / replicated / on
host, so the render is cheap (no gather) — `render_uv_figure` does the framework-agnostic
plot, this module is the thin wandb I/O the toy `eval_fn` calls.
"""

import io
from typing import Any

import numpy as np

from param_decomp.lm import DecomposedModel
from param_decomp.slow_eval import (
    PermutationMetricSpec,
    render_uv_figure,
    resolve_permutation_metrics,
)


def toy_uv_spec(lm: DecomposedModel, raw_cfg: dict[str, Any]) -> PermutationMetricSpec:
    """The permutation spec resolved over the toy's sites from the run config's
    `eval.metrics` (the raw schema dict — the toy's core `ExperimentConfig.eval` is `None`,
    so the metric list is re-validated here, as `slow_eval.eval_metrics_from_run_dir` does
    for the LM). `want_uv_plots` is True only when the config names `UVPlots`; the LM-only
    position-CI / identity metrics are ignored by the toy."""
    from pydantic import TypeAdapter

    from param_decomp.configs import AnyEvalMetricConfig

    raw_metrics = (raw_cfg.get("eval") or {}).get("metrics", [])
    adapter = TypeAdapter(AnyEvalMetricConfig)
    metrics = [adapter.validate_python(m) for m in raw_metrics]
    return resolve_permutation_metrics(lm.site_names, metrics)


def log_uv_figure(
    spec: PermutationMetricSpec,
    components_vu: dict[str, tuple[Any, Any]],
    probe_ci_upper: dict[str, Any],
    now_step: int,
    wandb_active: bool,
) -> None:
    """Render the `UVPlots` figure off the toy's on-host V/U and probe CI, log it to the live
    wandb run on the `_step` axis. No-op unless the config names `UVPlots` and wandb is on.
    `components_vu` / `probe_ci_upper` are the (already host-side) V/U pair and upper-leaky
    probe CI per site."""
    if not spec.want_uv_plots or not wandb_active:
        return
    components = {name: (np.asarray(V), np.asarray(U)) for name, (V, U) in components_vu.items()}
    perm_source = {name: np.asarray(probe_ci_upper[name]) for name in spec.permutation}
    figures = render_uv_figure(spec, components, perm_source)
    if not figures:
        return
    import wandb
    from PIL import Image

    payload = {f"slow_eval/{k}": wandb.Image(Image.open(io.BytesIO(v))) for k, v in figures.items()}
    wandb.log(payload, step=now_step)
