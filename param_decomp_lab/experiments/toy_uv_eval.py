"""Config-gated permutation figures for the positionless toys (TMS / ResidMLP).

Two figure metrics (SPEC S28), both driven off the toy's single-feature probe CI
`(n_features, C)` — the toys have no `(T, C)` position axis, so the probe CI plays the role
the LM's batch-mean position CI does:

- `PermutedCIPlots` — heatmaps the probe CI permuted toward each site's target shape
  (identity / dense). For the toys every site is an identity target, so a well-recovered
  decomposition shows the classic near-identity heatmap (one component per feature).
- `UVPlots` — heatmaps the per-site V/U weights with columns reordered by the same
  permutation.

Toy V/U + probe CI are small / replicated / already on host, so the render is cheap (no
gather). `slow_eval.render_permutation_figures` does the framework-agnostic plotting; this
module is the thin wandb I/O the toy `eval_fn` calls.
"""

import io
from typing import Any

import numpy as np

from param_decomp.lm import DecomposedModel
from param_decomp.slow_eval import (
    PermutationMetricSpec,
    PositionCI,
    render_permutation_figures,
    resolve_permutation_metrics,
)


def toy_permutation_spec(lm: DecomposedModel, raw_cfg: dict[str, Any]) -> PermutationMetricSpec:
    """The permutation spec resolved over the toy's sites from the run config's `eval.metrics`
    (the raw schema dict — the toy's core `ExperimentConfig.eval` is `None`, so the metric list
    is re-validated here, as `slow_eval.eval_metrics_from_run_dir` does for the LM)."""
    from pydantic import TypeAdapter

    from param_decomp.configs import AnyEvalMetricConfig

    raw_metrics = (raw_cfg.get("eval") or {}).get("metrics", [])
    adapter = TypeAdapter(AnyEvalMetricConfig)
    metrics = [adapter.validate_python(m) for m in raw_metrics]
    return resolve_permutation_metrics(lm.site_names, metrics)


def log_permutation_figures(
    spec: PermutationMetricSpec,
    components_vu: dict[str, tuple[Any, Any]],
    probe_ci_lower: dict[str, Any],
    probe_ci_upper: dict[str, Any],
    now_step: int,
    wandb_active: bool,
) -> None:
    """Render the config-gated permutation figures off the toy's on-host probe CI + V/U and log
    them to the live wandb run on the `_step` axis. No-op unless the config names at least one
    plot metric and wandb is on. `PermutedCIPlots` needs only the probe CI (cheap); `UVPlots`
    additionally reorders the on-host V/U pair."""
    if not spec.any_plots or not wandb_active:
        return
    position_ci = {
        name: PositionCI(
            lower=np.asarray(probe_ci_lower[name]), upper=np.asarray(probe_ci_upper[name])
        )
        for name in spec.permutation
    }
    components = (
        {name: (np.asarray(V), np.asarray(U)) for name, (V, U) in components_vu.items()}
        if spec.want_uv_plots
        else None
    )
    figures = render_permutation_figures(spec, position_ci, components)
    if not figures:
        return
    import wandb
    from PIL import Image

    payload = {f"slow_eval/{k}": wandb.Image(Image.open(io.BytesIO(v))) for k, v in figures.items()}
    wandb.log(payload, step=now_step)
