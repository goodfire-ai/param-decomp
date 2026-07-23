"""Figure metrics for the positionless toys (TMS / ResidMLP): the config-gated `UVPlots`
V/U heatmap, and the unconditional identity-permuted CI heatmap.

`UVPlots` is the one slow figure metric usable for any decomposition (SPEC S28): it heatmaps
the per-site V/U matrices with columns reordered by the same identity/dense permutation the
LM CI plots use. The toys have no `(T, C)` position axis, so they drive the column order off
their single-feature probe CI `(n_features, C)` instead. Toy V/U is small / replicated / on
host, so the render is cheap (no gather) — `render_uv_figure` does the framework-agnostic
plot, this module is the thin wandb I/O the toy `eval_fn` calls.

The permuted-CI heatmap (`log_permuted_ci_heatmap`) is the visual companion to the
`eval/identity_ci_error/<site>` scalar every toy already logs unconditionally: same
per-site target permutation (identity via Hungarian assignment, dense via column-mass
sort), rendered as an image instead of a discrete distance, so a recovered decomposition
is visibly the EXPECTED pattern rather than just a small number — for sites whose target
is dense (TMS's frozen `hidden_layers.*`, ResidMLP's `mlp_out`), that expected pattern is
a sorted dense block, not a diagonal; permuting those toward identity would make a
correctly-trained site look wrong. Unconditional — no config gate, no `EvalConfig`
required (toys have none)."""

import io
from typing import Any, Literal

import numpy as np

from param_decomp.core.model import DecomposedModel
from param_decomp.core.slow_eval import (
    PermutationMetricSpec,
    PositionCI,
    plot_permuted_ci_heatmaps,
    render_uv_figure,
    resolve_permutation_metrics,
)


def toy_uv_spec(model: DecomposedModel, raw_cfg: dict[str, Any]) -> PermutationMetricSpec:
    """The permutation spec resolved over the toy's sites from the run config's
    `eval.metrics` (the raw schema dict — the toy's core `ExperimentConfig.eval` is `None`,
    so the metric list is re-validated here, as `slow_eval.eval_metrics_from_run_dir` does
    for the LM). `want_uv_plots` is True only when the config names `UVPlots`; the LM-only
    position-CI / identity metrics are ignored by the toy."""
    from pydantic import TypeAdapter

    from param_decomp.core.configs import AnyEvalMetricConfig

    raw_metrics = (raw_cfg.get("eval") or {}).get("metrics", [])
    adapter = TypeAdapter(AnyEvalMetricConfig)
    metrics = [adapter.validate_python(m) for m in raw_metrics]
    return resolve_permutation_metrics(model.site_names, metrics)


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


def permuted_ci_heatmap_due(now_step: int, total_steps: int, save_every: int | None) -> bool:
    """Cadence for `log_permuted_ci_heatmap`: alongside each checkpoint (`save_every`, when
    set) and always at the final step — one heatmap per saved snapshot, cheaper than every
    `train_log_every` eval step over a 5k-100k-step toy run."""
    return now_step == total_steps or (save_every is not None and now_step % save_every == 0)


def log_permuted_ci_heatmap(
    ci_lower: dict[str, Any],
    ci_upper: dict[str, Any],
    permutation: dict[str, Literal["identity", "dense"]],
    now_step: int,
    wandb_active: bool,
) -> None:
    """Render + log the permuted CI heatmap for every site's single-feature-probe CI, each
    site permuted toward ITS target pattern (`permutation`; identity sites via Hungarian
    assignment, dense sites by column mass — `slow_eval.plot_permuted_ci_heatmaps`). No-op
    unless wandb is on."""
    if not wandb_active:
        return
    position_ci = {
        name: PositionCI(lower=np.asarray(ci_lower[name]), upper=np.asarray(ci_upper[name]))
        for name in ci_lower
    }
    lower_png, upper_png = plot_permuted_ci_heatmaps(position_ci, permutation)
    import wandb
    from PIL import Image

    wandb.log(
        {
            "slow_eval/figures/causal_importances": wandb.Image(Image.open(io.BytesIO(lower_png))),
            "slow_eval/figures/causal_importances_upper_leaky": wandb.Image(
                Image.open(io.BytesIO(upper_png))
            ),
        },
        step=now_step,
    )
