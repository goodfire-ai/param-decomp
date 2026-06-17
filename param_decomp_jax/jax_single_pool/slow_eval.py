"""JAX-native slow (plot-type) eval metrics, the offline counterpart of `eval.py`.

`eval.py` runs the FAST scalar tier in-loop (CE/KL, CI-L0, the fresh-PGD probe). The
SLOW tier is the heavy plot metrics deferred to this out-of-loop pass:
`CIHistograms`, `ComponentActivationDensity`, `CIMeanPerComponent` (the torch eval-
metric classes of the same names). Every one of them is a reduction over the per-site
causal-importance arrays from a masked-free forward, then a numpy/matplotlib plot. The
forward + reduction is JAX; the plotting is framework-agnostic (it mirrors the torch
`param_decomp_lab/eval_metrics/plotting.py` reductions on numpy arrays, no torch).

This runs as an OFFLINE pass over an on-disk checkpoint (`jsp-slow-eval <run_dir>`):
rebuild the JAX target from the run's
config, restore the `TrainState`, accumulate the reductions over `n_steps` eval batches,
render the figures, and log them under `slow_eval/*` into the run's wandb (the dedicated
`slow_eval/step` axis — the live run's `_step` has advanced, so an explicit `step=` write
would be dropped). No torch, no export round-trip.

Cross-batch reductions are exact under micro-batching: density/mean accumulate
SUM-over-positions + a position count, divided once at the end (token-weighted mean,
uniform `(B, T)` makes it the plain mean). `CIHistograms` caps its raw-value sample at
`n_batches_accum` batches, matching the torch metric's `n_batches_accum` early-stop.

It also computes the two SCALAR hidden-acts recon eval metrics (`CIHiddenActsReconLoss`,
`StochasticHiddenActsReconLoss`) natively — per decomposed site, the summed MSE between
the masked-model and target-model site OUTPUT activations, divided once by the element
count (`hidden_acts_eval.py`). Those ride the `masked_site_outputs` model seam (SPEC S31,
amended 2026-06-16 from keep-on-bridge) and are emitted as scalars under the torch log
keys (`<ClassName>/<site>` + a combined `<ClassName>`).
"""

import argparse
import io
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from jax_single_pool.checkpoint import make_checkpoint_manager, restore_step
from jax_single_pool.ci_fn import lower_leaky_hard_sigmoid
from jax_single_pool.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    load_run_dir_config,
)
from jax_single_pool.data import BatchSchedule, ShardServer, scan_shards
from jax_single_pool.hidden_acts_eval import (
    accumulate_hidden_acts,
    hidden_acts_log_entries,
    make_ci_hidden_acts_step,
    make_stochastic_hidden_acts_step,
)
from jax_single_pool.lm import DecomposedModel
from jax_single_pool.load_run import build_target
from jax_single_pool.run_state import build_optimizers, init_train_state
from jax_single_pool.sharding import dp_mesh
from param_decomp_config.routing import SamplingType


@dataclass(frozen=True)
class SiteReduction:
    """Per-site accumulators across the eval pass (all `(C,)` or scalar / capped sample).

    `density_counts[c]` = #(positions where `lower_leaky > threshold`); `ci_sums[c]` =
    Σ positions `lower_leaky`; `n_positions` = total positions seen (shared count for
    both means). `lower_sample` / `logits_sample` are flattened raw values from the first
    `n_batches_accum` batches, for the two `CIHistograms` histograms."""

    density_counts: np.ndarray
    ci_sums: np.ndarray
    n_positions: int
    lower_sample: np.ndarray
    logits_sample: np.ndarray


SlowEvalStep = Callable[
    [Any, Any, Float[Array, "*leading d"]],
    tuple[dict[str, Array], dict[str, Array], Array, dict[str, Array], dict[str, Array]],
]
"""`(ci_fn, frozen, residual) -> (density_counts, ci_sums, n_positions, flat_lower,
flat_logits)` — the per-batch reduction, pre-reduced over positions. The slow plot
metrics read only the CI arrays, so V/U (`components`) is not an input."""


def make_slow_eval_step(lm: DecomposedModel, ci_alive_threshold: float) -> SlowEvalStep:
    """Build the jit'd per-batch reduction `slow_eval_step(ci_fn, frozen, residual) ->
    ({site: density_counts}, {site: ci_sums}, n_positions, {site: flat lower},
    {site: flat logits})`. `lower`/`logits` are returned whole (the host caps the
    histogram sample); counts/sums are pre-reduced over positions."""
    site_names = lm.site_names

    @jax.jit
    def slow_eval_step(
        ci_fn: Any, frozen: Any, residual: Float[Array, "*leading d"]
    ) -> tuple[dict[str, Array], dict[str, Array], Array, dict[str, Array], dict[str, Array]]:
        # CI fn stays fp32 (its master dtype): torch offline-eval keeps V/U + CI fn fp32,
        # casting only the frozen target to bf16. The slow plot metrics are a
        # fp32-CI-fn readout, so we don't take eval.py's bf16-compute path here.
        site_inputs = {
            s: x.astype(jnp.float32) for s, x in lm.site_inputs(frozen, residual).items()
        }
        logits = ci_fn.site_logits(site_inputs)
        lower = {s: lower_leaky_hard_sigmoid(logits[s]) for s in site_names}

        density_counts = {
            s: (lower[s] > ci_alive_threshold)
            .astype(jnp.float32)
            .reshape(-1, lower[s].shape[-1])
            .sum(0)
            for s in site_names
        }
        ci_sums = {s: lower[s].reshape(-1, lower[s].shape[-1]).sum(0) for s in site_names}
        first = lower[site_names[0]]
        n_positions = jnp.asarray(math.prod(first.shape[:-1]), jnp.int32)
        flat_lower = {s: lower[s].reshape(-1) for s in site_names}
        flat_logits = {s: logits[s].reshape(-1) for s in site_names}
        return density_counts, ci_sums, n_positions, flat_lower, flat_logits

    return slow_eval_step


def accumulate_site_reductions(
    slow_eval_step: SlowEvalStep,
    ci_fn: Any,
    frozen: Any,
    residual_batches: list[Float[Array, "*leading d"]],
    n_batches_accum: int | None,
) -> dict[str, SiteReduction]:
    """Drive `slow_eval_step` over the eval batches and fold the per-batch reductions
    into one `SiteReduction` per site. `n_batches_accum` caps how many batches feed the
    `CIHistograms` raw-value sample (torch `n_batches_accum`); None keeps all."""
    assert residual_batches, "slow eval needs at least one batch"
    density: dict[str, np.ndarray] = {}
    sums: dict[str, np.ndarray] = {}
    lower_chunks: dict[str, list[np.ndarray]] = {}
    logits_chunks: dict[str, list[np.ndarray]] = {}
    total_positions = 0
    for batch_idx, residual in enumerate(residual_batches):
        d, s, n_pos, flat_lower, flat_logits = slow_eval_step(ci_fn, frozen, residual)
        total_positions += int(n_pos)
        keep_sample = n_batches_accum is None or batch_idx < n_batches_accum
        for site in d:
            counts, ci_sum = np.asarray(d[site]), np.asarray(s[site])
            density[site] = counts if batch_idx == 0 else density[site] + counts
            sums[site] = ci_sum if batch_idx == 0 else sums[site] + ci_sum
            if keep_sample:
                lower_chunks.setdefault(site, []).append(np.asarray(flat_lower[site]))
                logits_chunks.setdefault(site, []).append(np.asarray(flat_logits[site]))

    return {
        site: SiteReduction(
            density_counts=density[site],
            ci_sums=sums[site],
            n_positions=total_positions,
            lower_sample=np.concatenate(lower_chunks[site]),
            logits_sample=np.concatenate(logits_chunks[site]),
        )
        for site in density
    }


def _render_figure(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _grid_dims(n: int, max_rows: int = 6) -> tuple[int, int]:
    n_cols = (n + max_rows - 1) // max_rows
    n_rows = min(n, max_rows)
    return n_rows, n_cols


def plot_ci_value_histograms(samples: dict[str, np.ndarray], bins: int = 100) -> bytes:
    """Per-site histogram of flattened CI values (torch `plot_ci_values_histograms`)."""
    n_rows, n_cols = _grid_dims(len(samples))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    flat_axes = axs.T.ravel()
    for ax in flat_axes[len(samples) :]:
        ax.set_visible(False)
    for ax, (name, values) in zip(flat_axes, samples.items(), strict=False):
        ax.hist(values, bins=bins)
        ax.set_yscale("log")
        ax.set_title(f"Causal importances for {name.replace('.', '_')}")
        ax.set_xlabel("Causal importance value")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _render_figure(fig)


def plot_component_activation_density(densities: dict[str, np.ndarray], bins: int = 100) -> bytes:
    """Per-site histogram of per-component activation density (torch
    `plot_component_activation_density`)."""
    n_rows, n_cols = _grid_dims(len(densities))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    flat_axes = axs.T.ravel()
    for ax in flat_axes[len(densities) :]:
        ax.set_visible(False)
    for ax, (name, density) in zip(flat_axes, densities.items(), strict=False):
        ax.hist(density, bins=bins)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Activation density")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _render_figure(fig)


def plot_mean_component_cis_both_scales(
    mean_cis: dict[str, np.ndarray],
) -> tuple[bytes, bytes]:
    """Sorted-descending mean-CI scatter, linear and log y (torch
    `plot_mean_component_cis_both_scales`)."""
    sorted_data = {name: np.sort(v)[::-1] for name, v in mean_cis.items()}
    n_rows, n_cols = _grid_dims(len(sorted_data))
    images: list[bytes] = []
    for log_y in (False, True):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows), squeeze=False)
        flat_axes = axs.T.ravel()
        for ax in flat_axes[len(sorted_data) :]:
            ax.set_visible(False)
        for ax, (name, sorted_components) in zip(flat_axes, sorted_data.items(), strict=False):
            if log_y:
                ax.set_yscale("log")
            ax.scatter(range(len(sorted_components)), sorted_components, marker="x", s=10)
            ax.set_xlabel("Component")
            ax.set_ylabel("mean CI")
            ax.set_title(name, fontsize=10)
        fig.tight_layout()
        images.append(_render_figure(fig))
    return images[0], images[1]


def render_slow_eval_figures(
    reductions: dict[str, SiteReduction],
) -> dict[str, bytes]:
    """The three slow plot metrics as `{log_key: png_bytes}`, keyed exactly as torch
    logs them under `slow_eval/` (`figures/<key>` from each metric's `compute()`)."""
    lower_hist = plot_ci_value_histograms({s: r.lower_sample for s, r in reductions.items()})
    logits_hist = plot_ci_value_histograms({s: r.logits_sample for s, r in reductions.items()})
    assert all(r.n_positions > 0 for r in reductions.values())
    densities = {s: r.density_counts / r.n_positions for s, r in reductions.items()}
    mean_cis = {s: r.ci_sums / r.n_positions for s, r in reductions.items()}
    density_fig = plot_component_activation_density(densities)
    mean_linear, mean_log = plot_mean_component_cis_both_scales(mean_cis)
    return {
        "figures/causal_importance_values": lower_hist,
        "figures/causal_importance_values_pre_sigmoid": logits_hist,
        "figures/component_activation_density": density_fig,
        "figures/ci_mean_per_component": mean_linear,
        "figures/ci_mean_per_component_log": mean_log,
    }


@dataclass(frozen=True)
class SlowEvalOutput:
    """The offline slow-eval payload: plot `figures` ({log_key: png}) and scalar
    `hidden_acts` metrics ({torch_log_key: mse})."""

    figures: dict[str, bytes]
    hidden_acts: dict[str, float]


def _eval_config(cfg: ExperimentConfig) -> EvalConfig:
    assert cfg.eval is not None, f"{cfg.run_id}: no eval block — nothing to slow-eval"
    return cfg.eval


def compute_hidden_acts_metrics(
    lm: DecomposedModel,
    state: Any,
    frozen: Any,
    residual_batches: list[Float[Array, "*leading d"]],
    n_mask_samples: int,
    sampling: SamplingType,
    base_key: Array,
) -> dict[str, float]:
    """Both hidden-acts recon eval metrics over the eval batches, keyed by the torch
    `<ClassName>[/<site>]` log keys. `state.components`/`state.ci_fn` are the restored
    trajectory; `base_key` seeds the stochastic variant's per-batch draws."""
    ci_key, stoch_key = random.split(base_key)
    ci_step = make_ci_hidden_acts_step(lm)
    ci_reductions = accumulate_hidden_acts(
        ci_step, state.components, state.ci_fn, frozen, residual_batches, ci_key
    )
    stoch_step = make_stochastic_hidden_acts_step(lm, n_mask_samples, sampling)
    stoch_reductions = accumulate_hidden_acts(
        stoch_step, state.components, state.ci_fn, frozen, residual_batches, stoch_key
    )
    return {
        **hidden_acts_log_entries("CIHiddenActsReconLoss", ci_reductions),
        **hidden_acts_log_entries("StochasticHiddenActsReconLoss", stoch_reductions),
    }


def run_offline_slow_eval(run_dir: Path, cfg: ExperimentConfig, step: int) -> SlowEvalOutput:
    """Restore checkpoint `step` from `run_dir`'s ckpts, render the slow figures, and
    compute the scalar hidden-acts recon metrics. `run_dir` is the on-disk dir (the
    exporter takes it the same way); `cfg.run_dir` can differ when a run dir is read from
    a relocated copy. CPU-OK."""
    eval_cfg = _eval_config(cfg)
    mesh = dp_mesh()
    lm, frozen, prefix, prefix_residual_fn, _vocab_size = build_target(cfg, mesh)

    opt_vu, opt_ci, _schedules = build_optimizers(cfg)
    init_key, src_key, _run_key = random.split(random.PRNGKey(cfg.seed), 3)
    reference = init_train_state(cfg, lm, opt_vu, opt_ci, init_key, src_key, mesh)
    manager = make_checkpoint_manager(run_dir / "ckpts", cfg.cadence.keep_last)
    state = restore_step(manager, reference, step)

    data_cfg = cfg.data
    assert isinstance(data_cfg, DataConfig), "slow eval reads the LM parquet data path"
    schedule = BatchSchedule(scan_shards(data_cfg.dir), eval_cfg.batch_size, cfg.seed + 1)
    server = ShardServer(schedule, data_cfg.seq_len, jax.process_index(), jax.process_count())
    to_residual = jax.jit(prefix_residual_fn)
    residual_batches = [
        to_residual(prefix, jnp.asarray(server.local_batch(j))) for j in range(eval_cfg.n_steps)
    ]

    slow_eval_step = make_slow_eval_step(lm, eval_cfg.ci_alive_threshold)
    reductions = accumulate_site_reductions(
        slow_eval_step, state.ci_fn, frozen, residual_batches, _n_batches_accum(run_dir)
    )
    hidden_acts = compute_hidden_acts_metrics(
        lm, state, frozen, residual_batches, cfg.n_mask_samples, cfg.sampling,
        random.fold_in(random.PRNGKey(cfg.seed), step),
    )  # fmt: skip
    return SlowEvalOutput(figures=render_slow_eval_figures(reductions), hidden_acts=hidden_acts)


def _n_batches_accum(run_dir: Path) -> int | None:
    """The torch `CIHistograms.n_batches_accum` from the raw eval block (it's dropped by
    `EvalConfig`, which keeps only scalar-tier fields). None caps nothing."""
    import yaml

    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    for metric in raw["eval"]["metrics"]:
        if metric.get("type") == "CIHistograms":
            return metric.get("n_batches_accum")
    return None


def _write_output(output: SlowEvalOutput, out_dir: Path, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, png in output.figures.items():
        path = out_dir / f"{key.replace('/', '__')}_step{step}.png"
        path.write_bytes(png)
        print(f"wrote {path}", flush=True)
    scalars_path = out_dir / f"hidden_acts_recon_step{step}.json"
    scalars_path.write_text(json.dumps(output.hidden_acts, indent=2, sort_keys=True))
    print(f"wrote {scalars_path}", flush=True)
    for key in ("CIHiddenActsReconLoss", "StochasticHiddenActsReconLoss"):
        print(f"  {key} = {output.hidden_acts[key]:.6g}", flush=True)


def _log_to_wandb(cfg: ExperimentConfig, output: SlowEvalOutput, step: int) -> None:
    import wandb
    from PIL import Image

    assert cfg.wandb is not None, "no wandb config — pass --no-wandb to skip logging"
    wandb.init(
        id=cfg.run_id,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume="allow",
    )
    wandb.define_metric("slow_eval/step")
    wandb.define_metric("slow_eval/*", step_metric="slow_eval/step")
    payload: dict[str, Any] = {
        f"slow_eval/{k}": wandb.Image(Image.open(io.BytesIO(v))) for k, v in output.figures.items()
    }
    payload.update({f"slow_eval/loss/{k}": v for k, v in output.hidden_acts.items()})
    payload["slow_eval/step"] = step
    wandb.log(payload)
    wandb.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--no-wandb", action="store_true", help="write PNGs to disk only")
    args = ap.parse_args()
    jax.config.update("jax_platforms", "cpu")

    cfg = load_run_dir_config(args.run_dir)
    manager = make_checkpoint_manager(args.run_dir / "ckpts", cfg.cadence.keep_last)
    step = args.step if args.step is not None else manager.latest_step()
    assert step is not None, f"no checkpoints under {args.run_dir / 'ckpts'}"

    output = run_offline_slow_eval(args.run_dir, cfg, step)
    _write_output(output, args.run_dir / "slow_eval", step)
    if not args.no_wandb:
        _log_to_wandb(cfg, output, step)


if __name__ == "__main__":
    main()
