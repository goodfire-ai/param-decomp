"""Offline eval of a JAX-exported checkpoint: torch eval-metric parity on wandb.

The JAX single-pool trainer (pd-nano-jax) exports torch-format safetensors via
`jsp-export` — a full strict `LMComponentModel` state dict. This runner rebuilds the
torch model from a reference torch experiment yaml (the source of truth for the
`pd:` / `target:` / `data:` / `eval:` blocks), strict-loads the export, runs the
SLOW metrics from the yaml's `eval:` block (the heavy/plot ones the JAX in-loop eval
deliberately doesn't replicate) on the standard eval data stream, and logs the
results into the JAX run's wandb run. Fast scalars are the live trainer's job —
recomputing them here put a second writer on the `eval/*` keys.

The eval pass is fed as micro-batches: `eval.batch_size * eval.n_steps` total samples,
`--micro-batch-size` per forward. Every eval metric accumulates sums + counts across
`update` calls, so micro-batching is exact for the position-weighted means. (The two
exceptions are cosmetic/stochastic: `CIHistograms` with `n_batches_accum` caps the
histogram at that many micro-batches, and a `c`-scope PGD mask is shared
over a micro-batch rather than the full eval batch.)

Keys go under `slow_eval/<log_namespace>/<key>`, matching the in-train torch keys
byte-for-byte, logged retroactively onto the dedicated `slow_eval/step` axis (via
`wandb.define_metric`): the JAX run is live and its default `_step` axis has advanced
past the export step, so an explicit `wandb.log(step=<export step>)` would be
silently dropped.

Usage:
    pd-offline-eval <run>/export/model_<step>.safetensors path/to/reference.yaml \\
        [--step N] [--wandb-run-id ID] [--no-wandb] [--micro-batch-size M]
"""

import os
import sys
import time
from pathlib import Path
from typing import Any

import fire
import torch
import wandb
import yaml
from dotenv import load_dotenv
from safetensors.torch import load_file

from param_decomp.decomposition_targets import resolve_decomposition_targets
from param_decomp.log import logger
from param_decomp.metrics.base import Metric
from param_decomp.torch_helpers import loop_dataloader
from param_decomp.train_step import run_eval_pass
from param_decomp_config.experiment import WandbConfig
from param_decomp_config.lm import LMExperimentConfig
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.lm.run import build_lm_loader, build_target
from param_decomp_lab.experiments.lm.vendored.component_adapter import FsdpComponentAdapter
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.infra.wandb import get_wandb_entity, try_wandb
from param_decomp_lab.run_sink import _wandb_value
from param_decomp_lab.seed import set_seed


def _load_reference_config(config_path: Path) -> LMExperimentConfig:
    """Raw-HF Llama specs are normalized to the vendored view: `jsp-export` writes
    vendored-layout state-dict keys regardless of how the run's yaml named the
    target, so the eval model must be `VendoredLlama` (same weights) with the
    `model.`-prefixed site patterns stripped to the vendored module tree. The
    frozen target is forced to bf16 — that matches what the JAX run ACTUALLY
    trains with (its documented fp32-yaml divergence), and is therefore the more
    faithful eval reference."""
    raw = yaml.safe_load(config_path.read_text())
    assert "topology" not in raw.get("runtime", {}), (
        f"{config_path}: multi-pool reference yaml — needs the n-pool subsystems"
    )
    spec = raw["target"]["spec"]
    if spec.get("kind") == "hf" and spec.get("model_class") == "transformers.LlamaForCausalLM":
        raw["target"]["spec"] = {
            "kind": "hf_weights_in_vendored",
            "model_class": "param_decomp_lab.experiments.lm.vendored.llama_3_1.model.VendoredLlama",
            "model_name": spec["model_name"],
        }
        raw["target"]["weights_dtype"] = "bfloat16"
        for target in raw["pd"]["decomposition_targets"]:
            target["module_pattern"] = target["module_pattern"].removeprefix("model.")
    return LMExperimentConfig(**raw)


def _step_from_export_name(filename: str) -> int:
    assert filename.startswith("model_") and filename.endswith(".safetensors"), (
        f"expected `model_<step>.safetensors`, got {filename!r}"
    )
    return int(filename.removeprefix("model_").removesuffix(".safetensors"))


def _load_exported_component_model(
    cfg: LMExperimentConfig, export_path: Path, device: str
) -> LMComponentModel:
    """Rebuild the production-shape `LMComponentModel` and strict-load the JAX export.

    The export is a full fp32 state dict (frozen target + trainable V/U + CI fn).
    `load_state_dict` casts the frozen-target tensors into the dtype `build_target`
    chose (`target.weights_dtype` — bf16 for the reference run) while V/U + CI fn stay
    fp32, matching the in-train torch numerics.
    """
    target_model = build_target(cfg.target)
    target_model.requires_grad_(False)
    resolved_targets = resolve_decomposition_targets(
        target_model, list(cfg.pd.decomposition_targets)
    )
    component_model = LMComponentModel.build(
        target_model=target_model,
        decomposition_targets=resolved_targets,
        ci_config=cfg.pd.ci_config,
        sigmoid_type=cfg.pd.sigmoid_type,
    )
    state_dict = load_file(export_path)
    component_model.load_state_dict(state_dict, strict=True)
    del state_dict
    component_model.to(device)
    component_model.eval()
    return component_model


def _log_results_to_wandb(
    slow_metrics: dict[str, Any],
    *,
    wandb_cfg: WandbConfig,
    run_id: str,
    step: int,
) -> None:
    """Resume the JAX run and log the eval results, attributed to `step`.

    The JAX trainer logs `train/` + `eval/` keys at `step=<train step>` from its own
    process; we only ADD `slow_eval/` keys (disjoint namespace, so concurrent writes
    can't collide on a key). Keys ride the dedicated `slow_eval/step` axis — see the
    module docstring for why an explicit `wandb.log(step=...)` cannot work.
    """
    load_dotenv(override=True)
    wandb.init(
        id=run_id,
        project=wandb_cfg.project,
        entity=wandb_cfg.entity or get_wandb_entity(),
        resume="allow",
    )
    wandb.define_metric("slow_eval/step")
    wandb.define_metric("slow_eval/*", step_metric="slow_eval/step")
    slow_payload: dict[str, Any] = {
        f"slow_eval/{k}": _wandb_value(v) for k, v in slow_metrics.items()
    }
    slow_payload["slow_eval/step"] = step
    try_wandb(wandb.log, slow_payload)
    wandb.finish()


def main(
    export_path: str | Path,
    config: str | Path,
    *,
    step: int | None = None,
    wandb_run_id: str | None = None,
    no_wandb: bool = False,
    micro_batch_size: int = 8,
) -> None:
    """Eval a JAX-exported checkpoint with the reference yaml's eval metrics.

    Args:
        export_path: `model_<step>.safetensors` written by `jsp-export` (a full strict
            `LMComponentModel` state dict), conventionally at
            `<jax run dir>/export/model_<step>.safetensors`.
        config: Reference torch experiment yaml (single-pool `LMExperimentConfig` or
            `TwoPoolLMExperimentConfig` schema, detected by `runtime.topology`);
            its `pd:` / `target:` / `data:` / `eval:` blocks define the model shape,
            data stream, and metrics.
        step: Checkpoint step to attribute results to. Default: parsed from the export
            filename.
        wandb_run_id: The JAX run's wandb run id. Default: the export's run dir name
            (`<run>/export/model_<step>.safetensors` -> `<run>`).
        no_wandb: Dry run — print metrics, skip wandb.
        micro_batch_size: Per-forward batch size; the pass runs
            `eval.batch_size * eval.n_steps / micro_batch_size` micro-batches. The
            default is sized to one B200 (180 GiB): `CEandKLLosses` holds all six
            masked-variant logits + masks as live locals, which OOMs at 16.
    """
    export_path = Path(export_path)
    assert export_path.is_file(), f"export not found: {export_path}"
    if step is None:
        step = _step_from_export_name(export_path.name)
    if wandb_run_id is None:
        assert export_path.parent.name == "export", (
            f"cannot infer --wandb-run-id: expected `<run>/export/{export_path.name}`, "
            f"got {export_path}"
        )
        wandb_run_id = export_path.parents[1].name

    cfg = _load_reference_config(Path(config))
    assert cfg.eval is not None and cfg.eval.metrics, f"{config} has no eval metrics"
    if not no_wandb:
        assert cfg.wandb is not None, f"{config} has no `wandb:` block; pass --no-wandb"

    total_samples = cfg.eval.batch_size * cfg.eval.n_steps
    assert total_samples % micro_batch_size == 0, (
        f"micro_batch_size ({micro_batch_size}) must divide eval.batch_size * eval.n_steps "
        f"({total_samples})"
    )
    n_micro_steps = total_samples // micro_batch_size

    assert torch.cuda.is_available(), "offline eval requires a GPU"
    device = "cuda"
    set_seed(cfg.pd.seed)

    logger.info(f"offline_eval: {export_path.name} step={step} wandb_run_id={wandb_run_id}")
    component_model = _load_exported_component_model(cfg, export_path, device)
    adapter = FsdpComponentAdapter(component_model)

    eval_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="eval",
        device=device,
        batch_size=micro_batch_size,
        dist_state=None,
        seed=cfg.pd.seed,
    )

    slow_instances: dict[str, Metric[Any]] = {}
    for metric_cfg in cfg.eval.metrics:
        metric_class = EVAL_METRIC_CLASSES[metric_cfg.type]
        if not metric_class.slow:
            continue
        metric = metric_class(metric_cfg)
        metric.bind(model=adapter, device=device)
        metric_name = type(metric).__name__
        assert metric_name not in slow_instances, f"duplicate eval metric {metric_name!r}"
        slow_instances[metric_name] = metric
    assert slow_instances, "no slow metrics in eval.metrics — offline eval has nothing to do"

    start = time.perf_counter()
    fast_metrics, slow_metrics = run_eval_pass(
        eval_iterator=loop_dataloader(eval_loader),
        n_steps=n_micro_steps,
        slow_step=True,
        all_instances=slow_instances,
        step=step,
        device=device,
        wrapped_model=adapter,
        component_model=adapter,
        config=cfg.pd,
        reconstruction_loss=recon_loss_kl,
        autocast_bf16=cfg.runtime.autocast_bf16,
    )
    elapsed_s = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    logger.info(
        f"eval pass: {n_micro_steps} micro-batches of {micro_batch_size} "
        f"in {elapsed_s:.0f}s; peak GPU mem {peak_gib:.1f} GiB"
    )

    assert not fast_metrics, f"slow-only instances produced fast outputs: {sorted(fast_metrics)}"
    for key, value in slow_metrics.items():
        rendered = f"{value:.6f}" if isinstance(value, float) else f"<{type(value).__name__}>"
        print(f"slow_eval/{key}: {rendered}")

    if no_wandb:
        logger.info("--no-wandb: skipping wandb log")
        return
    assert cfg.wandb is not None
    _log_results_to_wandb(
        slow_metrics,
        wandb_cfg=cfg.wandb,
        run_id=wandb_run_id,
        step=step,
    )
    logger.info(f"logged slow_eval/ keys to wandb run {wandb_run_id} at step {step}")


def cli() -> None:
    fire.Fire(main)
    # The streaming fineweb loader leaves non-daemon parquet/arrow reader threads that
    # block interpreter shutdown indefinitely (reproducible by just iterating the
    # loader), which would wedge the GPU job after the eval is done. All results are
    # already flushed (stdout below; wandb via `wandb.finish()`), so hard-exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    cli()
