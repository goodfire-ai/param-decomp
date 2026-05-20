"""Standalone driver-mediated entry point for 2-pool training.

This is a sibling to ``run_pd`` — NOT a replacement and NOT a fork that
mutates ``RunConfig``. The single-pool ``run_pd`` stays untouched; 2-pool is
an exotic enough setup (heterogeneous GPU pools) that bolting an optional
field onto the mainline config doesn't earn its keep.

Pattern: load a regular ``RunConfig`` (via the existing YAML / driver
mechanism) for the model + dataloaders + PD knobs, plus a separate
``TwoPoolConfig`` for the topology. The driver is reused as-is; only the
inner trainer differs (``optimize_two_pool`` instead of ``optimize``).

Use cases:
- Benchmark scripts that need a real model + real data without re-implementing
  HF loaders (see ``param_decomp/scripts/two_pool_benchmark/qwen_2pool.py``
  for the current bespoke wiring this replaces).
- A future ``pd-run-two-pool`` CLI that loads two YAMLs (RunConfig + TwoPoolConfig)
  and dispatches here.
"""

# pyright: reportPrivateUsage=false

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.run import RunConfig
from param_decomp.two_pool.config import TwoPoolConfig
from param_decomp.two_pool.run import (
    PhaseProfiler,
    build_two_pool_runtime,
    optimize_two_pool,
)
from param_decomp.utils.module_utils import expand_module_patterns


def run_two_pool(
    *,
    run_cfg: RunConfig,
    two_pool_cfg: TwoPoolConfig,
    target: PDTarget,
    train_loader: DataLoader[Any],
    device: str,
    profiler: PhaseProfiler | None = None,
) -> None:
    """Run 2-pool training given a fully-materialized RunConfig + TwoPoolConfig.

    Caller is responsible for resolving the driver + materializing ``target``
    / ``train_loader`` (typically via ``materialize_run`` in ``run_pd``).

    The single-pool ``optimize`` integrates with the new metric registry,
    sinks, eval loop, and checkpointing. This 2-pool path currently only
    runs the training loop — eval / metrics / sinks / checkpoints are TODO
    and will be lifted from ``run_pd.optimize`` as needed.
    """
    # Resolve module patterns against the actual target to derive c_per_site
    # (same pattern as run_pd.optimize). Each owned site in the topology must
    # appear in the resolved module_path_info.
    module_path_info = expand_module_patterns(target.model, run_cfg.pd.all_module_info)
    c_per_site = {info.module_path: info.C for info in module_path_info}
    for bg in two_pool_cfg.block_groups:
        for site in bg.owned_sites:
            assert site in c_per_site, (
                f"site '{site}' in two_pool topology but not in pd.module_info "
                f"after pattern expansion. Available: {sorted(c_per_site)[:5]}…"
            )

    pool_runtime = build_two_pool_runtime(
        two_pool_cfg,
        c_per_site=c_per_site,
        ci_config=run_cfg.pd.ci_config,
        sigmoid_type=run_cfg.pd.sigmoid_type,
        run_batch=target.run_batch,
        reconstruction_loss=target.reconstruction_loss,
        bf16_autocast=run_cfg.runtime.autocast_bf16,
    )

    # Bridge DataLoader → batch_iter (the 2-pool inner loop currently takes
    # a step-indexed callable rather than a streaming iterator). Cycles
    # through the loader; same global batch is used by every rank since
    # optimize_two_pool slices internally per pool.
    loader_iter = iter(train_loader)

    def batch_iter(_step: int) -> Tensor:
        nonlocal loader_iter
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        if isinstance(batch, Tensor):
            return batch.to(device)
        if isinstance(batch, dict) and "input_ids" in batch:
            return batch["input_ids"].to(device)
        if isinstance(batch, list | tuple) and len(batch) > 0 and isinstance(batch[0], Tensor):
            return batch[0].to(device)
        raise TypeError(f"Unsupported batch type from DataLoader: {type(batch).__name__}")

    optimize_two_pool(
        target_model=target.model,
        pool_config=pool_runtime,
        device=torch.device(device),
        n_steps=run_cfg.pd.steps,
        batch_iter=batch_iter,
        profiler=profiler,
    )


__all__ = ["run_two_pool"]
