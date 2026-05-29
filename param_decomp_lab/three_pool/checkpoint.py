"""Offline checkpoint assembly for 3-pool training.

The train loop never assembles a checkpoint. Instead each rank writes a
self-contained partial to a shared-FS scratch dir (see
``ThreePoolTrainer.snapshot``): its owned model params (LW leaders → owned-sites
V/U; CI leader → CI fn), its optimizer state, and (PPGD) its sources. A separate
async SLURM job then reads every partial for a step and assembles the canonical
artifacts off the training critical path.

This module owns that offline assembly. It builds a full-decomposition
``ComponentModel`` on CPU as an assembly buffer, copies in every site's V/U + the
CI fn from the partials, and returns its ``state_dict()`` — the same flat schema
``load_component_model_from_checkpoint`` expects. No live ranks, no NCCL: the
assembly is pure file I/O + tensor copies.
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.ci_fns import CiConfig
from param_decomp.ci_sigmoids import SigmoidType
from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import DecompositionTarget


def site_to_component_prefix(site: str) -> str:
    """State-dict key prefix for a site's V/U params.

    ``ComponentModel`` registers components under an ``nn.ModuleDict`` keyed by
    the site with dots replaced by dashes (so dots don't nest module levels), so
    ``h.0.attn.q_proj`` → ``_components.h-0-attn-q_proj``.
    """
    return f"_components.{site.replace('.', '-')}"


def owned_model_state_keys(
    model_state_dict_keys: set[str], *, owned_sites: tuple[str, ...]
) -> set[str]:
    """The V/U state-dict keys for ``owned_sites`` (LW block leaders' partial)."""
    prefixes = tuple(f"{site_to_component_prefix(s)}." for s in owned_sites)
    return {k for k in model_state_dict_keys if k.startswith(prefixes)}


def ci_fn_state_keys(model_state_dict_keys: set[str]) -> set[str]:
    """The CI-fn state-dict keys (the CI pool leader's partial)."""
    return {k for k in model_state_dict_keys if k.startswith("ci_fn.")}


def assemble_model_state_dict_from_partials(
    *,
    partials: list[dict[str, Any]],
    target_model: nn.Module,
    run_batch: RunBatch,
    ci_config: CiConfig,
    sigmoid_type: SigmoidType,
    c_per_site: dict[str, int],
    all_sites: tuple[str, ...],
) -> dict[str, Tensor]:
    """Assemble the full ComponentModel state_dict from per-rank scratch partials.

    Each partial's ``model_params`` holds the CPU tensors that rank owns (a slice
    of its own ``component_model.state_dict()``): LW block leaders contribute their
    owned sites' ``_components.<site>.*``, the CI pool leader contributes ``ci_fn.*``.
    The union of every partial's keys must exactly cover the full-decomposition
    model's V/U + CI-fn keys (target-model params come from the fresh buffer).
    """
    # ComponentModel asserts the target has no trainable params; build_target only
    # `.eval()`s it, so freeze here (the training / load_component_model paths
    # freeze at their own construction sites).
    target_model.eval()
    target_model.requires_grad_(False)
    full_targets = [DecompositionTarget(module_path=s, C=c_per_site[s]) for s in all_sites]
    full_cm = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        decomposition_targets=full_targets,
        ci_config=ci_config,
        sigmoid_type=sigmoid_type,
    )

    # `ComponentModel` registers `target_model` as a submodule, so its frozen
    # weights appear in the state_dict under the `target_model.` prefix. They come
    # from the freshly-built buffer (which shares the real target_model), so the
    # partials only need to cover the V/U + CI-fn keys — everything NOT under
    # `target_model.`.
    expected_keys = set(full_cm.state_dict().keys())
    fillable_keys = {k for k in expected_keys if not k.startswith("target_model.")}

    collected: dict[str, Tensor] = {}
    for partial in partials:
        for k, v in partial["model_params"].items():
            assert k not in collected, f"duplicate model param {k!r} across partials"
            collected[k] = v

    assert set(collected.keys()) == fillable_keys, (
        "partials do not cover the full model state_dict:\n"
        f"  missing: {sorted(fillable_keys - set(collected))}\n"
        f"  extra:   {sorted(set(collected) - fillable_keys)}"
    )

    full_state = full_cm.state_dict()
    with torch.no_grad():
        for k, v in collected.items():
            full_state[k].copy_(v)

    return {k: v.cpu() for k, v in full_state.items()}
