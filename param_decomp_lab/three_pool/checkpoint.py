"""Offline checkpoint assembly for 3-pool training.

The train loop never assembles a checkpoint. Instead each rank writes a
self-contained partial to a shared-FS scratch dir (see
``ThreePoolTrainer.snapshot``): its owned model params (chunk leaders → owned-sites
V/U; CI leader → CI fn), its optimizer state, and (PPGD) its sources. A separate
async SLURM job then reads every partial for a step and assembles the canonical
artifacts off the training critical path.

This module owns that offline assembly. It builds a full-decomposition
``ComponentModel`` on CPU as an assembly buffer, copies in every site's V/U + the
CI fn from the partials, and returns its ``state_dict()`` — the same flat schema
``load_component_model_from_checkpoint`` expects. No live ranks, no NCCL: the
assembly is pure file I/O + tensor copies.
"""

import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.ci_sigmoids import SigmoidType
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp_config.ci_fn import CiConfig
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel


def site_to_component_prefix(site: str) -> str:
    """State-dict key prefix for a site's V/U params.

    ``LMComponentModel`` holds its components in-tree: a ``ComponentLinear`` /
    ``ComponentEmbedding`` swapped in at the site's path under the wrapped ``model``,
    with the V/U living on a ``.components`` submodule. So ``h.0.attn.q_proj`` →
    ``model.h.0.attn.q_proj.components`` (keys ``...components.V`` / ``.U`` / ``.bias``).
    The frozen ``model.<site>.target_weight`` / ``model.<site>.bias`` buffers are NOT
    under this prefix — they're rebuilt from the target, not saved in partials.
    """
    return f"model.{site}.components"


def owned_model_state_keys(model_state_dict_keys: set[str], *, sites: tuple[str, ...]) -> set[str]:
    """The V/U state-dict keys for ``sites`` (chunk leaders' partial)."""
    prefixes = tuple(f"{site_to_component_prefix(s)}." for s in sites)
    return {k for k in model_state_dict_keys if k.startswith(prefixes)}


def ci_fn_state_keys(model_state_dict_keys: set[str]) -> set[str]:
    """The CI-fn state-dict keys (the CI pool leader's partial)."""
    return {k for k in model_state_dict_keys if k.startswith("ci_fn.")}


def is_trainable_component_key(key: str) -> bool:
    """True for the V/U (`...components.*`) and CI-fn (`ci_fn.*`) keys — the only
    state worth persisting. The frozen target's `model.*` weights / per-site
    `target_weight` buffers are rebuilt from the freshly-loaded target, so they are
    NOT trained, NOT saved in partials, and must NOT be reloaded on resume (reloading
    the 8B frozen target across every rank is the resume I/O storm)."""
    return ".components." in key or key.startswith("ci_fn.")


def assemble_model_state_dict_from_partials(
    *,
    collected_model_params: dict[str, Tensor],
    target_model: nn.Module,
    run_batch: RunBatch,
    ci_config: CiConfig,
    sigmoid_type: SigmoidType,
    c_per_site: dict[str, int],
    all_sites: tuple[str, ...],
) -> dict[str, Tensor]:
    """Assemble the full LMComponentModel state_dict from the collected partial params.

    ``collected_model_params`` is the union of every rank's owned ``model_params``
    (the caller streams the partials and merges them, so this never holds all
    partials at once): chunk leaders contribute their owned sites'
    ``model.<site>.components.*``, the CI pool leader contributes ``ci_fn.*``. The
    union must exactly cover the full-decomposition model's V/U + CI-fn keys (frozen
    target params come from the fresh buffer).
    """
    del run_batch  # the vendored LMComponentModel calls the model directly (no run_batch)
    # Target-type validation lives in LMComponentModel.build → _componentize (single source of
    # truth); an unsupported target raises there.
    full_targets = [DecompositionTarget(module_path=s, C=c_per_site[s]) for s in all_sites]
    full_cm = LMComponentModel.build(
        target_model=target_model,
        decomposition_targets=full_targets,
        ci_config=ci_config,
        sigmoid_type=sigmoid_type,
    )

    # `LMComponentModel` holds the (frozen) target in-tree under the `model.` prefix,
    # along with the per-site `target_weight` / `bias` buffers. Those come from the
    # freshly-built buffer, so the partials only need to cover the trainable V/U
    # (`model.<site>.components.*`) + CI-fn (`ci_fn.*`) keys.
    expected_keys = set(full_cm.state_dict().keys())
    fillable_keys = {k for k in expected_keys if is_trainable_component_key(k)}

    assert set(collected_model_params.keys()) == fillable_keys, (
        "partials do not cover the full model state_dict:\n"
        f"  missing: {sorted(fillable_keys - set(collected_model_params))}\n"
        f"  extra:   {sorted(set(collected_model_params) - fillable_keys)}"
    )

    full_state = full_cm.state_dict()
    with torch.no_grad():
        for k, v in collected_model_params.items():
            full_state[k].copy_(v)

    return {k: v.cpu() for k, v in full_state.items()}
