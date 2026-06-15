"""On-loop sharded DCP save/load for the single-pool FSDP LM path.

The FSDP trainer never gathers the full model on the train loop. Each rank holds
its own DTensor shards of the trainable params (components V/U + CI fn) and the
matching optimizer state; this module persists those shards directly via
``torch.distributed.checkpoint`` (DCP) into ``<run_dir>/.dcp/step_<S>/`` — no
full-gather, no rank-0 bottleneck. Off-loop consolidation (``consolidate.py``)
reads the shards back into a full model to emit the downstream ``model_<S>.pth`` +
``training_<S>.pth``.

TRAINABLE-ONLY. The 8B frozen target is NOT saved: it's rebuilt from the vendored
weights at load time. ``StateDictOptions(ignore_frozen_params=True)`` drops every
``requires_grad=False`` param from the model state dict, so only the components'
V/U and the CI fn land in the shards. (The frozen target's per-site
``target_weight`` / ``bias`` are buffers, not params, and are excluded too — DCP's
model state dict carries only params under this option.)

KEY CONVENTION. ``save_dcp`` / ``load_dcp`` operate on the inner ``LMComponentModel``
(``fully_shard`` mutates it in place, so it is still sharded), NOT the
``FsdpComponentAdapter`` wrapping it. This keeps the DCP FQNs identical to a bare
``LMComponentModel.state_dict()`` (``model.<site>.components.*`` / ``ci_fn.*``) —
the same schema the 3-pool's ``is_trainable_component_key`` filter and the
downstream loaders expect — so consolidation can load the shards into a fresh,
unsharded ``LMComponentModel`` with no key translation.

Optimizers are keyed by their state-dict NAME (``"components"`` / ``"ci_fn"``) so a
resume binds each saved optimizer's state back to the right live optimizer. DCP's
optimizer state dict is already param-FQN-keyed (topology-independent), so resuming
into a different shard layout is handled by DCP itself.
"""

from pathlib import Path
from typing import Any, cast

from torch import nn
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.optim import Optimizer

from param_decomp.log import logger

type LossMetricStates = dict[str, dict[str, Any]]

DCP_DIRNAME = ".dcp"
"""Subdir under the run dir holding the per-step sharded checkpoints."""

_MODEL_KEY = "model"
_OPTIMIZERS_KEY = "optimizers"
_STEP_KEY = "step"
_LOSS_METRICS_KEY = "loss_metric_states"

_TRAINABLE_ONLY = StateDictOptions(ignore_frozen_params=True, strict=False)
"""Drop ``requires_grad=False`` params (the frozen 8B target) from the model state
dict, so only the trainable V/U + CI fn are sharded and saved. ``strict=False`` is
required on the SET side: with the frozen keys excluded, the underlying
``module.load_state_dict`` would otherwise reject the now-missing frozen keys."""


def _step_dir(run_dir: Path, step: int) -> Path:
    return run_dir / DCP_DIRNAME / f"step_{step}"


def save_dcp(
    model: nn.Module,
    optimizers: dict[str, Optimizer],
    *,
    step: int,
    loss_metric_states: LossMetricStates,
    out_dir: Path,
) -> None:
    """Sharded save of the trainable state into ``out_dir/.dcp/step_<step>/``.

    Collective over the whole process group. ``model`` is the inner (sharded)
    ``LMComponentModel``; ``optimizers`` maps a stable name (``"components"`` /
    ``"ci_fn"``) to its optimizer. The frozen target is excluded via
    ``ignore_frozen_params``.
    """
    model_sd = get_model_state_dict(model, options=_TRAINABLE_ONLY)
    optim_sds = {
        name: get_optimizer_state_dict(model, opt, options=_TRAINABLE_ONLY)
        for name, opt in optimizers.items()
    }
    state_dict = {
        _MODEL_KEY: model_sd,
        _OPTIMIZERS_KEY: optim_sds,
        _STEP_KEY: step,
        _LOSS_METRICS_KEY: loss_metric_states,
    }
    dcp.save(state_dict, checkpoint_id=str(_step_dir(out_dir, step)))
    logger.info(f"save_dcp: wrote sharded checkpoint to {_step_dir(out_dir, step)}")


def load_dcp(
    model: nn.Module,
    optimizers: dict[str, Optimizer],
    *,
    step: int,
    in_dir: Path,
    loss_metric_states: LossMetricStates,
) -> LossMetricStates:
    """Load ``in_dir/.dcp/step_<step>/`` in place into the sharded model + optimizers.

    Collective over the whole process group. Mutates ``model`` and ``optimizers``;
    returns the loaded loss-metric states for the trainer to apply.

    ``loss_metric_states`` is the FRESH skeleton (the live loss metrics'
    ``state_dict()`` after construction) that DCP fills in place. DCP only loads a
    non-tensor / tensor leaf when the destination already has that exact key, so a
    correctly-shaped skeleton is required — an empty dict silently loads nothing.
    The trainer's freshly-built metrics produce that shape.
    """
    step_dir = _step_dir(in_dir, step)
    assert step_dir.is_dir(), f"load_dcp: no DCP checkpoint at {step_dir}"

    model_sd = get_model_state_dict(model, options=_TRAINABLE_ONLY)
    optim_sds = {
        name: get_optimizer_state_dict(model, opt, options=_TRAINABLE_ONLY)
        for name, opt in optimizers.items()
    }
    # `dcp.load` mutates the entries in place; reading the model/optimizer/loss
    # state back through the typed locals (not the heterogeneous combined dict)
    # keeps their types concrete for the `set_*` calls below.
    state_dict = {
        _MODEL_KEY: model_sd,
        _OPTIMIZERS_KEY: optim_sds,
        _STEP_KEY: 0,
        _LOSS_METRICS_KEY: loss_metric_states,
    }
    dcp.load(state_dict, checkpoint_id=str(step_dir))

    set_model_state_dict(model, model_sd, options=_TRAINABLE_ONLY)
    for name, opt in optimizers.items():
        set_optimizer_state_dict(
            model, opt, optim_state_dict=optim_sds[name], options=_TRAINABLE_ONLY
        )

    saved_step = cast(int, state_dict[_STEP_KEY])
    assert saved_step == step, f"load_dcp: checkpoint step {saved_step} != requested {step}"
    return loss_metric_states


def latest_dcp_step(run_dir: Path) -> int | None:
    """Newest ``.dcp/step_<S>/`` step under ``run_dir``, or ``None`` if there are none."""
    dcp_root = run_dir / DCP_DIRNAME
    if not dcp_root.is_dir():
        return None
    steps: list[int] = []
    for d in dcp_root.glob("step_*"):
        if d.is_dir():
            steps.append(int(d.name.removeprefix("step_")))
    return max(steps) if steps else None
