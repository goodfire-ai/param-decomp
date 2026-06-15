"""Off-loop consolidation of a sharded FSDP DCP checkpoint into the downstream artifacts.

The FSDP train loop writes only sharded DCP shards (``checkpoint.save_dcp`` →
``<run_dir>/.dcp/step_<S>/``) and continues — it never assembles the full model.
This module does that assembly off the critical path (driven by the async SLURM
job that also runs the slow eval), mirroring ``three_pool/consolidate.py``'s
conventions (filenames ``model_<S>.pth`` / ``training_<S>.pth``, pruning,
idempotency, scratch cleanup) — but the DCP reader replaces the 3-pool's per-rank
partial reader.

Reconstructing the full state dict from shards
----------------------------------------------
DCP supports a single-process load of a checkpoint saved under any (sharded)
topology: ``dcp.load`` into a *regular*, unsharded module gives full
(non-sharded) tensors. So consolidation:

1. ``build_full_model()`` → a fresh, unsharded, CPU ``LMComponentModel`` (frozen
   target rebuilt from the vendored weights, trainable V/U + CI fn freshly
   initialised). This is the assembly buffer.
2. Build the two AdamW optimizers over its component / CI-fn params in the SAME
   split + order the trainer uses, so their param FQNs match the saved shards.
3. ``dcp.load`` the trainable-only shards into the model + optimizers (single
   process, no NCCL). The frozen target is untouched — it's not in the shards and
   comes from the fresh build.
4. ``model.state_dict()`` is now the FULL ``LMComponentModel`` state (frozen
   target + loaded trainable params) — the downstream ``model_<S>.pth`` artifact,
   loadable by ``load_vendored_component_model``.
5. Convert each optimizer's state to the by-NAME shape the ``TrainingState``
   resume path expects (``optimizer_state_by_name``), and assemble the
   ``TrainingState``.

Idempotent: a no-op if ``training_<S>.pth`` already exists (the async job may be
retried). DCP shards persist on failure → re-runnable.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch.nn as nn
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.optim import AdamW, Optimizer

from param_decomp.log import logger
from param_decomp.optimize import optimizer_state_by_name
from param_decomp.training_state import TrainingState
from param_decomp_config.pd import PDConfig
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.fsdp.checkpoint import DCP_DIRNAME
from param_decomp_lab.infra.run_files import save_file

_TRAINABLE_ONLY = StateDictOptions(ignore_frozen_params=True, strict=False)
"""``strict=False`` so ``set_model_state_dict`` tolerates the excluded frozen-target
keys (the frozen 8B target is rebuilt from vendored weights, not loaded from shards)."""

COMPONENTS_OPTIMIZER_NAME = "components"
CI_FN_OPTIMIZER_NAME = "ci_fn"


def _step_dir(run_dir: Path, step: int) -> Path:
    return run_dir / DCP_DIRNAME / f"step_{step}"


def _components_named_params(model: LMComponentModel) -> list[tuple[str, nn.Parameter]]:
    """``(name, param)`` for the V/U params, in trainer order. Names match the core
    ``Trainer._components_optimizer_named_params`` (``components.<path>.<pname>``)
    so the consolidated optimizer state resumes into either trainer."""
    out: list[tuple[str, nn.Parameter]] = []
    for module_path in model.target_module_paths:
        for pname, p in model.components[module_path].named_parameters():
            out.append((f"components.{module_path}.{pname}", p))
    return out


def _ci_fn_named_params(model: LMComponentModel) -> list[tuple[str, nn.Parameter]]:
    assert model.ci_fn is not None
    return [(f"ci_fn.{n}", p) for n, p in model.ci_fn.named_parameters()]


def _build_optimizers(model: LMComponentModel) -> dict[str, Optimizer]:
    """Fresh optimizers over the trainable params, split + ordered like the trainer.

    LR / betas / weight-decay are irrelevant here — only the param→FQN mapping
    matters, since this optimizer is just a load target whose state is then read
    back. The hyperparameters that govern training are restored from the config on
    resume, not from this throwaway optimizer.
    """
    component_params = [p for _, p in _components_named_params(model)]
    ci_fn_params = [p for _, p in _ci_fn_named_params(model)]
    return {
        COMPONENTS_OPTIMIZER_NAME: AdamW(component_params),
        CI_FN_OPTIMIZER_NAME: AdamW(ci_fn_params),
    }


def consolidate(
    run_dir: Path,
    step: int,
    *,
    build_full_model: Callable[[], LMComponentModel],
    pd_config: PDConfig,
    runtime_config_dump: dict[str, Any],
    keep_last_n_training: int | None,
) -> None:
    """Assemble + persist the step-``step`` checkpoint from its DCP shards.

    No-op when ``training_<step>.pth`` already exists. ``build_full_model`` returns
    a fresh unsharded CPU ``LMComponentModel`` (the assembly buffer);
    ``pd_config`` / ``runtime_config_dump`` populate the ``TrainingState`` configs
    (the DCP shards carry only model/optimizer/step/loss-metric state, not configs).
    ``keep_last_n_training`` prunes old ``training_<step>.pth`` (``None`` keeps all);
    ``model_<step>.pth`` files are never pruned.
    """
    training_path = run_dir / f"training_{step}.pth"
    if training_path.is_file():
        logger.info(f"consolidate: {training_path.name} already exists; skipping")
        return

    step_dir = _step_dir(run_dir, step)
    assert step_dir.is_dir(), (
        f"consolidate: no DCP shards at {step_dir} and no {training_path.name}"
    )

    model = build_full_model()
    assert isinstance(model, LMComponentModel)
    optimizers = _build_optimizers(model)

    # Load only the model + optimizer shards (+ step). Loss-metric states are NOT
    # reconstructed here: DCP loads a non-tensor/tensor leaf only into a key that
    # already exists in the destination, and a faithful loss-metric skeleton would
    # require instantiating + binding the metrics. The heavy loss-metric state is
    # the PPGD persistent sources (data-shaped) — the consolidated TrainingState
    # deliberately omits them (cf. the 3-pool's PPGD-shard policy). Resume-in-place
    # restores loss-metric state from the DCP shards via `load_dcp`; cross-run
    # resume from this artifact re-warms the adversary.
    model_sd = get_model_state_dict(model, options=_TRAINABLE_ONLY)
    optim_sds = {
        name: get_optimizer_state_dict(model, opt, options=_TRAINABLE_ONLY)
        for name, opt in optimizers.items()
    }
    state_dict = {
        "model": model_sd,
        "optimizers": optim_sds,
        "step": 0,
    }
    dcp.load(state_dict, checkpoint_id=str(step_dir))
    loaded_step = cast(int, state_dict["step"])
    assert loaded_step == step, f"consolidate: DCP step {loaded_step} != requested {step}"

    # `dcp.load` mutated `model_sd` / `optim_sds` in place; read them back through
    # the typed locals (not the heterogeneous combined dict) for concrete types.
    set_model_state_dict(model, model_sd, options=_TRAINABLE_ONLY)
    for name, opt in optimizers.items():
        set_optimizer_state_dict(
            model, opt, optim_state_dict=optim_sds[name], options=_TRAINABLE_ONLY
        )

    full_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    components_optimizer = optimizer_state_by_name(
        optimizers[COMPONENTS_OPTIMIZER_NAME], _components_named_params(model)
    )
    ci_fn_optimizer = optimizer_state_by_name(
        optimizers[CI_FN_OPTIMIZER_NAME], _ci_fn_named_params(model)
    )

    training_state = TrainingState(
        step=step,
        pd_config=pd_config.model_dump(),
        runtime_config=runtime_config_dump,
        component_model=full_model_state,
        components_optimizer=components_optimizer,
        ci_fn_optimizer=ci_fn_optimizer,
        loss_metrics={},
    )

    model_path = run_dir / f"model_{step}.pth"
    save_file(full_model_state, model_path)
    save_file(training_state, training_path)
    logger.info(f"consolidate: wrote {model_path.name} + {training_path.name}")

    if keep_last_n_training is not None:
        _prune_old_training(run_dir, keep_last_n=keep_last_n_training)


def _prune_old_training(run_dir: Path, *, keep_last_n: int) -> None:
    """Delete all but the last ``keep_last_n`` ``training_<step>.pth`` files.

    ``model_<step>.pth`` files are never pruned — they're the downstream artifact.
    """
    steps: list[int] = []
    for p in run_dir.glob("training_*.pth"):
        try:
            steps.append(int(p.stem.removeprefix("training_")))
        except ValueError:
            continue
    steps.sort()
    if len(steps) <= keep_last_n:
        return
    for step in steps[: len(steps) - keep_last_n]:
        # missing_ok: a concurrent per-step consolidation may have pruned it already.
        (run_dir / f"training_{step}.pth").unlink(missing_ok=True)
        logger.info(f"consolidate: pruned old training_{step}.pth")
