"""`ExperimentSpec`: lab-side compositional registry for experiments.

An `ExperimentSpec` is a plain dataclass that bundles the callables and pydantic types a
downstream tool needs to (re)build the target model and dataloaders for a given experiment.
It is **not** an inheritance hook — experiments register by constructing an `ExperimentSpec`
value with function pointers, not by subclassing.

Each in-repo experiment module exposes `EXPERIMENT_SPEC: ExperimentSpec` at the top level;
``EXPERIMENTS`` in :mod:`param_decomp_lab.experiments` is the dispatch dict keyed by name.

External users add their own experiments by constructing an `ExperimentSpec` in their package
and either adding it to `EXPERIMENTS` themselves or passing it directly to lab tools that
accept an `ExperimentSpec` argument.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss, RunBatch


@dataclass(frozen=True)
class ExperimentSpec:
    """How a saved PD run can be rebuilt from its serialized configs.

    Compositional: experiments populate this with plain functions. Lab-side tools
    (SavedRun, harvest, etc.) call these to materialize runtime objects.
    """

    name: str
    target_config_cls: type[BaseConfig]
    data_config_cls: type[BaseConfig]

    build_target: Callable[[Any], nn.Module]
    """Takes a validated target config; returns the target nn.Module."""

    build_train_loader: Callable[..., DataLoader[Any]]
    """Signature: (target_cfg, data_cfg, *, batch_size, device, dist_state=None, seed=0)."""

    build_eval_loader: Callable[..., DataLoader[Any]]
    """Signature: (target_cfg, data_cfg, *, batch_size, device, dist_state=None, seed=0)."""

    make_run_batch: Callable[[Any], RunBatch]
    """Takes target_cfg, returns the RunBatch callable (lets LM pick make_run_batch(output_extract))."""

    reconstruction_loss: ReconstructionLoss
    """Reconstruction loss for this experiment (typically `recon_loss_mse` or `recon_loss_kl`)."""

    def validate_target(self, raw: dict[str, Any]) -> Any:
        return self.target_config_cls.model_validate(raw)

    def validate_data(self, raw: dict[str, Any]) -> Any:
        return self.data_config_cls.model_validate(raw)
