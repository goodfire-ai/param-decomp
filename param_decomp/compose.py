"""Composition root: turn serializable configs into runtime objects.

This module is the single point where driver resolution and config-to-runtime
materialization happens. It keeps RunConfig as pure data (no driver loading in
from_dict) and makes the boundary explicit.

The two entry points:
- ``resolve_run(data)`` — parse raw dict into (RunConfig, ExperimentDriver)
- ``materialize_run(run_cfg, driver, ...)`` — turn config into runtime objects
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

from param_decomp.driver_path import load_driver
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.utils.distributed_utils import DistributedState

if TYPE_CHECKING:
    from param_decomp.experiments.driver import ExperimentDriver
    from param_decomp.run import RunConfig


@dataclass(frozen=True)
class RuntimeInputs:
    """Runtime objects built from a RunConfig via a driver."""

    target: PDTarget
    train_loader: DataLoader[Any]
    eval_loader: DataLoader[Any]


def resolve_run(data: dict[str, Any]) -> tuple["RunConfig", "ExperimentDriver[Any]"]:
    """Parse a dict into (RunConfig subtype, ExperimentDriver).

    Single composition root: loads the driver once, uses its ``config_type`` to
    validate, returns both. Callers who need the concrete RunConfig subtype
    narrow with ``isinstance(run_cfg, driver.config_type)``.

    Raises KeyError if ``driver_path`` is missing (required field).
    """
    from param_decomp.experiments.driver import ExperimentDriver
    from param_decomp.run import RunConfig

    driver_path = data["driver_path"]
    driver: ExperimentDriver[RunConfig] = load_driver(driver_path)
    run_cfg = driver.config_type.model_validate(data)
    return run_cfg, driver


def materialize_run(
    run_cfg: "RunConfig",
    driver: "ExperimentDriver[Any]",
    *,
    device: str,
    dist_state: DistributedState | None = None,
) -> RuntimeInputs:
    """Turn a RunConfig into runtime objects using the driver.

    The driver builds target model, train loader, and eval loader from the
    config. The target model is moved to the device.
    """
    target = driver.build_target(run_cfg)
    target.model.to(device)
    train_loader = driver.build_train_loader(run_cfg, device=device, dist_state=dist_state)
    eval_loader = driver.build_eval_loader(run_cfg, device=device, dist_state=dist_state)
    return RuntimeInputs(target=target, train_loader=train_loader, eval_loader=eval_loader)
