"""Lab-side orchestration for targeted parameter decomposition (tPD)."""

from typing import Any

from pydantic import NonNegativeFloat, PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp.configs import AnyLossMetricConfig
from param_decomp.distributed import get_distributed_state
from param_decomp.metrics.base import Metric
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.metrics.persistent_pgd_recon import (
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    validate_pgd_scope,
)
from param_decomp.metrics.stochastic_hidden_acts_recon import StochasticHiddenActsReconLossConfig
from param_decomp.metrics.unmasked_recon import UnmaskedReconLossConfig


class NontargetConfig[D: BaseConfig](BaseConfig):
    """Nontarget-pass settings for a targeted decomposition run.

    `data` reuses the experiment's data-config type and describes the broad nontarget
    distribution; `impmin_coeff_ratio` scales the importance-minimality coeff on the
    nontarget pass relative to the target pass.
    """

    data: D
    batch_size: PositiveInt
    eval_batch_size: PositiveInt
    impmin_coeff_ratio: NonNegativeFloat = 1.0


_EXCLUDED_NONTARGET_LOSS_CONFIGS = (
    UnmaskedReconLossConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    StochasticHiddenActsReconLossConfig,
)


def build_nontarget_loss_configs(
    loss_metrics: list[AnyLossMetricConfig],
    impmin_ratio: float,
    *,
    nontarget_batch_size: int,
) -> list[AnyLossMetricConfig]:
    """Derive the nontarget-pass loss set from the target-pass loss configs.

    Drops losses that are meaningless or unsafe with a forced-on delta (unmasked recon,
    both PPGD losses, hidden-acts recon); scales the importance-minimality coeff by
    `impmin_ratio`. The result must retain a full-model recon loss so the nontarget
    backward grads every parameter (a DDP requirement under the default reducer).
    """
    out: list[AnyLossMetricConfig] = []
    for cfg in loss_metrics:
        if isinstance(cfg, _EXCLUDED_NONTARGET_LOSS_CONFIGS):
            continue
        if isinstance(cfg, ImportanceMinimalityLossConfig) and cfg.coeff is not None:
            cfg = cfg.model_copy(update={"coeff": cfg.coeff * impmin_ratio})
        else:
            cfg = cfg.model_copy()
        out.append(cfg)

    dist_state = get_distributed_state()
    validate_pgd_scope(
        out,
        batch_size=nontarget_batch_size,
        world_size=dist_state.world_size if dist_state is not None else 1,
    )
    return out


def split_eval_metrics(
    metrics: list[Metric[Any]],
) -> tuple[list[Metric[Any]], list[Metric[Any]]]:
    """Partition instantiated eval metrics into `(target_metrics, nontarget_metrics)`.

    Routing reads each metric's `eval_distribution` marker: `"nontarget"` metrics go
    right (fed by the mirror nontarget eval loop under `delta_override(1.0)`); everything
    else goes left (normal target eval pass). New nontarget metrics just set the marker —
    no edit here is needed.
    """
    target_metrics = [m for m in metrics if m.eval_distribution != "nontarget"]
    nontarget_metrics: list[Metric[Any]] = [
        m for m in metrics if m.eval_distribution == "nontarget"
    ]
    return target_metrics, nontarget_metrics
