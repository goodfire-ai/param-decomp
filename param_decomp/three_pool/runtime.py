"""Internal step-context bundle for 3-pool training.

``_ThreePoolRuntime`` glues serializable config (``ThreePoolConfig``) with the
caller-supplied runtime callables and the derived per-site C from the actual
target. It's the parameter shape that ``step_ci`` / ``step_layerwise`` /
``step_ppgd`` consume; ``optimize_three_pool`` builds one internally per call.

Not part of the public API.
"""

from dataclasses import dataclass

from param_decomp.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.ci_fns import CiConfig
from param_decomp.ci_sigmoids import SigmoidType
from param_decomp.metrics.persistent_pgd_recon import PersistentPGDReconLossConfig
from param_decomp.three_pool.layout import LayerwiseBlockGroup

__all__ = ["_ThreePoolRuntime"]


@dataclass(frozen=True)
class _ThreePoolRuntime:
    """Internal bundle passed to the three step functions.

    Mirrors ``two_pool.runtime._TwoPoolRuntime`` but typed against the
    3-pool topology + the multi-rank CI pool. Coefficients and per-metric
    config are flattened in here so step functions don't have to re-parse
    ``pd_config.loss_metrics``.
    """

    # Topology
    ci_ranks: tuple[int, ...]
    layerwise_block_groups: tuple[LayerwiseBlockGroup, ...]
    ppgd_ranks: tuple[int, ...]
    batch_global: int

    # Per-site C (resolved from decomposition_targets)
    c_per_site: dict[str, int]

    # CI fn shape
    ci_config: CiConfig
    sigmoid_type: SigmoidType

    # Caller-supplied callables
    run_batch: RunBatch
    reconstruction_loss: ReconstructionLoss

    # PPGD config (the full LossMetricConfig, since PersistentPGDState
    # consumes several of its fields)
    ppgd_cfg: PersistentPGDReconLossConfig

    # Loss coefficients
    coeff_faith: float
    coeff_imp: float
    coeff_stoch: float
    coeff_ppgd: float

    # Importance minimality knobs
    imp_min_pnorm: float
    imp_min_beta: float
    imp_min_eps: float
    imp_min_p_anneal_start_frac: float
    imp_min_p_anneal_final_p: float | None
    imp_min_p_anneal_end_frac: float

    # Optimizer LRs (start values; schedules consumed by optimize_three_pool
    # which mutates optimizer.param_groups[i]['lr'] each step)
    lr_components: float
    lr_ci_fn: float

    # Substrate
    bf16_autocast: bool
    use_fused_kl: bool
