"""Internal step-context bundle for 2-pool training.

``_TwoPoolRuntime`` glues serializable config (``TwoPoolConfig``) with the
caller-supplied runtime callables and the derived per-site C from the actual
target. It's the parameter shape that ``step_pool_a``/``step_pool_b`` consume;
``optimize_two_pool`` builds one internally per call.

Not part of the public API.
"""

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from param_decomp.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.ci_fns import CiConfig
from param_decomp.ci_sigmoids import SigmoidType
from param_decomp.metrics.persistent_pgd_recon import PersistentPGDReconLossConfig
from param_decomp.two_pool.layout import BlockGroup

__all__ = ["_TwoPoolRuntime", "autocast_bf16"]


@dataclass(frozen=True)
class _TwoPoolRuntime:
    """Internal bundle passed to ``step_pool_a``/``step_pool_b``."""

    block_groups: tuple[BlockGroup, ...]
    pool_b_ranks: tuple[int, ...]
    batch_global: int
    c_per_site: dict[str, int]
    ci_config: CiConfig
    sigmoid_type: SigmoidType
    run_batch: RunBatch
    reconstruction_loss: ReconstructionLoss
    ppgd_cfg: PersistentPGDReconLossConfig
    coeff_faith: float
    coeff_imp: float
    coeff_stoch: float
    coeff_ppgd: float
    imp_min_pnorm: float
    imp_min_beta: float
    imp_min_eps: float
    imp_min_p_anneal_start_frac: float
    imp_min_p_anneal_final_p: float | None
    imp_min_p_anneal_end_frac: float
    lr_components: float
    lr_ci_fn: float
    grad_clip_norm_components: float | None
    grad_clip_norm_ci_fn: float | None
    bf16_autocast: bool
    use_fused_kl: bool


def autocast_bf16(enabled: bool) -> AbstractContextManager[Any]:
    """bf16 autocast on CUDA when enabled, no-op otherwise.

    Wrapping target/CI forward + PPGD forward in bf16 unlocks PyTorch's
    flash/cudnn SDPA backends (math is the only fp32 backend on H200 → ~6×
    slower attention). Faithfulness loss (small-number-sensitive) is kept
    OUTSIDE this block by the step functions.
    """
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
