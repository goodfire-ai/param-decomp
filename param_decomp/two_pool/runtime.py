"""Internal runtime bundle for 2-pool training.

``_TwoPoolRuntime`` glues serializable config (``TwoPoolConfig``) with the
runtime callables from ``PDTarget`` and the derived per-site C from the
actual installed module_path_info. ``build_two_pool_runtime`` constructs it
from the driver-extracted bits, and ``optimize_two_pool`` is the only public
entry point.

``_autocast`` + ``_seq_dims_from_batch_iter`` live here because they're shared
across the pool A/B step modules and need a stable canonical home.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from param_decomp.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.ci_fns import CiConfig
from param_decomp.ci_sigmoids import SigmoidType
from param_decomp.metrics.persistent_pgd_recon import PersistentPGDReconLossConfig
from param_decomp.two_pool.config import TwoPoolConfig
from param_decomp.two_pool.layout import BlockGroup


@dataclass(frozen=True)
class _TwoPoolRuntime:
    """Internal bundle passed to step_pool_a/b. Mixes serializable config (from
    ``TwoPoolConfig``) with the runtime callables from ``PDTarget`` and the
    derived per-site C from the actual installed module_path_info.

    Not part of the public API — call ``optimize_two_pool(target, run_cfg, ...)``
    from outside and let it build this internally.
    """

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
    bf16_autocast: bool
    use_fused_kl: bool


def autocast_bf16(enabled: bool) -> AbstractContextManager[Any]:
    """bf16 autocast on CUDA when enabled, no-op otherwise.

    Wrapping the target/CI forward + PPGD forward in bf16 unlocks PyTorch's
    flash/cudnn SDPA backends (math is the only fp32 backend on H200 → ~6×
    slower attention). Per microbench at b=66/s=1024, target_fwd drops 57 → 27ms.

    The faithfulness loss (‖W − VU.T‖²) is kept OUTSIDE this block because
    it's small-number-sensitive; everything else is robust to mixed precision.
    """
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def seq_dims_from_batch_iter(batch_iter: Callable[[int], Any]) -> tuple[int, ...]:
    """Peek at a single batch to determine seq dim(s). Assumes restartable iter.

    For token batches the shape is (B, S); returns (S,). For non-tensor batches
    the caller should supply batch_dims directly to PersistentPGDState.
    """
    sample = batch_iter(0)
    if isinstance(sample, Tensor):
        return tuple(sample.shape[1:])
    raise TypeError(f"Cannot infer seq dims from batch of type {type(sample).__name__}")


def build_two_pool_runtime(
    pool_config: TwoPoolConfig,
    *,
    batch_global: int,
    c_per_site: dict[str, int],
    ci_config: CiConfig,
    sigmoid_type: SigmoidType,
    run_batch: RunBatch,
    reconstruction_loss: ReconstructionLoss,
    ppgd_cfg: PersistentPGDReconLossConfig,
    coeff_faith: float,
    coeff_imp: float,
    coeff_stoch: float,
    coeff_ppgd: float,
    imp_min_pnorm: float,
    imp_min_beta: float,
    imp_min_eps: float,
    imp_min_p_anneal_start_frac: float,
    imp_min_p_anneal_final_p: float | None,
    imp_min_p_anneal_end_frac: float,
    lr_components: float,
    lr_ci_fn: float,
    bf16_autocast: bool,
    use_fused_kl: bool,
) -> _TwoPoolRuntime:
    """Glue: ``TwoPoolConfig`` + runtime callables + coefficients → runtime bundle.

    The driver path (``run_two_pool``) calls this after extracting everything from
    a ``RunConfig``; the toy benchmark scripts construct each arg directly.
    """
    block_groups = tuple(
        BlockGroup(ranks=tuple(bg.ranks), owned_sites=tuple(bg.owned_sites))
        for bg in pool_config.block_groups
    )
    return _TwoPoolRuntime(
        block_groups=block_groups,
        pool_b_ranks=tuple(pool_config.pool_b_ranks),
        batch_global=batch_global,
        c_per_site=c_per_site,
        ci_config=ci_config,
        sigmoid_type=sigmoid_type,
        run_batch=run_batch,
        reconstruction_loss=reconstruction_loss,
        ppgd_cfg=ppgd_cfg,
        coeff_faith=coeff_faith,
        coeff_imp=coeff_imp,
        coeff_stoch=coeff_stoch,
        coeff_ppgd=coeff_ppgd,
        imp_min_pnorm=imp_min_pnorm,
        imp_min_beta=imp_min_beta,
        imp_min_eps=imp_min_eps,
        imp_min_p_anneal_start_frac=imp_min_p_anneal_start_frac,
        imp_min_p_anneal_final_p=imp_min_p_anneal_final_p,
        imp_min_p_anneal_end_frac=imp_min_p_anneal_end_frac,
        lr_components=lr_components,
        lr_ci_fn=lr_ci_fn,
        bf16_autocast=bf16_autocast,
        use_fused_kl=use_fused_kl,
    )
