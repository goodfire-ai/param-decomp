"""Driver-mediated entry point for 2-pool training.

Standalone sibling to ``run_pd`` — does NOT mutate ``RunConfig`` or
``run_pd``. A 2-pool run is configured exactly like a normal SPD run
(``RunConfig`` with the standard ``pd`` / ``logging`` / ``runtime`` blocks
and ``loss_metrics``) plus a separate topology block (``TwoPoolConfig``).

External behaviour goal: same loss metrics, same LR schedules, same
faithfulness warmup, same logging shape as the single-pool path — just
under a different parallelism structure.

``validate_run_cfg_for_two_pool`` runs first and rejects:
  - any RunConfig missing one of the four loss metrics 2-pool implements
  - any RunConfig containing a loss metric 2-pool would silently ignore
  - batch_size that doesn't divide evenly across the topology
"""

# pyright: reportPrivateUsage=false

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from param_decomp.configs import PersistentPGDReconLossConfig
from param_decomp.driver_path import load_driver
from param_decomp.metrics.builtin.importance_minimality_loss import (
    ImportanceMinimalityLossConfig,
)
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.run import RunConfig
from param_decomp.run_sink import RunSink
from param_decomp.two_pool.config import TwoPoolConfig
from param_decomp.two_pool.run import (
    PhaseProfiler,
    build_two_pool_runtime,
    optimize_two_pool,
)
from param_decomp.utils.module_utils import expand_module_patterns

# Loss metrics the 2-pool training path implements. Each MUST be present in
# ``run_cfg.pd.loss_metrics`` with a non-None ``coeff``.
REQUIRED_LOSS_METRICS: tuple[str, ...] = (
    "FaithfulnessLoss",
    "ImportanceMinimalityLoss",
    "StochasticReconLayerwiseLoss",
    "PersistentPGDReconLoss",
)

# Loss metrics that would be silently ignored if set under a 2-pool RunConfig.
# Listed explicitly so misconfiguration is loud, not silent. If you want one
# of these to work under 2-pool, implement it in the 2-pool step functions
# and move it to REQUIRED_LOSS_METRICS (or a new OPTIONAL_LOSS_METRICS tier).
FORBIDDEN_LOSS_METRICS = (
    "StochasticReconLoss",
    "StochasticReconSubsetLoss",
    "StochasticReconSubsetCEAndKL",
    "PersistentPGDReconSubsetLoss",
    "CIMaskedReconLoss",
    "CIMaskedReconLayerwiseLoss",
    "CIMaskedReconSubsetLoss",
    "UnmaskedReconLoss",
    "PGDReconLoss",
    "PGDReconLayerwiseLoss",
    "PGDReconSubsetLoss",
    "CIMaskedAttnPatternsReconLoss",
    "StochasticAttnPatternsReconLoss",
    "StochasticHiddenActsReconLoss",
    "CIHiddenActsReconLoss",
)


def validate_run_cfg_for_two_pool(run_cfg: RunConfig, two_pool_cfg: TwoPoolConfig) -> None:
    """Fail loudly on any RunConfig misconfiguration the 2-pool path can't honour.

    Checks:
      1. ``pd.loss_metrics`` contains all of REQUIRED_LOSS_METRICS, each with a
         non-None ``coeff``.
      2. ``pd.loss_metrics`` contains none of FORBIDDEN_LOSS_METRICS (would be
         silently ignored otherwise).
      3. ``pd.batch_size`` divides evenly by ``N_per_block`` and by ``N_pool_b``.
    """
    pd = run_cfg.pd
    have = set(pd.loss_metrics)

    missing = sorted(set(REQUIRED_LOSS_METRICS) - have)
    assert not missing, (
        f"2-pool requires these metrics in pd.loss_metrics: {sorted(REQUIRED_LOSS_METRICS)}.\n"
        f"Missing: {missing}. Got: {sorted(have)}."
    )

    for name in REQUIRED_LOSS_METRICS:
        cfg = pd.loss_metrics[name]
        assert getattr(cfg, "coeff", None) is not None, (
            f"pd.loss_metrics[{name!r}].coeff is required for 2-pool training"
        )

    illegal = sorted(set(FORBIDDEN_LOSS_METRICS) & have)
    assert not illegal, (
        f"2-pool does not implement these loss metrics (they would be silently ignored): "
        f"{illegal}.\nRemove from pd.loss_metrics or extend the 2-pool path to handle them."
    )

    n_per_block = len(two_pool_cfg.block_groups[0].ranks)
    n_pool_b = len(two_pool_cfg.pool_b_ranks)
    bs = pd.batch_size
    assert bs % n_per_block == 0, (
        f"pd.batch_size ({bs}) must be divisible by N_per_block ({n_per_block}) "
        f"= len(block_groups[0].ranks)"
    )
    assert bs % n_pool_b == 0, (
        f"pd.batch_size ({bs}) must be divisible by N_pool_b ({n_pool_b}) = len(pool_b_ranks)"
    )

    assert pd.use_delta_component, (
        "2-pool path requires pd.use_delta_component=True (it's hardcoded in pool A's "
        "layerwise stoch recon + pool B's PPGD)."
    )


def run_two_pool(
    *,
    run_cfg: RunConfig,
    two_pool_cfg: TwoPoolConfig,
    target: PDTarget,
    train_loader: DataLoader[Any],
    device: str,
    profiler: PhaseProfiler | None = None,
    wandb_project: str | None = None,
) -> None:
    """Run 2-pool training given a fully-materialized RunConfig + TwoPoolConfig.

    Mirrors ``run_pd``'s composition shape — caller resolves the driver and
    materializes ``target`` / ``train_loader`` (typically via ``materialize_run``),
    we read everything else off ``run_cfg`` and hand to ``optimize_two_pool``.

    Eval loop isn't wired through yet — needs pool-aware MetricContext.
    """
    validate_run_cfg_for_two_pool(run_cfg, two_pool_cfg)

    driver = load_driver(run_cfg.driver_path)
    sink = RunSink.for_run(run_cfg, wandb_project=wandb_project, driver=driver)

    pd = run_cfg.pd
    runtime = run_cfg.runtime

    # ── Per-site C derived from PDConfig.module_info against the actual target. ──
    module_path_info = expand_module_patterns(target.model, pd.all_module_info)
    c_per_site = {info.module_path: info.C for info in module_path_info}
    for bg in two_pool_cfg.block_groups:
        for site in bg.owned_sites:
            assert site in c_per_site, (
                f"site '{site}' in two_pool topology but not in pd.module_info "
                f"after pattern expansion. Available: {sorted(c_per_site)[:5]}..."
            )

    # ── Loss coefficients + PPGD config from the regular pd.loss_metrics block.
    #    validate_run_cfg_for_two_pool already asserted coeff is non-None, so
    #    these casts are safe. ──
    def _coeff(name: str) -> float:
        c = pd.loss_metrics[name].coeff
        assert c is not None  # validated above
        return float(c)

    coeff_faith = _coeff("FaithfulnessLoss")
    coeff_imp = _coeff("ImportanceMinimalityLoss")
    coeff_stoch = _coeff("StochasticReconLayerwiseLoss")
    coeff_ppgd = _coeff("PersistentPGDReconLoss")
    ppgd_cfg = pd.loss_metrics["PersistentPGDReconLoss"]
    assert isinstance(ppgd_cfg, PersistentPGDReconLossConfig), (
        f"pd.loss_metrics['PersistentPGDReconLoss'] must be PersistentPGDReconLossConfig, "
        f"got {type(ppgd_cfg).__name__}"
    )
    imp_min_cfg = pd.loss_metrics["ImportanceMinimalityLoss"]
    assert isinstance(imp_min_cfg, ImportanceMinimalityLossConfig), (
        f"pd.loss_metrics['ImportanceMinimalityLoss'] must be ImportanceMinimalityLossConfig, "
        f"got {type(imp_min_cfg).__name__}"
    )
    assert ppgd_cfg.start_frac == 0.0, (
        "2-pool path does not implement PersistentPGDReconLoss.start_frac > 0 "
        "(PPGD always runs from step 0). Set start_frac to 0 or add gating."
    )

    # ── Optimizer LRs from pd.{components,ci_fn}_optimizer.lr_schedule.start_val. ──
    #    The LR schedule itself is threaded into optimize_two_pool so the loop
    #    can call get_scheduled_value per step (same as run_pd.optimize).
    components_lr_schedule = pd.components_optimizer.lr_schedule
    ci_fn_lr_schedule = pd.ci_fn_optimizer.lr_schedule

    pool_runtime = build_two_pool_runtime(
        two_pool_cfg,
        batch_global=pd.batch_size,
        c_per_site=c_per_site,
        ci_config=pd.ci_config,
        sigmoid_type=pd.sigmoid_type,
        run_batch=target.run_batch,
        reconstruction_loss=target.reconstruction_loss,
        ppgd_cfg=ppgd_cfg,
        coeff_faith=coeff_faith,
        coeff_imp=coeff_imp,
        coeff_stoch=coeff_stoch,
        coeff_ppgd=coeff_ppgd,
        imp_min_pnorm=imp_min_cfg.pnorm,
        imp_min_beta=imp_min_cfg.beta,
        imp_min_eps=imp_min_cfg.eps,
        imp_min_p_anneal_start_frac=imp_min_cfg.p_anneal_start_frac,
        imp_min_p_anneal_final_p=imp_min_cfg.p_anneal_final_p,
        imp_min_p_anneal_end_frac=imp_min_cfg.p_anneal_end_frac,
        lr_components=components_lr_schedule.start_val,
        lr_ci_fn=ci_fn_lr_schedule.start_val,
        bf16_autocast=runtime.autocast_bf16,
        use_fused_kl=two_pool_cfg.use_fused_kl,
    )

    # ── DataLoader → batch_iter bridge. Pool A's owned_sites slicing happens
    #    inside optimize_two_pool; here we just yield full-batch tensors. ──
    loader_iter = iter(train_loader)

    def batch_iter(_step: int) -> Tensor:
        nonlocal loader_iter
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        if isinstance(batch, Tensor):
            return batch.to(device)
        if isinstance(batch, dict) and "input_ids" in batch:
            return batch["input_ids"].to(device)
        if isinstance(batch, list | tuple) and len(batch) > 0 and isinstance(batch[0], Tensor):
            return batch[0].to(device)
        raise TypeError(f"Unsupported batch type from DataLoader: {type(batch).__name__}")

    optimize_two_pool(
        target_model=target.model,
        pool_config=pool_runtime,
        device=torch.device(device),
        n_steps=pd.steps,
        batch_iter=batch_iter,
        components_lr_schedule=components_lr_schedule,
        ci_fn_lr_schedule=ci_fn_lr_schedule,
        faithfulness_warmup_steps=pd.faithfulness_warmup_steps,
        faithfulness_warmup_lr=pd.faithfulness_warmup_lr,
        faithfulness_warmup_weight_decay=pd.faithfulness_warmup_weight_decay,
        profiler=profiler,
        sink=sink,
        logging_config=run_cfg.logging,
    )

    sink.finish()


__all__ = ["run_two_pool", "validate_run_cfg_for_two_pool"]
