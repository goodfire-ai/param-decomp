"""Watertight per-term equivalence: the flat (single-pool FSDP) PD step's loss terms
== the torch 2-pool's, on a fixed-seed tiny model. Standardises on the 2-pool's
semantics; where the flat path drives a core ``Metric`` and the 2-pool a pool-step
helper, this asserts they produce the SAME value AND the SAME gradient.

The recon (stochastic) term is proven separately and bit-identically in
``test_chunkwise_subset_recon_metric.py`` (the flat ``ChunkwiseSubsetReconLoss`` ==
the 2-pool's ``_run_routing_forwards``). Here we cover the other three terms:

  * **Faithfulness** — flat ``faithfulness_loss`` (core ``FaithfulnessLoss``) ==
    2-pool ``_faithfulness_loss`` (``step_chunkwise``). Both are ``Σ‖W−VU‖² /
    numel_global``; no RNG. Asserts value + V/U grad.
  * **Importance-minimality** — flat ``ImportanceMinimalityLoss`` (core, single-process)
    == 2-pool ``_importance_minimality_loss`` (``step_ci``). Both compose the SAME
    ``annealed_pnorm`` + ``per_component_lp_sums`` + ``finalize_imp_min``; at ``n_ci=1``
    the residual-trick global-sum is a no-op. Asserts value + CI grad, incl. the
    p-anneal schedule mid-anneal.
  * **PPGD** — flat ``PersistentPGDReconLoss`` (core metric;
    ``update``/``after_backward``) == the 2-pool adversary
    (``_warmup_and_recon`` + ``_autograd_grads_wrt_vu_ci_and_sources`` + ``_scale_grads``
    + ``state.step``). Both drive the SAME ``PersistentPGDState`` (same warmup, same
    ``mask = ci + (1−ci)·source`` interpolation, same minimax source step). PPGD recon
    is RNG-free given the initial sources, so copying the sources makes the two paths
    deterministic. At ``n_a = n_ppgd = 1`` the 2-pool's per-rank scale
    ``coeff/(n_examples·n_a)`` (V/U + CI) and ``1/n_examples`` (sources) coincide with
    the flat metric's ``∂(coeff·sum/n)``: the sources are graph leaves, so the one outer
    backward leaves ``coeff·∂(sum/n)/∂source`` in ``source.grad`` and ``after_backward``
    divides the coefficient back out. Asserts source / V/U / CI grads AND the stepped
    sources — this is the real grad check for the rerouted source gradient, against an
    independent raw-``autograd.grad`` reference. Parametrised over a power-of-two coeff
    (where backward-then-divide is bit-identical to the unscaled gradient, since scaling
    cotangents by 2^k is exact) and a non-power-of-two coeff (fp rounding, within
    ``assert_close`` tolerance).
"""

from typing import Any, cast, override

import pytest
import torch
import torch.nn as nn

from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import AllLayersRouter
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.faithfulness import FaithfulnessLoss
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLoss
from param_decomp.metrics.persistent_pgd_recon import PersistentPGDReconLoss
from param_decomp.metrics.persistent_pgd_state import PersistentPGDState
from param_decomp_config.ci_fn import LayerwiseCiConfig
from param_decomp_config.losses import (
    AdamPGDConfig,
    BSCScope,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
)
from param_decomp_config.schedule import ScheduleConfig
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_passthrough
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.three_pool.layout import Chunk
from param_decomp_lab.three_pool.recon_loss_strategy import ReconLossStrategy
from param_decomp_lab.three_pool.recon_plan import PerSitePlan
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime
from param_decomp_lab.three_pool.step_chunkwise import _faithfulness_loss
from param_decomp_lab.three_pool.step_ci import _importance_minimality_loss
from param_decomp_lab.three_pool.step_ppgd import (
    _autograd_grads_wrt_vu_ci_and_sources,
    _releaf_ci_fp32_for_grads,
    _warmup_and_recon,
)

_SITES = ("l0", "l1", "l2")
_C = 3
_D = 4
_BATCH = 2
_SEQ = 5

_PPGD_WARMUP = 2
_PPGD_INIT_SEED = 55

_IMP_PNORM = 2.0
_IMP_BETA = 0.5
_IMP_EPS = 1e-12

_CI_CONFIG = LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2])


class _ThreeLinearSeq(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        for name in _SITES:
            setattr(self, name, nn.Linear(d, d, bias=False))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in _SITES:
            x = torch.relu(getattr(self, name)(x))
        return x


def _make_model() -> ComponentModel:
    target = _ThreeLinearSeq(_D)
    target.requires_grad_(False)
    return ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        decomposition_targets=[DecompositionTarget(module_path=s, C=_C) for s in _SITES],
        ci_config=_CI_CONFIG,
        sigmoid_type="leaky_hard",
    )


def _vu_grads(model: ComponentModel) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, comp in model.components.items():
        assert comp.V.grad is not None and comp.U.grad is not None
        out[f"{name}.V"] = comp.V.grad.clone()
        out[f"{name}.U"] = comp.U.grad.clone()
    return out


def _numel_global(model: ComponentModel) -> int:
    return sum(model.target_weight(s).numel() for s in _SITES)


def _leaf_grad(t: torch.Tensor) -> torch.Tensor:
    assert t.grad is not None
    return t.grad.clone()


def _zero_grads(model: ComponentModel) -> None:
    for p in model.parameters():
        p.grad = None


def _build_runtime(
    model: ComponentModel,
    ppgd_cfg: PersistentPGDReconLossConfig,
    imp_cfg: ImportanceMinimalityLossConfig,
) -> _ThreePoolRuntime:
    """A real (single-process) ``_ThreePoolRuntime`` for the tiny model.

    The 2-pool step helpers consume the full runtime; building a genuine one (rather
    than a stand-in) keeps the equivalence assertion honest and the test type-safe.
    Only the imp-min knobs (read by ``_importance_minimality_loss``) and ``bf16_autocast``
    (read by ``_warmup_and_recon``) are load-bearing here; the rest are filled to be valid.
    """
    chunk = Chunk(ranks=(0,), sites=tuple(_SITES))
    return _ThreePoolRuntime(
        ci_ranks=(0,),
        chunks=(chunk,),
        ppgd_ranks=(0,),
        batch_global=_BATCH,
        c_per_site={s: _C for s in _SITES},
        ci_config=_CI_CONFIG,
        sigmoid_type="leaky_hard",
        run_batch=run_batch_passthrough,
        reconstruction_loss=recon_loss_mse,
        ppgd_cfg=ppgd_cfg,
        recon_plan=PerSitePlan(),
        n_est=len(_SITES),
        coeff_faith=1.0,
        coeff_imp=imp_cfg.coeff or 1.0,
        coeff_stoch=1.0,
        coeff_ppgd=ppgd_cfg.coeff or 1.0,
        log_name_faith="FaithfulnessLoss",
        log_name_imp="ImportanceMinimalityLoss",
        log_name_stoch="StochasticReconLayerwiseLoss",
        log_name_ppgd="PersistentPGDReconLoss",
        imp_min_pnorm=imp_cfg.pnorm,
        imp_min_beta=imp_cfg.beta,
        imp_min_eps=imp_cfg.eps,
        imp_min_p_anneal_start_frac=imp_cfg.p_anneal_start_frac,
        imp_min_p_anneal_final_p=imp_cfg.p_anneal_final_p,
        imp_min_p_anneal_end_frac=imp_cfg.p_anneal_end_frac,
        lr_components=0.0,
        lr_ci_fn=0.0,
        grad_clip_norm_components=None,
        grad_clip_norm_ci_fn=None,
        numel_global=_numel_global(model),
        bf16_autocast=False,
        use_fused_kl=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Faithfulness
# ──────────────────────────────────────────────────────────────────────────────


def test_faithfulness_flat_matches_two_pool() -> None:
    """Flat ``faithfulness_loss`` value + V/U grad == 2-pool ``_faithfulness_loss``."""
    torch.manual_seed(0)
    model = _make_model()
    numel_global = _numel_global(model)
    device = torch.device("cpu")
    coeff = 3.0

    # --- flat: core FaithfulnessLoss over global numel (run_loss_step normalises by
    # the global param count via faithfulness_loss; coeff applied by the caller) ---
    flat = FaithfulnessLoss(FaithfulnessLossConfig(coeff=coeff))
    flat.bind(model=model, device="cpu")
    _zero_grads(model)
    deltas_flat = model.calc_weight_deltas()
    ctx = _faith_ctx(model, deltas_flat, device)
    flat_loss = flat.update(ctx)
    (coeff * flat_loss).backward()
    vu_flat = _vu_grads(model)

    # --- 2-pool: _faithfulness_loss (numel_global denom) + chunk-leader backward ---
    _zero_grads(model)
    twopool_loss, _, _ = _faithfulness_loss(
        cast(LMComponentModel, cast(object, model)), device, numel_global
    )
    (coeff * twopool_loss).backward()
    vu_2p = _vu_grads(model)

    torch.testing.assert_close(flat_loss.detach(), twopool_loss.detach(), msg="faith value")
    for k in vu_flat:
        torch.testing.assert_close(vu_flat[k], vu_2p[k], msg=f"faith V/U grad: {k}")


def _faith_ctx(
    model: ComponentModel, weight_deltas: dict[str, torch.Tensor], device: torch.device
) -> MetricContext:
    """A minimal MetricContext for FaithfulnessLoss, which reads only weight_deltas
    (ci is required by the type but unused by the faithfulness path)."""
    from param_decomp.component_model import CIOutputs

    vals = {s: torch.rand(_BATCH, _SEQ, model.module_to_c[s], device=device) for s in _SITES}
    return MetricContext(
        model=cast(LMComponentModel, cast(object, model)),
        batch=torch.zeros(_BATCH, _SEQ, _D, device=device),
        target_out=torch.zeros(_BATCH, _SEQ, _D, device=device),
        pre_weight_acts={},
        ci=CIOutputs(lower_leaky=vals, upper_leaky=vals, pre_sigmoid=vals),
        weight_deltas=weight_deltas,
        step=0,
        total_steps=10,
        use_delta_component=True,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_mse,
        is_eval=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Importance-minimality
# ──────────────────────────────────────────────────────────────────────────────


def _run_imp(start_frac: float, final_p: float | None, step: int, total_steps: int) -> None:
    torch.manual_seed(1)
    model = _make_model()
    device = torch.device("cpu")
    coeff = 0.7
    ci_vals = {s: torch.rand(_BATCH, _SEQ, model.module_to_c[s]) for s in _SITES}

    cfg = ImportanceMinimalityLossConfig(
        coeff=coeff,
        pnorm=_IMP_PNORM,
        beta=_IMP_BETA,
        eps=_IMP_EPS,
        p_anneal_start_frac=start_frac,
        p_anneal_final_p=final_p,
        p_anneal_end_frac=1.0,
    )

    # --- flat: core ImportanceMinimalityLoss (single-process; no dist reduce) ---
    leaves_flat = {s: ci_vals[s].clone().requires_grad_(True) for s in _SITES}
    flat = ImportanceMinimalityLoss(cfg)
    flat.bind(model=model, device="cpu")
    ctx = _imp_ctx(model, leaves_flat, device, step, total_steps)
    flat.reset()
    flat_loss = flat.update(ctx)
    (coeff * flat_loss).backward()
    leaf_flat = {s: _leaf_grad(leaves_flat[s]) for s in _SITES}

    # --- 2-pool: _importance_minimality_loss (n_ci=1 → residual trick no-op).
    # The n_ci=1 single-process call never touches the CI-pool group, so the
    # group argument is unreached; the resolved layout always has a real group. ---
    leaves_2p = {s: ci_vals[s].clone().requires_grad_(True) for s in _SITES}
    runtime = _build_runtime(model, _ppgd_cfg(coeff=2.0), cfg)
    current_frac = step / total_steps
    twopool_loss = _importance_minimality_loss(
        leaves_2p,
        current_frac,
        runtime,
        ci_pool_group=cast("Any", None),
        n_ci_pool=1,
    )
    (coeff * twopool_loss).backward()
    leaf_2p = {s: _leaf_grad(leaves_2p[s]) for s in _SITES}

    torch.testing.assert_close(flat_loss.detach(), twopool_loss.detach(), msg="imp value")
    for s in _SITES:
        torch.testing.assert_close(leaf_flat[s], leaf_2p[s], msg=f"imp CI grad: {s}")


def test_importance_minimality_flat_matches_two_pool_no_anneal() -> None:
    """Flat ImportanceMinimalityLoss == 2-pool _importance_minimality_loss (no anneal)."""
    _run_imp(start_frac=1.0, final_p=None, step=3, total_steps=10)


def test_importance_minimality_flat_matches_two_pool_mid_anneal() -> None:
    """Same, mid p-anneal: both must read the SAME annealed p at this fraction."""
    _run_imp(start_frac=0.2, final_p=0.5, step=6, total_steps=10)


def _imp_ctx(
    model: ComponentModel,
    leaves: dict[str, torch.Tensor],
    device: torch.device,
    step: int,
    total_steps: int,
) -> MetricContext:
    from param_decomp.component_model import CIOutputs

    return MetricContext(
        model=cast(LMComponentModel, cast(object, model)),
        batch=torch.zeros(_BATCH, _SEQ, _D, device=device),
        target_out=torch.zeros(_BATCH, _SEQ, _D, device=device),
        pre_weight_acts={},
        ci=CIOutputs(lower_leaky=leaves, upper_leaky=leaves, pre_sigmoid=leaves),
        weight_deltas=model.calc_weight_deltas(),
        step=step,
        total_steps=total_steps,
        use_delta_component=True,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_mse,
        is_eval=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PPGD
# ──────────────────────────────────────────────────────────────────────────────


def _ppgd_cfg(coeff: float) -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        coeff=coeff,
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            eps=1e-8,
            lr_schedule=ScheduleConfig(start_val=0.01, fn_type="constant"),
        ),
        scope=BSCScope(),
        use_sigmoid_parameterization=False,
        n_warmup_steps=_PPGD_WARMUP,
        n_samples=1,
    )


def _make_ppgd_state(cfg: PersistentPGDReconLossConfig) -> PersistentPGDState:
    torch.manual_seed(_PPGD_INIT_SEED)
    return PersistentPGDState(
        module_to_c={s: _C for s in _SITES},
        batch_dims=(_BATCH, _SEQ),
        device=torch.device("cpu"),
        use_delta_component=True,
        optimizer_cfg=cfg.optimizer,
        scope=cfg.scope,
        use_sigmoid_parameterization=cfg.use_sigmoid_parameterization,
        n_warmup_steps=cfg.n_warmup_steps,
        n_samples=cfg.n_samples,
        router=AllLayersRouter(),
        reconstruction_loss=recon_loss_mse,
        replica_sync_group=None,
    )


@pytest.mark.parametrize("coeff", [2.0, 0.7])
def test_ppgd_flat_matches_two_pool(coeff: float) -> None:
    """Flat PersistentPGDReconLoss (metric hooks) == 2-pool adversary (pool-step path).

    Both drive the SAME PersistentPGDState. PPGD recon is RNG-free given the sources,
    so copying the (identically-seeded) initial sources makes both deterministic. At
    n_a=n_ppgd=1 the two normalisations coincide; asserts source / V/U / CI grads and
    the stepped sources match. The flat source grad arrives via the outer backward
    (coeff-scaled, unscaled in `after_backward`); the 2-pool reference computes it with
    raw `autograd.grad` — so this is the real grad check for the rerouting, at a
    power-of-two coeff (backward-then-divide exactly recovers the unscaled gradient)
    and a non-power-of-two coeff (fp rounding).
    """
    cfg = _ppgd_cfg(coeff)
    torch.manual_seed(7)
    model = _make_model()
    lm = cast(LMComponentModel, cast(object, model))
    batch = torch.randn(_BATCH, _SEQ, _D)
    with torch.no_grad():
        target = model(batch).detach()
    ci_vals = {s: torch.rand(_BATCH, _SEQ, model.module_to_c[s]) for s in _SITES}
    strategy = ReconLossStrategy.unfused(recon_loss_mse)

    # ====================== flat path (core metric) ======================
    metric = PersistentPGDReconLoss(cfg)
    metric.bind(model=model, device="cpu")
    state_flat = _make_ppgd_state(cfg)
    metric.state = state_flat
    _zero_grads(model)
    leaves_flat = {s: ci_vals[s].clone().requires_grad_(True) for s in _SITES}
    ctx = _ppgd_ctx(model, leaves_flat, batch, target)
    with strategy.context():
        live_loss = metric.update(ctx)
        assert live_loss is not None
        (coeff * live_loss).backward()  # V/U + CI + source grads via the one backward
    metric.after_backward()  # source grad / coeff + PGD source step
    vu_flat = _vu_grads(model)
    leaf_flat = {s: _leaf_grad(leaves_flat[s]) for s in _SITES}
    sources_flat = {s: state_flat.sources[s].detach().clone() for s in _SITES}

    # ====================== 2-pool path (pool-step helpers) ======================
    _zero_grads(model)
    state_2p = _make_ppgd_state(cfg)
    state_2p.update_lr(step=0, total_steps=1)
    weight_deltas = model.calc_weight_deltas()
    runtime = _build_runtime(model, cfg, _imp_cfg())
    with strategy.context():
        ci_scratch = _releaf_ci_fp32_for_grads({s: ci_vals[s].clone() for s in _SITES})
        recon = _warmup_and_recon(state_2p, lm, batch, target, ci_scratch, weight_deltas, runtime)
        raw = _autograd_grads_wrt_vu_ci_and_sources(
            recon.sum_loss, lm, ci_scratch, list(_SITES), state_2p.sources
        )
    # n_a = n_ppgd = 1: V/U + CI scale = coeff/(n_examples·1); sources = 1/n_examples.
    vu_and_ci_scale = coeff / recon.n_examples
    source_scale = 1.0 / recon.n_examples
    for d in (raw.v, raw.u, raw.ci):
        for g in d.values():
            g.mul_(vu_and_ci_scale)
    for g in raw.sources.values():
        g.mul_(source_scale)
    # Seed the model's V/U .grad like the chunk leader's _combine (here the only
    # contributor is PPGD), then read it.
    for s in _SITES:
        model.components[s].V.grad = raw.v[s]
        model.components[s].U.grad = raw.u[s]
    state_2p.step(state_2p.reduce_source_grads(raw.sources))
    vu_2p = {f"{s}.V": raw.v[s] for s in _SITES} | {f"{s}.U": raw.u[s] for s in _SITES}
    leaf_2p = {s: raw.ci[s] for s in _SITES}
    sources_2p = {s: state_2p.sources[s].detach().clone() for s in _SITES}

    # Guard against a vacuous pass: warmup + the final source step must actually move
    # the sources, and the V/U grad must be non-trivial.
    initial = _make_ppgd_state(cfg)
    for s in _SITES:
        assert not torch.allclose(sources_flat[s], initial.sources[s]), (
            f"ppgd sources unchanged for {s!r} — adversary step is a no-op, test is vacuous"
        )
    assert any(g.abs().sum() > 0 for g in vu_flat.values()), "ppgd V/U grad all-zero"

    for k in vu_flat:
        torch.testing.assert_close(vu_flat[k], vu_2p[k], msg=f"ppgd V/U grad: {k}")
    for s in _SITES:
        torch.testing.assert_close(leaf_flat[s], leaf_2p[s], msg=f"ppgd CI grad: {s}")
        torch.testing.assert_close(sources_flat[s], sources_2p[s], msg=f"ppgd stepped source: {s}")


def _ppgd_ctx(
    model: ComponentModel,
    leaves: dict[str, torch.Tensor],
    batch: torch.Tensor,
    target: torch.Tensor,
) -> MetricContext:
    from param_decomp.component_model import CIOutputs

    return MetricContext(
        model=cast(LMComponentModel, cast(object, model)),
        batch=batch,
        target_out=target,
        pre_weight_acts={},
        ci=CIOutputs(lower_leaky=leaves, upper_leaky=leaves, pre_sigmoid=leaves),
        weight_deltas=model.calc_weight_deltas(),
        step=0,
        total_steps=1,
        use_delta_component=True,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_mse,
        is_eval=False,
    )


def _imp_cfg() -> ImportanceMinimalityLossConfig:
    return ImportanceMinimalityLossConfig(
        coeff=0.7,
        pnorm=_IMP_PNORM,
        beta=_IMP_BETA,
        eps=_IMP_EPS,
        p_anneal_start_frac=1.0,
        p_anneal_final_p=None,
        p_anneal_end_frac=1.0,
    )
