"""Comprehensive distributed grad check for the 3-pool SUM-grad convention.

Validates the central claim of ``three_pool/SUM_GRAD_CONVENTION.md``: at a
NON-square topology with ALL loss terms enabled (faith + stoch + imp-min + ppgd
recon), the fully-reduced gradient on the REPLICATED params — the CI-fn weights
and the V/U weights — equals the single-process full-batch gradient for the
identical total loss.

Method. Build a tiny ``GPT2Simple``, a non-square ``World``
(``n_ci=4 != n_per_block=2``, 2 blocks, ``n_ppgd=2`` — exercises the CI-fine
routing regime, in-block DP replication, and cross-block summation), run ONE
3-pool step per pool up to (not including) ``optimizer.step`` (the optimizer is
neutralized so the assembled ``.grad`` survives), and compare:

  * the CI pool's fully-reduced CI-fn grad, and
  * each LW block's fully-reduced V/U grad

to a single-process reference that replays the identical loss on the full batch
with the SAME pinned RNG (stochastic mask noise ``u``, delta_mask, PPGD source
init + trajectory). The PPGD trajectory is matched by gathering the distributed
per-rank initial sources into the reference before warmup (PerBatchPerPosition
sources are per-position independent, so the gathered full-batch trajectory is
bit-identical to the per-rank-sliced one).

Run directly:
   torchrun --standalone --nproc_per_node=8 --master_port=29531 \
     param_decomp_lab/tests/test_three_pool_grad_check_distributed.py
or via pytest (spawns torchrun in a subprocess).
"""

# pyright: reportArgumentType=false, reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false, reportUnusedParameter=false
# pyright: reportUnusedVariable=false

import os
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch import Tensor

from param_decomp.ci_fns import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    GlobalSharedTransformerCiFn,
)
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import AllLayersRouter, make_mask_infos
from param_decomp.metrics.importance_minimality import (
    finalize_imp_min,
    per_component_lp_sums,
)
from param_decomp.metrics.persistent_pgd_recon import PersistentPGDReconLossConfig
from param_decomp.metrics.persistent_pgd_state import (
    AdamPGDConfig,
    PerBatchPerPositionScope,
    PersistentPGDState,
)
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse
from param_decomp_lab.distributed import cleanup_distributed, init_distributed
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.three_pool.context import CIContext, LWContext, PPGDContext
from param_decomp_lab.three_pool.layout import LayerwiseBlockGroup, build_world
from param_decomp_lab.three_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp_lab.three_pool.routing_plan import PerSitePlan
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime
from param_decomp_lab.three_pool.step_ci import step_ci
from param_decomp_lab.three_pool.step_layerwise import step_layerwise
from param_decomp_lab.three_pool.step_ppgd import step_ppgd

# ── Topology: non-square, multi-block, in-block DP, n_ppgd > 1 ────────────────
_CI_RANKS = [0, 1, 2, 3]  # n_ci = 4
_BLOCK0_RANKS = [4, 5]  # n_per_block = 2
_BLOCK1_RANKS = [6, 7]
_PPGD_RANKS = [8, 9]  # n_ppgd = 2
_WORLD_SIZE = 10
_BATCH_GLOBAL = 4
_SEQ_LEN = 3
_VOCAB = 16
_D_MODEL = 8
_N_HEAD = 2
_C = 2  # per-site component count

_SITES_BLOCK0 = ["h.0.attn.q_proj"]
_SITES_BLOCK1 = ["h.1.attn.q_proj"]
_ALL_SITES = _SITES_BLOCK0 + _SITES_BLOCK1

_COEFF_FAITH = 3.0
_COEFF_IMP = 0.7
_COEFF_STOCH = 5.0
_COEFF_PPGD = 2.0
_IMP_PNORM = 2.0
_IMP_BETA = 0.5
_IMP_EPS = 1e-12
_PPGD_WARMUP = 2

_MODEL_SEED = 1234
_BATCH_SEED = 99
_STOCH_RNG_SEED = 7  # pins u + delta_mask for stoch (per-site, per-call)
_PPGD_INIT_SEED = 55


def _build_target() -> GPT2Simple:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        block_size=_SEQ_LEN,
        vocab_size=_VOCAB,
        n_layer=2,
        n_head=_N_HEAD,
        n_embd=_D_MODEL,
        flash_attention=False,
    )
    torch.manual_seed(_MODEL_SEED)
    model = GPT2Simple(cfg)
    model.requires_grad_(False)
    return model


def _decomp_targets() -> list[DecompositionTarget]:
    return [DecompositionTarget(module_path=s, C=_C) for s in _ALL_SITES]


def _ci_config() -> GlobalCiConfig:
    # global_shared_transformer reads pre_weight_acts (independent of V/U), so CI
    # values don't depend on the component weights — matching production and
    # keeping the CI-fn grad cleanly separable from the V/U grad.
    return GlobalCiConfig(
        fn_type="global_shared_transformer",
        simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
            d_model=_D_MODEL,
            n_blocks=1,
            mlp_hidden_dim=[8],
            attn_config=AttnConfig(n_heads=_N_HEAD, max_len=_SEQ_LEN, rope_base=10000.0),
        ),
    )


def _ppgd_cfg() -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        coeff=_COEFF_PPGD,
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            eps=1e-8,
            lr_schedule=ScheduleConfig(start_val=0.01, fn_type="constant"),
        ),
        scope=PerBatchPerPositionScope(),
        use_sigmoid_parameterization=False,
        n_warmup_steps=_PPGD_WARMUP,
        n_samples=1,
    )


def _build_runtime(numel_global: int) -> _ThreePoolRuntime:
    block_groups = (
        LayerwiseBlockGroup(ranks=tuple(_BLOCK0_RANKS), owned_sites=tuple(_SITES_BLOCK0)),
        LayerwiseBlockGroup(ranks=tuple(_BLOCK1_RANKS), owned_sites=tuple(_SITES_BLOCK1)),
    )
    routing_plan = PerSitePlan()
    n_est = sum(routing_plan.n_forwards(bg.owned_sites) for bg in block_groups)
    return _ThreePoolRuntime(
        ci_ranks=tuple(_CI_RANKS),
        layerwise_block_groups=block_groups,
        ppgd_ranks=tuple(_PPGD_RANKS),
        batch_global=_BATCH_GLOBAL,
        c_per_site={s: _C for s in _ALL_SITES},
        ci_config=_ci_config(),
        sigmoid_type="leaky_hard",
        run_batch=lambda model, batch: model(batch),  # unused on CPU recon path
        reconstruction_loss=recon_loss_mse,
        ppgd_cfg=_ppgd_cfg(),
        routing_plan=routing_plan,
        n_est=n_est,
        coeff_faith=_COEFF_FAITH,
        coeff_imp=_COEFF_IMP,
        coeff_stoch=_COEFF_STOCH,
        coeff_ppgd=_COEFF_PPGD,
        log_name_faith="FaithfulnessLoss",
        log_name_imp="ImportanceMinimalityLoss",
        log_name_stoch="StochasticReconLayerwiseLoss",
        log_name_ppgd="PersistentPGDReconLoss",
        imp_min_pnorm=_IMP_PNORM,
        imp_min_beta=_IMP_BETA,
        imp_min_eps=_IMP_EPS,
        imp_min_p_anneal_start_frac=0.0,
        imp_min_p_anneal_final_p=None,
        imp_min_p_anneal_end_frac=1.0,
        lr_components=0.0,
        lr_ci_fn=0.0,
        grad_clip_norm_components=None,
        grad_clip_norm_ci_fn=None,
        numel_global=numel_global,
        bf16_autocast=False,
        use_fused_kl=False,
    )


def _build_component_model(rank: int) -> LMComponentModel:
    target = _build_target()
    # Seed identically on every rank so V/U + CI fn init match (DDP partners).
    torch.manual_seed(_MODEL_SEED + 1)
    cm = LMComponentModel.build(
        target_model=target,
        decomposition_targets=_decomp_targets(),
        ci_config=_ci_config(),
        sigmoid_type="leaky_hard",
    )
    # Exercise CI-fn activation checkpointing (production default) so the grad check
    # also validates that the checkpointed-block recompute reassembles identical grads.
    if cm.ci_fn is not None:
        for m in cm.ci_fn.modules():
            if isinstance(m, GlobalSharedTransformerCiFn):
                m.enable_activation_checkpointing()
    return cm


def _global_batch() -> Tensor:
    torch.manual_seed(_BATCH_SEED)
    return torch.randint(0, _VOCAB, (_BATCH_GLOBAL, _SEQ_LEN))


def _pin_stoch_rng() -> None:
    """Pin the RNG the stoch path consumes (u + delta_mask in _layerwise_one_site)."""
    torch.manual_seed(_STOCH_RNG_SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Distributed side
# ──────────────────────────────────────────────────────────────────────────────


def _neutralize_optimizer(opt: torch.optim.Optimizer) -> None:
    """Replace ``step`` with a no-op so the assembled .grad survives for capture."""
    opt.step = lambda *a, **k: None  # type: ignore[method-assign]


def _make_ppgd_state(world: Any, runtime: _ThreePoolRuntime, strategy: Any) -> PersistentPGDState:
    return PersistentPGDState(
        module_to_c=runtime.c_per_site,
        batch_dims=(world.batch_local_ppgd, _SEQ_LEN),
        device=torch.device("cpu"),
        use_delta_component=True,
        optimizer_cfg=runtime.ppgd_cfg.optimizer,
        scope=runtime.ppgd_cfg.scope,
        use_sigmoid_parameterization=runtime.ppgd_cfg.use_sigmoid_parameterization,
        n_warmup_steps=runtime.ppgd_cfg.n_warmup_steps,
        n_samples=runtime.ppgd_cfg.n_samples,
        router=AllLayersRouter(),
        reconstruction_loss=strategy.recon_loss,
    )


def _gather_initial_ppgd_sources(
    ppgd_state: PersistentPGDState | None, ctx: Any, world: Any
) -> dict[str, Tensor] | None:
    """All-gather each PPGD rank's pre-warmup sources so rank 0 can replay the
    exact PPGD trajectory in the reference.

    Uses ``all_gather_object`` on the WORLD default group (collective on every
    rank). Non-PPGD ranks contribute ``None``. Returns the reassembled
    full-batch sources on rank 0, ``None`` elsewhere.
    """
    payload: dict[str, Any] | None = None
    if isinstance(ctx, PPGDContext):
        assert ppgd_state is not None
        sl = ctx.role.batch_slice(world.batch_local_ppgd)
        start = sl.start if sl.start is not None else 0
        payload = {
            "start": start,
            "sources": {s: ppgd_state.sources[s].detach().clone() for s in ppgd_state.sources},
        }
    gathered: list[Any] = [None] * _WORLD_SIZE
    dist.all_gather_object(gathered, payload)
    if dist.get_rank() != 0:
        return None
    source_c = _C + 1  # use_delta_component
    full = {s: torch.zeros(_BATCH_GLOBAL, _SEQ_LEN, source_c) for s in _ALL_SITES}
    for item in gathered:
        if item is None:
            continue
        start = item["start"]
        for s, t in item["sources"].items():
            full[s][start : start + t.shape[0]] = t
    return full


def _run_distributed_step() -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
    """Run one 3-pool step on this rank.

    Returns ``(captured_grads, ref_ppgd_sources_or_None)`` where captured_grads
    is {param_key: fully_reduced_grad} for the params this rank leads, and the
    second element is the reassembled full-batch PPGD initial sources (rank 0
    only)."""
    from param_decomp_lab.three_pool.context import build_pool_context

    world = build_world(
        ci_ranks=list(_CI_RANKS),
        layerwise_block_groups=[
            LayerwiseBlockGroup(ranks=tuple(_BLOCK0_RANKS), owned_sites=tuple(_SITES_BLOCK0)),
            LayerwiseBlockGroup(ranks=tuple(_BLOCK1_RANKS), owned_sites=tuple(_SITES_BLOCK1)),
        ],
        ppgd_ranks=list(_PPGD_RANKS),
        batch_global=_BATCH_GLOBAL,
        pg_timeout=timedelta(seconds=120),
        device=None,
    )
    rank = dist.get_rank()
    ctx = build_pool_context(world, rank)
    cm = _build_component_model(rank)
    numel_global = sum(cm.target_weight(s).numel() for s in _ALL_SITES)

    # NOTE: the trainer drops pool-irrelevant params for memory; we keep all of
    # them so the layerwise ('mlp') CI fn — which reads component activations —
    # works on the CI pool too. Dropping is a memory opt, irrelevant to grads.
    runtime = _build_runtime(numel_global)
    strategy = LayerwiseLossStrategy.unfused(recon_loss_mse)
    batch = _global_batch()

    # PPGD state must exist BEFORE the source-gather collective (all ranks).
    ppgd_state: PersistentPGDState | None = None
    if isinstance(ctx, PPGDContext):
        torch.manual_seed(_PPGD_INIT_SEED + rank)
        ppgd_state = _make_ppgd_state(world, runtime, strategy)
    ref_sources = _gather_initial_ppgd_sources(ppgd_state, ctx, world)

    captured: dict[str, Tensor] = {}
    match ctx:
        case CIContext():
            ci_fn_params = list(cm.ci_fn.parameters())  # type: ignore[union-attr]
            opt = torch.optim.AdamW(ci_fn_params, lr=0.0)
            _neutralize_optimizer(opt)
            _pin_stoch_rng()
            step_ci(
                ctx,
                cm,
                opt,
                ci_fn_params,
                batch_T=batch,
                batch_T_plus_1=None,
                h_cache_T=None,
                cfg=runtime,
                current_frac_of_training=0.0,
                should_log=False,
            )
            if ctx.role.is_pool_leader:
                for n, p in cm.ci_fn.named_parameters():  # type: ignore[union-attr]
                    assert p.grad is not None, f"ci_fn.{n} grad is None"
                    captured[f"ci_fn.{n}"] = p.grad.detach().clone()
        case LWContext():
            comp_params = [p for s in ctx.role.owned_sites for p in cm.components[s].parameters()]
            opt = torch.optim.AdamW(comp_params, lr=0.0)
            _neutralize_optimizer(opt)
            _pin_stoch_rng()
            step_layerwise(ctx, cm, opt, comp_params, batch, runtime, strategy, should_log=False)
            if ctx.role.is_block_leader:
                for s in ctx.role.owned_sites:
                    for n, p in cm.components[s].named_parameters():
                        assert p.grad is not None, f"components.{s}.{n} grad is None"
                        captured[f"components.{s}.{n}"] = p.grad.detach().clone()
        case PPGDContext():
            assert ppgd_state is not None
            step_ppgd(
                ctx, cm, ppgd_state, batch, runtime, strategy, step=0, n_steps=1, should_log=False
            )
    return captured, ref_sources


# ──────────────────────────────────────────────────────────────────────────────
# Single-process reference (full batch, identical loss, pinned RNG)
# ──────────────────────────────────────────────────────────────────────────────


def _reference_grads(initial_ppgd_sources: dict[str, Tensor]) -> dict[str, Tensor]:
    """Full-batch single-process gradient for the identical total loss.

    Mirrors the per-term seeding the step functions apply, but on the global
    batch in one process: faith, stoch, imp-min, ppgd recon. Returns
    {param_key: grad} for ci_fn.* and components.<site>.*.
    """
    cm = _build_component_model(0)  # full model (CI fn + components)
    numel_global = sum(cm.target_weight(s).numel() for s in _ALL_SITES)
    runtime = _build_runtime(numel_global)
    strategy = LayerwiseLossStrategy.unfused(recon_loss_mse)
    batch = _global_batch()

    ci_fn_params = list(cm.ci_fn.parameters())  # type: ignore[union-attr]
    comp_params = [p for s in _ALL_SITES for p in cm.components[s].parameters()]
    for p in [*ci_fn_params, *comp_params]:
        p.grad = None

    n_sites_total = len(_ALL_SITES)

    # --- CI fn forward (full batch) ---
    _, cache = cm.forward_with_pre_weight_acts(batch)
    cache = {k: v.to(torch.float32) for k, v in cache.items()}
    ci = cm.calc_causal_importances(
        pre_weight_acts=cache, sampling="continuous", detach_inputs=False
    )

    # --- faith (full batch == single-pool) ---
    weight_deltas = cm.calc_weight_deltas()
    sum_sq = torch.zeros(())
    for d in weight_deltas.values():
        sum_sq = sum_sq + (d**2).sum()
    faith_loss = sum_sq / numel_global
    (_COEFF_FAITH * faith_loss).backward(retain_graph=True)

    # --- imp-min (full batch == single-pool exact) ---
    per_component_sums, n_examples = per_component_lp_sums(
        ci_upper_leaky=ci.upper_leaky, pnorm=_IMP_PNORM, eps=_IMP_EPS
    )
    imp_loss = finalize_imp_min(
        per_component_sums=per_component_sums, n_examples=n_examples, beta=_IMP_BETA
    )
    (_COEFF_IMP * imp_loss).backward(retain_graph=True)

    # --- stoch (per-site, pinned RNG matching _layerwise_one_site order) ---
    target_full = cm(batch).detach()
    _pin_stoch_rng()
    for s in _ALL_SITES:
        ci_s = ci.lower_leaky[s]
        u = torch.rand_like(ci_s)
        mask = ci_s + (1 - ci_s) * u
        delta = cm.target_weight(s) - cm.components[s].weight
        delta_mask = torch.rand(ci_s.shape[:-1], dtype=ci_s.dtype)
        mask_infos = make_mask_infos(
            {s: mask},
            weight_deltas_and_masks={s: (delta, delta_mask)},
            routing_masks="all",
        )
        pred = cm(batch, mask_infos=mask_infos)
        # The single-process recon's n_positions IS the global count (full batch),
        # matching the distributed code's ``n_positions_local * n_per_block``.
        loss_s, n_positions_global = strategy.recon_loss(pred=pred, target=target_full)
        denom = n_positions_global * n_sites_total
        (_COEFF_STOCH * loss_s / denom).backward(retain_graph=True)

    # --- ppgd recon (full batch, replayed sources) ---
    ppgd_state = PersistentPGDState(
        module_to_c=runtime.c_per_site,
        batch_dims=(_BATCH_GLOBAL, _SEQ_LEN),
        device=torch.device("cpu"),
        use_delta_component=True,
        optimizer_cfg=runtime.ppgd_cfg.optimizer,
        scope=runtime.ppgd_cfg.scope,
        use_sigmoid_parameterization=runtime.ppgd_cfg.use_sigmoid_parameterization,
        n_warmup_steps=runtime.ppgd_cfg.n_warmup_steps,
        n_samples=runtime.ppgd_cfg.n_samples,
        router=AllLayersRouter(),
        reconstruction_loss=strategy.recon_loss,
    )
    with torch.no_grad():
        for s in ppgd_state.sources:
            ppgd_state.sources[s].copy_(initial_ppgd_sources[s])
    ppgd_state.update_lr(step=0, total_steps=1)
    ci_scratch = {
        s: ci.lower_leaky[s].detach().to(torch.float32).clone().requires_grad_(True)
        for s in _ALL_SITES
    }
    ppgd_state.warmup(
        model=cm,
        batch=batch,
        target_out=target_full,
        ci=ci_scratch,
        weight_deltas=cm.calc_weight_deltas(),
    )
    ppgd_sum_loss, n_ppgd_examples = ppgd_state.compute_recon_sum_and_n(
        model=cm,
        batch=batch,
        target_out=target_full,
        ci=ci_scratch,
        weight_deltas=cm.calc_weight_deltas(),
    )
    # canonical PPGD loss: coeff * recon_sum / n_examples_global, on V/U AND ci_scratch.
    ppgd_scaled = _COEFF_PPGD * ppgd_sum_loss / n_ppgd_examples
    # V/U grads accumulate directly; ci_scratch grads must propagate through the
    # CI fn graph (the distributed run ships g_CI back and seeds lower_leaky).
    ci_scratch_grads = torch.autograd.grad(
        ppgd_scaled, list(ci_scratch.values()), retain_graph=True
    )
    ppgd_scaled.backward()  # populates V/U .grad for ppgd term
    torch.autograd.backward(
        tensors=[ci.lower_leaky[s] for s in _ALL_SITES],
        grad_tensors=list(ci_scratch_grads),
    )

    out: dict[str, Tensor] = {}
    for n, p in cm.ci_fn.named_parameters():  # type: ignore[union-attr]
        assert p.grad is not None
        out[f"ci_fn.{n}"] = p.grad.detach().clone()
    for s in _ALL_SITES:
        for n, p in cm.components[s].named_parameters():
            assert p.grad is not None
            out[f"components.{s}.{n}"] = p.grad.detach().clone()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────


def _run_test() -> None:
    init_distributed()
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == _WORLD_SIZE

        captured, ref_sources = _run_distributed_step()

        gathered: list[Any] = [None] * _WORLD_SIZE
        dist.all_gather_object(gathered, captured)

        if rank == 0:
            dist_grads: dict[str, Tensor] = {}
            for d in gathered:
                assert d is not None
                dist_grads.update(d)
            assert ref_sources is not None
            _report_and_assert(dist_grads, ref_sources)
    finally:
        cleanup_distributed()


def _report_and_assert(dist_grads: dict[str, Tensor], ref_sources: dict[str, Tensor]) -> None:
    ref_grads = _reference_grads(ref_sources)
    keys = sorted(dist_grads.keys())
    assert set(keys) == set(ref_grads.keys()), (
        f"param-key mismatch:\n dist={sorted(dist_grads)}\n ref ={sorted(ref_grads)}"
    )
    print("\n=== 3-pool SUM-grad convention: distributed vs single-process reference ===")
    print("topology: n_ci=4, n_per_block=2 (2 blocks), n_ppgd=2 (NON-square)")
    worst = 0.0
    for k in keys:
        dg, rg = dist_grads[k], ref_grads[k]
        assert dg.shape == rg.shape, f"{k}: shape {tuple(dg.shape)} vs {tuple(rg.shape)}"
        denom = rg.abs().max().clamp_min(1e-12)
        rel = (dg - rg).abs().max().item() / denom.item()
        # mean ratio over nonzero ref entries (a scalar "are they the same scale?")
        mask = rg.abs() > 1e-8
        ratio = (dg[mask] / rg[mask]).mean().item() if mask.any() else float("nan")
        worst = max(worst, rel)
        print(f"  {k:34s} max|Δ|/max|ref|={rel:.2e}  mean(dist/ref)={ratio:.6f}")
    print(f"worst relative error across all params: {worst:.2e}")
    for k in keys:
        torch.testing.assert_close(
            dist_grads[k],
            ref_grads[k],
            rtol=2e-4,
            atol=2e-5,
            msg=lambda m, k=k: f"grad mismatch on {k}:\n{m}",
        )
    print("PASS: all reduced grads match the single-process reference at non-square topology.")


if __name__ == "__main__":
    _run_test()


@pytest.mark.slow
class TestThreePoolGradCheckDistributed:
    def test_grad_check(self) -> None:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={_WORLD_SIZE}",
            "--master_port",
            "29531",
            str(Path(__file__).resolve()),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"grad check failed (code {result.returncode})")
        print(result.stdout)
