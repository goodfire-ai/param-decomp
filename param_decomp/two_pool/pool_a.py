"""Pool-A training step: target+CI forward, layerwise streaming loss, home losses, opt step.

Pool A trains V/U + CI fn. Each pool-A rank:

  1. Receives previous step's V/U updates (async). Runs target + CI forward.
  2. Sends per-site CI to pool B (async).
  3. Computes home losses (faithfulness, importance-minimality).
  4. Streaming layerwise loss — one site at a time, immediate backward, peak
     memory bounded to ~1×iter.
  5. Receives pool B's V/U + ci grad contributions.
  6. Combined backward through home losses + CI-fn graph (seeded by pool B's
     ci grads).
  7. In-block all-reduce, AdamW step.
  8. Async-sends updated V/U back to pool B.
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.component_model import ComponentModel
from param_decomp.grad_clip import cross_pool_clip_grad_norm
from param_decomp.masks import make_mask_infos
from param_decomp.metrics.importance_minimality import (
    _finalize as _finalize_imp_min,
)
from param_decomp.metrics.importance_minimality import (
    _get_linear_annealed_p,
    _per_component_sums,
)
from param_decomp.two_pool.layout import BlockDDPLayout
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp.two_pool.profiler import PhaseProfiler
from param_decomp.two_pool.runtime import _TwoPoolRuntime, autocast_bf16


def _faithfulness_loss(
    component_model: ComponentModel, device: torch.device, numel_global: int
) -> tuple[Tensor, Tensor, int]:
    """Standard faithfulness loss: ``‖W_target − VU.T‖²_F / numel_global`` over this
    rank's owned sites.

    Single-pool computes ``sum_sq_global / numel_global``; the per-element
    gradient on each ``V_i`` / ``U_i`` is then ``∝ 1 / numel_global``. To match
    that gradient scale in multi-pool, where each rank only sees its owned
    subset of sites, we still divide by ``numel_global`` (not the rank-local
    ``numel_owned``). Otherwise per-element gradients scale up by
    ``numel_global / numel_owned`` and the trajectory diverges from single-pool
    — most visibly during the unclipped faithfulness warmup, where 2-pool's
    V/U over-converges relative to 1-pool's after 400 steps.

    Returns ``(scalar_loss, sum_sq, numel_owned)`` — the raw ``numel_owned`` is
    still what the logger needs as the denominator for the ``SUM(num) /
    SUM(den)`` global-ratio reconstruction across blocks.
    """
    weight_deltas = component_model.calc_weight_deltas()
    sum_sq = torch.zeros((), device=device)
    numel_owned = 0
    for d in weight_deltas.values():
        sum_sq = sum_sq + (d**2).sum()
        numel_owned += d.numel()
    return sum_sq / numel_global, sum_sq, numel_owned


def _importance_minimality_loss(
    ci_upper: dict[str, Tensor],
    current_frac_of_training: float,
    cfg: _TwoPoolRuntime,
) -> Tensor:
    """Importance-minimality loss matching single-pool semantics.

    Mirrors ``param_decomp.metrics.builtin.importance_minimality_loss`` —
    annealed L_p penalty with a logarithmic beta term. ``world_size=1`` because
    the pool-A target+CI forward (phase a/1) runs on the FULL global batch
    (only the layerwise stoch loss slices the batch by within_block_idx, NOT
    the CI fn forward). So each rank's local ``sum`` over its owned sites
    already equals the global sum for those sites; the cross-block aggregation
    is SUM-across-disjoint-site-sets and lives entirely in the logger.
    """
    annealed_p = _get_linear_annealed_p(
        current_frac_of_training=current_frac_of_training,
        initial_p=cfg.imp_min_pnorm,
        p_anneal_start_frac=cfg.imp_min_p_anneal_start_frac,
        p_anneal_final_p=cfg.imp_min_p_anneal_final_p,
        p_anneal_end_frac=cfg.imp_min_p_anneal_end_frac,
    )
    per_component_sums, n_examples = _per_component_sums(
        ci_upper_leaky=ci_upper, pnorm=annealed_p, eps=cfg.imp_min_eps
    )
    return _finalize_imp_min(
        per_component_sums=per_component_sums,
        n_examples=n_examples,
        beta=cfg.imp_min_beta,
        world_size=1,
    )


def _layerwise_loss_streaming(
    component_model: ComponentModel,
    batch_local: Any,
    target_local: Tensor,
    ci_lower_leaves: dict[str, Tensor],
    owned_sites: tuple[str, ...],
    recon_loss: ReconstructionLoss,
    coeff_stoch: float,
    n_sites_total: int,
) -> tuple[float, float, int]:
    """Per-site layerwise loss, backpropagated per-iter so peak memory stays bounded.

    Each iter masks one site, runs the component-model forward to produce
    ``pred``, computes ``loss = recon_loss(pred, target_local)``, and backwards
    immediately. The iter-local autograd graph is then freed, bounding peak
    memory to ~1×iter instead of the ``N_sites×iter`` retain-graph would cost.

    ``recon_loss`` encapsulates the choice of fused-vs-unfused — when fused,
    pred/target are pre-LM-head hidden states and the kernel does LM head + KL
    in chunks. When unfused, pred/target are logits. See
    :class:`LayerwiseLossStrategy`.

    Per-site contribution is ``coeff_stoch * sum_kl_s / (n_positions *
    n_sites_total)`` — matches single-pool layerwise's
    ``coeff_stoch * sum_kl / (n_sites_total * n_positions)`` so the same YAML
    coefficient transfers between the two trainers.

    Returns ``(scalar_value, raw_num, raw_den)`` where:
      * ``scalar_value = total_value / n_owned`` is the per-rank logging scalar.
      * ``raw_num = total_value`` (sum over owned sites of ``loss / n_positions``)
        and ``raw_den = n_owned`` are the additive ingredients the logger
        combines as ``SUM(num) / SUM(den)`` across blocks to recover the global
        mean over (sites, positions). Intra-block AVG of ``raw_num`` is the
        cross-slice mean per site per position; intra-block AVG of ``raw_den``
        is trivial (identical on all partners).
    """
    n_owned = len(owned_sites)
    total_value = 0.0
    for s in owned_sites:
        ci_s = ci_lower_leaves[s]
        u = torch.rand_like(ci_s)
        mask = ci_s + (1 - ci_s) * u
        delta = component_model.target_weight(s) - component_model.components[s].weight
        delta_mask = torch.rand(ci_s.shape[:-1], device=ci_s.device, dtype=ci_s.dtype)
        mask_infos = make_mask_infos(
            {s: mask},
            weight_deltas_and_masks={s: (delta, delta_mask)},
            routing_masks="all",
        )
        pred = component_model(batch_local, mask_infos=mask_infos)
        loss, n_positions = recon_loss(pred=pred, target=target_local)
        scaled = coeff_stoch * loss / (n_positions * n_sites_total)
        scaled.backward()  # retain_graph=False — iter-local graph freed
        total_value += (loss / n_positions).item()
    return total_value / n_owned, total_value, n_owned


def step_pool_a(
    layout: BlockDDPLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    component_params: list[nn.Parameter],
    ci_fn_params: list[nn.Parameter],
    batch: Any,
    cfg: _TwoPoolRuntime,
    strategy: LayerwiseLossStrategy,
    current_frac_of_training: float,
    profiler: PhaseProfiler | None = None,
) -> dict[str, float]:
    """One training step on a pool-A rank."""
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)

    # 0. Wait for any pending async weight sends from the previous step before we
    #    risk modifying V/U via the optimizer step below.
    with p.phase("a/0_wait_prev_weight_send"):
        pending = getattr(component_model, "_pending_weight_sends", None)
        if pending is not None:
            for w in pending[0]:
                w.wait()
            component_model._pending_weight_sends = None  # type: ignore[attr-defined]

    # The strategy's context manager decides whether forwards return logits or
    # pre-LM-head hidden state for this step; recon_loss matches accordingly.
    with strategy.context(component_model.target_model):
        # 1. target + CI forward; CI fn graph retained.
        # bf16 autocast: target forward's SDPA only has a fast kernel for bf16/fp16
        # on H200 — fp32 falls back to the O(N²) math backend. Autocast on the
        # forward path saves ~30ms on target_fwd at our shape (see microbench).
        # Faithfulness loss is computed OUTSIDE autocast since it's fp32-sensitive.
        with p.phase("a/1_target_and_ci_fwd"), autocast_bf16(cfg.bf16_autocast):
            out = component_model(batch, cache_type="input")
            target_out = out.output  # logits or hidden state per strategy
            ci = component_model.calc_causal_importances(
                pre_weight_acts=out.cache,
                sampling="continuous",
                detach_inputs=False,
            )
            # Sanity print on step 0, leader only: confirms bf16 actually engaged.
            if not getattr(component_model, "_dtype_logged", False) and layout.my_rank == 0:
                sample_act = next(iter(out.cache.values()))
                print(
                    f"[two_pool sanity rank0] bf16_autocast={cfg.bf16_autocast}  "
                    f"use_fused_kl={cfg.use_fused_kl}  "
                    f"target_out.dtype={target_out.dtype}  "
                    f"target_out.shape={tuple(target_out.shape)}  "
                    f"cached_act.dtype={sample_act.dtype}  "
                    f"ci.lower_leaky.dtype={next(iter(ci.lower_leaky.values())).dtype}",
                    flush=True,
                )
                component_model._dtype_logged = True  # type: ignore[attr-defined]

        # 2. Cross-pool: send CI values to pool B (async — don't block on pool B's recv).
        with p.phase("a/2_async_send_ci"):
            ci_send_works, ci_send_buffers = layout.async_send_owned_ci_to_pool_b(
                {s: ci.lower_leaky[s] for s in layout.my_owned_sites}
            )

        # 3. Home losses (forward only; backward happens after streaming layerwise)
        device = target_out.device
        with p.phase("a/3_faith"):
            loss_faith, faith_sum_sq_t, faith_numel = _faithfulness_loss(
                component_model, device, cfg.numel_global
            )
        with p.phase("a/4_imp"):
            loss_imp = _importance_minimality_loss(ci.upper_leaky, current_frac_of_training, cfg)

        # 5. Streaming layerwise: re-leaf CI so each iter is autograd-independent
        #    from the CI-fn graph, then backprop per iter to bound peak memory.
        #    Accumulates V/U .grad and ci_lower_leaves[s].grad along the way.
        optimizer.zero_grad(set_to_none=True)
        with p.phase("a/5_layerwise"), autocast_bf16(cfg.bf16_autocast):
            sl = layout.my_batch_slice_a()
            batch_local = batch[sl] if isinstance(batch, Tensor) else batch
            target_local = target_out[sl].detach()
            # Detached leaves matching the sliced CI values — backward through
            # layerwise stops at these (does NOT traverse the CI-fn graph).
            ci_lower_leaves = {
                s: ci.lower_leaky[s][sl].detach().requires_grad_(True)
                for s in layout.my_owned_sites
            }
            loss_stoch_value, stoch_total_value, stoch_n_owned = _layerwise_loss_streaming(
                component_model,
                batch_local,
                target_local,
                ci_lower_leaves,
                layout.my_owned_sites,
                strategy.recon_loss,
                cfg.coeff_stoch,
                n_sites_total=len(cfg.c_per_site),
            )

    # 4. Cross-pool: receive per-site V/U grads + per-slice ci grads from pool B
    with p.phase("a/6_recv_grads_from_b"):
        v_templates = {s: component_model.components[s].V for s in layout.my_owned_sites}
        u_templates = {s: component_model.components[s].U for s in layout.my_owned_sites}
        ci_lower_owned_full = {s: ci.lower_leaky[s] for s in layout.my_owned_sites}
        v_grads, u_grads, ci_grads = layout.recv_grads_from_pool_b(
            v_templates,
            u_templates,
            ci_lower_owned_full,
        )

    # 6. Combined backward through home losses + CI-fn graph.
    #    V/U .grad already holds layerwise's contribution → ADD pool B's. CI
    #    grads to push through the CI-fn graph = (per-slice pool-B ci_grads)
    #    plus (full-batch layerwise contribution gathered from ci_lower_leaves).
    with p.phase("a/7_seed_and_backward"):
        # Combine ci grads: pool B's per-rank contribution is full-batch, but
        # layerwise touched only the rank's slice. Place leaf grads into the
        # right slice of a full-batch tensor that matches ci.lower_leaky's shape.
        combined_ci_grads: dict[str, Tensor] = {}
        for s in layout.my_owned_sites:
            grad = ci_grads[s].clone()
            grad[sl] += ci_lower_leaves[s].grad  # type: ignore[operator]
            combined_ci_grads[s] = grad
        # Add pool B's V/U contribution to existing layerwise grad
        for s in layout.my_owned_sites:
            comp = component_model.components[s]
            assert comp.V.grad is not None and comp.U.grad is not None, (
                "layerwise should have populated V/U .grad"
            )
            comp.V.grad.add_(v_grads[s])
            comp.U.grad.add_(u_grads[s])
        # Home (faith+imp) backward + push combined ci grads through CI-fn graph
        total_home = cfg.coeff_faith * loss_faith + cfg.coeff_imp * loss_imp
        torch.autograd.backward(
            tensors=[total_home, *(ci.lower_leaky[s] for s in layout.my_owned_sites)],
            grad_tensors=[None, *(combined_ci_grads[s] for s in layout.my_owned_sites)],
        )

    # 6. In-block DDP sync.
    with p.phase("a/8_in_block_allreduce"):
        layout.all_reduce_grads_in_block(all_params)

    # 6b. Cross-pool grad clip (matches single-pool semantics — clip on the
    # global norm summed across all pool-A blocks, not per-rank). Within a
    # block DDP partners hold identical grads after step 6, so the
    # all-reduce SUM over pool A double-counts by ``n_per_block``; divide
    # back out.
    with p.phase("a/8b_grad_clip"):
        n_per_block = layout.world.n_per_block
        if cfg.grad_clip_norm_components is not None:
            cross_pool_clip_grad_norm(
                component_params,
                cfg.grad_clip_norm_components,
                group=layout.world.pool_a_group,
                n_replicas=n_per_block,
            )
        if cfg.grad_clip_norm_ci_fn is not None:
            cross_pool_clip_grad_norm(
                ci_fn_params,
                cfg.grad_clip_norm_ci_fn,
                group=layout.world.pool_a_group,
                n_replicas=n_per_block,
            )

    # 7. AdamW step.
    with p.phase("a/9_opt_step"):
        optimizer.step()

    # 8. Cross-pool: ship updated V/U back to pool B (async).
    with p.phase("a/10_async_send_weights"):
        v_owned = {s: component_model.components[s].V for s in layout.my_owned_sites}
        u_owned = {s: component_model.components[s].U for s in layout.my_owned_sites}
        weight_send_works, weight_send_buffers = layout.async_send_updated_weights_to_pool_b(
            v_owned,
            u_owned,
        )

    # Make sure the async CI sends from step 2 are flushed before we touch the
    # source CI tensors next step.
    with p.phase("a/11_wait_async_ci_send"):
        for w in ci_send_works:
            w.wait()
        del ci_send_buffers  # release references
    # The weight sends are free to complete in the background; we wait on them
    # at start of next step (phase a/0).
    component_model._pending_weight_sends = (
        weight_send_works,
        weight_send_buffers,
    )  # type: ignore[attr-defined]

    return {
        "loss/faith": loss_faith.item(),
        "loss/imp": loss_imp.item(),
        "loss/stoch": loss_stoch_value,
        # Raw (numerator, denominator) per loss for cross-block aggregation in
        # the logger. Faith and stoch are ratios — the global ratio is
        # ``SUM(num) / SUM(den)`` across blocks, NOT the AVG of per-rank
        # ratios. Imp is a SUM-across-disjoint-site-sets: raw "num" is the
        # per-rank scalar, raw "den" is fixed at 1 so SUM(num)/SUM(den) gives
        # the cross-block SUM after dividing by n_blocks — which the logger
        # un-divides by multiplying back. (Equivalent to: aggregator does
        # straight cross-block SUM for imp; den=1 is a uniform tag.)
        "_raw/faith_num": faith_sum_sq_t.item(),
        "_raw/faith_den": float(faith_numel),
        "_raw/imp_num": loss_imp.item(),
        "_raw/stoch_num": stoch_total_value,
        "_raw/stoch_den": float(stoch_n_owned),
    }


def run_faithfulness_warmup_pool_a(
    *,
    component_model: ComponentModel,
    component_params: list[nn.Parameter],
    n_steps: int,
    lr: float,
    weight_decay: float,
    numel_global: int,
) -> None:
    """Single-pool-equivalent faithfulness warmup on pool A only.

    Pool B has no V/U params to optimize, so the warmup is a no-op there.
    Done once at startup so 2-pool runs initialize the same way as single-pool.
    """
    warmup_opt = torch.optim.AdamW(component_params, lr=lr, weight_decay=weight_decay)
    for _ in range(n_steps):
        warmup_opt.zero_grad()
        device = component_params[0].device
        loss, _, _ = _faithfulness_loss(component_model, device, numel_global)
        loss.backward()
        warmup_opt.step()
    del warmup_opt
    torch.cuda.empty_cache()
