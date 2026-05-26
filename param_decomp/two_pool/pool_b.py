# pyright: reportArgumentType=false

"""Pool-B training step: stateless PPGD inner loop + cross-pool send/recv.

Pool B is a PPGD replica. Each rank holds full-target V/U replicas (received
from pool A each step) and runs PPGD locally on its batch shard:

  1. Recv CI values from owning pool-A ranks (async, overlapped with
     target_fwd that doesn't depend on CI).
  2. Re-leaf CI as fp32 — it's wired in bf16 to save bandwidth, but pool A
     wants fp32 grads back.
  3. PPGD warmup refines the persistent adversarial sources in place.
  4. Final PPGD recon loss with refined sources → autograd.grad to extract
     V/U + CI gradients without polluting .grad.
  5. Source-tensor optimizer step on the same loss.
  6. SUM-reduce V/U grads within pool B (so pool A's incoming = full-batch).
  7. Ship grads to owning A ranks; receive updated V/U for next step.

``step_pool_b`` reads as ~25 lines of orchestration; the named helpers below
correspond one-to-one with the phases above. Profiler tags (``b/<n>_<name>``)
are preserved so HTA traces remain comparable.
"""

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from param_decomp.component_model import ComponentModel
from param_decomp.metrics.persistent_pgd_state import PersistentPGDState
from param_decomp.two_pool.layout import BlockDDPLayout
from param_decomp.two_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp.two_pool.profiler import PhaseProfiler
from param_decomp.two_pool.runtime import _TwoPoolRuntime, autocast_bf16


def step_pool_b(
    layout: BlockDDPLayout,
    component_model: ComponentModel,
    ppgd_state: PersistentPGDState,
    batch: Any,
    cfg: _TwoPoolRuntime,
    strategy: LayerwiseLossStrategy,
    step: int,
    n_steps: int,
    profiler: PhaseProfiler | None = None,
) -> dict[str, float]:
    """One training step on a pool-B rank.

    Reads as orchestration: each call delegates to a same-file helper that
    implements one numbered phase. See the module docstring for the phase
    list.
    """
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    device = next(component_model.parameters()).device
    batch_local = _slice_batch_for_pool_b(batch, layout)

    # Mirror single-pool's persistent_pgd_recon: advance LR once per step
    # before warmup/forward consume it.
    ppgd_state.update_lr(step=step, total_steps=n_steps)
    weight_deltas = component_model.calc_weight_deltas()

    # ``strategy.context`` controls whether forwards return logits or pre-LM-
    # head hidden state; ``ppgd_state``'s recon_loss was built to match.
    with strategy.context(component_model.target_model):
        ci_recv, ci_recv_works = _async_recv_ci_from_pool_a(layout, cfg, batch_local, device, p)
        target_out = _target_forward_no_grad(component_model, batch_local, cfg, p)
        _wait_ci_recv(ci_recv_works, p)
        ci_scratch = _releaf_ci_fp32_for_grads(ci_recv)

        _ppgd_inner_warmup(
            ppgd_state, component_model, batch_local, target_out, ci_scratch, weight_deltas, cfg, p
        )
        sum_loss, n_examples = _ppgd_recon_forward(
            ppgd_state, component_model, batch_local, target_out, ci_scratch, weight_deltas, cfg, p
        )

    # Scale by 1/N_pool_b so the upcoming SUM-reduce of V/U grads across pool B
    # produces the full-batch gradient (each rank contributes its slice mean ÷ N).
    total_ppgd = cfg.coeff_ppgd * (sum_loss / n_examples) / layout.world.n_pool_b

    v_grads, u_grads, ci_grads = _autograd_grad_for_vu_and_ci(
        total_ppgd, component_model, ci_scratch, layout, p
    )
    _ppgd_source_step_from_total_loss(ppgd_state, total_ppgd, p)
    _sum_reduce_vu_grads_within_pool_b(v_grads, u_grads, layout, p)
    _send_grads_to_pool_a(layout, v_grads, u_grads, ci_grads, p)
    _recv_updated_vu_from_pool_a(component_model, layout, p)

    return _step_metrics(sum_loss, n_examples)


# =============================================================================
# Phase helpers — each implements one numbered phase from the module docstring.
# =============================================================================


def _slice_batch_for_pool_b(batch: Any, layout: BlockDDPLayout) -> Any:
    """Pool B batch-shards across its ranks; pull this rank's slice."""
    sl = layout.my_batch_slice_b()
    return batch[sl] if isinstance(batch, Tensor) else batch


def _async_recv_ci_from_pool_a(
    layout: BlockDDPLayout,
    cfg: _TwoPoolRuntime,
    batch_local: Any,
    device: torch.device,
    p: PhaseProfiler,
) -> tuple[dict[str, Tensor], list[Any]]:
    """Phase b/1. Post irecvs for CI values from owning A ranks.

    Returns ``(ci_recv, ci_recv_works)``. The works run on NIC concurrently
    with the next phase's target_fwd on the GPU — pool B doesn't need CI for
    target_fwd, so we save the recv latency by overlapping.
    """
    with p.phase("b/1_post_async_recv_ci"):
        seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
        ci_recv, works = layout.async_recv_ci_from_owners(
            cfg.c_per_site,
            seq_len=seq_len,
            device=device,
            dtype=torch.float32,
        )
    # Pre-allocated buffers — values aren't filled until ``works`` are waited
    # on, but shapes are fixed at allocation time and the post-wait shape
    # must match what pool A sends. The assertion here costs nothing and
    # would catch a wrong c_per_site config or a per-rank batch size
    # mismatch between pools.
    batch_local_b = layout.world.batch_local_b
    for s, c in cfg.c_per_site.items():
        t = ci_recv[s]
        assert t.shape == (batch_local_b, seq_len, c), (
            f"ci_recv[{s!r}] shape {tuple(t.shape)} != expected ({batch_local_b}, {seq_len}, {c})"
        )
    return ci_recv, works


def _target_forward_no_grad(
    component_model: ComponentModel,
    batch_local: Any,
    cfg: _TwoPoolRuntime,
    p: PhaseProfiler,
) -> Tensor:
    """Phase b/2. Frozen target forward on the rank's batch slice.

    Detached + no_grad — we only need the output as the recon target. Runs
    in parallel with the CI recvs posted in phase b/1.
    """
    with p.phase("b/2_target_fwd"), torch.no_grad(), autocast_bf16(cfg.bf16_autocast):
        return component_model(batch_local).detach()


def _wait_ci_recv(ci_recv_works: list[Any], p: PhaseProfiler) -> None:
    """Phase b/3. Block on the irecvs from phase b/1 — should already be done."""
    with p.phase("b/3_wait_ci_recv"):
        for w in ci_recv_works:
            w.wait()


def _releaf_ci_fp32_for_grads(ci_recv: dict[str, Tensor]) -> dict[str, Tensor]:
    """Re-leaf received CI tensors as fp32 ``requires_grad=True``.

    CI is shipped in bf16 to save bandwidth; we upcast and re-leaf here so the
    leaf has fp32 ``.grad`` that pool A can merge into its fp32 layerwise grads.
    """
    return {
        s: v.detach().to(torch.float32).clone().requires_grad_(True) for s, v in ci_recv.items()
    }


def _ppgd_inner_warmup(
    ppgd_state: PersistentPGDState,
    component_model: ComponentModel,
    batch_local: Any,
    target_out: Tensor,
    ci_scratch: dict[str, Tensor],
    weight_deltas: dict[str, Tensor],
    cfg: _TwoPoolRuntime,
    p: PhaseProfiler,
) -> None:
    """Phase b/4. Refine the persistent adversarial sources in place."""
    with p.phase("b/4_ppgd_warmup"), autocast_bf16(cfg.bf16_autocast):
        ppgd_state.warmup(
            model=component_model,
            batch=batch_local,
            target_out=target_out,
            ci=ci_scratch,
            weight_deltas=weight_deltas,
        )


def _ppgd_recon_forward(
    ppgd_state: PersistentPGDState,
    component_model: ComponentModel,
    batch_local: Any,
    target_out: Tensor,
    ci_scratch: dict[str, Tensor],
    weight_deltas: dict[str, Tensor],
    cfg: _TwoPoolRuntime,
    p: PhaseProfiler,
) -> tuple[Tensor, int]:
    """Phase b/5. Final recon loss with the refined sources.

    Returns ``(sum_loss, n_examples)`` raw so the logger can SUM-reduce across
    pool B to recover ``global_mean = SUM(sum_loss) / SUM(n_examples)``.
    """
    with p.phase("b/5_ppgd_recon"), autocast_bf16(cfg.bf16_autocast):
        sum_loss, n_examples = ppgd_state.compute_recon_sum_and_n(
            model=component_model,
            batch=batch_local,
            target_out=target_out,
            ci=ci_scratch,
            weight_deltas=weight_deltas,
        )
    assert sum_loss.dim() == 0, f"sum_loss should be scalar; got {sum_loss.shape}"
    assert n_examples > 0, f"n_examples must be positive; got {n_examples}"
    return sum_loss, n_examples


def _autograd_grad_for_vu_and_ci(
    total_ppgd: Tensor,
    component_model: ComponentModel,
    ci_scratch: dict[str, Tensor],
    layout: BlockDDPLayout,
    p: PhaseProfiler,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Phase b/6. Extract V/U + CI gradients via ``torch.autograd.grad``.

    Uses ``autograd.grad`` rather than ``.backward()`` so we don't pollute the
    components' ``.grad`` accumulator — pool A owns the optimizer, and we'll
    ship these grads over the wire instead. ``retain_graph=True`` keeps the
    graph alive for the upcoming PPGD source-step (which traverses the same
    forward with respect to the source tensors).
    """
    with p.phase("b/6_backward"):
        all_sites = list(layout.world.all_sites)
        params: list[Tensor] = []
        for s in all_sites:
            params.append(component_model.components[s].V)
            params.append(component_model.components[s].U)
        ci_list = [ci_scratch[s] for s in all_sites]
        grads = torch.autograd.grad(total_ppgd, params + ci_list, retain_graph=True)

    n_sites = len(all_sites)
    v_grads = {s: grads[2 * i] for i, s in enumerate(all_sites)}
    u_grads = {s: grads[2 * i + 1] for i, s in enumerate(all_sites)}
    ci_grads = {s: grads[2 * n_sites + i] for i, s in enumerate(all_sites)}
    for s in all_sites:
        assert v_grads[s].shape == component_model.components[s].V.shape, (
            f"v_grad[{s!r}] shape {v_grads[s].shape} != "
            f"V.shape {component_model.components[s].V.shape}"
        )
        assert u_grads[s].shape == component_model.components[s].U.shape, (
            f"u_grad[{s!r}] shape {u_grads[s].shape} != "
            f"U.shape {component_model.components[s].U.shape}"
        )
        assert ci_grads[s].shape == ci_scratch[s].shape, (
            f"ci_grad[{s!r}] shape {ci_grads[s].shape} != ci_scratch shape {ci_scratch[s].shape}"
        )
    return v_grads, u_grads, ci_grads


def _ppgd_source_step_from_total_loss(
    ppgd_state: PersistentPGDState,
    total_ppgd: Tensor,
    p: PhaseProfiler,
) -> None:
    """Phase b/7. Update the persistent adversarial sources from the same loss.

    ``retain_graph=False`` here — the upstream call already used
    ``retain_graph=True`` to keep the graph alive for this final pass.
    """
    with p.phase("b/7_ppgd_source_step"):
        source_grads = ppgd_state.get_grads(total_ppgd, retain_graph=False)
        ppgd_state.step(source_grads)


def _sum_reduce_vu_grads_within_pool_b(
    v_grads: dict[str, Tensor],
    u_grads: dict[str, Tensor],
    layout: BlockDDPLayout,
    p: PhaseProfiler,
) -> None:
    """Phase b/8. SUM-reduce V/U grads within pool B.

    Combined with the ``/n_pool_b`` scaling on the loss, the SUM gives the
    full-batch gradient that pool A would have computed if it had owned PPGD.
    """
    with p.phase("b/8_pool_b_allreduce"):
        for s in layout.world.all_sites:
            dist.all_reduce(v_grads[s], op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)
            dist.all_reduce(u_grads[s], op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)


def _send_grads_to_pool_a(
    layout: BlockDDPLayout,
    v_grads: dict[str, Tensor],
    u_grads: dict[str, Tensor],
    ci_grads: dict[str, Tensor],
    p: PhaseProfiler,
) -> None:
    """Phase b/9. Ship V/U + CI grads to the owning pool-A ranks."""
    with p.phase("b/9_send_grads_to_a"):
        layout.send_pool_b_grads_to_owners(v_grads, u_grads, ci_grads)


def _recv_updated_vu_from_pool_a(
    component_model: ComponentModel,
    layout: BlockDDPLayout,
    p: PhaseProfiler,
) -> None:
    """Phase b/10. Receive updated V/U from pool A and copy in place.

    In-place copy avoids reallocating ``components[s].V`` / ``.U`` (which would
    detach them from the optimizer if pool B had one — defense in depth even
    though pool B doesn't).
    """
    with p.phase("b/10_recv_weights"):
        all_sites = layout.world.all_sites
        v_templates = {s: component_model.components[s].V for s in all_sites}
        u_templates = {s: component_model.components[s].U for s in all_sites}
        v_new, u_new = layout.recv_updated_weights_from_owners(v_templates, u_templates)
        with torch.no_grad():
            for s in all_sites:
                component_model.components[s].V.copy_(v_new[s])
                component_model.components[s].U.copy_(u_new[s])


def _step_metrics(sum_loss: Tensor, n_examples: int) -> dict[str, float]:
    """Per-step metrics dict: per-rank display scalar + raw (num, den) for the
    cross-pool-B logger reduction.

    Pool B batch-shards across its ranks. Global mean per position is
    ``SUM(sum_loss) / SUM(n_examples)`` across the pool, NOT AVG of per-rank
    ratios. ``n_examples`` is the same on every PPGD rank so AVG happens to
    coincide, but reporting raw keeps the contract uniform with pool A's
    cross-block reductions.
    """
    return {
        "loss/ppgd": (sum_loss / n_examples).item(),
        "_raw/ppgd_num": sum_loss.item(),
        "_raw/ppgd_den": float(n_examples),
    }
