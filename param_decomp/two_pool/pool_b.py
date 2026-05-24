# pyright: reportArgumentType=false

"""Pool-B training step: stateless PPGD inner loop + cross-pool send/recv.

Pool B is a PPGD replica:

  1. Async-receives CI values from owning pool-A ranks; runs target forward
     (in parallel with the recv).
  2. Re-leafs CI as fp32 (it's wired in bf16 to save bandwidth).
  3. PPGD warmup — refines the persistent adversarial sources.
  4. Final PPGD recon loss with the refined sources → backwards for V/U + CI
     grads.
  5. Updates the persistent sources.
  6. SUM-reduces V/U grads within pool B.
  7. Sends grads to owning A ranks; receives back the updated V/U.
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
    """One training step on a pool-B rank using PersistentPGDState."""
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    device = next(component_model.parameters()).device

    # Mirror the single-pool persistent_pgd_recon metric: bump the PPGD LR
    # schedule once per training step before warmup/forward consume it.
    ppgd_state.update_lr(step=step, total_steps=n_steps)

    weight_deltas = component_model.calc_weight_deltas()

    sl = layout.my_batch_slice_b()
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch

    # The strategy's context manager controls whether the model's forwards
    # return logits or pre-LM-head hidden state. ppgd_state's recon_loss was
    # constructed from strategy.recon_loss to match.
    with strategy.context(component_model.target_model):
        # 1. Async: post irecvs for CI values from owning A ranks. The recvs run
        #    concurrently with target_fwd below — pool B doesn't actually need CI for
        #    target_fwd, so we save the recv latency by overlapping.
        with p.phase("b/1_post_async_recv_ci"):
            seq_len = batch_local.shape[1] if batch_local.ndim >= 2 else 1
            ci_recv, ci_recv_works = layout.async_recv_ci_from_owners(
                cfg.c_per_site,
                seq_len=seq_len,
                device=device,
                dtype=torch.float32,
            )

        # 2. Target forward (frozen, no grad) — runs in parallel with the CI recvs.
        with p.phase("b/2_target_fwd"), torch.no_grad(), autocast_bf16(cfg.bf16_autocast):
            target_out = component_model(batch_local).detach()

        # Now block on the CI recvs — should already be done by the time we get here.
        with p.phase("b/3_wait_ci_recv"):
            for w in ci_recv_works:
                w.wait()

        # 3. Re-leaf CI so we can produce ci grads to send back to pool A.
        # CI is received in bf16 (wire dtype); upcast to fp32 here so the leaf has
        # fp32 .grad that pool A can merge into its fp32 layerwise grads.
        ci_scratch = {
            s: v.detach().to(torch.float32).clone().requires_grad_(True) for s, v in ci_recv.items()
        }

        # 4. PPGD warmup — refines the persistent adversarial sources in-place
        with p.phase("b/4_ppgd_warmup"), autocast_bf16(cfg.bf16_autocast):
            ppgd_state.warmup(
                model=component_model,
                batch=batch_local,
                target_out=target_out,
                ci=ci_scratch,
                weight_deltas=weight_deltas,
            )

        # 5. Final PPGD recon loss with refined sources
        with p.phase("b/5_ppgd_recon"), autocast_bf16(cfg.bf16_autocast):
            sum_loss, n_examples = ppgd_state.compute_recon_sum_and_n(
                model=component_model,
                batch=batch_local,
                target_out=target_out,
                ci=ci_scratch,
                weight_deltas=weight_deltas,
            )
            loss_ppgd = sum_loss / n_examples
    # Scale by 1/N so a SUM-reduce of V/U grads across pool B equals the full-batch grad.
    total_ppgd = cfg.coeff_ppgd * loss_ppgd / layout.world.n_pool_b

    # 6. Extract V/U + ci_scratch grads via autograd (no .grad pollution)
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

    # 7. Update the persistent adversarial sources from the same loss
    with p.phase("b/7_ppgd_source_step"):
        source_grads = ppgd_state.get_grads(total_ppgd, retain_graph=False)
        ppgd_state.step(source_grads)

    # 8. SUM-reduce V/U grads within pool B
    with p.phase("b/8_pool_b_allreduce"):
        for s in all_sites:
            dist.all_reduce(v_grads[s], op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)
            dist.all_reduce(u_grads[s], op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)

    # 9. Send grads to owning A ranks
    with p.phase("b/9_send_grads_to_a"):
        layout.send_pool_b_grads_to_owners(v_grads, u_grads, ci_grads)

    # 10. Receive updated V/U from owning A ranks
    with p.phase("b/10_recv_weights"):
        v_templates = {s: component_model.components[s].V for s in all_sites}
        u_templates = {s: component_model.components[s].U for s in all_sites}
        v_new, u_new = layout.recv_updated_weights_from_owners(v_templates, u_templates)
        with torch.no_grad():
            for s in all_sites:
                component_model.components[s].V.copy_(v_new[s])
                component_model.components[s].U.copy_(u_new[s])

    return {"loss/ppgd": loss_ppgd.item()}
