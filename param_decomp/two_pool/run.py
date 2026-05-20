"""`optimize_two_pool` — a parallel orchestrator to `run_pd.optimize` that bakes in
the 2-pool training strategy.

Same composition primitives as the single-pool path (ComponentModel, PersistentPGDState,
ReconstructionLoss, AdamW), wired differently:

  - On pool A ranks, ComponentModel is built with `module_path_info` restricted to the
    rank's owned sites. Pool A computes target+CI forward, per-site layerwise loss
    (a serial M_local-iteration loop), faithfulness, importance-minimality, runs the
    combined backward seeded by pool B's ci grads, in-block all-reduce, and the
    AdamW optimizer step.

  - On pool B ranks, ComponentModel is built with `module_path_info` covering every
    site (full replica for the full-model PPGD forward). Pool B uses the actual
    PersistentPGDState (PerBatchPerPositionScope, persistent sources across steps),
    sends per-site V/U + per-slice ci grads back to pool A, and receives updated V/U
    after pool A's optimizer step.

This module deliberately stays at the same composition level as `run_pd.optimize` —
it doesn't introduce new lower-level primitives, it just orchestrates the existing
ones under a different topology. Conditionals stay out of `run_pd.optimize`; the
2-pool strategy lives here.

Current status: functional minimal version. The benchmark in
`param_decomp/scripts/two_pool_benchmark/two_pool.py` exercises this path. Eval,
wandb logging, full metric registry integration, and checkpointing are TODO and
should be lifted from `run_pd.optimize` as they become needed.
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false

import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from param_decomp.configs import CiConfig, PersistentPGDReconLossConfig
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import make_mask_infos
from param_decomp.models.sigmoids import SigmoidType
from param_decomp.persistent_pgd import PersistentPGDState
from param_decomp.two_pool.install import (
    build_pool_a_module_path_info,
    build_pool_b_module_path_info,
)
from param_decomp.two_pool.layout import BlockDDPLayout, BlockGroup

# ───────────────────────── Public config ─────────────────────────


@dataclass
class PhaseProfiler:
    """Optional inline per-phase timer for `optimize_two_pool`.

    Each phase wraps a chunk of compute/comm with cuda.synchronize() + perf_counter
    on entry and exit. Synchronizing serializes async ops — so this is for
    DIAGNOSING where time goes, not for production runs. Disable when measuring
    real wall-clock; enable when chasing Amdahl's-law-style bottlenecks.
    """

    times: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    enabled: bool = True

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.times[name].append((time.perf_counter() - t0) * 1000.0)

    def report(self, warmup: int = 0) -> str:
        lines = []
        for name, vals in self.times.items():
            vals = vals[warmup:]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            lines.append(
                f"  {name:32s} avg={avg:9.2f}ms  min={min(vals):8.2f}ms  max={max(vals):8.2f}ms"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class TwoPoolConfig:
    """Topology and per-pool knobs for `optimize_two_pool`.

    Block groups and pool-B ranks are explicit so this config can describe any
    topology the BlockDDPLayout supports — including the cross-node round-robin
    layouts we measured in nano.

    `c_per_site` is the component count per decomposed module. The matching
    `ci_config` (LayerwiseCiConfig instance) governs how per-module CI fns are built.
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
    coeff_faith: float = 1e6
    coeff_imp: float = 1e-4
    coeff_stoch: float = 0.5
    coeff_ppgd: float = 0.5
    lr_components: float = 5e-5
    lr_ci_fn: float = 5e-5


# ───────────────────────── Helpers ─────────────────────────


def _faithfulness_loss(
    component_model: ComponentModel,
    device: torch.device,
) -> Tensor:
    """Standard faithfulness loss: ‖W_target − VU.T‖²_F / numel, summed across sites."""
    weight_deltas = component_model.calc_weight_deltas()
    sum_sq = torch.zeros((), device=device)
    numel = 0
    for d in weight_deltas.values():
        sum_sq = sum_sq + (d**2).sum()
        numel += d.numel()
    return sum_sq / numel


def _importance_minimality_loss(
    ci_upper: dict[str, Tensor],
    device: torch.device,
    p: float = 1.0,
    eps: float = 1e-12,
) -> Tensor:
    """L_p importance penalty summed across owned sites."""
    total = torch.zeros((), device=device)
    for v in ci_upper.values():
        vals = (v + eps).pow(p)
        batch_seq_dims = tuple(range(vals.ndim - 1))
        sum_c = vals.sum(dim=batch_seq_dims)
        import math

        mean_c = sum_c / math.prod(vals.shape[:-1])
        total = total + mean_c.sum()
    return total


def _layerwise_loss_local(
    component_model: ComponentModel,
    batch_local: Any,
    target_logits_local: Tensor,
    ci_lower_local: dict[str, Tensor],
    owned_sites: tuple[str, ...],
    recon_loss: ReconstructionLoss,
) -> Tensor:
    """Single-site layerwise loss serially over owned sites on a pool-A rank."""
    losses: list[Tensor] = []
    for s in owned_sites:
        ci_s = ci_lower_local[s]
        u = torch.rand_like(ci_s)
        mask = ci_s + (1 - ci_s) * u
        mask_infos = make_mask_infos({s: mask}, routing_masks="all")
        pred = component_model(batch_local, mask_infos=mask_infos)
        loss, _ = recon_loss(pred=pred, target=target_logits_local)
        losses.append(loss)
    return torch.stack(losses).mean()


# ───────────────────────── Pool A step ─────────────────────────


def step_pool_a(
    layout: BlockDDPLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    batch: Any,
    cfg: TwoPoolConfig,
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

    # 1. target + CI forward; CI fn graph retained.
    with p.phase("a/1_target_and_ci_fwd"):
        out = component_model(batch, cache_type="input")
        target_logits = out.output
        ci = component_model.calc_causal_importances(
            pre_weight_acts=out.cache,
            sampling="continuous",
            detach_inputs=False,
        )

    # 2. Cross-pool: send CI values to pool B (async — don't block on pool B's recv).
    with p.phase("a/2_async_send_ci"):
        ci_send_works, ci_send_buffers = layout.async_send_owned_ci_to_pool_b(
            {s: ci.lower_leaky[s] for s in layout.my_owned_sites}
        )

    # 3. Home losses (forward)
    device = target_logits.device
    with p.phase("a/3_faith"):
        loss_faith = _faithfulness_loss(component_model, device)
    with p.phase("a/4_imp"):
        loss_imp = _importance_minimality_loss(ci.upper_leaky, device)
    with p.phase("a/5_layerwise"):
        sl = layout.my_batch_slice_a()
        batch_local = batch[sl] if isinstance(batch, Tensor) else batch
        target_local = target_logits[sl].detach()
        ci_local = {s: ci.lower_leaky[s][sl] for s in layout.my_owned_sites}
        loss_stoch = _layerwise_loss_local(
            component_model,
            batch_local,
            target_local,
            ci_local,
            layout.my_owned_sites,
            cfg.reconstruction_loss,
        )

    total_home = (
        cfg.coeff_faith * loss_faith + cfg.coeff_imp * loss_imp + cfg.coeff_stoch * loss_stoch
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

    # 5. Seed V/U .grad with pool-B contribution; combined backward.
    with p.phase("a/7_seed_and_backward"):
        optimizer.zero_grad(set_to_none=True)
        for s in layout.my_owned_sites:
            comp = component_model.components[s]
            comp.V.grad = v_grads[s]
            comp.U.grad = u_grads[s]
        torch.autograd.backward(
            tensors=[total_home, *(ci.lower_leaky[s] for s in layout.my_owned_sites)],
            grad_tensors=[None, *(ci_grads[s] for s in layout.my_owned_sites)],
        )

    # 6. In-block DDP sync.
    with p.phase("a/8_in_block_allreduce"):
        layout.all_reduce_grads_in_block(all_params)

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
    component_model._pending_weight_sends = weight_send_works, weight_send_buffers

    return {
        "loss/faith": loss_faith.item(),
        "loss/imp": loss_imp.item(),
        "loss/stoch": loss_stoch.item(),
    }


# ───────────────────────── Pool B step ─────────────────────────


def step_pool_b(
    layout: BlockDDPLayout,
    component_model: ComponentModel,
    ppgd_state: PersistentPGDState,
    batch: Any,
    cfg: TwoPoolConfig,
    profiler: PhaseProfiler | None = None,
) -> dict[str, float]:
    """One training step on a pool-B rank using PersistentPGDState."""
    p = profiler if profiler is not None else PhaseProfiler(enabled=False)
    device = next(component_model.parameters()).device

    sl = layout.my_batch_slice_b()
    batch_local = batch[sl] if isinstance(batch, Tensor) else batch

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

    # 2. Target forward (frozen, no grad) — runs in parallel with the CI recvs above.
    with p.phase("b/2_target_fwd"), torch.no_grad():
        target_logits = component_model(batch_local)

    # Now block on the CI recvs — should already be done by the time we get here.
    with p.phase("b/3_wait_ci_recv"):
        for w in ci_recv_works:
            w.wait()

    # 3. Re-leaf CI so we can produce ci grads to send back to pool A
    ci_scratch = {s: v.detach().clone().requires_grad_(True) for s, v in ci_recv.items()}

    # 4. PPGD warmup — refines the persistent adversarial sources in-place
    with p.phase("b/4_ppgd_warmup"):
        ppgd_state.warmup(
            model=component_model,
            batch=batch_local,
            target_out=target_logits.detach(),
            ci=ci_scratch,
            weight_deltas=None,
        )

    # 5. Final PPGD recon loss with refined sources
    with p.phase("b/5_ppgd_recon"):
        loss_ppgd = ppgd_state.compute_recon_loss(
            model=component_model,
            batch=batch_local,
            target_out=target_logits.detach(),
            ci=ci_scratch,
            weight_deltas=None,
        )
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


# ───────────────────────── Public entry point ─────────────────────────


def optimize_two_pool(
    target_model: nn.Module,
    pool_config: TwoPoolConfig,
    device: torch.device,
    n_steps: int,
    batch_iter: Callable[[int], Any],
    *,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
    enable_tf32: bool = True,
    fused_optimizer: bool = True,
    profiler: PhaseProfiler | None = None,
) -> tuple[ComponentModel, BlockDDPLayout, PhaseProfiler | None]:
    """Train a ComponentModel under the 2-pool strategy.

    Composes the same primitives that `run_pd.optimize` uses (ComponentModel,
    PersistentPGDState, ReconstructionLoss, AdamW) but orchestrated under a
    `BlockDDPLayout`. Pool A and pool B do different work and exchange gradients
    via the layout's cross-pool comm methods.

    Args:
        target_model: A frozen target whose decomposable modules' paths appear in
            `pool_config.c_per_site`. Must already be on `device` (or this function
            will move ComponentModel — target stays on whatever device the caller put it).
        pool_config: Topology + per-pool knobs. See `TwoPoolConfig`.
        device: The CUDA device for this rank.
        n_steps: Number of training steps to run.
        batch_iter: Callable taking step idx → batch (anything `run_batch` accepts).
        on_step: Optional callback invoked after each step with (step, metrics dict).

    Returns:
        (component_model, layout) so callers can introspect post-training state.

    The function assumes `dist.init_process_group` has already been called.
    """
    assert dist.is_initialized(), (
        "init the distributed process group before calling optimize_two_pool"
    )
    rank = dist.get_rank()

    # TF32 matmuls are ~2-3x faster on H200 with sub-ULP precision loss — fine
    # for SPD training where we already use fp32 throughout.
    if enable_tf32:
        torch.set_float32_matmul_precision("high")

    from param_decomp.two_pool.layout import build_block_ddp_world

    world = build_block_ddp_world(
        block_groups=list(pool_config.block_groups),
        pool_b_ranks=list(pool_config.pool_b_ranks),
        batch_global=pool_config.batch_global,
    )
    layout = BlockDDPLayout.from_world(world, rank)

    if layout.my_pool == "a":
        mpi = build_pool_a_module_path_info(layout, pool_config.c_per_site)
    else:
        mpi = build_pool_b_module_path_info(layout, pool_config.c_per_site)

    target_model.requires_grad_(False)
    component_model = ComponentModel(
        target_model=target_model,
        run_batch=pool_config.run_batch,
        module_path_info=mpi,
        ci_config=pool_config.ci_config,
        sigmoid_type=pool_config.sigmoid_type,
    ).to(device)

    optimizer: torch.optim.Optimizer | None = None
    all_params: list[nn.Parameter] = []
    ppgd_state: PersistentPGDState | None = None

    if layout.my_pool == "a":
        component_params: list[nn.Parameter] = []
        for name in component_model.target_module_paths:
            component_params.extend(component_model.components[name].parameters())
        ci_fn_params = list(component_model.ci_fn.parameters())
        all_params = component_params + ci_fn_params
        # One optimizer covering both component params and CI fn — matches the
        # benchmark's behaviour and is a fine default. Split-optimizer (one per group)
        # would be a natural future option mirroring `run_pd.optimize`.
        # fused=True uses a CUDA-fused kernel, noticeably faster on H200 vs the
        # Python foreach implementation that's the default.
        optimizer = torch.optim.AdamW(
            [
                {"params": component_params, "lr": pool_config.lr_components},
                {"params": ci_fn_params, "lr": pool_config.lr_ci_fn},
            ],
            weight_decay=0.0,
            fused=fused_optimizer,
        )
    else:
        ppgd_state = PersistentPGDState(
            module_to_c=pool_config.c_per_site,
            batch_dims=(layout.world.batch_local_b, *_seq_dims_from_batch_iter(batch_iter)),
            device=device,
            use_delta_component=False,
            cfg=pool_config.ppgd_cfg,
            reconstruction_loss=pool_config.reconstruction_loss,
        )

    for step in range(n_steps):
        batch = batch_iter(step)
        if layout.my_pool == "a":
            assert optimizer is not None
            metrics = step_pool_a(
                layout,
                component_model,
                optimizer,
                all_params,
                batch,
                pool_config,
                profiler=profiler,
            )
        else:
            assert ppgd_state is not None
            metrics = step_pool_b(
                layout,
                component_model,
                ppgd_state,
                batch,
                pool_config,
                profiler=profiler,
            )

        if on_step is not None:
            on_step(step, metrics)

    return component_model, layout, profiler


def _seq_dims_from_batch_iter(batch_iter: Callable[[int], Any]) -> tuple[int, ...]:
    """Peek at a single batch to determine seq dim(s). Assumes the dataloader is restartable.

    For token batches the shape is (B, S); we return (S,). For arbitrary batch shapes the
    caller can override by computing batch_dims manually.
    """
    sample = batch_iter(0)
    if isinstance(sample, Tensor):
        return tuple(sample.shape[1:])
    raise TypeError(f"Cannot infer seq dims from batch of type {type(sample).__name__}")
