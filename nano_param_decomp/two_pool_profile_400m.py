"""Wall-clock profile of stage 5 at ~400M target / ~4B CI fn scale.

Scaled-up version of `two_pool_profile.py`. Same topology (3 block groups × 2 ranks +
2 pool B ranks = 8 GPUs), but each block group owns 4 transformer blocks (= 28 sites)
so we get 12 transformer blocks total — needed to hit the 400M target size with only
6 pool A ranks.

Sizing:
  target_model  ≈ 390M params  (12 transformer blocks, d_model=1280, d_mlp=5120)
  ci_fn (total) ≈ 3.4B params  (per-module MLPs with hidden=22000)
  per pool-A rank CI fn: ~1.1B params
  per pool-A rank: ~20GB GPU memory (fp32 + AdamW state for CI fns dominates)

Run:
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \\
        -m nano_param_decomp.two_pool_profile_400m
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

import math
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from nano_param_decomp.run import (
    ComponentLinear,
    Config,
    PersistentPGD,
    anneal_p,
    clear_wrapper_masks,
    kl_logits,
)
from nano_param_decomp.two_pool.block_ddp import (
    BlockDDPLayout,
    build_block_ddp_world,
    build_ci_fns_for_block_ddp,
    install_components_for_block_ddp,
)
from nano_param_decomp.two_pool_stage2 import ModuleCIFn, ci_forward
from nano_param_decomp.two_pool_stage4 import (
    Tiny6BlockTransformer,
    faith_loss_owned,
    imp_loss_owned,
    sites_for_block,
)

# --- Topology ---

N_BLOCK_GROUPS = 3
N_PER_BLOCK_GROUP = 2  # in-block DDP size
N_POOL_B = 2

# Each block group owns 4 transformer blocks → 12 transformer blocks total → 84 sites.
N_TRANSFORMER_BLOCKS = 12
BLOCKS_PER_GROUP = N_TRANSFORMER_BLOCKS // N_BLOCK_GROUPS

# --- Model dimensions ---

VOCAB = 32000
D_MODEL = 1280
N_HEADS = 16
D_MLP = 5120
BATCH = 4
SEQ_LEN = 64
C = 64
CI_HIDDEN = 22000

WARMUP_STEPS = 2
PROFILE_STEPS = 6


# --- Timer ---


class StepTimer:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.times: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.times[name].append(elapsed_ms)

    def report(self, warmup: int) -> None:
        print(f"\n[rank{self.rank}] phase wall-clock (skipping first {warmup} steps):", flush=True)
        total_avg = 0.0
        for name, vals in self.times.items():
            vals = vals[warmup:]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            mn = min(vals)
            mx = max(vals)
            total_avg += avg
            print(
                f"  {name:32s} avg={avg:9.2f}ms  min={mn:9.2f}ms  max={mx:9.2f}ms  (n={len(vals)})",
                flush=True,
            )
        print(f"  {'TOTAL_PER_STEP':32s} avg={total_avg:9.2f}ms", flush=True)


# --- Layerwise loss (per-site, timed) ---


def single_site_layerwise_loss_local_timed(
    timer: StepTimer,
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    owned_sites: tuple[str, ...],
    input_ids_local: Tensor,
    target_logits_local: Tensor,
    ci_lower_local: dict[str, Tensor],
) -> Tensor:
    losses: list[Tensor] = []
    for s in owned_sites:
        with timer.phase("a/layerwise/per_site"):
            ci = ci_lower_local[s]
            u = torch.rand_like(ci)
            mask = ci + (1 - ci) * u
            delta_mask = torch.rand(*ci.shape[:-1], device=ci.device, dtype=ci.dtype)
            for name, w in wrappers.items():
                if name == s:
                    w.mode = "component"
                    w.mask = mask
                    w.delta_mask = delta_mask
                    w.routing_mask = None
                else:
                    w.mode = "target"
                    w.mask = None
                    w.delta_mask = None
                    w.routing_mask = None
            pred = target_model(input_ids_local)
            losses.append(kl_logits(pred, target_logits_local))
    clear_wrapper_masks(wrappers)
    return torch.stack(losses).mean()


# --- Pool A step ---


def pool_a_step(
    timer: StepTimer,
    layout: BlockDDPLayout,
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ci_fns: dict[str, ModuleCIFn],
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    cfg: Config,
    imp_p: float,
) -> dict[str, float]:
    with timer.phase("a/target_fwd"):
        clear_wrapper_masks(wrappers)
        target_logits = target_model(input_ids)

    with timer.phase("a/ci_fwd"):
        acts_owned = {s: wrappers[s].last_input for s in layout.my_owned_sites}
        ci_lower_owned, ci_upper_owned = ci_forward(ci_fns, acts_owned)

    with timer.phase("a/send_ci_to_b"):
        layout.send_owned_ci_to_pool_b(ci_lower_owned)

    with timer.phase("a/faith"):
        loss_faith = faith_loss_owned(wrappers, layout.my_owned_sites)
    with timer.phase("a/imp"):
        loss_imp = imp_loss_owned(ci_upper_owned, imp_p, cfg.imp_eps, cfg.imp_beta)

    with timer.phase("a/layerwise_total"):
        sl = layout.my_batch_slice_a()
        loss_stoch = single_site_layerwise_loss_local_timed(
            timer,
            target_model,
            wrappers,
            layout.my_owned_sites,
            input_ids[sl],
            target_logits[sl].detach(),
            {s: ci_lower_owned[s][sl] for s in layout.my_owned_sites},
        )

    total_home = (
        cfg.coeff_faith * loss_faith + cfg.coeff_imp * loss_imp + cfg.coeff_stoch * loss_stoch
    )

    with timer.phase("a/recv_grads_from_b"):
        v_grads, u_grads, ci_grads = layout.recv_grads_from_pool_b(wrappers, ci_lower_owned)

    with timer.phase("a/seed_and_backward"):
        optimizer.zero_grad(set_to_none=True)
        for s in layout.my_owned_sites:
            wrappers[s].V.grad = v_grads[s]
            wrappers[s].U.grad = u_grads[s]
        torch.autograd.backward(
            tensors=[total_home, *(ci_lower_owned[s] for s in layout.my_owned_sites)],
            grad_tensors=[None, *(ci_grads[s] for s in layout.my_owned_sites)],
        )

    with timer.phase("a/in_block_allreduce"):
        params_to_reduce: list[nn.Parameter] = []
        for s in layout.my_owned_sites:
            params_to_reduce.extend([wrappers[s].V, wrappers[s].U])
        for f in ci_fns.values():
            params_to_reduce.extend(list(f.parameters()))
        layout.all_reduce_grads_in_block(params_to_reduce)

    with timer.phase("a/opt_step"):
        optimizer.step()

    with timer.phase("a/send_weights_to_b"):
        v_owned = {s: wrappers[s].V for s in layout.my_owned_sites}
        u_owned = {s: wrappers[s].U for s in layout.my_owned_sites}
        layout.send_updated_weights_to_pool_b(v_owned, u_owned)

    return {
        "loss/faith": loss_faith.item(),
        "loss/imp": loss_imp.item(),
        "loss/stoch": loss_stoch.item(),
    }


# --- Pool B step ---


def pool_b_step(
    timer: StepTimer,
    layout: BlockDDPLayout,
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ppgd: PersistentPGD,
    input_ids_full: Tensor,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    sl = layout.my_batch_slice_b()
    input_ids = input_ids_full[sl]

    with timer.phase("b/recv_ci_from_a"):
        ci_recv = layout.recv_ci_from_owners(
            wrappers, seq_len=input_ids.shape[1], device=device, dtype=torch.float32
        )

    with timer.phase("b/target_fwd"):
        clear_wrapper_masks(wrappers)
        with torch.no_grad():
            target_logits = target_model(input_ids)

    ci_scratch = {n: v.detach().clone().requires_grad_(True) for n, v in ci_recv.items()}

    with timer.phase("b/ppgd_warmup"):
        ppgd.warmup(target_model, wrappers, input_ids, target_logits, ci_scratch, lr=cfg.ppgd_lr)

    with timer.phase("b/ppgd_recon"):
        loss_ppgd = ppgd.recon_loss(target_model, wrappers, input_ids, target_logits, ci_scratch)
        total_ppgd = cfg.coeff_ppgd * loss_ppgd / layout.world.n_pool_b

    with timer.phase("b/backward"):
        params: list[Tensor] = []
        for site in layout.world.all_sites:
            params.extend([wrappers[site].V, wrappers[site].U])
        ci_list = [ci_scratch[s] for s in layout.world.all_sites]
        grads = torch.autograd.grad(total_ppgd, params + ci_list)

    n_sites = len(layout.world.all_sites)
    v_grads = {s: grads[2 * i] for i, s in enumerate(layout.world.all_sites)}
    u_grads = {s: grads[2 * i + 1] for i, s in enumerate(layout.world.all_sites)}
    ci_grads = {s: grads[2 * n_sites + i] for i, s in enumerate(layout.world.all_sites)}

    with timer.phase("b/pool_b_allreduce"):
        for s in layout.world.all_sites:
            dist.all_reduce(v_grads[s], op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)
            dist.all_reduce(u_grads[s], op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)

    with timer.phase("b/send_grads_to_a"):
        layout.send_pool_b_grads_to_owners(v_grads, u_grads, ci_grads)

    with timer.phase("b/recv_weights_from_a"):
        v_new, u_new = layout.recv_updated_weights_from_owners(wrappers)
        with torch.no_grad():
            for s in layout.world.all_sites:
                wrappers[s].V.copy_(v_new[s])
                wrappers[s].U.copy_(u_new[s])

    return {"loss/ppgd": loss_ppgd.item()}


# --- Main ---


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected = N_BLOCK_GROUPS * N_PER_BLOCK_GROUP + N_POOL_B
    assert world_size == expected, f"need {expected} ranks (got {world_size})"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    block_groups = [
        [g * N_PER_BLOCK_GROUP + k for k in range(N_PER_BLOCK_GROUP)] for g in range(N_BLOCK_GROUPS)
    ]
    # Each block group owns BLOCKS_PER_GROUP consecutive transformer blocks.
    block_owned_sites = [
        [
            s
            for tb in range(g * BLOCKS_PER_GROUP, (g + 1) * BLOCKS_PER_GROUP)
            for s in sites_for_block(tb)
        ]
        for g in range(N_BLOCK_GROUPS)
    ]
    pool_b_ranks = list(
        range(N_BLOCK_GROUPS * N_PER_BLOCK_GROUP, N_BLOCK_GROUPS * N_PER_BLOCK_GROUP + N_POOL_B)
    )

    world = build_block_ddp_world(
        block_groups=block_groups,
        block_owned_sites=block_owned_sites,
        pool_b_ranks=pool_b_ranks,
        batch_global=BATCH,
    )
    layout = BlockDDPLayout.from_world(world, rank)
    print(
        f"[rank{rank}] pool={layout.my_pool} "
        f"block_idx={layout.my_block_idx} within_block={layout.my_within_block_idx} "
        f"owned_sites={len(layout.my_owned_sites)} slice={layout.my_slice_idx}",
        flush=True,
    )

    cfg = Config(
        C_per_module={},
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        n_steps=WARMUP_STEPS + PROFILE_STEPS,
        ppgd_inner_steps=2,
    )

    torch.manual_seed(0)
    target = Tiny6BlockTransformer(VOCAB, D_MODEL, N_TRANSFORMER_BLOCKS, N_HEADS, D_MLP)
    all_sites = [s for sites in block_owned_sites for s in sites]
    c_per_site = {s: C for s in all_sites}
    cfg.C_per_module = c_per_site

    wrappers = install_components_for_block_ddp(target, layout, c_per_site)
    target = target.to(device)
    for w in wrappers.values():
        w.to(device)

    ci_fns = build_ci_fns_for_block_ddp(
        layout, wrappers, c_per_site, hidden=CI_HIDDEN, leaky_alpha=cfg.leaky_alpha
    )
    for f in ci_fns.values():
        f.to(device)

    optimizer: torch.optim.Optimizer | None = None
    if layout.my_pool == "a":
        params: list[nn.Parameter] = []
        for s in layout.my_owned_sites:
            params.extend([wrappers[s].V, wrappers[s].U])
        for f in ci_fns.values():
            params.extend(list(f.parameters()))
        optimizer = torch.optim.AdamW(params, lr=cfg.main_lr, weight_decay=0.0)

    ppgd: PersistentPGD | None = None
    if layout.my_pool == "b":
        torch.manual_seed(42 + (layout.my_slice_idx or 0))
        ppgd = PersistentPGD(wrappers, layout.world.batch_local_b, SEQ_LEN, device, cfg)

    data_rng = torch.Generator(device=device).manual_seed(0)

    def make_batch(step: int) -> Tensor:
        data_rng.manual_seed(step * 7919 + 17)
        return torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device, generator=data_rng)

    imp_p = anneal_p(0, cfg.n_steps, cfg.p_start, cfg.p_end)
    timer = StepTimer(rank)

    if rank == 0:
        target_params = sum(p.numel() for p in target.parameters())
        per_rank_wrapper_params = sum(p.numel() for w in wrappers.values() for p in (w.V, w.U))
        per_rank_ci_fn_params = sum(p.numel() for f in ci_fns.values() for p in f.parameters())
        total_ci_fn_global = per_rank_ci_fn_params * N_BLOCK_GROUPS  # one copy per group
        print(
            f"[rank0] target_params={target_params:,} "
            f"per_rank_wrappers={per_rank_wrapper_params:,} "
            f"per_rank_ci_fn={per_rank_ci_fn_params:,} "
            f"total_ci_fn_across_pool_a={total_ci_fn_global:,}",
            flush=True,
        )
        print(
            f"[rank0] batch={BATCH} seq={SEQ_LEN} d_model={D_MODEL} d_mlp={D_MLP} "
            f"n_blocks={N_TRANSFORMER_BLOCKS} blocks_per_group={BLOCKS_PER_GROUP} ci_hidden={CI_HIDDEN}",
            flush=True,
        )
        print(f"[rank0] warmup={WARMUP_STEPS} profile_steps={PROFILE_STEPS}", flush=True)

    if rank == 0:
        mem = torch.cuda.memory_allocated(device) / 1e9
        max_mem = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[rank0] before training: allocated={mem:.2f}GB max={max_mem:.2f}GB", flush=True)

    for step in range(cfg.n_steps):
        input_ids = make_batch(step)
        torch.manual_seed(100 + step * 1000 + rank)

        with timer.phase("STEP_TOTAL"):
            if layout.my_pool == "a":
                metrics = pool_a_step(
                    timer, layout, target, wrappers, ci_fns, optimizer, input_ids, cfg, imp_p
                )
            else:
                metrics = pool_b_step(timer, layout, target, wrappers, ppgd, input_ids, cfg, device)

        if rank == 0:
            mem = torch.cuda.memory_allocated(device) / 1e9
            max_mem = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"[rank0] step={step} allocated={mem:.2f}GB peak={max_mem:.2f}GB", flush=True)

        for k, v in metrics.items():
            if not math.isfinite(v):
                print(f"[rank{rank}] NaN/Inf at step {step}: {k}={v}", flush=True)
                dist.destroy_process_group()
                raise SystemExit(1)

    timer.report(warmup=WARMUP_STEPS)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
