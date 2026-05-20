"""Stage 5: in-block DDP on pool A coexisting with DP on pool B.

Topology: 3 blocks × 2 ranks per block = 6 pool A ranks; 2 pool B ranks DP-2.
Total 8 GPUs.

Each pool-A block group has 2 ranks holding replicated V/U + per-module CI fns +
optimizer state for the block's 7 sites. The 2 ranks share work via batch DP: each
sees half of the global batch and runs its layerwise iterations on its slice.

Three collectives per step (the derisk goal):
  1. Pool A in-block all-reduce (mean) on V/U + CI fn grads — combines per-slice
     layerwise contributions across the 2 block-mates.
  2. Pool B all-reduce (SUM, with 1/N scaling) on V/U grads — standard DP-2 PPGD.
  3. Cross-pool send/recv between block leaders and pool B ranks.

Run:
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \\
        -m nano_param_decomp.two_pool_stage5
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

import math
import os

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

# --- Topology constants ---

N_BLOCKS = 3
N_PER_BLOCK = 2
N_POOL_B = 2


# --- Per-site layerwise loss, but now on a local batch slice ---


def single_site_layerwise_loss_local(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    owned_sites: tuple[str, ...],
    input_ids_local: Tensor,
    target_logits_local: Tensor,
    ci_lower_local: dict[str, Tensor],
) -> Tensor:
    """Layerwise loss on the local batch slice.

    `ci_lower_local` is sliced ci for this rank's batch — shape [B_local_a, S, C] per site.
    Each owned site s is routed in turn (component mode), others in target mode.
    """
    losses: list[Tensor] = []
    for s in owned_sites:
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
    layout: BlockDDPLayout,
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ci_fns: dict[str, ModuleCIFn],
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    cfg: Config,
    imp_p: float,
) -> dict[str, float]:
    # Full-batch target + CI forward (replicated within block group — cheap relative to layerwise).
    clear_wrapper_masks(wrappers)
    target_logits = target_model(input_ids)
    acts_owned = {s: wrappers[s].last_input for s in layout.my_owned_sites}
    ci_lower_owned, ci_upper_owned = ci_forward(ci_fns, acts_owned)

    # Block leader sends full-batch CI (sliced per B rank) to pool B.
    layout.send_owned_ci_to_pool_b(ci_lower_owned)

    # Faith + imp on full batch — replicated identical on every block-group rank.
    loss_faith = faith_loss_owned(wrappers, layout.my_owned_sites)
    loss_imp = imp_loss_owned(ci_upper_owned, imp_p, cfg.imp_eps, cfg.imp_beta)

    # Layerwise on the LOCAL batch slice (this is where the in-block DP win lives).
    sl = layout.my_batch_slice_a()
    loss_stoch = single_site_layerwise_loss_local(
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

    # Recv B-contributed grads — block leader recvs from pool B, then broadcasts within block group.
    v_grads, u_grads, ci_grads = layout.recv_grads_from_pool_b(wrappers, ci_lower_owned)

    optimizer.zero_grad(set_to_none=True)
    for s in layout.my_owned_sites:
        wrappers[s].V.grad = v_grads[s]
        wrappers[s].U.grad = u_grads[s]

    # Combined home backward (seeds ci_lower with B's ci grads as extra roots).
    torch.autograd.backward(
        tensors=[total_home, *(ci_lower_owned[s] for s in layout.my_owned_sites)],
        grad_tensors=[None, *(ci_grads[s] for s in layout.my_owned_sites)],
    )

    # In-block all-reduce (mean) — combines per-slice layerwise contributions across the block group.
    params_to_reduce: list[nn.Parameter] = []
    for s in layout.my_owned_sites:
        params_to_reduce.extend([wrappers[s].V, wrappers[s].U])
    for f in ci_fns.values():
        params_to_reduce.extend(list(f.parameters()))
    layout.all_reduce_grads_in_block(params_to_reduce)

    optimizer.step()

    # Block leader sends updated V/U to pool B.
    v_owned = {s: wrappers[s].V for s in layout.my_owned_sites}
    u_owned = {s: wrappers[s].U for s in layout.my_owned_sites}
    layout.send_updated_weights_to_pool_b(v_owned, u_owned)

    return {
        "loss/faith": loss_faith.item(),
        "loss/imp": loss_imp.item(),
        "loss/stoch": loss_stoch.item(),
    }


# --- Pool B step (essentially unchanged from stage 4) ---


def pool_b_step(
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

    ci_recv = layout.recv_ci_from_owners(
        wrappers, seq_len=input_ids.shape[1], device=device, dtype=torch.float32
    )

    clear_wrapper_masks(wrappers)
    with torch.no_grad():
        target_logits = target_model(input_ids)

    ci_scratch = {n: v.detach().clone().requires_grad_(True) for n, v in ci_recv.items()}

    ppgd.warmup(target_model, wrappers, input_ids, target_logits, ci_scratch, lr=cfg.ppgd_lr)
    loss_ppgd = ppgd.recon_loss(target_model, wrappers, input_ids, target_logits, ci_scratch)
    total_ppgd = cfg.coeff_ppgd * loss_ppgd / layout.world.n_pool_b

    params: list[Tensor] = []
    for site in layout.world.all_sites:
        params.extend([wrappers[site].V, wrappers[site].U])
    ci_list = [ci_scratch[s] for s in layout.world.all_sites]
    grads = torch.autograd.grad(total_ppgd, params + ci_list)

    n_sites = len(layout.world.all_sites)
    v_grads = {s: grads[2 * i] for i, s in enumerate(layout.world.all_sites)}
    u_grads = {s: grads[2 * i + 1] for i, s in enumerate(layout.world.all_sites)}
    ci_grads = {s: grads[2 * n_sites + i] for i, s in enumerate(layout.world.all_sites)}

    for s in layout.world.all_sites:
        dist.all_reduce(v_grads[s], op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)
        dist.all_reduce(u_grads[s], op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)

    layout.send_pool_b_grads_to_owners(v_grads, u_grads, ci_grads)

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
    expected = N_BLOCKS * N_PER_BLOCK + N_POOL_B
    assert world_size == expected, f"stage 5 needs {expected} ranks (got {world_size})"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    block_groups = [[b * N_PER_BLOCK + k for k in range(N_PER_BLOCK)] for b in range(N_BLOCKS)]
    block_owned_sites = [sites_for_block(b) for b in range(N_BLOCKS)]
    pool_b_ranks = list(range(N_BLOCKS * N_PER_BLOCK, N_BLOCKS * N_PER_BLOCK + N_POOL_B))

    vocab, d, n_heads, d_mlp, batch_size, seq_len, C = 64, 32, 4, 64, 4, 8, 4

    world = build_block_ddp_world(
        block_groups=block_groups,
        block_owned_sites=block_owned_sites,
        pool_b_ranks=pool_b_ranks,
        batch_global=batch_size,
    )
    layout = BlockDDPLayout.from_world(world, rank)
    print(
        f"[rank{rank}] pool={layout.my_pool} block_idx={layout.my_block_idx} "
        f"within_block={layout.my_within_block_idx} owned={len(layout.my_owned_sites)} "
        f"slice={layout.my_slice_idx}",
        flush=True,
    )

    cfg = Config(
        C_per_module={},
        batch_size=batch_size,
        seq_len=seq_len,
        n_steps=30,
        ppgd_inner_steps=2,
    )

    torch.manual_seed(0)
    target = Tiny6BlockTransformer(vocab, d, N_BLOCKS, n_heads, d_mlp)
    all_sites = [s for sites in block_owned_sites for s in sites]
    c_per_site = {s: C for s in all_sites}
    cfg.C_per_module = c_per_site

    wrappers = install_components_for_block_ddp(target, layout, c_per_site)
    target = target.to(device)
    for w in wrappers.values():
        w.to(device)

    ci_fns = build_ci_fns_for_block_ddp(
        layout, wrappers, c_per_site, hidden=32, leaky_alpha=cfg.leaky_alpha
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
        ppgd = PersistentPGD(wrappers, layout.world.batch_local_b, seq_len, device, cfg)

    data_rng = torch.Generator(device=device).manual_seed(0)

    def make_batch(step: int) -> Tensor:
        data_rng.manual_seed(step * 7919 + 17)
        return torch.randint(0, vocab, (batch_size, seq_len), device=device, generator=data_rng)

    imp_p = anneal_p(0, cfg.n_steps, cfg.p_start, cfg.p_end)

    print(f"[rank{rank}] starting ({cfg.n_steps} steps)", flush=True)
    for step in range(cfg.n_steps):
        input_ids = make_batch(step)
        torch.manual_seed(100 + step * 1000 + rank)

        if layout.my_pool == "a":
            metrics = pool_a_step(
                layout, target, wrappers, ci_fns, optimizer, input_ids, cfg, imp_p
            )
        else:
            metrics = pool_b_step(layout, target, wrappers, ppgd, input_ids, cfg, device)

        if step % 3 == 0:
            msg = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(f"[rank{rank}] step={step} {msg}", flush=True)

        for k, v in metrics.items():
            if not math.isfinite(v):
                print(f"[rank{rank}] NaN/Inf at step {step}: {k}={v}", flush=True)
                dist.destroy_process_group()
                raise SystemExit(1)

    print(f"[rank{rank}] done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
