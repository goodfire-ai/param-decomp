"""Stage 4: block-wise sharding in pool A + true single-site layerwise loss.

Refactored on top of `nano_param_decomp.two_pool.{World, TwoPoolLayout}` — no more
`RANK_ROLES` dict or ad-hoc rank/role helpers; the layout is the single source of
truth and orchestrates all cross-pool comm.

Target: Tiny6BlockTransformer — 6 transformer blocks, each with 7 decomposable matrices
(Q, K, V, O for attention; gate, up, down for SwiGLU MLP). 42 sites total.

Pool A (ranks 0-5): each rank owns ONE block — V/U + per-module CI fns + optimizer state
for that block's 7 sites only. Non-owned sites stay as the original `nn.Linear` (no
phantom V/U allocation).

Pool B (ranks 6-7): DP-2 with batch sliced in half. Both B ranks hold replicated V/U
for ALL 42 sites (the full-model PPGD needs every site in component mode at once).

Per-site layerwise loss: each pool-A rank loops over its 7 owned sites. For each
iteration, that single site is in component mode (uses its V/U + a sampled mask) and
the other sites pass through `W_target` (or, on this rank, the original `nn.Linear`).

Run:
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \\
        -m nano_param_decomp.two_pool_stage4
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

import math
import os
from typing import override

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nano_param_decomp.run import (
    ComponentLinear,
    Config,
    PersistentPGD,
    anneal_p,
    clear_wrapper_masks,
    kl_logits,
)
from nano_param_decomp.two_pool import (
    TwoPoolLayout,
    build_ci_fns_for_layout,
    build_world,
    install_components_for_layout,
)
from nano_param_decomp.two_pool_stage2 import ModuleCIFn, ci_forward

# --- Topology constants — used only at startup to build World ---

N_POOL_A = 6
N_POOL_B = 2
N_BLOCKS = 6  # == N_POOL_A; one block per pool-A rank


# --- Tiny 6-block transformer ---


class TinyAttention(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class TinySwiGLU(nn.Module):
    def __init__(self, d: int, d_mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d_mlp, bias=False)
        self.up_proj = nn.Linear(d, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, d_mlp: int) -> None:
        super().__init__()
        self.attn = TinyAttention(d, n_heads)
        self.mlp = TinySwiGLU(d, d_mlp)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(F.rms_norm(x, (x.shape[-1],)))
        x = x + self.mlp(F.rms_norm(x, (x.shape[-1],)))
        return x


class Tiny6BlockTransformer(nn.Module):
    def __init__(self, vocab: int, d: int, n_blocks: int, n_heads: int, d_mlp: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([TinyBlock(d, n_heads, d_mlp) for _ in range(n_blocks)])
        self.unembed = nn.Linear(d, vocab, bias=False)

    @override
    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = F.rms_norm(x, (x.shape[-1],))
        return self.unembed(x)


def sites_for_block(block_idx: int) -> list[str]:
    return [
        f"blocks.{block_idx}.attn.q_proj",
        f"blocks.{block_idx}.attn.k_proj",
        f"blocks.{block_idx}.attn.v_proj",
        f"blocks.{block_idx}.attn.o_proj",
        f"blocks.{block_idx}.mlp.gate_proj",
        f"blocks.{block_idx}.mlp.up_proj",
        f"blocks.{block_idx}.mlp.down_proj",
    ]


# --- Per-site layerwise loss ---


def single_site_layerwise_loss(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    owned_sites: tuple[str, ...],
    input_ids: Tensor,
    target_logits: Tensor,
    ci_lower_owned: dict[str, Tensor],
) -> Tensor:
    """For each owned site s: full target forward with s in component mode, others (which
    on this rank may not even have wrappers — they stay as nn.Linear) in target mode.
    """
    losses: list[Tensor] = []
    for s in owned_sites:
        ci = ci_lower_owned[s]
        u = torch.rand_like(ci)
        mask = ci + (1 - ci) * u
        delta_mask = torch.rand(*ci.shape[:-1], device=ci.device, dtype=ci.dtype)

        # Owned wrappers: route s, target-mode the others.
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

        pred = target_model(input_ids)
        losses.append(kl_logits(pred, target_logits))

    clear_wrapper_masks(wrappers)
    return torch.stack(losses).mean()


def faith_loss_owned(wrappers: dict[str, ComponentLinear], owned_sites: tuple[str, ...]) -> Tensor:
    sum_sq = torch.zeros((), device=wrappers[owned_sites[0]].V.device)
    numel = 0
    for s in owned_sites:
        delta = wrappers[s].weight_delta()
        sum_sq = sum_sq + delta.pow(2).sum()
        numel += delta.numel()
    return sum_sq / numel


def imp_loss_owned(ci_upper_owned: dict[str, Tensor], p: float, eps: float, beta: float) -> Tensor:
    total = torch.zeros((), device=next(iter(ci_upper_owned.values())).device)
    for v in ci_upper_owned.values():
        vals = (v + eps).pow(p)
        batch_seq_dims = tuple(range(vals.ndim - 1))
        sum_c = vals.sum(dim=batch_seq_dims)
        n = math.prod(vals.shape[:-1])
        mean_c = sum_c / n
        total = total + (mean_c + beta * mean_c * torch.log2(1 + sum_c)).sum()
    return total


# --- Pool A step ---


def pool_a_step(
    layout: TwoPoolLayout,
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ci_fns: dict[str, ModuleCIFn],
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    cfg: Config,
    imp_p: float,
) -> dict[str, float]:
    clear_wrapper_masks(wrappers)
    target_logits = target_model(input_ids)

    # CI fwd for OWNED sites only — uses cached last_input from the target-mode forward.
    acts_owned = {s: wrappers[s].last_input for s in layout.my_owned_sites}
    ci_lower_owned, ci_upper_owned = ci_forward(ci_fns, acts_owned)

    layout.send_owned_ci_to_pool_b(ci_lower_owned)

    loss_faith = faith_loss_owned(wrappers, layout.my_owned_sites)
    loss_imp = imp_loss_owned(ci_upper_owned, imp_p, cfg.imp_eps, cfg.imp_beta)
    loss_stoch = single_site_layerwise_loss(
        target_model, wrappers, layout.my_owned_sites, input_ids, target_logits, ci_lower_owned
    )
    total_home = (
        cfg.coeff_faith * loss_faith + cfg.coeff_imp * loss_imp + cfg.coeff_stoch * loss_stoch
    )

    v_grads, u_grads, ci_grads = layout.recv_grads_from_pool_b(wrappers, ci_lower_owned)

    optimizer.zero_grad(set_to_none=True)
    for s in layout.my_owned_sites:
        wrappers[s].V.grad = v_grads[s]
        wrappers[s].U.grad = u_grads[s]

    torch.autograd.backward(
        tensors=[total_home, *(ci_lower_owned[s] for s in layout.my_owned_sites)],
        grad_tensors=[None, *(ci_grads[s] for s in layout.my_owned_sites)],
    )
    optimizer.step()

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
    layout: TwoPoolLayout,
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ppgd: PersistentPGD,
    input_ids_full: Tensor,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    sl = layout.my_batch_slice()
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
    assert world_size == N_POOL_A + N_POOL_B
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    pool_a_ranks = list(range(N_POOL_A))
    pool_b_ranks = list(range(N_POOL_A, N_POOL_A + N_POOL_B))

    all_sites = [s for b in range(N_BLOCKS) for s in sites_for_block(b)]
    # Each pool-A rank owns exactly one block.
    site_owner = {
        s: pool_a_ranks[block_idx]
        for block_idx in range(N_BLOCKS)
        for s in sites_for_block(block_idx)
    }

    vocab, d, n_heads, d_mlp, batch_size, seq_len, C = 64, 32, 4, 64, 4, 8, 4

    world = build_world(
        pool_a_ranks=pool_a_ranks,
        pool_b_ranks=pool_b_ranks,
        all_sites=all_sites,
        site_owner=site_owner,
        batch_global=batch_size,
    )
    layout = TwoPoolLayout.from_world(world, rank)
    print(
        f"[rank{rank}] pool={layout.my_pool} owned={layout.my_owned_sites} slice={layout.my_slice_idx}",
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
    c_per_site = {s: C for s in all_sites}
    cfg.C_per_module = c_per_site

    wrappers = install_components_for_layout(target, layout, c_per_site)
    target = target.to(device)
    for w in wrappers.values():
        w.to(device)

    ci_fns = build_ci_fns_for_layout(
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
