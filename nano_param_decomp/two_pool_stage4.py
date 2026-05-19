"""Stage 4: block-wise sharding in pool A + true single-site layerwise loss.

Target: Tiny6BlockTransformer — 6 transformer blocks, each with 7 decomposable matrices
(Q, K, V, O for attention; gate, up, down for SwiGLU MLP). 42 sites total.

Pool A (ranks 0-5, 6 GPUs): each rank owns ONE block (its 7 matrices' V/U + per-module CI
fns + optimizer state). Other ranks' wrappers exist locally but stay in target mode and
their V/U is never used in this rank's forwards (still allocated for now — site-skipping
the wrapper allocation is a memory optimization for later).

Pool B (ranks 6-7, 2 GPUs): DP-2 with batch sliced in half. Both B ranks hold replicated
V/U for ALL 42 sites (the full-model PPGD needs every site in component mode at once).

Per-site layerwise loss: each pool-A rank loops over its 7 owned sites. For each iteration,
that single site is in component mode (uses its V/U + a sampled mask) and the other 41
sites are in target mode (pass through W_target). Full target forward+backward per
iteration. M_local = 7 iterations per A rank per step.

Run:
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \\
        -m nano_param_decomp.two_pool_stage4
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

import math
import os
from typing import Any, override

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
    install_components,
    kl_logits,
    set_wrapper_masks,
)
from nano_param_decomp.two_pool_stage2 import ModuleCIFn, build_ci_fns, ci_forward

# --- Topology ---

N_POOL_A = 6
N_POOL_B = 2
N_BLOCKS = 6   # one block per pool-A rank
POOL_A_RANKS = list(range(N_POOL_A))
POOL_B_RANKS = list(range(N_POOL_A, N_POOL_A + N_POOL_B))
A_LEADER = POOL_A_RANKS[0]


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
    def __init__(
        self, vocab: int, d: int, n_blocks: int, n_heads: int, d_mlp: int
    ) -> None:
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


# Site paths per block, in deterministic order.
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


def all_site_paths() -> list[str]:
    return [s for b in range(N_BLOCKS) for s in sites_for_block(b)]


def block_idx_for_pool_a_rank(rank: int) -> int:
    assert rank in POOL_A_RANKS, rank
    return rank - POOL_A_RANKS[0]


def owning_a_rank_for_site(site: str) -> int:
    """sites are blocks.{i}.something — owning A rank is POOL_A_RANKS[i]."""
    block_idx = int(site.split(".")[1])
    return POOL_A_RANKS[block_idx]


def role_for(rank: int) -> dict[str, Any]:
    if rank in POOL_A_RANKS:
        return {
            "pool": "a",
            "is_leader": rank == A_LEADER,
            "owned_sites": sites_for_block(block_idx_for_pool_a_rank(rank)),
        }
    if rank in POOL_B_RANKS:
        return {
            "pool": "b",
            "is_leader": rank == POOL_B_RANKS[0],
            "slice_idx": POOL_B_RANKS.index(rank),
        }
    raise ValueError(f"rank {rank} not in any pool")


# --- Comm helpers ---

def _send(t: Tensor, dst: int) -> None:
    dist.send(t.contiguous(), dst=dst)


def _recv_like(template: Tensor, src: int) -> Tensor:
    buf = torch.empty_like(template)
    dist.recv(buf, src=src)
    return buf


# --- Per-site layerwise stoch loss ---

def single_site_layerwise_loss(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    owned_sites: list[str],
    input_ids: Tensor,
    target_logits: Tensor,
    ci_lower_owned: dict[str, Tensor],
) -> Tensor:
    """For each owned site s: full target forward with s in component mode (mask sampled
    from ci_lower[s]) and every other site in target mode. KL vs target_logits. Mean across
    iterations.

    Equivalent to "one module masked at a time" — true layerwise SPD loss.
    """
    losses: list[Tensor] = []
    for s in owned_sites:
        ci = ci_lower_owned[s]
        u = torch.rand_like(ci)
        mask = ci + (1 - ci) * u
        delta_mask = torch.rand(*ci.shape[:-1], device=ci.device, dtype=ci.dtype)

        # All wrappers default to target mode; the one being routed flips to component.
        for name, w in wrappers.items():
            if name == s:
                w.mode = "component"
                w.mask = mask
                w.delta_mask = delta_mask
                w.routing_mask = None  # route everywhere — single-site layerwise
            else:
                w.mode = "target"
                w.mask = None
                w.delta_mask = None
                w.routing_mask = None

        try:
            pred = target_model(input_ids)
            losses.append(kl_logits(pred, target_logits))
        finally:
            pass

    clear_wrapper_masks(wrappers)
    return torch.stack(losses).mean()


# --- Faith and imp losses, computed over owned sites only ---

def faith_loss_owned(wrappers: dict[str, ComponentLinear], owned_sites: list[str]) -> Tensor:
    """Sum-of-squared weight-delta error over owned sites, normalized by their total numel.

    Each pool-A rank's partial faith only depends on its owned V/U via wrappers[s].weight_delta()
    — non-owned wrappers' V/U is not referenced.
    """
    sum_sq = torch.zeros((), device=wrappers[owned_sites[0]].V.device)
    numel = 0
    for s in owned_sites:
        delta = wrappers[s].weight_delta()
        sum_sq = sum_sq + delta.pow(2).sum()
        numel += delta.numel()
    return sum_sq / numel


def imp_loss_owned(
    ci_upper_owned: dict[str, Tensor], p: float, eps: float, beta: float
) -> Tensor:
    """Importance minimality summed over owned sites. Each term touches only its own
    site's CI fn output → backward only touches the owning rank's CI fn weights.
    """
    total = torch.zeros((), device=next(iter(ci_upper_owned.values())).device)
    for v in ci_upper_owned.values():
        vals = (v + eps).pow(p)
        batch_seq_dims = tuple(range(vals.ndim - 1))
        sum_c = vals.sum(dim=batch_seq_dims)
        n = math.prod(vals.shape[:-1])
        mean_c = sum_c / n
        # world_size=1 — this rank's contribution standalone; do NOT all-reduce across pool A.
        total = total + (mean_c + beta * mean_c * torch.log2(1 + sum_c)).sum()
    return total


# --- Pool A step ---

def pool_a_step(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ci_fns_owned: dict[str, ModuleCIFn],
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    cfg: Config,
    imp_p: float,
    all_sites: list[str],
    role: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    B_global = input_ids.shape[0]
    B_local_b = B_global // N_POOL_B
    assert B_global % N_POOL_B == 0

    owned_sites: list[str] = role["owned_sites"]
    my_rank = dist.get_rank()

    # --- Forward (every A rank — identical compute through frozen target) ---
    clear_wrapper_masks(wrappers)
    target_logits = target_model(input_ids)

    # CI fwd for OWNED sites only — uses cached last_input from the target-mode forward above.
    acts_owned = {s: wrappers[s].last_input for s in owned_sites}
    ci_lower_owned, ci_upper_owned = ci_forward(ci_fns_owned, acts_owned)

    # --- Phase A: A → B sends ci_lower slices, per site, per B-rank slice ---
    for site in all_sites:
        owner = owning_a_rank_for_site(site)
        if my_rank == owner:
            for slice_idx in range(N_POOL_B):
                sl = slice(slice_idx * B_local_b, (slice_idx + 1) * B_local_b)
                _send(ci_lower_owned[site][sl].detach(), dst=POOL_B_RANKS[slice_idx])

    # --- Home loss forwards (partial — owned sites only) ---
    loss_faith = faith_loss_owned(wrappers, owned_sites)
    loss_imp = imp_loss_owned(ci_upper_owned, imp_p, cfg.imp_eps, cfg.imp_beta)
    loss_stoch = single_site_layerwise_loss(
        target_model, wrappers, owned_sites, input_ids, target_logits, ci_lower_owned
    )
    total_home = (
        cfg.coeff_faith * loss_faith
        + cfg.coeff_imp * loss_imp
        + cfg.coeff_stoch * loss_stoch
    )

    # --- Phase B: B → A V/U grads (B leader → owning A rank) ---
    v_grads_owned: dict[str, Tensor] = {}
    u_grads_owned: dict[str, Tensor] = {}
    for site in all_sites:
        owner = owning_a_rank_for_site(site)
        if my_rank == owner:
            v_grads_owned[site] = _recv_like(wrappers[site].V, src=POOL_B_RANKS[0])
            u_grads_owned[site] = _recv_like(wrappers[site].U, src=POOL_B_RANKS[0])

    # --- Phase C: B → A ci_scratch.grad slices (per B rank → owning A rank, concat in batch) ---
    ci_grads_owned: dict[str, Tensor] = {}
    for site in all_sites:
        owner = owning_a_rank_for_site(site)
        if my_rank == owner:
            slices: list[Tensor] = []
            B_S_C = (B_local_b, ci_lower_owned[site].shape[1], ci_lower_owned[site].shape[2])
            for slice_idx in range(N_POOL_B):
                tmpl = torch.empty(B_S_C, device=device, dtype=ci_lower_owned[site].dtype)
                slices.append(_recv_like(tmpl, src=POOL_B_RANKS[slice_idx]))
            ci_grads_owned[site] = torch.cat(slices, dim=0)

    # --- Seed V/U .grad with B's contribution + single combined backward ---
    optimizer.zero_grad(set_to_none=True)
    for s in owned_sites:
        wrappers[s].V.grad = v_grads_owned[s]
        wrappers[s].U.grad = u_grads_owned[s]

    torch.autograd.backward(
        tensors=[total_home, *(ci_lower_owned[s] for s in owned_sites)],
        grad_tensors=[None, *(ci_grads_owned[s] for s in owned_sites)],
    )

    optimizer.step()

    # --- Phase D: A → B updated V/U (owner → each B rank) ---
    for site in all_sites:
        owner = owning_a_rank_for_site(site)
        if my_rank == owner:
            for b_dst in POOL_B_RANKS:
                _send(wrappers[site].V.detach(), dst=b_dst)
                _send(wrappers[site].U.detach(), dst=b_dst)

    return {
        "loss/faith": loss_faith.item(),
        "loss/imp": loss_imp.item(),
        "loss/stoch": loss_stoch.item(),
    }


# --- Pool B step ---

def pool_b_step(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ppgd: PersistentPGD,
    input_ids_full: Tensor,
    cfg: Config,
    all_sites: list[str],
    role: dict[str, Any],
    device: torch.device,
    pool_b_group,
) -> dict[str, float]:
    B_global = input_ids_full.shape[0]
    B_local = B_global // N_POOL_B
    slice_idx = role["slice_idx"]
    sl = slice(slice_idx * B_local, (slice_idx + 1) * B_local)
    input_ids = input_ids_full[sl]

    # --- Phase A: recv ci_lower per site from owning A rank ---
    ci_recv: dict[str, Tensor] = {}
    for site in all_sites:
        owner = owning_a_rank_for_site(site)
        tmpl = torch.empty(B_local, input_ids.shape[1], wrappers[site].C, device=device)
        ci_recv[site] = _recv_like(tmpl, src=owner)

    # --- Target forward on local slice (frozen) ---
    clear_wrapper_masks(wrappers)
    with torch.no_grad():
        target_logits = target_model(input_ids)

    ci_scratch = {n: v.detach().clone().requires_grad_(True) for n, v in ci_recv.items()}

    # --- PPGD on full site set, local batch slice ---
    ppgd.warmup(target_model, wrappers, input_ids, target_logits, ci_scratch, lr=cfg.ppgd_lr)
    loss_ppgd = ppgd.recon_loss(target_model, wrappers, input_ids, target_logits, ci_scratch)
    total_ppgd = cfg.coeff_ppgd * loss_ppgd / N_POOL_B

    params: list[Tensor] = []
    for site in all_sites:
        params.extend([wrappers[site].V, wrappers[site].U])
    ci_list = [ci_scratch[s] for s in all_sites]
    grads = torch.autograd.grad(total_ppgd, params + ci_list)

    v_grads = {s: grads[2 * i] for i, s in enumerate(all_sites)}
    u_grads = {s: grads[2 * i + 1] for i, s in enumerate(all_sites)}
    ci_grads = {s: grads[2 * len(all_sites) + i] for i, s in enumerate(all_sites)}

    # All-reduce V/U grads within pool B (SUM — already scaled by 1/N_POOL_B).
    for s in all_sites:
        dist.all_reduce(v_grads[s], op=dist.ReduceOp.SUM, group=pool_b_group)
        dist.all_reduce(u_grads[s], op=dist.ReduceOp.SUM, group=pool_b_group)

    # --- Phase B: B leader sends V/U grads to owning A rank ---
    if role["is_leader"]:
        for site in all_sites:
            owner = owning_a_rank_for_site(site)
            _send(v_grads[site], dst=owner)
            _send(u_grads[site], dst=owner)

    # --- Phase C: each B rank sends its ci_scratch.grad slice to owning A rank ---
    for site in all_sites:
        owner = owning_a_rank_for_site(site)
        _send(ci_grads[site], dst=owner)

    # --- Phase D: recv updated V/U from owning A rank ---
    for site in all_sites:
        owner = owning_a_rank_for_site(site)
        v_new = _recv_like(wrappers[site].V, src=owner)
        u_new = _recv_like(wrappers[site].U, src=owner)
        with torch.no_grad():
            wrappers[site].V.copy_(v_new)
            wrappers[site].U.copy_(u_new)

    return {"loss/ppgd": loss_ppgd.item()}


# --- Main ---

def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == N_POOL_A + N_POOL_B, (
        f"stage 4 needs exactly {N_POOL_A + N_POOL_B} ranks (got {world_size})"
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    role = role_for(rank)
    print(f"[rank{rank}] role={role} device={device}", flush=True)

    pool_b_group = dist.new_group(ranks=POOL_B_RANKS)

    # Tiny6BlockTransformer — small enough to be fast, big enough that block-wise sharding is meaningful.
    vocab, d, n_heads, d_mlp, batch_size, seq_len, C = 64, 32, 4, 64, 4, 8, 4
    cfg = Config(
        C_per_module={},
        batch_size=batch_size,
        seq_len=seq_len,
        n_steps=30,  # 7-forwards-per-rank-per-step makes this slower; keep modest.
        ppgd_inner_steps=2,
    )

    torch.manual_seed(0)
    target = Tiny6BlockTransformer(vocab, d, N_BLOCKS, n_heads, d_mlp)
    all_sites = all_site_paths()
    C_per_module = {s: C for s in all_sites}
    cfg.C_per_module = C_per_module

    wrappers = install_components(target, C_per_module)
    target = target.to(device)
    for w in wrappers.values():
        w.to(device)

    # CI fns: pool A ranks build only their OWNED sites. Pool B doesn't build any.
    ci_fns_owned: dict[str, ModuleCIFn] = {}
    if role["pool"] == "a":
        d_in_per_module = {s: int(wrappers[s].W_target.shape[1]) for s in role["owned_sites"]}
        owned_c = {s: C for s in role["owned_sites"]}
        ci_fns_owned = build_ci_fns(d_in_per_module, owned_c, hidden=32, leaky_alpha=cfg.leaky_alpha)
        for f in ci_fns_owned.values():
            f.to(device)

    optimizer: torch.optim.Optimizer | None = None
    if role["pool"] == "a":
        params: list[nn.Parameter] = []
        for s in role["owned_sites"]:
            params.extend([wrappers[s].V, wrappers[s].U])
        for f in ci_fns_owned.values():
            params.extend(list(f.parameters()))
        optimizer = torch.optim.AdamW(params, lr=cfg.main_lr, weight_decay=0.0)

    ppgd: PersistentPGD | None = None
    if role["pool"] == "b":
        torch.manual_seed(42 + role["slice_idx"])
        B_local = batch_size // N_POOL_B
        ppgd = PersistentPGD(wrappers, B_local, seq_len, device, cfg)

    data_rng = torch.Generator(device=device).manual_seed(0)

    def make_batch(step: int) -> Tensor:
        data_rng.manual_seed(step * 7919 + 17)
        return torch.randint(0, vocab, (batch_size, seq_len), device=device, generator=data_rng)

    imp_p = anneal_p(0, cfg.n_steps, cfg.p_start, cfg.p_end)

    print(f"[rank{rank}] starting ({cfg.n_steps} steps)", flush=True)
    for step in range(cfg.n_steps):
        input_ids = make_batch(step)
        torch.manual_seed(100 + step * 1000 + rank)

        if role["pool"] == "a":
            metrics = pool_a_step(
                target, wrappers, ci_fns_owned, optimizer, input_ids, cfg, imp_p,
                all_sites, role, device,
            )
        else:
            metrics = pool_b_step(
                target, wrappers, ppgd, input_ids, cfg, all_sites, role, device, pool_b_group,
            )

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
