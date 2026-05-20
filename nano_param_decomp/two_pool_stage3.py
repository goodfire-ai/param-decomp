"""Stage 3: asymmetric 6+2 pools with DP inside pool B.

Pool A (ranks 0-5, 6 GPUs): fully replicated. V/U + per-module CI fns + optimizer all on
every A rank. Each A rank does identical compute.

Pool B (ranks 6-7, 2 GPUs): DP-2. Each B rank holds replicated V/U, no CI fn, persistent
PGD sources scoped to its batch slice. Each B rank processes B_global/2 of the batch.

Cross-pool comm:
  - A leader (rank 0) → each B rank: that B rank's slice of ci_lower per site
  - Each B rank → A leader: its slice of ci_scratch.grad per site, plus its (scaled)
    V/U grads
  - A leader broadcasts pool-B-contributed V/U grads + concatenated ci_scratch.grad
    to all other pool A ranks via POOL_A_GROUP

Grad scaling: each B rank scales its loss by 1/N_B before backward. Then:
  - V/U grads: all-reduced (SUM) within pool B → equals full-batch V/U grad.
  - ci_scratch.grad: concatenated across B ranks in batch dim → equals full-batch ci grad.

Run:
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \\
        -m nano_param_decomp.two_pool_stage3
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

import math
import os
from typing import Any

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
    faithfulness_loss,
    importance_minimality_loss,
    install_components,
    stochastic_recon_loss,
)
from nano_param_decomp.two_pool_stage1 import TinyLM
from nano_param_decomp.two_pool_stage2 import ModuleCIFn, build_ci_fns, ci_forward

# --- Topology ---

N_POOL_A = 6
N_POOL_B = 2
POOL_A_RANKS = list(range(N_POOL_A))
POOL_B_RANKS = list(range(N_POOL_A, N_POOL_A + N_POOL_B))
A_LEADER = POOL_A_RANKS[0]


def role_for(rank: int) -> dict[str, Any]:
    if rank in POOL_A_RANKS:
        return {"pool": "a", "is_leader": rank == A_LEADER}
    if rank in POOL_B_RANKS:
        return {
            "pool": "b",
            "is_leader": rank == POOL_B_RANKS[0],
            "slice_idx": POOL_B_RANKS.index(rank),
        }
    raise ValueError(f"rank {rank} not in any pool")


# --- Comm helpers ---


def _send_tensor(t: Tensor, dst: int) -> None:
    dist.send(t.contiguous(), dst=dst)


def _recv_tensor_like(template: Tensor, src: int) -> Tensor:
    buf = torch.empty_like(template)
    dist.recv(buf, src=src)
    return buf


def _broadcast_in_pool_a(t: Tensor, pool_a_group: dist.ProcessGroup) -> None:
    """In-place broadcast from A_LEADER to all pool A ranks."""
    dist.broadcast(t, src=A_LEADER, group=pool_a_group)


# --- Pool A step ---


def pool_a_step(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    ci_fns: dict[str, ModuleCIFn],
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    cfg: Config,
    imp_p: float,
    site_names: list[str],
    role: dict[str, Any],
    device: torch.device,
    pool_a_group: dist.ProcessGroup,
) -> dict[str, float]:
    B_global = input_ids.shape[0]
    B_local_b = B_global // N_POOL_B
    assert B_global % N_POOL_B == 0, f"B_global={B_global} not divisible by N_POOL_B={N_POOL_B}"

    # Forward (identical on every pool-A rank).
    clear_wrapper_masks(wrappers)
    target_logits = target_model(input_ids)
    acts = {n: wrappers[n].last_input for n in site_names}
    ci_lower, ci_upper = ci_forward(ci_fns, acts)

    # A leader sends per-B-rank slices of ci_lower per site. Other A ranks idle on send.
    if role["is_leader"]:
        for slice_idx in range(N_POOL_B):
            b_dst = POOL_B_RANKS[slice_idx]
            sl = slice(slice_idx * B_local_b, (slice_idx + 1) * B_local_b)
            for n in site_names:
                _send_tensor(ci_lower[n][sl].detach(), dst=b_dst)

    # Home loss forwards (every A rank — they're identical).
    loss_faith = faithfulness_loss(wrappers)
    loss_imp = importance_minimality_loss(ci_upper, imp_p, cfg.imp_eps, cfg.imp_beta, world_size=1)
    loss_stoch = stochastic_recon_loss(target_model, wrappers, input_ids, target_logits, ci_lower)
    total_home = (
        cfg.coeff_faith * loss_faith + cfg.coeff_imp * loss_imp + cfg.coeff_stoch * loss_stoch
    )

    # Receive grads from pool B (A leader only) then broadcast within pool A.
    v_grads_full: dict[str, Tensor] = {}
    u_grads_full: dict[str, Tensor] = {}
    ci_grads_full: dict[str, Tensor] = {}

    if role["is_leader"]:
        # V/U grads (already SUM-reduced inside pool B) — A leader recvs from B leader.
        for n in site_names:
            v_grads_full[n] = _recv_tensor_like(wrappers[n].V, src=POOL_B_RANKS[0])
        for n in site_names:
            u_grads_full[n] = _recv_tensor_like(wrappers[n].U, src=POOL_B_RANKS[0])
        # ci_scratch.grad slices from each B rank, concat in batch dim.
        for n in site_names:
            slices: list[Tensor] = []
            for slice_idx in range(N_POOL_B):
                template = torch.empty(
                    B_local_b, ci_lower[n].shape[1], ci_lower[n].shape[2], device=device
                )
                slices.append(_recv_tensor_like(template, src=POOL_B_RANKS[slice_idx]))
            ci_grads_full[n] = torch.cat(slices, dim=0)
    else:
        # Allocate empty buffers to participate in the broadcast.
        for n in site_names:
            v_grads_full[n] = torch.empty_like(wrappers[n].V)
            u_grads_full[n] = torch.empty_like(wrappers[n].U)
            ci_grads_full[n] = torch.empty_like(ci_lower[n])

    # Broadcast pool-B-contributed grads to all pool A ranks.
    for n in site_names:
        _broadcast_in_pool_a(v_grads_full[n], pool_a_group)
        _broadcast_in_pool_a(u_grads_full[n], pool_a_group)
        _broadcast_in_pool_a(ci_grads_full[n], pool_a_group)

    # Seed V/U .grad with B's contribution; the home backward accumulates onto them.
    optimizer.zero_grad(set_to_none=True)
    for n in site_names:
        wrappers[n].V.grad = v_grads_full[n]
        wrappers[n].U.grad = u_grads_full[n]

    # Combined home backward with ci_lower as extra roots seeded by B's ci grads.
    torch.autograd.backward(
        tensors=[total_home, *(ci_lower[n] for n in site_names)],
        grad_tensors=[None, *(ci_grads_full[n] for n in site_names)],
    )

    optimizer.step()

    # A leader sends updated V/U to all B ranks. (Other A ranks have the same values; no
    # internal pool-A sync needed because they all ran identical compute.)
    if role["is_leader"]:
        for n in site_names:
            for b_dst in POOL_B_RANKS:
                _send_tensor(wrappers[n].V.detach(), dst=b_dst)
            for b_dst in POOL_B_RANKS:
                _send_tensor(wrappers[n].U.detach(), dst=b_dst)

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
    site_names: list[str],
    role: dict[str, Any],
    device: torch.device,
    pool_b_group: dist.ProcessGroup,
) -> dict[str, float]:
    B_global = input_ids_full.shape[0]
    B_local = B_global // N_POOL_B
    sl = slice(role["slice_idx"] * B_local, (role["slice_idx"] + 1) * B_local)
    input_ids = input_ids_full[sl]

    # Receive ci_lower slice (for our batch slice only) from A leader, per site.
    ci_recv: dict[str, Tensor] = {}
    for n in site_names:
        template = torch.empty(B_local, input_ids.shape[1], wrappers[n].C, device=device)
        ci_recv[n] = _recv_tensor_like(template, src=A_LEADER)

    # Target forward on our slice — frozen target so no_grad is fine.
    clear_wrapper_masks(wrappers)
    with torch.no_grad():
        target_logits = target_model(input_ids)

    # Re-leaf for grad.
    ci_scratch = {n: v.detach().clone().requires_grad_(True) for n, v in ci_recv.items()}

    # PPGD warmup + recon loss against ci_scratch.
    ppgd.warmup(target_model, wrappers, input_ids, target_logits, ci_scratch, lr=cfg.ppgd_lr)
    loss_ppgd = ppgd.recon_loss(target_model, wrappers, input_ids, target_logits, ci_scratch)

    # Scale by 1/N_POOL_B so that SUM-reduce of V/U grads across pool B == full-batch
    # gradient (kl_logits is mean over the local slice; summing N scaled means gives the
    # full-batch mean).
    total_ppgd = cfg.coeff_ppgd * loss_ppgd / N_POOL_B

    params: list[Tensor] = []
    for n in site_names:
        params.extend([wrappers[n].V, wrappers[n].U])
    ci_scratch_list = [ci_scratch[n] for n in site_names]
    grads = torch.autograd.grad(total_ppgd, params + ci_scratch_list)

    v_grads = {n: grads[2 * i] for i, n in enumerate(site_names)}
    u_grads = {n: grads[2 * i + 1] for i, n in enumerate(site_names)}
    ci_grads = {n: grads[2 * len(site_names) + i] for i, n in enumerate(site_names)}

    # All-reduce V/U grads within pool B (SUM — they're already scaled by 1/N).
    for n in site_names:
        dist.all_reduce(v_grads[n], op=dist.ReduceOp.SUM, group=pool_b_group)
        dist.all_reduce(u_grads[n], op=dist.ReduceOp.SUM, group=pool_b_group)

    # B leader sends the all-reduced V/U grads to A.
    if role["is_leader"]:
        for n in site_names:
            _send_tensor(v_grads[n], dst=A_LEADER)
        for n in site_names:
            _send_tensor(u_grads[n], dst=A_LEADER)

    # Each B rank sends its ci_scratch.grad slice to A leader (A concatenates in batch dim).
    for n in site_names:
        _send_tensor(ci_grads[n], dst=A_LEADER)

    # Recv updated V/U from A.
    for n in site_names:
        v_new = _recv_tensor_like(wrappers[n].V, src=A_LEADER)
        with torch.no_grad():
            wrappers[n].V.copy_(v_new)
    for n in site_names:
        u_new = _recv_tensor_like(wrappers[n].U, src=A_LEADER)
        with torch.no_grad():
            wrappers[n].U.copy_(u_new)

    # Report the unscaled loss for logging.
    return {"loss/ppgd": loss_ppgd.item()}


# --- Main ---


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == N_POOL_A + N_POOL_B, (
        f"stage 3 needs exactly {N_POOL_A + N_POOL_B} ranks (got {world_size})"
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    role = role_for(rank)
    print(f"[rank{rank}] role={role} device={device}", flush=True)

    pool_a_group = dist.new_group(ranks=POOL_A_RANKS)
    pool_b_group = dist.new_group(ranks=POOL_B_RANKS)

    # Bigger TinyLM so the asymmetry is visible (3 layers → 7 sites; still tiny).
    vocab, d, n_layers, batch_size, seq_len, C = 32, 16, 3, 4, 8, 4

    cfg = Config(
        C_per_module={},
        batch_size=batch_size,
        seq_len=seq_len,
        n_steps=50,
        ppgd_inner_steps=2,
    )

    torch.manual_seed(0)
    target = TinyLM(vocab, d, n_layers)
    C_per_module = {f"layers.{i}.fc{j}": C for i in range(n_layers) for j in (1, 2)} | {
        "unembed": C
    }
    cfg.C_per_module = C_per_module
    site_names = sorted(C_per_module)

    wrappers = install_components(target, C_per_module)
    target = target.to(device)
    for w in wrappers.values():
        w.to(device)

    ci_fns: dict[str, ModuleCIFn] = {}
    if role["pool"] == "a":
        d_in_per_module = {n: int(wrappers[n].W_target.shape[1]) for n in site_names}
        ci_fns = build_ci_fns(d_in_per_module, C_per_module, hidden=32, leaky_alpha=cfg.leaky_alpha)
        for n in site_names:
            ci_fns[n].to(device)

    optimizer: torch.optim.Optimizer | None = None
    if role["pool"] == "a":
        params: list[nn.Parameter] = []
        for w in wrappers.values():
            params.extend([w.V, w.U])
        for f in ci_fns.values():
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

    sampling_seed_base = 100
    imp_p = anneal_p(0, cfg.n_steps, cfg.p_start, cfg.p_end)

    print(f"[rank{rank}] starting training ({cfg.n_steps} steps)", flush=True)
    for step in range(cfg.n_steps):
        input_ids = make_batch(step)
        torch.manual_seed(sampling_seed_base + step)

        if role["pool"] == "a":
            metrics = pool_a_step(
                target,
                wrappers,
                ci_fns,
                optimizer,
                input_ids,
                cfg,
                imp_p,
                site_names,
                role,
                device,
                pool_a_group,
            )
        else:
            metrics = pool_b_step(
                target,
                wrappers,
                ppgd,
                input_ids,
                cfg,
                site_names,
                role,
                device,
                pool_b_group,
            )

        if step % 5 == 0 and (role.get("is_leader") or role["pool"] == "b"):
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
