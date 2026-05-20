"""2-pool benchmark using param_decomp's ComponentModel + the layout module.

Same model dimensions as `vanilla.py` for direct apples-to-apples comparison.

Topology: 3 block groups × 2 ranks (in-block DDP-2) + 2 pool B ranks (DP-2) = 8 GPUs.
Each pool A rank owns 2 transformer blocks (14 sites). Pool B ranks hold all 42 sites
in component mode and run the full-model adversarial recon.

Each pool A rank constructs a ComponentModel with `module_path_info` restricted to its
owned sites (via `param_decomp.two_pool.install.build_pool_a_module_path_info`). Each
pool B rank constructs a ComponentModel covering every site
(`build_pool_b_module_path_info`).

Run on 8 GPUs single-node:
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \\
        -m param_decomp.scripts.two_pool_benchmark.two_pool
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false

import math
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from param_decomp.configs import (
    LayerwiseCiConfig,
    PerBatchPerPositionScope,
    PersistentPGDReconLossConfig,
    ScheduleConfig,
    SignPGDConfig,
)
from param_decomp.models.batch_and_loss_fns import recon_loss_kl, run_batch_passthrough
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import make_mask_infos
from param_decomp.persistent_pgd import PersistentPGDState
from param_decomp.scripts.two_pool_benchmark._tiny_model import TinyTransformer, sites_for_block
from param_decomp.two_pool import (
    BlockDDPLayout,
    build_block_ddp_world,
    build_pool_a_module_path_info,
    build_pool_b_module_path_info,
)

# Identical dims to vanilla.py.
VOCAB = 8192
D_MODEL = 768
N_HEADS = 12
D_MLP = 3072
N_TRANSFORMER_BLOCKS = 6
BATCH = 8
SEQ_LEN = 64
C = 32
CI_HIDDEN = 1024

# Topology
N_BLOCK_GROUPS = 3
N_PER_BLOCK_GROUP = 2
N_POOL_B = 2
BLOCKS_PER_GROUP = N_TRANSFORMER_BLOCKS // N_BLOCK_GROUPS  # 2

WARMUP_STEPS = 2
PROFILE_STEPS = 4


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
            self.times[name].append((time.perf_counter() - start) * 1000.0)

    def report(self, warmup: int) -> None:
        print(f"\n[two_pool rank{self.rank}] phase wall-clock (skipping first {warmup}):", flush=True)
        total = 0.0
        for name, vals in self.times.items():
            vals = vals[warmup:]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            total += avg
            print(
                f"  {name:30s} avg={avg:9.2f}ms  min={min(vals):9.2f}ms  max={max(vals):9.2f}ms",
                flush=True,
            )
        print(f"  {'TOTAL_PER_STEP':30s} avg={total:9.2f}ms", flush=True)


def pool_a_step(
    timer: StepTimer,
    layout: BlockDDPLayout,
    component_model: ComponentModel,
    optimizer: torch.optim.Optimizer,
    all_params: list[nn.Parameter],
    input_ids: Tensor,
    all_sites: list[str],
) -> dict[str, float]:
    """One training step for a pool-A rank."""

    with timer.phase("a/target_and_ci_fwd"):
        out = component_model(input_ids, cache_type="input")
        target_logits = out.output
        ci = component_model.calc_causal_importances(
            pre_weight_acts=out.cache, sampling="continuous", detach_inputs=False,
        )

    # ci.lower_leaky / upper_leaky here only cover layout.my_owned_sites — pool A only
    # has wrappers/ci fns for those.
    owned = list(layout.my_owned_sites)

    with timer.phase("a/send_ci_to_b"):
        ci_owned_for_send = {s: ci.lower_leaky[s] for s in owned}
        layout.send_owned_ci_to_pool_b(ci_owned_for_send)

    with timer.phase("a/faith"):
        weight_deltas = component_model.calc_weight_deltas()
        sum_sq = torch.zeros((), device=input_ids.device)
        numel = 0
        for d in weight_deltas.values():
            sum_sq = sum_sq + (d ** 2).sum()
            numel += d.numel()
        loss_faith = sum_sq / numel

    with timer.phase("a/imp"):
        imp_total = torch.zeros((), device=input_ids.device)
        for v in ci.upper_leaky.values():
            vals = (v + 1e-12).pow(1.0)
            sum_c = vals.sum(dim=tuple(range(vals.ndim - 1)))
            mean_c = sum_c / math.prod(vals.shape[:-1])
            imp_total = imp_total + mean_c.sum()
        loss_imp = imp_total

    with timer.phase("a/layerwise_total"):
        sl = layout.my_batch_slice_a()
        input_local = input_ids[sl]
        target_local = target_logits[sl].detach()
        ci_local = {s: ci.lower_leaky[s][sl] for s in owned}
        losses: list[Tensor] = []
        for s in owned:
            ci_s = ci_local[s]
            u = torch.rand_like(ci_s)
            mask = ci_s + (1 - ci_s) * u
            mask_infos = make_mask_infos({s: mask}, routing_masks="all")
            pred = component_model(input_local, mask_infos=mask_infos)
            losses.append(
                nn.functional.kl_div(
                    torch.log_softmax(pred, dim=-1),
                    torch.softmax(target_local, dim=-1),
                    reduction="batchmean",
                )
            )
        loss_stoch = torch.stack(losses).mean()

    total_home = 1e6 * loss_faith + 1e-4 * loss_imp + 0.5 * loss_stoch

    # Receive grads from pool B
    with timer.phase("a/recv_grads_from_b"):
        v_templates = {s: component_model.components[s].V for s in owned}  # type: ignore[attr-defined]
        u_templates = {s: component_model.components[s].U for s in owned}  # type: ignore[attr-defined]
        ci_lower_owned_full = {s: ci.lower_leaky[s] for s in owned}
        v_grads, u_grads, ci_grads = layout.recv_grads_from_pool_b(
            v_templates, u_templates, ci_lower_owned_full,
        )

    with timer.phase("a/seed_and_backward"):
        optimizer.zero_grad(set_to_none=True)
        for s in owned:
            comp = component_model.components[s]
            comp.V.grad = v_grads[s]  # type: ignore[attr-defined]
            comp.U.grad = u_grads[s]  # type: ignore[attr-defined]
        # Combined backward seeded by PPGD's ci grads at ci_lower
        torch.autograd.backward(
            tensors=[total_home, *(ci.lower_leaky[s] for s in owned)],
            grad_tensors=[None, *(ci_grads[s] for s in owned)],
        )

    with timer.phase("a/in_block_allreduce"):
        layout.all_reduce_grads_in_block(all_params)

    with timer.phase("a/opt_step"):
        optimizer.step()

    with timer.phase("a/send_weights_to_b"):
        v_owned = {s: component_model.components[s].V for s in owned}  # type: ignore[attr-defined]
        u_owned = {s: component_model.components[s].U for s in owned}  # type: ignore[attr-defined]
        layout.send_updated_weights_to_pool_b(v_owned, u_owned)

    return {
        "loss/faith": loss_faith.item(),
        "loss/imp": loss_imp.item(),
        "loss/stoch": loss_stoch.item(),
    }


def pool_b_step(
    timer: StepTimer,
    layout: BlockDDPLayout,
    component_model: ComponentModel,
    ppgd_state: PersistentPGDState,
    input_ids_full: Tensor,
    all_sites: list[str],
    site_to_c: dict[str, int],
    device: torch.device,
) -> dict[str, float]:
    """One training step for a pool-B rank using the real PersistentPGDState.

    The adversarial sources persist across training steps (per-batch-per-position
    scope, so each pool-B rank's sources are local to its batch slice). Per step
    we run n_warmup PGD inner steps that refine the sources, then compute the
    final recon loss with the refined sources for backward.
    """

    sl = layout.my_batch_slice_b()
    input_ids = input_ids_full[sl]

    with timer.phase("b/recv_ci_from_a"):
        ci_recv = layout.recv_ci_from_owners(
            site_to_c, seq_len=input_ids.shape[1], device=device, dtype=torch.float32,
        )

    with timer.phase("b/target_fwd"):
        with torch.no_grad():
            target_logits = component_model(input_ids)

    # Re-leaf the CI so we can backprop through it to get ci grads for pool A.
    ci_scratch = {s: v.detach().clone().requires_grad_(True) for s, v in ci_recv.items()}

    # PPGD warmup: refines the persistent adversarial sources in-place.
    with timer.phase("b/ppgd_warmup"):
        ppgd_state.warmup(
            model=component_model,
            batch=input_ids,
            target_out=target_logits.detach(),
            ci=ci_scratch,
            weight_deltas=None,
        )

    # Final PPGD recon loss with the (now-refined) sources.
    with timer.phase("b/ppgd_recon"):
        loss_ppgd = ppgd_state.compute_recon_loss(
            model=component_model,
            batch=input_ids,
            target_out=target_logits.detach(),
            ci=ci_scratch,
            weight_deltas=None,
        )
        # Scale by 1/N so that SUM-reduce of V/U grads across pool B equals
        # the full-batch gradient.
        total_ppgd = 0.5 * loss_ppgd / layout.world.n_pool_b

    with timer.phase("b/backward"):
        params: list[Tensor] = []
        for s in all_sites:
            params.append(component_model.components[s].V)  # type: ignore[attr-defined]
            params.append(component_model.components[s].U)  # type: ignore[attr-defined]
        ci_list = [ci_scratch[s] for s in all_sites]
        # Retain graph so we can also update the persistent sources from this loss.
        grads = torch.autograd.grad(total_ppgd, params + ci_list, retain_graph=True)

    # Update the persistent sources from the same loss (this is what makes PPGD
    # "persistent": the source state evolves across training steps).
    with timer.phase("b/ppgd_source_step"):
        source_grads = ppgd_state.get_grads(total_ppgd, retain_graph=False)
        ppgd_state.step(source_grads)

    n_sites = len(all_sites)
    v_grads = {s: grads[2 * i] for i, s in enumerate(all_sites)}
    u_grads = {s: grads[2 * i + 1] for i, s in enumerate(all_sites)}
    ci_grads = {s: grads[2 * n_sites + i] for i, s in enumerate(all_sites)}

    with timer.phase("b/pool_b_allreduce"):
        for s in all_sites:
            dist.all_reduce(v_grads[s], op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)
            dist.all_reduce(u_grads[s], op=dist.ReduceOp.SUM, group=layout.world.pool_b_group)

    with timer.phase("b/send_grads_to_a"):
        layout.send_pool_b_grads_to_owners(v_grads, u_grads, ci_grads)

    with timer.phase("b/recv_weights_from_a"):
        v_templates = {s: component_model.components[s].V for s in all_sites}  # type: ignore[attr-defined]
        u_templates = {s: component_model.components[s].U for s in all_sites}  # type: ignore[attr-defined]
        v_new, u_new = layout.recv_updated_weights_from_owners(v_templates, u_templates)
        with torch.no_grad():
            for s in all_sites:
                component_model.components[s].V.copy_(v_new[s])  # type: ignore[attr-defined]
                component_model.components[s].U.copy_(u_new[s])  # type: ignore[attr-defined]

    return {"loss/ppgd": loss_ppgd.item()}


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
        [g * N_PER_BLOCK_GROUP + k for k in range(N_PER_BLOCK_GROUP)]
        for g in range(N_BLOCK_GROUPS)
    ]
    block_owned_sites = [
        [s for tb in range(g * BLOCKS_PER_GROUP, (g + 1) * BLOCKS_PER_GROUP)
         for s in sites_for_block(tb)]
        for g in range(N_BLOCK_GROUPS)
    ]
    pool_b_ranks = list(range(
        N_BLOCK_GROUPS * N_PER_BLOCK_GROUP, N_BLOCK_GROUPS * N_PER_BLOCK_GROUP + N_POOL_B,
    ))

    world = build_block_ddp_world(
        block_groups=block_groups, block_owned_sites=block_owned_sites,
        pool_b_ranks=pool_b_ranks, batch_global=BATCH,
    )
    layout = BlockDDPLayout.from_world(world, rank)

    all_sites = [s for sites in block_owned_sites for s in sites]
    c_per_site = {s: C for s in all_sites}

    torch.manual_seed(0)
    target = TinyTransformer(VOCAB, D_MODEL, N_TRANSFORMER_BLOCKS, N_HEADS, D_MLP)
    target.requires_grad_(False)

    if layout.my_pool == "a":
        module_path_info = build_pool_a_module_path_info(layout, c_per_site)
    else:
        module_path_info = build_pool_b_module_path_info(layout, c_per_site)

    ci_config = LayerwiseCiConfig(fn_type="vector_mlp", hidden_dims=[CI_HIDDEN])

    target = target.to(device)
    component_model = ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        module_path_info=module_path_info,
        ci_config=ci_config,
        sigmoid_type="leaky_hard",
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
        optimizer = torch.optim.AdamW(all_params, lr=5e-5, weight_decay=0.0)
    else:
        # Pool B: build the persistent PGD state for adversarial source training.
        # PerBatchPerPositionScope → each pool-B rank's sources are local to its batch slice
        # (and PersistentPGDState's _skip_all_reduce auto-flips, which is what we want here:
        # pool B does its own cross-rank V/U all-reduce explicitly inside the step).
        ppgd_cfg = PersistentPGDReconLossConfig(
            coeff=1.0,
            scope=PerBatchPerPositionScope(),
            optimizer=SignPGDConfig(lr_schedule=ScheduleConfig(start_val=0.01)),
            n_warmup_steps=2,
            n_samples=1,
            use_sigmoid_parameterization=False,
        )
        ppgd_state = PersistentPGDState(
            module_to_c=c_per_site,
            batch_dims=(layout.world.batch_local_b, SEQ_LEN),
            device=device,
            use_delta_component=False,
            cfg=ppgd_cfg,
            reconstruction_loss=recon_loss_kl,
        )

    if rank == 0:
        n_target = sum(p.numel() for p in target.parameters())
        n_comp = sum(p.numel() for p in component_model._components.parameters())
        n_ci = sum(p.numel() for p in component_model.ci_fn.parameters())
        print(f"[two_pool] 2-POOL  (6A + 2B, using param_decomp.ComponentModel + layout)", flush=True)
        print(
            f"[two_pool] target~{n_target/1e6:.1f}M  pool-A per-rank components={n_comp/1e6:.1f}M  "
            f"ci_fn={n_ci/1e6:.1f}M",
            flush=True,
        )
        print(
            f"[two_pool] batch={BATCH} (A_local={layout.world.batch_local_a} "
            f"B_local={layout.world.batch_local_b}) seq={SEQ_LEN} "
            f"d={D_MODEL} d_mlp={D_MLP} n_blocks={N_TRANSFORMER_BLOCKS} ci_hidden={CI_HIDDEN}",
            flush=True,
        )

    timer = StepTimer(rank)
    data_rng = torch.Generator(device=device).manual_seed(0)

    def make_batch(step: int) -> Tensor:
        data_rng.manual_seed(step * 7919 + 17)
        return torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device, generator=data_rng)

    for step in range(WARMUP_STEPS + PROFILE_STEPS):
        input_ids = make_batch(step)
        torch.manual_seed(100 + step * 1000 + rank)

        with timer.phase("STEP_TOTAL"):
            if layout.my_pool == "a":
                assert optimizer is not None
                metrics = pool_a_step(
                    timer, layout, component_model, optimizer, all_params, input_ids, all_sites,
                )
            else:
                assert ppgd_state is not None
                metrics = pool_b_step(
                    timer, layout, component_model, ppgd_state, input_ids, all_sites, c_per_site, device,
                )

        if rank in (0, pool_b_ranks[0]):
            mem = torch.cuda.memory_allocated(device) / 1e9
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            print(
                f"[two_pool rank{rank}/{layout.my_pool}] step={step} "
                f"mem={mem:.2f}GB peak={peak:.2f}GB "
                f"{' '.join(f'{k}={v:.4g}' for k, v in metrics.items())}",
                flush=True,
            )

    timer.report(warmup=WARMUP_STEPS)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
