"""Vanilla single-pool DDP baseline using param_decomp's ComponentModel.

Every rank holds a full ComponentModel — components V/U at every site, the per-module
CI fn for every site, plus AdamW state for both. Each rank runs:

  1. target_model + CI fn forward (via ComponentModel.calc_causal_importances)
  2. faithfulness loss (sum of squared weight-deltas)
  3. importance-minimality loss (cheap, on ci_upper)
  4. per-site layerwise loss: serial M-iteration loop, one site routed at a time
  5. all-sites masked forward + an inline adversarial source PGD inner step
  6. backward + DDP all-reduce + AdamW step

This is the apples-to-apples baseline that the 2-pool design competes with. It
only fits at moderate scales because per-rank memory has to hold everything.

Run on 8 GPUs single-node:
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \\
        -m param_decomp.scripts.two_pool_benchmark.vanilla
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

from param_decomp.configs import LayerwiseCiConfig
from param_decomp.models.batch_and_loss_fns import run_batch_passthrough
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import make_mask_infos
from param_decomp.scripts.two_pool_benchmark._tiny_model import TinyTransformer, sites_for_block
from param_decomp.utils.module_utils import ModulePathInfo

VOCAB = 8192
D_MODEL = 768
N_HEADS = 12
D_MLP = 3072
N_TRANSFORMER_BLOCKS = 6
BATCH = 8
SEQ_LEN = 64
C = 32
CI_HIDDEN = 1024   # vector_mlp grows fast; keep this modest so vanilla DDP-N fits

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
        if self.rank != 0:
            return
        print(f"\n[vanilla rank{self.rank}] phase wall-clock (skipping first {warmup}):", flush=True)
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


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    assert BATCH % world_size == 0
    batch_local = BATCH // world_size

    torch.manual_seed(0)
    target = TinyTransformer(VOCAB, D_MODEL, N_TRANSFORMER_BLOCKS, N_HEADS, D_MLP)
    target.requires_grad_(False)

    all_sites = [s for b in range(N_TRANSFORMER_BLOCKS) for s in sites_for_block(b)]
    module_path_info = [ModulePathInfo(module_path=s, C=C) for s in all_sites]

    ci_config = LayerwiseCiConfig(fn_type="vector_mlp", hidden_dims=[CI_HIDDEN])

    target = target.to(device)
    component_model = ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        module_path_info=module_path_info,
        ci_config=ci_config,
        sigmoid_type="leaky_hard",
    ).to(device)

    component_params: list[nn.Parameter] = []
    for name in component_model.target_module_paths:
        component_params.extend(component_model.components[name].parameters())
    ci_fn_params = list(component_model.ci_fn.parameters())
    all_params = component_params + ci_fn_params
    optimizer = torch.optim.AdamW(all_params, lr=5e-5, weight_decay=0.0)

    if rank == 0:
        n_comp = sum(p.numel() for p in component_params)
        n_ci = sum(p.numel() for p in ci_fn_params)
        n_target = sum(p.numel() for p in target.parameters())
        print(f"[vanilla] VANILLA DDP-{world_size}  (using param_decomp.ComponentModel)", flush=True)
        print(
            f"[vanilla] target~{n_target/1e6:.1f}M  components={n_comp/1e6:.1f}M  "
            f"ci_fn={n_ci/1e6:.1f}M  trainable_per_rank={(n_comp+n_ci)/1e6:.1f}M",
            flush=True,
        )
        print(
            f"[vanilla] batch={BATCH} (local={batch_local}) seq={SEQ_LEN} "
            f"d={D_MODEL} d_mlp={D_MLP} n_blocks={N_TRANSFORMER_BLOCKS} ci_hidden={CI_HIDDEN}",
            flush=True,
        )

    timer = StepTimer(rank)
    data_rng = torch.Generator(device=device).manual_seed(0)

    def make_batch(step: int) -> Tensor:
        data_rng.manual_seed(step * 7919 + 17)
        full = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device, generator=data_rng)
        return full[rank * batch_local : (rank + 1) * batch_local]

    for step in range(WARMUP_STEPS + PROFILE_STEPS):
        input_ids = make_batch(step)
        torch.manual_seed(100 + step * 1000 + rank)

        with timer.phase("STEP_TOTAL"):
            # 1. target + CI forward
            with timer.phase("target_and_ci_fwd"):
                output_with_cache = component_model(input_ids, cache_type="input")
                target_logits = output_with_cache.output
                ci = component_model.calc_causal_importances(
                    pre_weight_acts=output_with_cache.cache,
                    sampling="continuous",
                    detach_inputs=False,
                )

            # 2. faith
            with timer.phase("faith"):
                weight_deltas = component_model.calc_weight_deltas()
                sum_sq = torch.zeros((), device=device)
                numel = 0
                for d in weight_deltas.values():
                    sum_sq = sum_sq + (d ** 2).sum()
                    numel += d.numel()
                loss_faith = sum_sq / numel

            # 3. imp (cheap, on ci_upper)
            with timer.phase("imp"):
                imp_total = torch.zeros((), device=device)
                for v in ci.upper_leaky.values():
                    p = 1.0
                    vals = (v + 1e-12).pow(p)
                    sum_c = vals.sum(dim=tuple(range(vals.ndim - 1)))
                    mean_c = sum_c / math.prod(vals.shape[:-1])
                    imp_total = imp_total + mean_c.sum()
                loss_imp = imp_total

            # 4. per-site layerwise: serial M-iteration loop
            with timer.phase("layerwise_total"):
                layerwise_losses: list[Tensor] = []
                for site in all_sites:
                    ci_s = ci.lower_leaky[site]
                    u = torch.rand_like(ci_s)
                    mask = ci_s + (1 - ci_s) * u
                    mask_infos = make_mask_infos({site: mask}, routing_masks="all")
                    pred = component_model(input_ids, mask_infos=mask_infos)
                    layerwise_losses.append(
                        nn.functional.kl_div(
                            torch.log_softmax(pred, dim=-1),
                            torch.softmax(target_logits.detach(), dim=-1),
                            reduction="batchmean",
                        )
                    )
                loss_stoch = torch.stack(layerwise_losses).mean()

            # 5. ppgd: all sites masked, one inline adversarial step
            with timer.phase("ppgd"):
                with torch.no_grad():
                    adv_masks_raw = {
                        s: ci.lower_leaky[s] + (1 - ci.lower_leaky[s]) * torch.rand_like(ci.lower_leaky[s])
                        for s in all_sites
                    }
                adv_masks = {s: m.detach().requires_grad_(True) for s, m in adv_masks_raw.items()}

                # One PGD inner step on the masks
                mask_infos_adv = make_mask_infos(adv_masks, routing_masks="all")
                pred_adv = component_model(input_ids, mask_infos=mask_infos_adv)
                inner_loss = nn.functional.kl_div(
                    torch.log_softmax(pred_adv, dim=-1),
                    torch.softmax(target_logits.detach(), dim=-1),
                    reduction="batchmean",
                )
                inner_grads = torch.autograd.grad(inner_loss, list(adv_masks.values()), retain_graph=False)
                with torch.no_grad():
                    for s, g in zip(adv_masks, inner_grads, strict=True):
                        adv_masks[s].add_(0.01 * g.sign())
                        adv_masks[s].clamp_(0.0, 1.0)

                # Final ppgd recon with the updated masks (graph-attached this time)
                final_masks = {
                    s: ci.lower_leaky[s] + (1 - ci.lower_leaky[s]) * adv_masks[s].detach()
                    for s in all_sites
                }
                mask_infos_final = make_mask_infos(final_masks, routing_masks="all")
                pred_ppgd = component_model(input_ids, mask_infos=mask_infos_final)
                loss_ppgd = nn.functional.kl_div(
                    torch.log_softmax(pred_ppgd, dim=-1),
                    torch.softmax(target_logits.detach(), dim=-1),
                    reduction="batchmean",
                )

            total = 1e6 * loss_faith + 1e-4 * loss_imp + 0.5 * loss_stoch + 0.5 * loss_ppgd

            with timer.phase("backward"):
                optimizer.zero_grad(set_to_none=True)
                total.backward()

            with timer.phase("ddp_allreduce"):
                for p in all_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            with timer.phase("opt_step"):
                optimizer.step()

        if rank == 0:
            mem = torch.cuda.memory_allocated(device) / 1e9
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            print(
                f"[vanilla rank0] step={step} mem={mem:.2f}GB peak={peak:.2f}GB "
                f"faith={loss_faith.item():.4g} stoch={loss_stoch.item():.4g} "
                f"ppgd={loss_ppgd.item():.4g}",
                flush=True,
            )

    timer.report(warmup=WARMUP_STEPS)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
