"""Disentangle every block's MLP on *stochastic*-masked forwards (no compression).

Unlike `run.py`/`run_mse.py`, the replacement MLPs are trained against the *stochastic*-
masked forward (masks drawn uniformly in `[ci.lower_leaky, 1]`, the same sampling real PD
uses for stochastic reconstruction), not the CI-masked one. The same stochastic mask draw
gates both the teacher (original MLP) and the student (replacement MLP) forward.

The replacement is not a bottleneck — it aims to *disentangle* how output subcomponent
activations arise from input subcomponent activations:

  - `N = C_out * d_expand` GeLU neurons. Each neuron connects to exactly one output
    subcomponent (so each output subcomponent is fed by `d_expand` neurons), via a scalar
    weight `w_neuron` (`C_out x d_expand`). The neurons read all input subcomponents through
    a dense `W_in` (`C_in x N`).
  - A dense linear bypass `W_bypass` (`C_in x C_out`) maps input subcomponents straight to
    output subcomponents, summed with the neuron contribution before the shared down mask.

Input/output subcomponent spaces stay frozen (`V_cfc`, `U_down` from the decomposition);
only `W_in`, `w_neuron`, `W_bypass` train. Attention is left untouched (runs masked by the
same masks via the component model). The objective is the output-logit KL of the replacement
(student) forward against the original-MLP (teacher) forward under the *same* masks — i.e.
KL(original-masked ‖ replacement-masked) at the final logits — optionally split across three
masking regimes:

  - `stochastic_coeff` weights KL under one *stochastic* mask draw (masks in `[ci, 1]`) with
    *subset routing over the MLP blocks*: each position routes a uniform-k random subset of the
    4 block MLPs to the masked path; the rest fall back to the original full MLP (attention is
    masked everywhere). This is PD's stochastic-recon-subset regime at the MLP-block granularity.
  - `adversarial_coeff` weights KL under *persistent adversarial* full-layer masks
    (`ci + (1-ci)*src`), where per-component sources persist across training steps and ascend
    (Adam) to maximize the replacement's KL — PD's PPGD-Recon adversary, but scoring the
    replacement's output fidelity. No subset routing here (every layer masked).
  - `unmasked_coeff` weights KL with *all component masks set to 1 and no weight-delta
    residual* (the delta component is the one thing left off) — PD's `UnmaskedReconLoss`
    regime, driving the replacement to reproduce the all-components-on forward. No routing.

With `adversarial_coeff == 0` and `unmasked_coeff == 0` (defaults) only the stochastic term
runs. No sparsity penalty. Residual-stream relative MSEs are still logged as evals
(`eval/resid_relmse_total`).

Run: python -m param_decomp_lab.experiments.mlp_compress.run_component --d_expand D [--steps N] ...
"""

import glob
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, override

import einops
import fire
import torch
import wandb
from dotenv import load_dotenv
from torch import Tensor, nn

from param_decomp.component_model import ComponentModel
from param_decomp.components import LinearComponents, init_param_
from param_decomp.distributed import (
    all_reduce,
    avg_metrics_across_ranks,
    is_main_process,
    seed_per_rank,
)
from param_decomp.masks import (
    AllLayersRouter,
    ComponentsMaskInfo,
    calc_stochastic_component_mask_info,
    make_mask_infos,
    sample_uniform_k_subset_routing_masks,
)
from param_decomp.metrics.persistent_pgd_state import AdamPGDConfig, make_ppgd_optimizer
from param_decomp.metrics.pgd_utils import PGDInitStrategy, get_pgd_init_tensor
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.batch_and_loss_fns import calc_kl_divergence_lm
from param_decomp_lab.distributed import cleanup_distributed, get_device, init_distributed
from param_decomp_lab.experiments.lm.data import LMDataConfig
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader
from param_decomp_lab.experiments.mlp_compress.run import RUN_DIR
from param_decomp_lab.experiments.mlp_compress.run_attn import mlp_module_names
from param_decomp_lab.experiments.mlp_compress.run_mse import (
    capture_residuals,
    relative_mse,
    resid_point_names,
)
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT

OUT_BASE = PARAM_DECOMP_OUT_DIR / "runs/s-55ea3f9b/component_mlp"


def read_dataset_from_local_cache(data_cfg: LMDataConfig) -> LMDataConfig:
    """Rewrite a streaming HF-hub data config to stream the same shards from the local cache.

    This cluster's egress to the HF Xet CDN flakily 408s mid-stream, which kills long runs.
    The full dataset is already in the shared HF cache, so we read the cached parquet shards
    directly via the `parquet` builder (no hub access). `data_files` is a per-split dict keyed
    by `train_split`/`eval_split` to match the split `build_lm_loader` requests.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_cache = f"{HF_HUB_CACHE}/datasets--{data_cfg.dataset_name.replace('/', '--')}"
    data_dirs = sorted(glob.glob(f"{repo_cache}/snapshots/*/data"))
    assert data_dirs, f"dataset not in local HF cache: {data_cfg.dataset_name}"
    data_dir = data_dirs[-1]
    files = {
        split: sorted(glob.glob(f"{data_dir}/{split}-*.parquet"))
        for split in (data_cfg.train_split, data_cfg.eval_split)
    }
    assert all(files.values()), f"missing cached shards under {data_dir}: {files}"
    return data_cfg.model_copy(
        update={"dataset_name": "parquet", "data_files": files, "revision": None}
    )


class ComponentMLP(nn.Module):
    """MLP replacement in subcomponent space: dense bypass + sparsely-wired GeLU neurons.

    Operates on the frozen subcomponent reader `V_cfc` (`d_in x C_in`) and writer `U_down`
    (`C_out x d_out`). With `m_cfc`/`m_down` the (stochastic) component masks, forward is:

        a_in   = (x @ V_cfc) * m_cfc                       # [..., C_in]
        bypass = a_in @ W_bypass                           # [..., C_out]
        h      = gelu(a_in @ W_in)                         # [..., N], N = C_out * d_expand
        neur   = (h.view(..., C_out, d_expand) * w_neuron).sum(-1)   # [..., C_out]
        a_out  = (bypass + neur) * m_down                  # [..., C_out]
        out    = a_out @ U_down                            # [..., d_out]

    Each neuron writes to a single output subcomponent (the `w_neuron` per-neuron scalar),
    so the wiring from neurons to output subcomponents is block-diagonal by construction.
    `d_expand=0` removes the neuron path entirely (output = masked bypass).
    """

    def __init__(self, V_cfc: Tensor, U_down: Tensor, d_expand: int, activation: nn.Module):
        super().__init__()
        self.register_buffer("V_cfc", V_cfc.clone())
        self.register_buffer("U_down", U_down.clone())
        C_in = V_cfc.shape[1]
        C_out = U_down.shape[0]
        self.C_out = C_out
        self.d_expand = d_expand
        n_neurons = C_out * d_expand
        self.W_bypass = nn.Parameter(torch.empty(C_in, C_out))
        self.W_in = nn.Parameter(torch.empty(C_in, n_neurons))
        self.w_neuron = nn.Parameter(torch.empty(C_out, d_expand))
        init_param_(self.W_bypass, fan_val=C_in, nonlinearity="linear")
        if d_expand > 0:
            init_param_(self.W_in, fan_val=C_in, nonlinearity="linear")
            init_param_(self.w_neuron, fan_val=d_expand, nonlinearity="linear")
        self.activation = activation

    @override
    def forward(self, x: Tensor, m_cfc: Tensor, m_down: Tensor) -> Tensor:
        assert isinstance(self.V_cfc, Tensor) and isinstance(self.U_down, Tensor)
        a_in = einops.einsum(x, self.V_cfc, "... d, d C -> ... C") * m_cfc
        bypass = einops.einsum(a_in, self.W_bypass, "... Ci, Ci Co -> ... Co")
        hidden = self.activation(einops.einsum(a_in, self.W_in, "... Ci, Ci N -> ... N"))
        hidden = einops.rearrange(hidden, "... (C d) -> ... C d", C=self.C_out, d=self.d_expand)
        neuron_out = einops.einsum(hidden, self.w_neuron, "... C d, C d -> ... C")
        a_out = (bypass + neuron_out) * m_down
        return einops.einsum(a_out, self.U_down, "... C, C d -> ... d")


def compute_stochastic_masks(
    comp_model: ComponentModel, batch: Tensor
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, Tensor], Tensor]:
    """Target forward -> CI (lower_leaky) -> one stochastic mask draw in [ci, 1].

    Returns (stochastic_mask_infos, ci_lower_leaky, target_logits). The mask draw is shared
    by the teacher (original MLP) and student (replacement MLP) forwards. `ci` is returned so
    the fixed-mask evals (ci / unmasked / rounded / adversarial) can reuse it.
    """
    out = comp_model(batch, cache_type="input")
    ci = comp_model.calc_causal_importances(out.cache, sampling="continuous").lower_leaky
    stoch = calc_stochastic_component_mask_info(ci, "continuous", None, AllLayersRouter())
    return stoch, ci, out.output


def compute_stochastic_subset_masks(
    comp_model: ComponentModel, batch: Tensor, n_blocks: int
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, Tensor], Tensor, dict[int, Tensor]]:
    """Target forward -> CI -> one stochastic mask draw, with subset routing over the MLP blocks.

    Attention components route everywhere (`"all"`); each block's MLP is routed as a unit via a
    uniform-k subset draw (its `c_fc` and `c_proj` share one per-position routing mask). At
    positions where a block's MLP is not routed, both teacher and student fall back to the
    original full MLP, so the masked replacement is only scored where the block is routed.

    Returns (mask_infos, ci_lower_leaky, target_logits, mlp_routing). `mlp_routing[block]` is the
    per-position bool mask the student hook uses to blend the replacement against the original MLP.
    """
    out = comp_model(batch, cache_type="input")
    ci = comp_model.calc_causal_importances(out.cache, sampling="continuous").lower_leaky
    leading_dims = tuple(next(iter(ci.values())).shape[:-1])
    device = next(iter(ci.values())).device
    block_routing = sample_uniform_k_subset_routing_masks(
        leading_dims, [str(b) for b in range(n_blocks)], device
    )
    name_to_routing: dict[str, Tensor] = {}
    mlp_routing: dict[int, Tensor] = {}
    for b in range(n_blocks):
        cfc_name, down_name = mlp_module_names(b)
        routing = block_routing[str(b)]
        name_to_routing[cfc_name] = routing
        name_to_routing[down_name] = routing
        mlp_routing[b] = routing
    mask_infos = {
        name: ComponentsMaskInfo(
            component_mask=c + (1 - c) * torch.rand_like(c),
            routing_mask=name_to_routing.get(name, "all"),
        )
        for name, c in ci.items()
    }
    return mask_infos, ci, out.output, mlp_routing


AdvSourceScope = Literal["per_batch_per_position", "shared_across_batch"]


class PersistentAdversarialMasks:
    """Per-component adversarial sources in `[0, 1]`, persisting across training steps.

    Sources ascend (Adam, via `AdamPGDOptimizer`) to *maximize* the replacement's residual
    relative MSE. Masks are `ci + (1 - ci) * source`, matching PD's PGD-Recon construction; the
    adversary here scores the replacement's fidelity rather than the recon loss.

    `scope` controls the source's leading (batch/sequence) dims, mirroring PD's PPGD scopes:
    `per_batch_per_position` gives an independent persistent source per batch element and position
    (what the real s-55ea3f9b run uses for its training PPGD); `shared_across_batch` keeps one
    source vector reused across all positions (what PD's PGD-Recon and our eval-time PGD use).
    """

    def __init__(
        self,
        component_c: dict[str, int],
        batch_dims: tuple[int, ...],
        scope: AdvSourceScope,
        device: torch.device | str,
        init: PGDInitStrategy,
        beta1: float,
        beta2: float,
        lr: float,
    ) -> None:
        match scope:
            case "per_batch_per_position":
                leading_dims = tuple(batch_dims)
            case "shared_across_batch":
                leading_dims = tuple(1 for _ in batch_dims)
        self.sources = {
            name: get_pgd_init_tensor(init, (*leading_dims, c), device).requires_grad_(True)
            for name, c in component_c.items()
        }
        cfg = AdamPGDConfig(
            beta1=beta1, beta2=beta2, lr_schedule=ScheduleConfig(start_val=lr, fn_type="constant")
        )
        self.optimizer = make_ppgd_optimizer(cfg)
        self.optimizer.init_state(self.sources)

    def masks(self, ci: dict[str, Tensor]) -> dict[str, Tensor]:
        batch_dims = next(iter(ci.values())).shape[:-1]
        return {k: ci[k] + (1 - ci[k]) * self.sources[k].expand(*batch_dims, -1) for k in ci}

    def ascend(self, grads: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.optimizer.step(self.sources, grads)
            for source in self.sources.values():
                source.clamp_(0.0, 1.0)


@contextmanager
def mlp_blocks_replaced(
    mlp_modules: dict[int, nn.Module],
    component_mlps: dict[int, ComponentMLP],
    mlp_masks: dict[int, tuple[Tensor, Tensor]],
    mlp_routing: dict[int, Tensor] | None,
) -> Iterator[None]:
    """Swap each block's MLP for its `ComponentMLP`. With `mlp_routing`, blend the replacement
    against the original MLP per position (replacement where routed, original elsewhere); with
    `None`, the replacement output is used everywhere.
    """
    handles = []
    for block, mlp in mlp_modules.items():
        compressed = component_mlps[block]
        m_cfc, m_down = mlp_masks[block]
        routing = None if mlp_routing is None else mlp_routing[block]

        def make_hook(
            compressed: ComponentMLP, m_cfc: Tensor, m_down: Tensor, routing: Tensor | None
        ):
            def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
                replaced = compressed(args[0], m_cfc, m_down)
                if routing is None:
                    return replaced
                return torch.where(routing[..., None], replaced, output)

            return hook

        handles.append(mlp.register_forward_hook(make_hook(compressed, m_cfc, m_down, routing)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def student_forward(
    comp_model: ComponentModel,
    batch: Tensor,
    stoch: dict[str, ComponentsMaskInfo],
    mlp_modules: dict[int, nn.Module],
    component_mlps: dict[int, ComponentMLP],
    mlp_routing: dict[int, Tensor] | None,
) -> Tensor:
    """Stochastic-masked forward with every block's MLP swapped for its `ComponentMLP`.

    Attention components keep their stochastic masks (run normally); MLP component keys are
    dropped from `mask_infos` and the MLP modules are hooked to emit the replacement output
    gated by the same stochastic masks. `mlp_routing` blends the replacement against the
    original MLP per position (`None` = replace everywhere).
    """
    drop: set[str] = set()
    mlp_masks: dict[int, tuple[Tensor, Tensor]] = {}
    for block in mlp_modules:
        cfc_name, down_name = mlp_module_names(block)
        drop.add(cfc_name)
        drop.add(down_name)
        mlp_masks[block] = (stoch[cfc_name].component_mask, stoch[down_name].component_mask)
    student_mask_infos = {k: v for k, v in stoch.items() if k not in drop}
    with mlp_blocks_replaced(mlp_modules, component_mlps, mlp_masks, mlp_routing):
        return comp_model(batch, mask_infos=student_mask_infos)


def masked_student_kl_to_teacher(
    comp_model: ComponentModel,
    batch: Tensor,
    mask_infos: dict[str, ComponentsMaskInfo],
    mlp_modules: dict[int, nn.Module],
    component_mlps: dict[int, ComponentMLP],
    mlp_routing: dict[int, Tensor] | None,
) -> Tensor:
    """KL(original-masked forward ‖ replacement-masked forward) at the output logits.

    Teacher (original MLPs under `mask_infos`) is a detached target; gradient flows through the
    replacement (student) only. `mlp_routing` blends the replacement against the original MLP per
    position (`None` = replace the MLP everywhere).
    """
    with torch.no_grad():
        teacher_logits = comp_model(batch, mask_infos=mask_infos)
    student_logits = student_forward(
        comp_model, batch, mask_infos, mlp_modules, component_mlps, mlp_routing
    )
    return calc_kl_divergence_lm(pred=student_logits.float(), target=teacher_logits.float())


def masked_student_kl_to_target(
    comp_model: ComponentModel,
    batch: Tensor,
    component_masks: dict[str, Tensor],
    target_logits: Tensor,
    mlp_modules: dict[int, nn.Module],
    component_mlps: dict[int, ComponentMLP],
) -> float:
    """KL(replacement forward under `component_masks` ‖ original target model)."""
    student_logits = student_forward(
        comp_model, batch, make_mask_infos(component_masks), mlp_modules, component_mlps, None
    )
    return calc_kl_divergence_lm(pred=student_logits.float(), target=target_logits.float()).item()


def adversarial_student_kl_to_target(
    comp_model: ComponentModel,
    batch: Tensor,
    ci: dict[str, Tensor],
    target_logits: Tensor,
    mlp_modules: dict[int, nn.Module],
    component_mlps: dict[int, ComponentMLP],
    init: PGDInitStrategy,
    step_size: float,
    n_steps: int,
) -> float:
    """Worst-case KL(replacement forward ‖ target) over PGD-Recon-style adversarial masks.

    Mirrors `param_decomp.metrics.pgd_utils`: per-component adversarial *sources* in `[0, 1]`,
    interpolated with CI as `mask = ci + (1 - ci) * source`, optimized by sign-PGD to *maximize*
    the divergence of the replacement forward from the target. Sources are shared across all
    leading (batch + sequence) dims (`shared_across_batch`). Single-GPU, so the DDP all-reduce
    /broadcast of the library path is unnecessary.
    """
    batch_dims = next(iter(ci.values())).shape[:-1]
    sources = {
        name: get_pgd_init_tensor(
            init, (*[1 for _ in batch_dims], c.shape[-1]), c.device
        ).requires_grad_(True)
        for name, c in ci.items()
    }

    def student_kl() -> Tensor:
        masks = {k: ci[k] + (1 - ci[k]) * sources[k].expand(*batch_dims, -1) for k in ci}
        student_logits = student_forward(
            comp_model, batch, make_mask_infos(masks), mlp_modules, component_mlps, None
        )
        return calc_kl_divergence_lm(pred=student_logits.float(), target=target_logits.float())

    for _ in range(n_steps):
        with torch.enable_grad():
            loss = student_kl()
        grads = torch.autograd.grad(loss, list(sources.values()))
        with torch.no_grad():
            for name, grad in zip(sources, grads, strict=True):
                sources[name].add_(step_size * grad.sign())
                sources[name].clamp_(0.0, 1.0)

    return student_kl().item()


def main(
    d_expand: int,
    steps: int = 20_000,
    batch_size: int = 32,
    lr: float = 1e-3,
    warmup_steps: int = 200,
    final_lr_frac: float = 0.1,
    stochastic_coeff: float = 1.0,
    adversarial_coeff: float = 0.0,
    unmasked_coeff: float = 0.0,
    adv_lr: float = 0.01,
    adv_beta1: float = 0.0,
    adv_beta2: float = 0.99,
    adv_n_warmup: int = 1,
    adv_init: PGDInitStrategy = "random",
    adv_scope: AdvSourceScope = "per_batch_per_position",
    eval_every: int = 250,
    n_eval_batches: int = 4,
    save_every: int = 2_500,
    pgd_init: PGDInitStrategy = "random",
    pgd_step_size: float = 0.1,
    pgd_n_steps: int = 20,
    seed: int = 0,
    use_wandb: bool = True,
) -> None:
    load_dotenv(REPO_ROOT / ".env")
    assert torch.cuda.is_available(), "needs a GPU"
    dist_state = init_distributed()
    world_size = dist_state.world_size if dist_state is not None else 1
    rank = dist_state.rank if dist_state is not None else 0
    device = get_device()
    torch.manual_seed(seed)
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # noqa: E731

    out_dir = OUT_BASE / f"dexp{d_expand}_{time.strftime('%Y%m%d_%H%M%S')}"
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=False)

    pd_run = SavedLMRun.from_path(RUN_DIR)
    comp_model = pd_run.load_model().to(device)
    comp_model.eval()
    comp_model.requires_grad_(False)

    n_blocks = 0
    while mlp_module_names(n_blocks)[0] in comp_model.components:
        n_blocks += 1
    assert n_blocks == 4, f"expected 4 blocks, found {n_blocks}"

    mlp_modules: dict[int, nn.Module] = {}
    component_mlps: dict[int, ComponentMLP] = {}
    for block in range(n_blocks):
        cfc_name, down_name = mlp_module_names(block)
        cfc = comp_model.components[cfc_name]
        down = comp_model.components[down_name]
        assert isinstance(cfc, LinearComponents) and isinstance(down, LinearComponents)
        assert cfc.bias is None and down.bias is None, "target MLP is bias-free"
        assert cfc.d_out == down.d_in == 3072
        mlp = comp_model.target_model.get_submodule(f"h.{block}.mlp")
        mlp_modules[block] = mlp
        gelu = mlp.gelu
        assert isinstance(gelu, nn.Module)
        component_mlps[block] = ComponentMLP(
            V_cfc=cfc.V.data,
            U_down=down.U.data,
            d_expand=d_expand,
            activation=gelu,
        ).to(device)

    params = [p for c in component_mlps.values() for p in c.parameters()]
    n_trainable = sum(p.numel() for p in params if p.requires_grad)
    point_names = resid_point_names(n_blocks)
    C_in = comp_model.components[mlp_module_names(0)[0]].C
    C_out = comp_model.components[mlp_module_names(0)[1]].C
    config = {
        "run": "s-55ea3f9b (via p-55ea3f9b)",
        "objective": "output_kl_divergence",
        "stochastic_regime": "stochastic_recon_subset_over_mlp_blocks",
        "adversarial_regime": "persistent_pgd_full_layer",
        "unmasked_regime": "all_components_on_no_delta",
        "stochastic_coeff": stochastic_coeff,
        "adversarial_coeff": adversarial_coeff,
        "unmasked_coeff": unmasked_coeff,
        "adv_lr": adv_lr,
        "adv_beta1": adv_beta1,
        "adv_beta2": adv_beta2,
        "adv_n_warmup": adv_n_warmup,
        "adv_init": adv_init,
        "adv_scope": adv_scope,
        "n_resid_points": len(point_names),
        "n_blocks": n_blocks,
        "d_expand": d_expand,
        "n_neurons_per_block": C_out * d_expand,
        "C_in": C_in,
        "C_out": C_out,
        "mlp_activation": "gelu",
        "pgd_init": pgd_init,
        "pgd_step_size": pgd_step_size,
        "pgd_n_steps": pgd_n_steps,
        "steps": steps,
        "batch_size": batch_size,
        "world_size": world_size,
        "effective_batch_size": batch_size * world_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "final_lr_frac": final_lr_frac,
        "seed": seed,
        "n_trainable_params": n_trainable,
    }
    if is_main_process():
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        print(f"out_dir: {out_dir}")
        print(f"config: {json.dumps(config, indent=2)}")

    is_adversarial = adversarial_coeff > 0
    mode_tag = "adversarial" if is_adversarial else "stochastic"
    wb = None
    if use_wandb and is_main_process():
        wb = wandb.init(
            project="spd",
            name=f"component-mlp-{mode_tag}-dexp{d_expand}-s-55ea3f9b",
            group=f"component_mlp_{mode_tag}_dexpand_sweep",
            tags=["component_mlp", mode_tag, "output_kl"],
            config=config,
        )
        print(f"wandb: {wb.url}")

    data_cfg = read_dataset_from_local_cache(pd_run.cfg.data)
    train_loader = build_lm_loader(
        pd_run.cfg.target,
        data_cfg,
        split="train",
        device=device,
        batch_size=batch_size,
        seed=seed + rank,
    )
    eval_loader = build_lm_loader(
        pd_run.cfg.target,
        data_cfg,
        split="eval",
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    eval_iter = iter(eval_loader)
    eval_batches = [next(eval_iter).to(device) for _ in range(n_eval_batches)]

    opt = torch.optim.Adam(params, lr=lr)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        cos = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return final_lr_frac + (1 - final_lr_frac) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Fixed eval context: one stochastic mask draw per eval batch, frozen for the whole run.
    eval_ctx = []
    with torch.no_grad(), autocast():
        for b in eval_batches:
            stoch, ci, target_logits = compute_stochastic_masks(comp_model, b)
            with capture_residuals(comp_model, n_blocks) as tr:
                teacher_logits = comp_model(b, mask_infos=stoch)
            teacher_resids = {k: v.detach().clone() for k, v in tr.items()}
            assert set(teacher_resids) == set(point_names)
            eval_ctx.append((b, stoch, ci, target_logits, teacher_logits, teacher_resids))

    with torch.no_grad(), autocast():
        kl_teacher_vs_target = sum(
            calc_kl_divergence_lm(pred=tl.float(), target=gl.float()).item()
            for _b, _st, _ci, gl, tl, _tr in eval_ctx
        ) / len(eval_ctx)
    references = {"kl_stoch_masked_vs_target": kl_teacher_vs_target}
    if is_main_process():
        print(f"references: {references}")
        (out_dir / "references.json").write_text(json.dumps(references, indent=2))
        if wb is not None:
            wb.summary.update(references)

    def run_eval() -> dict[str, float]:
        kl_vs_teacher = 0.0
        kl_vs_target = 0.0
        relmse_total = 0.0
        kl_ci = 0.0
        kl_unmasked = 0.0
        kl_rounded = 0.0
        kl_adversarial = 0.0
        with torch.no_grad(), autocast():
            for b, stoch, ci, target_logits, teacher_logits, teacher_resids in eval_ctx:
                with capture_residuals(comp_model, n_blocks) as sr:
                    student_logits = student_forward(
                        comp_model, b, stoch, mlp_modules, component_mlps, None
                    )
                kl_vs_teacher += calc_kl_divergence_lm(
                    pred=student_logits.float(), target=teacher_logits.float()
                ).item()
                kl_vs_target += calc_kl_divergence_lm(
                    pred=student_logits.float(), target=target_logits.float()
                ).item()
                relmse_total += sum(
                    relative_mse(sr[k].float(), teacher_resids[k].float()).item()
                    for k in point_names
                )
                ones = {k: torch.ones_like(v) for k, v in ci.items()}
                rounded = {k: (v > 0).to(v.dtype) for k, v in ci.items()}
                kl_ci += masked_student_kl_to_target(
                    comp_model, b, ci, target_logits, mlp_modules, component_mlps
                )
                kl_unmasked += masked_student_kl_to_target(
                    comp_model, b, ones, target_logits, mlp_modules, component_mlps
                )
                kl_rounded += masked_student_kl_to_target(
                    comp_model, b, rounded, target_logits, mlp_modules, component_mlps
                )
                kl_adversarial += adversarial_student_kl_to_target(
                    comp_model,
                    b,
                    ci,
                    target_logits,
                    mlp_modules,
                    component_mlps,
                    init=pgd_init,
                    step_size=pgd_step_size,
                    n_steps=pgd_n_steps,
                )
        n = len(eval_ctx)
        return {
            "eval/kl_student_vs_teacher_stoch": kl_vs_teacher / n,
            "eval/kl_student_vs_target": kl_vs_target / n,
            "eval/resid_relmse_total": relmse_total / n,
            "eval/kl_ci_masked_vs_target": kl_ci / n,
            "eval/kl_unmasked_vs_target": kl_unmasked / n,
            "eval/kl_rounded_vs_target": kl_rounded / n,
            "eval/kl_adversarial_vs_target": kl_adversarial / n,
        }

    if world_size > 1:
        seed_per_rank(seed)

    adversarial = None
    if adversarial_coeff > 0:
        sample_ci = eval_ctx[0][2]
        component_c = {name: c.shape[-1] for name, c in sample_ci.items()}
        batch_dims = next(iter(sample_ci.values())).shape[:-1]
        adversarial = PersistentAdversarialMasks(
            component_c, batch_dims, adv_scope, device, adv_init, adv_beta1, adv_beta2, adv_lr
        )

    metrics_path = out_dir / "metrics.jsonl"
    train_iter = iter(train_loader)
    last_log_time = time.time()
    for step in range(steps):
        batch = next(train_iter).to(device)

        with torch.no_grad(), autocast():
            subset_mask_infos, ci, _, mlp_routing = compute_stochastic_subset_masks(
                comp_model, batch, n_blocks
            )

        # Each regime's KL builds a full student autograd graph. Backward each term as soon as
        # it is computed (gradient accumulation) rather than summing into one loss and a single
        # backward — numerically identical, but only one student graph is alive at a time, which
        # keeps peak activation memory to a single forward instead of stacking all three.
        opt.zero_grad(set_to_none=True)
        if adversarial is not None:
            for source in adversarial.sources.values():
                source.grad = None

        with autocast():
            kl_stoch = masked_student_kl_to_teacher(
                comp_model, batch, subset_mask_infos, mlp_modules, component_mlps, mlp_routing
            )
        (stochastic_coeff * kl_stoch).backward()

        kl_unmasked = None
        if unmasked_coeff > 0:
            ones = {k: torch.ones_like(v) for k, v in ci.items()}
            with autocast():
                kl_unmasked = masked_student_kl_to_teacher(
                    comp_model, batch, make_mask_infos(ones), mlp_modules, component_mlps, None
                )
            (unmasked_coeff * kl_unmasked).backward()

        kl_adv = None
        if adversarial is not None:
            for _ in range(adv_n_warmup):
                with autocast():
                    warm_loss = adversarial_coeff * masked_student_kl_to_teacher(
                        comp_model,
                        batch,
                        make_mask_infos(adversarial.masks(ci)),
                        mlp_modules,
                        component_mlps,
                        None,
                    )
                warm_grads = torch.autograd.grad(warm_loss, list(adversarial.sources.values()))
                adversarial.ascend(dict(zip(adversarial.sources, warm_grads, strict=True)))
            with autocast():
                kl_adv = masked_student_kl_to_teacher(
                    comp_model,
                    batch,
                    make_mask_infos(adversarial.masks(ci)),
                    mlp_modules,
                    component_mlps,
                    None,
                )
            (adversarial_coeff * kl_adv).backward()

        loss = (
            stochastic_coeff * kl_stoch.item()
            + (unmasked_coeff * kl_unmasked.item() if kl_unmasked is not None else 0.0)
            + (adversarial_coeff * kl_adv.item() if kl_adv is not None else 0.0)
        )
        if adversarial is not None:
            grads: dict[str, Tensor] = {}
            for name, src in adversarial.sources.items():
                assert src.grad is not None
                grads[name] = src.grad
            adversarial.ascend(grads)
        if world_size > 1:
            for p in params:
                assert p.grad is not None
                all_reduce(p.grad)
                p.grad /= world_size
        opt.step()
        scheduler.step()

        train_scalars: dict[str, float] = {
            "train/loss": loss,
            "train/stochastic_subset_kl": kl_stoch.item(),
        }
        if kl_unmasked is not None:
            train_scalars["train/unmasked_kl"] = kl_unmasked.item()
        if kl_adv is not None:
            train_scalars["train/adversarial_kl"] = kl_adv.item()
        train_scalars = dict(avg_metrics_across_ranks(train_scalars, device))

        if not is_main_process():
            continue

        record: dict[str, float] = {
            "step": step,
            "lr": float(scheduler.get_last_lr()[0]),
            **train_scalars,
        }
        if step % eval_every == 0 or step == steps - 1:
            record |= run_eval()
            now = time.time()
            record["steps_per_s"] = eval_every / (now - last_log_time)
            last_log_time = now
            print(json.dumps(record))
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        if wb is not None:
            wb.log(record, step=step)

        if (step > 0 and step % save_every == 0) or step == steps - 1:
            torch.save(
                {b: c.state_dict() for b, c in component_mlps.items()},
                out_dir / f"component_mlp_{step}.pt",
            )

    if is_main_process():
        final = run_eval()
        print(f"final: {final} | references: {references}")
        (out_dir / "final.json").write_text(json.dumps(final | references, indent=2))
        if wb is not None:
            wb.summary.update(final)
            wb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(main)
