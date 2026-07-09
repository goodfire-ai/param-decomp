"""Distill all block MLPs of s-55ea3f9b to a smaller neuron dim simultaneously, on CI-masked forwards.

Same recipe as `run.py`, but every block's MLP is replaced by its own compressed bottleneck
within a single shared masked forward pass — so the replacements are trained jointly, each
block's compressed MLP seeing inputs produced by the (compressed) earlier blocks. CI masks for
all blocks are reused from the teacher pass. Loss: KL(teacher || student) on output logits.

Run: python -m param_decomp_lab.experiments.mlp_compress.run_all --n_compressed N [--steps N] ...
"""

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager

import fire
import torch
import wandb
from dotenv import load_dotenv
from torch import Tensor, nn

from param_decomp.component_model import ComponentModel
from param_decomp.components import LinearComponents
from param_decomp.masks import ComponentsMaskInfo, make_mask_infos
from param_decomp_lab.batch_and_loss_fns import calc_kl_divergence_lm
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader
from param_decomp_lab.experiments.mlp_compress.run import (
    RUN_DIR,
    CompressedMaskedMLP,
    compute_ci_and_teacher,
)
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT

OUT_BASE = PARAM_DECOMP_OUT_DIR / "runs/s-55ea3f9b/all_mlp_compress"


def block_module_names(block: int) -> tuple[str, str]:
    return f"h.{block}.mlp.c_fc", f"h.{block}.mlp.down_proj"


@contextmanager
def mlps_replaced(
    mlp_modules: dict[int, nn.Module],
    compressed_by_block: dict[int, CompressedMaskedMLP],
    ci: dict[str, Tensor],
) -> Iterator[None]:
    handles = []
    for block, mlp in mlp_modules.items():
        cfc_name, down_name = block_module_names(block)
        compressed = compressed_by_block[block]
        ci_cfc, ci_down = ci[cfc_name], ci[down_name]

        def make_hook(compressed: CompressedMaskedMLP, ci_cfc: Tensor, ci_down: Tensor):
            def hook(_module: nn.Module, args: tuple[Tensor, ...], _output: Tensor) -> Tensor:
                return compressed(args[0], ci_cfc, ci_down)

            return hook

        handles.append(mlp.register_forward_hook(make_hook(compressed, ci_cfc, ci_down)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def student_forward(
    comp_model: ComponentModel,
    batch: Tensor,
    ci: dict[str, Tensor],
    mask_infos: dict[str, ComponentsMaskInfo],
    mlp_modules: dict[int, nn.Module],
    compressed_by_block: dict[int, CompressedMaskedMLP],
) -> Tensor:
    drop = {name for block in mlp_modules for name in block_module_names(block)}
    student_mask_infos = {k: v for k, v in mask_infos.items() if k not in drop}
    with mlps_replaced(mlp_modules, compressed_by_block, ci):
        return comp_model(batch, mask_infos=student_mask_infos)


def main(
    n_compressed: int,
    steps: int = 20_000,
    batch_size: int = 32,
    lr: float = 1e-3,
    warmup_steps: int = 200,
    final_lr_frac: float = 0.1,
    eval_every: int = 250,
    n_eval_batches: int = 4,
    save_every: int = 2_500,
    seed: int = 0,
    use_wandb: bool = True,
) -> None:
    load_dotenv(REPO_ROOT / ".env")
    assert torch.cuda.is_available(), "needs a GPU"
    device = "cuda"
    torch.manual_seed(seed)
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # noqa: E731

    out_dir = OUT_BASE / f"n{n_compressed}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)

    pd_run = SavedLMRun.from_path(RUN_DIR)
    comp_model = pd_run.load_model().to(device)
    comp_model.eval()
    comp_model.requires_grad_(False)

    n_blocks = 0
    while block_module_names(n_blocks)[0] in comp_model.components:
        n_blocks += 1
    assert n_blocks == 4, f"expected 4 blocks, found {n_blocks}"

    mlp_modules: dict[int, nn.Module] = {}
    compressed_by_block: dict[int, CompressedMaskedMLP] = {}
    for block in range(n_blocks):
        cfc_name, down_name = block_module_names(block)
        cfc = comp_model.components[cfc_name]
        down = comp_model.components[down_name]
        assert isinstance(cfc, LinearComponents) and isinstance(down, LinearComponents)
        assert cfc.bias is None and down.bias is None, "target MLP is bias-free"
        assert cfc.d_out == down.d_in == 3072
        mlp = comp_model.target_model.get_submodule(f"h.{block}.mlp")
        mlp_modules[block] = mlp
        gelu = mlp.gelu
        assert isinstance(gelu, nn.Module)
        compressed_by_block[block] = CompressedMaskedMLP(
            V_cfc=cfc.V.data,
            U_down=down.U.data,
            n_compressed=n_compressed,
            activation=gelu,
            bypass=False,
        ).to(device)

    params = [p for c in compressed_by_block.values() for p in c.parameters()]
    n_trainable = sum(p.numel() for p in params if p.requires_grad)
    config = {
        "run": "s-55ea3f9b (via p-55ea3f9b)",
        "n_blocks": n_blocks,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "final_lr_frac": final_lr_frac,
        "seed": seed,
        "d_mlp": 3072,
        "n_compressed": n_compressed,
        "n_trainable_params": n_trainable,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"out_dir: {out_dir}")
    print(f"config: {json.dumps(config, indent=2)}")

    wb = None
    if use_wandb:
        wb = wandb.init(
            project="spd",
            name=f"compress-allmlp-n{n_compressed}-s-55ea3f9b",
            group="allmlp_compress_nsweep",
            tags=["mlp_compress", "all_mlp"],
            config=config,
        )
        print(f"wandb: {wb.url}")

    train_loader = build_lm_loader(
        pd_run.cfg.target,
        pd_run.cfg.data,
        split="train",
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    eval_loader = build_lm_loader(
        pd_run.cfg.target,
        pd_run.cfg.data,
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

    eval_ctx = []
    with torch.no_grad(), autocast():
        for b in eval_batches:
            ci, mask_infos, target_logits, teacher_logits = compute_ci_and_teacher(comp_model, b)
            eval_ctx.append((b, ci, mask_infos, target_logits, teacher_logits))

    with torch.no_grad(), autocast():
        kl_teacher_vs_target = 0.0
        kl_all_mlps_zeroed_vs_teacher = 0.0
        for b, ci, _mask_infos, target_logits, teacher_logits in eval_ctx:
            kl_teacher_vs_target += calc_kl_divergence_lm(
                pred=teacher_logits.float(), target=target_logits.float()
            ).item()
            zeroed_ci = dict(ci)
            for block in range(n_blocks):
                cfc_name, down_name = block_module_names(block)
                zeroed_ci[cfc_name] = torch.zeros_like(ci[cfc_name])
                zeroed_ci[down_name] = torch.zeros_like(ci[down_name])
            zeroed_logits = comp_model(b, mask_infos=make_mask_infos(zeroed_ci))
            kl_all_mlps_zeroed_vs_teacher += calc_kl_divergence_lm(
                pred=zeroed_logits.float(), target=teacher_logits.float()
            ).item()
        kl_teacher_vs_target /= len(eval_ctx)
        kl_all_mlps_zeroed_vs_teacher /= len(eval_ctx)
    references = {
        "kl_ci_masked_vs_target": kl_teacher_vs_target,
        "kl_all_mlps_zeroed_vs_teacher": kl_all_mlps_zeroed_vs_teacher,
    }
    print(f"references: {references}")
    (out_dir / "references.json").write_text(json.dumps(references, indent=2))
    if wb is not None:
        wb.summary.update(references)

    def run_eval() -> dict[str, float]:
        kl_vs_teacher = 0.0
        kl_vs_target = 0.0
        with torch.no_grad(), autocast():
            for b, ci, mask_infos, target_logits, teacher_logits in eval_ctx:
                student_logits = student_forward(
                    comp_model, b, ci, mask_infos, mlp_modules, compressed_by_block
                ).float()
                kl_vs_teacher += calc_kl_divergence_lm(
                    pred=student_logits, target=teacher_logits.float()
                ).item()
                kl_vs_target += calc_kl_divergence_lm(
                    pred=student_logits, target=target_logits.float()
                ).item()
        return {
            "eval/kl_student_vs_teacher": kl_vs_teacher / len(eval_ctx),
            "eval/kl_student_vs_target": kl_vs_target / len(eval_ctx),
        }

    metrics_path = out_dir / "metrics.jsonl"
    train_iter = iter(train_loader)
    last_log_time = time.time()
    for step in range(steps):
        batch = next(train_iter).to(device)

        with torch.no_grad(), autocast():
            ci, mask_infos, _, teacher_logits = compute_ci_and_teacher(comp_model, batch)
        with autocast():
            student_logits = student_forward(
                comp_model, batch, ci, mask_infos, mlp_modules, compressed_by_block
            )
        loss = calc_kl_divergence_lm(pred=student_logits.float(), target=teacher_logits.float())

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        record: dict[str, float] = {
            "step": step,
            "train/kl": loss.item(),
            "lr": float(scheduler.get_last_lr()[0]),
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
                {block: c.state_dict() for block, c in compressed_by_block.items()},
                out_dir / f"compressed_{step}.pt",
            )

    final = run_eval()
    print(f"final: {final} | references: {references}")
    (out_dir / "final.json").write_text(json.dumps(final | references, indent=2))
    if wb is not None:
        wb.summary.update(final)
        wb.finish()


if __name__ == "__main__":
    fire.Fire(main)
