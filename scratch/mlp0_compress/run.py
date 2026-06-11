"""Distill block-0 MLP of s-55ea3f9b to a 10x smaller neuron dim on CI-masked forwards.

Teacher: the conventional CI-masked forward (`make_mask_infos(ci.lower_leaky)`, no weight
delta — same convention as the `kl_ci_masked` eval). Student: the identical masked forward,
except block-0's MLP output is recomputed through a compressed bottleneck — frozen `V_cfc`
(768xC_cfc) and `U_down` (C_down x768) from the decomposition, with new trainable
`U'` (C_cfc x n_compressed) and `V'` (n_compressed x C_down), GELU in between. Block-0 CI
masks are reused from the teacher pass (the CI fn reads 3072-dim `down_proj` pre-acts,
which don't exist in the compressed pass). Loss: KL(teacher || student) on output logits.

Run: python scratch/mlp0_compress/run.py [--steps N] [--batch_size B] ...
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import einops
import fire
import torch
import wandb
from dotenv import load_dotenv
from torch import Tensor, nn

from param_decomp.component_model import ComponentModel
from param_decomp.components import LinearComponents, init_param_
from param_decomp.masks import make_mask_infos
from param_decomp_lab.batch_and_loss_fns import calc_kl_divergence_lm
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT

RUN_DIR = PARAM_DECOMP_OUT_DIR / "runs/p-55ea3f9b"  # current-schema port of s-55ea3f9b
OUT_BASE = PARAM_DECOMP_OUT_DIR / "runs/s-55ea3f9b/mlp0_compress"
CFC = "h.0.mlp.c_fc"
DOWN = "h.0.mlp.down_proj"


class CompressedMaskedMLP(nn.Module):
    """Block-0 MLP with the neuron dimension compressed, operating in CI-mask space.

    Forward: `x @ V_cfc * ci_cfc @ U_compressed -> GELU -> @ V_compressed * ci_down @ U_down`.
    `V_cfc` / `U_down` are frozen buffers copied from the decomposition; only
    `U_compressed` / `V_compressed` train. The target MLP has no biases.
    """

    V_cfc: Tensor
    U_down: Tensor

    def __init__(
        self,
        V_cfc: Tensor,
        U_down: Tensor,
        n_compressed: int,
        gelu: nn.Module,
    ):
        super().__init__()
        self.register_buffer("V_cfc", V_cfc.clone())
        self.register_buffer("U_down", U_down.clone())
        C_cfc = V_cfc.shape[1]
        C_down = U_down.shape[0]
        self.U_compressed = nn.Parameter(torch.empty(C_cfc, n_compressed))
        self.V_compressed = nn.Parameter(torch.empty(n_compressed, C_down))
        init_param_(self.U_compressed, fan_val=C_cfc, nonlinearity="linear")
        init_param_(self.V_compressed, fan_val=n_compressed, nonlinearity="linear")
        self.gelu = gelu

    def forward(
        self,
        x: Tensor,
        ci_cfc: Tensor,
        ci_down: Tensor,
    ) -> Tensor:
        comp_acts_cfc = einops.einsum(x, self.V_cfc, "... d_in, d_in C -> ... C") * ci_cfc
        hidden = self.gelu(
            einops.einsum(comp_acts_cfc, self.U_compressed, "... C, C n -> ... n")
        )
        comp_acts_down = (
            einops.einsum(hidden, self.V_compressed, "... n, n C -> ... C") * ci_down
        )
        return einops.einsum(comp_acts_down, self.U_down, "... C, C d_out -> ... d_out")


@contextmanager
def mlp0_replaced(
    mlp0: nn.Module, compressed: CompressedMaskedMLP, ci_holder: dict[str, Tensor]
) -> Iterator[None]:
    def hook(_module: nn.Module, args: tuple[Tensor, ...], _output: Tensor) -> Tensor:
        return compressed(args[0], ci_holder[CFC], ci_holder[DOWN])

    handle = mlp0.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def compute_ci_and_teacher(
    comp_model: ComponentModel, batch: Tensor
) -> tuple[dict[str, Tensor], dict[str, object], Tensor, Tensor]:
    """Target forward -> CI -> conventional CI-masked forward.

    Returns (ci_lower_leaky, mask_infos, target_logits, teacher_logits).
    """
    out = comp_model(batch, cache_type="input")
    ci = comp_model.calc_causal_importances(out.cache, sampling="continuous").lower_leaky
    mask_infos = make_mask_infos(ci)
    teacher_logits = comp_model(batch, mask_infos=mask_infos)
    return ci, mask_infos, out.output, teacher_logits


def student_forward(
    comp_model: ComponentModel,
    batch: Tensor,
    ci: dict[str, Tensor],
    mask_infos: dict[str, object],
    mlp0: nn.Module,
    compressed: CompressedMaskedMLP,
) -> Tensor:
    student_mask_infos = {k: v for k, v in mask_infos.items() if k not in (CFC, DOWN)}
    ci_holder = {CFC: ci[CFC], DOWN: ci[DOWN]}
    with mlp0_replaced(mlp0, compressed, ci_holder):
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

    cfc = comp_model.components[CFC]
    down = comp_model.components[DOWN]
    assert isinstance(cfc, LinearComponents) and isinstance(down, LinearComponents)
    assert cfc.bias is None and down.bias is None, "target MLP is bias-free"
    d_mlp = cfc.d_out
    assert d_mlp == down.d_in == 3072
    assert 0 < n_compressed <= d_mlp

    mlp0 = comp_model.target_model.get_submodule("h.0.mlp")
    compressed = CompressedMaskedMLP(
        V_cfc=cfc.V.data, U_down=down.U.data, n_compressed=n_compressed, gelu=mlp0.gelu
    ).to(device)

    n_trainable = sum(p.numel() for p in compressed.parameters() if p.requires_grad)
    config = {
        "run": "s-55ea3f9b (via p-55ea3f9b)",
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "final_lr_frac": final_lr_frac,
        "seed": seed,
        "d_mlp": d_mlp,
        "n_compressed": n_compressed,
        "C_cfc": cfc.C,
        "C_down": down.C,
        "n_trainable_params": n_trainable,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"out_dir: {out_dir}")
    print(f"config: {json.dumps(config, indent=2)}")

    wb = None
    if use_wandb:
        wb = wandb.init(
            project="spd",
            name=f"compress-mlp0-n{n_compressed}-s-55ea3f9b",
            group="mlp0_compress_nsweep",
            tags=["mlp0_compress"],
            config=config,
        )
        print(f"wandb: {wb.url}")

    train_loader = build_lm_loader(
        pd_run.cfg.target, pd_run.cfg.data, split="train", device=device,
        batch_size=batch_size, seed=seed,
    )
    eval_loader = build_lm_loader(
        pd_run.cfg.target, pd_run.cfg.data, split="eval", device=device,
        batch_size=batch_size, seed=seed,
    )
    eval_iter = iter(eval_loader)
    eval_batches = [next(eval_iter).to(device) for _ in range(n_eval_batches)]

    opt = torch.optim.Adam(compressed.parameters(), lr=lr)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        cos = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return final_lr_frac + (1 - final_lr_frac) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Fixed eval context: CI, masks, target + teacher logits per eval batch (CI fn is
    # deterministic given the batch, so compute once).
    eval_ctx = []
    with torch.no_grad(), autocast():
        for b in eval_batches:
            ci, mask_infos, target_logits, teacher_logits = compute_ci_and_teacher(comp_model, b)
            eval_ctx.append((b, ci, mask_infos, target_logits, teacher_logits))

    # Reference points (computed once).
    with torch.no_grad(), autocast():
        kl_teacher_vs_target = 0.0
        kl_mlp0_zeroed_vs_teacher = 0.0
        for b, ci, mask_infos, target_logits, teacher_logits in eval_ctx:
            kl_teacher_vs_target += calc_kl_divergence_lm(
                pred=teacher_logits.float(), target=target_logits.float()
            ).item()
            zeroed_ci = dict(ci) | {CFC: torch.zeros_like(ci[CFC]), DOWN: torch.zeros_like(ci[DOWN])}
            zeroed_logits = comp_model(b, mask_infos=make_mask_infos(zeroed_ci))
            kl_mlp0_zeroed_vs_teacher += calc_kl_divergence_lm(
                pred=zeroed_logits.float(), target=teacher_logits.float()
            ).item()
        kl_teacher_vs_target /= len(eval_ctx)
        kl_mlp0_zeroed_vs_teacher /= len(eval_ctx)
    references = {
        "kl_ci_masked_vs_target": kl_teacher_vs_target,
        "kl_mlp0_zeroed_vs_teacher": kl_mlp0_zeroed_vs_teacher,
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
                    comp_model, b, ci, mask_infos, mlp0, compressed
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
            student_logits = student_forward(comp_model, batch, ci, mask_infos, mlp0, compressed)
        loss = calc_kl_divergence_lm(pred=student_logits.float(), target=teacher_logits.float())

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        record: dict[str, float] = {
            "step": step,
            "train/kl": loss.item(),
            "lr": scheduler.get_last_lr()[0],
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
            torch.save(compressed.state_dict(), out_dir / f"compressed_{step}.pt")

    final = run_eval()
    print(f"final: {final} | references: {references}")
    (out_dir / "final.json").write_text(json.dumps(final | references, indent=2))
    if wb is not None:
        wb.summary.update(final)
        wb.finish()


if __name__ == "__main__":
    fire.Fire(main)
