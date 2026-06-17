"""Same attention+MLP replacement as `run_attn.py`, but trained on residual-stream MSE.

Instead of distilling logits via KL, this trains the replacements to reconstruct the teacher's
residual stream at 8 points — the activations after each attention sublayer and after each MLP
sublayer, across all 4 blocks (post-attn residual = input to `rms_2`; post-mlp residual = block
output). The training loss is the *sum* of those 8 per-point *relative* MSEs (each point's
squared error normalized by that point's teacher squared norm, so the 8 points are weighted
evenly) on the CI-masked (teacher) forward. KL(student‖teacher) and KL(student‖target) are
still reported as eval metrics.

All decomposition components, the CI function, and the CI masks stay frozen; only the small
attention (fewer heads) and MLP (n_compressed bottleneck) transformations train.

Run: python scratch/mlp0_compress/run_mse.py --n_heads H --n_compressed N [--steps ...] ...
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import fire
import torch
import wandb
from dotenv import load_dotenv
from torch import Tensor, nn

from param_decomp.component_model import ComponentModel
from param_decomp.components import LinearComponents
from param_decomp_lab.batch_and_loss_fns import calc_kl_divergence_lm
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT
from run import CompressedMaskedMLP, RUN_DIR, compute_ci_and_teacher
from run_attn import (
    ORIG_HEAD_DIM,
    ORIG_N_HEADS,
    CompressedMaskedAttention,
    attn_module_names,
    blocks_replaced,
    mlp_module_names,
    student_forward,
)

OUT_BASE = PARAM_DECOMP_OUT_DIR / "runs/s-55ea3f9b/attn_mlp_relmse"


L0_THRESHOLD = 0.0


def relative_mse(student: Tensor, teacher: Tensor) -> Tensor:
    """Scale-invariant per-point MSE: ||student - teacher||^2 / ||teacher||^2."""
    return (student - teacher).pow(2).mean() / teacher.pow(2).mean()


def lp_sparsity_penalty(hidden: Tensor, p: float) -> Tensor:
    """Mean over tokens of the L_p norm of the neuron-activation vector: (Σ_n |h_n|^p)^(1/p).

    The `eps` inside the base keeps the gradient finite where activations hit exactly 0 —
    at p < 1 the bare `|h|^(p-1)` term diverges there (and ReLU sends every off-neuron to
    exact 0), which otherwise produces NaN gradients.
    """
    eps = 1e-6
    return (hidden.abs() + eps).pow(p).sum(dim=-1).pow(1.0 / p).mean()


def l0_per_token(hidden: Tensor, threshold: float) -> float:
    """Mean over tokens of the count of neuron activations exceeding `threshold` in magnitude."""
    return (hidden.abs() > threshold).float().sum(dim=-1).mean().item()


def resid_point_names(n_blocks: int) -> list[str]:
    names = []
    for b in range(n_blocks):
        names.append(f"post_attn_{b}")
        names.append(f"post_mlp_{b}")
    return names


@contextmanager
def capture_residuals(comp_model: ComponentModel, n_blocks: int) -> Iterator[dict[str, Tensor]]:
    """Capture the residual stream after each attn and mlp sublayer of every block.

    post_attn_b = input to `h.b.rms_2` (residual after the attention add).
    post_mlp_b  = output of block `h.b` (residual after the MLP add).
    """
    resids: dict[str, Tensor] = {}
    handles = []
    for b in range(n_blocks):
        block = comp_model.target_model.get_submodule(f"h.{b}")
        rms_2 = comp_model.target_model.get_submodule(f"h.{b}.rms_2")

        def make_pre(key: str):
            def hook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
                resids[key] = args[0]

            return hook

        def make_post(key: str):
            def hook(_module: nn.Module, _args: tuple[Tensor, ...], output: Tensor) -> None:
                resids[key] = output

            return hook

        handles.append(rms_2.register_forward_pre_hook(make_pre(f"post_attn_{b}")))
        handles.append(block.register_forward_hook(make_post(f"post_mlp_{b}")))
    try:
        yield resids
    finally:
        for handle in handles:
            handle.remove()


def main(
    n_heads: int,
    head_dim: int,
    n_compressed: int,
    sparsity_coeff: float,
    bypass: bool,
    p_norm: float = 0.9,
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

    bypass_tag = "bp1" if bypass else "bp0"
    out_dir = (
        OUT_BASE
        / f"relu_h{n_heads}_d{head_dim}_n{n_compressed}_sp{sparsity_coeff:g}_p{p_norm:g}_{bypass_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    pd_run = SavedLMRun.from_path(RUN_DIR)
    comp_model = pd_run.load_model().to(device)
    comp_model.eval()
    comp_model.requires_grad_(False)

    attn0 = comp_model.target_model.get_submodule("h.0.attn")
    assert attn0.rotary_adjacent_pairs is False, "compressed attn uses half-split rotate_half"
    rotary_sin, rotary_cos = attn0.calculate_sin_cos_rotary(
        head_dim, attn0.n_ctx, base=attn0.rotary_base
    )
    rotary_cos, rotary_sin = rotary_cos.to(device), rotary_sin.to(device)

    n_blocks = 0
    while mlp_module_names(n_blocks)[0] in comp_model.components:
        n_blocks += 1
    assert n_blocks == 4, f"expected 4 blocks, found {n_blocks}"

    mlp_modules: dict[int, nn.Module] = {}
    attn_modules: dict[int, nn.Module] = {}
    compressed_mlp: dict[int, CompressedMaskedMLP] = {}
    compressed_attn: dict[int, CompressedMaskedAttention] = {}
    for block in range(n_blocks):
        cfc_name, down_name = mlp_module_names(block)
        cfc = comp_model.components[cfc_name]
        down = comp_model.components[down_name]
        assert isinstance(cfc, LinearComponents) and isinstance(down, LinearComponents)
        assert cfc.bias is None and down.bias is None, "target MLP is bias-free"
        assert cfc.d_out == down.d_in == 3072
        mlp = comp_model.target_model.get_submodule(f"h.{block}.mlp")
        mlp_modules[block] = mlp
        compressed_mlp[block] = CompressedMaskedMLP(
            V_cfc=cfc.V.data,
            U_down=down.U.data,
            n_compressed=n_compressed,
            activation=nn.ReLU(),
            bypass=bypass,
        ).to(device)

        q_name, k_name, v_name, o_name = attn_module_names(block)
        q, k, v, o = (comp_model.components[n] for n in (q_name, k_name, v_name, o_name))
        assert all(isinstance(c, LinearComponents) for c in (q, k, v, o))
        assert q.bias is None and o.bias is None, "target attn is bias-free"
        assert q.d_in == k.d_in == v.d_in == 768 and o.d_out == 768
        attn = comp_model.target_model.get_submodule(f"h.{block}.attn")
        assert attn.head_dim == ORIG_HEAD_DIM
        compressed_attn[block] = CompressedMaskedAttention(
            V_q=q.V.data, V_k=k.V.data, V_v=v.V.data, U_o=o.U.data,
            n_heads=n_heads, head_dim=head_dim, rotary_cos=rotary_cos, rotary_sin=rotary_sin,
        ).to(device)
        attn_modules[block] = attn

    params = [
        p for c in (*compressed_mlp.values(), *compressed_attn.values()) for p in c.parameters()
    ]
    n_trainable = sum(p.numel() for p in params if p.requires_grad)
    point_names = resid_point_names(n_blocks)
    config = {
        "run": "s-55ea3f9b (via p-55ea3f9b)",
        "objective": "residual_stream_relative_mse + Lp_sparsity",
        "n_resid_points": len(point_names),
        "n_blocks": n_blocks,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "n_compressed_mlp": n_compressed,
        "mlp_activation": "relu",
        "linear_bypass": bypass,
        "sparsity_coeff": sparsity_coeff,
        "p_norm": p_norm,
        "l0_threshold": L0_THRESHOLD,
        "orig_n_heads": ORIG_N_HEADS,
        "orig_head_dim": ORIG_HEAD_DIM,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "final_lr_frac": final_lr_frac,
        "seed": seed,
        "n_trainable_params": n_trainable,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"out_dir: {out_dir}")
    print(f"config: {json.dumps(config, indent=2)}")

    wb = None
    if use_wandb:
        wb = wandb.init(
            project="spd",
            name=f"relmse-relu-{bypass_tag}-sp{sparsity_coeff:g}-attn{n_heads}h-d{head_dim}-mlp{n_compressed}-s-55ea3f9b",
            group="attn_mlp_relmse_sparsity_relu_bypass_sweep" if bypass else "attn_mlp_relmse_sparsity_relu_sweep",
            tags=["mlp_compress", "attn_compress", "resid_relmse", "lp_sparsity", "relu"]
            + (["bypass"] if bypass else []),
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
            with capture_residuals(comp_model, n_blocks) as tr:
                ci, mask_infos, target_logits, teacher_logits = compute_ci_and_teacher(comp_model, b)
            teacher_resids = {k: v.detach().clone() for k, v in tr.items()}
            assert set(teacher_resids) == set(point_names)
            eval_ctx.append((b, ci, mask_infos, target_logits, teacher_logits, teacher_resids))

    with torch.no_grad(), autocast():
        kl_teacher_vs_target = sum(
            calc_kl_divergence_lm(pred=tl.float(), target=gl.float()).item()
            for _b, _ci, _mi, gl, tl, _tr in eval_ctx
        ) / len(eval_ctx)
    references = {"kl_ci_masked_vs_target": kl_teacher_vs_target}
    print(f"references: {references}")
    (out_dir / "references.json").write_text(json.dumps(references, indent=2))
    if wb is not None:
        wb.summary.update(references)

    def run_eval() -> dict[str, float]:
        kl_vs_teacher = 0.0
        kl_vs_target = 0.0
        relmse_total = 0.0
        sparsity_total = 0.0
        l0_total = 0.0
        with torch.no_grad(), autocast():
            for b, ci, mask_infos, target_logits, teacher_logits, teacher_resids in eval_ctx:
                with capture_residuals(comp_model, n_blocks) as sr:
                    student_logits = student_forward(
                        comp_model, b, ci, mask_infos, mlp_modules, attn_modules,
                        compressed_mlp, compressed_attn,
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
                hiddens = [compressed_mlp[blk].last_hidden.float() for blk in range(n_blocks)]
                sparsity_total += sum(lp_sparsity_penalty(h, p_norm).item() for h in hiddens)
                l0_total += sum(l0_per_token(h, L0_THRESHOLD) for h in hiddens) / n_blocks
        n = len(eval_ctx)
        l0_mean = l0_total / n
        return {
            "eval/kl_student_vs_teacher": kl_vs_teacher / n,
            "eval/kl_student_vs_target": kl_vs_target / n,
            "eval/resid_relmse_total": relmse_total / n,
            "eval/lp_sparsity_total": sparsity_total / n,
            "eval/l0_per_token": l0_mean,
            "eval/l0_frac": l0_mean / n_compressed if n_compressed > 0 else 0.0,
        }

    metrics_path = out_dir / "metrics.jsonl"
    train_iter = iter(train_loader)
    last_log_time = time.time()
    for step in range(steps):
        batch = next(train_iter).to(device)

        with torch.no_grad(), autocast():
            with capture_residuals(comp_model, n_blocks) as tr:
                ci, mask_infos, _, _ = compute_ci_and_teacher(comp_model, batch)
            teacher_resids = {k: v.detach() for k, v in tr.items()}
        with autocast(), capture_residuals(comp_model, n_blocks) as sr:
            student_forward(
                comp_model, batch, ci, mask_infos, mlp_modules, attn_modules,
                compressed_mlp, compressed_attn,
            )
        relmse_terms = {
            k: relative_mse(sr[k].float(), teacher_resids[k].float()) for k in point_names
        }
        relmse_total = sum(relmse_terms.values())
        sparsity_terms = {
            blk: lp_sparsity_penalty(compressed_mlp[blk].last_hidden.float(), p_norm)
            for blk in range(n_blocks)
        }
        sparsity_total = sum(sparsity_terms.values())
        loss = relmse_total + sparsity_coeff * sparsity_total

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        record: dict[str, float] = {
            "step": step,
            "train/loss": loss.item(),
            "train/resid_relmse_total": relmse_total.item(),
            "train/lp_sparsity_total": sparsity_total.item(),
            "lr": scheduler.get_last_lr()[0],
        }
        for k, v in relmse_terms.items():
            record[f"train/relmse_{k}"] = v.item()
        if step % eval_every == 0 or step == steps - 1:
            record |= run_eval()
            now = time.time()
            record["steps_per_s"] = eval_every / (now - last_log_time)
            last_log_time = now
            print(json.dumps({k: record[k] for k in record if not k.startswith("train/relmse_")}))
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        if wb is not None:
            wb.log(record, step=step)

        if (step > 0 and step % save_every == 0) or step == steps - 1:
            torch.save(
                {
                    "mlp": {b: c.state_dict() for b, c in compressed_mlp.items()},
                    "attn": {b: c.state_dict() for b, c in compressed_attn.items()},
                },
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
