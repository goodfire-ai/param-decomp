"""Fit a target component's activations as a function of source component activations.

Loads a trained SPD model, runs forward passes to collect component activations,
and fits a GeLU-gated linear model predicting a specific mlp.down_proj component
from the preceding mlp.c_fc components in the same layer.

The prediction is: sum over h of GeLU(source_acts * W_gelu[:, h] + b_gelu[:, h]) @ W_gelu_out[:, h]
  Each source component i has hidden_width GeLU neurons (default 1), each with its own
  weight W_gelu[i, h] and bias b_gelu[i, h]. The GeLU outputs are linearly combined.

Usage:
    python scripts/fit_component.py --config scripts/fit_component_config.yaml
"""

import argparse
import math
from pathlib import Path
from typing import Any

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.models.components import LinearComponents
from spd.utils.general_utils import bf16_autocast, extract_batch_data


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_module_path(layer_idx: int, module_suffix: str) -> str:
    return f"h.{layer_idx}.mlp.{module_suffix}"


@torch.no_grad()
def collect_component_acts(
    model: ComponentModel,
    batch: Tensor,
    source_path: str,
    target_path: str,
    target_component_idx: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run a forward pass and extract component activations.

    Computes the target using only the component weights (no delta):
        target = GeLU(source_acts @ U_c_fc + bias) @ V_down[:, target_component_idx]
    Also returns the actual target from the full forward pass for comparison.

    Returns:
        (source_acts, target_scalar, actual_target): all float tensors.
        source_acts is (B, S, C_source), target_scalar and actual_target are (B, S).
    """
    out: OutputWithCache = model(batch, cache_type="input")
    all_acts = model.get_all_component_acts(out.cache)

    source_acts = all_acts[source_path]  # (B, S, C_source)

    # Actual target from full forward pass (includes delta)
    target_acts = all_acts[target_path]  # (B, S, C_target)
    actual_target = target_acts[..., target_component_idx]  # (B, S)

    # Compute target from component weights only (excluding delta)
    source_comp = model.components[source_path]
    target_comp = model.components[target_path]
    assert isinstance(source_comp, LinearComponents) and isinstance(target_comp, LinearComponents)
    c_fc_out = einops.einsum(source_acts, source_comp.U, "... C, C d_out -> ... d_out")
    if source_comp.bias is not None:
        c_fc_out = c_fc_out + source_comp.bias
    gelu_out = F.gelu(c_fc_out)
    target_scalar = einops.einsum(
        gelu_out, target_comp.V[:, target_component_idx], "... d, d -> ..."
    )

    return source_acts.float(), target_scalar.float(), actual_target.float()


def compute_variance_explained(predictions: Tensor, targets: Tensor) -> float:
    ss_res = ((targets - predictions) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit component activations")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    wandb_path = cfg["wandb_path"]
    layer_idx: int = cfg["layer_idx"]
    target_component_idx: int = cfg["target_component_idx"]
    batch_size: int = cfg["batch_size"]
    n_steps: int = cfg["n_steps"]
    lr = float(cfg.get("lr", 1e-3))
    lr_decay_factor = float(cfg.get("lr_decay_factor", 1.0))
    hidden_width: int = cfg.get("hidden_width", 1)
    interact_width: int = cfg.get("interact_width", 8)
    n_eval_batches: int = cfg.get("n_eval_batches", 10)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    source_path = build_module_path(layer_idx, "c_fc")
    target_path = build_module_path(layer_idx, "down_proj")

    # --- Load model --- #
    logger.info(f"Loading model from {wandb_path}")
    run_info = SPDRunInfo.from_path(wandb_path)
    spd_config = run_info.config
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    model.eval()

    assert source_path in model.target_module_paths, (
        f"{source_path} not in model target modules: {model.target_module_paths}"
    )
    assert target_path in model.target_module_paths, (
        f"{target_path} not in model target modules: {model.target_module_paths}"
    )

    c_source = model.module_to_c[source_path]
    c_target = model.module_to_c[target_path]
    assert target_component_idx < c_target, (
        f"target_component_idx={target_component_idx} >= C_target={c_target}"
    )
    logger.info(
        f"Source: {source_path} (C={c_source}), "
        f"Target: {target_path}[{target_component_idx}] (C={c_target})"
    )

    # --- Load data --- #
    task_config = spd_config.task_config
    assert isinstance(task_config, LMTaskConfig)

    train_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=spd_config.tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=task_config.shuffle_each_epoch,
        seed=task_config.dataset_seed,
    )
    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=spd_config.seed,
    )
    train_iter = loop_dataloader(train_loader)

    eval_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=spd_config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
        seed=task_config.dataset_seed,
    )
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=batch_size,
        buffer_size=task_config.buffer_size,
        global_seed=spd_config.seed + 1,
    )
    eval_iter = loop_dataloader(eval_loader)

    # --- Set up fit parameters --- #
    # Kaiming normal: std = sqrt(2 / fan_in)
    W_skip = nn.Parameter(torch.randn(c_source, device=device) * math.sqrt(2.0 / c_source))
    bias = nn.Parameter(torch.zeros(1, device=device))
    params: list[nn.Parameter] = [bias, W_skip]
    if hidden_width > 0:
        # GeLU-gated linear: sum_h GeLU(source_acts * W_gelu[:, h] + b_gelu[:, h]) @ W_gelu_out[:, h]
        # W_gelu is a stack of c_source matrices each with fan_in=1
        W_gelu = nn.Parameter(torch.randn(c_source, hidden_width, device=device))
        b_gelu = nn.Parameter(torch.randn(c_source, hidden_width, device=device))
        # W_gelu_out is effectively one matrix with fan_in=c_source*hidden_width
        W_gelu_out = nn.Parameter(
            torch.randn(c_source, hidden_width, device=device)
            * math.sqrt(2.0 / (c_source * hidden_width))
        )
        params.extend([W_gelu, b_gelu, W_gelu_out])
    else:
        W_gelu = W_gelu_out = b_gelu = None
    if interact_width > 0:
        W_interact = nn.Parameter(
            torch.randn(c_source, interact_width, device=device) * math.sqrt(2.0 / c_source)
        )
        W_interact_out = nn.Parameter(
            torch.randn(interact_width, device=device) * math.sqrt(2.0 / interact_width)
        )
        b_interact = nn.Parameter(torch.zeros(interact_width, device=device))
        params.extend([W_interact, W_interact_out, b_interact])
    else:
        W_interact = W_interact_out = b_interact = None

    optimizer = torch.optim.AdamW(params, lr=lr)

    # Cosine decay: lr -> lr * lr_decay_factor over training
    lr_final = lr * lr_decay_factor
    use_lr_schedule = lr_decay_factor < 1.0

    def get_lr(step: int) -> float:
        if not use_lr_schedule:
            return lr
        progress = step / max(n_steps - 1, 1)
        # Cosine decay from lr to lr_final
        return lr_final + 0.5 * (lr - lr_final) * (1.0 + math.cos(math.pi * progress))

    def predict(source_acts: Tensor) -> Tensor:
        out = einops.einsum(source_acts, W_skip, "B S C, C -> B S") + bias  # (B, S)
        if W_gelu is not None:
            assert W_gelu_out is not None and b_gelu is not None
            expanded = source_acts.unsqueeze(-1) * W_gelu + b_gelu  # (B, S, C, H)
            gelu_out = F.gelu(expanded)  # (B, S, C, H)
            out = out + (gelu_out * W_gelu_out).sum(dim=(-2, -1))
        if W_interact is not None:
            assert W_interact_out is not None and b_interact is not None
            interact_out = F.gelu(
                einops.einsum(source_acts, W_interact, "B S C, C I -> B S I") + b_interact
            )  # (B, S, I)
            out = out + einops.einsum(interact_out, W_interact_out, "B S I, I -> B S")
        return out

    # --- Training loop --- #
    lr_str = f", cosine LR {lr} -> {lr_final}" if use_lr_schedule else ""
    mode_str = f"GeLU-gated linear (H={hidden_width})" + lr_str
    logger.info(f"Training {mode_str} fit for {n_steps} steps...")

    # Log metrics at initialisation (before any training)
    with torch.no_grad():
        init_batch = extract_batch_data(next(train_iter)).to(device, non_blocking=True)
        with bf16_autocast(enabled=spd_config.autocast_bf16):
            init_source, init_target, _ = collect_component_acts(
                model, init_batch, source_path, target_path, target_component_idx
            )
        init_pred = predict(init_source)
        init_mse = ((init_pred - init_target) ** 2).mean().item()
        init_r2 = compute_variance_explained(init_pred, init_target)
        logger.info(f"Init: MSE={init_mse:.6f} R²={init_r2:.4f}")

    last_train_mse = 0.0
    last_train_var_explained = 0.0

    # Accumulate SS_res and SS_tot over a window for smoothed R²
    window_ss_res = 0.0
    window_ss_tot = 0.0
    window_mse_sum = 0.0
    window_count = 0

    for step in range(n_steps):
        if use_lr_schedule:
            current_lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

        batch = extract_batch_data(next(train_iter)).to(device, non_blocking=True)

        with bf16_autocast(enabled=spd_config.autocast_bf16):
            source_acts, target_scalar, _ = collect_component_acts(
                model, batch, source_path, target_path, target_component_idx
            )

        prediction = predict(source_acts)
        mse_loss = ((prediction - target_scalar) ** 2).mean()

        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_detached = prediction.detach()
            window_ss_res += ((target_scalar - pred_detached) ** 2).sum().item()
            window_ss_tot += ((target_scalar - target_scalar.mean()) ** 2).sum().item()
            window_mse_sum += mse_loss.item()
            window_count += 1

        if step % 1000 == 0 or step == n_steps - 1:
            window_r2 = 1.0 - window_ss_res / window_ss_tot if window_ss_tot > 0 else 0.0
            window_mse = window_mse_sum / window_count
            logger.info(f"Step {step:>6d}/{n_steps}: MSE={window_mse:.6f} R²={window_r2:.4f}")

            last_train_mse = window_mse
            last_train_var_explained = window_r2

            # Reset window
            window_ss_res = 0.0
            window_ss_tot = 0.0
            window_mse_sum = 0.0
            window_count = 0

    # --- Evaluation --- #
    logger.info(f"Evaluating on {n_eval_batches} val batches...")
    eval_mse_sum = 0.0
    eval_ss_res = 0.0
    eval_ss_tot = 0.0
    # Metrics vs actual target (includes delta)
    eval_actual_mse_sum = 0.0
    eval_actual_ss_res = 0.0
    eval_actual_ss_tot = 0.0
    eval_count = 0

    with torch.no_grad():
        for _ in range(n_eval_batches):
            batch = extract_batch_data(next(eval_iter)).to(device, non_blocking=True)

            with bf16_autocast(enabled=spd_config.autocast_bf16):
                source_acts, target_scalar, actual_target = collect_component_acts(
                    model, batch, source_path, target_path, target_component_idx
                )

            prediction = predict(source_acts)
            mse = ((prediction - target_scalar) ** 2).mean()
            eval_mse_sum += mse.item()

            eval_ss_res += ((target_scalar - prediction) ** 2).sum().item()
            eval_ss_tot += ((target_scalar - target_scalar.mean()) ** 2).sum().item()

            # Vs actual target
            eval_actual_mse_sum += ((prediction - actual_target) ** 2).mean().item()
            eval_actual_ss_res += ((actual_target - prediction) ** 2).sum().item()
            eval_actual_ss_tot += ((actual_target - actual_target.mean()) ** 2).sum().item()

            eval_count += 1

    eval_mse = eval_mse_sum / eval_count
    eval_var_explained = 1.0 - eval_ss_res / eval_ss_tot if eval_ss_tot > 0 else 0.0
    eval_actual_mse = eval_actual_mse_sum / eval_count
    eval_actual_var_explained = (
        1.0 - eval_actual_ss_res / eval_actual_ss_tot if eval_actual_ss_tot > 0 else 0.0
    )

    # --- Results --- #
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Target: {target_path} component {target_component_idx}")
    print(f"Source: {source_path} ({c_source} components)")
    print(f"Mode: {mode_str}")
    print()
    print(f"Last training step:  MSE={last_train_mse:.6f}, R²={last_train_var_explained:.4f}")
    print(f"Evaluation ({eval_count} batches): MSE={eval_mse:.6f}, R²={eval_var_explained:.4f}")
    print(
        f"Eval vs actual target:       MSE={eval_actual_mse:.6f}, R²={eval_actual_var_explained:.4f}"
    )
    print(f"\nBias: {bias.item():.6f}")

    # --- Test prompt visualization --- #
    test_prompt: str | None = cfg.get("test_prompt")
    if test_prompt is not None:
        assert spd_config.tokenizer_name is not None
        tokenizer = AutoTokenizer.from_pretrained(spd_config.tokenizer_name)
        token_ids = torch.tensor(
            [tokenizer.encode(test_prompt)],
            device=device,
        )
        tokens: list[str] = [
            tokenizer.decode(t)  # pyright: ignore[reportAttributeAccessIssue]
            for t in token_ids[0].tolist()
        ]
        seq_len = token_ids.shape[1]

        with torch.no_grad(), bf16_autocast(enabled=spd_config.autocast_bf16):
            source_acts, target_scalar, actual_scalar = collect_component_acts(
                model, token_ids, source_path, target_path, target_component_idx
            )
        with torch.no_grad():
            pred_scalar = predict(source_acts)

        target_vals = target_scalar[0].cpu().tolist()
        actual_vals = actual_scalar[0].cpu().tolist()
        pred_vals = pred_scalar[0].cpu().tolist()

        print("\n" + "=" * 80)
        print("TEST PROMPT VISUALIZATION")
        print("=" * 80)
        print(f"Prompt: {test_prompt!r}")
        print(
            f"{'Pos':>4}  {'Token':<20}  {'Target':>12}  {'Actual':>12}  {'Predicted':>12}  {'Error':>12}"
        )
        print("-" * 80)
        for pos in range(seq_len):
            target_v = target_vals[pos]
            actual_v = actual_vals[pos]
            pred_v = pred_vals[pos]
            err = pred_v - target_v
            print(
                f"{pos:>4}  {tokens[pos]:<20}  {target_v:>12.6f}  {actual_v:>12.6f}"
                f"  {pred_v:>12.6f}  {err:>+12.6f}"
            )

        # Plot
        fig, ax = plt.subplots(figsize=(max(12, seq_len * 0.5), 5))
        positions = list(range(seq_len))
        ax.plot(positions, target_vals, "o-", label="Target (no delta)", markersize=5)
        ax.plot(positions, actual_vals, "^:", label="Actual (with delta)", markersize=5)
        ax.plot(positions, pred_vals, "s--", label="Predicted", markersize=5)
        ax.set_xticks(positions)
        ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=8)
        ax.set_xlabel("Token position")
        ax.set_ylabel("Component activation")
        ax.set_title(
            f"Component {target_component_idx} ({target_path}): "
            f"target vs predicted (R²={eval_var_explained:.4f})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = Path(f"fit_component_{target_component_idx}_test_prompt.png")
        fig.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
