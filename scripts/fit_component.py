"""Fit a target component's activations as a function of source component activations.

Loads a trained SPD model, runs forward passes to collect component activations,
and fits a linear + nonlinear MLP model predicting a specific mlp.down_proj component
from the preceding mlp.c_fc components in the same layer.

The prediction is: linear_term + mlp_term + bias
  linear_term = source_acts @ W_linear
  mlp_term = act_fn(source_acts @ W_hidden + b_hidden) @ W_out

Usage:
    python scripts/fit_component.py --config scripts/fit_component_config.yaml
"""

import argparse
import math
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader, loop_dataloader
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
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
) -> tuple[Tensor, Tensor]:
    """Run a forward pass and extract component activations.

    Returns:
        (source_acts, target_scalar): source_acts is (B, S, C_source),
        target_scalar is (B, S).
    """
    out: OutputWithCache = model(batch, cache_type="input")
    all_acts = model.get_all_component_acts(out.cache)

    source_acts = all_acts[source_path]  # (B, S, C_source)
    target_acts = all_acts[target_path]  # (B, S, C_target)
    target_scalar = target_acts[..., target_component_idx]  # (B, S)

    return source_acts.float(), target_scalar.float()


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
    lr: float = cfg.get("lr", 1e-3)
    lr_decay_factor: float = cfg.get("lr_decay_factor", 1.0)
    mlp_width: int = cfg.get("mlp_width", 0)
    act_fn_name: Literal["relu", "gelu"] = cfg.get("act_fn", "relu")
    sparsity_coeff: float = cfg.get("sparsity_coeff", 0.0)
    sparsity_p: float = cfg.get("sparsity_p", 1.0)
    l0_threshold: float = cfg.get("l0_threshold", 1e-4)
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
    act_fn = F.gelu if act_fn_name == "gelu" else F.relu

    # Linear term: source_acts @ W_linear
    W_linear = nn.Parameter(torch.zeros(c_source, device=device))
    bias = nn.Parameter(torch.zeros(1, device=device))
    params: list[nn.Parameter] = [W_linear, bias]

    # MLP term: act_fn(source_acts @ W_hidden + b_hidden) @ W_out
    has_mlp = mlp_width > 0
    W_hidden = nn.Parameter(torch.randn(c_source, mlp_width, device=device) * 0.01)
    b_hidden = nn.Parameter(torch.randn(mlp_width, device=device) * 0.01)
    W_out = nn.Parameter(torch.randn(mlp_width, device=device) * 0.01)
    if has_mlp:
        params.extend([W_hidden, b_hidden, W_out])

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
        pred = source_acts @ W_linear + bias
        if has_mlp:
            hidden = act_fn(source_acts @ W_hidden + b_hidden)  # (B, S, mlp_width)
            pred = pred + hidden @ W_out  # (B, S)
        return pred

    use_sparsity = has_mlp and sparsity_coeff > 0.0

    def calc_sparsity_loss() -> Tensor:
        """Penalise effective magnitude of each MLP weight/bias scaled by its W_out."""
        # W_hidden: (C_source, mlp_width), W_out: (mlp_width,)
        w_penalty = (W_hidden.abs() * W_out.abs().unsqueeze(0)).pow(sparsity_p).sum()
        b_penalty = (b_hidden.abs() * W_out.abs()).pow(sparsity_p).sum()
        return w_penalty + b_penalty

    def calc_mlp_l0() -> int:
        """Count weights whose effective magnitude |W_hidden[i,j] * W_out[j]| exceeds threshold.

        Also counts biases: |b_hidden[j] * W_out[j]|.
        """
        w_effective = W_hidden.detach().abs() * W_out.detach().abs().unsqueeze(0)
        b_effective = b_hidden.detach().abs() * W_out.detach().abs()
        return int(
            (w_effective > l0_threshold).sum().item() + (b_effective > l0_threshold).sum().item()
        )

    # --- Training loop --- #
    mlp_str = f"linear + {act_fn_name.upper()} MLP (width={mlp_width})" if has_mlp else "linear"
    lr_str = f", cosine LR {lr} -> {lr_final}" if use_lr_schedule else ""
    sparse_str = f", sparsity={sparsity_coeff} p={sparsity_p}" if use_sparsity else ""
    mode_str = mlp_str + lr_str + sparse_str
    logger.info(f"Training {mode_str} fit for {n_steps} steps...")
    last_train_mse = 0.0
    last_train_var_explained = 0.0
    last_train_sparsity = 0.0

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
            source_acts, target_scalar = collect_component_acts(
                model, batch, source_path, target_path, target_component_idx
            )

        prediction = predict(source_acts)
        mse_loss = ((prediction - target_scalar) ** 2).mean()

        loss = mse_loss
        if use_sparsity:
            s_loss = calc_sparsity_loss()
            loss = loss + sparsity_coeff * s_loss

        optimizer.zero_grad()
        loss.backward()
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
            log_parts = [
                f"Step {step:>6d}/{n_steps}:",
                f"MSE={window_mse:.6f}",
                f"R²={window_r2:.4f}",
            ]
            if use_sparsity:
                log_parts.append(f"sparse={s_loss.item():.6f}")  # pyright: ignore[reportPossiblyUnboundVariable]
                log_parts.append(f"L0={calc_mlp_l0()}")
            logger.info(" ".join(log_parts))

            # Save last window stats before resetting
            last_train_mse = window_mse
            last_train_var_explained = window_r2
            if use_sparsity:
                last_train_sparsity = s_loss.item()  # pyright: ignore[reportPossiblyUnboundVariable]

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
    eval_count = 0

    with torch.no_grad():
        for _ in range(n_eval_batches):
            batch = extract_batch_data(next(eval_iter)).to(device, non_blocking=True)

            with bf16_autocast(enabled=spd_config.autocast_bf16):
                source_acts, target_scalar = collect_component_acts(
                    model, batch, source_path, target_path, target_component_idx
                )

            prediction = predict(source_acts)
            mse = ((prediction - target_scalar) ** 2).mean()
            eval_mse_sum += mse.item()

            eval_ss_res += ((target_scalar - prediction) ** 2).sum().item()
            eval_ss_tot += ((target_scalar - target_scalar.mean()) ** 2).sum().item()
            eval_count += 1

    eval_mse = eval_mse_sum / eval_count
    eval_var_explained = 1.0 - eval_ss_res / eval_ss_tot if eval_ss_tot > 0 else 0.0

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
    if use_sparsity:
        print(f"\nSparsity loss (train): {last_train_sparsity:.6f}")
        print(f"Sparsity loss (eval):  {calc_sparsity_loss().item():.6f}")
        total_mlp_params = c_source * mlp_width + mlp_width  # W_hidden + b_hidden
        print(f"MLP weight L0: {calc_mlp_l0()} / {total_mlp_params} (threshold={l0_threshold})")
    print(f"\nBias: {bias.item():.6f}")

    # Top 50 linear weights by magnitude
    magnitudes = W_linear.detach().abs()
    top_indices = magnitudes.argsort(descending=True)[:50]

    print(f"\nTop 50 linear weights (of {c_source} total):")
    print(f"{'Rank':>4}  {'Component':>10}  {'Weight':>12}  {'|Weight|':>12}")
    print("-" * 44)
    for rank, idx in enumerate(top_indices):
        w = W_linear[idx].item()
        print(f"{rank + 1:>4}  {idx.item():>10}  {w:>12.6f}  {abs(w):>12.6f}")

    if has_mlp:
        print(f"\n{act_fn_name.upper()} MLP hidden biases (width={mlp_width}):")
        for i in range(mlp_width):
            print(f"  neuron {i}: bias={b_hidden[i].item():.6f}, W_out={W_out[i].item():.6f}")

        print(f"\nTop 10 input weights per {act_fn_name.upper()} neuron:")
        for i in range(mlp_width):
            col = W_hidden[:, i].detach()
            top_k = col.abs().argsort(descending=True)[:10]
            entries = [f"{idx.item()}:{col[idx].item():+.4f}" for idx in top_k]
            print(f"  neuron {i} (out={W_out[i].item():+.6f}): {', '.join(entries)}")


if __name__ == "__main__":
    main()
