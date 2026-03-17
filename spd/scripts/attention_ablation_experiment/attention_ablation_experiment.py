"""Measure effects of ablating attention heads or SPD parameter components.

Supports three ablation modes for components:
  - deterministic: all-ones masks as baseline, zero out target components
  - stochastic: CI-based masks with stochastic sources, target CI forced to 0
  - adversarial: PGD-optimized worst-case masks, target CI forced to 0

Usage:
    # Head ablation
    python -m spd.scripts.attention_ablation_experiment.attention_ablation_experiment \
        wandb:goodfire/spd/runs/<run_id> --heads L0H3,L1H5

    # Component ablation (deterministic)
    python -m spd.scripts.attention_ablation_experiment.attention_ablation_experiment \
        wandb:goodfire/spd/runs/<run_id> --components "h.0.attn.q_proj:3,h.1.attn.k_proj:7"

    # Component ablation (stochastic)
    python -m spd.scripts.attention_ablation_experiment.attention_ablation_experiment \
        wandb:goodfire/spd/runs/<run_id> --components "h.0.attn.q_proj:3" \
        --ablation_mode stochastic --n_mask_samples 10

    # Component ablation (adversarial / PGD)
    python -m spd.scripts.attention_ablation_experiment.attention_ablation_experiment \
        wandb:goodfire/spd/runs/<run_id> --components "h.0.attn.q_proj:3" \
        --ablation_mode adversarial --pgd_steps 50 --pgd_step_size 0.01
"""

import math
import random
import re
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import fire
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import LMTaskConfig, PGDConfig, SamplingType
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.pretrain.models.llama_simple_mlp import CausalSelfAttention, LlamaSimpleMLP
from spd.routing import AllLayersRouter
from spd.spd_types import ModelPath
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent

AblationMode = Literal["deterministic", "stochastic", "adversarial"]


@dataclass
class ComponentHeadAblation:
    """Subtract a component's contribution from a specific head's q/k at a position."""

    layer: int
    qk: Literal["q", "k"]
    v_col: Tensor  # (d_in,)
    u_row: Tensor  # (d_out,)
    head: int
    pos: int


def _extract_component_vectors(
    spd_model: ComponentModel,
    module_name: str,
    comp_idx: int,
) -> tuple[Tensor, Tensor]:
    """Return (V_col, U_row) for a specific component. Both detached."""
    components = spd_model.components[module_name]
    return components.V[:, comp_idx].detach(), components.U[comp_idx, :].detach()


# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────


def parse_heads(spec: str) -> list[tuple[int, int]]:
    """Parse "L0H3,L1H5" → [(0, 3), (1, 5)]."""
    heads: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        m = re.fullmatch(r"L(\d+)H(\d+)", token)
        assert m is not None, f"Bad head spec: {token!r}, expected e.g. L0H3"
        heads.append((int(m.group(1)), int(m.group(2))))
    return heads


def parse_components(spec: str) -> list[tuple[str, int]]:
    """Parse "h.0.attn.q_proj:3,h.1.attn.k_proj:7" → [("h.0.attn.q_proj", 3), ...]."""
    components: list[tuple[str, int]] = []
    for token in spec.split(","):
        token = token.strip()
        parts = token.rsplit(":", 1)
        assert len(parts) == 2, f"Bad component spec: {token!r}, expected e.g. h.0.attn.q_proj:3"
        components.append((parts[0], int(parts[1])))
    return components


# ──────────────────────────────────────────────────────────────────────────────
# Patched attention forward (context manager)
# ──────────────────────────────────────────────────────────────────────────────

AttentionPatterns = dict[int, Float[Tensor, "n_heads T T"]]
ValueVectors = dict[int, Float[Tensor, "n_heads T head_dim"]]
AttnOutputs = dict[int, Float[Tensor, "T d_model"]]


class AttentionData(NamedTuple):
    patterns: AttentionPatterns  # layer → (n_heads, T, T)
    values: ValueVectors  # layer → (n_heads, T, head_dim)
    attn_outputs: AttnOutputs  # layer → (T, d_model)


@contextmanager
def patched_attention_forward(
    target_model: LlamaSimpleMLP,
    head_pos_ablations: list[tuple[int, int, int]] | None = None,
    value_pos_ablations: list[tuple[int, int]] | None = None,
    value_head_pos_ablations: list[tuple[int, int, int]] | None = None,
    component_head_ablations: list[ComponentHeadAblation] | None = None,
) -> Generator[AttentionData]:
    """Replace each CausalSelfAttention.forward to capture attention patterns and values.

    Yields AttentionData containing:
      - patterns: layer_index → attention pattern tensor (n_heads, T, T), mean over batch
      - values: layer_index → value vectors (n_heads, T, head_dim), mean over batch
    """
    patterns: AttentionPatterns = {}
    values: ValueVectors = {}
    attn_outs: AttnOutputs = {}
    originals: dict[int, object] = {}

    for layer_idx, block in enumerate(target_model._h):
        attn: CausalSelfAttention = block.attn
        originals[layer_idx] = attn.forward

        def _make_patched_forward(attn_module: CausalSelfAttention, li: int) -> object:
            def _patched_forward(
                x: Float[Tensor, "batch pos d_model"],
                attention_mask: Int[Tensor, "batch offset_pos"] | None = None,
                position_ids: Int[Tensor, "batch pos"] | None = None,
                _past_key_value: tuple[Tensor, Tensor] | None = None,
            ) -> Float[Tensor, "batch pos d_model"]:
                B, T, C = x.size()

                q = attn_module.q_proj(x)
                k = attn_module.k_proj(x)
                v = attn_module.v_proj(x)

                if component_head_ablations is not None:
                    for abl in component_head_ablations:
                        if abl.layer == li and abl.pos < T:
                            comp_act = x[:, abl.pos, :] @ abl.v_col.to(x.device)
                            comp_out = comp_act.unsqueeze(-1) * abl.u_row.to(x.device)
                            hd = attn_module.head_dim
                            hs = slice(abl.head * hd, (abl.head + 1) * hd)
                            if abl.qk == "q":
                                q[:, abl.pos, hs] -= comp_out[:, hs]
                            else:
                                k[:, abl.pos, hs] -= comp_out[:, hs]

                q = q.view(B, T, attn_module.n_head, attn_module.head_dim).transpose(1, 2)
                k = k.view(B, T, attn_module.n_key_value_heads, attn_module.head_dim).transpose(
                    1, 2
                )
                v = v.view(B, T, attn_module.n_key_value_heads, attn_module.head_dim).transpose(
                    1, 2
                )

                if position_ids is None:
                    if attention_mask is not None:
                        position_ids = attn_module.get_offset_position_ids(0, attention_mask)
                    else:
                        position_ids = torch.arange(T, device=x.device).unsqueeze(0)

                position_ids = position_ids.clamp(max=attn_module.n_ctx - 1)
                cos = attn_module.rotary_cos[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
                sin = attn_module.rotary_sin[position_ids].to(q.dtype)  # pyright: ignore[reportIndexIssue]
                q, k = attn_module.apply_rotary_pos_emb(q, k, cos, sin)

                if attn_module.use_grouped_query_attention and attn_module.repeat_kv_heads > 1:
                    k = k.repeat_interleave(attn_module.repeat_kv_heads, dim=1)
                    v = v.repeat_interleave(attn_module.repeat_kv_heads, dim=1)

                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn_module.head_dim))
                causal_mask = attn_module.bias[:, :, :T, :T]  # pyright: ignore[reportIndexIssue]
                att = att.masked_fill(causal_mask == 0, float("-inf"))
                att = F.softmax(att, dim=-1)

                patterns[li] = att.float().mean(dim=0).detach().cpu()

                if value_pos_ablations is not None:
                    for abl_layer, abl_pos in value_pos_ablations:
                        if abl_layer == li and abl_pos < T:
                            v[:, :, abl_pos, :] = 0.0
                if value_head_pos_ablations is not None:
                    for abl_layer, abl_head, abl_pos in value_head_pos_ablations:
                        if abl_layer == li and abl_pos < T:
                            v[:, abl_head, abl_pos, :] = 0.0

                y = att @ v  # (B, n_head, T, head_dim)

                if head_pos_ablations is not None:
                    for abl_layer, abl_head, abl_pos in head_pos_ablations:
                        if abl_layer == li and abl_pos < T:
                            y[:, abl_head, abl_pos, :] = 0.0

                values[li] = v.float().mean(dim=0).detach().cpu()

                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = attn_module.o_proj(y)

                attn_outs[li] = y.float().mean(dim=0).detach().cpu()

                return y

            return _patched_forward

        attn.forward = _make_patched_forward(attn, layer_idx)  # pyright: ignore[reportAttributeAccessIssue]

    try:
        yield AttentionData(patterns, values, attn_outs)
    finally:
        for layer_idx, block in enumerate(target_model._h):
            block.attn.forward = originals[layer_idx]  # pyright: ignore[reportAttributeAccessIssue]


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────


def plot_attention_grid(
    patterns: AttentionPatterns,
    title: str,
    path: Path,
    max_pos: int,
) -> None:
    n_layers = len(patterns)
    n_heads = patterns[0].shape[0]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 3, n_layers * 3), squeeze=False)

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            ax = axes[layer_idx, h]
            pat = patterns[layer_idx][h, :max_pos, :max_pos].numpy()
            ax.imshow(pat, aspect="auto", cmap="viridis", vmin=0)
            ax.set_title(f"L{layer_idx}H{h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_attention_diff(
    baseline: AttentionPatterns,
    ablated: AttentionPatterns,
    title: str,
    path: Path,
    max_pos: int,
) -> None:
    n_layers = len(baseline)
    n_heads = baseline[0].shape[0]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 3, n_layers * 3), squeeze=False)

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            ax = axes[layer_idx, h]
            diff = (
                ablated[layer_idx][h, :max_pos, :max_pos]
                - baseline[layer_idx][h, :max_pos, :max_pos]
            ).numpy()
            vmax = max(abs(diff.min()), abs(diff.max()), 1e-8)
            ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"L{layer_idx}H{h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_value_norms(
    values: ValueVectors,
    title: str,
    path: Path,
    max_pos: int,
) -> None:
    n_layers = len(values)
    fig, axes = plt.subplots(n_layers, 1, figsize=(8, n_layers * 2.5), squeeze=False)

    for layer_idx in range(n_layers):
        ax = axes[layer_idx, 0]
        norms = values[layer_idx][:, :max_pos, :].norm(dim=-1).numpy()  # (n_heads, max_pos)
        im = ax.imshow(norms, aspect="auto", cmap="viridis")
        ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("Position", fontsize=8)
        n_heads = norms.shape[0]
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_value_norms_diff(
    baseline_values: ValueVectors,
    ablated_values: ValueVectors,
    title: str,
    path: Path,
    max_pos: int,
) -> None:
    n_layers = len(baseline_values)
    fig, axes = plt.subplots(n_layers, 1, figsize=(8, n_layers * 2.5), squeeze=False)

    for layer_idx in range(n_layers):
        ax = axes[layer_idx, 0]
        baseline_norms = baseline_values[layer_idx][:, :max_pos, :].norm(dim=-1)
        ablated_norms = ablated_values[layer_idx][:, :max_pos, :].norm(dim=-1)
        diff = (ablated_norms - baseline_norms).numpy()
        vmax = max(abs(diff.min()), abs(diff.max()), 1e-8)
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("Position", fontsize=8)
        n_heads = diff.shape[0]
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def compute_ablation_metrics(
    baseline_attn_outputs: AttnOutputs,
    ablated_attn_outputs: AttnOutputs,
) -> tuple[AttnOutputs, AttnOutputs]:
    """Compute per-position normalized inner product and cosine similarity.

    Normalized IP: dot(baseline, ablated) / ||baseline||². 1.0 at unaffected positions.
    Cosine sim: cos(baseline, ablated). 1.0 at unaffected positions.

    Returns (normalized_ips, cosine_sims) where each is layer → (T,).
    """
    normalized_ips: AttnOutputs = {}
    cosine_sims: AttnOutputs = {}
    for layer_idx in baseline_attn_outputs:
        baseline = baseline_attn_outputs[layer_idx]
        ablated = ablated_attn_outputs[layer_idx]
        ip = (baseline * ablated).sum(dim=-1)
        baseline_norm_sq = (baseline * baseline).sum(dim=-1)
        normalized_ips[layer_idx] = ip / baseline_norm_sq.clamp(min=1e-8)
        norms_product = baseline.norm(dim=-1) * ablated.norm(dim=-1)
        cosine_sims[layer_idx] = ip / norms_product.clamp(min=1e-8)
    return normalized_ips, cosine_sims


def plot_per_position_line(
    values: AttnOutputs,
    title: str,
    path: Path,
    max_pos: int,
    baseline_y: float = 0.0,
    ylim: tuple[float, float] | None = None,
) -> None:
    n_layers = len(values)
    fig, axes = plt.subplots(n_layers, 1, figsize=(10, n_layers * 2), squeeze=False)

    for layer_idx in range(n_layers):
        ax = axes[layer_idx, 0]
        vals = values[layer_idx][:max_pos].numpy()
        ax.plot(vals, linewidth=0.8)
        ax.axhline(y=baseline_y, color="gray", linewidth=0.5, linestyle="--")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("Position", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def compute_ablation_metrics_at_pos(
    baseline_attn_outputs: AttnOutputs,
    ablated_attn_outputs: AttnOutputs,
    pos: int,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute normalized inner product and cosine similarity at a single position.

    Returns (normalized_ips, cosine_sims) — layer -> scalar.
    """
    normalized_ips: dict[int, float] = {}
    cosine_sims: dict[int, float] = {}
    for layer_idx in baseline_attn_outputs:
        baseline_vec = baseline_attn_outputs[layer_idx][pos]
        ablated_vec = ablated_attn_outputs[layer_idx][pos]
        ip = (baseline_vec * ablated_vec).sum().item()
        baseline_norm_sq = (baseline_vec * baseline_vec).sum().item()
        normalized_ips[layer_idx] = ip / max(baseline_norm_sq, 1e-8)
        norms_product = baseline_vec.norm().item() * ablated_vec.norm().item()
        cosine_sims[layer_idx] = ip / max(norms_product, 1e-8)
    return normalized_ips, cosine_sims


def plot_output_similarity_bars(
    means: dict[int, float],
    stds: dict[int, float],
    title: str,
    path: Path,
) -> None:
    layers = sorted(means.keys())
    mean_vals = [means[li] for li in layers]
    std_vals = [stds[li] for li in layers]
    layer_labels = [f"L{li}" for li in layers]

    fig, ax = plt.subplots(figsize=(max(4, len(layers) * 1.2), 4))
    ax.bar(layer_labels, mean_vals, yerr=std_vals, capsize=4, color="steelblue", alpha=0.8)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Token prediction table & stats
# ──────────────────────────────────────────────────────────────────────────────


def log_prediction_table(
    input_ids: Tensor,
    baseline_logits: Tensor,
    ablated_logits: Tensor,
    tokenizer: object,
    last_n: int = 20,
) -> int:
    """Log per-position prediction changes. Returns count of changed positions."""
    seq_len = input_ids.shape[0]
    baseline_probs = F.softmax(baseline_logits, dim=-1)
    ablated_probs = F.softmax(ablated_logits, dim=-1)

    baseline_top = baseline_probs.argmax(dim=-1)
    ablated_top = ablated_probs.argmax(dim=-1)

    changed_mask = baseline_top != ablated_top
    changed_positions = changed_mask.nonzero(as_tuple=True)[0].tolist()
    show_positions = set(changed_positions) | set(range(max(0, seq_len - last_n), seq_len))

    decode = tokenizer.decode  # pyright: ignore[reportAttributeAccessIssue]
    lines = [f"{'Pos':>4} | {'Token':>10} | {'Baseline (prob)':>20} | {'Ablated (prob)':>20} | Chg"]
    lines.append("-" * 85)

    for pos in sorted(show_positions):
        tok = decode([input_ids[pos].item()]).replace("\n", "\\n")
        b_id = int(baseline_top[pos].item())
        a_id = int(ablated_top[pos].item())
        b_tok = decode([b_id]).replace("\n", "\\n")
        a_tok = decode([a_id]).replace("\n", "\\n")
        b_prob = baseline_probs[pos, b_id].item()
        a_prob = ablated_probs[pos, a_id].item()
        changed = " *" if pos in changed_positions else ""
        lines.append(
            f"{pos:>4} | {tok:>10} | {b_tok:>10} ({b_prob:.3f}) | {a_tok:>10} ({a_prob:.3f}) |{changed}"
        )

    logger.info("Prediction table:\n" + "\n".join(lines))
    return len(changed_positions)


def calc_mean_kl_divergence(
    baseline_logits: Tensor,
    ablated_logits: Tensor,
) -> float:
    """KL(baseline || ablated) averaged over positions, for first item in batch."""
    baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
    ablated_log_probs = F.log_softmax(ablated_logits, dim=-1)
    kl = F.kl_div(ablated_log_probs, baseline_log_probs.exp(), reduction="batchmean")
    return kl.item()


# ──────────────────────────────────────────────────────────────────────────────
# Component mask construction
# ──────────────────────────────────────────────────────────────────────────────


def _build_deterministic_masks(
    model: ComponentModel,
    ablated_components: list[tuple[str, int]],
    batch_shape: tuple[int, int],
    device: torch.device,
    ablation_pos: int,
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, ComponentsMaskInfo]]:
    """Build all-ones baseline and ablated mask_infos for deterministic mode.

    Masks have shape (batch, seq_len, C) and the target component is zeroed only at
    ablation_pos.
    """
    baseline_masks: dict[str, Float[Tensor, "batch seq_len C"]] = {}
    ablated_masks: dict[str, Float[Tensor, "batch seq_len C"]] = {}

    for module_name in model.target_module_paths:
        c = model.module_to_c[module_name]
        baseline_masks[module_name] = torch.ones(*batch_shape, c, device=device)
        ablated_masks[module_name] = torch.ones(*batch_shape, c, device=device)

    for module_name, comp_idx in ablated_components:
        assert module_name in ablated_masks, f"Module {module_name!r} not in model"
        ablated_masks[module_name][:, ablation_pos, comp_idx] = 0.0

    return make_mask_infos(baseline_masks), make_mask_infos(ablated_masks)


def _build_deterministic_masks_multi_pos(
    model: ComponentModel,
    component_positions: list[tuple[str, int, int]],
    batch_shape: tuple[int, int],
    device: torch.device,
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, ComponentsMaskInfo]]:
    """Build masks where each component is zeroed at its own position.

    component_positions: list of (module_name, comp_idx, pos).
    """
    baseline_masks: dict[str, Float[Tensor, "batch seq_len C"]] = {}
    ablated_masks: dict[str, Float[Tensor, "batch seq_len C"]] = {}

    for module_name in model.target_module_paths:
        c = model.module_to_c[module_name]
        baseline_masks[module_name] = torch.ones(*batch_shape, c, device=device)
        ablated_masks[module_name] = torch.ones(*batch_shape, c, device=device)

    for module_name, comp_idx, pos in component_positions:
        assert module_name in ablated_masks, f"Module {module_name!r} not in model"
        ablated_masks[module_name][:, pos, comp_idx] = 0.0

    return make_mask_infos(baseline_masks), make_mask_infos(ablated_masks)


def _infer_layer_from_components(parsed_components: list[tuple[str, int]]) -> int:
    """Extract layer index from component module paths (e.g. 'h.1.attn.q_proj' → 1)."""
    layers: set[int] = set()
    for module_name, _ in parsed_components:
        m = re.search(r"h\.(\d+)\.", module_name)
        assert m is not None, f"Cannot infer layer from {module_name!r}"
        layers.add(int(m.group(1)))
    assert len(layers) == 1, f"All components must be in the same layer, got layers {layers}"
    return layers.pop()


def _build_prev_token_component_positions(
    parsed_components: list[tuple[str, int]],
    t: int,
) -> list[tuple[str, int, int]]:
    """Assign positions based on module type: q_proj → t, k_proj → t-1."""
    positions: list[tuple[str, int, int]] = []
    for module_name, comp_idx in parsed_components:
        if "q_proj" in module_name:
            positions.append((module_name, comp_idx, t))
        elif "k_proj" in module_name:
            positions.append((module_name, comp_idx, t - 1))
        else:
            raise AssertionError(
                f"prev_token_test only supports q_proj/k_proj components, got {module_name!r}"
            )
    return positions


def _build_component_head_ablations(
    spd_model: ComponentModel,
    parsed_components: list[tuple[str, int]],
    heads: list[tuple[int, int]],
    t: int,
) -> list[ComponentHeadAblation]:
    """Build per-head component ablations: q components at t, k components at t-1."""
    ablations: list[ComponentHeadAblation] = []
    for module_name, comp_idx in parsed_components:
        v_col, u_row = _extract_component_vectors(spd_model, module_name, comp_idx)
        if "q_proj" in module_name:
            qk: Literal["q", "k"] = "q"
            pos = t
        elif "k_proj" in module_name:
            qk = "k"
            pos = t - 1
        else:
            raise AssertionError(
                f"per-head component ablation only supports q_proj/k_proj, got {module_name!r}"
            )
        layer = _infer_layer_from_components([(module_name, comp_idx)])
        for _layer, head in heads:
            ablations.append(ComponentHeadAblation(layer, qk, v_col, u_row, head, pos))
    return ablations


def _build_stochastic_masks(
    _model: ComponentModel,
    ci: dict[str, Float[Tensor, "batch C"]],
    ablated_components: list[tuple[str, int]],
    sampling: SamplingType,
    ablation_pos: int,
    seq_len: int,
) -> tuple[dict[str, ComponentsMaskInfo], dict[str, ComponentsMaskInfo]]:
    """Build stochastic mask_infos: baseline uses original CI, ablated zeros target CIs.

    CI is expanded to (batch, seq_len, C) and the target component CI is zeroed only at
    ablation_pos.
    """
    router = AllLayersRouter()

    expanded_ci = {k: v.unsqueeze(1).expand(-1, seq_len, -1).clone() for k, v in ci.items()}
    baseline_mask_infos = calc_stochastic_component_mask_info(expanded_ci, sampling, None, router)

    ablated_ci = {k: v.clone() for k, v in expanded_ci.items()}
    for module_name, comp_idx in ablated_components:
        assert module_name in ablated_ci, f"Module {module_name!r} not in model"
        ablated_ci[module_name][:, ablation_pos, comp_idx] = 0.0
    ablated_mask_infos = calc_stochastic_component_mask_info(ablated_ci, sampling, None, router)

    return baseline_mask_infos, ablated_mask_infos


def _build_adversarial_masks(
    model: ComponentModel,
    batch: Int[Tensor, "batch pos"],
    ci: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... vocab"],
    ablated_components: list[tuple[str, int]],
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    """Run PGD for baseline and ablated, return (baseline_loss, ablated_loss)."""
    from spd.metrics.pgd_utils import pgd_masked_recon_loss_update

    router = AllLayersRouter()

    baseline_sum_loss, baseline_n = pgd_masked_recon_loss_update(
        model, batch, ci, None, target_out, "kl", router, pgd_config
    )

    ablated_ci = {k: v.clone() for k, v in ci.items()}
    for module_name, comp_idx in ablated_components:
        assert module_name in ablated_ci, f"Module {module_name!r} not in model"
        ablated_ci[module_name][..., comp_idx] = 0.0

    ablated_sum_loss, ablated_n = pgd_masked_recon_loss_update(
        model, batch, ablated_ci, None, target_out, "kl", router, pgd_config
    )

    return baseline_sum_loss / baseline_n, ablated_sum_loss / ablated_n


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample result + accumulation helpers
# ──────────────────────────────────────────────────────────────────────────────


class SampleResult(NamedTuple):
    baseline_patterns: AttentionPatterns
    ablated_patterns: AttentionPatterns
    baseline_values: ValueVectors
    ablated_values: ValueVectors
    baseline_attn_outputs: AttnOutputs
    ablated_attn_outputs: AttnOutputs
    baseline_logits: Tensor  # (batch, pos, vocab)
    ablated_logits: Tensor  # (batch, pos, vocab)


class PrevTokenSampleResult(NamedTuple):
    baseline_attn_outputs: AttnOutputs
    a_attn_outputs: AttnOutputs
    b_all_attn_outputs: AttnOutputs
    b_specific_attn_outputs: AttnOutputs
    ab_all_attn_outputs: AttnOutputs
    ab_specific_attn_outputs: AttnOutputs
    baseline_logits: Tensor
    a_logits: Tensor
    a_b_all_logits: Tensor
    ab_specific_logits: Tensor


def _add_patterns(accum: AttentionPatterns, new: AttentionPatterns) -> None:
    for layer_idx, pat in new.items():
        if layer_idx in accum:
            accum[layer_idx] = accum[layer_idx] + pat
        else:
            accum[layer_idx] = pat.clone()


def _scale_patterns(accum: AttentionPatterns, n: int) -> AttentionPatterns:
    return {k: v / n for k, v in accum.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Head ablation
# ──────────────────────────────────────────────────────────────────────────────


def _run_head_ablation(
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_heads: list[tuple[int, int]],
    ablation_pos: int,
) -> SampleResult:
    with patched_attention_forward(target_model) as baseline_data:
        baseline_logits, _ = target_model(input_ids)

    pos_ablations = [(layer, head, ablation_pos) for layer, head in parsed_heads]
    with patched_attention_forward(target_model, head_pos_ablations=pos_ablations) as ablated_data:
        ablated_logits, _ = target_model(input_ids)

    assert baseline_logits is not None and ablated_logits is not None
    return SampleResult(
        baseline_data.patterns,
        ablated_data.patterns,
        baseline_data.values,
        ablated_data.values,
        baseline_data.attn_outputs,
        ablated_data.attn_outputs,
        baseline_logits,
        ablated_logits,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Component ablation
# ──────────────────────────────────────────────────────────────────────────────


def _run_component_ablation(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_components: list[tuple[str, int]],
    ablation_mode: AblationMode,
    n_mask_samples: int,
    pgd_steps: int,
    pgd_step_size: float,
    ablation_pos: int,
) -> SampleResult:
    match ablation_mode:
        case "deterministic":
            return _run_deterministic_component_ablation(
                spd_model, target_model, input_ids, parsed_components, ablation_pos
            )
        case "stochastic":
            return _run_stochastic_component_ablation(
                spd_model, target_model, input_ids, parsed_components, n_mask_samples, ablation_pos
            )
        case "adversarial":
            return _run_adversarial_component_ablation(
                spd_model,
                target_model,
                input_ids,
                parsed_components,
                pgd_steps,
                pgd_step_size,
                ablation_pos,
            )


def _run_deterministic_component_ablation(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_components: list[tuple[str, int]],
    ablation_pos: int,
) -> SampleResult:
    batch_shape = (input_ids.shape[0], input_ids.shape[1])
    baseline_mask_infos, ablated_mask_infos = _build_deterministic_masks(
        spd_model, parsed_components, batch_shape, input_ids.device, ablation_pos
    )

    with patched_attention_forward(target_model) as baseline_data:
        baseline_out = spd_model(input_ids, mask_infos=baseline_mask_infos)
    assert isinstance(baseline_out, Tensor)

    with patched_attention_forward(target_model) as ablated_data:
        ablated_out = spd_model(input_ids, mask_infos=ablated_mask_infos)
    assert isinstance(ablated_out, Tensor)

    return SampleResult(
        baseline_data.patterns,
        ablated_data.patterns,
        baseline_data.values,
        ablated_data.values,
        baseline_data.attn_outputs,
        ablated_data.attn_outputs,
        baseline_out,
        ablated_out,
    )


def _run_stochastic_component_ablation(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_components: list[tuple[str, int]],
    n_mask_samples: int,
    ablation_pos: int,
) -> SampleResult:
    output_with_cache = spd_model(input_ids, cache_type="input")
    assert isinstance(output_with_cache, OutputWithCache)
    ci = spd_model.calc_causal_importances(output_with_cache.cache, "continuous").lower_leaky

    baseline_logits_accum: Tensor | None = None
    ablated_logits_accum: Tensor | None = None
    sample_baseline_patterns: AttentionPatterns = {}
    sample_ablated_patterns: AttentionPatterns = {}
    sample_baseline_values: ValueVectors = {}
    sample_ablated_values: ValueVectors = {}
    sample_baseline_attn_outs: AttnOutputs = {}
    sample_ablated_attn_outs: AttnOutputs = {}

    stoch_seq_len = input_ids.shape[1]
    for _s in range(n_mask_samples):
        baseline_mask_infos, ablated_mask_infos = _build_stochastic_masks(
            spd_model, ci, parsed_components, "continuous", ablation_pos, stoch_seq_len
        )

        with patched_attention_forward(target_model) as b_data:
            b_out = spd_model(input_ids, mask_infos=baseline_mask_infos)
        assert isinstance(b_out, Tensor)

        with patched_attention_forward(target_model) as a_data:
            a_out = spd_model(input_ids, mask_infos=ablated_mask_infos)
        assert isinstance(a_out, Tensor)

        if baseline_logits_accum is None:
            baseline_logits_accum = b_out
            ablated_logits_accum = a_out
        else:
            baseline_logits_accum = baseline_logits_accum + b_out
            assert ablated_logits_accum is not None
            ablated_logits_accum = ablated_logits_accum + a_out

        _add_patterns(sample_baseline_patterns, b_data.patterns)
        _add_patterns(sample_ablated_patterns, a_data.patterns)
        _add_patterns(sample_baseline_values, b_data.values)
        _add_patterns(sample_ablated_values, a_data.values)
        _add_patterns(sample_baseline_attn_outs, b_data.attn_outputs)
        _add_patterns(sample_ablated_attn_outs, a_data.attn_outputs)

    assert baseline_logits_accum is not None and ablated_logits_accum is not None
    return SampleResult(
        _scale_patterns(sample_baseline_patterns, n_mask_samples),
        _scale_patterns(sample_ablated_patterns, n_mask_samples),
        _scale_patterns(sample_baseline_values, n_mask_samples),
        _scale_patterns(sample_ablated_values, n_mask_samples),
        _scale_patterns(sample_baseline_attn_outs, n_mask_samples),
        _scale_patterns(sample_ablated_attn_outs, n_mask_samples),
        baseline_logits_accum / n_mask_samples,
        ablated_logits_accum / n_mask_samples,
    )


def _run_adversarial_component_ablation(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_components: list[tuple[str, int]],
    pgd_steps: int,
    pgd_step_size: float,
    ablation_pos: int,
) -> SampleResult:
    output_with_cache = spd_model(input_ids, cache_type="input")
    assert isinstance(output_with_cache, OutputWithCache)
    ci = spd_model.calc_causal_importances(output_with_cache.cache, "continuous").lower_leaky

    target_out = output_with_cache.output

    pgd_config = PGDConfig(
        init="random",
        step_size=pgd_step_size,
        n_steps=pgd_steps,
        mask_scope="unique_per_datapoint",
    )

    baseline_loss, ablated_loss = _build_adversarial_masks(
        spd_model, input_ids, ci, target_out, parsed_components, pgd_config
    )
    logger.info(
        f"PGD losses — baseline: {baseline_loss.item():.4f}, ablated: {ablated_loss.item():.4f}"
    )

    # Capture attention patterns with deterministic masks for visualization
    batch_shape = (input_ids.shape[0], input_ids.shape[1])
    baseline_mask_infos, ablated_mask_infos = _build_deterministic_masks(
        spd_model, parsed_components, batch_shape, input_ids.device, ablation_pos
    )

    with patched_attention_forward(target_model) as baseline_data:
        baseline_out = spd_model(input_ids, mask_infos=baseline_mask_infos)
    assert isinstance(baseline_out, Tensor)

    with patched_attention_forward(target_model) as ablated_data:
        ablated_out = spd_model(input_ids, mask_infos=ablated_mask_infos)
    assert isinstance(ablated_out, Tensor)

    return SampleResult(
        baseline_data.patterns,
        ablated_data.patterns,
        baseline_data.values,
        ablated_data.values,
        baseline_data.attn_outputs,
        ablated_data.attn_outputs,
        baseline_out,
        ablated_out,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Previous-token redundancy test
# ──────────────────────────────────────────────────────────────────────────────


def _capture_attn_outputs(
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    head_pos_ablations: list[tuple[int, int, int]] | None = None,
    value_pos_ablations: list[tuple[int, int]] | None = None,
    value_head_pos_ablations: list[tuple[int, int, int]] | None = None,
    component_head_ablations: list[ComponentHeadAblation] | None = None,
    spd_model: ComponentModel | None = None,
    mask_infos: dict[str, ComponentsMaskInfo] | None = None,
) -> tuple[AttnOutputs, Tensor]:
    """Run a forward pass capturing attention outputs and logits."""
    with patched_attention_forward(
        target_model,
        head_pos_ablations,
        value_pos_ablations,
        value_head_pos_ablations,
        component_head_ablations,
    ) as data:
        if spd_model is not None:
            out = spd_model(input_ids, mask_infos=mask_infos)
            assert isinstance(out, Tensor)
        else:
            out, _ = target_model(input_ids)
            assert out is not None
    return data.attn_outputs, out


def _run_prev_token_head_ablation(
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_heads: list[tuple[int, int]],
    value_heads: list[tuple[int, int]],
    t: int,
) -> PrevTokenSampleResult:
    head_abl = [(layer, head, t) for layer, head in parsed_heads]
    layer = parsed_heads[0][0]
    val_all = [(layer, t - 1)]
    val_specific = [(layer, head, t - 1) for layer, head in value_heads]

    baseline_outs, baseline_logits = _capture_attn_outputs(target_model, input_ids)
    a_outs, a_logits = _capture_attn_outputs(target_model, input_ids, head_pos_ablations=head_abl)
    b_all_outs, _b_all_logits = _capture_attn_outputs(
        target_model, input_ids, value_pos_ablations=val_all
    )
    b_spec_outs, _b_spec_logits = _capture_attn_outputs(
        target_model, input_ids, value_head_pos_ablations=val_specific
    )
    ab_all_outs, a_b_all_logits = _capture_attn_outputs(
        target_model, input_ids, head_pos_ablations=head_abl, value_pos_ablations=val_all
    )
    ab_spec_outs, a_b_spec_logits = _capture_attn_outputs(
        target_model, input_ids, head_pos_ablations=head_abl, value_head_pos_ablations=val_specific
    )

    return PrevTokenSampleResult(
        baseline_outs,
        a_outs,
        b_all_outs,
        b_spec_outs,
        ab_all_outs,
        ab_spec_outs,
        baseline_logits,
        a_logits,
        a_b_all_logits,
        a_b_spec_logits,
    )


def _run_prev_token_component_ablation(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_components: list[tuple[str, int]],
    value_heads: list[tuple[int, int]],
    t: int,
) -> PrevTokenSampleResult:
    layer = _infer_layer_from_components(parsed_components)
    component_positions = _build_prev_token_component_positions(parsed_components, t)
    batch_shape = (input_ids.shape[0], input_ids.shape[1])

    baseline_masks, ablated_masks = _build_deterministic_masks_multi_pos(
        spd_model, component_positions, batch_shape, input_ids.device
    )
    val_all = [(layer, t - 1)]
    val_specific = [(layer, head, t - 1) for layer, head in value_heads]

    baseline_outs, baseline_logits = _capture_attn_outputs(
        target_model, input_ids, spd_model=spd_model, mask_infos=baseline_masks
    )
    a_outs, a_logits = _capture_attn_outputs(
        target_model, input_ids, spd_model=spd_model, mask_infos=ablated_masks
    )
    b_all_outs, _b_all_logits = _capture_attn_outputs(
        target_model,
        input_ids,
        value_pos_ablations=val_all,
        spd_model=spd_model,
        mask_infos=baseline_masks,
    )
    b_spec_outs, _b_spec_logits = _capture_attn_outputs(
        target_model,
        input_ids,
        value_head_pos_ablations=val_specific,
        spd_model=spd_model,
        mask_infos=baseline_masks,
    )
    ab_all_outs, a_b_all_logits = _capture_attn_outputs(
        target_model,
        input_ids,
        value_pos_ablations=val_all,
        spd_model=spd_model,
        mask_infos=ablated_masks,
    )
    ab_spec_outs, a_b_spec_logits = _capture_attn_outputs(
        target_model,
        input_ids,
        value_head_pos_ablations=val_specific,
        spd_model=spd_model,
        mask_infos=ablated_masks,
    )

    return PrevTokenSampleResult(
        baseline_outs,
        a_outs,
        b_all_outs,
        b_spec_outs,
        ab_all_outs,
        ab_spec_outs,
        baseline_logits,
        a_logits,
        a_b_all_logits,
        a_b_spec_logits,
    )


def _run_prev_token_head_restricted_component_ablation(
    spd_model: ComponentModel,
    target_model: LlamaSimpleMLP,
    input_ids: Int[Tensor, "batch pos"],
    parsed_components: list[tuple[str, int]],
    restrict_heads: list[tuple[int, int]],
    value_heads: list[tuple[int, int]],
    t: int,
) -> PrevTokenSampleResult:
    """Per-head component ablation: subtract component contributions from specific heads only."""
    layer = _infer_layer_from_components(parsed_components)
    comp_head_abls = _build_component_head_ablations(
        spd_model, parsed_components, restrict_heads, t
    )
    batch_shape = (input_ids.shape[0], input_ids.shape[1])

    # All-ones baseline masks (SPD model reconstructs original output)
    baseline_masks, _ = _build_deterministic_masks(spd_model, [], batch_shape, input_ids.device, t)
    val_all = [(layer, t - 1)]
    val_specific = [(layer, head, t - 1) for layer, head in value_heads]

    baseline_outs, baseline_logits = _capture_attn_outputs(
        target_model, input_ids, spd_model=spd_model, mask_infos=baseline_masks
    )
    a_outs, a_logits = _capture_attn_outputs(
        target_model,
        input_ids,
        component_head_ablations=comp_head_abls,
        spd_model=spd_model,
        mask_infos=baseline_masks,
    )
    b_all_outs, _b_all_logits = _capture_attn_outputs(
        target_model,
        input_ids,
        value_pos_ablations=val_all,
        spd_model=spd_model,
        mask_infos=baseline_masks,
    )
    b_spec_outs, _b_spec_logits = _capture_attn_outputs(
        target_model,
        input_ids,
        value_head_pos_ablations=val_specific,
        spd_model=spd_model,
        mask_infos=baseline_masks,
    )
    ab_all_outs, a_b_all_logits = _capture_attn_outputs(
        target_model,
        input_ids,
        component_head_ablations=comp_head_abls,
        value_pos_ablations=val_all,
        spd_model=spd_model,
        mask_infos=baseline_masks,
    )
    ab_spec_outs, a_b_spec_logits = _capture_attn_outputs(
        target_model,
        input_ids,
        component_head_ablations=comp_head_abls,
        value_head_pos_ablations=val_specific,
        spd_model=spd_model,
        mask_infos=baseline_masks,
    )

    return PrevTokenSampleResult(
        baseline_outs,
        a_outs,
        b_all_outs,
        b_spec_outs,
        ab_all_outs,
        ab_spec_outs,
        baseline_logits,
        a_logits,
        a_b_all_logits,
        a_b_spec_logits,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Offset sweep
# ──────────────────────────────────────────────────────────────────────────────


def _run_offset_sweep(
    target_model: LlamaSimpleMLP,
    spd_model: ComponentModel | None,
    loader: Iterable[dict[str, Tensor]],
    is_head_ablation: bool,
    parsed_heads: list[tuple[int, int]],
    parsed_components: list[tuple[str, int]],
    parsed_restrict_heads: list[tuple[int, int]],
    n_samples: int,
    max_offsets: int,
    max_pos: int,
    seq_len: int,
    run_id: str,
    label: str,
    sim_dir: Path,
    column_name: str,
    device: torch.device,
) -> None:
    """Sweep value ablation across offsets 1..max_offsets to profile which positions matter."""
    if is_head_ablation:
        layer = parsed_heads[0][0]
    else:
        layer = _infer_layer_from_components(parsed_components)

    # offset -> layer -> list[float]
    base_vs_a_nips: dict[str, dict[int, list[float]]] = {"nip": {}, "cos": {}}
    base_vs_ab_by_offset: dict[int, dict[str, dict[int, list[float]]]] = {
        offset: {"nip": {}, "cos": {}} for offset in range(1, max_offsets + 1)
    }
    n_processed = 0

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if i >= n_samples:
                break

            input_ids: Int[Tensor, "batch pos"] = batch_data[column_name][:, :seq_len].to(device)

            sample_seq_len = input_ids.shape[1]
            rng = random.Random(i)
            t = rng.randint(max_offsets, min(sample_seq_len, max_pos) - 1)

            # Compute baseline and A once
            if is_head_ablation:
                head_abl = [(ly, hd, t) for ly, hd in parsed_heads]
                baseline_outs, _ = _capture_attn_outputs(target_model, input_ids)
                a_outs, _ = _capture_attn_outputs(
                    target_model, input_ids, head_pos_ablations=head_abl
                )
            elif parsed_restrict_heads:
                assert spd_model is not None
                comp_head_abls = _build_component_head_ablations(
                    spd_model, parsed_components, parsed_restrict_heads, t
                )
                batch_shape = (input_ids.shape[0], input_ids.shape[1])
                baseline_masks, _ = _build_deterministic_masks(
                    spd_model, [], batch_shape, input_ids.device, t
                )
                baseline_outs, _ = _capture_attn_outputs(
                    target_model, input_ids, spd_model=spd_model, mask_infos=baseline_masks
                )
                a_outs, _ = _capture_attn_outputs(
                    target_model,
                    input_ids,
                    component_head_ablations=comp_head_abls,
                    spd_model=spd_model,
                    mask_infos=baseline_masks,
                )
            else:
                assert spd_model is not None
                component_positions = _build_prev_token_component_positions(parsed_components, t)
                batch_shape = (input_ids.shape[0], input_ids.shape[1])
                baseline_masks, ablated_masks = _build_deterministic_masks_multi_pos(
                    spd_model, component_positions, batch_shape, input_ids.device
                )
                baseline_outs, _ = _capture_attn_outputs(
                    target_model, input_ids, spd_model=spd_model, mask_infos=baseline_masks
                )
                a_outs, _ = _capture_attn_outputs(
                    target_model, input_ids, spd_model=spd_model, mask_infos=ablated_masks
                )

            base_vs_a_nip, base_vs_a_cos = compute_ablation_metrics_at_pos(baseline_outs, a_outs, t)
            _accum_comparison(base_vs_a_nips, base_vs_a_nip, base_vs_a_cos)

            # Sweep offsets
            for offset in range(1, max_offsets + 1):
                val_pos = t - offset
                assert val_pos >= 0
                val_all = [(layer, val_pos)]

                if is_head_ablation:
                    head_abl = [(ly, hd, t) for ly, hd in parsed_heads]
                    ab_outs, _ = _capture_attn_outputs(
                        target_model,
                        input_ids,
                        head_pos_ablations=head_abl,
                        value_pos_ablations=val_all,
                    )
                elif parsed_restrict_heads:
                    assert spd_model is not None
                    cha = _build_component_head_ablations(
                        spd_model, parsed_components, parsed_restrict_heads, t
                    )
                    bs = (input_ids.shape[0], input_ids.shape[1])
                    bm, _ = _build_deterministic_masks(spd_model, [], bs, input_ids.device, t)
                    ab_outs, _ = _capture_attn_outputs(
                        target_model,
                        input_ids,
                        component_head_ablations=cha,
                        value_pos_ablations=val_all,
                        spd_model=spd_model,
                        mask_infos=bm,
                    )
                else:
                    assert spd_model is not None
                    cp = _build_prev_token_component_positions(parsed_components, t)
                    bs = (input_ids.shape[0], input_ids.shape[1])
                    _, am = _build_deterministic_masks_multi_pos(
                        spd_model, cp, bs, input_ids.device
                    )
                    ab_outs, _ = _capture_attn_outputs(
                        target_model,
                        input_ids,
                        value_pos_ablations=val_all,
                        spd_model=spd_model,
                        mask_infos=am,
                    )

                base_vs_ab_nip, base_vs_ab_cos = compute_ablation_metrics_at_pos(
                    baseline_outs, ab_outs, t
                )
                _accum_comparison(base_vs_ab_by_offset[offset], base_vs_ab_nip, base_vs_ab_cos)

            n_processed += 1
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{n_samples} samples")

    assert n_processed > 0, "No samples processed"

    # Compute means
    base_vs_a_nip_mean = {
        li: torch.tensor(vs).mean().item() for li, vs in base_vs_a_nips["nip"].items()
    }

    # Plot offset profile for each layer
    layers = sorted(base_vs_a_nip_mean.keys())
    offsets = list(range(1, max_offsets + 1))

    for metric_key, metric_name in [("nip", "NIP"), ("cos", "Cosine Sim")]:
        fig, axes = plt.subplots(len(layers), 1, figsize=(10, len(layers) * 2.5), squeeze=False)

        for li_idx, li in enumerate(layers):
            ax = axes[li_idx, 0]
            means = [
                torch.tensor(base_vs_ab_by_offset[o][metric_key][li]).mean().item() for o in offsets
            ]
            stds = [
                torch.tensor(base_vs_ab_by_offset[o][metric_key][li]).std().item() for o in offsets
            ]

            ax.errorbar(offsets, means, yerr=stds, fmt="o-", capsize=3, markersize=4)
            base_val = torch.tensor(base_vs_a_nips[metric_key][li]).mean().item()
            ax.axhline(
                y=base_val, color="red", linewidth=0.8, linestyle="--", label="Baseline vs A"
            )
            ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle=":")
            ax.set_ylabel(f"Layer {li}", fontsize=9)
            ax.set_xlabel("Offset (t - offset)", fontsize=8)
            ax.legend(fontsize=7)

        fig.suptitle(
            f"{run_id} | {metric_name} offset profile (n={n_processed})",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        path = sim_dir / f"offset_profile_{metric_key}_{label}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")

    # Log summary
    logger.section(f"Offset sweep (n={n_processed})")
    logger.info(f"Baseline vs A NIP: {base_vs_a_nip_mean}")
    for offset in offsets:
        nip_means = {
            li: torch.tensor(vs).mean().item()
            for li, vs in base_vs_ab_by_offset[offset]["nip"].items()
        }
        logger.info(f"  offset={offset}: Baseline vs AB NIP = {nip_means}")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def _make_metric_bucket() -> dict[str, dict[int, list[float]]]:
    return {"nip": {}, "cos": {}}


@dataclass
class _PrevTokenAggStats:
    n_samples: int = 0
    base_vs_a: dict[str, dict[int, list[float]]] = field(default_factory=_make_metric_bucket)
    base_vs_b_all: dict[str, dict[int, list[float]]] = field(default_factory=_make_metric_bucket)
    base_vs_b_specific: dict[str, dict[int, list[float]]] = field(
        default_factory=_make_metric_bucket
    )
    base_vs_ab_all: dict[str, dict[int, list[float]]] = field(default_factory=_make_metric_bucket)
    base_vs_ab_specific: dict[str, dict[int, list[float]]] = field(
        default_factory=_make_metric_bucket
    )
    a_vs_ab_all: dict[str, dict[int, list[float]]] = field(default_factory=_make_metric_bucket)
    a_vs_ab_specific: dict[str, dict[int, list[float]]] = field(default_factory=_make_metric_bucket)


def _accum_comparison(
    bucket: dict[str, dict[int, list[float]]],
    nip: dict[int, float],
    cos: dict[int, float],
) -> None:
    for layer_idx, val in nip.items():
        bucket["nip"].setdefault(layer_idx, []).append(val)
    for layer_idx, val in cos.items():
        bucket["cos"].setdefault(layer_idx, []).append(val)


def _run_prev_token_loop(
    target_model: LlamaSimpleMLP,
    spd_model: ComponentModel | None,
    loader: Iterable[dict[str, Tensor]],
    is_head_ablation: bool,
    parsed_heads: list[tuple[int, int]],
    parsed_components: list[tuple[str, int]],
    parsed_value_heads: list[tuple[int, int]],
    parsed_restrict_heads: list[tuple[int, int]],
    n_samples: int,
    max_plot_samples: int,
    max_pos: int,
    seq_len: int,
    run_id: str,
    label: str,
    sim_dir: Path,
    column_name: str,
    device: torch.device,
) -> None:
    stats = _PrevTokenAggStats()
    comparisons = [
        ("base_vs_a", "Baseline vs A"),
        ("base_vs_b_all", "Baseline vs B(all)"),
        ("base_vs_b_specific", "Baseline vs B(specific)"),
        ("base_vs_ab_all", "Baseline vs A+B(all)"),
        ("base_vs_ab_specific", "Baseline vs A+B(specific)"),
        ("a_vs_ab_all", "A vs A+B(all)"),
        ("a_vs_ab_specific", "A vs A+B(specific)"),
    ]

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if i >= n_samples:
                break

            input_ids: Int[Tensor, "batch pos"] = batch_data[column_name][:, :seq_len].to(device)

            sample_seq_len = input_ids.shape[1]
            rng = random.Random(i)
            t = rng.randint(1, min(sample_seq_len, max_pos) - 1)

            if is_head_ablation:
                result = _run_prev_token_head_ablation(
                    target_model, input_ids, parsed_heads, parsed_value_heads, t
                )
            elif parsed_restrict_heads:
                assert spd_model is not None
                result = _run_prev_token_head_restricted_component_ablation(
                    spd_model,
                    target_model,
                    input_ids,
                    parsed_components,
                    parsed_restrict_heads,
                    parsed_value_heads,
                    t,
                )
            else:
                assert spd_model is not None
                result = _run_prev_token_component_ablation(
                    spd_model, target_model, input_ids, parsed_components, parsed_value_heads, t
                )

            b = result.baseline_attn_outputs
            pairs = [
                (b, result.a_attn_outputs),
                (b, result.b_all_attn_outputs),
                (b, result.b_specific_attn_outputs),
                (b, result.ab_all_attn_outputs),
                (b, result.ab_specific_attn_outputs),
                (result.a_attn_outputs, result.ab_all_attn_outputs),
                (result.a_attn_outputs, result.ab_specific_attn_outputs),
            ]

            for (tag, desc), (out_a, out_b) in zip(comparisons, pairs, strict=True):
                nip_at_t, cos_at_t = compute_ablation_metrics_at_pos(out_a, out_b, t)
                _accum_comparison(getattr(stats, tag), nip_at_t, cos_at_t)

                if i < max_plot_samples:
                    sample_nip, sample_cos = compute_ablation_metrics(out_a, out_b)
                    plot_per_position_line(
                        sample_nip,
                        f"{run_id} | {desc} NIP sample {i} (t={t})",
                        sim_dir / f"{tag}_nip_sample{i}_{label}.png",
                        max_pos,
                        baseline_y=1.0,
                        ylim=(-1, 1),
                    )
                    plot_per_position_line(
                        sample_cos,
                        f"{run_id} | {desc} cos sample {i} (t={t})",
                        sim_dir / f"{tag}_cosine_sim_sample{i}_{label}.png",
                        max_pos,
                        baseline_y=1.0,
                        ylim=(-1, 1),
                    )

            stats.n_samples += 1
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{n_samples} samples")

    assert stats.n_samples > 0, "No samples processed"

    for tag, desc in comparisons:
        bucket = getattr(stats, tag)
        nip_means = {li: torch.tensor(vs).mean().item() for li, vs in bucket["nip"].items()}
        nip_stds = {li: torch.tensor(vs).std().item() for li, vs in bucket["nip"].items()}
        cos_means = {li: torch.tensor(vs).mean().item() for li, vs in bucket["cos"].items()}
        cos_stds = {li: torch.tensor(vs).std().item() for li, vs in bucket["cos"].items()}

        plot_output_similarity_bars(
            nip_means,
            nip_stds,
            f"{run_id} | {desc} NIP (n={stats.n_samples})",
            sim_dir / f"{tag}_nip_bars_{label}.png",
        )
        plot_output_similarity_bars(
            cos_means,
            cos_stds,
            f"{run_id} | {desc} cos (n={stats.n_samples})",
            sim_dir / f"{tag}_cosine_sim_bars_{label}.png",
        )

        logger.section(f"{desc} at position t")
        for layer_idx in sorted(nip_means.keys()):
            logger.info(
                f"  Layer {layer_idx}: "
                f"NIP = {nip_means[layer_idx]:.4f} ± {nip_stds[layer_idx]:.4f}, "
                f"cos = {cos_means[layer_idx]:.4f} ± {cos_stds[layer_idx]:.4f}"
            )

    logger.info(f"All plots saved to {sim_dir}")


@dataclass
class _AggStats:
    total_changed: int = 0
    total_positions: int = 0
    total_kl: float = 0.0
    n_samples: int = 0
    pos_ip_samples: dict[int, list[float]] = field(default_factory=dict)
    pos_cos_samples: dict[int, list[float]] = field(default_factory=dict)


def run_attention_ablation(
    wandb_path: ModelPath,
    heads: str | None = None,
    components: str | None = None,
    ablation_mode: AblationMode = "deterministic",
    n_samples: int = 10,
    max_plot_samples: int = 6,
    batch_size: int = 1,
    n_mask_samples: int = 10,
    pgd_steps: int = 50,
    pgd_step_size: float = 0.01,
    max_pos: int = 128,
    prev_token_test: bool = False,
    value_heads: str | None = None,
    restrict_to_heads: str | None = None,
    offset_sweep: int = 0,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    assert (heads is None) != (components is None), "Provide exactly one of --heads or --components"
    if prev_token_test:
        assert value_heads is not None, "--value_heads required when --prev_token_test is set"
    if restrict_to_heads is not None:
        assert components is not None, "--restrict_to_heads requires --components"
        assert prev_token_test or offset_sweep > 0, (
            "--restrict_to_heads requires --prev_token_test or --offset_sweep"
        )
    is_head_ablation = heads is not None
    parsed_heads = parse_heads(heads) if heads else []
    parsed_components = parse_components(components) if components else []
    parsed_value_heads = parse_heads(value_heads) if value_heads else []
    parsed_restrict_heads = parse_heads(restrict_to_heads) if restrict_to_heads else []

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model: ComponentModel | None = None
    if is_head_ablation:
        assert config.pretrained_model_name is not None
        target_model = LlamaSimpleMLP.from_pretrained(config.pretrained_model_name)
        target_model.eval()
        target_model.requires_grad_(False)
        for block in target_model._h:
            block.attn.flash_attention = False
        target_model = target_model.to(device)
    else:
        spd_model = ComponentModel.from_run_info(run_info)
        spd_model.eval()
        spd_model = spd_model.to(device)
        target_model = spd_model.target_model
        assert isinstance(target_model, LlamaSimpleMLP)
        for block in target_model._h:
            block.attn.flash_attention = False

    seq_len = target_model.config.n_ctx

    # Data loader
    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig)
    dataset_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
    )
    loader, tokenizer = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=1000,
    )

    def _short_module(m: str) -> str:
        return m.replace("_proj", "")

    if is_head_ablation:
        label = "_".join(f"L{layer}H{head}" for layer, head in parsed_heads)
    else:
        mode_suffix = "" if ablation_mode == "deterministic" else f"_{ablation_mode}"
        label = "_".join(f"{_short_module(m)}:{c}" for m, c in parsed_components) + mode_suffix
    logger.section(f"Attention ablation: {label}")
    logger.info(f"run_id={run_id}, device={device}, n_samples={n_samples}")

    attn_dir = out_dir / "attention_patterns"
    value_dir = out_dir / "value_norms"
    sim_dir = out_dir / "output_similarity"
    attn_dir.mkdir(parents=True, exist_ok=True)
    value_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    if offset_sweep > 0:
        _run_offset_sweep(
            target_model=target_model,
            spd_model=spd_model,
            loader=loader,
            is_head_ablation=is_head_ablation,
            parsed_heads=parsed_heads,
            parsed_components=parsed_components,
            parsed_restrict_heads=parsed_restrict_heads,
            n_samples=n_samples,
            max_offsets=offset_sweep,
            max_pos=max_pos,
            seq_len=seq_len,
            run_id=run_id,
            label=label,
            sim_dir=sim_dir,
            column_name=task_config.column_name,
            device=device,
        )
        return

    if prev_token_test:
        _run_prev_token_loop(
            target_model=target_model,
            spd_model=spd_model,
            loader=loader,
            is_head_ablation=is_head_ablation,
            parsed_heads=parsed_heads,
            parsed_components=parsed_components,
            parsed_value_heads=parsed_value_heads,
            parsed_restrict_heads=parsed_restrict_heads,
            n_samples=n_samples,
            max_plot_samples=max_plot_samples,
            max_pos=max_pos,
            seq_len=seq_len,
            run_id=run_id,
            label=label,
            sim_dir=sim_dir,
            column_name=task_config.column_name,
            device=device,
        )
        return

    accum_baseline_patterns: AttentionPatterns = {}
    accum_ablated_patterns: AttentionPatterns = {}
    accum_baseline_values: ValueVectors = {}
    accum_ablated_values: ValueVectors = {}
    stats = _AggStats()

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if i >= n_samples:
                break

            input_ids: Int[Tensor, "batch pos"] = batch_data[task_config.column_name][
                :, :seq_len
            ].to(device)

            sample_seq_len = input_ids.shape[1]
            rng = random.Random(i)
            ablation_pos = rng.randint(0, min(sample_seq_len, max_pos) - 1)

            if is_head_ablation:
                result = _run_head_ablation(target_model, input_ids, parsed_heads, ablation_pos)
            else:
                assert spd_model is not None
                result = _run_component_ablation(
                    spd_model,
                    target_model,
                    input_ids,
                    parsed_components,
                    ablation_mode,
                    n_mask_samples,
                    pgd_steps,
                    pgd_step_size,
                    ablation_pos,
                )

            if i < max_plot_samples:
                # Per-sample attention plots
                plot_attention_grid(
                    result.baseline_patterns,
                    f"{run_id} | Sample {i} baseline (pos={ablation_pos})",
                    attn_dir / f"baseline_sample{i}_{label}.png",
                    max_pos,
                )
                plot_attention_grid(
                    result.ablated_patterns,
                    f"{run_id} | Sample {i} ablated (pos={ablation_pos})",
                    attn_dir / f"ablated_sample{i}_{label}.png",
                    max_pos,
                )
                plot_attention_diff(
                    result.baseline_patterns,
                    result.ablated_patterns,
                    f"{run_id} | Sample {i} diff (pos={ablation_pos})",
                    attn_dir / f"diff_sample{i}_{label}.png",
                    max_pos,
                )

                # Per-sample value norm plots
                plot_value_norms(
                    result.baseline_values,
                    f"{run_id} | Sample {i} value norms baseline",
                    value_dir / f"baseline_sample{i}_{label}.png",
                    max_pos,
                )
                plot_value_norms(
                    result.ablated_values,
                    f"{run_id} | Sample {i} value norms ablated",
                    value_dir / f"ablated_sample{i}_{label}.png",
                    max_pos,
                )
                plot_value_norms_diff(
                    result.baseline_values,
                    result.ablated_values,
                    f"{run_id} | Sample {i} value norms diff",
                    value_dir / f"diff_sample{i}_{label}.png",
                    max_pos,
                )

                # Per-sample per-position line plots (sanity check)
                sample_ip, sample_cos = compute_ablation_metrics(
                    result.baseline_attn_outputs, result.ablated_attn_outputs
                )
                plot_per_position_line(
                    sample_ip,
                    f"{run_id} | Sample {i} normalized IP (ablated pos={ablation_pos})",
                    sim_dir / f"normalized_ip_sample{i}_{label}.png",
                    max_pos,
                    baseline_y=1.0,
                    ylim=(-1, 1),
                )
                plot_per_position_line(
                    sample_cos,
                    f"{run_id} | Sample {i} cosine sim (ablated pos={ablation_pos})",
                    sim_dir / f"cosine_sim_sample{i}_{label}.png",
                    max_pos,
                    baseline_y=1.0,
                    ylim=(-1, 1),
                )

            # Position-specific scalar measurement at ablated position
            pos_ip, pos_cos = compute_ablation_metrics_at_pos(
                result.baseline_attn_outputs,
                result.ablated_attn_outputs,
                ablation_pos,
            )
            for layer_idx, val in pos_ip.items():
                stats.pos_ip_samples.setdefault(layer_idx, []).append(val)
            for layer_idx, val in pos_cos.items():
                stats.pos_cos_samples.setdefault(layer_idx, []).append(val)

            # Per-sample prediction table
            n_changed = log_prediction_table(
                input_ids[0], result.baseline_logits[0], result.ablated_logits[0], tokenizer
            )

            # Accumulate stats
            stats.total_changed += n_changed
            stats.total_positions += sample_seq_len
            stats.total_kl += calc_mean_kl_divergence(
                result.baseline_logits[0], result.ablated_logits[0]
            )
            stats.n_samples += 1

            # Accumulate for mean plots
            _add_patterns(accum_baseline_patterns, result.baseline_patterns)
            _add_patterns(accum_ablated_patterns, result.ablated_patterns)
            _add_patterns(accum_baseline_values, result.baseline_values)
            _add_patterns(accum_ablated_values, result.ablated_values)
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{n_samples} samples")

    assert stats.n_samples > 0, "No samples processed"

    # Mean attention plots
    mean_baseline_patterns = _scale_patterns(accum_baseline_patterns, stats.n_samples)
    mean_ablated_patterns = _scale_patterns(accum_ablated_patterns, stats.n_samples)

    plot_attention_grid(
        mean_baseline_patterns,
        f"{run_id} | Baseline mean attention (n={stats.n_samples})",
        attn_dir / f"baseline_mean_{label}.png",
        max_pos,
    )
    plot_attention_grid(
        mean_ablated_patterns,
        f"{run_id} | Ablated mean attention (n={stats.n_samples})",
        attn_dir / f"ablated_mean_{label}.png",
        max_pos,
    )
    plot_attention_diff(
        mean_baseline_patterns,
        mean_ablated_patterns,
        f"{run_id} | Attention diff mean (n={stats.n_samples})",
        attn_dir / f"diff_mean_{label}.png",
        max_pos,
    )

    # Mean value norm plots
    mean_baseline_values = _scale_patterns(accum_baseline_values, stats.n_samples)
    mean_ablated_values = _scale_patterns(accum_ablated_values, stats.n_samples)

    plot_value_norms(
        mean_baseline_values,
        f"{run_id} | Baseline mean value norms (n={stats.n_samples})",
        value_dir / f"baseline_mean_{label}.png",
        max_pos,
    )
    plot_value_norms(
        mean_ablated_values,
        f"{run_id} | Ablated mean value norms (n={stats.n_samples})",
        value_dir / f"ablated_mean_{label}.png",
        max_pos,
    )
    plot_value_norms_diff(
        mean_baseline_values,
        mean_ablated_values,
        f"{run_id} | Value norms diff mean (n={stats.n_samples})",
        value_dir / f"diff_mean_{label}.png",
        max_pos,
    )

    # Position-specific bar charts (mean ± std across samples)
    ip_means = {li: torch.tensor(vs).mean().item() for li, vs in stats.pos_ip_samples.items()}
    ip_stds = {li: torch.tensor(vs).std().item() for li, vs in stats.pos_ip_samples.items()}
    cos_means = {li: torch.tensor(vs).mean().item() for li, vs in stats.pos_cos_samples.items()}
    cos_stds = {li: torch.tensor(vs).std().item() for li, vs in stats.pos_cos_samples.items()}

    plot_output_similarity_bars(
        ip_means,
        ip_stds,
        f"{run_id} | Normalized IP at ablated pos (n={stats.n_samples})",
        sim_dir / f"normalized_ip_bars_{label}.png",
    )
    plot_output_similarity_bars(
        cos_means,
        cos_stds,
        f"{run_id} | Cosine sim at ablated pos (n={stats.n_samples})",
        sim_dir / f"cosine_sim_bars_{label}.png",
    )

    # Summary stats
    frac_changed = stats.total_changed / stats.total_positions
    mean_kl = stats.total_kl / stats.n_samples
    logger.section("Summary")
    logger.values(
        {
            "n_samples": stats.n_samples,
            "frac_top1_changed": f"{frac_changed:.4f}",
            "mean_kl_divergence": f"{mean_kl:.6f}",
        }
    )

    logger.section("Position-specific similarity at ablated position")
    for layer_idx in sorted(ip_means.keys()):
        logger.info(
            f"  Layer {layer_idx}: "
            f"NIP = {ip_means[layer_idx]:.4f} ± {ip_stds[layer_idx]:.4f}, "
            f"cos = {cos_means[layer_idx]:.4f} ± {cos_stds[layer_idx]:.4f}"
        )

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(run_attention_ablation)
