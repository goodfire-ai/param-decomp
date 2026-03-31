"""Plot fractional attention change for top QK component pairs.

For each of the top N pairs (ranked by attention contribution at offset 1),
ablates that pair and computes the fractional attention change normalized by
the cross-head mean baseline at each offset.

Usage:
    python -m spd.scripts.attention_ablation_experiment.plot_multi_pair_frac \
        wandb:goodfire/spd/runs/s-275c8f21 \
        --layer 1 --top_n 10 --n_samples 256
"""

import random
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.scripts.attention_ablation_experiment.attention_ablation_experiment import (
    _build_deterministic_masks_multi_pos,
    _build_prev_token_component_positions,
    parse_components,
    patched_attention_forward,
)
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).parent
QK_CONTRIB_OUT = Path(__file__).parent.parent / "plot_qk_c_attention_contributions" / "out"


def _load_top_pairs(run_id: str, layer: int, top_n: int, rank_offset: int) -> list[tuple[int, int]]:
    """Load attention contribution cache and return top N (q_idx, k_idx) pairs.

    Pairs are ranked by absolute mean-across-heads contribution at rank_offset.
    """
    cache_path = QK_CONTRIB_OUT / run_id / "cache" / f"layer{layer}.npz"
    assert cache_path.exists(), f"No attention contribution cache at {cache_path}"
    data = np.load(cache_path)
    W = data["W"]  # (n_offsets, n_q_heads, n_q_alive, n_k_alive)
    q_alive = data["q_alive"].tolist()
    k_alive = data["k_alive"].tolist()
    offsets = data["offsets"].tolist()

    offset_idx = offsets.index(rank_offset)
    W_at_offset = W[offset_idx]  # (n_q_heads, n_q_alive, n_k_alive)
    W_mean = np.abs(W_at_offset).mean(axis=0)  # (n_q_alive, n_k_alive)

    flat_ranked = np.argsort(W_mean.ravel())[::-1]
    n_k = len(k_alive)
    pairs = []
    for idx in flat_ranked[:top_n]:
        qi, ki = divmod(int(idx), n_k)
        pairs.append((q_alive[qi], k_alive[ki]))
    return pairs


def plot_multi_pair_frac(
    wandb_path: ModelPath,
    layer: int = 1,
    top_n: int = 10,
    rank_offset: int = 1,
    n_samples: int = 256,
    max_offset_show: int = 20,
    seed: int = 42,
) -> None:
    """Plot fractional attention change for top QK pairs.

    Args:
        wandb_path: WandB run path.
        layer: Layer index for the attention components.
        top_n: Number of top pairs to plot.
        rank_offset: RoPE offset used to rank pairs by attention contribution.
        n_samples: Number of dataset samples to average over.
        max_offset_show: Maximum offset to show in plots.
        seed: Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    pairs = _load_top_pairs(run_id, layer, top_n, rank_offset)
    logger.info(f"Top {top_n} pairs (by offset {rank_offset}): {pairs}")

    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config

    spd_model = ComponentModel.from_run_info(run_info)
    spd_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spd_model = spd_model.to(device)
    target_model = spd_model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    for block in target_model._h:
        block.attn.flash_attention = False

    seq_len = target_model.config.n_ctx
    n_heads = target_model.config.n_head
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
    loader, _tokenizer = create_data_loader(
        dataset_config=dataset_config, batch_size=1, buffer_size=1000
    )

    out_dir = SCRIPT_DIR / "out" / run_id / "multi_pair_frac"
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets = list(range(max_offset_show + 1))

    # accum[pair_idx][head][offset] = list of per-sample attention values
    baseline_accum: dict[int, dict[int, list[float]]] = {
        h: {o: [] for o in offsets} for h in range(n_heads)
    }
    pair_accums: list[dict[int, dict[int, list[float]]]] = [
        {h: {o: [] for o in offsets} for h in range(n_heads)} for _ in range(top_n)
    ]

    q_path = f"h.{layer}.attn.q_proj"
    k_path = f"h.{layer}.attn.k_proj"

    logger.section(f"Multi-pair fractional attention change (n={n_samples}, top {top_n})")

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if i >= n_samples:
                break
            input_ids: Int[Tensor, "batch pos"] = batch_data[task_config.column_name][
                :, :seq_len
            ].to(device)

            sample_seq_len = input_ids.shape[1]
            rng = random.Random(i)
            t = rng.randint(max_offset_show, min(sample_seq_len, 128) - 1)
            bs = (input_ids.shape[0], input_ids.shape[1])

            # SPD baseline (shared across all pairs)
            baseline_components = parse_components(f"{q_path}:0,{k_path}:0")
            cp_baseline = _build_prev_token_component_positions(baseline_components, t)
            baseline_masks, _ = _build_deterministic_masks_multi_pos(
                spd_model, cp_baseline, bs, input_ids.device
            )

            with patched_attention_forward(target_model) as d:
                spd_model(input_ids, mask_infos=baseline_masks)
            spd_pat = d.patterns

            for h in range(n_heads):
                for o in offsets:
                    kp = t - o
                    if kp >= 0:
                        baseline_accum[h][o].append(spd_pat[layer][h, t, kp].item())

            # Ablate each pair
            for pi, (q_idx, k_idx) in enumerate(pairs):
                components_str = f"{q_path}:{q_idx},{k_path}:{k_idx}"
                parsed = parse_components(components_str)
                cp = _build_prev_token_component_positions(parsed, t)
                _, ablated_masks = _build_deterministic_masks_multi_pos(
                    spd_model, cp, bs, input_ids.device
                )

                with patched_attention_forward(target_model) as d:
                    spd_model(input_ids, mask_infos=ablated_masks)
                abl_pat = d.patterns

                for h in range(n_heads):
                    for o in offsets:
                        kp = t - o
                        if kp >= 0:
                            pair_accums[pi][h][o].append(abl_pat[layer][h, t, kp].item())

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{n_samples}")

    # Compute cross-head mean baseline per offset
    mean_baseline_by_offset = {
        o: np.mean([v for h in range(n_heads) for v in baseline_accum[h][o]]) for o in offsets
    }

    # Plot: one subplot per head, one line per pair
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(n_heads, 1, figsize=(14, n_heads * 2.2), squeeze=False)
    for h in range(n_heads):
        ax = axes[h, 0]
        for pi, (q_idx, k_idx) in enumerate(pairs):
            frac_means = []
            for o in offsets:
                norm = mean_baseline_by_offset[o]
                if norm > 1e-8:
                    sample_fracs = [
                        (a - b) / norm
                        for a, b in zip(pair_accums[pi][h][o], baseline_accum[h][o], strict=True)
                    ]
                else:
                    sample_fracs = [0.0] * len(baseline_accum[h][o])
                frac_means.append(np.mean(sample_fracs))
            ax.plot(
                offsets,
                frac_means,
                color=cmap(pi % 10),
                linewidth=1.2,
                label=f"Q{q_idx}→K{k_idx}",
            )
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylabel(f"H{h}", fontsize=10, fontweight="bold")
        ax.set_xlim(max_offset_show + 0.5, -0.5)
        ax.set_xticks(offsets)
        if h == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=2)
        if h == n_heads - 1:
            ax.set_xlabel("Offset from query position", fontsize=9)

    fig.suptitle(
        f"Layer {layer} fractional attention change per QK pair (n={n_samples}, top {top_n})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / f"frac_top{top_n}_n{n_samples}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    fire.Fire(plot_multi_pair_frac)
