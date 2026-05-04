"""Plot k-v component co-activation heatmaps from harvest co-occurrence data.

For each layer, produces five heatmaps showing how k_proj and v_proj components
co-activate across the dataset:
  - Raw co-occurrence count (how many tokens where both fired)
  - Phi coefficient (correlation of binary firing indicators)
  - Jaccard similarity (intersection over union of firing sets)
  - P(V | K) conditional probability (fraction of K-active tokens where V is also active)
  - P(K | V) conditional probability (fraction of V-active tokens where K is also active)

All metrics are derived from the pre-computed CorrelationStorage in the harvest data.

Usage:
    python -m param_decomp.scripts.plot_kv_coactivation.plot_kv_coactivation \
        wandb:goodfire/spd/runs/<run_id>
"""

import re
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

from param_decomp.harvest.repo import HarvestRepo
from param_decomp.harvest.schemas import ComponentSummary
from param_decomp.harvest.storage import CorrelationStorage
from param_decomp.log import logger
from param_decomp.param_decomp_types import ModelPath
from param_decomp.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
MIN_MEAN_CI = 0.001

BURNT_ORANGE_CMAP = LinearSegmentedColormap.from_list("burnt_orange", ["white", "#BF5700"])


def _get_alive_indices(summary: dict[str, ComponentSummary], module_path: str) -> list[int]:
    """Return component indices for a module sorted by CI descending, filtered by threshold."""
    components = [
        (s.component_idx, s.mean_activations["causal_importance"])
        for s in summary.values()
        if s.layer == module_path and s.mean_activations["causal_importance"] > MIN_MEAN_CI
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _correlation_indices(
    corr: CorrelationStorage, module_path: str, alive_indices: list[int]
) -> list[int]:
    """Map (module_path, component_idx) pairs to indices in CorrelationStorage."""
    return [corr.key_to_idx[f"{module_path}:{idx}"] for idx in alive_indices]


def _compute_raw_cooccurrence(
    count_ij: torch.Tensor, k_corr_idx: list[int], v_corr_idx: list[int]
) -> NDArray[np.floating]:
    k_idx = torch.tensor(k_corr_idx)
    v_idx = torch.tensor(v_corr_idx)
    sub = count_ij[v_idx[:, None], k_idx[None, :]].float()
    return sub.numpy()


def _compute_phi_coefficient(
    count_ij: torch.Tensor,
    count_i: torch.Tensor,
    count_total: int,
    k_corr_idx: list[int],
    v_corr_idx: list[int],
) -> NDArray[np.floating]:
    k_idx = torch.tensor(k_corr_idx)
    v_idx = torch.tensor(v_corr_idx)
    a = count_ij[v_idx[:, None], k_idx[None, :]].float()
    n_k = count_i[k_idx].float()  # (n_k_alive,)
    n_v = count_i[v_idx].float()  # (n_v_alive,)
    n = float(count_total)

    numerator = n * a - n_v[:, None] * n_k[None, :]
    denominator = torch.sqrt(n_v[:, None] * (n - n_v[:, None]) * n_k[None, :] * (n - n_k[None, :]))
    phi = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(a))
    return phi.numpy()


def _compute_jaccard(
    count_ij: torch.Tensor,
    count_i: torch.Tensor,
    k_corr_idx: list[int],
    v_corr_idx: list[int],
) -> NDArray[np.floating]:
    k_idx = torch.tensor(k_corr_idx)
    v_idx = torch.tensor(v_corr_idx)
    intersection = count_ij[v_idx[:, None], k_idx[None, :]].float()
    union = count_i[v_idx].float()[:, None] + count_i[k_idx].float()[None, :] - intersection
    jaccard = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
    return jaccard.numpy()


def _compute_conditional_prob(
    count_ij: torch.Tensor,
    count_i: torch.Tensor,
    k_corr_idx: list[int],
    v_corr_idx: list[int],
    condition_on: str,
) -> NDArray[np.floating]:
    """P(V|K) when condition_on="k", P(K|V) when condition_on="v"."""
    k_idx = torch.tensor(k_corr_idx)
    v_idx = torch.tensor(v_corr_idx)
    # count_ij[v, k] = number of tokens where both v and k are active
    joint = count_ij[v_idx[:, None], k_idx[None, :]].float()
    if condition_on == "k":
        denom = count_i[k_idx].float()[None, :]
    else:
        denom = count_i[v_idx].float()[:, None]
    return torch.where(denom > 0, joint / denom, torch.zeros_like(joint)).numpy()


def _plot_heatmap(
    data: NDArray[np.floating],
    k_alive: list[int],
    v_alive: list[int],
    layer_idx: int,
    run_id: str,
    metric_name: str,
    cmap: str | LinearSegmentedColormap,
    vmin: float | None,
    vmax: float | None,
    out_dir: Path,
) -> None:
    n_v, n_k = data.shape
    fig, ax = plt.subplots(figsize=(max(8, n_k * 0.25), max(4, n_v * 0.18)))

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=metric_name)

    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"C{idx}" for idx in k_alive], fontsize=7, rotation=90)
    ax.set_xlabel("k_proj component (sorted by CI)")

    ax.set_yticks(range(n_v))
    ax.set_yticklabels([f"C{idx}" for idx in v_alive], fontsize=7)
    ax.set_ylabel("v_proj component (sorted by CI)")

    fig.suptitle(
        f"{run_id}  |  Layer {layer_idx} — k/v {metric_name}  (ci>{MIN_MEAN_CI})",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.12)

    path = out_dir / f"layer{layer_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_pkv_combined(
    p_v_given_k: NDArray[np.floating],
    p_k_given_v: NDArray[np.floating],
    k_alive: list[int],
    v_alive: list[int],
    layer_idx: int,
    out_dir: Path,
) -> None:
    """Side-by-side P(V|K) and P(K|V) heatmaps, sharing one row axis."""
    n_v, n_k = p_v_given_k.shape
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(max(8, n_k * 0.25), max(4, n_v * 0.12)),
        constrained_layout=True,
    )

    im = ax_left.imshow(p_v_given_k, aspect="auto", cmap=BURNT_ORANGE_CMAP, vmin=0, vmax=1)
    ax_right.imshow(p_k_given_v, aspect="auto", cmap=BURNT_ORANGE_CMAP, vmin=0, vmax=1)

    for ax, title in [(ax_left, r"$P(V_c \mid K_c)$"), (ax_right, r"$P(K_c \mid V_c)$")]:
        ax.set_xticks(range(n_k))
        ax.set_xticklabels([f"C{idx}" for idx in k_alive], fontsize=7, rotation=90)
        ax.set_xlabel("k_proj subcomponent (sorted by CI)")
        ax.set_yticks(range(n_v))
        ax.set_yticklabels([f"C{idx}" for idx in v_alive], fontsize=7)
        ax.set_ylabel("v_proj subcomponent (sorted by CI)")
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=[ax_left, ax_right], shrink=0.8, pad=0.02, label="probability")

    path = out_dir / f"layer{layer_idx}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _get_n_layers(summary: dict[str, ComponentSummary]) -> int:
    """Infer number of layers from summary keys like 'h.0.attn.k_proj'."""
    layer_indices = {
        int(m.group(1)) for s in summary.values() if (m := re.match(r"h\.(\d+)\.", s.layer))
    }
    assert layer_indices, "No layer indices found in summary"
    return max(layer_indices) + 1


def plot_kv_coactivation(wandb_path: ModelPath) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))

    out_base = SCRIPT_DIR / "out" / run_id
    raw_dir = out_base / "ci_cooccurrence"
    phi_dir = out_base / "phi_coefficient"
    jaccard_dir = out_base / "jaccard"
    p_v_given_k_dir = out_base / "p_v_given_k"
    p_k_given_v_dir = out_base / "p_k_given_v"
    p_kv_combined_dir = out_base / "p_kv_combined"
    for d in (raw_dir, phi_dir, jaccard_dir, p_v_given_k_dir, p_k_given_v_dir, p_kv_combined_dir):
        d.mkdir(parents=True, exist_ok=True)

    repo = HarvestRepo.open_most_recent(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()

    corr = repo.get_correlations()
    assert corr is not None, f"No correlation data found for {run_id}"
    logger.info(
        f"Loaded correlations: {len(corr.component_keys)} components, {corr.count_total} tokens"
    )

    n_layers = _get_n_layers(summary)
    for layer_idx in range(n_layers):
        k_path = f"h.{layer_idx}.attn.k_proj"
        v_path = f"h.{layer_idx}.attn.v_proj"

        k_alive = _get_alive_indices(summary, k_path)
        v_alive = _get_alive_indices(summary, v_path)
        logger.info(f"Layer {layer_idx}: {len(k_alive)} k components, {len(v_alive)} v components")

        if not k_alive or not v_alive:
            logger.info(f"Layer {layer_idx}: skipping (no alive k or v components)")
            continue

        k_corr_idx = _correlation_indices(corr, k_path, k_alive)
        v_corr_idx = _correlation_indices(corr, v_path, v_alive)

        # CI co-occurrence
        raw = _compute_raw_cooccurrence(corr.count_ij, k_corr_idx, v_corr_idx)
        _plot_heatmap(
            raw,
            k_alive,
            v_alive,
            layer_idx,
            run_id,
            "CI co-occurrence",
            BURNT_ORANGE_CMAP,
            0,
            None,
            raw_dir,
        )

        # Phi coefficient
        phi = _compute_phi_coefficient(
            corr.count_ij, corr.count_i, corr.count_total, k_corr_idx, v_corr_idx
        )
        phi_abs_max = float(np.abs(phi).max()) or 1.0
        _plot_heatmap(
            phi,
            k_alive,
            v_alive,
            layer_idx,
            run_id,
            "phi coefficient",
            "RdBu_r",
            -phi_abs_max,
            phi_abs_max,
            phi_dir,
        )

        # Jaccard
        jacc = _compute_jaccard(corr.count_ij, corr.count_i, k_corr_idx, v_corr_idx)
        _plot_heatmap(
            jacc,
            k_alive,
            v_alive,
            layer_idx,
            run_id,
            "Jaccard similarity",
            BURNT_ORANGE_CMAP,
            0,
            None,
            jaccard_dir,
        )

        # P(V | K)
        p_v_given_k = _compute_conditional_prob(
            corr.count_ij, corr.count_i, k_corr_idx, v_corr_idx, condition_on="k"
        )
        _plot_heatmap(
            p_v_given_k,
            k_alive,
            v_alive,
            layer_idx,
            run_id,
            "P(V | K)",
            BURNT_ORANGE_CMAP,
            0,
            None,
            p_v_given_k_dir,
        )

        # P(K | V)
        p_k_given_v = _compute_conditional_prob(
            corr.count_ij, corr.count_i, k_corr_idx, v_corr_idx, condition_on="v"
        )
        _plot_heatmap(
            p_k_given_v,
            k_alive,
            v_alive,
            layer_idx,
            run_id,
            "P(K | V)",
            BURNT_ORANGE_CMAP,
            0,
            None,
            p_k_given_v_dir,
        )

        # Combined P(V | K) + P(K | V)
        _plot_pkv_combined(
            p_v_given_k,
            p_k_given_v,
            k_alive,
            v_alive,
            layer_idx,
            p_kv_combined_dir,
        )

    logger.info(f"All plots saved to {out_base}")


if __name__ == "__main__":
    fire.Fire(plot_kv_coactivation)
