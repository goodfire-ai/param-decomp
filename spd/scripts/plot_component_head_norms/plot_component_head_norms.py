"""Plot subcomponent-to-head Frobenius norm heatmaps from an SPD run.

For each layer and attention projection (q/k/v/o_proj), produces a heatmap where:
  - x-axis: attention head index
  - y-axis: alive subcomponents (sorted by mean_ci descending, top N)
  - color: Frobenius norm of the subcomponent's weight slice for that head

Head slicing:
  - q/k/v_proj: heads partition rows (output dim). Head h = rows [h*head_dim : (h+1)*head_dim].
  - o_proj: heads partition columns (input dim). Head h = cols [h*head_dim : (h+1)*head_dim].

Usage:
    python -m spd.scripts.plot_component_head_norms.plot_component_head_norms wandb:goodfire/spd/runs/<run_id>
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentSummary
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearComponents
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.spd_types import ModelPath
from spd.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent
PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
MIN_MEAN_CI = 0.01


def _get_alive_indices(
    summary: dict[str, ComponentSummary], module_path: str, min_mean_ci: float
) -> list[int]:
    """Return component indices for a module sorted by CI descending, filtered by threshold."""
    components = [
        (s.component_idx, s.mean_activations["causal_importance"])
        for s in summary.values()
        if s.layer == module_path and s.mean_activations["causal_importance"] > min_mean_ci
    ]
    components.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in components]


def _component_weight(component: LinearComponents, idx: int) -> torch.Tensor:
    """Weight matrix for a single subcomponent: outer(U[c], V[:, c]) -> (d_out, d_in)."""
    return torch.outer(component.U[idx], component.V[:, idx])


def _head_norms(
    component: LinearComponents,
    alive_indices: list[int],
    proj_name: str,
    head_dim: int,
    n_heads: int,
) -> NDArray[np.floating]:
    """Compute (n_alive, n_heads) array of Frobenius norms per subcomponent per head."""
    norms = np.zeros((len(alive_indices), n_heads), dtype=np.float32)
    for row, c_idx in enumerate(alive_indices):
        w = _component_weight(component, c_idx).float()  # (d_out, d_in)
        for h in range(n_heads):
            if proj_name in ("q_proj", "k_proj", "v_proj"):
                head_slice = w[h * head_dim : (h + 1) * head_dim, :]
            else:
                head_slice = w[:, h * head_dim : (h + 1) * head_dim]
            norms[row, h] = torch.linalg.norm(head_slice).item()
    return norms


def _plot_heatmap(
    norms: NDArray[np.floating],
    alive_indices: list[int],
    n_heads: int,
    layer_idx: int,
    proj_name: str,
    run_id: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.8), max(6, len(alive_indices) * 0.25)))

    im = ax.imshow(norms, aspect="auto", cmap="Purples", vmin=0)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Frobenius norm")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_xlabel("Head")

    ax.set_yticks(range(len(alive_indices)))
    ax.set_yticklabels([f"C{idx}" for idx in alive_indices], fontsize=7)
    ax.set_ylabel("Component (sorted by CI)")

    fig.suptitle(f"{run_id}  |  Layer {layer_idx} — {proj_name}", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.08)

    path = out_dir / f"layer{layer_idx}_{proj_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_qk_combined(
    q_norms: NDArray[np.floating],
    k_norms: NDArray[np.floating],
    q_alive: list[int],
    k_alive: list[int],
    n_heads: int,
    layer_idx: int,
    out_dir: Path,
) -> None:
    max_rows = max(len(q_alive), len(k_alive))
    fig, (ax_q, ax_k) = plt.subplots(
        1,
        2,
        figsize=(max(5, n_heads * 0.6) * 2 + 1, max(4, max_rows * 0.18)),
        sharey=False,
    )

    vmax = max(q_norms.max(), k_norms.max())
    im = None

    for ax, norms, alive, proj_short, subtitle in [
        (ax_q, q_norms, q_alive, "q", r"$W_Q$"),
        (ax_k, k_norms, k_alive, "k", r"$W_K$"),
    ]:
        im = ax.imshow(norms, aspect="auto", cmap="Purples", vmin=0, vmax=vmax)

        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
        ax.set_xlabel("Head")

        ax.set_yticks(range(len(alive)))
        ax.set_yticklabels([f"{proj_short}.{idx}" for idx in alive], fontsize=7)
        if ax is ax_q:
            ax.set_ylabel("Component ID (sorted by mean CI)")
        ax.set_title(subtitle, fontsize=11)

    fig.tight_layout(w_pad=4)
    assert im is not None
    fig.colorbar(im, ax=[ax_q, ax_k], shrink=0.8, pad=0.01, label="Frobenius norm")
    path = out_dir / f"layer{layer_idx}_qk_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_component_head_norms(wandb_path: ModelPath) -> None:
    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = SPDRunInfo.from_path(wandb_path)

    out_dir = SCRIPT_DIR / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ComponentModel.from_run_info(run_info)
    model.eval()

    repo = HarvestRepo.open_most_recent(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"
    summary = repo.get_summary()

    target_model = model.target_model
    assert isinstance(target_model, LlamaSimpleMLP)
    blocks = target_model._h
    head_dim = blocks[0].attn.head_dim
    n_layers = len(blocks)
    logger.info(f"Model: {n_layers} layers, head_dim={head_dim}")

    with torch.no_grad():
        for layer_idx in range(n_layers):
            layer_data: dict[str, tuple[NDArray[np.floating], list[int], int]] = {}

            for proj_name in PROJ_NAMES:
                module_path = f"h.{layer_idx}.attn.{proj_name}"

                component = model.components[module_path]
                assert isinstance(component, LinearComponents)

                alive_indices = _get_alive_indices(summary, module_path, MIN_MEAN_CI)
                logger.info(
                    f"{module_path}: {len(alive_indices)} components with mean_ci > {MIN_MEAN_CI}"
                )

                if proj_name in ("q_proj", "k_proj", "v_proj"):
                    n_heads = component.U.shape[1] // head_dim
                else:
                    n_heads = component.V.shape[0] // head_dim
                norms = _head_norms(component, alive_indices, proj_name, head_dim, n_heads)
                layer_data[proj_name] = (norms, alive_indices, n_heads)

                _plot_heatmap(
                    norms,
                    alive_indices,
                    n_heads,
                    layer_idx,
                    proj_name,
                    run_id,
                    out_dir,
                )

            q_norms, q_alive, n_heads = layer_data["q_proj"]
            k_norms, k_alive, _ = layer_data["k_proj"]
            _plot_qk_combined(q_norms, k_norms, q_alive, k_alive, n_heads, layer_idx, out_dir)

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire(plot_component_head_norms)
