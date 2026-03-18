"""Cluster-based geometric comparison of SPD models.

Compares how different clustering runs group subcomponents by computing cosine
similarity between cluster-level parameter vectors. Processes one module at a time
to avoid materializing full parameter-space vectors.

Usage:
    python spd/scripts/compare_models/compare_clusters.py run <config.yaml>
    python spd/scripts/compare_models/compare_clusters.py replot <output_dir>
"""

from collections import defaultdict
from pathlib import Path
from typing import ClassVar

import fire
import matplotlib
import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from pydantic import ConfigDict, Field
from torch import Tensor

from spd.base_config import BaseConfig
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_history import MergeHistory
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import Components
from spd.scripts.compare_models.compare_models import (
    max_match_stats,
    resolve_output_dir,
)
from spd.settings import SPD_OUT_DIR
from spd.utils.run_utils import save_file
from spd.utils.target_ci_solutions import permute_to_identity

matplotlib.use("Agg")

# {group_id: {module_name: [subcomp_indices]}}
GroupModuleIndices = dict[int, dict[str, list[int]]]

MAX_DENSE_BYTES = 4 * 1024**3  # 4 GB max per dense matrix chunk


class CompareClusterSide(BaseConfig):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    spd_model_path: str
    clustering_run_id: str
    iteration: int = Field(..., ge=0)


class CompareClustersConfig(BaseConfig):
    side_a: CompareClusterSide
    side_b: CompareClusterSide
    output_dir: str | None = None


def parse_label(label: str) -> tuple[str, int]:
    module, idx_str = label.rsplit(":", 1)
    return module, int(idx_str)


def _build_group_module_indices(
    merge: GroupMerge, labels: list[str], model: ComponentModel
) -> GroupModuleIndices:
    """Map each (group_id, module_name) to the subcomponent indices it contains."""
    result: GroupModuleIndices = defaultdict(lambda: defaultdict(list))
    for label_idx, label in enumerate(labels):
        group_id = int(merge.group_idxs[label_idx].item())
        module_name, subcomp_idx = parse_label(label)
        assert module_name in model.components, f"Unknown module: {module_name}"
        result[group_id][module_name].append(subcomp_idx)
    return dict(result)


def _compute_cluster_weight_row(
    comp: Components,
    subcomp_indices: list[int],
) -> Float[Tensor, " n_params"]:
    """Compute flattened weight vector for one cluster in one module."""
    idx = torch.tensor(subcomp_indices, dtype=torch.long)
    weight = comp.V[:, idx].float() @ comp.U[idx, :].float()
    return weight.reshape(-1)


def compute_cluster_cosine_sim(
    model_a: ComponentModel,
    merge_a: GroupMerge,
    labels_a: list[str],
    model_b: ComponentModel,
    merge_b: GroupMerge,
    labels_b: list[str],
) -> Float[Tensor, "ka kb"]:
    """Compute cosine similarity between clusters from two clustering runs.

    Processes one module at a time. For each module, computes per-cluster weight
    vectors and accumulates dot products and squared norms. Only clusters that
    have subcomponents in the current module contribute non-zero rows.
    """
    k_a, k_b = merge_a.k_groups, merge_b.k_groups
    gmi_a = _build_group_module_indices(merge_a, labels_a, model_a)
    gmi_b = _build_group_module_indices(merge_b, labels_b, model_b)

    all_modules = set(model_a.components.keys()) | set(model_b.components.keys())

    dot_matrix = torch.zeros(k_a, k_b)
    sq_norms_a = torch.zeros(k_a)
    sq_norms_b = torch.zeros(k_b)

    for module_name in sorted(all_modules):
        if module_name not in model_a.components or module_name not in model_b.components:
            logger.warning(f"Module {module_name} only in one model, skipping")
            continue

        comp_a = model_a.components[module_name]
        comp_b = model_b.components[module_name]
        n_params = comp_a.V.shape[0] * comp_a.U.shape[1]

        # Collect which groups have subcomponents in this module
        active_a = {
            g: gmi_a[g][module_name] for g in range(k_a) if g in gmi_a and module_name in gmi_a[g]
        }
        active_b = {
            g: gmi_b[g][module_name] for g in range(k_b) if g in gmi_b and module_name in gmi_b[g]
        }

        if not active_a and not active_b:
            continue

        # Compute weight vectors only for active clusters
        vecs_a = {
            g: _compute_cluster_weight_row(comp_a, indices) for g, indices in active_a.items()
        }
        vecs_b = {
            g: _compute_cluster_weight_row(comp_b, indices) for g, indices in active_b.items()
        }

        # Accumulate squared norms
        for g, vec in vecs_a.items():
            sq_norms_a[g] += vec.square().sum()
        for g, vec in vecs_b.items():
            sq_norms_b[g] += vec.square().sum()

        # Accumulate dot products (only between active pairs)
        if active_a and active_b:
            # Stack into dense matrices for matmul
            a_ids = sorted(active_a.keys())
            b_ids = sorted(active_b.keys())
            mat_a = torch.stack([vecs_a[g] for g in a_ids])
            mat_b = torch.stack([vecs_b[g] for g in b_ids])
            dots = mat_a @ mat_b.T
            for i, g_a in enumerate(a_ids):
                for j, g_b in enumerate(b_ids):
                    dot_matrix[g_a, g_b] += dots[i, j]

        logger.info(
            f"  {module_name}: {len(active_a)} active A, {len(active_b)} active B "
            f"({n_params:,} params)"
        )

    eps = 1e-12
    norms_a = sq_norms_a.sqrt().clamp_min(eps)
    norms_b = sq_norms_b.sqrt().clamp_min(eps)
    return dot_matrix / torch.outer(norms_a, norms_b)


def save_cluster_heatmap(
    sim_matrix: Float[Tensor, "ka kb"],
    output_path: Path,
    title: str,
) -> None:
    k_a, k_b = sim_matrix.shape
    data = sim_matrix.numpy()

    cell_size = max(0.05, min(1.0, 40 / max(k_a, k_b)))
    fig, ax = plt.subplots(figsize=(k_b * cell_size + 2, k_a * cell_size + 2))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    if k_a <= 20 and k_b <= 20:
        for i in range(k_a):
            for j in range(k_b):
                val = data[i, j]
                color = "black" if val > 0.5 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10)
        ax.set_xticks(range(k_b))
        ax.set_yticks(range(k_a))

    ax.set_title(title)
    ax.set_xlabel("Cluster (side B)")
    ax.set_ylabel("Cluster (side A)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def format_results_markdown(
    sim_matrix: Float[Tensor, "ka kb"],
    config: CompareClustersConfig,
) -> str:
    lines: list[str] = []
    lines.append("# Cluster Comparison Results\n")
    lines.append(
        f"**Side A**: model `{config.side_a.spd_model_path}`, "
        f"clustering `{config.side_a.clustering_run_id}`, "
        f"iteration {config.side_a.iteration}"
    )
    lines.append(
        f"**Side B**: model `{config.side_b.spd_model_path}`, "
        f"clustering `{config.side_b.clustering_run_id}`, "
        f"iteration {config.side_b.iteration}\n"
    )

    k_a, k_b = sim_matrix.shape
    lines.append(f"- **Clusters A**: {k_a}")
    lines.append(f"- **Clusters B**: {k_b}\n")

    mean, std, min_v, max_v = max_match_stats(sim_matrix)
    lines.append("## Max-match similarity (A -> B)\n")
    lines.append(f"- Mean: {mean:.4f}")
    lines.append(f"- Std: {std:.4f}")
    lines.append(f"- Min: {min_v:.4f}")
    lines.append(f"- Max: {max_v:.4f}\n")

    mean, std, min_v, max_v = max_match_stats(sim_matrix.T)
    lines.append("## Max-match similarity (B -> A)\n")
    lines.append(f"- Mean: {mean:.4f}")
    lines.append(f"- Std: {std:.4f}")
    lines.append(f"- Min: {min_v:.4f}")
    lines.append(f"- Max: {max_v:.4f}\n")

    if k_a <= 20 and k_b <= 20:
        lines.append("## Similarity matrix\n")
        header = "| | " + " | ".join(f"B{j}" for j in range(k_b)) + " |"
        sep = "|---|" + "|".join("---:" for _ in range(k_b)) + "|"
        lines.append(header)
        lines.append(sep)
        for i in range(k_a):
            cells = " | ".join(f"{sim_matrix[i, j]:.4f}" for j in range(k_b))
            lines.append(f"| A{i} | {cells} |")
        lines.append("")

    return "\n".join(lines)


def _load_side(
    side: CompareClusterSide,
) -> tuple[ComponentModel, GroupMerge, list[str]]:
    logger.info(f"Loading SPD model: {side.spd_model_path}")
    model = ComponentModel.from_pretrained(side.spd_model_path)
    model.eval()
    model.requires_grad_(False)

    history_path = SPD_OUT_DIR / "clustering" / "runs" / side.clustering_run_id / "history.zip"
    assert history_path.exists(), f"No history at {history_path}"
    logger.info(f"Loading clustering history: {side.clustering_run_id}")
    history = MergeHistory.read(history_path)

    assert side.iteration < history.n_iters_current, (
        f"Iteration {side.iteration} out of bounds (max {history.n_iters_current - 1})"
    )
    merge = history[side.iteration].merges
    labels = list(history.labels)

    logger.info(
        f"  iteration {side.iteration}: {merge.k_groups} clusters, {len(labels)} subcomponents"
    )
    return model, merge, labels


def main(config_path: Path | str) -> None:
    config = CompareClustersConfig.from_file(config_path)
    output_dir = resolve_output_dir(config.output_dir)

    model_a, merge_a, labels_a = _load_side(config.side_a)
    if config.side_a.spd_model_path == config.side_b.spd_model_path:
        model_b = model_a
        logger.info("Reusing SPD model for side B (same path)")
    else:
        model_b, _, _ = _load_side(config.side_b)

    history_path_b = (
        SPD_OUT_DIR / "clustering" / "runs" / config.side_b.clustering_run_id / "history.zip"
    )
    assert history_path_b.exists(), f"No history at {history_path_b}"
    history_b = MergeHistory.read(history_path_b)
    assert config.side_b.iteration < history_b.n_iters_current
    merge_b = history_b[config.side_b.iteration].merges
    labels_b = list(history_b.labels)
    logger.info(
        f"Side B: iteration {config.side_b.iteration}: "
        f"{merge_b.k_groups} clusters, {len(labels_b)} subcomponents"
    )

    pair_name = (
        f"{config.side_a.clustering_run_id}_iter{config.side_a.iteration}"
        f"_vs_{config.side_b.clustering_run_id}_iter{config.side_b.iteration}"
    )
    pair_dir = output_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing cosine similarity matrix...")
    sim_matrix = compute_cluster_cosine_sim(model_a, merge_a, labels_a, model_b, merge_b, labels_b)

    _, col_perm = permute_to_identity(sim_matrix)
    sim_matrix_permuted = sim_matrix[:, col_perm]

    torch.save(sim_matrix, pair_dir / "sim_matrix.pt")
    a_stats = max_match_stats(sim_matrix)
    b_stats = max_match_stats(sim_matrix.T)
    save_file(
        {
            "a_to_b_mean": a_stats[0],
            "a_to_b_std": a_stats[1],
            "a_to_b_min": a_stats[2],
            "a_to_b_max": a_stats[3],
            "b_to_a_mean": b_stats[0],
            "b_to_a_std": b_stats[1],
            "b_to_a_min": b_stats[2],
            "b_to_a_max": b_stats[3],
            "k_groups_a": merge_a.k_groups,
            "k_groups_b": merge_b.k_groups,
        },
        pair_dir / "results.json",
    )
    (pair_dir / "results.md").write_text(format_results_markdown(sim_matrix, config))
    save_cluster_heatmap(
        sim_matrix_permuted,
        pair_dir / "sim_heatmap.png",
        title=(
            f"Cluster cosine sim: {config.side_a.clustering_run_id} "
            f"vs {config.side_b.clustering_run_id}"
        ),
    )

    logger.info(f"Results saved to {pair_dir}")
    logger.info(
        f"  A->B max-match: mean={a_stats[0]:.4f}, min={a_stats[2]:.4f}, max={a_stats[3]:.4f}"
    )


def replot(output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    sim_matrix = torch.load(output_dir / "sim_matrix.pt", weights_only=True)
    _, col_perm = permute_to_identity(sim_matrix)
    save_cluster_heatmap(
        sim_matrix[:, col_perm],
        output_dir / "sim_heatmap.png",
        title="Cluster cosine similarity",
    )
    logger.info(f"Replotted heatmap in {output_dir}")


if __name__ == "__main__":
    fire.Fire({"run": main, "replot": replot})
