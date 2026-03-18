"""Cluster-based geometric comparison of SPD models.

Compares how different clustering runs group subcomponents by computing cosine
similarity between cluster-level parameter vectors. Each cluster is represented
sparsely as a dict of per-module weight matrices (sum of its subcomponents' rank-1
contributions), avoiding materialization of full parameter-space vectors.

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
from spd.scripts.compare_models.compare_models import (
    max_match_stats,
    resolve_output_dir,
)
from spd.settings import SPD_OUT_DIR
from spd.utils.run_utils import save_file
from spd.utils.target_ci_solutions import permute_to_identity

matplotlib.use("Agg")

# A cluster's weight representation: only modules where it has subcomponents
ClusterWeights = dict[str, Tensor]


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


def build_cluster_weights(
    model: ComponentModel,
    merge: GroupMerge,
    labels: list[str],
) -> list[ClusterWeights]:
    """Build sparse weight representations for each cluster.

    Each cluster is a dict mapping module names to (d_in, d_out) weight matrices,
    computed as V[:, indices] @ U[indices, :] for the subcomponents in that cluster.
    """
    # Group label indices by (group_id, module_name)
    group_module_indices: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for label_idx, label in enumerate(labels):
        group_id = int(merge.group_idxs[label_idx].item())
        module_name, subcomp_idx = parse_label(label)
        assert module_name in model.components, f"Unknown module: {module_name}"
        group_module_indices[group_id][module_name].append(subcomp_idx)

    cluster_weights: list[ClusterWeights] = []
    for group_id in range(merge.k_groups):
        weights: ClusterWeights = {}
        module_indices = group_module_indices.get(group_id, {})
        for module_name, subcomp_indices in module_indices.items():
            comp = model.components[module_name]
            idx_tensor = torch.tensor(subcomp_indices, dtype=torch.long)
            V_subset = comp.V[:, idx_tensor].float()
            U_subset = comp.U[idx_tensor, :].float()
            weights[module_name] = V_subset @ U_subset
        cluster_weights.append(weights)

    return cluster_weights


MAX_DENSE_BYTES = 4 * 1024**3  # 4 GB max per dense matrix chunk


def sparse_cosine_sim_matrix(
    clusters_a: list[ClusterWeights],
    clusters_b: list[ClusterWeights],
) -> Float[Tensor, "ka kb"]:
    """Compute cosine similarity between all pairs of clusters.

    Vectorized per-module: for each module, stacks cluster weights into dense
    matrices and accumulates dot products via matmul. Large modules are processed
    in chunks to limit memory usage.
    """
    k_a, k_b = len(clusters_a), len(clusters_b)

    all_modules: set[str] = set()
    for c in clusters_a:
        all_modules.update(c.keys())
    for c in clusters_b:
        all_modules.update(c.keys())

    dot_matrix = torch.zeros(k_a, k_b)
    sq_norms_a = torch.zeros(k_a)
    sq_norms_b = torch.zeros(k_b)

    for module_name in sorted(all_modules):
        sample = next(
            (c[module_name] for c in (*clusters_a, *clusters_b) if module_name in c), None
        )
        assert sample is not None
        n_params = sample.numel()

        # Determine chunk size to stay within memory budget
        max_k = max(k_a, k_b)
        max_cols = MAX_DENSE_BYTES // (max_k * 4)  # 4 bytes per float32
        chunk_size = max(1, min(n_params, max_cols))

        flat_a = {
            i: c[module_name].reshape(-1) for i, c in enumerate(clusters_a) if module_name in c
        }
        flat_b = {
            j: c[module_name].reshape(-1) for j, c in enumerate(clusters_b) if module_name in c
        }

        for start in range(0, n_params, chunk_size):
            end = min(start + chunk_size, n_params)

            mat_a = torch.zeros(k_a, end - start)
            for i, vec in flat_a.items():
                mat_a[i] = vec[start:end]

            mat_b = torch.zeros(k_b, end - start)
            for j, vec in flat_b.items():
                mat_b[j] = vec[start:end]

            dot_matrix += mat_a @ mat_b.T
            sq_norms_a += mat_a.square().sum(dim=1)
            sq_norms_b += mat_b.square().sum(dim=1)

        logger.info(f"  Processed module {module_name} ({n_params:,} params)")

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

    cell_size = max(1.0, 40 / max(k_a, k_b))
    fig, ax = plt.subplots(figsize=(k_b * cell_size + 2, k_a * cell_size + 2))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    # Annotate cells with values when small enough
    if k_a <= 20 and k_b <= 20:
        for i in range(k_a):
            for j in range(k_b):
                val = data[i, j]
                color = "black" if val > 0.5 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("Cluster (side B)")
    ax.set_ylabel("Cluster (side A)")
    ax.set_xticks(range(k_b))
    ax.set_yticks(range(k_a))

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

    # Max-match stats (A → B)
    mean, std, min_v, max_v = max_match_stats(sim_matrix)
    lines.append("## Max-match similarity (A → B)\n")
    lines.append(f"- Mean: {mean:.4f}")
    lines.append(f"- Std: {std:.4f}")
    lines.append(f"- Min: {min_v:.4f}")
    lines.append(f"- Max: {max_v:.4f}\n")

    # Max-match stats (B → A)
    mean, std, min_v, max_v = max_match_stats(sim_matrix.T)
    lines.append("## Max-match similarity (B → A)\n")
    lines.append(f"- Mean: {mean:.4f}")
    lines.append(f"- Std: {std:.4f}")
    lines.append(f"- Min: {min_v:.4f}")
    lines.append(f"- Max: {max_v:.4f}\n")

    # Full matrix
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

    # Load both sides (reuse model if same path)
    model_a, merge_a, labels_a = _load_side(config.side_a)
    if config.side_a.spd_model_path == config.side_b.spd_model_path:
        model_b = model_a
        logger.info("Reusing SPD model for side B (same path)")
    else:
        model_b = ComponentModel.from_pretrained(config.side_b.spd_model_path)
        model_b.eval()
        model_b.requires_grad_(False)

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

    # Build cluster weights
    logger.info("Building cluster weights for side A...")
    clusters_a = build_cluster_weights(model_a, merge_a, labels_a)
    logger.info("Building cluster weights for side B...")
    clusters_b = build_cluster_weights(model_b, merge_b, labels_b)

    # Save intermediate results
    pair_name = (
        f"{config.side_a.clustering_run_id}_iter{config.side_a.iteration}"
        f"_vs_{config.side_b.clustering_run_id}_iter{config.side_b.iteration}"
    )
    pair_dir = output_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving cluster weights...")
    torch.save(clusters_a, pair_dir / "cluster_weights_a.pt")
    torch.save(clusters_b, pair_dir / "cluster_weights_b.pt")

    # Compute cosine similarity
    logger.info("Computing cosine similarity matrix...")
    sim_matrix = sparse_cosine_sim_matrix(clusters_a, clusters_b)

    # Permute columns for diagonal-like heatmap
    _, col_perm = permute_to_identity(sim_matrix)
    sim_matrix_permuted = sim_matrix[:, col_perm]

    # Save results
    torch.save(sim_matrix, pair_dir / "sim_matrix.pt")
    save_file(
        {
            "a_to_b_mean": max_match_stats(sim_matrix)[0],
            "a_to_b_std": max_match_stats(sim_matrix)[1],
            "a_to_b_min": max_match_stats(sim_matrix)[2],
            "a_to_b_max": max_match_stats(sim_matrix)[3],
            "b_to_a_mean": max_match_stats(sim_matrix.T)[0],
            "b_to_a_std": max_match_stats(sim_matrix.T)[1],
            "b_to_a_min": max_match_stats(sim_matrix.T)[2],
            "b_to_a_max": max_match_stats(sim_matrix.T)[3],
            "k_groups_a": merge_a.k_groups,
            "k_groups_b": merge_b.k_groups,
        },
        pair_dir / "results.json",
    )
    (pair_dir / "results.md").write_text(format_results_markdown(sim_matrix, config))
    save_cluster_heatmap(
        sim_matrix_permuted,
        pair_dir / "sim_heatmap.png",
        title=f"Cluster cosine sim: {config.side_a.clustering_run_id} vs {config.side_b.clustering_run_id}",
    )

    logger.info(f"Results saved to {pair_dir}")
    a_mean, _, a_min, a_max = max_match_stats(sim_matrix)
    logger.info(f"  A→B max-match: mean={a_mean:.4f}, min={a_min:.4f}, max={a_max:.4f}")


def replot(output_dir: Path | str) -> None:
    """Regenerate heatmap from saved sim_matrix.pt."""
    output_dir = Path(output_dir)
    sim_matrix = torch.load(output_dir / "sim_matrix.pt", weights_only=True)
    _, col_perm = permute_to_identity(sim_matrix)
    sim_matrix_permuted = sim_matrix[:, col_perm]
    save_cluster_heatmap(
        sim_matrix_permuted,
        output_dir / "sim_heatmap.png",
        title="Cluster cosine similarity",
    )
    logger.info(f"Replotted heatmap in {output_dir}")


if __name__ == "__main__":
    fire.Fire({"run": main, "replot": replot})
