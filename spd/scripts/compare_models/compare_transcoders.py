"""Geometric consistency comparison for transcoders and CLTs.

Compares decoder weight directions across seeds by computing pairwise cosine
similarity of decoder rows. Supports both single-layer transcoders (per-layer
artifacts) and cross-layer transcoders (single artifact with triangular decoders).

Usage:
    python spd/scripts/compare_models/compare_transcoders.py run <config.yaml>
    python spd/scripts/compare_models/compare_transcoders.py replot <output_dir>
"""

import itertools
import json
from pathlib import Path
from typing import Any, Literal

import fire
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from jaxtyping import Float
from pydantic import Field
from torch import Tensor

from spd.base_config import BaseConfig
from spd.log import logger
from spd.scripts.compare_models.compare_models import (
    HEATMAP_DPI,
    HEATMAP_MAX_INCHES,
    HEATMAP_PIXELS_PER_CELL,
    max_match_stats,
    resolve_output_dir,
)
from spd.utils.run_utils import save_file
from spd.utils.target_ci_solutions import permute_to_identity

matplotlib.use("Agg")


class CompareTranscodersConfig(BaseConfig):
    model_type: Literal["transcoder", "clt"]
    wandb_project: str
    run_ids: list[str]
    alive_masks_dir: str | None = Field(
        None,
        description="Directory containing alive mask .pt files (tc_<run_id>.pt or clt_<run_id>.pt)",
    )
    output_dir: str | None = None
    label: str = Field("", description="Human-readable label for the output directory name")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _download_run_artifacts(wandb_project: str, run_id: str) -> list[tuple[str, Path]]:
    """Download model artifacts for a run. Returns list of (artifact_name, local_dir)."""
    api = wandb.Api()
    run = api.run(f"{wandb_project}/{run_id}")
    results: list[tuple[str, Path]] = []
    for art in run.logged_artifacts():
        if art.type != "model":
            continue
        from spd.settings import SPD_OUT_DIR

        dest = SPD_OUT_DIR / "checkpoints" / f"{wandb_project.replace('/', '_')}_{art.name}"
        complete_marker = dest / ".complete"
        if not complete_marker.exists():
            logger.info(f"Downloading artifact {art.name}...")
            art.download(root=str(dest))
            complete_marker.touch()
        results.append((art.name, dest))
    return results


def load_transcoder_decoders(
    wandb_project: str, run_id: str
) -> dict[int, Float[Tensor, "dict_size d_out"]]:
    """Load per-layer W_dec matrices for a transcoder run."""
    artifacts = _download_run_artifacts(wandb_project, run_id)
    decoders: dict[int, Tensor] = {}
    for name, path in artifacts:
        # Artifact names like "local_mse_k16_seed0_checkpoint_layer2_final:v0"
        for part in name.split("_"):
            if part.startswith("layer"):
                layer_idx = int(part.replace("layer", ""))
                sd = torch.load(path / "encoder.pt", map_location="cpu", weights_only=True)
                decoders[layer_idx] = sd["W_dec"].float()
                break
    assert decoders, f"No layer artifacts found for run {run_id}"
    return decoders


def load_clt_decoders(
    wandb_project: str, run_id: str
) -> dict[int, Float[Tensor, "dict_size concat_d_out"]]:
    """Load per-source-layer decoder matrices for a CLT run.

    For source layer i, concatenates decoder writes across target layers i..n-1
    into a single vector per feature: shape (dict_size, (n-i)*d_out).
    """
    artifacts = _download_run_artifacts(wandb_project, run_id)
    model_arts = [(n, p) for n, p in artifacts if "checkpoint" in n and "history" not in n]
    assert len(model_arts) == 1, f"Expected 1 model artifact for CLT, got {len(model_arts)}"
    _, path = model_arts[0]
    sd = torch.load(path / "encoder.pt", map_location="cpu", weights_only=True)

    with open(path / "config.json") as f:
        cfg = json.load(f)
    layers: list[int] = cfg["layers"]
    assert isinstance(layers, list)

    decoders: dict[int, Tensor] = {}
    for i in layers:
        # W_dec.{i} shape: (n_target_layers, dict_size, d_out)
        # Concatenate across target layers → (dict_size, n_target_layers * d_out)
        w_dec = sd[f"W_dec.{i}"].float()
        n_targets, dict_size, d_out = w_dec.shape
        decoders[i] = w_dec.permute(1, 0, 2).reshape(dict_size, n_targets * d_out)

    return decoders


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compute_decoder_cosine_sim(
    dec_a: Float[Tensor, "da d"],
    dec_b: Float[Tensor, "db d"],
) -> Float[Tensor, "da db"]:
    """Cosine similarity between decoder rows, absolute value."""
    norm_a = F.normalize(dec_a, p=2, dim=1)
    norm_b = F.normalize(dec_b, p=2, dim=1)
    return (norm_a @ norm_b.T).abs()


def save_heatmap(
    sim_matrix: Float[Tensor, "da db"],
    output_path: Path,
    title: str,
) -> None:
    data = sim_matrix.numpy()
    n_rows, n_cols = data.shape

    width = min(n_cols * HEATMAP_PIXELS_PER_CELL / HEATMAP_DPI + 2, HEATMAP_MAX_INCHES)
    height = min(n_rows * HEATMAP_PIXELS_PER_CELL / HEATMAP_DPI + 2, HEATMAP_MAX_INCHES)

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Model B features")
    ax.set_ylabel("Model A features")
    fig.tight_layout()
    fig.savefig(output_path, dpi=HEATMAP_DPI)
    plt.close(fig)


def compare_decoder_pair(
    dec_a: Float[Tensor, "da d"],
    dec_b: Float[Tensor, "db d"],
    pair_dir: Path,
    title: str,
) -> dict[str, float]:
    """Compare two sets of decoder vectors, save results to pair_dir."""
    pair_dir.mkdir(parents=True, exist_ok=True)

    sim = compute_decoder_cosine_sim(dec_a, dec_b)
    _, col_perm = permute_to_identity(sim)
    sim_permuted = sim[:, col_perm]

    torch.save(sim, pair_dir / "sim_matrix.pt")

    a_stats = max_match_stats(sim)
    b_stats = max_match_stats(sim.T)
    results = {
        "a_to_b_mean": a_stats[0],
        "a_to_b_std": a_stats[1],
        "a_to_b_min": a_stats[2],
        "a_to_b_max": a_stats[3],
        "b_to_a_mean": b_stats[0],
        "b_to_a_std": b_stats[1],
        "b_to_a_min": b_stats[2],
        "b_to_a_max": b_stats[3],
        "n_features_a": dec_a.shape[0],
        "n_features_b": dec_b.shape[0],
    }
    save_file(results, pair_dir / "results.json")
    save_heatmap(sim_permuted, pair_dir / "sim_heatmap.png", title=title)

    return results


def format_summary_markdown(
    config: CompareTranscodersConfig,
    layer_results: dict[tuple[str, str], dict[int, dict[str, float]]],
) -> str:
    lines: list[str] = []
    lines.append(f"# {config.model_type.upper()} Geometric Consistency\n")
    lines.append(f"- **Project**: `{config.wandb_project}`")
    lines.append(f"- **Runs**: {', '.join(config.run_ids)}")
    lines.append(f"- **Pairs**: {len(layer_results)}\n")
    all_layers = sorted({layer for lr in layer_results.values() for layer in lr})

    # Summary table: per-layer averages across all pairs
    lines.append("## Summary (averaged across pairs)\n")
    lines.append("| Layer | Mean | Std | Min | Max |")
    lines.append("|-------|-----:|----:|----:|----:|")
    overall_means: list[float] = []
    for layer in all_layers:
        means = [lr[layer]["a_to_b_mean"] for lr in layer_results.values() if layer in lr]
        stds = [lr[layer]["a_to_b_std"] for lr in layer_results.values() if layer in lr]
        mins = [lr[layer]["a_to_b_min"] for lr in layer_results.values() if layer in lr]
        maxs = [lr[layer]["a_to_b_max"] for lr in layer_results.values() if layer in lr]
        avg_mean = sum(means) / len(means)
        avg_std = sum(stds) / len(stds)
        avg_min = sum(mins) / len(mins)
        avg_max = sum(maxs) / len(maxs)
        overall_means.append(avg_mean)
        lines.append(
            f"| {layer} | {avg_mean:.4f} | {avg_std:.4f} | {avg_min:.4f} | {avg_max:.4f} |"
        )
    lines.append(f"| **All layers** | **{sum(overall_means) / len(overall_means):.4f}** | | | |")
    lines.append("")

    # Per-pair table with per-layer breakdown
    lines.append("## Per-pair results\n")
    layer_headers = " | ".join(f"L{layer}" for layer in all_layers)
    lines.append(f"| Pair | {layer_headers} | Mean |")
    lines.append("|------|" + "|".join("----:" for _ in all_layers) + "|-----:|")
    for (id_a, id_b), lr in layer_results.items():
        layer_vals = " | ".join(f"{lr[layer]['a_to_b_mean']:.4f}" for layer in all_layers)
        pair_mean = sum(lr[layer]["a_to_b_mean"] for layer in all_layers) / len(all_layers)
        lines.append(f"| {id_a} vs {id_b} | {layer_vals} | {pair_mean:.4f} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def main(config_path: Path | str) -> None:
    config = CompareTranscodersConfig.from_file(config_path)
    assert len(config.run_ids) >= 2, "Need at least 2 runs to compare"

    base_output_dir = resolve_output_dir(config.output_dir)
    label = config.label or f"{config.model_type}_{config.wandb_project.split('/')[-1]}"
    output_dir = base_output_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(itertools.combinations(range(len(config.run_ids)), 2))
    logger.info(f"Comparing {len(config.run_ids)} {config.model_type} runs ({len(pairs)} pairs)")

    # Load all decoders
    all_decoders: dict[str, dict[int, Tensor]] = {}
    for run_id in config.run_ids:
        logger.info(f"Loading decoders for {run_id}...")
        if config.model_type == "transcoder":
            all_decoders[run_id] = load_transcoder_decoders(config.wandb_project, run_id)
        else:
            all_decoders[run_id] = load_clt_decoders(config.wandb_project, run_id)

    # Load alive masks if provided
    all_alive_masks: dict[str, dict[int, Tensor]] | None = None
    if config.alive_masks_dir is not None:
        masks_dir = Path(config.alive_masks_dir)
        all_alive_masks = {}
        prefix = "tc" if config.model_type == "transcoder" else "clt"
        for run_id in config.run_ids:
            mask_path = masks_dir / f"{prefix}_{run_id}.pt"
            assert mask_path.exists(), f"No alive mask at {mask_path}"
            all_alive_masks[run_id] = torch.load(mask_path, weights_only=True)
            n_total = sum(m.shape[0] for m in all_alive_masks[run_id].values())
            n_alive = sum(m.sum().item() for m in all_alive_masks[run_id].values())
            logger.info(f"  {run_id} alive: {int(n_alive)}/{n_total}")

    pairwise_results: dict[tuple[str, str], dict[str, Any]] = {}
    pairwise_layer_results: dict[tuple[str, str], dict[int, dict[str, float]]] = {}

    for idx, (i, j) in enumerate(pairs):
        id_a, id_b = config.run_ids[i], config.run_ids[j]
        logger.info(f"Pair {idx + 1}/{len(pairs)}: {id_a} vs {id_b}")

        dec_a = all_decoders[id_a]
        dec_b = all_decoders[id_b]
        pair_dir = output_dir / f"{id_a}_vs_{id_b}"

        # Per-layer comparisons
        layers = sorted(set(dec_a.keys()) & set(dec_b.keys()))
        layer_results: dict[int, dict[str, float]] = {}
        all_layer_stats: list[float] = []

        for layer in layers:
            da = dec_a[layer]
            db = dec_b[layer]

            # Filter to alive features if masks provided
            if all_alive_masks is not None:
                mask_a = all_alive_masks[id_a][layer]
                mask_b = all_alive_masks[id_b][layer]
                da = da[mask_a]
                db = db[mask_b]
                logger.info(f"  layer {layer}: {da.shape[0]} alive A, {db.shape[0]} alive B")

            layer_dir = pair_dir / f"layer_{layer}"
            title = f"{config.model_type} layer {layer}: {id_a} vs {id_b}"
            r = compare_decoder_pair(da, db, layer_dir, title)
            layer_results[layer] = r
            all_layer_stats.append(r["a_to_b_mean"])
            logger.info(f"  layer {layer}: mean={r['a_to_b_mean']:.4f}")

        # Aggregate across layers — include full per-layer stats
        aggregate: dict[str, Any] = {
            "a_to_b_mean": sum(all_layer_stats) / len(all_layer_stats),
            "per_layer": {str(layer): lr for layer, lr in layer_results.items()},
        }
        save_file(aggregate, pair_dir / "results.json")
        pairwise_results[(id_a, id_b)] = aggregate
        pairwise_layer_results[(id_a, id_b)] = layer_results

    # Save summary
    summary_data = {
        "config": {
            "model_type": config.model_type,
            "wandb_project": config.wandb_project,
            "run_ids": config.run_ids,
        },
        "pairwise": {f"{a}_vs_{b}": v for (a, b), v in pairwise_results.items()},
    }
    save_file(summary_data, output_dir / "multi_summary.json")
    (output_dir / "multi_summary.md").write_text(
        format_summary_markdown(config, pairwise_layer_results)
    )

    logger.info(f"All comparisons complete! Results saved to {output_dir}")
    for (id_a, id_b), r in pairwise_results.items():
        logger.info(f"  {id_a} vs {id_b}: mean={r['a_to_b_mean']:.4f}")


def replot(output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    for sim_path in output_dir.rglob("sim_matrix.pt"):
        sim = torch.load(sim_path, weights_only=True)
        _, col_perm = permute_to_identity(sim)
        save_heatmap(
            sim[:, col_perm],
            sim_path.parent / "sim_heatmap.png",
            title=sim_path.parent.name,
        )
    logger.info(f"Replotted heatmaps in {output_dir}")


if __name__ == "__main__":
    fire.Fire({"run": main, "replot": replot})
