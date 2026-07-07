"""Plot the initial (pre-merge) coactivation matrix, ordered by hierarchical clustering.

For a harvest snapshot at `<path>/memberships.npz` (sparse CSC of shape
(n_samples, n_components)), this computes:

  - `coact = M.T @ M`  (component, component) integer co-firing counts
  - `p_both = coact / n_samples`  (= P(i fires ∧ j fires) per sample)
  - Jaccard distance `d(i,j) = 1 - coact / (s_i + s_j - coact)`  (s_i = diag(coact))

Hierarchical clustering uses `scipy.cluster.hierarchy.linkage` (average linkage by
default) on the condensed Jaccard distance; rows/cols are reordered by `leaves_list`.

Outputs in `<out_dir>/`:
  - `coactivation_clustered.npz` — coact, s_diag, leaf ordering, linkage Z, labels
  - `coactivation_clustered.pdf` — vector plot, zoomable to read every label
  - `coactivation_clustered.png` — preview raster

Usage:
    # Full pipeline (compute + plot):
    python -m spd.clustering.scripts.plot_initial_coactivation <harvest_dir> [...]

    # Replot from a previously saved .npz (skips M.T @ M and linkage):
    python -m spd.clustering.scripts.plot_initial_coactivation --replot_from <npz_path>
"""

import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.typing import NDArray
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from spd.log import logger

SCRIPT_DIR = Path(__file__).resolve().parent

MODULE_SHORTCODE = {
    "attn.q_proj": "q",
    "attn.k_proj": "k",
    "attn.v_proj": "v",
    "attn.o_proj": "o",
    "mlp.gate_proj": "g",
    "mlp.up_proj": "u",
    "mlp.down_proj": "d",
}


def _short_label(full_label: str) -> str:
    """`h.0.attn.k_proj:7` -> `0k:7`. Falls back to the full label if the format is unexpected."""
    layer_module, _, cidx = full_label.rpartition(":")
    if not layer_module.startswith("h."):
        return full_label
    parts = layer_module.split(".", 2)
    if len(parts) < 3:
        return full_label
    layer = parts[1]
    rest = parts[2]
    short = MODULE_SHORTCODE.get(rest, rest)
    return f"{layer}{short}:{cidx}"


def _compute_coactivation(memberships_csc: sparse.csc_matrix) -> NDArray[np.int64]:
    logger.info(
        f"computing M.T @ M for shape {memberships_csc.shape} (nnz={memberships_csc.nnz:,})"
    )
    m_i32 = memberships_csc.astype(np.int32)
    coact_sparse = m_i32.T @ m_i32
    coact = coact_sparse.toarray().astype(np.int64)
    assert coact.shape[0] == coact.shape[1] == memberships_csc.shape[1]
    return coact


def _jaccard_distance(coact: NDArray[np.int64]) -> NDArray[np.float32]:
    s = np.diagonal(coact).astype(np.float64)
    union = s[:, None] + s[None, :] - coact.astype(np.float64)
    assert (union >= 0).all()
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, coact.astype(np.float64) / union, 0.0)
    np.fill_diagonal(sim, 1.0)
    dist = (1.0 - sim).astype(np.float32)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def _cluster_ordering(
    dist: NDArray[np.float32], method: str
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    n = dist.shape[0]
    logger.info(f"linkage(method={method}) on {n} components")
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    order = leaves_list(Z)
    return order, Z


def _module_color_strip(
    labels: list[str], order: NDArray[np.intp]
) -> tuple[NDArray[np.float32], dict[str, tuple[float, float, float, float]]]:
    """Build a small color strip mapping each ordered component to its 'layer.module' identity."""
    modules: list[str] = []
    for lab in labels:
        layer_module = lab.rsplit(":", 1)[0]
        modules.append(layer_module)
    unique = sorted(set(modules))
    cmap = plt.get_cmap("tab20")
    color_for_module = {m: cmap(i % cmap.N) for i, m in enumerate(unique)}
    strip = np.array([color_for_module[modules[i]] for i in order], dtype=np.float32)
    return strip, color_for_module


def _make_plot(
    coact: NDArray[np.int64],
    order: NDArray[np.intp],
    Z: NDArray[np.float64],
    labels: list[str],
    n_samples: int,
    out_path: Path,
    method: str,
    title_extra: str,
    figsize: float,
    dpi: int,
    label_fontsize: float | None,
    save_png: bool,
) -> None:
    n_components = coact.shape[0]
    short_labels = [_short_label(labels[i]) for i in order]

    p_both = coact.astype(np.float64) / n_samples
    reordered = p_both[order][:, order]

    strip, color_for_module = _module_color_strip(labels, order)

    if label_fontsize is None:
        max_label_pts = (figsize * 72.0) / max(n_components, 1)
        label_fontsize = float(np.clip(max_label_pts * 0.9, 1.5, 7.0))
    show_tick_labels = (figsize * 72.0) / max(n_components, 1) >= 1.5

    fig = plt.figure(figsize=(figsize + 2.5, figsize + 1.0))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[0.08, 0.025, 1.0],
        height_ratios=[0.06, 1.0],
        wspace=0.01,
        hspace=0.01,
    )

    ax_dendro_left = fig.add_subplot(gs[1, 0])
    ax_strip_top = fig.add_subplot(gs[0, 2])
    ax_strip_left = fig.add_subplot(gs[1, 1])
    ax_heat = fig.add_subplot(gs[1, 2])

    dendrogram(Z, orientation="left", ax=ax_dendro_left, no_labels=True, color_threshold=0)
    ax_dendro_left.invert_yaxis()
    ax_dendro_left.set_xticks([])
    for spine in ax_dendro_left.spines.values():
        spine.set_visible(False)

    ax_strip_top.imshow(strip[None, :, :], aspect="auto")
    ax_strip_top.set_xticks([])
    ax_strip_top.set_yticks([])
    ax_strip_left.imshow(strip[:, None, :], aspect="auto")
    ax_strip_left.set_xticks([])
    ax_strip_left.set_yticks([])

    vmax = float(reordered.max())
    floor = 1.0 / n_samples
    norm = LogNorm(vmin=max(floor, 1e-9), vmax=max(vmax, 10.0 * floor))
    im = ax_heat.imshow(
        np.clip(reordered, floor, None),
        aspect="equal",
        norm=norm,
        cmap="viridis",
        interpolation="nearest",
    )

    if show_tick_labels:
        positions = np.arange(n_components)
        ax_heat.set_xticks(positions)
        ax_heat.set_yticks(positions)
        ax_heat.set_xticklabels(
            short_labels, fontsize=label_fontsize, rotation=90, family="monospace"
        )
        ax_heat.set_yticklabels(short_labels, fontsize=label_fontsize, family="monospace")
        ax_heat.tick_params(axis="both", which="both", length=0, pad=1)
    else:
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])
        logger.info(
            f"skipping per-component tick labels (n={n_components} too dense for figsize={figsize})"
        )
    ax_heat.set_xlabel(
        f"components ({n_components}), reordered by hierarchical clustering · "
        "labels: <layer><q|k|v|o|g|u|d>:<cIdx>"
    )

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01)
    cbar.set_label("P(both fire) per sample (log scale)")

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=4, label=module)
        for module, color in color_for_module.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(8, math.ceil(len(legend_handles) / 2)),
        fontsize=8,
        bbox_to_anchor=(0.55, -0.005),
        frameon=False,
    )

    fig.suptitle(
        f"Initial coactivation P(i ∧ j) | linkage={method} (Jaccard) | "
        f"n_samples={n_samples:,}, k={n_components}{title_extra}"
    )
    pdf_path = out_path / "coactivation_clustered.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info(f"wrote {pdf_path}")
    if save_png:
        png_path = out_path / "coactivation_clustered.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"wrote {png_path}")
    plt.close(fig)


def _parse_zoom_range(zoom_range: str, n: int) -> tuple[int, int]:
    start_str, _, end_str = zoom_range.partition(":")
    start = int(start_str)
    end = int(end_str)
    assert 0 <= start < end <= n, f"zoom range {zoom_range} must satisfy 0 <= start < end <= {n}"
    return start, end


def main(
    harvest_dir: str | None = None,
    replot_from: str | None = None,
    out_dir: str | None = None,
    method: str = "average",
    top_n: int | None = None,
    figsize: float = 32.0,
    dpi: int = 200,
    label_fontsize: float | None = None,
    save_png: bool = True,
    zoom_range: str | None = None,
) -> None:
    assert (harvest_dir is None) ^ (replot_from is None), (
        "specify exactly one of --harvest_dir or --replot_from"
    )

    if replot_from is not None:
        npz_path = Path(replot_from)
        assert npz_path.is_file(), f"{npz_path} does not exist"
        data = np.load(npz_path, allow_pickle=False)
        coact = data["coact"].astype(np.int64)
        order = data["order"].astype(np.intp)
        Z = data["linkage"].astype(np.float64)
        labels = [str(x) for x in data["labels"]]
        n_samples = int(data["n_samples"][0])
        out_path = Path(out_dir) if out_dir is not None else npz_path.parent
        title_extra = f" · replot from {npz_path.name}"
    else:
        assert harvest_dir is not None
        harvest_path = Path(harvest_dir)
        assert harvest_path.is_dir(), f"{harvest_dir} is not a directory"

        import json

        metadata = json.loads((harvest_path / "metadata.json").read_text())
        labels = list(metadata["labels"])
        n_samples = int(metadata["n_samples"])
        n_components = len(labels)
        logger.info(f"loaded metadata: n_samples={n_samples:,} n_components={n_components}")

        memberships_csc = sparse.load_npz(harvest_path / "memberships.npz").tocsc()
        assert memberships_csc.shape == (n_samples, n_components)

        if top_n is not None and top_n < n_components:
            counts = np.asarray(memberships_csc.sum(axis=0)).ravel()
            keep = np.argsort(counts)[::-1][:top_n]
            keep.sort()
            memberships_csc = memberships_csc[:, keep].tocsc()
            labels = [labels[i] for i in keep]
            n_components = top_n
            logger.info(f"filtered to top {top_n} components by firing count")

        coact = _compute_coactivation(memberships_csc)
        s_diag = np.diagonal(coact).astype(np.int64)

        dist = _jaccard_distance(coact)
        order, Z = _cluster_ordering(dist, method=method)

        out_path = Path(out_dir) if out_dir is not None else SCRIPT_DIR / "out" / harvest_path.name
        out_path.mkdir(parents=True, exist_ok=True)

        np.savez(
            out_path / "coactivation_clustered.npz",
            coact=coact,
            s_diag=s_diag,
            order=order,
            linkage=Z,
            labels=np.array(labels),
            n_samples=np.array([n_samples]),
        )
        logger.info(f"wrote {out_path / 'coactivation_clustered.npz'}")
        title_extra = f" · harvest {harvest_path.name}"

    if zoom_range is not None:
        start, end = _parse_zoom_range(zoom_range, len(order))
        sub_order = order[start:end]
        coact_sub = coact[sub_order][:, sub_order]
        order_sub = np.arange(end - start, dtype=np.intp)
        sub_dist = _jaccard_distance(coact_sub)
        condensed = squareform(sub_dist, checks=False)
        Z_sub = linkage(condensed, method=method)
        order_sub = leaves_list(Z_sub)
        labels_sub = [labels[i] for i in sub_order]
        zoom_out_path = out_path / f"zoom_{start}_{end}"
        zoom_out_path.mkdir(parents=True, exist_ok=True)
        title_extra = title_extra + f" · zoom [{start}:{end}]"
        _make_plot(
            coact=coact_sub,
            order=order_sub,
            Z=Z_sub,
            labels=labels_sub,
            n_samples=n_samples,
            out_path=zoom_out_path,
            method=method,
            title_extra=title_extra,
            figsize=figsize,
            dpi=dpi,
            label_fontsize=label_fontsize,
            save_png=save_png,
        )
    else:
        _make_plot(
            coact=coact,
            order=order,
            Z=Z,
            labels=labels,
            n_samples=n_samples,
            out_path=out_path,
            method=method,
            title_extra=title_extra,
            figsize=figsize,
            dpi=dpi,
            label_fontsize=label_fontsize,
            save_png=save_png,
        )


if __name__ == "__main__":
    fire.Fire(main)
