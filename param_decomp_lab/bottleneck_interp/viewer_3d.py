"""Interactive 3D PCA + grand-tour viewer for a point cloud.

Adapted from sp-viz's `analysis/viewer_3d.py` — the generic
`build_latent_viewer_data(points, thumbnails, ...)` path plus its HTML/three.js
template and grand-tour renderer. The SAE-atom path (decoder/J PCA + synthetic
atoms, which needed sp-viz's `TopologyArtifacts`) is dropped.

`build_latent_viewer_data` PCAs `points (N, D)` into 3D, renders each point as an
RGB thumbnail from the atlas, colours points by group, and supports a grand tour
through PCs. `write_viewer_html(data, out_path, ...)` injects the payload into a
self-contained HTML file (three.js loaded from CDN via importmap).
"""

import base64
import io
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------
# Generic helpers (no domain types)
# --------------------------------------------------------------------------


_VAR_EXPLAINED_EPS = 1e-9
"""Tiny floor below which a PC is treated as a true zero and dropped.
Anything above this (i.e. > 0% to display precision) is kept."""


def _trim_zero_components(
    coords: np.ndarray, var_explained: list[float]
) -> tuple[np.ndarray, list[float]]:
    """Keep only leading PCs whose variance-explained is > 0 (epsilon)."""
    n = 0
    for v in var_explained:
        if v > _VAR_EXPLAINED_EPS:
            n += 1
        else:
            break
    n = max(n, 3)  # always retain at least 3 for the axis UI
    return coords[:, :n], var_explained[:n]


def _pca_rows(X: np.ndarray) -> dict[str, Any]:
    """Mean-centred SVD on rows of ``X``. Returns float32 coords (N, k)
    and variance-explained list (length k), with trailing-zero PCs trimmed.
    """
    centered = np.ascontiguousarray(X, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt.T  # (N, min(N, D))
    total = float((sv**2).sum())
    var_explained = (sv**2 / max(total, 1e-30)).tolist()
    coords, var_explained = _trim_zero_components(coords.astype(np.float32), var_explained)
    return {"coords": coords, "var_explained": var_explained}


def _pack_coords(coords: np.ndarray) -> dict[str, Any]:
    """Float32 coords as base64, per-axis pre-normalised to unit std so JS
    doesn't redo the work on every slider tweak."""
    arr = np.ascontiguousarray(coords, dtype=np.float32)
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    normed = ((arr - mu) / sd).astype(np.float32)
    return {
        "buffer_b64": base64.b64encode(normed.tobytes()).decode("ascii"),
        "n_components": int(normed.shape[1]),
    }


def _build_atlas_from_thumbnails(thumbnails: np.ndarray) -> tuple[str, int]:
    """Pack ``(N, 3, P, P)`` float-`[0,1]` thumbnails into one square PNG atlas.

    Returns ``(base64_png, atlas_side)`` so JS can compute per-point UV offsets.
    """
    n, c, p, p2 = thumbnails.shape
    assert c == 3 and p == p2, f"Expected (N, 3, P, P); got {thumbnails.shape}"
    side = int(np.ceil(np.sqrt(n)))
    atlas = np.ones((side * p, side * p, 3), dtype=np.float32)
    for i in range(n):
        rgb_np = np.transpose(thumbnails[i], (1, 2, 0))  # (P, P, 3)
        r, c_col = divmod(i, side)
        atlas[r * p : (r + 1) * p, c_col * p : (c_col + 1) * p] = rgb_np
    atlas_u8 = (atlas * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(atlas_u8).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), side


def _empty_synthetics_block() -> dict[str, Any]:
    """Placeholder for viewers that don't use atom-interpolation synthetics."""
    empty_i32 = np.zeros(0, dtype=np.int32).tobytes()
    empty_f32 = np.zeros(0, dtype=np.float32).tobytes()
    return {
        "n": 0,
        "k": 0,
        "steps": 0,
        "src_b64": base64.b64encode(empty_i32).decode("ascii"),
        "dst_b64": base64.b64encode(empty_i32).decode("ascii"),
        "alpha_b64": base64.b64encode(empty_f32).decode("ascii"),
    }


def _empty_knn_block() -> dict[str, Any]:
    """Placeholder for viewers without a neighbour graph."""
    empty_i32 = np.zeros(0, dtype=np.int32).tobytes()
    return {"k": 0, "indices_b64": base64.b64encode(empty_i32).decode("ascii")}


def _emit_viewer_json(
    *,
    n_points: int,
    patch_size: int,
    atlas_b64: str,
    atlas_side: int,
    labels: list[int],
    clusters: list[dict[str, Any]],
    firing_rate: list[float],
    point_indices: list[int],
    bases: dict[str, dict[str, Any]],
    synthetics_block: dict[str, Any],
    knn_block: dict[str, Any],
    viewer_meta: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON payload consumed by the HTML template."""
    return {
        "n_atoms": int(n_points),
        "patch_size": int(patch_size),
        "atlas_side": int(atlas_side),
        "atlas_b64": atlas_b64,
        "labels": labels,
        "clusters": clusters,
        "firing_rate": firing_rate,
        "atom_indices": point_indices,
        "bases": bases,
        "synthetics": synthetics_block,
        "knn": knn_block,
        "viewer_meta": viewer_meta,
    }


# --------------------------------------------------------------------------
# Adapter: SAE-atom data
# --------------------------------------------------------------------------


def _knn_indices_full(points: np.ndarray, k: int, batch_size: int = 2048) -> np.ndarray:
    """Per-point indices of its ``k`` nearest neighbours in latent-space
    Euclidean distance (self excluded). Returns (N, k) int32.

    Uses GPU-batched ``torch.cdist`` + ``topk`` if CUDA is available; falls
    back to sklearn's brute NearestNeighbors otherwise. The GPU path scales
    happily to ~100K points × 128 dims (where sklearn brute would crawl).
    """
    n = int(points.shape[0])
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return np.zeros((n, 0), dtype=np.int32)
    if torch.cuda.is_available():
        device = "cuda"
        t = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).to(device)
        out = torch.empty((n, k_eff), dtype=torch.long, device=device)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            d = torch.cdist(t[start:end], t)  # (chunk, n)
            # Exclude self by setting the diagonal of this chunk to +inf.
            rows = torch.arange(start, end, device=device)
            d[torch.arange(end - start, device=device), rows] = float("inf")
            _, idx = d.topk(k_eff, largest=False, dim=1)
            out[start:end] = idx
        return out.cpu().numpy().astype(np.int32)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(points)
    _, idx = nn.kneighbors(points)
    return idx[:, 1:].astype(np.int32)


def build_latent_viewer_data(
    points: np.ndarray,
    thumbnails: np.ndarray | dict[str, np.ndarray],
    labels: np.ndarray,
    groups: list[dict[str, Any]],
    *,
    patch_size: int,
    extra_bases: dict[str, np.ndarray] | None = None,
    point_indices: np.ndarray | None = None,
    point_scalar: np.ndarray | None = None,
    point_label: str = "point",
    edge_k_max: int = 20,
    atlas_labels: dict[str, str] | None = None,
    primary_basis: Literal["pca", "raw"] = "pca",
    point_active_mask: np.ndarray | None = None,
    manifold_filter_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build viewer JSON for an arbitrary (points, thumbnails) cloud.

    ``points`` (N, D): vectors to PCA-project into 3D.
    ``thumbnails``: either ``(N, 3, P, P)`` float in [0,1] (single atlas) or a
        dict ``{atlas_name: (N, 3, P, P)}`` to expose a dropdown in the viewer
        that switches between alternative thumbnail renderings of the same points.
        All atlases must have identical N and P.
    ``labels`` (N,): group id per point (matches ``groups[i]["id"]``).
    ``groups``: ``[{id, size, regime, label, colour: [r,g,b]}, ...]``.
        ``regime`` and ``label`` show up in the sidebar; ``label`` overrides
        the default "#id regime n=size" if present.
    ``extra_bases``: optional additional ``{name: (N, k) coords}`` bases.
        The primary basis (``"latent_pca"``) is always computed from ``points``.
    ``point_indices``: optional ints shown in hover tooltip. Defaults to ``range(N)``.
    ``point_scalar``: optional float per point (e.g. swept-dim id, perturbation
        magnitude) shown in hover tooltip alongside the group. Defaults to ``0``.
    ``point_label``: label prefix for hover tooltip ("point", "latent", etc.).
    ``edge_k_max``: how many nearest neighbours to compute per point for the
        viewer's k-NN edge graph. 0 disables.
    ``atlas_labels``: optional human-readable label per atlas name (multi-atlas
        only); shown in the dropdown. Defaults to the atlas name.
    """
    n = int(points.shape[0])

    # Normalise into a {name → array} dict so the rest of the function is uniform.
    if isinstance(thumbnails, dict):
        assert thumbnails, "thumbnails dict must not be empty"
        thumbnails_dict = thumbnails
    else:
        thumbnails_dict = {"default": thumbnails}
    for name, arr in thumbnails_dict.items():
        assert arr.shape[0] == n, f"atlas {name!r}: {arr.shape[0]} thumbs != {n} points"
    assert labels.shape == (n,), f"labels shape {labels.shape} != ({n},)"

    atlases: dict[str, dict[str, Any]] = {}
    for name, arr in thumbnails_dict.items():
        b64, side = _build_atlas_from_thumbnails(arr)
        atlases[name] = {
            "b64": b64,
            "side": int(side),
            "label": (atlas_labels or {}).get(name, name),
        }
    default_atlas_name = next(iter(atlases))
    atlas_b64 = atlases[default_atlas_name]["b64"]
    atlas_side = atlases[default_atlas_name]["side"]

    # Always emit both bases so the user can flip at runtime via the radio UI
    # — PCA aligns axes with leading variance, raw shows the underlying units.
    pca_res = _pca_rows(points)
    pca_basis = {
        "label": "Latent-space PCA",
        **_pack_coords(pca_res["coords"]),
        "var_explained": pca_res["var_explained"],
        "supports_synthetics": False,
        "axis_prefix": "PC",
        "axis_noun": "PCs",
    }
    raw_coords = np.ascontiguousarray(points, dtype=np.float32)
    raw_per_dim_var = raw_coords.var(axis=0)
    raw_total_var = float(raw_per_dim_var.sum())
    raw_basis = {
        "label": "Raw bottleneck dims",
        **_pack_coords(raw_coords),
        "var_explained": (raw_per_dim_var / max(raw_total_var, 1e-30)).tolist(),
        "supports_synthetics": False,
        "axis_prefix": "Dim",
        "axis_noun": "dims",
    }
    # Default-first: dict insertion order = render order; first basis = default.
    bases: dict[str, dict[str, Any]] = (
        {"pca": pca_basis, "raw": raw_basis}
        if primary_basis == "pca"
        else {"raw": raw_basis, "pca": pca_basis}
    )
    if extra_bases:
        for name, coords in extra_bases.items():
            bases[name] = {
                "label": name,
                **_pack_coords(coords),
                "var_explained": [1.0 / coords.shape[1]] * coords.shape[1],
                "supports_synthetics": False,
            }

    if point_indices is None:
        point_indices = np.arange(n, dtype=int)
    if point_scalar is None:
        point_scalar = np.zeros(n, dtype=float)

    if edge_k_max > 0 and n > 1:
        knn = _knn_indices_full(points, edge_k_max)
        knn_block = {
            "k": int(knn.shape[1]),
            "indices_b64": base64.b64encode(knn.tobytes()).decode("ascii"),
        }
    else:
        knn_block = _empty_knn_block()

    payload = _emit_viewer_json(
        n_points=n,
        patch_size=patch_size,
        atlas_b64=atlas_b64,
        atlas_side=atlas_side,
        labels=labels.astype(int).tolist(),
        clusters=list(groups),
        firing_rate=point_scalar.astype(float).tolist(),
        point_indices=point_indices.astype(int).tolist(),
        bases=bases,
        synthetics_block=_empty_synthetics_block(),
        knn_block=knn_block,
        viewer_meta={
            "point_label": point_label,
            "group_word": "group",
            "scalar_word": "value",
            "show_unvalidated_row": False,
        },
    )
    # Multi-atlas extension: lets the HTML viewer expose a dropdown when more
    # than one atlas is provided. Backward-compatible — single-atlas callers
    # still get a hidden dropdown.
    payload["atlases"] = atlases
    payload["default_atlas"] = default_atlas_name

    # Per-manifold filter: callers that know which instances were active in each
    # sample can pass a (N, n_instances) bool mask, and the viewer exposes a
    # 6×8-style toggle grid so the user can subset to "samples where manifold K
    # is active". Hidden when not provided.
    if point_active_mask is not None:
        assert point_active_mask.shape[0] == n, (
            f"point_active_mask shape {point_active_mask.shape} != ({n}, n_instances)"
        )
        mask_u8 = np.ascontiguousarray(point_active_mask, dtype=np.uint8)
        payload["point_active_mask_b64"] = base64.b64encode(mask_u8.tobytes()).decode("ascii")
        payload["point_active_mask_n_inst"] = int(point_active_mask.shape[1])
        payload["manifold_filter_meta"] = manifold_filter_meta or {}
    return payload


# --------------------------------------------------------------------------
# HTML renderer
# --------------------------------------------------------------------------


_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>3D PCA viewer — __TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #0a0a0a; color: #e4e4e7;
               font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
  #scene { position: fixed; inset: 0; }
  #sidebar { position: fixed; top: 0; left: 0; height: 100%; width: 580px;
             background: rgba(15, 15, 20, 0.92); border-right: 1px solid #27272a;
             overflow-y: auto; padding: 18px 16px; box-sizing: border-box;
             font-size: 13px; line-height: 1.45; }
  #sidebar h1 { font-size: 14px; margin: 0 0 4px 0; }
  #sidebar h2 { font-size: 12px; margin: 16px 0 6px; text-transform: uppercase;
                letter-spacing: 0.05em; color: #a1a1aa; }
  /* Collapsible panels: a clickable header with a disclosure arrow toggles its body. */
  .panel { border-top: 1px solid #1f1f23; }
  .panel > .ph { cursor: pointer; user-select: none; display: flex; align-items: center;
                 gap: 6px; margin: 0; padding: 12px 0 8px; }
  .panel > .ph::before { content: '\\25be'; font-size: 10px; color: #71717a;
                         transition: transform 0.12s; flex: 0 0 auto; }
  .panel.collapsed > .ph::before { transform: rotate(-90deg); }
  .panel.collapsed > .pb { display: none; }
  .pb { padding-bottom: 6px; }
  #sidebar label { display: block; margin: 4px 0; }
  #sidebar select, #sidebar input[type="text"] { width: 100%; padding: 4px 6px;
       background: #18181b; color: #e4e4e7; border: 1px solid #3f3f46;
       border-radius: 3px; font-size: 12px; }
  .row { display: flex; gap: 6px; margin: 4px 0; }
  .row > * { flex: 1; }
  .axis-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .axis-name { width: 14px; font-weight: 600; color: #a1a1aa; font-size: 12px; }
  .axis-slider { flex: 1; min-width: 0; }
  .axis-label { font-size: 11px; min-width: 90px; text-align: right;
                font-variant-numeric: tabular-nums; }
  .cluster-row { display: flex; align-items: center; gap: 6px;
                 padding: 2px 4px; cursor: pointer; border-radius: 3px; }
  .cluster-row:hover { background: #27272a; }
  .swatch { width: 12px; height: 12px; border-radius: 2px; flex: 0 0 12px;
            border: 1px solid #3f3f46; }
  .muted { color: #71717a; font-size: 11px; }
  .comp-id { color: #71717a; font-size: 10px; margin-left: 4px; }
  #hover { position: fixed; left: 596px; top: 12px; padding: 8px 10px;
           background: rgba(15,15,20,0.92); border: 1px solid #27272a;
           border-radius: 4px; font-size: 12px; pointer-events: none;
           display: none; }
  /* Bottom bar: the focused token's sequence on one line, centred on that token,
     flanked by the step arrows; a back button restores the previous focus. */
  #seqBar { position: fixed; left: 600px; right: 12px; bottom: 12px; display: none;
            align-items: center; gap: 6px; padding: 6px 8px;
            background: rgba(15,15,20,0.94); border: 1px solid #27272a;
            border-radius: 6px; }
  #seqLine { flex: 1; overflow: hidden; white-space: nowrap; min-width: 0;
             font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
             font-size: 12px; line-height: 1.6; scroll-behavior: smooth; }
  #seqLine .focus-tok { background: #fde047; color: #000; border-radius: 2px; }
  #seqBar button { flex: 0 0 auto; }
  #seqBar .seq-meta { flex: 0 0 auto; font-size: 11px; color: #71717a;
                      max-width: 180px; overflow: hidden; text-overflow: ellipsis;
                      white-space: nowrap; }
  button { background: #3f3f46; color: #e4e4e7; border: none; padding: 4px 8px;
           border-radius: 3px; cursor: pointer; font-size: 12px; }
  button:hover { background: #52525b; }
  #legend { font-size: 11px; }
  .hidden { display: none !important; }
  #manifoldGrid { display: grid; gap: 2px; }
  .mf-cell { cursor: pointer; border: 1px solid #3f3f46; border-radius: 2px;
             aspect-ratio: 1 / 1; opacity: 0.35; }
  .mf-cell.on { opacity: 1.0; border-color: #f4f4f5; }
</style>
</head>
<body>
<div id="scene"></div>
<div id="sidebar">
  <h1>3D viewer</h1>
  <div class="muted">__SUBTITLE__</div>
  <div class="muted" id="visibleCount" style="margin-top:6px;"></div>

  <section class="panel" id="seqSection" style="display:none;">
    <h2 class="ph">Sequence</h2>
    <div class="pb">
      <div class="muted" id="seqHint">click a point — its sequence shows along the bottom</div>
      <label><input type="checkbox" id="flowLines"> flow line for this sequence (rainbow by position)</label>
      <label><input type="checkbox" id="showPivot"> marker sphere on selected point</label>
      <div style="margin-top:8px">
        <input type="range" id="flowOpacity" min="0.05" max="1" value="0.95" step="0.05" style="width:100%">
        <div class="muted" id="flowOpacityLabel">flow line opacity 0.95</div>
      </div>
      <div style="margin-top:8px">
        <input type="range" id="flowLocality" min="2" max="512" value="512" step="2" style="width:100%">
        <div class="muted" id="flowLocalityLabel">locality focus: off (whole sequence)</div>
      </div>
    </div>
  </section>

  <section class="panel" id="atlasSection">
    <h2 class="ph">Thumbnail type</h2>
    <div class="pb"><select id="atlasSelect"></select></div>
  </section>

  <section class="panel">
    <h2 class="ph">Thumbnail size</h2>
    <div class="pb">
      <input type="range" id="thumbSize" min="0.2" max="50" value="14" step="0.2" style="width:100%">
    </div>
  </section>

  <section class="panel" id="synthSection">
    <h2 class="ph">Synthetic atoms</h2>
    <div class="pb">
      <label><input type="checkbox" id="showSynth"> Show interpolated atoms</label>
      <div class="muted" id="synthHint"></div>
    </div>
  </section>

  <section class="panel" id="knnSection">
    <h2 class="ph">Neighbour graph</h2>
    <div class="pb">
      <label><input type="checkbox" id="showLines"> Show k-NN edges</label>
      <div style="margin-top:8px">
        <input type="range" id="edgeK" min="1" max="20" value="5" step="1" style="width:100%">
        <div class="muted" id="edgeKLabel">k = 5 neighbours per atom</div>
      </div>
      <div style="margin-top:8px">
        <input type="range" id="edgeWidth" min="0.5" max="6" value="2" step="0.1" style="width:100%">
        <div class="muted" id="edgeWidthLabel">line thickness 2.0 px</div>
      </div>
      <div class="muted" id="linesHint"></div>
    </div>
  </section>

  <section class="panel" id="basisSection">
    <h2 class="ph">Projection basis</h2>
    <div class="pb"><div id="basisRadios"></div></div>
  </section>

  <section class="panel">
    <h2 class="ph">Axes</h2>
    <div class="pb">
      <div class="axis-row" data-axis="0">
        <span class="axis-name">X</span>
        <input type="range" class="axis-slider" min="1" step="any" data-wheel-step="1">
        <span class="axis-label muted"></span>
      </div>
      <div class="axis-row" data-axis="1">
        <span class="axis-name">Y</span>
        <input type="range" class="axis-slider" min="1" step="any" data-wheel-step="1">
        <span class="axis-label muted"></span>
      </div>
      <div class="axis-row" data-axis="2">
        <span class="axis-name">Z</span>
        <input type="range" class="axis-slider" min="1" step="any" data-wheel-step="1">
        <span class="axis-label muted"></span>
      </div>
      <div class="muted" id="varex"></div>
    </div>
  </section>

  <section class="panel">
    <h2 class="ph">Clusters</h2>
    <div class="pb">
      <div class="row"><button id="allOn">all</button><button id="allOff">none</button></div>
      <div id="clusterList"></div>
    </div>
  </section>

  <section class="panel" id="manifoldFilterSection">
    <h2 class="ph">Manifold filter</h2>
    <div class="pb">
      <div class="row"><button id="mfAll">all</button><button id="mfNone">none</button></div>
      <div class="muted" id="mfMode" style="margin-bottom:6px">
        <label><input type="radio" name="mfmode" value="any" checked> any active</label>
        <label><input type="radio" name="mfmode" value="all"> all active</label>
      </div>
      <div id="manifoldGrid"></div>
      <div class="muted" id="mfHint" style="margin-top:6px"></div>
    </div>
  </section>

  <section class="panel" id="rimSection" style="display:none;">
    <h2 class="ph">Rim source</h2>
    <div class="pb">
      <select id="rimSource"></select>
      <div id="overlayItems" style="margin-top:6px; max-height:320px; overflow:auto;"></div>
    </div>
  </section>

  <section class="panel">
    <h2 class="ph">Border thickness</h2>
    <div class="pb">
      <input type="range" id="borderSize" min="0" max="10" value="0" step="1" style="width:100%">
      <div class="muted" id="borderLabel">no border</div>
    </div>
  </section>

  <section class="panel collapsed">
    <h2 class="ph">Grand tour</h2>
    <div class="pb">
      <button id="tourBtn">Start grand tour</button>
      <button id="resetBtn" disabled>Reset</button>
      <div style="margin-top:8px">
        <input type="range" id="tourLegSec" min="1" max="400" value="120" step="1" style="width:100%">
        <div class="muted" id="tourLegLabel">120 sec per leg</div>
      </div>
      <div style="margin-top:8px">
        <input type="range" id="tourRotSpeed" min="0" max="10" value="0" step="0.1" style="width:100%">
        <div class="muted" id="tourRotSpeedLabel">camera rotation off (drag up to spin)</div>
      </div>
      <div style="margin-top:8px">
        <input type="range" id="tourProgress" min="0" max="1" value="0" step="any" data-wheel-step="0.01" style="width:100%">
        <div class="muted" id="tourProgressLabel">Tour position — start to drag</div>
      </div>
      <div class="muted" id="tourHint">Deterministic rolling-window walk through projection axes in variance-descending order. K_tour is the smallest top-K whose cumulative variance explained ≥ 95% of the current basis. Scrub the progress slider to seek anywhere without changing play/pause state. Reset returns to the start and pauses.</div>
    </div>
  </section>
</div>
<div id="hover"></div>
<div id="seqBar">
  <button id="seqBack" title="back to previous focus (history)" disabled>&#8617; back</button>
  <button id="seqPrev" title="previous token (&larr;)" disabled>&#9664;</button>
  <div id="seqLine"></div>
  <button id="seqNext" title="next token (&rarr;)" disabled>&#9654;</button>
  <span class="seq-meta" id="seqMeta"></span>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.169.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.169.0/examples/jsm/"
  }
}
</script>

<script id="viewer-data" type="application/json">__DATA_JSON__</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

const DATA = JSON.parse(document.getElementById('viewer-data').textContent);
const N = DATA.n_atoms;
const VIEWER_META = DATA.viewer_meta || {};
const POINT_LABEL = VIEWER_META.point_label || 'atom';

// Decode any base64 payload into a typed array view.
function _decodeB64(b64, TypedArray) {
    const bin = atob(b64);
    const len = bin.length;
    const buf = new Uint8Array(len);
    for (let i = 0; i < len; i++) buf[i] = bin.charCodeAt(i);
    return new TypedArray(buf.buffer);
}
for (const key of Object.keys(DATA.bases)) {
    const b = DATA.bases[key];
    b.coords_flat = _decodeB64(b.buffer_b64, Float32Array);
    b.stride = b.n_components;
    delete b.buffer_b64;
}

const SYN_SRC   = _decodeB64(DATA.synthetics.src_b64,   Int32Array);
const SYN_DST   = _decodeB64(DATA.synthetics.dst_b64,   Int32Array);
const SYN_ALPHA = _decodeB64(DATA.synthetics.alpha_b64, Float32Array);
const N_SYN     = SYN_SRC.length;
const TOTAL     = N + N_SYN;

const KNN   = _decodeB64(DATA.knn.indices_b64, Int32Array);
const KNN_K = DATA.knn.k;

const HAS_SYNTHETICS = N_SYN > 0;
const HAS_KNN_UI = KNN_K > 0;
if (!HAS_SYNTHETICS) document.getElementById('synthSection').classList.add('hidden');
if (!HAS_KNN_UI) document.getElementById('knnSection').classList.add('hidden');
if (Object.keys(DATA.bases).length <= 1) {
    document.getElementById('basisSection').classList.add('hidden');
}
// Multi-atlas dropdown: only useful when more than one atlas is provided.
const ATLASES = DATA.atlases || {};
const ATLAS_NAMES = Object.keys(ATLASES);
if (ATLAS_NAMES.length <= 1) {
    document.getElementById('atlasSection').classList.add('hidden');
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 5000);
camera.position.set(2, 2, 4);
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
document.getElementById('scene').appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = false;

// --- Texture atlas ---
const atlasImg = new Image();
atlasImg.src = 'data:image/png;base64,' + DATA.atlas_b64;
const atlasTex = new THREE.Texture(atlasImg);
atlasTex.magFilter = THREE.NearestFilter;
atlasTex.minFilter = THREE.NearestFilter;
atlasTex.colorSpace = THREE.NoColorSpace;
atlasImg.onload = () => { atlasTex.needsUpdate = true; };
const SIDE = DATA.atlas_side;
const CELL = 1.0 / SIDE;

function atomCellOrigin(atomIdx, outUV, outOff) {
    const r = Math.floor(atomIdx / SIDE);
    const c = atomIdx % SIDE;
    outUV[outOff + 0] = c * CELL;
    outUV[outOff + 1] = 1.0 - (r + 1) * CELL;
}

const positions = new Float32Array(TOTAL * 3);
const uvA = new Float32Array(TOTAL * 2);
const uvB = new Float32Array(TOTAL * 2);
const alphas = new Float32Array(TOTAL);
const tints = new Float32Array(TOTAL * 3);
const iVisible = new Float32Array(TOTAL).fill(1.0);
for (let s = 0; s < N_SYN; s++) iVisible[N + s] = 0.0;

const clusterColour = new Map();
for (const c of DATA.clusters) clusterColour.set(c.id, c.colour);
const origTint = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {
    const col = clusterColour.get(DATA.labels[i]) || [0.42, 0.42, 0.42];
    origTint[i * 3 + 0] = col[0];
    origTint[i * 3 + 1] = col[1];
    origTint[i * 3 + 2] = col[2];
    atomCellOrigin(i, uvA, i * 2);
    atomCellOrigin(i, uvB, i * 2);
    alphas[i] = 0.0;
    tints[i * 3 + 0] = origTint[i * 3 + 0];
    tints[i * 3 + 1] = origTint[i * 3 + 1];
    tints[i * 3 + 2] = origTint[i * 3 + 2];
}
for (let s = 0; s < N_SYN; s++) {
    const src = SYN_SRC[s];
    const dst = SYN_DST[s];
    const slot = N + s;
    atomCellOrigin(src, uvA, slot * 2);
    atomCellOrigin(dst, uvB, slot * 2);
    alphas[slot] = SYN_ALPHA[s];
    const a = SYN_ALPHA[s];
    const ia = 1.0 - a;
    tints[slot * 3 + 0] = ia * origTint[src * 3 + 0] + a * origTint[dst * 3 + 0];
    tints[slot * 3 + 1] = ia * origTint[src * 3 + 1] + a * origTint[dst * 3 + 1];
    tints[slot * 3 + 2] = ia * origTint[src * 3 + 2] + a * origTint[dst * 3 + 2];
}

const geom = new THREE.PlaneGeometry(1, 1);
const instGeom = new THREE.InstancedBufferGeometry().copy(geom);
instGeom.instanceCount = TOTAL;
instGeom.setAttribute('iOffset',  new THREE.InstancedBufferAttribute(positions, 3));
instGeom.setAttribute('iUVa',     new THREE.InstancedBufferAttribute(uvA, 2));
instGeom.setAttribute('iUVb',     new THREE.InstancedBufferAttribute(uvB, 2));
instGeom.setAttribute('iAlpha',   new THREE.InstancedBufferAttribute(alphas, 1));
instGeom.setAttribute('iTint',    new THREE.InstancedBufferAttribute(tints, 3));
instGeom.setAttribute('iVisible', new THREE.InstancedBufferAttribute(iVisible, 1));
// Per-instance hover flag: the point under the cursor gets a forced white rim so you can
// see which thumbnail a click will select (the nearest-in-screen-space pick can sit behind
// the one you meant). Independent of the rim-overlay tint and the border-thickness slider.
const iHover = new Float32Array(TOTAL);
instGeom.setAttribute('iHover',   new THREE.InstancedBufferAttribute(iHover, 1));

const material = new THREE.ShaderMaterial({
    uniforms: {
        uAtlas:  { value: atlasTex },
        uCell:   { value: CELL },
        uThumb:  { value: 0.14 },
        uBorder: { value: 0.0 },
    },
    vertexShader: `
        attribute vec3 iOffset;
        attribute vec2 iUVa;
        attribute vec2 iUVb;
        attribute float iAlpha;
        attribute vec3 iTint;
        attribute float iVisible;
        attribute float iHover;
        varying vec2 vCellUV;
        varying vec2 vUVa;
        varying vec2 vUVb;
        varying float vAlpha;
        varying vec3 vTint;
        varying float vVisible;
        varying float vHover;
        uniform float uThumb;
        void main() {
            vVisible = iVisible;
            vTint = iTint;
            vHover = iHover;
            vCellUV = uv;
            vUVa = iUVa;
            vUVb = iUVb;
            vAlpha = iAlpha;
            vec4 viewCentre = modelViewMatrix * vec4(iOffset, 1.0);
            viewCentre.xy += position.xy * uThumb;
            gl_Position = projectionMatrix * viewCentre;
        }
    `,
    fragmentShader: `
        uniform sampler2D uAtlas;
        uniform float uBorder;
        uniform float uCell;
        varying vec2 vCellUV;
        varying vec2 vUVa;
        varying vec2 vUVb;
        varying float vAlpha;
        varying vec3 vTint;
        varying float vVisible;
        varying float vHover;

        vec3 sampleBlended(vec2 cellUV) {
            vec3 a = texture2D(uAtlas, vUVa + cellUV * uCell).rgb;
            if (vAlpha <= 0.0) return a;
            vec3 b = texture2D(uAtlas, vUVb + cellUV * uCell).rgb;
            return mix(a, b, vAlpha);
        }

        void main() {
            if (vVisible < 0.5) discard;
            // The hovered thumbnail always gets a white rim, even when the border slider is
            // at zero, so the click target is unambiguous.
            float border = uBorder;
            vec3 tint = vTint;
            if (vHover > 0.5) { border = max(uBorder, 0.12); tint = vec3(1.0); }
            if (border <= 0.0) {
                gl_FragColor = vec4(sampleBlended(vCellUV), 1.0);
                return;
            }
            float d = min(min(vCellUV.x, 1.0 - vCellUV.x),
                          min(vCellUV.y, 1.0 - vCellUV.y));
            if (d < border) {
                gl_FragColor = vec4(tint, 1.0);
            } else {
                float t = 1.0 / (1.0 - 2.0 * border);
                vec2 inner = (vCellUV - border) * t;
                gl_FragColor = vec4(sampleBlended(inner), 1.0);
            }
        }
    `,
});

const mesh = new THREE.Mesh(instGeom, material);
mesh.frustumCulled = false;
scene.add(mesh);

// --- Rim overlays: the iTint rim channel is fed by a selectable source. 'cluster' uses
// the per-point cluster colour (origTint); any DATA.overlays entry (e.g. module/component
// CI-activity) rims points by which of its selected items are active there. ---
const OVERLAYS = DATA.overlays || {};
const overlaySelected = {};
const overlayThreshold = {};
for (const k of Object.keys(OVERLAYS)) {
    const ov = OVERLAYS[k];
    if (!ov.items) continue;  // 'picker' overlays (component) have no item list
    ov.colourById = new Map();
    for (const it of ov.items) ov.colourById.set(it.id, it.colour);
    overlaySelected[k] = new Set(ov.items.map(it => it.id));
    overlayThreshold[k] = (ov.default_threshold ?? 0.05);  // scored overlays only
    ov.score_max = ov.score_max ?? 1;  // slider/threshold scale for scored overlays
}
let rimMode = 'cluster';
let selectedPickerId = null;  // for 'picker' overlays (component)
const DIM_TINT = [0.16, 0.16, 0.18];
const PICKER_HILITE = [1.0, 0.85, 0.2];

// A point's rim under an overlay names one item active there.
// 'set' overlays use point_items (membership) -> first selected member.
// 'scored' overlays use point_scores (uint8, scaled so slider units = ov.score_max) ->
// the DOMINANT selected item (argmax score), provided it clears the threshold floor. This
// makes the module overlay a "which module is most engaged here" map rather than a
// first-match that saturates on one colour.
function overlayItemForPoint(ov, sel, i) {
    if (ov.point_scores) {
        const thr = overlayThreshold[rimMode] / ov.score_max * 255;
        const sc = ov.point_scores[i];
        let bestId = -1, bestSc = thr;
        for (const it of ov.items) {
            if (sel.has(it.id) && sc[it.id] >= bestSc) { bestSc = sc[it.id]; bestId = it.id; }
        }
        return bestId;
    }
    const items = ov.point_items[i];
    for (let j = 0; j < items.length; j++) if (sel.has(items[j])) return items[j];
    return -1;
}

function applyTint() {
    if (rimMode === 'cluster' || !OVERLAYS[rimMode]) {
        for (let i = 0; i < N; i++) {
            tints[i*3] = origTint[i*3];
            tints[i*3+1] = origTint[i*3+1];
            tints[i*3+2] = origTint[i*3+2];
        }
    } else if (OVERLAYS[rimMode].kind === 'picker') {
        for (let i = 0; i < N; i++) {
            tints[i*3] = DIM_TINT[0]; tints[i*3+1] = DIM_TINT[1]; tints[i*3+2] = DIM_TINT[2];
        }
        const pts = (selectedPickerId != null) ? OVERLAYS[rimMode].index[selectedPickerId] : null;
        if (pts) for (const p of pts) {
            tints[p*3] = PICKER_HILITE[0]; tints[p*3+1] = PICKER_HILITE[1]; tints[p*3+2] = PICKER_HILITE[2];
        }
    } else {
        const ov = OVERLAYS[rimMode];
        const sel = overlaySelected[rimMode];
        for (let i = 0; i < N; i++) {
            const id = overlayItemForPoint(ov, sel, i);
            const col = id >= 0 ? ov.colourById.get(id) : DIM_TINT;
            tints[i*3] = col[0]; tints[i*3+1] = col[1]; tints[i*3+2] = col[2];
        }
    }
    instGeom.attributes.iTint.needsUpdate = true;
}

// A cluster-list-style row: checkbox + colour swatch + label, matching the Clusters UI.
function overlayRow(colour, label, checked, onToggle) {
    const row = document.createElement('label');
    row.className = 'cluster-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = checked;
    cb.addEventListener('change', () => onToggle(cb.checked));
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = `rgb(${Math.round(255*colour[0])},${Math.round(255*colour[1])},${Math.round(255*colour[2])})`;
    const txt = document.createElement('span');
    txt.textContent = label;
    row.append(cb, sw, txt);
    return row;
}

function renderOverlayItems() {
    const box = document.getElementById('overlayItems');
    box.innerHTML = '';
    if (rimMode === 'cluster' || !OVERLAYS[rimMode]) return;
    const ov = OVERLAYS[rimMode];

    // 'picker' overlays (component): a search box filters the same cluster-row format,
    // each row a single-select toggle.
    if (ov.kind === 'picker') {
        const search = document.createElement('input');
        search.type = 'text';
        search.placeholder = 'search component (e.g. h.2.attn.q_proj#15)';
        search.style.width = '100%';
        const listEl = document.createElement('div');
        listEl.style.cssText = 'margin-top:4px;';
        const refresh = () => {
            const q = search.value.toLowerCase();
            listEl.innerHTML = '';
            // Match on the human label (autointerp) when present, else the component id.
            const labelOf = (id) => (ov.labels && ov.labels[id]) || ov.names[id];
            const ids = Object.keys(ov.names)
                .filter(id => labelOf(id).toLowerCase().includes(q) || ov.names[id].toLowerCase().includes(q))
                .sort((a, b) => ov.index[b].length - ov.index[a].length)
                .slice(0, 1000);
            for (const id of ids) {
                const row = overlayRow(
                    PICKER_HILITE,
                    `${labelOf(id)} (${ov.index[id].length})`,
                    id === selectedPickerId,
                    () => {
                        selectedPickerId = (selectedPickerId === id) ? null : id;
                        applyTint();
                        refresh();
                    },
                );
                // When an autointerp label is shown primarily, keep the raw component id as
                // secondary muted text so it stays identifiable.
                if (ov.labels && ov.labels[id]) {
                    const cid = document.createElement('span');
                    cid.className = 'comp-id';
                    cid.textContent = ov.names[id];
                    row.appendChild(cid);
                }
                listEl.appendChild(row);
            }
        };
        search.addEventListener('input', refresh);
        box.append(search, listEl);
        refresh();
        return;
    }

    // 'scored' overlays (module): a single module is rimmed at a time. A scrubber (◀/▶ +
    // slider) selects which one; the engagement-floor slider sets how strongly it must be
    // engaged (× its own mean) for a point to be coloured. One-at-a-time reads far better
    // than the old multi-checkbox list.
    const sel = overlaySelected[rimMode];
    if (ov.point_scores) {
        // The selection is exactly one module; derive the scrub index from it.
        if (sel.size !== 1) { sel.clear(); sel.add(ov.items[0].id); }
        let idx = ov.items.findIndex(it => sel.has(it.id));
        if (idx < 0) idx = 0;

        const floorLab = document.createElement('div');
        floorLab.className = 'muted';
        const floor = document.createElement('input');
        floor.type = 'range'; floor.min = '0'; floor.max = String(ov.score_max);
        floor.step = String(ov.score_max / 100);
        floor.value = String(overlayThreshold[rimMode]);
        floor.style.width = '100%';
        const setFloorLab = () => {
            floorLab.textContent = `min engagement ${(+floor.value).toFixed(2)}× its mean`;
        };
        setFloorLab();
        floor.addEventListener('input', () => {
            overlayThreshold[rimMode] = parseFloat(floor.value);
            setFloorLab(); applyTint();
        });

        const pickRow = document.createElement('div');
        pickRow.className = 'row';
        pickRow.style.alignItems = 'center';
        const prev = document.createElement('button'); prev.textContent = '\\u25c0'; prev.style.flex = '0 0 auto';
        const next = document.createElement('button'); next.textContent = '\\u25b6'; next.style.flex = '0 0 auto';
        const scrub = document.createElement('input');
        scrub.type = 'range'; scrub.min = '0'; scrub.max = String(ov.items.length - 1);
        scrub.step = '1'; scrub.value = String(idx); scrub.style.flex = '1';
        const nameRow = document.createElement('div');
        nameRow.className = 'cluster-row';
        const setItem = (i) => {
            idx = Math.max(0, Math.min(ov.items.length - 1, i));
            const it = ov.items[idx];
            sel.clear(); sel.add(it.id);
            scrub.value = String(idx);
            nameRow.innerHTML = '';
            const sw = document.createElement('span');
            sw.className = 'swatch';
            sw.style.background = `rgb(${Math.round(255*it.colour[0])},${Math.round(255*it.colour[1])},${Math.round(255*it.colour[2])})`;
            const txt = document.createElement('span');
            txt.textContent = `${it.name}  (${idx + 1}/${ov.items.length})`;
            nameRow.append(sw, txt);
            applyTint();
        };
        prev.addEventListener('click', () => setItem(idx - 1));
        next.addEventListener('click', () => setItem(idx + 1));
        scrub.addEventListener('input', () => setItem(parseInt(scrub.value, 10)));
        pickRow.append(prev, scrub, next);
        box.append(floor, floorLab, pickRow, nameRow);
        setItem(idx);
        return;
    }

    // 'set' overlays (membership): the multi-checkbox list with all/none.
    const controls = document.createElement('div');
    controls.className = 'row';
    const allBtn = document.createElement('button');
    allBtn.textContent = 'all';
    const noneBtn = document.createElement('button');
    noneBtn.textContent = 'none';
    const setAll = (on) => {
        if (on) for (const it of ov.items) sel.add(it.id); else sel.clear();
        renderOverlayItems(); applyTint();
    };
    allBtn.addEventListener('click', () => setAll(true));
    noneBtn.addEventListener('click', () => setAll(false));
    controls.append(allBtn, noneBtn);
    box.appendChild(controls);

    for (const it of ov.items) {
        box.appendChild(overlayRow(it.colour, it.name, sel.has(it.id), (on) => {
            if (on) sel.add(it.id); else sel.delete(it.id);
            applyTint();
        }));
    }
}

if (Object.keys(OVERLAYS).length > 0) {
    document.getElementById('rimSection').style.display = 'block';
    const rimSourceEl = document.getElementById('rimSource');
    rimSourceEl.innerHTML = '<option value="cluster">cluster</option>'
        + Object.keys(OVERLAYS).map(k => `<option value="${k}">${k}</option>`).join('');
    rimSourceEl.addEventListener('change', (e) => {
        rimMode = e.target.value;
        renderOverlayItems();
        applyTint();
    });
}

// Collapsible sidebar panels: clicking a panel header toggles its body.
for (const head of document.querySelectorAll('.panel > .ph')) {
    head.addEventListener('click', () => head.parentElement.classList.toggle('collapsed'));
}

// --- k-NN edge mesh (only meaningful when DATA.knn.k > 0) ---
const MAX_EDGES = Math.max(1, N * KNN_K);
const lineFlat = new Float32Array(MAX_EDGES * 6);
const lineGeom = new LineSegmentsGeometry();
lineGeom.setPositions(lineFlat);
const lineMat = new LineMaterial({
    color: 0xb8c0cc,
    linewidth: 2.0,
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    worldUnits: false,
});
lineMat.resolution.set(window.innerWidth, window.innerHeight);
const lineMesh = new LineSegments2(lineGeom, lineMat);
lineMesh.computeLineDistances = () => lineMesh;
lineMesh.frustumCulled = false;
lineMesh.visible = false;
scene.add(lineMesh);

let edgeK = Math.min(5, KNN_K);

// --- Per-sequence flow lines (B1): connect consecutive positions of the selected sequence,
// coloured as a rainbow gradient that sweeps with token position along the sequence. ---
function hueToRgb(h) {  // h in [0,1] -> RGB on the saturated rainbow (HSV s=v=1)
    const r = Math.abs(h * 6 - 3) - 1;
    const g = 2 - Math.abs(h * 6 - 2);
    const b = 2 - Math.abs(h * 6 - 4);
    return [Math.min(1, Math.max(0, r)), Math.min(1, Math.max(0, g)), Math.min(1, Math.max(0, b))];
}
const flowFlat = new Float32Array(Math.max(1, N) * 6);
const flowColFlat = new Float32Array(Math.max(1, N) * 6);
const flowGeom = new LineSegmentsGeometry();
flowGeom.setPositions(flowFlat);
flowGeom.setColors(flowColFlat);
const flowMat = new LineMaterial({
    vertexColors: true, linewidth: 2.5, transparent: true, opacity: 0.95,
    depthWrite: false, worldUnits: false,
});
flowMat.resolution.set(window.innerWidth, window.innerHeight);
const flowMesh = new LineSegments2(flowGeom, flowMat);
flowMesh.computeLineDistances = () => flowMesh;
flowMesh.frustumCulled = false;
flowMesh.visible = false;
scene.add(flowMesh);
let flowEnabled = false;
let flowOpacity = 0.95;       // overall flow-line transparency (material opacity)
let flowLocalitySigma = Infinity;  // gaussian width (in tokens) around the focused position;
                                   // Infinity = no locality fade (whole sequence equally lit)

function updateFlowLines() {
    if (!SEQ || !flowEnabled || pivotLock < 0) { flowMesh.visible = false; return; }
    const targetSeq = SEQ.point_seq[pivotLock];
    const focusPos = SEQ.point_pos[pivotLock];
    const seqLen = SEQ.seq_len;
    const twoSigSq = 2 * flowLocalitySigma * flowLocalitySigma;
    // Locality fade: scale each endpoint's colour toward black (the dark background) by a
    // gaussian in distance-from-focus, so far-away parts of the sequence soft-hide.
    const wt = (pos) => (twoSigSq === Infinity ? 1 : Math.exp(-((pos - focusPos) ** 2) / twoSigSq));
    let outIdx = 0, colIdx = 0, edgeCount = 0;
    for (let i = 0; i + 1 < N; i++) {
        if (SEQ.point_seq[i] !== targetSeq || SEQ.point_seq[i + 1] !== targetSeq) continue;
        flowFlat[outIdx++] = positions[i * 3]; flowFlat[outIdx++] = positions[i * 3 + 1]; flowFlat[outIdx++] = positions[i * 3 + 2];
        flowFlat[outIdx++] = positions[(i + 1) * 3]; flowFlat[outIdx++] = positions[(i + 1) * 3 + 1]; flowFlat[outIdx++] = positions[(i + 1) * 3 + 2];
        // Each endpoint takes the hue of its own token position (vertex-colour interpolation
        // sweeps the rainbow along the line), dimmed by the locality gaussian.
        const wa = wt(SEQ.point_pos[i]), wb = wt(SEQ.point_pos[i + 1]);
        const ca = hueToRgb(SEQ.point_pos[i] / Math.max(1, seqLen - 1));
        const cb = hueToRgb(SEQ.point_pos[i + 1] / Math.max(1, seqLen - 1));
        flowColFlat[colIdx++] = ca[0]*wa; flowColFlat[colIdx++] = ca[1]*wa; flowColFlat[colIdx++] = ca[2]*wa;
        flowColFlat[colIdx++] = cb[0]*wb; flowColFlat[colIdx++] = cb[1]*wb; flowColFlat[colIdx++] = cb[2]*wb;
        edgeCount++;
    }
    flowGeom.setPositions(flowFlat.subarray(0, edgeCount * 6));
    flowGeom.setColors(flowColFlat.subarray(0, edgeCount * 6));
    flowGeom.instanceCount = edgeCount;
    flowMat.opacity = flowOpacity;
    flowMesh.visible = edgeCount > 0;
}

function updateLines() {
    if (!HAS_KNN_UI || !lineMesh.visible) return;
    const k = Math.min(edgeK, KNN_K);
    let outIdx = 0;
    let edgeCount = 0;
    for (let i = 0; i < N; i++) {
        const visI = iVisible[i] > 0.5;
        if (!visI) continue;
        const px = positions[i * 3 + 0];
        const py = positions[i * 3 + 1];
        const pz = positions[i * 3 + 2];
        for (let j = 0; j < k; j++) {
            const nbr = KNN[i * KNN_K + j];
            if (iVisible[nbr] < 0.5) continue;
            lineFlat[outIdx++] = px;
            lineFlat[outIdx++] = py;
            lineFlat[outIdx++] = pz;
            lineFlat[outIdx++] = positions[nbr * 3 + 0];
            lineFlat[outIdx++] = positions[nbr * 3 + 1];
            lineFlat[outIdx++] = positions[nbr * 3 + 2];
            edgeCount++;
        }
    }
    lineGeom.attributes.instanceStart.data.needsUpdate = true;
    lineGeom.instanceCount = edgeCount;
}

// --- Per-cluster visibility ---
const visiblePerCluster = new Set(DATA.clusters.map(c => c.id));
const validatedIds = new Set(DATA.clusters.map(c => c.id));
let showUnvalidated = false;
let showSynthetics = false;

// Optional per-point active-manifold mask: (N × N_INST) packed Uint8 (0/1).
// When present, the sidebar shows a clickable manifold-filter grid; samples
// with no overlap with the user's selection (in the chosen mode) are hidden.
const HAS_MANIFOLD_FILTER = !!DATA.point_active_mask_b64;
const N_INST = DATA.point_active_mask_n_inst || 0;
const MASK = HAS_MANIFOLD_FILTER ? _decodeB64(DATA.point_active_mask_b64, Uint8Array) : null;
const selectedManifolds = new Set();
let manifoldFilterMode = 'any';

function isOriginalVisible(i) {
    const lab = DATA.labels[i];
    const clusterOk = validatedIds.has(lab) ? visiblePerCluster.has(lab) : showUnvalidated;
    if (!clusterOk) return false;
    if (!HAS_MANIFOLD_FILTER || selectedManifolds.size === 0) return true;
    const base = i * N_INST;
    if (manifoldFilterMode === 'any') {
        for (const k of selectedManifolds) {
            if (MASK[base + k]) return true;
        }
        return false;
    }
    for (const k of selectedManifolds) {
        if (!MASK[base + k]) return false;
    }
    return true;
}

function basisSupportsSynth() {
    return !!DATA.bases[currentBasis].supports_synthetics;
}

function applyVisibility() {
    for (let i = 0; i < N; i++) {
        iVisible[i] = isOriginalVisible(i) ? 1.0 : 0.0;
    }
    const synthOn = HAS_SYNTHETICS && showSynthetics && basisSupportsSynth();
    if (!synthOn) {
        for (let s = 0; s < N_SYN; s++) iVisible[N + s] = 0.0;
    } else {
        for (let s = 0; s < N_SYN; s++) {
            const src = SYN_SRC[s];
            const dst = SYN_DST[s];
            iVisible[N + s] =
                (isOriginalVisible(src) && isOriginalVisible(dst)) ? 1.0 : 0.0;
        }
    }
    instGeom.attributes.iVisible.needsUpdate = true;
    updateLines();
    updateFlowLines();
    updateOrbitTarget();
    updateVisibleCount();
}

function updateVisibleCount() {
    let n = 0;
    for (let i = 0; i < N; i++) if (iVisible[i] > 0.5) n++;
    const el = document.getElementById('visibleCount');
    if (el) el.textContent =
        `Showing ${n.toLocaleString()} / ${N.toLocaleString()} ${POINT_LABEL}s`;
}

// Centroid of currently-visible points → OrbitControls target, so toggling
// clusters re-centres the orbit on whatever the user is looking at. Camera
// position is kept (no jump); only the look-at point shifts. Falls back to
// the previous target if everything is hidden.
function updateOrbitTarget() {
    // Locked pivot: orbit around a chosen point (set by clicking it) so local
    // structure can be inspected. Clearing the lock (click empty space) restores
    // the auto-centroid behaviour below.
    if (pivotLock >= 0) {
        controls.target.set(
            positions[pivotLock * 3], positions[pivotLock * 3 + 1], positions[pivotLock * 3 + 2]
        );
        pivotMarker.position.copy(controls.target);
        pivotMarker.visible = showPivotMarker;
        controls.update();
        return;
    }
    pivotMarker.visible = false;
    let cx = 0, cy = 0, cz = 0, count = 0;
    for (let i = 0; i < TOTAL; i++) {
        if (iVisible[i] < 0.5) continue;
        cx += positions[i * 3 + 0];
        cy += positions[i * 3 + 1];
        cz += positions[i * 3 + 2];
        count++;
    }
    if (count === 0) return;
    controls.target.set(cx / count, cy / count, cz / count);
    controls.update();
}

// --- Axis pick + basis switch ---
const BASIS_KEYS = Object.keys(DATA.bases);
let currentBasis = BASIS_KEYS[0];
let axisIdx = [0, 1, 2];
const axisRows = document.querySelectorAll('.axis-row');

function clampAxisIdx() {
    const ve = DATA.bases[currentBasis].var_explained;
    for (let s = 0; s < 3; s++) {
        if (axisIdx[s] >= ve.length) axisIdx[s] = ve.length - 1;
    }
}

function configureAxisSliders() {
    const ve = DATA.bases[currentBasis].var_explained;
    const k = ve.length;
    axisRows.forEach((row, s) => {
        const slider = row.querySelector('.axis-slider');
        slider.max = String(k);
        slider.value = String(axisIdx[s] + 1);
        updateAxisLabel(s);
    });
}

function axisPrefix() {
    return DATA.bases[currentBasis].axis_prefix || 'PC';
}
function axisNoun() {
    return DATA.bases[currentBasis].axis_noun || 'PCs';
}

function updateAxisLabel(s) {
    const ve = DATA.bases[currentBasis].var_explained;
    const i = axisIdx[s];
    const row = axisRows[s];
    row.querySelector('.axis-label').textContent =
        `${axisPrefix()}${i + 1} (${(100 * ve[i]).toFixed(2)}%)`;
}

function syncAxisSlidersToTour() {
    if (tourState === 'stopped' || !tourFrameCurr) return;
    const cycle = tourFrameCurr.length / 3 | 0;
    const ve = DATA.bases[currentBasis].var_explained;
    const pos = tourLegIdx + tourT;
    const prefix = axisPrefix();
    for (let s = 0; s < 3; s++) {
        const offset = (tourStartAxes[s] + pos) % cycle;
        const slider = axisRows[s].querySelector('.axis-slider');
        slider.value = String(offset + 1);
        const dom = Math.round(offset) % cycle;
        axisRows[s].querySelector('.axis-label').textContent =
            `${prefix}${dom + 1} (${(100 * ve[dom]).toFixed(2)}%)`;
    }
}

function updateVarExSummary() {
    const ve = DATA.bases[currentBasis].var_explained;
    const sum = ve[axisIdx[0]] + ve[axisIdx[1]] + ve[axisIdx[2]];
    const label = DATA.bases[currentBasis].label || currentBasis;
    document.getElementById('varex').textContent =
        `Selected ${axisNoun()} account for ${(100*sum).toFixed(2)}% of ${label} variance.`;
}

function applyProjection() {
    const basis = DATA.bases[currentBasis];
    const flat = basis.coords_flat;
    const stride = basis.stride;
    const ax0 = axisIdx[0], ax1 = axisIdx[1], ax2 = axisIdx[2];
    for (let i = 0; i < N; i++) {
        const base = i * stride;
        positions[i * 3 + 0] = flat[base + ax0];
        positions[i * 3 + 1] = flat[base + ax1];
        positions[i * 3 + 2] = flat[base + ax2];
    }
    if (HAS_SYNTHETICS && basisSupportsSynth()) {
        for (let s = 0; s < N_SYN; s++) {
            const src = SYN_SRC[s];
            const dst = SYN_DST[s];
            const a = SYN_ALPHA[s];
            const ia = 1.0 - a;
            const slot = (N + s) * 3;
            const bs = src * stride, bd = dst * stride;
            positions[slot + 0] = ia * flat[bs + ax0] + a * flat[bd + ax0];
            positions[slot + 1] = ia * flat[bs + ax1] + a * flat[bd + ax1];
            positions[slot + 2] = ia * flat[bs + ax2] + a * flat[bd + ax2];
        }
    } else {
        for (let s = 0; s < N_SYN; s++) {
            const slot = (N + s) * 3;
            positions[slot + 0] = 0.0;
            positions[slot + 1] = 0.0;
            positions[slot + 2] = 0.0;
        }
    }
    instGeom.attributes.iOffset.needsUpdate = true;
    updateVarExSummary();
    updateLines();
    updateFlowLines();
    updateOrbitTarget();
}

// --- Grand tour ---
const K_TOUR_VAR_THRESHOLD = 0.95;
let tourState = 'stopped';
let tourFrameCurr = null;
let tourFrameTarg = null;
let tourT = 0;
let tourLegIdx = 0;
let tourStartAxes = [0, 0, 0];
let tourLegSec = 120;
let tourSpeed = 1 / (tourLegSec * 60);
let tourRotSpeed = 0.0;
let suppressProgressUpdate = false;
let tourProgressEl = null;
let tourProgressLabelEl = null;

function tourK() {
    const ve = DATA.bases[currentBasis].var_explained;
    let cum = 0;
    for (let k = 0; k < ve.length; k++) {
        cum += ve[k];
        if (cum >= K_TOUR_VAR_THRESHOLD) return k + 1;
    }
    return ve.length;
}

function makeAxisAlignedFrame(K, idx0, idx1, idx2) {
    const F = new Float32Array(3 * K);
    const used = new Set();
    const idxs = [idx0, idx1, idx2];
    for (let r = 0; r < 3; r++) {
        let p = ((idxs[r] % K) + K) % K;
        while (used.has(p)) p = (p + 1) % K;
        F[r * K + p] = 1.0;
        used.add(p);
    }
    return F;
}

function frameForLeg(K, legIdx) {
    return makeAxisAlignedFrame(
        K,
        tourStartAxes[0] + legIdx,
        tourStartAxes[1] + legIdx,
        tourStartAxes[2] + legIdx,
    );
}

function tourCycleLength() {
    return tourFrameCurr ? (tourFrameCurr.length / 3 | 0) : tourK();
}

function formatProgressLabel(legIdxFloat, cycle) {
    const legOneIdx = (Math.floor(legIdxFloat) % cycle) + 1;
    const tourFrac = legIdxFloat / cycle;
    return `Leg ${legOneIdx} / ${cycle} — ${(tourFrac * 100).toFixed(2)}% through tour`;
}

function updateProgressSliderFromState() {
    const cycle = tourCycleLength();
    const legPos = (tourLegIdx + tourT) % cycle;
    tourProgressLabelEl.textContent = formatProgressLabel(legPos, cycle);
    if (!suppressProgressUpdate) {
        tourProgressEl.value = String(legPos / cycle);
    }
}

function gramSchmidt(F, K) {
    for (let r = 0; r < 3; r++) {
        const rowOff = r * K;
        for (let q = 0; q < r; q++) {
            const qOff = q * K;
            let dot = 0;
            for (let k = 0; k < K; k++) dot += F[rowOff + k] * F[qOff + k];
            for (let k = 0; k < K; k++) F[rowOff + k] -= dot * F[qOff + k];
        }
        let nrm = 0;
        for (let k = 0; k < K; k++) nrm += F[rowOff + k] * F[rowOff + k];
        nrm = Math.sqrt(nrm);
        if (nrm < 1e-12) { F[rowOff] = 1; nrm = 1; }
        const inv = 1 / nrm;
        for (let k = 0; k < K; k++) F[rowOff + k] *= inv;
    }
}

function updateTourButtons() {
    const tb = document.getElementById('tourBtn');
    const rb = document.getElementById('resetBtn');
    if (tourState === 'stopped') {
        tb.textContent = 'Start grand tour';
        rb.disabled = true;
    } else if (tourState === 'playing') {
        tb.textContent = 'Pause';
        rb.disabled = false;
    } else {
        tb.textContent = 'Resume';
        rb.disabled = false;
    }
}

function initTourState(legIdx = 0, t = 0) {
    const K = tourK();
    tourStartAxes = [axisIdx[0], axisIdx[1], axisIdx[2]];
    tourLegIdx = legIdx;
    tourT = t;
    tourFrameCurr = frameForLeg(K, legIdx);
    tourFrameTarg = frameForLeg(K, legIdx + 1);
}

function setPlaying() {
    tourState = 'playing';
    controls.autoRotate = tourRotSpeed > 0;
    controls.autoRotateSpeed = tourRotSpeed;
    updateTourButtons();
}

function setPaused() {
    tourState = 'paused';
    controls.autoRotate = false;
    updateTourButtons();
}

function startTour() {
    if (tourState !== 'stopped') return;
    initTourState();
    setPlaying();
    updateProgressSliderFromState();
}

function pauseOrResume() {
    if (tourState === 'stopped') {
        startTour();
    } else if (tourState === 'playing') {
        setPaused();
    } else {
        setPlaying();
    }
}

function resetTour() {
    if (tourState === 'stopped') return;
    const K = tourFrameCurr.length / 3 | 0;
    tourLegIdx = 0;
    tourT = 0;
    tourFrameCurr = frameForLeg(K, 0);
    tourFrameTarg = frameForLeg(K, 1);
    setPaused();
    applyProjection();
    updateProgressSliderFromState();
}

function stopTour() {
    if (tourState === 'stopped') return;
    tourState = 'stopped';
    controls.autoRotate = false;
    axisRows.forEach((row, s) => {
        row.querySelector('.axis-slider').value = String(axisIdx[s] + 1);
        updateAxisLabel(s);
    });
    applyProjection();
    updateTourButtons();
}

function seekTour(frac) {
    if (tourState === 'stopped') {
        initTourState();
        setPaused();
    }
    const K = tourFrameCurr.length / 3 | 0;
    const cycle = K;
    const legPos = Math.max(0, Math.min(cycle - 1e-9, frac * cycle));
    tourLegIdx = Math.floor(legPos);
    tourT = legPos - tourLegIdx;
    tourFrameCurr = frameForLeg(K, tourLegIdx);
    tourFrameTarg = frameForLeg(K, tourLegIdx + 1);
    if (tourState === 'paused') projectViaTourFrame();
}

function tourStep() {
    if (tourState !== 'playing') return;
    tourT += tourSpeed;
    if (tourT >= 1.0) {
        const K = tourFrameTarg.length / 3 | 0;
        tourLegIdx += 1;
        tourT = 0;
        tourFrameCurr = tourFrameTarg;
        tourFrameTarg = frameForLeg(K, tourLegIdx + 1);
    }
    updateProgressSliderFromState();
    projectViaTourFrame();
}

function projectViaTourFrame() {
    if (!tourFrameCurr || !tourFrameTarg) return;
    const K = tourFrameCurr.length / 3 | 0;
    const t = 0.5 - 0.5 * Math.cos(Math.PI * tourT);
    const V = new Float32Array(3 * K);
    for (let i = 0; i < V.length; i++) V[i] = (1 - t) * tourFrameCurr[i] + t * tourFrameTarg[i];
    gramSchmidt(V, K);
    const basis = DATA.bases[currentBasis];
    const flat = basis.coords_flat;
    const stride = basis.stride;
    const KEFF = Math.min(K, stride);
    for (let i = 0; i < N; i++) {
        const base = i * stride;
        let x = 0, y = 0, z = 0;
        for (let k = 0; k < KEFF; k++) {
            const c = flat[base + k];
            x += V[k]         * c;
            y += V[K + k]     * c;
            z += V[2 * K + k] * c;
        }
        positions[i * 3 + 0] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;
    }
    if (HAS_SYNTHETICS && showSynthetics && basisSupportsSynth()) {
        for (let s = 0; s < N_SYN; s++) {
            const src = SYN_SRC[s];
            const dst = SYN_DST[s];
            const a = SYN_ALPHA[s];
            const ia = 1 - a;
            const slot = (N + s) * 3;
            positions[slot + 0] = ia * positions[src * 3 + 0] + a * positions[dst * 3 + 0];
            positions[slot + 1] = ia * positions[src * 3 + 1] + a * positions[dst * 3 + 1];
            positions[slot + 2] = ia * positions[src * 3 + 2] + a * positions[dst * 3 + 2];
        }
    }
    instGeom.attributes.iOffset.needsUpdate = true;
    updateLines();
    syncAxisSlidersToTour();
}

// --- Sidebar wiring ---
// Basis radios are generated from DATA.bases so any number of bases works.
const basisRadios = document.getElementById('basisRadios');
BASIS_KEYS.forEach((key, idx) => {
    const label = document.createElement('label');
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'basis';
    radio.value = key;
    if (idx === 0) radio.checked = true;
    label.appendChild(radio);
    label.appendChild(document.createTextNode(' ' + (DATA.bases[key].label || key)));
    basisRadios.appendChild(label);
    radio.addEventListener('change', () => {
        if (!radio.checked) return;
        stopTour();
        currentBasis = key;
        clampAxisIdx();
        configureAxisSliders();
        applyProjection();
        applyVisibility();
        updateSynthHint();
    });
});

axisRows.forEach((row, s) => {
    const slider = row.querySelector('.axis-slider');
    slider.addEventListener('input', e => {
        stopTour();
        const snapped = Math.round(parseFloat(e.target.value));
        e.target.value = String(snapped);
        axisIdx[s] = snapped - 1;
        updateAxisLabel(s);
        applyProjection();
    });
});

document.querySelectorAll('input[type="range"]').forEach(slider => {
    slider.addEventListener('wheel', e => {
        e.preventDefault();
        const dir = -Math.sign(e.deltaY);
        if (dir === 0) return;
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        let step;
        if (slider.dataset.wheelStep) {
            step = parseFloat(slider.dataset.wheelStep);
        } else {
            const stepAttr = slider.getAttribute('step');
            step = (stepAttr === 'any' || stepAttr === null)
                ? (max - min) / 100
                : parseFloat(stepAttr);
        }
        const cur = parseFloat(slider.value);
        const next = Math.max(min, Math.min(max, cur + dir * step));
        if (next === cur) return;
        slider.value = String(next);
        slider.dispatchEvent(new Event('input'));
    }, { passive: false });
});
document.getElementById('thumbSize').addEventListener('input', e => {
    material.uniforms.uThumb.value = parseFloat(e.target.value) / 100.0;
});

// Populate the atlas dropdown. All atlases share the same N, side, patch_size,
// so swapping one in just means replacing the texture's image source.
const atlasSelectEl = document.getElementById('atlasSelect');
for (const name of ATLAS_NAMES) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = ATLASES[name].label || name;
    atlasSelectEl.appendChild(opt);
}
function loadAtlas(name) {
    const atlas = ATLASES[name];
    if (!atlas) return;
    const img = new Image();
    img.onload = () => { atlasTex.image = img; atlasTex.needsUpdate = true; };
    img.src = 'data:image/png;base64,' + atlas.b64;
}
atlasSelectEl.value = DATA.default_atlas || ATLAS_NAMES[0] || '';
atlasSelectEl.addEventListener('change', e => loadAtlas(e.target.value));
document.getElementById('borderSize').addEventListener('input', e => {
    const pct = parseInt(e.target.value, 10);
    material.uniforms.uBorder.value = pct / 100.0;
    document.getElementById('borderLabel').textContent =
        pct === 0 ? 'no border' : `${pct}% of each edge (set 0 to hide)`;
});

const showSynthEl = document.getElementById('showSynth');
function updateSynthHint() {
    if (!HAS_SYNTHETICS) return;
    const el = document.getElementById('synthHint');
    const synStr = `${N_SYN.toLocaleString()} synthetic atoms ` +
                   `(${DATA.synthetics.k}-NN × ${DATA.synthetics.steps} steps)`;
    if (!basisSupportsSynth()) {
        el.textContent = `${synStr} — hidden in this basis.`;
        showSynthEl.disabled = true;
    } else {
        showSynthEl.disabled = false;
        el.textContent = showSynthEl.checked
            ? `${synStr} — visible.`
            : `${synStr} — off.`;
    }
}
if (HAS_SYNTHETICS) {
    showSynthEl.addEventListener('change', () => {
        showSynthetics = showSynthEl.checked;
        applyVisibility();
        updateSynthHint();
    });
}

if (HAS_KNN_UI) {
    function updateLinesHint() {
        document.getElementById('linesHint').textContent =
            `up to ${(N * edgeK).toLocaleString()} edges (k=${edgeK}, ` +
            `each ${POINT_LABEL} to its k closest neighbours).`;
    }
    updateLinesHint();
    document.getElementById('showLines').addEventListener('change', e => {
        lineMesh.visible = e.target.checked;
        updateLines();
    });
    document.getElementById('edgeK').addEventListener('input', e => {
        edgeK = parseInt(e.target.value, 10);
        document.getElementById('edgeKLabel').textContent =
            `k = ${edgeK} neighbours per ${POINT_LABEL}`;
        updateLinesHint();
        updateLines();
    });
    document.getElementById('edgeWidth').addEventListener('input', e => {
        const w = parseFloat(e.target.value);
        lineMat.linewidth = w;
        document.getElementById('edgeWidthLabel').textContent = `line thickness ${w.toFixed(1)} px`;
    });
}

document.getElementById('tourBtn').addEventListener('click', pauseOrResume);
document.getElementById('resetBtn').addEventListener('click', resetTour);
document.getElementById('tourLegSec').addEventListener('input', e => {
    tourLegSec = parseInt(e.target.value, 10);
    tourSpeed = 1 / (tourLegSec * 60);
    document.getElementById('tourLegLabel').textContent = `${tourLegSec} sec per leg`;
});
document.getElementById('tourRotSpeed').addEventListener('input', e => {
    tourRotSpeed = parseFloat(e.target.value);
    controls.autoRotateSpeed = tourRotSpeed;
    controls.autoRotate = tourState === 'playing' && tourRotSpeed > 0;
    document.getElementById('tourRotSpeedLabel').textContent =
        tourRotSpeed === 0
            ? 'camera rotation off'
            : `camera rotation × ${tourRotSpeed.toFixed(1)} (0 = off)`;
});
tourProgressEl = document.getElementById('tourProgress');
tourProgressLabelEl = document.getElementById('tourProgressLabel');
tourProgressEl.addEventListener('input', e => {
    suppressProgressUpdate = true;
    seekTour(parseFloat(e.target.value));
    if (tourFrameCurr) {
        const cycle = tourFrameCurr.length / 3 | 0;
        tourProgressLabelEl.textContent = formatProgressLabel(tourLegIdx + tourT, cycle);
    }
});
function resumeProgressSync() { suppressProgressUpdate = false; }
tourProgressEl.addEventListener('change', resumeProgressSync);
tourProgressEl.addEventListener('pointerup', resumeProgressSync);
tourProgressEl.addEventListener('mouseup', resumeProgressSync);
tourProgressEl.addEventListener('touchend', resumeProgressSync);

const list = document.getElementById('clusterList');
function clusterRowText(c) {
    if (c.label) return c.label;
    return `#${c.id} ${c.regime} n=${c.size}`;
}
for (const c of DATA.clusters) {
    const row = document.createElement('label');
    row.className = 'cluster-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.addEventListener('change', () => {
        if (cb.checked) visiblePerCluster.add(c.id);
        else visiblePerCluster.delete(c.id);
        applyVisibility();
    });
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = `rgb(${Math.round(255*c.colour[0])},${Math.round(255*c.colour[1])},${Math.round(255*c.colour[2])})`;
    const txt = document.createElement('span');
    txt.textContent = clusterRowText(c);
    row.append(cb, sw, txt);
    list.appendChild(row);
}
if (VIEWER_META.show_unvalidated_row) {
    const row = document.createElement('label');
    row.className = 'cluster-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = showUnvalidated;
    cb.addEventListener('change', () => { showUnvalidated = cb.checked; applyVisibility(); });
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = '#6b7280';
    const txt = document.createElement('span');
    txt.textContent = 'unvalidated atoms';
    row.append(cb, sw, txt);
    list.appendChild(row);
}
document.getElementById('allOn').addEventListener('click', () => {
    for (const c of DATA.clusters) visiblePerCluster.add(c.id);
    showUnvalidated = true;
    document.querySelectorAll('#clusterList input[type=checkbox]').forEach(cb => cb.checked = true);
    applyVisibility();
});
document.getElementById('allOff').addEventListener('click', () => {
    visiblePerCluster.clear();
    showUnvalidated = false;
    document.querySelectorAll('#clusterList input[type=checkbox]').forEach(cb => cb.checked = false);
    applyVisibility();
});

// Manifold-filter grid: one clickable cell per instance, arranged as
// n_variants rows × n_types cols (matching the mask-bitmap thumbnail layout).
const mfSection = document.getElementById('manifoldFilterSection');
if (!HAS_MANIFOLD_FILTER) {
    mfSection.classList.add('hidden');
} else {
    const meta = DATA.manifold_filter_meta || {};
    const nTypes = meta.n_types || 8;
    const nVariants = meta.n_variants || (N_INST / Math.max(nTypes, 1));
    const typeColours = meta.type_colours || [];
    const typeNames = meta.type_names || [];
    const grid = document.getElementById('manifoldGrid');
    grid.style.gridTemplateColumns = `repeat(${nTypes}, 1fr)`;

    function cellRgb(typeIdx) {
        const c = typeColours[typeIdx] || [0.6, 0.6, 0.6];
        return `rgb(${Math.round(255*c[0])},${Math.round(255*c[1])},${Math.round(255*c[2])})`;
    }
    function refreshHint() {
        const sel = selectedManifolds.size;
        const hint = document.getElementById('mfHint');
        if (sel === 0) hint.textContent = 'no filter — showing all samples';
        else hint.textContent = `${sel} manifold${sel === 1 ? '' : 's'} selected (${manifoldFilterMode} mode)`;
    }
    for (let v = 0; v < nVariants; v++) {
        for (let t = 0; t < nTypes; t++) {
            const inst = t * nVariants + v;
            if (inst >= N_INST) continue;
            const cell = document.createElement('div');
            cell.className = 'mf-cell';
            cell.style.background = cellRgb(t);
            cell.title = `${typeNames[t] || `type${t}`}#${v}`;
            cell.addEventListener('click', () => {
                if (selectedManifolds.has(inst)) {
                    selectedManifolds.delete(inst);
                    cell.classList.remove('on');
                } else {
                    selectedManifolds.add(inst);
                    cell.classList.add('on');
                }
                refreshHint();
                applyVisibility();
            });
            grid.appendChild(cell);
        }
    }
    document.getElementById('mfAll').addEventListener('click', () => {
        for (let k = 0; k < N_INST; k++) selectedManifolds.add(k);
        grid.querySelectorAll('.mf-cell').forEach(c => c.classList.add('on'));
        refreshHint(); applyVisibility();
    });
    document.getElementById('mfNone').addEventListener('click', () => {
        selectedManifolds.clear();
        grid.querySelectorAll('.mf-cell').forEach(c => c.classList.remove('on'));
        refreshHint(); applyVisibility();
    });
    document.querySelectorAll('input[name=mfmode]').forEach(r => {
        r.addEventListener('change', e => {
            manifoldFilterMode = e.target.value;
            refreshHint(); applyVisibility();
        });
    });
    refreshHint();
}

// --- Hover tooltip ---
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const hoverDiv = document.getElementById('hover');
renderer.domElement.addEventListener('pointermove', (ev) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
});

// --- Click-to-pivot: orbit around a clicked point (click empty space to clear) ---
let pivotLock = -1;
let showPivotMarker = false;  // marker sphere off by default (toggled in the Sequence panel)
const pivotMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.15, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.2 })
);
pivotMarker.visible = false;
scene.add(pivotMarker);

const SEQ = DATA.seq || null;
const seqSectionEl = document.getElementById('seqSection');
const seqHintEl = document.getElementById('seqHint');
const seqBarEl = document.getElementById('seqBar');
const seqLineEl = document.getElementById('seqLine');
const seqMetaEl = document.getElementById('seqMeta');
const seqPrevEl = document.getElementById('seqPrev');
const seqNextEl = document.getElementById('seqNext');
const seqBackEl = document.getElementById('seqBack');
// (sequence, position) -> point index, so the prev/next arrows can walk the centred token
// along its sequence one token at a time.
const seqPosToPoint = new Map();
// Focus history: each selectPoint pushes the previous focus so 'back' can undo a misclick
// (clicking a thumbnail behind the intended one) or retrace a step-through.
const focusHistory = [];
if (SEQ) {
    seqSectionEl.style.display = 'block';
    for (let i = 0; i < N; i++) seqPosToPoint.set(SEQ.point_seq[i] * 100000 + SEQ.point_pos[i], i);
    document.getElementById('flowLines').addEventListener('change', (e) => {
        flowEnabled = e.target.checked;
        updateFlowLines();
    });
    document.getElementById('flowOpacity').addEventListener('input', (e) => {
        flowOpacity = parseFloat(e.target.value);
        document.getElementById('flowOpacityLabel').textContent = `flow line opacity ${flowOpacity.toFixed(2)}`;
        updateFlowLines();
    });
    document.getElementById('flowLocality').addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        const lab = document.getElementById('flowLocalityLabel');
        if (v >= parseFloat(e.target.max)) {
            flowLocalitySigma = Infinity;
            lab.textContent = 'locality focus: off (whole sequence)';
        } else {
            flowLocalitySigma = v;
            lab.textContent = `locality focus: σ ≈ ${v} tokens around centre`;
        }
        updateFlowLines();
    });
    document.getElementById('showPivot').addEventListener('change', (e) => {
        showPivotMarker = e.target.checked;
        updateOrbitTarget();
    });
    seqPrevEl.addEventListener('click', () => stepSequence(-1));
    seqNextEl.addEventListener('click', () => stepSequence(1));
    seqBackEl.addEventListener('click', goBack);
}

function stepSequence(delta) {
    if (!SEQ || pivotLock < 0) return;
    const s = SEQ.point_seq[pivotLock], p = SEQ.point_pos[pivotLock];
    const next = seqPosToPoint.get(s * 100000 + (p + delta));
    if (next !== undefined) selectPoint(next);
}

function goBack() {
    if (focusHistory.length === 0) return;
    const prev = focusHistory.pop();
    selectPoint(prev, true);  // don't re-push: this IS the undo
}

function renderSequence(idx) {
    if (!SEQ) return;
    const s = SEQ.point_seq[idx], p = SEQ.point_pos[idx];
    const esc = (t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/\\n/g, '\\u23ce');  // show newlines as a glyph on one line
    seqLineEl.innerHTML = SEQ.tokens[s].map((t, i) => i === p
        ? `<span class="focus-tok" id="seqFocusTok">${esc(t)}</span>`
        : `<span>${esc(t)}</span>`).join('');
    seqMetaEl.textContent = `seq ${s} · pos ${p}`;
    seqHintEl.textContent = `showing sequence ${s}, position ${p} (bottom bar)`;
    seqPrevEl.disabled = !seqPosToPoint.has(s * 100000 + (p - 1));
    seqNextEl.disabled = !seqPosToPoint.has(s * 100000 + (p + 1));
    seqBackEl.disabled = focusHistory.length === 0;
    // Scroll the single line so the focused token sits in the centre of the bar.
    const tok = document.getElementById('seqFocusTok');
    if (tok) seqLineEl.scrollLeft = tok.offsetLeft - seqLineEl.clientWidth / 2 + tok.offsetWidth / 2;
}

function selectPoint(idx, fromBack = false) {
    if (!fromBack && pivotLock >= 0 && pivotLock !== idx) focusHistory.push(pivotLock);
    pivotLock = idx;
    updateOrbitTarget();
    if (SEQ) { seqBarEl.style.display = 'flex'; renderSequence(idx); }
    updateFlowLines();
}
function clearPivot() {
    pivotLock = -1;
    updateOrbitTarget();
    if (SEQ) {
        seqBarEl.style.display = 'none';
        seqPrevEl.disabled = true; seqNextEl.disabled = true;
    }
}

let downX = 0, downY = 0, downT = 0;
renderer.domElement.addEventListener('pointerdown', (e) => {
    downX = e.clientX; downY = e.clientY; downT = performance.now();
});
renderer.domElement.addEventListener('pointerup', (e) => {
    const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
    if (moved > 5 || performance.now() - downT > 400) return;  // a drag/rotate, not a click
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    const idx = pickNearest();
    if (idx >= 0) selectPoint(idx); else clearPivot();
});

// Left/Right arrow keys step the focused token along its sequence (mirrors the bottom-bar
// arrows). Ignored while typing in a field so the component search box still works.
window.addEventListener('keydown', (e) => {
    if (!SEQ || pivotLock < 0) return;
    const t = document.activeElement;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) return;
    if (e.key === 'ArrowLeft') { stepSequence(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { stepSequence(1); e.preventDefault(); }
});

function pickNearest() {
    const v = new THREE.Vector3();
    let best = -1, bestD = Infinity;
    for (let i = 0; i < N; i++) {
        if (iVisible[i] < 0.5) continue;
        v.set(positions[i*3], positions[i*3+1], positions[i*3+2]);
        v.project(camera);
        const dx = v.x - pointer.x, dy = v.y - pointer.y;
        const d = dx*dx + dy*dy;
        if (d < bestD) { bestD = d; best = i; }
    }
    return (best >= 0 && bestD < 0.005) ? best : -1;
}
let hoveredPoint = -1;
function setHovered(idx) {
    if (idx === hoveredPoint) return;
    if (hoveredPoint >= 0) iHover[hoveredPoint] = 0;
    hoveredPoint = idx;
    if (hoveredPoint >= 0) iHover[hoveredPoint] = 1;
    instGeom.attributes.iHover.needsUpdate = true;
    renderer.domElement.style.cursor = hoveredPoint >= 0 ? 'pointer' : 'default';
}
function updateHover() {
    const best = pickNearest();
    setHovered(best);
    if (best >= 0) {
        const c = DATA.labels[best];
        const cm = DATA.clusters.find(x => x.id === c);
        const rate = DATA.firing_rate[best];
        const idx = DATA.atom_indices[best];
        const groupWord = VIEWER_META.group_word || 'cluster';
        const scalarWord = VIEWER_META.scalar_word || 'firing rate';
        const cmTail = cm
            ? (cm.label ? ` (${cm.label})` : ` (${cm.regime}, n=${cm.size})`)
            : ' (unvalidated)';
        hoverDiv.style.display = 'block';
        hoverDiv.innerHTML = `<b>${POINT_LABEL} #${idx}</b> &nbsp;`
            + `${groupWord}: ${c}${cmTail}<br>`
            + `${scalarWord}: ${rate.toFixed(3)}`;
    } else {
        hoverDiv.style.display = 'none';
    }
}

// --- Boot ---
clampAxisIdx();
configureAxisSliders();
applyProjection();
applyVisibility();
updateSynthHint();
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    lineMat.resolution.set(window.innerWidth, window.innerHeight);
    flowMat.resolution.set(window.innerWidth, window.innerHeight);
});

function animate() {
    requestAnimationFrame(animate);
    if (tourState === 'playing') tourStep();
    controls.update();
    updateHover();
    renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""


def _default_subtitle(data: dict[str, Any], run_id: str) -> str:
    n = data["n_atoms"]
    point_label = data.get("viewer_meta", {}).get("point_label", "atom")
    parts = [f"run <code>{run_id}</code>", f"{n:,} {point_label}s"]
    for name, basis in data["bases"].items():
        label = basis.get("label", name)
        parts.append(f"{len(basis['var_explained'])} PCs in <i>{label}</i>")
    return " — ".join(parts)


def write_viewer_html(
    data: dict[str, Any], out_path: Path, run_id: str, subtitle: str | None = None
) -> None:
    """Write a self-contained interactive viewer HTML."""
    if subtitle is None:
        subtitle = _default_subtitle(data, run_id)
    payload = json.dumps(data)
    html = (
        _VIEWER_TEMPLATE.replace("__TITLE__", run_id)
        .replace("__SUBTITLE__", subtitle)
        .replace("__DATA_JSON__", payload)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
