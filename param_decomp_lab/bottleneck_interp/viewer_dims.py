"""3D UMAP of the bottleneck code DIMS, mapped into the grand-tour viewer.

Each point is one of the D code dims, represented by its activation profile across a
sample of positions (the columns of the code matrix). UMAP (correlation metric) embeds
dims that co-activate similarly near each other; PCA of the same profiles is exposed as
an alternate basis. Each dim's thumbnail is its top-activating token; points are coloured
by firing-rate band.

This is the feature-space view (cf. UMAP-of-SAE-features), distinct from the
position-manifold view in `viewer_codes.py`.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import umap
from transformers import AutoTokenizer

from param_decomp_lab.bottleneck_interp.geometry import load_codes
from param_decomp_lab.bottleneck_interp.viewer_3d import build_latent_viewer_data, write_viewer_html
from param_decomp_lab.bottleneck_interp.viewer_codes import token_thumbnails

_BANDS = [(0.0, 0.05, "rare <5%"), (0.05, 0.25, "low"), (0.25, 0.5, "mid"), (0.5, 1.01, "high")]
_BAND_COLOUR = [[0.84, 0.15, 0.16], [1.0, 0.6, 0.2], [0.17, 0.63, 0.17], [0.12, 0.47, 0.71]]


def firing_band(rate: float) -> int:
    for i, (lo, hi, _) in enumerate(_BANDS):
        if lo <= rate < hi:
            return i
    return len(_BANDS) - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n_profile", type=int, default=50000, help="positions per dim profile")
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--min_dist", type=float, default=0.1)
    ap.add_argument("--patch", type=int, default=40)
    ap.add_argument("--run_id", default="bneck-1e-2-dims")
    args = ap.parse_args()

    meta = json.loads((args.codes / "meta.json").read_text())
    tokenizer: Any = AutoTokenizer.from_pretrained(meta["tokenizer_name"])
    sequences = torch.load(args.codes / "sequences.pt")
    flat_tokens = sequences.reshape(-1)

    codes = load_codes(args.codes, args.n_profile)[: args.n_profile].float()  # (P, D)
    n_pos, dim = codes.shape
    profiles = codes.T.contiguous().numpy()  # (D, P): each dim's activation profile
    print(f"dims {dim}, profile length {n_pos}")

    umap_xyz = umap.UMAP(
        n_components=3,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="correlation",
        random_state=0,
    ).fit_transform(profiles)

    # PCA-reduce the 50k-dim profiles so the viewer's PCA/raw bases stay small; the
    # leading PCs preserve the dim-to-dim structure that its PCA basis would compute.
    centered = profiles - profiles.mean(axis=0, keepdims=True)
    vt = np.linalg.svd(centered, full_matrices=False)[2][:50]
    points = (centered @ vt.T).astype(np.float32)

    firing_rate = (codes != 0).float().mean(dim=0).numpy()  # (D,)
    # each dim's top-activating token (within the sampled positions)
    top_pos = codes.abs().argmax(dim=0).numpy()  # (D,) position index per dim
    top_token_ids = flat_tokens[torch.from_numpy(top_pos.astype(np.int64))].tolist()

    labels = np.array([firing_band(float(r)) for r in firing_rate], dtype=int)
    groups: list[dict[str, object]] = []
    for b, (_, _, name) in enumerate(_BANDS):
        groups.append(
            {
                "id": b,
                "size": int((labels == b).sum()),
                "regime": "band",
                "label": f"{name} ({int((labels == b).sum())} dims)",
                "colour": _BAND_COLOUR[b],
            }
        )

    thumbs = token_thumbnails(top_token_ids, tokenizer, args.patch)

    data = build_latent_viewer_data(
        points=points,
        thumbnails=thumbs,
        labels=labels,
        groups=groups,
        patch_size=args.patch,
        extra_bases={"umap3d": np.asarray(umap_xyz, dtype=np.float32)},
        point_indices=np.arange(dim),
        point_scalar=firing_rate,
        point_label="code-dim",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_viewer_html(data, args.out, run_id=args.run_id, subtitle="bottleneck dims — UMAP + PCA")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
