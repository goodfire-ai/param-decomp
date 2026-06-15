"""Build the 3D grand-tour viewer over bottleneck code vectors.

Each point is one token-position's code `z`; PCA'd to 3D this is the code manifold.
Points are coloured by k-means region and each carries a thumbnail rendering its focus
token, so hovering/zooming shows what token sits where on the manifold.

Reads a context-preserving harvest (codes + sequences.pt) and writes a self-contained
HTML viewer.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer

from param_decomp_lab.bottleneck_interp.geometry import load_codes
from param_decomp_lab.bottleneck_interp.viewer_3d import (
    build_latent_viewer_data,
    write_viewer_html,
)

_PALETTE = [
    [0.12, 0.47, 0.71],
    [1.00, 0.50, 0.05],
    [0.17, 0.63, 0.17],
    [0.84, 0.15, 0.16],
    [0.58, 0.40, 0.74],
    [0.55, 0.34, 0.29],
    [0.89, 0.47, 0.76],
    [0.50, 0.50, 0.50],
    [0.74, 0.74, 0.13],
    [0.09, 0.75, 0.81],
    [0.68, 0.78, 0.91],
    [1.00, 0.73, 0.47],
    [0.60, 0.87, 0.54],
    [1.00, 0.60, 0.59],
    [0.77, 0.69, 0.84],
    [0.77, 0.61, 0.58],
    [0.97, 0.71, 0.82],
    [0.78, 0.78, 0.78],
    [0.86, 0.86, 0.55],
    [0.62, 0.85, 0.90],
]


def token_thumbnails(token_ids: list[int], tokenizer: Any, patch: int) -> np.ndarray:
    """(N, 3, patch, patch) float[0,1] images, each rendering one token's text."""
    font = ImageFont.load_default(size=max(10, patch // 3))
    out = np.zeros((len(token_ids), 3, patch, patch), dtype=np.float32)
    for i, tid in enumerate(token_ids):
        text = tokenizer.decode([tid]).strip()[:6] or "·"
        img = Image.new("RGB", (patch, patch), (24, 24, 28))
        ImageDraw.Draw(img).text((2, patch // 3), text, fill=(235, 235, 235), font=font)
        out[i] = np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, type=Path, help="context-preserving harvest dir")
    ap.add_argument("--out", required=True, type=Path, help="output .html path")
    ap.add_argument("--n_points", type=int, default=8000)
    ap.add_argument("--n_regions", type=int, default=20)
    ap.add_argument("--patch", type=int, default=40)
    ap.add_argument("--run_id", default="bneck-1e-2")
    ap.add_argument("--umap", action="store_true", help="add a 3D UMAP basis of the code vectors")
    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    args = ap.parse_args()

    meta = json.loads((args.codes / "meta.json").read_text())
    tokenizer: Any = AutoTokenizer.from_pretrained(meta["tokenizer_name"])
    sequences = torch.load(args.codes / "sequences.pt")
    flat_tokens = sequences.reshape(-1)

    codes_all = load_codes(args.codes, args.n_points * 4)
    sel = torch.randperm(min(codes_all.shape[0], flat_tokens.shape[0]))[: args.n_points]
    codes = codes_all[sel].float()
    tokens = flat_tokens[sel]
    print(f"points {tuple(codes.shape)}")

    labels = MiniBatchKMeans(
        n_clusters=args.n_regions, random_state=0, n_init="auto", batch_size=4096
    ).fit_predict(codes.numpy())

    groups: list[dict[str, object]] = []
    for r in range(args.n_regions):
        idx = np.where(labels == r)[0]
        top = Counter(tokens[idx].tolist()).most_common(4)
        label = " ".join(repr(tokenizer.decode([t]).strip()) for t, _ in top)
        groups.append(
            {
                "id": r,
                "size": int(len(idx)),
                "regime": "region",
                "label": f"#{r} {label}",
                "colour": _PALETTE[r % len(_PALETTE)],
            }
        )

    thumbs = token_thumbnails(tokens.tolist(), tokenizer, args.patch)
    code_l0 = (codes != 0).float().sum(dim=1).numpy()

    extra_bases: dict[str, np.ndarray] | None = None
    if args.umap:
        import umap

        print(f"UMAP over {codes.shape[0]} code vectors...")
        xyz = umap.UMAP(
            n_components=3,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric="cosine",
            random_state=0,
        ).fit_transform(codes.numpy())
        extra_bases = {"umap3d": np.asarray(xyz, dtype=np.float32)}

    data = build_latent_viewer_data(
        points=codes.numpy(),
        thumbnails=thumbs,
        labels=labels.astype(int),
        groups=groups,
        patch_size=args.patch,
        extra_bases=extra_bases,
        point_scalar=code_l0,
        point_label="token-position",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_viewer_html(data, args.out, run_id=args.run_id, subtitle="bottleneck code manifold")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
