"""Build the 3D grand-tour viewer over bottleneck code vectors.

Each point is one token-position's code `z`; PCA'd to 3D this is the code manifold.
Points are sampled as whole sequences (so a sequence's positions stay together for the
flow lines and the click-to-show-sequence sidebar), coloured by k-means region, and each
carries a thumbnail rendering its focus token.

Reads a context-preserving harvest (codes + sequences.pt) and writes a self-contained
HTML viewer.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from param_decomp_lab.bottleneck_interp.harvest_io import (
    REGION_PALETTE,
    kmeans_regions,
    load_harvest,
    token_thumbnails,
    top_tokens,
)
from param_decomp_lab.bottleneck_interp.viewer_3d import (
    build_latent_viewer_data,
    write_viewer_html,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, type=Path, help="context-preserving harvest dir")
    ap.add_argument("--out", required=True, type=Path, help="output .html path")
    ap.add_argument("--n_sequences", type=int, default=60, help="whole sequences to sample")
    ap.add_argument("--n_regions", type=int, default=20)
    ap.add_argument("--patch", type=int, default=40)
    ap.add_argument("--run_id", default="bneck-1e-2")
    ap.add_argument("--umap", action="store_true", help="add a 3D UMAP basis of the code vectors")
    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    args = ap.parse_args()

    seq_len = json.loads((args.codes / "meta.json").read_text())["seq_len"]
    pool = args.n_sequences * seq_len * 4
    h = load_harvest(args.codes, pool)
    n_seq_avail = h.codes.shape[0] // seq_len
    chosen = torch.randperm(n_seq_avail)[: args.n_sequences]
    # gather every position of each chosen sequence, in order, so points stay contiguous
    flat_idx = (chosen[:, None] * seq_len + torch.arange(seq_len)).reshape(-1)
    codes = h.codes[flat_idx]
    tokens = h.flat_tokens[flat_idx]
    n_chosen = len(chosen)
    point_seq = torch.arange(n_chosen).repeat_interleave(seq_len)
    point_pos = torch.arange(seq_len).repeat(n_chosen)
    seq_tokens = [[h.tokenizer.decode([int(t)]) for t in h.sequences[int(c)]] for c in chosen]
    print(f"points {tuple(codes.shape)} from {n_chosen} sequences of length {seq_len}")

    labels = kmeans_regions(codes, args.n_regions)

    groups: list[dict[str, object]] = []
    for r in range(args.n_regions):
        idx = np.where(labels == r)[0]
        top = " ".join(t for t, _ in top_tokens(tokens[idx], h.tokenizer, 4))
        groups.append(
            {
                "id": r,
                "size": int(len(idx)),
                "regime": "region",
                "label": f"#{r} {top}",
                "colour": REGION_PALETTE[r % len(REGION_PALETTE)],
            }
        )

    thumbs = token_thumbnails(tokens.tolist(), h.tokenizer, args.patch)
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
    # Sequence panel: per-point (sequence, position) + each sequence's decoded tokens, so
    # the viewer can show a clicked point's whole sequence and draw per-sequence flow lines.
    data["seq"] = {
        "point_seq": point_seq.tolist(),
        "point_pos": point_pos.tolist(),
        "tokens": seq_tokens,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_viewer_html(data, args.out, run_id=args.run_id, subtitle="bottleneck code manifold")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
