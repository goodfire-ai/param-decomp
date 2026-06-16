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
from typing import cast

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

# One colour per module *type*; the layer is disambiguated by the item name.
_MODULE_TYPE_COLOUR = {
    "mlp.c_fc": [0.12, 0.47, 0.71],
    "mlp.down_proj": [0.09, 0.75, 0.81],
    "attn.q_proj": [0.84, 0.15, 0.16],
    "attn.k_proj": [1.00, 0.50, 0.05],
    "attn.v_proj": [0.58, 0.40, 0.74],
    "attn.o_proj": [0.17, 0.63, 0.17],
}


def _module_colour(module_name: str) -> list[float]:
    """RGB for a module path like 'h.0.mlp.c_fc', keyed on the type suffix."""
    for suffix, colour in _MODULE_TYPE_COLOUR.items():
        if module_name.endswith(suffix):
            return colour
    return [0.5, 0.5, 0.5]


def _component_overlay(
    coo: dict[str, torch.Tensor], component_names: list[str], flat_idx: torch.Tensor
) -> dict[str, object]:
    """Invert the active-component COO onto the sampled points: {component id -> sampled
    point ids where active} + names, for the viewer's component picker."""
    remap = torch.full((int(flat_idx.max()) + 1,), -1, dtype=torch.long)
    remap[flat_idx] = torch.arange(len(flat_idx))
    pt, cp = coo["point"].long(), coo["comp"].long()
    in_range = pt < remap.shape[0]
    pt, cp = pt[in_range], cp[in_range]
    keep = remap[pt] >= 0
    sampled_pt = remap[pt[keep]].numpy()
    comp = cp[keep].numpy()
    order = comp.argsort(kind="stable")
    comp_s, pt_s = comp[order], sampled_pt[order]
    uniq, starts = np.unique(comp_s, return_index=True)
    bounds = list(starts) + [len(comp_s)]
    index = {str(int(c)): pt_s[bounds[k] : bounds[k + 1]].tolist() for k, c in enumerate(uniq)}
    names = {str(int(c)): component_names[int(c)] for c in uniq}
    return {"kind": "picker", "index": index, "names": names}


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
    ap.add_argument(
        "--module_threshold",
        type=float,
        default=1.0,
        help="module rim floor: min engagement (× the module's own mean) to colour a point",
    )
    ap.add_argument(
        "--component_labels",
        type=Path,
        default=None,
        help="optional JSON {component_name: autointerp label} shown primarily in the "
        "component picker (id kept as secondary text)",
    )
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
        "seq_len": int(seq_len),
    }

    # Module rim overlay (B4): per point, each module's engagement *relative to its own
    # mean* over the sample (frac / mean_frac). The viewer rims each point by its dominant
    # selected module (argmax) with the slider as a floor. Raw fractions are tiny and very
    # uneven across module types, so a plain-fraction threshold saturated on one colour or
    # went all-grey; relative engagement spreads the map across modules and reveals
    # position-specific specialisation.
    if h.module_frac is not None and h.module_names is not None:
        frac = h.module_frac[flat_idx].float()  # [n_points, M] in [0, 1]
        rel = frac / (frac.mean(dim=0, keepdim=True) + 1e-8)  # ×-its-own-mean engagement
        score_max = 8.0  # cap; ≥8× the module's mean all map to full intensity
        items = [
            {"id": j, "name": name, "colour": _module_colour(name)}
            for j, name in enumerate(h.module_names)
        ]
        point_scores = (rel.clamp(0, score_max) / score_max * 255).round().to(torch.uint8).tolist()
        data.setdefault("overlays", {})["module"] = {
            "items": items,
            "point_scores": point_scores,
            "score_max": score_max,
            "default_threshold": args.module_threshold,
        }

    # Component rim overlay (B3): for each global component active somewhere in the sample,
    # the list of sampled points where it is CI-active. The viewer's picker selects one
    # component by name and rims those points. With --component_labels, the picker shows the
    # autointerp label primarily and the raw component id secondarily.
    if h.active_components is not None and h.component_names is not None:
        overlay = _component_overlay(h.active_components, h.component_names, flat_idx)
        if args.component_labels is not None:
            name_to_label: dict[str, str] = json.loads(args.component_labels.read_text())
            names = cast(dict[str, str], overlay["names"])
            overlay["labels"] = {
                cid: name_to_label[name] for cid, name in names.items() if name in name_to_label
            }
        data.setdefault("overlays", {})["component"] = overlay

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_viewer_html(data, args.out, run_id=args.run_id, subtitle="bottleneck code manifold")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
