"""Link bottleneck-code structure to token contexts.

Two views, both reading the context-preserving harvest (`sequences.pt` + flat codes):
  - regions: k-means the code vectors into K manifold regions; for each, the top tokens
    and a few context windows of member positions.
  - dims: for chosen code dims, the context windows of the top-|z| positions (separately
    for +z and -z, since the gate is sign-preserving).

Context window for global position i is sequences[i // seq_len, (i-w) : (i+w+1)].
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer

from param_decomp_lab.bottleneck_interp.geometry import load_codes


@dataclass
class Region:
    region: int
    size: int
    frac: float
    top_tokens: list[tuple[str, int]]
    examples: list[str]


@dataclass
class DimExamples:
    dim: int
    top_pos: list[str]
    top_neg: list[str]


def context_window(
    sequences: torch.Tensor, seq_len: int, tokenizer: Any, pos: int, half: int
) -> str:
    """Decoded window around a flat position, with the focus token marked [[...]]."""
    s, p = divmod(pos, seq_len)
    lo, hi = max(0, p - half), min(seq_len, p + half + 1)
    row = sequences[s]
    before = tokenizer.decode(row[lo:p].tolist())
    focus = tokenizer.decode([int(row[p])])
    after = tokenizer.decode(row[p + 1 : hi].tolist())
    return f"{before}[[{focus}]]{after}".replace("\n", "\\n")


def top_tokens(tokens: torch.Tensor, tokenizer: Any, k: int) -> list[tuple[str, int]]:
    counts = Counter(tokens.tolist())
    return [(repr(tokenizer.decode([t])), n) for t, n in counts.most_common(k)]


def cluster_regions(
    codes: Float[torch.Tensor, "n D"],
    sequences: torch.Tensor,
    seq_len: int,
    tokenizer: Any,
    n_regions: int,
    n_examples: int,
    half: int,
) -> list[Region]:
    x = codes.numpy()
    km = MiniBatchKMeans(n_clusters=n_regions, random_state=0, n_init="auto", batch_size=4096)
    labels = km.fit_predict(x)
    flat_tokens = sequences.reshape(-1)
    out: list[Region] = []
    for r in range(n_regions):
        idx = np.where(labels == r)[0]
        examples = [
            context_window(sequences, seq_len, tokenizer, int(i), half)
            for i in idx[torch.randperm(len(idx))[:n_examples].numpy()]
        ]
        out.append(
            Region(
                region=r,
                size=int(len(idx)),
                frac=round(len(idx) / len(labels), 4),
                top_tokens=top_tokens(flat_tokens[idx], tokenizer, 12),
                examples=examples,
            )
        )
    out.sort(key=lambda region: -region.size)
    return out


def dim_examples(
    codes: Float[torch.Tensor, "n D"],
    sequences: torch.Tensor,
    seq_len: int,
    tokenizer: Any,
    dims: list[int],
    n_examples: int,
    half: int,
) -> list[DimExamples]:
    out: list[DimExamples] = []
    for d in dims:
        col = codes[:, d]
        windows: dict[int, list[str]] = {}
        for sign in (1, -1):
            top = torch.topk(col * sign, n_examples).indices
            top = top[col[top] * sign > 0]
            windows[sign] = [
                context_window(sequences, seq_len, tokenizer, int(i), half) for i in top
            ]
        out.append(DimExamples(dim=d, top_pos=windows[1], top_neg=windows[-1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, type=Path, help="context-preserving harvest dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n_samples", type=int, default=400_000)
    ap.add_argument("--n_regions", type=int, default=40)
    ap.add_argument("--region_examples", type=int, default=8)
    ap.add_argument("--dims", type=int, nargs="*", default=[])
    ap.add_argument("--dim_examples", type=int, default=15)
    ap.add_argument("--half", type=int, default=8)
    args = ap.parse_args()

    meta = json.loads((args.codes / "meta.json").read_text())
    seq_len = meta["seq_len"]
    tokenizer = AutoTokenizer.from_pretrained(meta["tokenizer_name"])
    sequences = torch.load(args.codes / "sequences.pt")
    codes = load_codes(args.codes, args.n_samples)[: args.n_samples]
    # codes are in flat sequence-major order; align sequences to the same positions
    n_seq = args.n_samples // seq_len
    sequences = sequences[:n_seq]
    codes = codes[: n_seq * seq_len]
    print(f"codes {tuple(codes.shape)}  sequences {tuple(sequences.shape)}")

    args.out.mkdir(parents=True, exist_ok=True)
    regions = cluster_regions(
        codes, sequences, seq_len, tokenizer, args.n_regions, args.region_examples, args.half
    )
    (args.out / "regions.json").write_text(json.dumps([asdict(r) for r in regions], indent=2))
    print(f"wrote {args.n_regions} regions -> {args.out / 'regions.json'}")
    for r in regions[:8]:
        toks = ", ".join(t for t, _ in r.top_tokens[:6])
        print(f"  region {r.region:>3} (n={r.size:>6}): {toks}")

    if args.dims:
        dims = dim_examples(
            codes, sequences, seq_len, tokenizer, args.dims, args.dim_examples, args.half
        )
        (args.out / "dim_examples.json").write_text(json.dumps([asdict(d) for d in dims], indent=2))
        print(f"wrote {len(args.dims)} dim example sets -> {args.out / 'dim_examples.json'}")


if __name__ == "__main__":
    main()
