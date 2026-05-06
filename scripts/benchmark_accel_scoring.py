"""Benchmark optional rank-one scoring backends.

This synthetic harness measures the acceleration seam independently from model
loading, decomposition training, JSONL writes, and dataset preprocessing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from param_decomp.accel import score_rank_one_linear_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "python", "rust"], default="python")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--in-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--components", type=int, default=256)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--with-slices", action="store_true")
    return parser.parse_args()


def make_bundle(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    inputs = rng.normal(size=(args.rows, args.in_dim)).astype(np.float32)
    weights = rng.normal(scale=0.05, size=(args.out_dim, args.in_dim)).astype(np.float32)
    bias = rng.normal(scale=0.01, size=(args.out_dim,)).astype(np.float32)
    labels = rng.integers(0, args.out_dim, size=(args.rows,), dtype=np.int64)
    reference_logits = inputs @ weights.T + bias
    components_u = rng.normal(scale=0.05, size=(args.components, args.out_dim)).astype(np.float32)
    components_v = rng.normal(scale=0.05, size=(args.components, args.in_dim)).astype(np.float32)
    bundle: dict[str, Any] = {
        "inputs": inputs,
        "labels": labels,
        "reference_logits": reference_logits,
        "components_u": components_u,
        "components_v": components_v,
        "component_ids": [f"synthetic:{index:05d}" for index in range(args.components)],
    }
    if args.with_slices:
        even = np.arange(0, args.rows, 2, dtype=np.int64)
        odd = np.arange(1, args.rows, 2, dtype=np.int64)
        bundle.update(
            {
                "row_indices": np.arange(args.rows, dtype=np.int64),
                "slice_names": ["even", "odd"],
                "slice_offsets": np.array([0, len(even), len(even) + len(odd)], dtype=np.int64),
                "slice_indices": np.concatenate([even, odd]).astype(np.int64),
            }
        )
    return bundle


def main() -> None:
    args = parse_args()
    bundle = make_bundle(args)
    elapsed_values = []
    records_count = 0
    for _ in range(args.repeats):
        started = time.perf_counter()
        records = score_rank_one_linear_components(
            **bundle,
            backend=args.backend,
            rust_threads=args.threads,
        )
        elapsed = time.perf_counter() - started
        elapsed_values.append(elapsed)
        records_count = len(records)
    best_elapsed = min(elapsed_values)
    result = {
        "backend": args.backend,
        "rows": args.rows,
        "in_dim": args.in_dim,
        "out_dim": args.out_dim,
        "components": args.components,
        "threads": args.threads,
        "repeats": args.repeats,
        "with_slices": args.with_slices,
        "best_elapsed_seconds": round(best_elapsed, 6),
        "component_scores_per_sec": round(records_count / best_elapsed, 3),
        "records": records_count,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
