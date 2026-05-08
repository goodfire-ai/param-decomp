"""Summarize a torch CUDA memory snapshot dumped by the scaling-investigation profile.

Reads `memory_snapshot.pickle` and prints the top allocations by size at the snapshot
moment, plus a rough breakdown by allocation type (params/grads/optimizer/activations
based on lifetime and origin).

Usage:
    python scripts/analyze_memory_snapshot.py \\
        /path/to/decompositions/<run-id>/memory_snapshot.pickle
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path


def main(path: str) -> None:
    p = Path(path)
    assert p.exists(), f"snapshot not found: {p}"
    with open(p, "rb") as f:
        snapshot = pickle.load(f)

    segments = snapshot["segments"]
    print(f"Snapshot: {p}")
    print(f"Segments: {len(segments)}")
    total_alloc = sum(seg["allocated_size"] for seg in segments)
    total_reserved = sum(seg["total_size"] for seg in segments)
    print(f"Allocated: {total_alloc / 1e9:.2f} GB across all segments")
    print(f"Reserved:  {total_reserved / 1e9:.2f} GB")

    # Walk all live blocks across all segments, group by stream + size
    live_blocks: list[dict] = []
    for seg in segments:
        for block in seg["blocks"]:
            if block["state"] == "active_allocated":
                live_blocks.append(block)

    print(
        f"Live blocks: {len(live_blocks)}, total {sum(b['size'] for b in live_blocks) / 1e9:.2f} GB"
    )

    # Top 30 largest live allocations with their first-frame stack
    live_blocks.sort(key=lambda b: -b["size"])
    print("\n--- Top 30 live allocations ---")
    for i, b in enumerate(live_blocks[:30]):
        size_mb = b["size"] / 1e6
        frames = b.get("frames", [])
        frame_str = "<no frames>"
        for fr in frames:
            fn = fr.get("filename", "")
            name = fr.get("name", "")
            if "param_decomp" in fn or "ZeroRedundancyOptimizer" in name or "AdamW" in name:
                frame_str = f"{fn.rsplit('/', 1)[-1]}:{fr.get('line', '?')} ({name})"
                break
        else:
            if frames:
                fr = frames[0]
                frame_str = (
                    f"{fr.get('filename', '?').rsplit('/', 1)[-1]}:{fr.get('line', '?')} "
                    f"({fr.get('name', '?')})"
                )
        print(f"  [{i:2d}] {size_mb:>9.1f} MB   {frame_str}")

    # Bucket by frame source
    print("\n--- Aggregated by source file (>50 MB total only) ---")
    by_file: Counter[str] = Counter()
    for b in live_blocks:
        frames = b.get("frames", [])
        for fr in frames:
            fn = fr.get("filename", "")
            if "param_decomp" in fn or "torch/optim" in fn or "torch/nn" in fn:
                key = fn.rsplit("/", 2)
                key_str = "/".join(key[-2:]) if len(key) > 1 else fn
                by_file[key_str] += b["size"]
                break
    for fn, total in sorted(by_file.items(), key=lambda kv: -kv[1])[:20]:
        if total >= 50e6:
            print(f"  {total / 1e9:>6.2f} GB   {fn}")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: analyze_memory_snapshot.py <path/to/snapshot.pickle>"
    main(sys.argv[1])
