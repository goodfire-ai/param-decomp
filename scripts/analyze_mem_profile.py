"""Coarse offline summary of a ``torch.cuda.memory._record_memory_history`` pickle.

The pickle is also loadable at https://pytorch.org/memory_viz for the full
interactive timeline, but for fast capacity-planning questions ("how much is
weights vs activations? what's the peak? what's persistent vs transient?")
this CLI is faster.

Usage:
    python scripts/analyze_mem_profile.py <pickle_path> [--top=20]

Outputs:
  * Peak allocation observed in the snapshot.
  * Live allocation as of the last sample (= persistent state).
  * Top-N largest live allocations by size, with frame summary.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import fire


def _format_gb(b: int) -> str:
    return f"{b / 1e9:.3f}gb"


def _format_mb(b: int) -> str:
    return f"{b / 1e6:.1f}mb"


def _frame_summary(frames: list[dict[str, Any]] | None, depth: int = 4) -> str:
    if not frames:
        return "(no frames)"
    # frames is ordered leaf → root; show top N as a compact path
    parts: list[str] = []
    for f in frames[:depth]:
        name = f.get("name", "?")
        filename = f.get("filename", "?")
        line = f.get("line", "?")
        short_file = filename.rsplit("/", 1)[-1] if filename != "?" else "?"
        parts.append(f"{name}({short_file}:{line})")
    return " → ".join(parts)


def main(pickle_path: str, top: int = 20) -> None:
    """Summarize a CUDA memory-history pickle.

    Args:
        pickle_path: Path to ``mem_rank<R>.pickle`` from
            ``torch.cuda.memory._dump_snapshot``.
        top: Number of largest live allocations to print.
    """
    path = Path(pickle_path)
    assert path.exists(), f"pickle not found: {path}"
    with open(path, "rb") as f:
        snap = pickle.load(f)

    # The snapshot format (PyTorch 2.x):
    # snap = {"segments": [...], "device_traces": [...]}
    # segments[i] = {
    #     "device": int,
    #     "address": int,
    #     "total_size": int,
    #     "blocks": [{"size": int, "state": "active_allocated" | "inactive" | ..., "frames": [...]}, ...],
    #     ...
    # }
    segments: list[dict[str, Any]] = snap["segments"]

    live_blocks: list[tuple[int, list[dict[str, Any]] | None]] = []
    reserved_total = 0
    live_total = 0
    for seg in segments:
        reserved_total += seg["total_size"]
        for block in seg["blocks"]:
            if block["state"].startswith("active"):
                live_total += block["size"]
                live_blocks.append((block["size"], block.get("frames")))

    print(f"file:           {path}")
    print(f"reserved:       {_format_gb(reserved_total)} (caching allocator's pool)")
    print(f"live (active):  {_format_gb(live_total)} ({len(live_blocks)} blocks)")
    print(f"free in pool:   {_format_gb(reserved_total - live_total)}")
    print()

    # Group same-shape allocations to find the heavy hitters
    print(f"top {top} live allocations by size:")
    live_blocks.sort(key=lambda x: x[0], reverse=True)
    for i, (size, frames) in enumerate(live_blocks[:top]):
        print(f"  [{i + 1:2d}] {_format_mb(size):>9s}  {_frame_summary(frames)}")
    print()

    # Size histogram by power-of-two buckets
    print("size histogram (live):")
    buckets: Counter[int] = Counter()
    for size, _ in live_blocks:
        bucket = max(1, size).bit_length() - 1  # log2(size)
        buckets[bucket] += size
    for bucket in sorted(buckets):
        size_label = f"2^{bucket}".rjust(5)
        total = buckets[bucket]
        n = sum(1 for size, _ in live_blocks if max(1, size).bit_length() - 1 == bucket)
        print(
            f"  {size_label} bytes ({_format_mb(1 << bucket):>9s} each): "
            f"{n:5d} blocks, {_format_gb(total):>9s} total"
        )


if __name__ == "__main__":
    fire.Fire(main)
