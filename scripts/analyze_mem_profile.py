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


def _leaf_frame_key(frames: list[dict[str, Any]] | None) -> str:
    """Identify a frame group by its top user-code frame.

    Skips C++ allocator-internal frames and PyTorch internal modules so we
    end up keying on the actual lab function that asked for the allocation
    (e.g. ``make_components``, ``_layerwise_one_site``, ``calc_weight_deltas``).
    """
    if not frames:
        return "(no frames)"
    skip_substrings = (
        "/torch/cuda/",
        "/torch/_utils.py",
        "/torch/nn/modules/module.py",
        "/torch/storage.py",
        "/torch/_tensor.py",
        "/torch/autograd/",
        "memory_snapshot",
        "CUDACachingAllocator",
    )
    for f in frames:
        filename = f.get("filename", "")
        if not any(s in filename for s in skip_substrings):
            short = filename.rsplit("/", 1)[-1] if filename else "?"
            return f"{f.get('name', '?')}({short}:{f.get('line', '?')})"
    # All frames were filtered out — fall back to topmost
    return _frame_summary(frames, depth=1)


def _scan_peak_from_traces(snap: dict[str, Any]) -> tuple[int, bool] | None:
    """Walk ``device_traces`` to find peak total allocated bytes over time.

    Returns ``(peak_bytes, truncated)`` or ``None`` if traces are absent.
    ``truncated`` is True when the trace buffer rolled over (max_entries hit),
    in which case peak is a LOWER BOUND only — we lost the build-up phase.

    Truncation heuristic: PyTorch records events in a circular buffer; if the
    earliest event is a free (rather than an alloc or a segment_alloc), we
    started recording mid-run.
    """
    traces = snap.get("device_traces")
    if not traces:
        return None
    events: list[dict[str, Any]] = []
    for dev_events in traces:
        events.extend(dev_events)
    if not events:
        return None
    truncated = events[0].get("action", "") in ("free_completed", "free_requested")
    current = 0
    peak = 0
    for ev in events:
        action = ev.get("action", "")
        size = ev.get("size", 0)
        if action == "alloc":
            current += size
            if current > peak:
                peak = current
        elif action == "free_completed":
            current -= size
    return peak, truncated


def main(pickle_path: str, top: int = 20) -> None:
    """Summarize a CUDA memory-history pickle.

    Args:
        pickle_path: Path to ``mem_rank<R>.pickle`` from
            ``torch.cuda.memory._dump_snapshot``.
        top: Number of largest live allocation groups (by leaf Python frame).
    """
    path = Path(pickle_path)
    assert path.exists(), f"pickle not found: {path}"
    with open(path, "rb") as f:
        snap = pickle.load(f)

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

    peak_info = _scan_peak_from_traces(snap)

    print(f"file:           {path}")
    print(f"reserved:       {_format_gb(reserved_total)} (caching allocator's pool)")
    print(
        f"live (active):  {_format_gb(live_total)} ({len(live_blocks)} blocks)  ← persistent state"
    )
    if peak_info is not None:
        peak, truncated = peak_info
        if truncated:
            print(
                f"peak observed:  ≥{_format_gb(peak)}  (trace buffer truncated; "
                "lower bound only — use the per-phase PD_PHASE_TRACE log for accurate peaks)"
            )
        else:
            print(f"peak observed:  {_format_gb(peak)}  ← max during run")
            print(f"transient peak: {_format_gb(peak - live_total)}  ← scales with batch")
    print(f"free in pool:   {_format_gb(reserved_total - live_total)}")
    print()

    # Group live allocations by leaf Python frame (heavy hitter buckets).
    by_leaf: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for size, frames in live_blocks:
        key = _leaf_frame_key(frames)
        by_leaf[key] += size
        counts[key] += 1

    print(f"top {top} live allocation groups (by leaf Python frame):")
    for key, total in by_leaf.most_common(top):
        n = counts[key]
        print(f"  {_format_gb(total):>9s} ({n:4d} blocks)  {key}")
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
