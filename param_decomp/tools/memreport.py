"""Rigorous GPU-memory attribution from XLA's compile-time dump — no byte-arithmetic guessing.

A run launched with `XLA_FLAGS=--xla_dump_to=<dir>` writes, per compiled module:
  *-buffer-assignment.txt   — every buffer: `(size=N, offset=M): dtype[shape]`
  *-memory-usage-report.txt — buffers ranked by cumulative size, with their shapes

This parses the jit_step module's report and prints the largest buffers (size + shapes +
how many live copies), so "what is the 358GB?" is a one-command, factual answer instead of
matching a byte count to a shape product. Usage:

    python -m param_decomp.tools.memreport <dump_dir|run_dir> [--top N]

The dump lands at runs/<id>/hlo when the launcher dumps HLO (see launch.py XLA_FLAGS).
"""

import re
import sys
from pathlib import Path


def find_report(root: Path) -> Path:
    for pat in ("**/*jit_step*memory-usage-report.txt", "**/*jit_step*buffer-assignment.txt"):
        hits = sorted(root.glob(pat), key=lambda p: p.stat().st_size, reverse=True)
        if hits:
            return hits[0]
    raise SystemExit(f"no jit_step memory-usage-report / buffer-assignment under {root}")


def main() -> None:
    root = Path(sys.argv[1])
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 20
    f = find_report(root)
    print(f"report: {f.name}")
    text = f.read_text()

    total = re.search(r"Total bytes:\s*\d+\s*\(([\d.]+\w+)\)", text)
    if total:
        print(f"peak/total: {total.group(1)}")

    if "memory-usage-report" in f.name:
        # rows: cumulative; size; offset; n_values; shapes
        rows = []
        for line in text.splitlines():
            m = re.match(r"\s*[\d.]+\w+\(\s*\d+%\);\s*([\d.]+\w+);\s*\d+;\s*(\d+);\s*(.+)", line)
            if m:
                rows.append((m.group(1), int(m.group(2)), m.group(3).strip()))
        # already roughly size-sorted; print the biggest distinct shapes
        print(f"\n=== top {top} buffers (size; n_live; shapes) ===")
        for size, n, shapes in rows[:top]:
            print(f"  {size:>9}  ×{n:<3}  {shapes[:90]}")
    else:
        # buffer-assignment: parse `(size=N ...): dtype[shape]`, aggregate by shape
        from collections import Counter

        by_shape: Counter[str] = Counter()
        bytes_by_shape: dict[str, int] = {}
        for sz, shape in re.findall(r"size=(\d+)[^)]*\):\s*([a-z0-9]+\[[\d,]*\])", text):
            by_shape[shape] += 1
            bytes_by_shape[shape] = int(sz)
        print(f"\n=== top {top} buffer shapes (total_GB; count; shape) ===")
        ranked = sorted(by_shape, key=lambda s: bytes_by_shape[s] * by_shape[s], reverse=True)
        for shape in ranked[:top]:
            tot = bytes_by_shape[shape] * by_shape[shape] / 1e9
            print(f"  {tot:8.1f} GB  ×{by_shape[shape]:<4} {shape}")


if __name__ == "__main__":
    main()
