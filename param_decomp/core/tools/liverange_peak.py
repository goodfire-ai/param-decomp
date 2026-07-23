"""Live-range peak-composition analysis for an XLA jit_step dump.

Joins the buffer-assignment (size+shape per buffer) with the live-range file (start-end
program point per buffer), sweeps program points to find the TRUE peak (max simultaneous
live bytes), and decomposes WHAT is co-resident at that peak — the honest attribution the
static slab report can't give (slabs share offsets across temporally-disjoint buffers).

    python liverange_peak.py <dump_dir>
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
ba = sorted(root.glob("*jit_step*buffer-assignment.txt"), key=lambda p: p.stat().st_size)[-1]
lr = sorted(root.glob("*jit_step*live-range.txt"), key=lambda p: p.stat().st_size)[-1]

# buffer-assignment: " value: <id NAME @off> (size=N,offset=M): dtype[shape]{layout}"
size: dict[str, int] = {}
shape: dict[str, str] = {}
for m in re.finditer(
    r"value: <\d+ (\S+) @\d+> \(size=(\d+),[^)]*\):\s*([a-z0-9]+\[[0-9,]*\])", ba.read_text()
):
    name, sz, shp = m.group(1), int(m.group(2)), m.group(3)
    name = name if "{" in name else name + "{}"
    size[name] = sz
    shape[name] = shp

# live-range: "  NAME{idx}:start-end"  (after the BufferLiveRange: header)
text = lr.read_text().split("BufferLiveRange:", 1)[1]
events: list[tuple[int, int, str]] = []  # (start, end, name)
for m in re.finditer(r"^\s+(\S+):(\d+)-(\d+)\s*$", text, re.M):
    name, s, e = m.group(1), int(m.group(2)), int(m.group(3))
    if name in size:
        events.append((s, e, name))

# sweep-line over program points: +size at start, -size at end (inclusive end)
deltas: dict[int, int] = defaultdict(int)
for s, e, name in events:
    deltas[s] += size[name]
    deltas[e + 1] -= size[name]
cur = 0
peak = 0
peak_t = 0
for t in sorted(deltas):
    cur += deltas[t]
    if cur > peak:
        peak, peak_t = cur, t

GB = 1024**3
print(f"buffers joined: {len(events)}   peak = {peak / GB:.2f} GiB at program point {peak_t}")

# composition at peak: buffers live across peak_t, aggregated by shape
live_by_shape: dict[str, list[int]] = defaultdict(list)
for s, e, name in events:
    if s <= peak_t <= e:
        live_by_shape[shape[name]].append(size[name])
print("\n=== peak co-resident composition by shape (GiB; count) ===")
ranked = sorted(live_by_shape.items(), key=lambda kv: -sum(kv[1]))
for shp, szs in ranked[:24]:
    print(f"  {sum(szs) / GB:7.2f} GB  x{len(szs):<4} {shp}")
