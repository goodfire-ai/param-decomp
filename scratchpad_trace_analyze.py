"""Parse a jax/chrome trace.json.gz: GPU-busy vs idle per step + time per named scope."""

import collections
import glob
import gzip
import json
import os
import sys

path = sys.argv[1]
if os.path.isdir(path) or "*" in path or path.endswith("/"):
    cands = glob.glob(path + "/**/*.trace.json.gz", recursive=True) + glob.glob(
        path + "/**/trace.json.gz", recursive=True
    )
    assert cands, f"no trace.json.gz under {path}"
    path = max(cands, key=lambda p: __import__("os").path.getsize(p))
print(f"trace: {path}")

with gzip.open(path) as f:
    data = json.load(f)
events = data["traceEvents"]

# identify tracks (pid/tid -> name) from metadata events
pid_name, tid_name = {}, {}
for e in events:
    if e.get("ph") == "M" and e.get("name") == "process_name":
        pid_name[e["pid"]] = e["args"].get("name", "")
    if e.get("ph") == "M" and e.get("name") == "thread_name":
        tid_name[(e["pid"], e["tid"])] = e["args"].get("name", "")

# complete events with duration
X = [e for e in events if e.get("ph") == "X" and "dur" in e]


# GPU streams: tracks whose process/thread name mentions a GPU/stream
def is_gpu(e):
    pn = pid_name.get(e["pid"], "").lower()
    tn = tid_name.get((e["pid"], e["tid"]), "").lower()
    return (
        "gpu" in pn
        or "stream" in tn
        or "/device:gpu" in pn
        or "tensorflow" in pn
        and "stream" in tn
    )


print("=== tracks (pid/process, tid/thread) ===")
for pid, nm in sorted(pid_name.items()):
    threads = [tn for (p, t), tn in tid_name.items() if p == pid]
    print(f"  pid {pid}: {nm!r}  threads={threads[:6]}")

gpu_ev = [e for e in X if is_gpu(e)]
all_min = min((e["ts"] for e in X), default=0)
all_max = max((e["ts"] + e["dur"] for e in X), default=0)
print(
    f"trace span: {(all_max - all_min) / 1e6:.3f}s, {len(X)} complete events, {len(gpu_ev)} on gpu-ish tracks"
)


# union of GPU-busy intervals (merge overlaps) -> busy time
def union(evs):
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs)
    tot, cs, ce = 0.0, None, None
    for s, en in iv:
        if cs is None:
            cs, ce = s, en
        elif s <= ce:
            ce = max(ce, en)
        else:
            tot += ce - cs
            cs, ce = s, en
    if cs is not None:
        tot += ce - cs
    return tot


busy = union(gpu_ev) / 1e6
span = (all_max - all_min) / 1e6
print(f"GPU-busy (union): {busy:.3f}s  /  span {span:.3f}s  ->  occupancy {100 * busy / span:.0f}%")

# time per named scope (top-level pd_* and big kernels)
by_name = collections.Counter()
for e in gpu_ev:
    by_name[e["name"]] += e["dur"]
print("=== top GPU ops by total dur (ms) ===")
for n, d in by_name.most_common(15):
    print(f"  {d / 1e3:9.1f} ms  {n[:80]}")

# track-level breakdown (which stream/track is busy)
by_track = collections.Counter()
for e in gpu_ev:
    by_track[tid_name.get((e["pid"], e["tid"]), f"pid{e['pid']}")] += e["dur"]
print("=== busy by track (ms) ===")
for n, d in by_track.most_common(10):
    print(f"  {d / 1e3:9.1f} ms  {n[:60]}")
