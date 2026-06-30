"""Gather-vs-compute overlap from an UNCAPPED xplane.pb (no 1M-event JSON cap).

For each GPU device plane: classify every kernel into gather (NCCL collectives) /
compute (everything else on a compute stream) / memcpy, then sweep-line over the
intervals to report union(gather), union(compute), both-active, and busy%. Both-active
~0 => gathers are exposed (serial); high => overlap is happening.

Usage: python xplane_overlap.py <xplane.pb>  [also prints top kernels by total dur]
"""

import collections
import re
import sys

import google.protobuf.runtime_version as _rv

_rv.ValidateProtobufRuntimeVersion = lambda *a, **k: None
from xplane_pb2 import XSpace

PS_PER_NS = 1000.0

COLL = re.compile(
    r"all-?gather|allgather|reduce-?scatter|all-?reduce|collective-?permute|"
    r"ppermute|nccl|rccl|async-(start|done)|collective",
    re.I,
)
MEMCPY = re.compile(r"memcpy|memset|copy-start|copy-done|hbm|d2d|h2d|d2h", re.I)


def classify(name):
    if COLL.search(name):
        return "gather"
    if MEMCPY.search(name):
        return "memcpy"
    return "compute"


def sweep(intervals):
    pts = []
    for s, e, k in intervals:
        pts.append((s, 1, k))
        pts.append((e, -1, k))
    pts.sort()
    g = c = 0
    last = None
    gu = cu = both = busy = 0.0
    lo = min(s for s, _, _ in intervals)
    hi = max(e for _, e, _ in intervals)
    for t, d, k in pts:
        if last is not None and t > last:
            dt = t - last
            if g > 0:
                gu += dt
            if c > 0:
                cu += dt
            if g > 0 and c > 0:
                both += dt
            if g > 0 or c > 0:
                busy += dt
        if k == "gather":
            g += d
        elif k == "compute":
            c += d
        last = t
    return hi - lo, gu, cu, both, busy


def main(path):
    xs = XSpace()
    with open(path, "rb") as f:
        xs.ParseFromString(f.read())

    gpu_planes = [p for p in xs.planes if "/device:GPU" in p.name]
    print(f"file: {path.split('/')[-1]}")
    print(f"GPU device planes: {len(gpu_planes)}  (one per local GPU)\n")

    for plane in gpu_planes:
        emeta = plane.event_metadata
        iv = []
        dur_by_name = collections.Counter()
        cnt_by_kind = collections.Counter()
        for line in plane.lines:
            base = line.timestamp_ns
            for ev in line.events:
                name = emeta[ev.metadata_id].name
                s = base + ev.offset_ps / PS_PER_NS
                d = ev.duration_ps / PS_PER_NS
                if d <= 0:
                    continue
                k = classify(name)
                cnt_by_kind[k] += 1
                if k in ("gather", "compute"):
                    iv.append((s, s + d, k))
                    dur_by_name[(k, name)] += d
        if not iv:
            continue
        span, gu, cu, both, busy = sweep(iv)
        print(f"=== {plane.name[:48]} — {len(iv)} kernels  {dict(cnt_by_kind)} ===")
        print(f"  span          {span/1e6:8.2f} ms")
        print(f"  busy (g|c)    {busy/1e6:8.2f} ms  ({100*busy/span:4.1f}% of span)")
        print(f"  gather union  {gu/1e6:8.2f} ms")
        print(f"  compute union {cu/1e6:8.2f} ms")
        print(f"  BOTH active   {both/1e6:8.2f} ms  = {100*both/max(gu,1):4.1f}% of gather, "
              f"{100*both/max(cu,1):4.1f}% of compute")
        serial = gu + cu - both
        print(f"  serial (g+c-both) {serial/1e6:7.2f} ms   "
              f"ideal-overlap floor (max) {max(gu,cu)/1e6:.2f} ms")
        print(f"  -> {'OVERLAP' if both > 0.15*gu else 'GATHERS EXPOSED'}  "
              f"(prize: {serial/1e6:.1f} -> {max(gu,cu)/1e6:.1f} ms = "
              f"{serial/max(max(gu,cu),1):.2f}x)")
        print("  top kernels by total dur (ms):")
        for (k, name), d in dur_by_name.most_common(12):
            print(f"    {d/1e6:8.2f}  [{k:7s}] {name[:70]}")
        print()
        break  # one representative GPU is enough; remove to see all


if __name__ == "__main__":
    main(sys.argv[1])
