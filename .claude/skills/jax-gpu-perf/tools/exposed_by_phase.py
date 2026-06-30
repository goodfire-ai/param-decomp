"""Localize EXPOSED (non-overlapped) gather time by named-scope PHASE.

For one GPU plane: classify kernels gather/compute, recover each event's `pd_*` phase from
its scoped hlo op-name (jit(step)/pd_<phase>/...), merge the compute-union, and for each
gather interval compute the portion NOT covered by any compute = exposed. Sum total +
exposed gather per phase. Exposed gather is what coalescing/overlap must target; gather
already hidden under compute is free.

Usage: python exposed_by_phase.py <xplane.pb>
"""
import collections
import re
import sys

import google.protobuf.runtime_version as _rv

_rv.ValidateProtobufRuntimeVersion = lambda *a, **k: None
from xplane_pb2 import XSpace

PS = 1000.0
COLL = re.compile(r"all-?gather|reduce-?scatter|all-?reduce|collective-?permute|nccl|async-(start|done)", re.I)
MEMCPY = re.compile(r"memcpy|memset|copy-start|copy-done", re.I)
PHASE = re.compile(r"(pd_[a-z_]+)")


def classify(n):
    if COLL.search(n): return "gather"
    if MEMCPY.search(n): return "memcpy"
    return "compute"


def merge(intervals):
    if not intervals: return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out


def covered(s, e, union):
    """length of [s,e] covered by the sorted disjoint `union`."""
    import bisect
    lo = bisect.bisect_left([u[1] for u in union], s)
    tot = 0.0
    for i in range(lo, len(union)):
        us, ue = union[i]
        if us >= e: break
        tot += max(0.0, min(e, ue) - max(s, us))
    return tot


def main(path):
    xs = XSpace(); xs.ParseFromString(open(path, "rb").read())
    gpu = [p for p in xs.planes if "/device:GPU" in p.name][0]
    emeta = gpu.event_metadata
    smeta = {sm.id: sm.name for sm in gpu.stat_metadata.values()}

    def scope_of(ev):
        for s in ev.stats:
            key = smeta.get(s.metadata_id, "")
            val = s.str_value or (smeta.get(s.ref_value, "") if s.ref_value else "")
            m = PHASE.search(val) or (PHASE.search(key) if key else None)
            if m: return m.group(1)
        # fall back: the event's own metadata name may carry the scope
        return "?"

    gathers = []  # (s,e,phase)
    computes = []
    for line in gpu.lines:
        base = line.timestamp_ns
        for ev in line.events:
            nm = emeta[ev.metadata_id].name
            k = classify(nm)
            if k == "memcpy": continue
            s = base + ev.offset_ps / PS; e = s + ev.duration_ps / PS
            if e <= s: continue
            if k == "gather": gathers.append((s, e, scope_of(ev)))
            else: computes.append((s, e))

    cunion = merge([(s, e) for s, e in computes])
    tot = collections.Counter(); exp = collections.Counter()
    for s, e, ph in gathers:
        dur = e - s; ov = covered(s, e, cunion)
        tot[ph] += dur; exp[ph] += dur - ov

    gt = sum(tot.values()); ge = sum(exp.values())
    print(f"plane {gpu.name}  | gathers={len(gathers)} compute_kernels={len(computes)}")
    print(f"TOTAL gather {gt/1e6:.0f}ms   EXPOSED {ge/1e6:.0f}ms ({100*ge/gt:.0f}%)   hidden {100*(gt-ge)/gt:.0f}%\n")
    print(f"{'phase':32s} {'gather(ms)':>11s} {'exposed(ms)':>12s} {'exp%':>6s}")
    for ph, _ in tot.most_common():
        print(f"{ph:32s} {tot[ph]/1e6:11.0f} {exp[ph]/1e6:12.0f} {100*exp[ph]/max(tot[ph],1):6.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
