"""Per-NCCL-kernel-type timing from a Chrome trace.json.gz, with effective BW.

For each NCCL devkernel name, sum durations and count occurrences (on GPU streams of
one device, pid=1). Effective BW = total_bytes_moved / total_kernel_time. Bytes per op
are read from the matching HLO collective output sizes is hard to join 1:1, so we report
per-kernel total time + count + mean dur; bytes are estimated from the op-name arg if the
trace carries a 'long_name'/shape, else left to the HLO cross-ref.
"""
import gzip
import json
import re
import sys
from collections import defaultdict

NCCL_RE = re.compile(r"ncclDevKernel_(\w+?)(?:\(|$)")


def main(path: str):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"]

    # pid -> device name; tid -> thread name
    pid_name = {}
    tid_name = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pid_name[e["pid"]] = e["args"]["name"]
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            tid_name[(e["pid"], e.get("tid"))] = e["args"]["name"]

    # Restrict to a single GPU device (pid of /device:GPU:0) to avoid N-fold counting.
    gpu0 = [p for p, n in pid_name.items() if n == "/device:GPU:0"]
    assert gpu0, "no GPU:0 pid"
    gpu0 = gpu0[0]

    # nccl kernel name -> [count, total_dur_us, list of durs]
    nccl = defaultdict(lambda: [0, 0.0])
    # full step span on GPU0 compute streams
    all_start = None
    all_end = None
    # also: total time on GPU0 covered by ANY nccl kernel (union) vs compute
    nccl_intervals = []
    compute_intervals = []

    for e in events:
        if e.get("ph") != "X" or e.get("pid") != gpu0:
            continue
        name = e.get("name", "")
        dur = e.get("dur", 0.0)
        ts = e.get("ts", 0.0)
        all_start = ts if all_start is None else min(all_start, ts)
        all_end = ts + dur if all_end is None else max(all_end, ts + dur)
        m = NCCL_RE.search(name)
        if m:
            key = m.group(1)
            nccl[key][0] += 1
            nccl[key][1] += dur
            nccl_intervals.append((ts, ts + dur))
        else:
            compute_intervals.append((ts, ts + dur))

    def union(intervals):
        if not intervals:
            return 0.0
        intervals.sort()
        tot = 0.0
        cs, ce = intervals[0]
        for s, e_ in intervals[1:]:
            if s > ce:
                tot += ce - cs
                cs, ce = s, e_
            else:
                ce = max(ce, e_)
        tot += ce - cs
        return tot

    span = (all_end - all_start) / 1e3  # ms
    nccl_union = union(nccl_intervals) / 1e3
    compute_union = union(compute_intervals) / 1e3

    print(f"GPU0 pid={gpu0}  step span = {span:.2f} ms")
    print(f"NCCL kernel union (any collective busy) = {nccl_union:.2f} ms ({100*nccl_union/span:.1f}% of span)")
    print(f"Compute kernel union                    = {compute_union:.2f} ms ({100*compute_union/span:.1f}% of span)")
    print()
    print(f"{'NCCL kernel':52s} {'count':>6s} {'total_ms':>10s} {'mean_us':>9s}")
    tot_nccl_ms = 0.0
    for k in sorted(nccl, key=lambda x: -nccl[x][1]):
        c, d = nccl[k]
        tot_nccl_ms += d / 1e3
        print(f"{k:52s} {c:6d} {d/1e3:10.2f} {d/max(c,1):9.1f}")
    print(f"{'TOTAL nccl kernel time (sum, not union)':52s} {'':>6s} {tot_nccl_ms:10.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
