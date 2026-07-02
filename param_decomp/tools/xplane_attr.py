"""Full-step GPU time attribution from an uncapped `*.xplane.pb` profile.

The chrome `trace.json.gz` next to it caps at 1M events (a partial step at full32L
scale); the raw xplane protobuf is uncapped. This reads the per-GPU device planes and
reports, per step-window: busy / compute / comms unions, comms split by NCCL collective
kind, EXPOSED comms (not overlapped with any compute kernel on the same GPU), and idle.

Zero deps: a hand-rolled protobuf wire scan of the four fields we need (plane name,
line timestamp, event metadata-id/offset/duration, event-metadata id/name), so it runs
in the repo venv without a generated xplane_pb2.

Usage: python -m param_decomp.tools.xplane_attr <path/to/rank.xplane.pb> [label]
"""

import collections
import sys
from dataclasses import dataclass, field

_PS = 1e-12


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not b & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: bytes):
    """Yield `(field_number, wire_type, scalar_or_bytes)` over one message's wire bytes."""
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        fieldnum, wire = tag >> 3, tag & 7
        match wire:
            case 0:
                val, pos = _read_varint(buf, pos)
                yield fieldnum, wire, val
            case 2:
                length, pos = _read_varint(buf, pos)
                yield fieldnum, wire, buf[pos : pos + length]
                pos += length
            case 1:
                yield fieldnum, wire, buf[pos : pos + 8]
                pos += 8
            case 5:
                yield fieldnum, wire, buf[pos : pos + 4]
                pos += 4
            case _:
                raise AssertionError(f"unexpected wire type {wire} for field {fieldnum}")


@dataclass
class GpuPlane:
    name: str
    event_names: dict[int, str] = field(default_factory=dict)
    # (start_ps, end_ps, metadata_id) per kernel event, all lines/streams merged
    events: list[tuple[int, int, int]] = field(default_factory=list)


def _parse_event(buf: bytes, line_base_ps: int) -> tuple[int, int, int]:
    metadata_id = offset_ps = duration_ps = 0
    for fieldnum, _, val in _iter_fields(buf):
        match fieldnum:
            case 1:
                assert isinstance(val, int)
                metadata_id = val
            case 2:
                assert isinstance(val, int)
                offset_ps = val
            case 3:
                assert isinstance(val, int)
                duration_ps = val
            case _:
                pass
    start = line_base_ps + offset_ps
    return start, start + duration_ps, metadata_id


def _parse_gpu_plane(buf: bytes, name: str) -> GpuPlane:
    plane = GpuPlane(name=name)
    for fieldnum, _, val in _iter_fields(buf):
        match fieldnum:
            case 3:  # XPlane.lines
                assert isinstance(val, bytes)
                timestamp_ns = 0
                event_bufs: list[bytes] = []
                for lf, _, lv in _iter_fields(val):
                    match lf:
                        case 3:
                            assert isinstance(lv, int)
                            timestamp_ns = lv
                        case 4:
                            assert isinstance(lv, bytes)
                            event_bufs.append(lv)
                        case _:
                            pass
                base_ps = timestamp_ns * 1000
                plane.events.extend(_parse_event(eb, base_ps) for eb in event_bufs)
            case 4:  # XPlane.event_metadata: map<int64, XEventMetadata>
                assert isinstance(val, bytes)
                key = 0
                md_name = ""
                for mf, _, mv in _iter_fields(val):
                    match mf:
                        case 1:
                            assert isinstance(mv, int)
                            key = mv
                        case 2:
                            assert isinstance(mv, bytes)
                            for ef, _, ev in _iter_fields(mv):
                                if ef == 2:
                                    assert isinstance(ev, bytes)
                                    md_name = ev.decode()
                        case _:
                            pass
                plane.event_names[key] = md_name
            case _:
                pass
    return plane


def iter_gpu_planes(path: str):
    """Stream the XSpace top level, parsing ONLY `/device:GPU:*` planes (the host plane
    is by far the largest and irrelevant here)."""
    with open(path, "rb") as f:
        buf = f.read()
    for fieldnum, _, plane_buf in _iter_fields(buf):
        if fieldnum != 1:  # XSpace.planes
            continue
        assert isinstance(plane_buf, bytes)
        name = ""
        for pf, _, pv in _iter_fields(plane_buf):
            if pf == 2:
                assert isinstance(pv, bytes)
                name = pv.decode()
                break
        if name.startswith("/device:GPU:"):
            yield _parse_gpu_plane(plane_buf, name)


def union_ps(intervals: list[tuple[int, int]]) -> int:
    intervals.sort()
    total = 0
    cur_s: int | None = None
    cur_e = 0
    for s, e in intervals:
        if cur_s is None or s > cur_e:
            if cur_s is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def intersect_ps(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    a.sort()
    b.sort()
    i = j = total = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def nccl_category(kernel_name: str) -> str:
    if "AllGather" in kernel_name:
        return "AllGather"
    if "AllReduce" in kernel_name:
        return "AllReduce"
    if "ReduceScatter" in kernel_name:
        return "ReduceScatter"
    if "SendRecv" in kernel_name:
        return "SendRecv/Permute"
    return "NCCL-other"


def analyze(path: str, label: str) -> None:
    per_gpu: dict[str, list[float]] = collections.defaultdict(list)
    nccl_counts: collections.Counter[str] = collections.Counter()
    nccl_sum_ps: collections.Counter[str] = collections.Counter()
    n_gpus = 0

    for plane in iter_gpu_planes(path):
        n_gpus += 1
        comms: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        compute: list[tuple[int, int]] = []
        everything: list[tuple[int, int]] = []
        for s, e, metadata_id in plane.events:
            everything.append((s, e))
            name = plane.event_names[metadata_id]
            if "nccl" in name.lower():
                cat = nccl_category(name)
                comms[cat].append((s, e))
                if plane.name == "/device:GPU:0":
                    nccl_counts[cat] += 1
                    nccl_sum_ps[cat] += e - s
            else:
                compute.append((s, e))
        window = max(e for _, e in everything) - min(s for s, _ in everything)
        all_comms = [iv for ivs in comms.values() for iv in ivs]
        busy = union_ps(everything)
        comms_u = union_ps(all_comms)
        per_gpu["window"].append(window * _PS)
        per_gpu["busy"].append(busy * _PS)
        per_gpu["compute"].append(union_ps(compute) * _PS)
        per_gpu["comms"].append(comms_u * _PS)
        per_gpu["exposed_comms"].append((comms_u - intersect_ps(all_comms, compute)) * _PS)
        per_gpu["idle"].append((window - busy) * _PS)
        for cat, ivs in comms.items():
            per_gpu[f"comms:{cat}"].append(union_ps(ivs) * _PS)
            per_gpu[f"exposed:{cat}"].append((union_ps(ivs) - intersect_ps(ivs, compute)) * _PS)

    assert n_gpus, f"no /device:GPU:* planes in {path}"
    window = sum(per_gpu["window"]) / n_gpus
    print(f"===== {label}  ({path.rsplit('/', 1)[-1]}, {n_gpus} GPUs)")
    print(f"window {window:.3f}s (mean over GPUs); mean seconds [fraction of window]")
    keys = [
        "busy",
        "compute",
        "comms",
        "exposed_comms",
        "idle",
        *sorted(k for k in per_gpu if k.startswith("comms:")),
        *sorted(k for k in per_gpu if k.startswith("exposed:")),
    ]
    for k in keys:
        mean = sum(per_gpu[k]) / len(per_gpu[k])
        print(f"  {k:24s} {mean:7.3f}s  [{mean / window * 100:5.1f}%]")
    print("  GPU:0 nccl kernels (count / summed duration):")
    for cat, n in nccl_counts.most_common():
        print(f"    {cat:18s} {n:6d} kernels  sum {nccl_sum_ps[cat] * _PS:7.3f}s")


if __name__ == "__main__":
    assert len(sys.argv) >= 2, __doc__
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
