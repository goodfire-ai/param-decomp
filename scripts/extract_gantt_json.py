"""Extract per-pool Gantt data from 3-pool torch.profiler traces → dashboard JSON.

Produces, for one representative ProfilerStep per pool: the compute/nccl/idle summary, a
binned step-relative GPU-occupancy timeline (compute & nccl fractions per bin), and the
CPU-side NCCL time by op kind (recv = blocking wait on another pool). No per-phase data —
this trace has no phase annotations, so granularity is compute vs comm vs idle.

Usage: python scripts/extract_gantt_json.py <trace_dir> <out_json>
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_3pool_trace import (  # noqa: E402
    _is_gpu_kernel,
    _is_nccl,
    _nccl_op_kind,
    clip,
    merged_length,
)

NBINS = 240


def extract_pool(path: Path) -> dict:
    d = json.loads(path.read_text())
    ev = d["traceEvents"]
    rank = d["distributedInfo"]["rank"]
    pool = path.stem.replace("trace_", "").split("_rank")[0]

    steps = [e for e in ev if e.get("ph") == "X" and "ProfilerStep" in str(e.get("name", ""))]
    by_name: dict[str, dict] = {}
    for e in steps:
        if e["name"] not in by_name or e["dur"] > by_name[e["name"]]["dur"]:
            by_name[e["name"]] = e
    steps_meta = sorted(by_name.values(), key=lambda e: e["ts"])
    assert steps_meta, f"no ProfilerStep in {path}"
    # representative = median-wall step (steady state)
    sm = sorted(steps_meta, key=lambda e: e["dur"])[len(steps_meta) // 2]
    w0, w1 = sm["ts"], sm["ts"] + sm["dur"]
    wall = sm["dur"]

    compute, nccl = [], []
    for e in ev:
        if not _is_gpu_kernel(e):
            continue
        iv = (e["ts"], e["ts"] + e.get("dur", 0.0))
        (nccl if _is_nccl(str(e.get("name", ""))) else compute).append(iv)
    comp = clip(compute, (w0, w1))
    ncl = clip(nccl, (w0, w1))

    binw = wall / NBINS
    bins = []
    for i in range(NBINS):
        b0, b1 = w0 + i * binw, w0 + (i + 1) * binw
        cf = merged_length(clip(comp, (b0, b1))) / binw if binw else 0.0
        nf = merged_length(clip(ncl, (b0, b1))) / binw if binw else 0.0
        bins.append([round(min(cf, 1.0), 3), round(min(nf, 1.0), 3)])

    op_time: dict[str, float] = {}
    for e in ev:
        if (
            e.get("cat") == "user_annotation"
            and _is_nccl(str(e.get("name", "")))
            and w0 <= e["ts"] < w1
        ):
            k = _nccl_op_kind(str(e["name"]))
            op_time[k] = op_time.get(k, 0.0) + e.get("dur", 0.0)

    compute_ms = merged_length(comp) / 1000
    nccl_ms = merged_length(ncl) / 1000
    busy_ms = merged_length(comp + ncl) / 1000
    wall_ms = wall / 1000
    return {
        "pool": pool,
        "rank": rank,
        "wall_ms": round(wall_ms, 1),
        "compute_ms": round(compute_ms, 1),
        "nccl_ms": round(nccl_ms, 1),
        "idle_ms": round(wall_ms - busy_ms, 1),
        "idle_pct": round(100 * (wall_ms - busy_ms) / wall_ms, 1),
        "nccl_by_op_ms": {
            k: round(v / 1000, 1) for k, v in sorted(op_time.items(), key=lambda x: -x[1])
        },
        "bins": bins,
    }


def main() -> None:
    trace_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    paths = sorted(trace_dir.glob("trace_*.json"))
    assert paths, f"no traces in {trace_dir}"
    order = {"ci": 0, "layerwise": 1, "ppgd": 2}
    pools = []
    for p in paths:
        print(f"extracting {p.name} ({p.stat().st_size / 1e6:.0f} MB) ...", flush=True)
        pools.append(extract_pool(p))
    pools.sort(key=lambda x: order.get(x["pool"], 9))
    payload = {
        "source": f"b256 prod 160-GPU profile · {trace_dir.name} · representative steady step",
        "nbins": NBINS,
        "step_ms": round(max(p["wall_ms"] for p in pools), 1),
        "pools": pools,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out)
    print(f"wrote {out}")
    for p in pools:
        print(
            f"  {p['pool']:>9} rank{p['rank']}: wall {p['wall_ms']}ms  compute {p['compute_ms']}ms  "
            f"nccl {p['nccl_ms']}ms  idle {p['idle_ms']}ms ({p['idle_pct']}%)  {p['nccl_by_op_ms']}"
        )


if __name__ == "__main__":
    main()
