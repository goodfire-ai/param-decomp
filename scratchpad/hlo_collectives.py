"""Parse XLA after_optimizations HLO: collectives by type, mesh axis, byte volume.

Reports per (op_type, group_size, shapes) the count and per-rank bytes moved.
Group size is read from the replica_groups mesh spec (the axis(es) the collective runs over).
"""
import re
import sys
from collections import defaultdict

DTYPE_BYTES = {"f32": 4, "bf16": 2, "f16": 2, "s32": 4, "s8": 1, "f8e4m3fn": 1, "f8e5m2": 1, "pred": 1, "u32": 4}

# matches e.g. bf16[4096,2048]{1,0}  or  bf16[188743680]{0}
SHAPE_RE = re.compile(r"(f32|bf16|f16|s32|s8|f8e4m3fn|f8e5m2|pred|u32)\[([0-9,]*)\]")
# mesh['axis_0'=8,'axis_1'=1,'axis_2'=4] ... {'axis_0'} or {'axis_0','axis_2'}
MESH_RE = re.compile(r"mesh\[([^\]]*)\]")
AXES_USED_RE = re.compile(r"\}\s*\{([^}]*)\}")  # the {'axis_0'} after device_ids=...


def elems(dims: str) -> int:
    if dims == "":
        return 1
    n = 1
    for d in dims.split(","):
        n *= int(d)
    return n


def shape_bytes(dtype: str, dims: str) -> int:
    return elems(dims) * DTYPE_BYTES[dtype]


def parse_mesh(line: str) -> dict[str, int]:
    m = MESH_RE.search(line)
    if not m:
        return {}
    out = {}
    for part in m.group(1).split(","):
        kv = part.split("=")
        if len(kv) == 2:
            out[kv[0].strip().strip("'")] = int(kv[1])
    return out


def axes_used(line: str) -> list[str]:
    # the set after device_ids=(...) e.g. {'axis_0'} ; fall back to last {...} group of axis names
    cands = re.findall(r"\{([^}]*axis_[^}]*)\}", line)
    if not cands:
        return []
    grp = cands[-1]
    return [a.strip().strip("'") for a in grp.split(",") if "axis" in a]


def group_size(line: str) -> int:
    mesh = parse_mesh(line)
    used = axes_used(line)
    if not mesh or not used:
        return 0
    g = 1
    for a in used:
        g *= mesh.get(a, 1)
    return g


def op_output_bytes(line: str) -> int:
    """Bytes of the *output* tuple of the collective (all gathered shapes summed)."""
    # take shapes only up to the op name / args — output shapes precede the '(' of args.
    # Heuristic: sum all shape matches that appear before the first '(' that opens args.
    head = line.split(" all-gather", 1)[0] if "all-gather" in line else line
    head = re.split(r"all-reduce|reduce-scatter|collective-permute|all-to-all", head)[0]
    total = 0
    for dt, dims in SHAPE_RE.findall(head):
        total += shape_bytes(dt, dims)
    return total


def main(path: str):
    # type -> (group_size) -> [count, total_output_bytes]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # distinct gather output sizes
    gather_sizes = defaultdict(int)  # bytes -> count
    with open(path) as f:
        for line in f:
            for op in ("all-gather-start", "all-reduce-start", "reduce-scatter", "all-to-all", "collective-permute-start"):
                if f" {op}(" in line or "= " in line and op in line and "-done" not in line and "get-tuple" not in line:
                    if op not in line or "-done" in line:
                        continue
                    if op == "reduce-scatter" and "reduce-scatter(" not in line:
                        continue
                    gs = group_size(line)
                    ob = op_output_bytes(line)
                    base = op.replace("-start", "")
                    agg[base][gs][0] += 1
                    agg[base][gs][1] += ob
                    if base == "all-gather":
                        gather_sizes[ob] += 1
                    break
    print("=== collectives by (type, group_size): count, total_output_bytes(GB) ===")
    for t in sorted(agg):
        for gs in sorted(agg[t]):
            c, b = agg[t][gs]
            print(f"{t:24s} group={gs:3d}  count={c:6d}  total_out={b/1e9:8.3f} GB  avg/op={b/max(c,1)/1e6:8.2f} MB")
    print("\n=== distinct all-gather output sizes (top 30 by count) ===")
    for sz, c in sorted(gather_sizes.items(), key=lambda x: -x[1])[:30]:
        print(f"{sz/1e6:9.3f} MB   x{c}")
    print(f"\ndistinct gather sizes: {len(gather_sizes)}")


if __name__ == "__main__":
    main(sys.argv[1])
