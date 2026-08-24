"""Collective census over compiled HLO text: which collectives run inside while loops
vs entry, and which cross replicate groups — the structural gate for the deferred
(chained-reduced) master reduction. Fail-closed: unparseable headers, missing loops,
and unparseable replica_groups all raise rather than degrade the census."""

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

_COLLECTIVE_OP = re.compile(
    r"\b(all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute|collective-broadcast)"
    r"(?:-start)?\("
)
_REDUCTIONS = {"all-reduce", "reduce-scatter"}


def _computations(hlo: str) -> dict[str, list[str]]:
    comps: dict[str, list[str]] = {}
    # Computation headers start at column 0: `name (params) -> type {` (jitted
    # as_text), or `name {` from printers that omit the signature (proto->text).
    header = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s*(?:\([^)]*\)[^{]*)?\{\s*$")
    current: str | None = None
    for line in hlo.splitlines():
        if current is None:
            m = header.match(line)
            if m is not None:
                current = str(m.group(1))
                comps[current] = []
            else:
                assert not (line and not line[0].isspace() and line.rstrip().endswith("{")), (
                    f"unrecognized computation header (census would silently drop its "
                    f"body): {line.strip()!r}"
                )
        elif line.strip() == "}":
            current = None
        else:
            assert current is not None
            comps[current].append(line)
    assert comps, "no computations parsed from HLO text"
    return comps


def _loop_computations(comps: dict[str, list[str]]) -> set[str]:
    roots: set[str] = set()
    for lines in comps.values():
        for line in lines:
            if " while(" not in line:
                continue
            for key in ("body", "condition"):
                m = re.search(rf"{key}=%?([\w.\-]+)", line)
                assert m is not None, f"while op without {key}=: {line.strip()}"
                roots.add(m.group(1))
    closure: set[str] = set()
    frontier = list(roots)
    while frontier:
        name = frontier.pop()
        if name in closure:
            continue
        closure.add(name)
        for line in comps[name]:
            for ref in re.findall(r"%([\w.\-]+)", line):
                if ref in comps and ref not in closure:
                    frontier.append(ref)
    return closure


def _replica_groups(line: str, n_devices: int) -> list[list[int]]:
    m = re.search(
        r"replica_groups=mesh\[([^\]]*)\]"
        r"(?:,\s*device_ids=\((?:\[([\d,]+)\](?:T\(([\d,]+)\))?|([\d,]+))\))?"
        r"\s*\{([^}]*)\}",
        line,
    )
    if m is not None:
        mesh_axes, dims, perm, flat, coll = m.groups()
        axes = re.findall(r"'([\w]+)'=(\d+)", mesh_axes)
        assert axes, f"unparseable mesh axes: {line.strip()}"
        names = [n for n, _ in axes]
        sizes = [int(s) for _, s in axes]
        if flat is not None:
            ids = np.array([int(v) for v in flat.split(",")])
        elif dims is not None:
            ids = np.arange(int(np.prod(sizes))).reshape([int(d) for d in dims.split(",")])
            if perm is not None:
                ids = ids.transpose([int(p) for p in perm.split(",")])
        else:
            ids = np.arange(int(np.prod(sizes)))
        # device_ids grids may omit size-1 mesh axes; reshape to the full mesh
        assert ids.size == int(np.prod(sizes)), (ids.shape, sizes)
        ids = ids.reshape(sizes)
        collective_axes = set(re.findall(r"'([\w]+)'", coll))
        assert collective_axes and collective_axes <= set(names), (
            f"bad mesh collective axes: {line.strip()}"
        )
        coll_idx = [i for i, n in enumerate(names) if n in collective_axes]
        other_idx = [i for i, n in enumerate(names) if n not in collective_axes]
        group_size = int(np.prod([sizes[i] for i in coll_idx]))
        return ids.transpose(other_idx + coll_idx).reshape(-1, group_size).tolist()

    m = re.search(
        r"replica_groups=(\{\{[\d,{}\s]*\}\}|\{\}|\[\d+,\d+\]<=\[[\d,]+\](?:T\([\d,]+\))?)",
        line,
    )
    assert m is not None, f"collective without parseable replica_groups: {line.strip()}"
    spec = m.group(1)
    if spec == "{}":
        return [list(range(n_devices))]
    if spec.startswith("{{"):
        return [
            [int(d) for d in grp.split(",")] for grp in re.findall(r"\{([\d,\s]+)\}", spec[1:-1])
        ]
    m = re.fullmatch(r"\[(\d+),(\d+)\]<=\[([\d,]+)\](?:T\(([\d,]+)\))?", spec)
    assert m is not None
    n_groups, group_size = int(m.group(1)), int(m.group(2))
    dims = [int(d) for d in m.group(3).split(",")]
    ids = np.arange(int(np.prod(dims))).reshape(dims)
    if m.group(4) is not None:
        ids = ids.transpose([int(p) for p in m.group(4).split(",")])
    return ids.reshape(n_groups, group_size).tolist()


def _spans_replicate(line: str, op: str, replica_stride: int, n_devices: int) -> bool:
    """Whether the collective crosses replicate groups (device id // stride, with the
    mesh grid replicate-major)."""
    if op == "collective-permute":
        pairs = re.findall(r"\{(\d+),(\d+)\}", line)
        return any(int(a) // replica_stride != int(b) // replica_stride for a, b in pairs)
    return any(
        len({d // replica_stride for d in grp}) > 1 for grp in _replica_groups(line, n_devices)
    )


def _reduces_only_scalars(line: str, op_start: int) -> bool:
    return re.search(r"\[\d", line[:op_start]) is None


_DTYPE_BYTES = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "bf16": 2,
    "f16": 2,
    "s16": 2,
    "u16": 2,
    "f32": 4,
    "s32": 4,
    "u32": 4,
    "f64": 8,
    "s64": 8,
    "u64": 8,
}
_RESULT_SHAPE = re.compile(r"\b(pred|s8|u8|bf16|f16|s16|u16|f32|s32|u32|f64|s64|u64)\[([\d,]*)\]")


def _result_bytes(line: str, op_start: int) -> int:
    """Total bytes of the collective's result shapes (the text left of the op name —
    covers both single results and result tuples)."""
    total = 0
    for dtype, dims in _RESULT_SHAPE.findall(line[:op_start]):
        n = 1
        for d in dims.split(","):
            if d:
                n *= int(d)
        total += n * _DTYPE_BYTES[dtype]
    assert total > 0, (
        f"no result shape parsed (dtype outside _DTYPE_BYTES would fail open and slip "
        f"under the sanctioned-smalls bound): {line.strip()!r}"
    )
    return total


@dataclass(frozen=True)
class CollectiveCensus:
    """`counts` keys are `region:op[xrep]`; `in_loop_cross_replicate_bytes` carries one
    result-size entry per in-loop cross-replicate collective, so gates can bound the
    sanctioned smalls (replicated-persisted bias/source whole-batch sums) without a
    reintroduced weight-gradient reduction hiding behind them."""

    counts: dict[str, int]
    in_loop_cross_replicate_bytes: tuple[int, ...]
    exit_reductions: int

    @property
    def in_loop_cross_replicate(self) -> int:
        return len(self.in_loop_cross_replicate_bytes)


def collective_census(hlo: str, *, replica_stride: int, n_devices: int) -> CollectiveCensus:
    """Census the compiled module. `replica_stride` = fsdp*tp for the replicate-major
    `(replicate, fsdp, tp)` grid. Requires at least one while loop: a fully unrolled
    scan would classify everything as entry and pass the in-loop gate vacuously."""
    comps = _computations(hlo)
    loop_names = _loop_computations(comps)
    assert loop_names, (
        "no while loop in the module — an unrolled scan would make the in-loop census "
        "vacuous; census only scan-shaped programs"
    )
    counts: Counter[str] = Counter()
    in_loop_bytes: list[int] = []
    exit_reductions = 0
    for name, lines in comps.items():
        region = "loop" if name in loop_names else "entry"
        for line in lines:
            m = _COLLECTIVE_OP.search(line)
            if m is None:
                continue
            op = m.group(1)
            xrep = _spans_replicate(line, op, replica_stride, n_devices)
            counts[f"{region}:{op}" + ("[xrep]" if xrep else "")] += 1
            if region == "loop" and xrep:
                in_loop_bytes.append(_result_bytes(line, m.start()))
            if (
                region == "entry"
                and xrep
                and op in _REDUCTIONS
                and not _reduces_only_scalars(line, m.start())
            ):
                exit_reductions += 1
    return CollectiveCensus(
        counts=dict(counts),
        in_loop_cross_replicate_bytes=tuple(in_loop_bytes),
        exit_reductions=exit_reductions,
    )
