"""Rigorous GPU-memory attribution from XLA's compile-time dump — no byte-arithmetic guessing.

A run launched with `XLA_FLAGS=--xla_dump_to=<dir>` writes, per compiled module:
  *-memory-usage-report.pb  — buffers ranked by cumulative size (peak total + top entries)
  *.hlo.pb                  — the HloProto: the module plus its BufferAssignmentProto,
                              whose whole-module heap-simulator trace is the live-range
                              record (one ALLOC/FREE/SHARE event stream in schedule order)

The report section answers "what are the biggest buffers?"; the live-range section
answers the sharper question "what is co-resident at the arena's PEAK program point?" —
per buffer: bytes, shape, defining instruction, jax op path, and the [born, dies) event
range. Usage:

    python -m param_decomp.core.tools.memreport <dump_dir|run_dir> [--top N]

The dump lands at runs/<id>/hlo when the launcher dumps HLO (see launch.py XLA_FLAGS).
All proto schemas here are reversed from dumps (no public .proto ships with jaxlib) and
parse fail-closed: unknown wire types and unmodeled structure raise.
"""

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# ── protobuf wire format ─────────────────────────────────────────────────────────────


def walk_fields(buf: memoryview) -> Iterator[tuple[int, int | memoryview]]:
    """Yield `(field_number, value)` — varints as int, length-delimited/fixed as memoryview.
    Truncation (an OOM-killed writer's partial dump) raises rather than yielding a
    plausible prefix."""
    i = 0
    n = len(buf)
    while i < n:
        shift, tag = 0, 0
        while True:
            assert i < n, "truncated proto: varint runs past EOF"
            b = buf[i]
            tag |= (b & 0x7F) << shift
            i += 1
            if not b & 0x80:
                break
            shift += 7
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            shift, val = 0, 0
            while True:
                assert i < n, "truncated proto: varint runs past EOF"
                b = buf[i]
                val |= (b & 0x7F) << shift
                i += 1
                if not b & 0x80:
                    break
                shift += 7
            yield field, val
        elif wire == 1:
            assert i + 8 <= n, "truncated proto: fixed64 runs past EOF"
            yield field, buf[i : i + 8]
            i += 8
        elif wire == 2:
            shift, length = 0, 0
            while True:
                assert i < n, "truncated proto: varint runs past EOF"
                b = buf[i]
                length |= (b & 0x7F) << shift
                i += 1
                if not b & 0x80:
                    break
                shift += 7
            assert i + length <= n, "truncated proto: length-delimited field runs past EOF"
            yield field, buf[i : i + length]
            i += length
        elif wire == 5:
            assert i + 4 <= n, "truncated proto: fixed32 runs past EOF"
            yield field, buf[i : i + 4]
            i += 4
        else:
            raise AssertionError(f"unknown wire type {wire} for field {field}")
    assert i == n


def _int(value: int | memoryview) -> int:
    assert isinstance(value, int)
    return value


def _sub(value: int | memoryview) -> memoryview:
    assert isinstance(value, memoryview)
    return value


def _repeated_ints(value: int | memoryview) -> list[int]:
    """A repeated varint field arrives either one-at-a-time (int) or packed (bytes)."""
    if isinstance(value, int):
        return [value]
    out = []
    i, n = 0, len(value)
    while i < n:
        shift, val = 0, 0
        while True:
            b = value[i]
            val |= (b & 0x7F) << shift
            i += 1
            if not b & 0x80:
                break
            shift += 7
        out.append(val)
    return out


# ── memory-usage-report.pb ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReportBuffer:
    """One buffer of the proto memory-usage report, byte-accounted by XLA itself."""

    size_bytes: int
    cumulative_bytes: int
    instructions: tuple[str, ...]


def parse_report_proto(raw: bytes) -> tuple[int, list[ReportBuffer]]:
    """`(peak_total_bytes, buffers sorted as ranked by XLA)`. Schema: top field 1 =
    report; report field 1 = peak total bytes, repeated field 3 = buffer entry
    {1: rank, 2: size bytes, 3: cumulative bytes, 4: f64 cumulative fraction,
    5 (repeated): instruction names, 6: rendering}."""
    (report,) = [v for f, v in walk_fields(memoryview(raw)) if f == 1]
    peak = 0
    buffers: list[ReportBuffer] = []
    for field, value in walk_fields(_sub(report)):
        if field == 1:
            peak = _int(value)
        elif field == 3:
            size, cumulative, names = 0, 0, []
            for entry_field, entry_value in walk_fields(_sub(value)):
                if entry_field == 2:
                    size = _int(entry_value)
                elif entry_field == 3:
                    cumulative = _int(entry_value)
                elif entry_field == 5:
                    names.append(bytes(_sub(entry_value)).decode())
            assert size and names, "report entry without size/instructions — schema drift"
            assert cumulative >= (buffers[-1].cumulative_bytes if buffers else 0), (
                "report entries out of cumulative order — schema drift"
            )
            buffers.append(
                ReportBuffer(
                    size_bytes=size, cumulative_bytes=cumulative, instructions=tuple(names)
                )
            )
    assert peak and buffers, "report parsed empty — schema drift"
    return peak, buffers


# ── hlo.pb: buffer assignment + heap-simulator live ranges ───────────────────────────


class HeapEventKind(Enum):
    ALLOC = 0
    FREE = 1
    SHARE_WITH = 2


@dataclass(frozen=True)
class HeapEvent:
    kind: HeapEventKind
    buffer_id: int
    instruction_name: str
    share_with_canonical_id: int | None


@dataclass(frozen=True)
class LogicalBuffer:
    size_bytes: int
    defining_instruction_id: int
    shape_index: tuple[int, ...]


@dataclass(frozen=True)
class BufferAssignment:
    """The byte-relevant slice of XLA's BufferAssignmentProto (HloProto field 3).
    `arena_allocation_index` is the one allocation every heap-traced buffer is assigned
    into — the temporally-packed slab; every other allocation is static for the step."""

    buffers: dict[int, LogicalBuffer]
    buffer_arena_offsets: dict[int, int]
    allocation_sizes: tuple[int, ...]
    entry_parameter_allocation_indices: frozenset[int]
    arena_allocation_index: int
    events: tuple[HeapEvent, ...]


def parse_buffer_assignment(hlo_proto: memoryview) -> BufferAssignment:
    """Schema: BufferAssignmentProto {1: logical_buffer {1: id, 2: size, 3: location
    {3 (repeated): shape_index, 4: instruction id}}, 3: allocation {1: index, 2: size,
    5: is_entry_computation_parameter, 9: assigned {1: buffer id, 2: offset, 3: size}},
    4: heap_simulator_trace {1: event {1: kind, 2: buffer id, 4: instruction name,
    5: share-with canonical id}, 2: whole_module}}. Exactly one whole-module trace
    expected."""
    (assignment,) = [v for f, v in walk_fields(hlo_proto) if f == 3]
    buffers: dict[int, LogicalBuffer] = {}
    allocations: dict[int, tuple[int, bool]] = {}
    allocation_of_buffer: dict[int, int] = {}
    offset_of_buffer: dict[int, int] = {}
    traces: list[tuple[bool, tuple[HeapEvent, ...]]] = []
    for field, value in walk_fields(_sub(assignment)):
        if field == 1:
            buffer_id, size, instruction_id, shape_index = 0, 0, 0, []
            for f2, v2 in walk_fields(_sub(value)):
                if f2 == 1:
                    buffer_id = _int(v2)
                elif f2 == 2:
                    size = _int(v2)
                elif f2 == 3:
                    for f3, v3 in walk_fields(_sub(v2)):
                        if f3 == 3:
                            shape_index.extend(_repeated_ints(v3))
                        elif f3 == 4:
                            instruction_id = _int(v3)
            buffers[buffer_id] = LogicalBuffer(size, instruction_id, tuple(shape_index))
        elif field == 3:
            index, size, is_param = 0, 0, False
            assigned: list[tuple[int, int]] = []
            for f2, v2 in walk_fields(_sub(value)):
                if f2 == 1:
                    index = _int(v2)
                elif f2 == 2:
                    size = _int(v2)
                elif f2 == 5:
                    is_param = bool(_int(v2))
                elif f2 == 9:
                    entry_buffer_id, offset = 0, 0
                    for f3, v3 in walk_fields(_sub(v2)):
                        if f3 == 1:
                            entry_buffer_id = _int(v3)
                        elif f3 == 2:
                            offset = _int(v3)
                    assigned.append((entry_buffer_id, offset))
            allocations[index] = (size, is_param)
            for entry_buffer_id, offset in assigned:
                allocation_of_buffer[entry_buffer_id] = index
                offset_of_buffer[entry_buffer_id] = offset
        elif field == 4:
            whole_module = False
            events: list[HeapEvent] = []
            for f2, v2 in walk_fields(_sub(value)):
                if f2 == 1:
                    kind, buffer_id, name, share_with = 0, 0, "", None
                    for f3, v3 in walk_fields(_sub(v2)):
                        if f3 == 1:
                            kind = _int(v3)
                        elif f3 == 2:
                            buffer_id = _int(v3)
                        elif f3 == 4:
                            name = bytes(_sub(v3)).decode()
                        elif f3 == 5:
                            share_with = _int(v3)
                    events.append(HeapEvent(HeapEventKind(kind), buffer_id, name, share_with))
                elif f2 == 2:
                    whole_module = bool(_int(v2))
            traces.append((whole_module, tuple(events)))
    ((whole_module, trace_events),) = traces
    assert whole_module, "expected a whole-module heap-simulator trace"
    assert allocations, "no buffer allocations parsed — schema drift"
    n = len(allocations)
    assert sorted(allocations) == list(range(n)), "allocation indices not dense"
    (arena_index,) = {
        allocation_of_buffer[e.buffer_id] for e in trace_events if e.kind is HeapEventKind.ALLOC
    }
    return BufferAssignment(
        buffers=buffers,
        buffer_arena_offsets={
            b: offset_of_buffer[b] for b, i in allocation_of_buffer.items() if i == arena_index
        },
        allocation_sizes=tuple(allocations[i][0] for i in range(n)),
        entry_parameter_allocation_indices=frozenset(i for i in range(n) if allocations[i][1]),
        arena_allocation_index=arena_index,
        events=trace_events,
    )


# ── hlo.pb: instruction index (names, shapes, jax op paths) ──────────────────────────

_PRIMITIVE_TYPES = {
    1: "pred", 2: "s8", 3: "s16", 4: "s32", 5: "s64", 6: "u8", 7: "u16", 8: "u32",
    9: "u64", 10: "f16", 11: "f32", 12: "f64", 13: "tuple", 14: "opaque", 15: "c64",
    16: "bf16", 17: "token", 18: "c128", 19: "f8e5m2", 20: "f8e4m3fn", 21: "s4",
    22: "u4", 23: "f8e4m3b11fnuz", 24: "f8e5m2fnuz", 25: "f8e4m3fnuz", 26: "f8e4m3",
    27: "f8e3m4", 28: "s2", 29: "u2", 30: "f4e2m1fn", 31: "f8e8m0fnu",
}  # fmt: skip


def _render_shape(shape: memoryview, shape_index: tuple[int, ...]) -> str:
    """ShapeProto {2: element_type, 3 (repeated): dimensions, 4 (repeated): tuple_shapes},
    descended through `shape_index` for a buffer inside a tuple-producing instruction."""
    if shape_index:
        tuple_shapes = [v for f, v in walk_fields(shape) if f == 4]
        return _render_shape(_sub(tuple_shapes[shape_index[0]]), shape_index[1:])
    element_type, dims = 0, []
    for field, value in walk_fields(shape):
        if field == 2:
            element_type = _int(value)
        elif field == 3:
            dims.extend(_repeated_ints(value))
    return f"{_PRIMITIVE_TYPES[element_type]}[{','.join(map(str, dims))}]"


@dataclass(frozen=True)
class Instruction:
    name: str
    shape: memoryview
    op_path: str


def parse_instruction_index(hlo_proto: memoryview) -> dict[int, Instruction]:
    """Instruction id → (name, shape proto, jax metadata op path), from the HloModuleProto
    (HloProto field 1; computations at field 3, instructions at computation field 2;
    instruction {1: name, 3: shape, 7: metadata {2: op_name}, 35: id})."""
    (module,) = [v for f, v in walk_fields(hlo_proto) if f == 1]
    index: dict[int, Instruction] = {}
    for field, computation in walk_fields(_sub(module)):
        if field != 3:
            continue
        for f2, instruction in walk_fields(_sub(computation)):
            if f2 != 2:
                continue
            name, shape, op_path, instruction_id = "", memoryview(b""), "", 0
            for f3, v3 in walk_fields(_sub(instruction)):
                if f3 == 1:
                    name = bytes(_sub(v3)).decode()
                elif f3 == 3:
                    shape = _sub(v3)
                elif f3 == 7:
                    for f4, v4 in walk_fields(_sub(v3)):
                        if f4 == 2:
                            op_path = bytes(_sub(v4)).decode()
                elif f3 == 35:
                    instruction_id = _int(v3)
            index[instruction_id] = Instruction(name, shape, op_path)
    assert index, "no instructions parsed — schema drift"
    return index


# ── live-range sweep ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LiveRange:
    """One canonical (bytes-owning) buffer's residency: live over events [born, dies)."""

    buffer_id: int
    size_bytes: int
    born: int
    dies: int


@dataclass(frozen=True)
class ArenaLiveness:
    peak_bytes: int
    peak_event: int
    n_events: int
    ranges: tuple[LiveRange, ...]

    def live_at(self, event: int) -> tuple[LiveRange, ...]:
        return tuple(r for r in self.ranges if r.born <= event < r.dies)


def sweep_live_ranges(assignment: BufferAssignment) -> ArenaLiveness:
    """Replay the heap-simulator trace. SHARE_WITH joins the canonical buffer's group
    without new bytes; a group's bytes release at its last FREE. A SHARE_WITH against a
    dead canonical (the collective-pipeliner's loop-carried double-buffer chunks) is a
    rebirth: the same chunk re-allocates, opening a fresh live range under the canonical
    id. The peak event is the program point of maximum simultaneous live bytes."""
    canonical: dict[int, int] = {}
    group_refs: dict[int, int] = {}
    born: dict[int, int] = {}
    ranges: list[LiveRange] = []
    live = peak = peak_event = 0

    def allocate(owner: int, size: int, event_index: int) -> None:
        nonlocal live, peak, peak_event
        group_refs[owner] = 1
        born[owner] = event_index
        live += size
        if live > peak:
            peak, peak_event = live, event_index

    for i, event in enumerate(assignment.events):
        match event.kind:
            case HeapEventKind.ALLOC:
                assert event.buffer_id not in canonical
                canonical[event.buffer_id] = event.buffer_id
                allocate(event.buffer_id, assignment.buffers[event.buffer_id].size_bytes, i)
            case HeapEventKind.SHARE_WITH:
                assert event.share_with_canonical_id is not None
                owner = canonical.get(event.share_with_canonical_id, event.share_with_canonical_id)
                canonical[event.buffer_id] = owner
                if owner in born:
                    group_refs[owner] += 1
                else:
                    assert (
                        assignment.buffers[event.buffer_id].size_bytes
                        == assignment.buffers[owner].size_bytes
                    )
                    allocate(owner, assignment.buffers[owner].size_bytes, i)
            case HeapEventKind.FREE:
                owner = canonical.pop(event.buffer_id)
                group_refs[owner] -= 1
                if group_refs[owner] == 0:
                    size = assignment.buffers[owner].size_bytes
                    live -= size
                    ranges.append(LiveRange(owner, size, born.pop(owner), i))
    assert live == 0 and not canonical and not born, "trace did not free every buffer"
    return ArenaLiveness(peak, peak_event, len(assignment.events), tuple(ranges))


# ── CLI ──────────────────────────────────────────────────────────────────────────────

GIB = 2**30


def _print_report_section(path: Path, top: int) -> None:
    print(f"report: {path.name}")
    peak, buffers = parse_report_proto(path.read_bytes())
    print(f"peak/total: {peak / GIB:.2f} GiB")
    print(f"\n=== top {top} buffers (size; cumulative; instructions) ===")
    for b in buffers[:top]:
        names = ", ".join(b.instructions)[:100]
        print(f"  {b.size_bytes / GIB:7.2f} GiB  cum {b.cumulative_bytes / GIB:7.2f} GiB  {names}")


def _print_live_range_section(path: Path, top: int) -> None:
    print(f"\nhlo proto: {path.name}")
    hlo_proto = memoryview(path.read_bytes())
    assignment = parse_buffer_assignment(hlo_proto)
    instructions = parse_instruction_index(hlo_proto)
    liveness = sweep_live_ranges(assignment)

    arena = assignment.allocation_sizes[assignment.arena_allocation_index]
    parameters = sum(
        assignment.allocation_sizes[i] for i in assignment.entry_parameter_allocation_indices
    )
    static = sum(assignment.allocation_sizes) - arena - parameters
    assert liveness.peak_bytes <= arena, "swept peak exceeds the arena allocation"
    peak_intervals = sorted(
        (assignment.buffer_arena_offsets[r.buffer_id], r.size_bytes)
        for r in liveness.live_at(liveness.peak_event)
    )
    for (offset, size), (next_offset, _) in zip(peak_intervals, peak_intervals[1:], strict=False):
        assert offset + size <= next_offset, "peak-live buffers overlap in the arena"
    print(
        f"allocations: parameters {parameters / GIB:.2f} GiB"
        f" + other static {static / GIB:.2f} GiB"
        f" + arena {arena / GIB:.2f} GiB (fragmentation {(arena - liveness.peak_bytes) / GIB:.2f})"
    )
    print(
        f"arena peak: {liveness.peak_bytes / GIB:.2f} GiB live at event"
        f" {liveness.peak_event}/{liveness.n_events}"
        f" ({assignment.events[liveness.peak_event].instruction_name})"
    )
    print(f"\n=== top {top} buffers live at the peak (size; born-dies; shape; instruction) ===")
    at_peak = sorted(liveness.live_at(liveness.peak_event), key=lambda r: -r.size_bytes)
    for r in at_peak[:top]:
        buffer = assignment.buffers[r.buffer_id]
        instruction = instructions[buffer.defining_instruction_id]
        shape = _render_shape(instruction.shape, buffer.shape_index)
        print(
            f"  {r.size_bytes / GIB:7.2f} GiB  [{r.born:>5}, {r.dies:>5})  {shape:<24}"
            f" {instruction.name}  ({instruction.op_path[:80]})"
        )


def main() -> None:
    root = Path(sys.argv[1])
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 20
    reports = sorted(
        root.glob("**/*jit_step*memory-usage-report.pb"), key=lambda p: p.stat().st_size
    )
    hlo_protos = sorted(root.glob("**/*jit_step*.hlo.pb"), key=lambda p: p.stat().st_size)
    assert reports or hlo_protos, f"no jit_step memory-usage-report.pb / hlo.pb under {root}"
    if reports:
        _print_report_section(reports[-1], top)
    if hlo_protos:
        _print_live_range_section(hlo_protos[-1], top)


if __name__ == "__main__":
    main()
