"""Pins memreport's proto decode and heap-trace live-range sweep semantics."""

from param_decomp.core.tools.memreport import (
    BufferAssignment,
    HeapEvent,
    HeapEventKind,
    LogicalBuffer,
    parse_buffer_assignment,
    parse_report_proto,
    sweep_live_ranges,
)


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field(number: int, payload: int | bytes) -> bytes:
    if isinstance(payload, int):
        return _varint(number << 3) + _varint(payload)
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _event(kind: int, buffer_id: int, name: str, share_with: int | None = None) -> bytes:
    body = _field(1, kind) + _field(2, buffer_id) + _field(4, name.encode())
    if share_with is not None:
        body += _field(5, share_with)
    return body


def _assignment(buffers: dict[int, int], events: list[bytes]) -> BufferAssignment:
    """Encode a BufferAssignmentProto with one static param allocation (index 0) and one
    arena (index 1) holding every buffer at offset 64*id, then decode it."""
    logical = b"".join(
        _field(1, _field(1, b) + _field(2, size) + _field(3, _field(4, 1000 + b)))
        for b, size in buffers.items()
    )
    param_allocation = _field(3, _field(1, 0) + _field(2, 512) + _field(5, 1))
    arena_assigned = b"".join(
        _field(9, _field(1, b) + _field(2, 64 * b) + _field(3, size)) for b, size in buffers.items()
    )
    arena_allocation = _field(3, _field(1, 1) + _field(2, 4096) + arena_assigned)
    trace = _field(4, b"".join(_field(1, e) for e in events) + _field(2, 1))
    raw = _field(1, b"module bytes") + _field(
        3, logical + param_allocation + arena_allocation + trace
    )
    return parse_buffer_assignment(memoryview(raw))


def test_buffer_assignment_proto_decode() -> None:
    a = _assignment(
        {1: 100, 2: 40},
        [
            _event(0, 1, "fusion.1"),
            _event(0, 2, "fusion.2"),
            _event(1, 1, "fusion.1"),
            _event(1, 2, "fusion.2"),
        ],
    )
    assert a.buffers == {
        1: LogicalBuffer(size_bytes=100, defining_instruction_id=1001, shape_index=()),
        2: LogicalBuffer(size_bytes=40, defining_instruction_id=1002, shape_index=()),
    }
    assert a.allocation_sizes == (512, 4096)
    assert a.entry_parameter_allocation_indices == frozenset({0})
    assert a.arena_allocation_index == 1
    assert a.buffer_arena_offsets == {1: 64, 2: 128}
    assert a.events[0] == HeapEvent(HeapEventKind.ALLOC, 1, "fusion.1", None)
    assert a.events[2].kind is HeapEventKind.FREE


def test_sweep_share_groups_rebirth_and_peak() -> None:
    a = _assignment(
        {1: 100, 2: 100, 3: 40, 4: 100, 5: 10},
        [
            _event(0, 1, "a"),  # e0: live 100
            _event(2, 2, "b", share_with=1),  # e1: joins group 1, live 100
            _event(1, 1, "a"),  # e2: group 1 still referenced by 2
            _event(0, 3, "c"),  # e3: live 140
            _event(1, 2, "b"),  # e4: group 1 dies, live 40
            _event(2, 4, "d", share_with=1),  # e5: REBIRTH of chunk 1, live 140
            _event(0, 5, "e"),  # e6: live 150 — the peak
            _event(1, 4, "d"),  # e7: group 1 dies again, live 50
            _event(1, 3, "c"),  # e8: live 10
            _event(1, 5, "e"),  # e9: live 0
        ],
    )
    liveness = sweep_live_ranges(a)
    assert (liveness.peak_bytes, liveness.peak_event, liveness.n_events) == (150, 6, 10)
    assert {(r.buffer_id, r.born, r.dies) for r in liveness.live_at(6)} == {
        (1, 5, 7),
        (3, 3, 8),
        (5, 6, 9),
    }
    assert {(r.buffer_id, r.size_bytes, r.born, r.dies) for r in liveness.ranges} == {
        (1, 100, 0, 4),
        (1, 100, 5, 7),
        (3, 40, 3, 8),
        (5, 10, 6, 9),
    }


def test_truncated_proto_raises() -> None:
    """A partial dump (OOM-killed writer) must raise, not decode a plausible prefix."""
    entry = _field(1, 1) + _field(2, 100) + _field(3, 100) + _field(5, b"fusion.1")
    raw = _field(1, _field(1, 4096) + _field(3, entry))
    for cut in (len(raw) - 1, len(raw) - 5):
        try:
            parse_report_proto(raw[:cut])
        except (AssertionError, IndexError):
            continue
        raise AssertionError(f"truncated proto at {cut} decoded silently")


def test_report_proto_decode() -> None:
    entry = (
        _field(1, 1)
        + _field(2, 100)
        + _field(3, 100)
        + _field(5, b"fusion.1")
        + _field(5, b"copy.2")
    )
    raw = _field(1, _field(1, 4096) + _field(3, entry))
    peak, buffers = parse_report_proto(raw)
    assert peak == 4096
    ((size, cumulative, instructions),) = [
        (b.size_bytes, b.cumulative_bytes, b.instructions) for b in buffers
    ]
    assert (size, cumulative, instructions) == (100, 100, ("fusion.1", "copy.2"))
