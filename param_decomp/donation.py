"""Loud verification that a donating jitted step actually reused its input buffers.

Dispatch-time donation failure is SILENT: the `Some donated buffers were not usable`
warning is lowering-time only (jax `mlir.py`), equinox's `donate="all"` suppresses even
that, and on GPU a runtime-blocked donation falls back to a fresh allocation + copy with
no signal at all — the step then peaks ~one full copy of the donated state above steady
state (the resume OOM; lore 2026-07-03--resume-oom-is-buffer-donation-asymmetry).
Aliasing is observable only at the buffer-pointer level, so callers snapshot the input
pointers before the one step they distrust and check reuse after.
"""

import jax


def buffer_bytes_by_ptr(tree: object) -> dict[int, int]:
    """Device-buffer pointer -> shard nbytes over every addressable shard of `tree`'s
    array leaves. Holds no array references (the per-shard views die with the
    comprehension), so a snapshot cannot itself block donation."""
    return {
        shard.data.unsafe_buffer_pointer(): shard.data.nbytes
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
        for shard in leaf.addressable_shards
    }


def reused_fraction(in_buffers: dict[int, int], out_tree: object) -> float:
    """Bytes-weighted fraction of `in_buffers` reappearing under `out_tree`. Pointer SETS,
    not positions: XLA may cross-match same-aval leaves (e.g. Adam m against v)."""
    total = sum(in_buffers.values())
    reused = sum(
        nbytes
        for ptr, nbytes in buffer_bytes_by_ptr(out_tree).items()
        if in_buffers.get(ptr) == nbytes
    )
    return reused / total


def warn_if_not_donated(in_buffers: dict[int, int], out_tree: object, what: str) -> None:
    """Print `DONATION FAILED` when <95% of the snapshotted input bytes were reused."""
    fraction = reused_fraction(in_buffers, out_tree)
    if fraction < 0.95:
        total = sum(in_buffers.values())
        print(
            f"[rank {jax.process_index()}] DONATION FAILED on {what}: only "
            f"{fraction * total / 2**30:.2f}/{total / 2**30:.2f} GiB of input buffers "
            "were reused; this step peaked ~one full copy of the donated state above "
            "steady state (lore 2026-07-03--resume-oom-is-buffer-donation-asymmetry)",
            flush=True,
        )
