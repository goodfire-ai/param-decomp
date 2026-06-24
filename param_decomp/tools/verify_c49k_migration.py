"""Prove the C49k checkpoint migration is VALUE-exact, not just structurally sound.

`migrate_c49k_checkpoint.py` is a pure copy + reshape (re-key + squeeze the legacy
leading singleton) — no recompute. So every leaf of the migrated tree MUST equal its
source leaf bit-for-bit once the singleton is squeezed and the keys are re-mapped. A
swapped `V`<->`U`, a mis-mapped `g/u/d`->`gate/up/down`, or a wrong squeeze axis would
pass the migration's structure-only checks (shapes / step / finiteness) yet silently
corrupt the fine-tune base; only a per-leaf value comparison against the CORRECTLY-mapped
source leaf catches it.

The comparison table here is the INVERSE of `migrate_c49k_checkpoint.py`'s remap, built
from that tool's own constants (`KIND_TO_SITE_SUFFIX`, `SOURCE_STATE_KEY`) so the mapping
under test and the mapping applied are the same object. We stream leaf-by-leaf: restore
one source leaf + its migrated counterpart single-device on CPU, compare, free, advance —
never holding both 47 GB trees at once (that OOM-killed the migration under the
interactive cgroup cap).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import jax
import numpy as np
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import SingleDeviceSharding
from jax.tree_util import KeyEntry

from param_decomp.tools.migrate_c49k_checkpoint import (
    KIND_TO_SITE_SUFFIX,
    SOURCE_STATE_KEY,
)

DEFAULT_SRC = Path(
    "/mnt/data/artifacts/mechanisms/param-decomp/jax_runs/jax-l18-C49k-200k/preserved_ckpts/175000"
)
DEFAULT_DST = Path("/mnt/data/artifacts/mechanisms/param-decomp/runs/p-bd3cd4d4/ckpts/175000")


@dataclass(frozen=True)
class LeafPair:
    """One migrated leaf and the source leaf it must equal, under `group` in the report."""

    group: str
    dst_keystr: str
    src_keystr: str
    squeeze_singleton: bool


def _decomp_vu_pairs(group: str, dst_prefix: str, src_prefix: str) -> list[LeafPair]:
    """The 6 V/U component leaves: OLD flat `{V,U}{g,u,d}` (3-D, leading singleton) ->
    NEW `vu['<site>'][0|1]` (2-D), V->[0] U->[1], the singleton squeezed. Mirrors the
    migration's `_old_decomp_vu_to_new` exactly (same `KIND_TO_SITE_SUFFIX`, same
    V->0 / U->1, same squeeze)."""
    pairs: list[LeafPair] = []
    for kind, suffix in KIND_TO_SITE_SUFFIX.items():
        site = f"layers.18.mlp.{suffix}"
        for vu_index, vu_letter in ((0, "V"), (1, "U")):
            pairs.append(
                LeafPair(
                    group=group,
                    dst_keystr=f"{dst_prefix}['vu']['{site}'][{vu_index}]",
                    src_keystr=f"{src_prefix}['{vu_letter}{kind}']",
                    squeeze_singleton=True,
                )
            )
    return pairs


def _ci_fn_pairs(group: str, dst_prefix: str, src_prefix: str) -> list[LeafPair]:
    """ci_fn sub-tree: IDENTICAL leaf names old and new -> straight copy, no squeeze."""
    scalars = ("in_proj_w", "in_proj_b", "out_w", "out_b", "inv_freq")
    block_fields = ("wq", "wk", "wv", "wo", "w1", "b1", "w2", "b2")
    pairs: list[LeafPair] = []
    for field in scalars:
        pairs.append(LeafPair(group, f"{dst_prefix}['{field}']", f"{src_prefix}['{field}']", False))
    for i in range(4):
        for field in block_fields:
            sub = f"['blocks'][{i}]['{field}']"
            pairs.append(LeafPair(group, dst_prefix + sub, src_prefix + sub, False))
    return pairs


def build_leaf_pairs() -> list[LeafPair]:
    """The full inverse remap: every migrated leaf -> the source leaf it must equal."""
    pairs: list[LeafPair] = []

    pairs += _decomp_vu_pairs("components", "['components']", "['components']")

    co_dst = "['components_opt_state'][1][0]"
    co_src = "['components_opt_state'][1][0]"
    pairs.append(
        LeafPair("components_opt_state", f"{co_dst}['count']", f"{co_src}['count']", False)
    )
    pairs.append(
        LeafPair(
            "components_opt_state",
            "['components_opt_state'][1][2]['count']",
            "['components_opt_state'][1][2]['count']",
            False,
        )
    )
    for moment in ("mu", "nu"):
        pairs += _decomp_vu_pairs(
            "components_opt_state", f"{co_dst}['{moment}']", f"{co_src}['{moment}']"
        )

    pairs += _ci_fn_pairs("ci_fn", "['ci_fn']", "['ci_fn']")

    cio_dst = "['ci_fn_opt_state'][0]"
    cio_src = "['ci_fn_opt_state'][0]"
    pairs.append(LeafPair("ci_fn_opt_state", f"{cio_dst}['count']", f"{cio_src}['count']", False))
    pairs.append(
        LeafPair(
            "ci_fn_opt_state",
            "['ci_fn_opt_state'][2]['count']",
            "['ci_fn_opt_state'][2]['count']",
            False,
        )
    )
    for moment in ("mu", "nu"):
        pairs += _ci_fn_pairs("ci_fn_opt_state", f"{cio_dst}['{moment}']", f"{cio_src}['{moment}']")

    adv_dst = f"['adversaries']['{SOURCE_STATE_KEY}']"
    sites = [f"layers.18.mlp.{suffix}" for suffix in KIND_TO_SITE_SUFFIX.values()]
    for site in sites:
        pairs.append(
            LeafPair(
                "sources",
                f"{adv_dst}['sources']['{site}']",
                f"['sources']['{site}']",
                False,
            )
        )
    so_dst = f"{adv_dst}['opt_state']"
    for moment in ("m", "v"):
        for site in sites:
            pairs.append(
                LeafPair(
                    "sources_opt_state",
                    f"{so_dst}['{moment}']['{site}']",
                    f"['sources_adam_state']['{moment}']['{site}']",
                    False,
                )
            )
    pairs.append(
        LeafPair(
            "sources_opt_state",
            f"{so_dst}['step_count']",
            "['sources_adam_state']['step_count']",
            False,
        )
    )

    pairs.append(LeafPair("step", "['step']", "['step']", False))
    return pairs


@runtime_checkable
class ShapeDtypeLeaf(Protocol):
    """The shape/dtype-carrying interface orbax metadata leaves expose (`ArrayMetadata` /
    `ScalarMetadata`); the only thing the restore-target builders read off a leaf."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...


def _restore_single_leaf(
    ckpt_default: epath.Path, keystr: str, sharding: SingleDeviceSharding
) -> np.ndarray:
    """Restore exactly ONE leaf (by keystr) from an orbax checkpoint, single-device CPU,
    every other leaf a PLACEHOLDER (never read from disk / never materialized)."""
    meta = ocp.PyTreeCheckpointHandler().metadata(ckpt_default).tree

    def pick(path: tuple[KeyEntry, ...], leaf: ShapeDtypeLeaf) -> object:
        if jax.tree_util.keystr(path) == keystr:
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)
        return ocp.PLACEHOLDER

    def rargs(path: tuple[KeyEntry, ...], leaf: ShapeDtypeLeaf) -> ocp.RestoreArgs:
        if jax.tree_util.keystr(path) == keystr:
            return ocp.ArrayRestoreArgs(sharding=sharding, dtype=leaf.dtype)
        return ocp.RestoreArgs()

    target = jax.tree_util.tree_map_with_path(pick, meta)
    restore_args = jax.tree_util.tree_map_with_path(rargs, meta)
    with ocp.Checkpointer(ocp.PyTreeCheckpointHandler()) as ckptr:
        restored = ckptr.restore(
            ckpt_default, args=ocp.args.PyTreeRestore(item=target, restore_args=restore_args)
        )
    leaves = [leaf for _, leaf in jax.tree_util.tree_flatten_with_path(restored)[0]]
    materialized = [np.asarray(leaf) for leaf in leaves if leaf is not ocp.PLACEHOLDER]
    assert len(materialized) == 1, (
        f"expected exactly one leaf for {keystr}, got {len(materialized)}"
    )
    return materialized[0]


@dataclass
class LeafVerdict:
    pair: LeafPair
    equal: bool
    detail: str


def verify_pair(
    src_default: epath.Path,
    dst_default: epath.Path,
    pair: LeafPair,
    sharding: SingleDeviceSharding,
) -> LeafVerdict:
    src = _restore_single_leaf(src_default, pair.src_keystr, sharding)
    dst = _restore_single_leaf(dst_default, pair.dst_keystr, sharding)
    src_cmp = np.squeeze(src, axis=0) if pair.squeeze_singleton else src
    shape_dtype_ok = src_cmp.shape == dst.shape and src_cmp.dtype == dst.dtype
    equal = bool(shape_dtype_ok and np.array_equal(src_cmp, dst))
    if equal:
        detail = f"{dst.shape} {dst.dtype}"
    elif not shape_dtype_ok:
        detail = f"SHAPE/DTYPE: src{src_cmp.shape}{src_cmp.dtype} vs dst{dst.shape}{dst.dtype}"
    else:
        diff = np.abs(src_cmp.astype(np.float64) - dst.astype(np.float64))
        n_diff = int(np.count_nonzero(src_cmp != dst))
        detail = f"VALUE: {n_diff} elems differ, max|Δ|={float(diff.max()):.3e}"
    return LeafVerdict(pair=pair, equal=equal, detail=detail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="OLD orbax ckpt step dir")
    parser.add_argument(
        "--dst", type=Path, default=DEFAULT_DST, help="MIGRATED orbax ckpt step dir"
    )
    args = parser.parse_args()

    src_default = epath.Path(args.src.resolve() / "default")
    dst_default = epath.Path(args.dst.resolve() / "default")
    assert src_default.exists(), f"src not found: {src_default}"
    assert dst_default.exists(), f"dst not found: {dst_default}"

    cpus = jax.devices("cpu")
    assert cpus, "no CPU device"
    sharding = SingleDeviceSharding(cpus[0])

    pairs = build_leaf_pairs()
    print(f"src : {src_default}")
    print(f"dst : {dst_default}")
    print(f"verifying {len(pairs)} leaves (streamed single-device CPU, one pair at a time)\n")

    src_keystrs = {
        jax.tree_util.keystr(p)
        for p, _ in jax.tree_util.tree_flatten_with_path(
            ocp.PyTreeCheckpointHandler().metadata(src_default).tree
        )[0]
    }
    dst_keystrs = {
        jax.tree_util.keystr(p)
        for p, _ in jax.tree_util.tree_flatten_with_path(
            ocp.PyTreeCheckpointHandler().metadata(dst_default).tree
        )[0]
    }
    mapped_src = {p.src_keystr for p in pairs}
    mapped_dst = {p.dst_keystr for p in pairs}
    assert mapped_src == src_keystrs, (
        "remap does not cover the source tree 1:1",
        sorted(src_keystrs - mapped_src),
        sorted(mapped_src - src_keystrs),
    )
    assert mapped_dst == dst_keystrs, (
        "remap does not cover the migrated tree 1:1",
        sorted(dst_keystrs - mapped_dst),
        sorted(mapped_dst - dst_keystrs),
    )
    print("coverage: every source leaf and every migrated leaf is mapped exactly once\n")

    verdicts = [verify_pair(src_default, dst_default, pair, sharding) for pair in pairs]
    for v in verdicts:
        status = "PASS" if v.equal else "FAIL"
        print(f"  [{status}] {v.pair.dst_keystr}  <-  {v.pair.src_keystr}  {v.detail}")

    groups = sorted({p.group for p in pairs})
    print("\nper-group:")
    all_pass = True
    for group in groups:
        gv = [v for v in verdicts if v.pair.group == group]
        n_pass = sum(1 for v in gv if v.equal)
        ok = n_pass == len(gv)
        all_pass = all_pass and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {group:24s} {n_pass}/{len(gv)} leaves bit-identical")

    print()
    if all_pass:
        print(f"VERDICT: PASS — all {len(verdicts)} leaves bit-identical under the remap.")
        print("The migrated 175k checkpoint is VALUE-EXACT.")
    else:
        failed = [v for v in verdicts if not v.equal]
        print(
            f"VERDICT: FAIL — {len(failed)}/{len(verdicts)} leaves mismatch. MIGRATION IS CORRUPT:"
        )
        for v in failed:
            print(f"  {v.pair.dst_keystr}  <-  {v.pair.src_keystr}  {v.detail}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
