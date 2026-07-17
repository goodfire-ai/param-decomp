"""Strip a finished run's pre-split orbax checkpoint to its decomposition item.

A pre-split trainer checkpoint is one orbax item (`default`) dominated by
resume-only state: the persistent-PGD adversary (sources + their Adam moments)
and the V/U + CI-fn optimizer moments are ~90% of the bytes; the trained
decomposition (`components` + `ci_fn`) that consumers actually restore is the
small rest. Stripping rewrites `ckpts/<step>` into the split two-item layout
(SPEC S22) with ONLY the `decomposition` item — this is the pre-split-run
migration the split's clean format break requires, minus the training item.
Consumers (`restore_decomposition*`) load a stripped checkpoint natively; the
run can no longer warm-resume training — only strip finished or abandoned runs.

Dry-run by default: prints the kept/dropped byte split from checkpoint metadata.
`--apply` stages the stripped checkpoint next to `ckpts/`, verifies its tree
against the original's metadata, then swaps it in. (If interrupted between the
swap's delete and rename, the staged copy survives under `ckpts_strip_staging/`.)

    python -m param_decomp.tools.strip_checkpoint <run_dir> [--step N] [--apply]
"""

import argparse
import math
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import orbax.checkpoint as ocp

DECOMPOSITION_KEYS = ("components", "ci_fn")
DROPPABLE_KEYS = frozenset(
    {
        "step",
        "adversaries",
        "sources",
        "sources_opt_state",
        "components_opt_state",
        "ci_fn_opt_state",
    }
)


def open_manager(ckpts_dir: Path) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        ckpts_dir.resolve(),
        options=ocp.CheckpointManagerOptions(enable_async_checkpointing=False),
    )


def leaf_bytes(tree: Any) -> int:
    leaves = [leaf for leaf in jax.tree.leaves(tree) if hasattr(leaf, "shape")]
    assert leaves, "metadata subtree has no array leaves"
    return sum(math.prod(leaf.shape) * np.dtype(leaf.dtype).itemsize for leaf in leaves)


def tree_shapes(tree: Any) -> Any:
    return jax.tree.map(lambda leaf: (tuple(leaf.shape), str(np.dtype(leaf.dtype))), tree)


def du_bytes(path: Path) -> int:
    du = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, check=True)
    return int(du.stdout.split()[0])


def read_tree_metadata(item_dir: Path) -> Mapping[str, Any]:
    metadata = ocp.PyTreeCheckpointer().metadata(item_dir).item_metadata
    assert metadata is not None, f"no pytree metadata under {item_dir}"
    return cast(Mapping[str, Any], metadata)


def restore_decomposition_subtrees(
    manager: ocp.CheckpointManager, step: int, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Partial-restore only the decomposition subtrees of a pre-split (single
    `default`-item) checkpoint, as host numpy (no shardings)."""
    item = {
        key: jax.tree.map(lambda m: jax.ShapeDtypeStruct(m.shape, m.dtype), metadata[key])
        for key in DECOMPOSITION_KEYS
    }
    restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), item)
    restored = manager.restore(
        step,
        args=ocp.args.PyTreeRestore(item=item, restore_args=restore_args, partial_restore=True),
    )
    return cast(dict[str, Any], restored)


def main() -> None:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--step", type=int, help="checkpoint step (default: latest)")
    parser.add_argument(
        "--apply", action="store_true", help="actually strip (default: dry-run report)"
    )
    args = parser.parse_args()

    ckpts_dir = args.run_dir / "ckpts"
    assert ckpts_dir.is_dir(), f"no ckpts dir under {args.run_dir}"
    manager = open_manager(ckpts_dir)
    step = args.step if args.step is not None else manager.latest_step()
    assert step is not None, f"no checkpoints under {ckpts_dir}"
    assert step in manager.all_steps(), f"step {step} not in {sorted(manager.all_steps())}"
    assert (ckpts_dir / str(step) / "default").is_dir(), (
        f"{ckpts_dir / str(step)} is not a pre-split single-item checkpoint "
        "(already split/stripped?)"
    )

    metadata = read_tree_metadata(ckpts_dir / str(step) / "default")
    present: set[str] = set(metadata.keys())
    keep: set[str] = set(DECOMPOSITION_KEYS)
    assert keep <= present, f"checkpoint lacks {keep - present}"
    dropped_keys = present - keep
    assert dropped_keys <= DROPPABLE_KEYS, (
        f"unknown training-state keys: {dropped_keys - DROPPABLE_KEYS}"
    )

    kept = {key: leaf_bytes(metadata[key]) for key in DECOMPOSITION_KEYS}
    dropped = {key: leaf_bytes(metadata[key]) for key in sorted(dropped_keys) if key != "step"}
    on_disk = du_bytes(ckpts_dir / str(step))
    total_logical = sum(kept.values()) + sum(dropped.values())
    print(f"{args.run_dir.name} ckpts/{step}: {on_disk / 2**30:.1f} GiB on disk")
    for label, group in (("keep", kept), ("drop", dropped)):
        for key, size in sorted(group.items(), key=lambda kv: -kv[1]):
            print(f"  {label}  {size / 2**30:9.2f} GiB  {key}")
    kept_frac = sum(kept.values()) / total_logical
    print(
        f"stripped size ~{on_disk * kept_frac / 2**30:.1f} GiB "
        f"({kept_frac:.1%} of logical bytes kept)"
    )
    if not args.apply:
        print("dry run — pass --apply to strip")
        return

    decomposition = restore_decomposition_subtrees(manager, step, metadata)
    staging_dir = args.run_dir / "ckpts_strip_staging"
    assert not staging_dir.exists(), (
        f"leftover staging dir from an interrupted strip: {staging_dir}"
    )
    staging_manager = open_manager(staging_dir)
    staging_manager.save(
        step, args=ocp.args.Composite(decomposition=ocp.args.StandardSave(decomposition))
    )
    staging_manager.wait_until_finished()

    staged_metadata = read_tree_metadata(staging_dir / str(step) / "decomposition")
    assert set(staged_metadata.keys()) == keep, staged_metadata.keys()
    for key in DECOMPOSITION_KEYS:
        assert tree_shapes(staged_metadata[key]) == tree_shapes(metadata[key]), (
            f"staged {key} tree does not match the original"
        )

    shutil.rmtree(ckpts_dir / str(step))
    (staging_dir / str(step)).rename(ckpts_dir / str(step))
    shutil.rmtree(staging_dir)
    print(f"stripped: {on_disk / 2**30:.1f} -> {du_bytes(ckpts_dir / str(step)) / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
