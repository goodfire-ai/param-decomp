"""One-off migration: the frozen C49k clone's orbax checkpoint -> the current trainer's
`TrainState` layout, so a fine-tune can `restore_latest` from it.

The C49k run (`jax-l18-C49k-200k`, Llama-3.1-8B layer-18 MLP, C=49152, written
~2026-06-15 by a since-superseded frozen clone) saved a `TrainState` whose pytree keys
differ from HEAD's in three places. This script restores the OLD tree leaf-by-leaf into
host arrays (single-device CPU), rebuilds it under the CURRENT key structure, and
`save_state`s it so `restore_latest` finds it unchanged.

The remap (OLD leaf -> NEW leaf), confirmed against orbax metadata of both trees:

  components            OLD components.{Vg,Ug, Vu,Uu, Vd,Ud}  shape (1, *, *)
                        NEW components.vu[<site>][0]=V, [1]=U  shape (*, *)
                        g->gate_proj, u->up_proj, d->down_proj; V->[0], U->[1];
                        the OLD leading singleton axis is SQUEEZED (the clone stored a
                        legacy (1, d_in, C) / (1, C, d_out); HEAD's DecompVU is 2-D).

  components_opt_state  OLD [1][0].{mu,nu}.{Vg..Ud}  ->  NEW [1][0].{mu,nu}.vu[<site>][0|1]
                        same squeeze; the two optax `count` scalars ([1][0], [1][2])
                        copy through unchanged.

  ci_fn / ci_fn_opt_state   IDENTICAL leaf names old and new -> straight copy.

  adversaries
                        OLD sources.<short_site>                  (no state-key level)
                        NEW adversaries.<state_key>.sources.<short_site>   state_key =
                        the PersistentPGD term's instance key ("PersistentPGDReconLoss").
                        OLD sources_adam_state.{m,v}.<site> + .step_count
                        NEW adversaries.<state_key>.opt_state.{m,v}.<site> + .step_count.
                        Site names already match (short `layers.18.mlp.*_proj`).

  step                  scalar, copied through (restores as 175000).

The run's pinned `config.yaml` predates the `run_id` convention and declares the frozen
target as fp32 (the clone ran bf16 anyway); the migration mints a fresh `run_id` and
stamps `weights_dtype: bfloat16` into the destination's config copies so the current
trainer's loader assertions pass on resume. The frozen target is NOT in the checkpoint
(rebuilt from HF on resume), so no 8B weights are touched here.
"""

import argparse
import secrets
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import yaml
from etils import epath
from jax.sharding import SingleDeviceSharding
from jaxtyping import Array

from param_decomp.checkpoint import make_checkpoint_manager, restore_latest, save_state
from param_decomp.experiments.llama8b_real import _random_decomposed_lm
from param_decomp.run_state import build_optimizers, init_train_state
from param_decomp.sharding import dp_mesh
from param_decomp.targets.llama8b import llama31_8b_config, llama_site_specs
from param_decomp.train import COMPUTE_DT, TrainState, cast_floating
from param_decomp_lab.experiments.lm.config import build_from_schema

KIND_TO_SITE_SUFFIX = {"g": "gate_proj", "u": "up_proj", "d": "down_proj"}
SOURCE_STATE_KEY = "PersistentPGDReconLoss"


def _restore_old_tree(src_ckpt: Path) -> dict[str, Any]:
    """Restore the OLD checkpoint onto an abstract pytree mirroring its own metadata,
    every leaf placed single-device on the one CPU device. Returns the OLD-layout pytree."""
    cpus = jax.devices("cpu")
    assert cpus, "no CPU device"
    sharding = SingleDeviceSharding(cpus[0])
    handler = ocp.StandardCheckpointHandler()
    meta_tree = handler.metadata(epath.Path(src_ckpt / "default")).tree

    def abstract(leaf: Any) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)

    reference = jax.tree.map(abstract, meta_tree)
    with ocp.StandardCheckpointer() as ckptr:
        restored = ckptr.restore(epath.Path(src_ckpt / "default"), target=reference)
    assert isinstance(restored, dict)
    return restored


def _old_decomp_vu_to_new(old_vu: dict[str, Array]) -> dict[str, Array]:
    """OLD flat {Vg, Ug, Vu, Uu, Vd, Ud} (each `(1, *, *)`) -> values keyed by the
    NEW keystr suffix `['<site>'][0|1]` (V->0, U->1), the leading singleton SQUEEZED."""
    out: dict[str, Array] = {}
    for kind, suffix in KIND_TO_SITE_SUFFIX.items():
        site = f"layers.18.mlp.{suffix}"
        out[f"['{site}'][0]"] = jnp.squeeze(old_vu[f"V{kind}"], axis=0)
        out[f"['{site}'][1]"] = jnp.squeeze(old_vu[f"U{kind}"], axis=0)
    return out


def _flatten_old(old: dict[str, Any]) -> dict[str, Array]:
    """Flatten the OLD-layout restored pytree to `{NEW keystr -> array}` — the value each
    leaf of the current reference `TrainState` (addressed by `jax.tree_util.keystr`) must
    take. Encodes the whole remap; every reference leaf must be hit exactly once."""
    out: dict[str, Array] = {}

    for suffix, value in _old_decomp_vu_to_new(old["components"]).items():
        out[f".components.vu{suffix}"] = value

    co = old["components_opt_state"]
    out[".components_opt_state[1][0].count"] = co[1][0]["count"]
    out[".components_opt_state[1][2].count"] = co[1][2]["count"]
    for moment in ("mu", "nu"):
        for suffix, value in _old_decomp_vu_to_new(co[1][0][moment]).items():
            out[f".components_opt_state[1][0].{moment}.vu{suffix}"] = value

    def add_ci_fn(prefix: str, ci: dict[str, Any]) -> None:
        out[f"{prefix}.in_proj_w"] = ci["in_proj_w"]
        out[f"{prefix}.in_proj_b"] = ci["in_proj_b"]
        out[f"{prefix}.out_w"] = ci["out_w"]
        out[f"{prefix}.out_b"] = ci["out_b"]
        out[f"{prefix}.inv_freq"] = ci["inv_freq"]
        for i, block in enumerate(ci["blocks"]):
            for field in ("wq", "wk", "wv", "wo", "w1", "b1", "w2", "b2"):
                out[f"{prefix}.blocks[{i}].{field}"] = block[field]

    add_ci_fn(".ci_fn", old["ci_fn"])
    cio = old["ci_fn_opt_state"]
    out[".ci_fn_opt_state[0].count"] = cio[0]["count"]
    out[".ci_fn_opt_state[2].count"] = cio[2]["count"]
    add_ci_fn(".ci_fn_opt_state[0].mu", cio[0]["mu"])
    add_ci_fn(".ci_fn_opt_state[0].nu", cio[0]["nu"])

    adam = old["sources_adam_state"]
    adv = f".adversaries['{SOURCE_STATE_KEY}']"
    for site, src in old["sources"].items():
        out[f"{adv}.sources['{site}']"] = src
    for moment in ("m", "v"):
        for site, value in adam[moment].items():
            out[f"{adv}.opt_state.{moment}['{site}']"] = value
    out[f"{adv}.opt_state.step_count"] = adam["step_count"]

    out[".step"] = old["step"]
    return out


def _build_new_state(old: dict[str, Any], reference: TrainState) -> TrainState:
    """Place every OLD leaf onto the current reference `TrainState` by keystr, asserting
    a 1:1 shape/dtype match — the structure is the reference's, the values are the OLD's."""
    new_by_keystr = _flatten_old(old)
    ref_leaves = jax.tree_util.tree_flatten_with_path(reference)[0]
    assert len(ref_leaves) == len(new_by_keystr), (
        len(ref_leaves),
        len(new_by_keystr),
        sorted(set(new_by_keystr) - {jax.tree_util.keystr(p) for p, _ in ref_leaves}),
    )

    def place(path: Any, ref_leaf: Array) -> Array:
        keystr = jax.tree_util.keystr(path)
        assert keystr in new_by_keystr, f"no OLD value for reference leaf {keystr}"
        value = new_by_keystr[keystr]
        assert value.shape == ref_leaf.shape and value.dtype == ref_leaf.dtype, (
            keystr,
            (value.shape, value.dtype),
            (ref_leaf.shape, ref_leaf.dtype),
        )
        return value

    return jax.tree_util.tree_map_with_path(place, reference)


def _build_reference(
    run_dir: Path, run_id: str, mesh: Any, *, abstract: bool
) -> tuple[TrainState, Any, Any]:
    """`(reference TrainState, cfg, lm)`. `abstract=True` returns the reference as a tree
    of `ShapeDtypeStruct`s via `jax.eval_shape` (zero host memory) — used as the remap
    target template and the orbax restore/save reference, so the full fp32 reference never
    coexists with the 47 GB restored tree under the session's ~78 GB cgroup cap."""
    schema_raw = yaml.safe_load((run_dir / "experiment_config.yaml").read_text())
    schema_raw["target"]["weights_dtype"] = "bfloat16"
    schema_raw["run_name"] = run_dir.name
    schema_raw.setdefault("runtime", {})["remat_recon_forwards"] = False
    cfg = build_from_schema(schema_raw, run_id)
    llama_cfg = llama31_8b_config()
    sites = llama_site_specs(llama_cfg, cfg.target.sites)
    # The migration only needs `lm`'s STATIC config (sites / leading_axes) to build the
    # reference TrainState — the frozen weights never enter; build them abstractly (no alloc).
    lm = eqx.filter_eval_shape(_random_decomposed_lm, llama_cfg, sites, jax.random.PRNGKey(0))
    opt_vu, opt_ci, _ = build_optimizers(cfg.pd)
    init_key, src_key = jax.random.split(jax.random.PRNGKey(cfg.pd.seed))
    build = lambda: init_train_state(
        cfg.pd, lm, cfg.ci_fn, cfg.data, opt_vu, opt_ci, init_key, src_key, mesh
    )
    reference = jax.eval_shape(build) if abstract else build()
    return reference, cfg, lm


def _write_dst_config(src_run_dir: Path, dst_run_dir: Path, run_id: str) -> None:
    """Write the destination run dir's single self-contained `config.yaml`: the source
    run's old schema yaml (`experiment_config.yaml`) with `weights_dtype: bfloat16` and
    the run-instance fields (run_name, minted run_id, destination out_dir) stamped in, so
    the current trainer's single-file loader assertions pass."""
    schema = yaml.safe_load((src_run_dir / "experiment_config.yaml").read_text())
    schema["target"]["weights_dtype"] = "bfloat16"
    schema["run_name"] = src_run_dir.name
    schema["run_id"] = run_id
    schema["out_dir"] = str(dst_run_dir.parent)
    (dst_run_dir / "config.yaml").write_text(yaml.safe_dump(schema, sort_keys=False))


def _verify(dst_run_dir: Path, reference: TrainState, lm: Any, cfg: Any) -> None:
    manager = make_checkpoint_manager(dst_run_dir / "ckpts", cfg.cadence.keep_last)
    restored = restore_latest(manager, reference)
    assert restored is not None, "no checkpoint found after migration"
    state, step = restored
    print(f"  restore_latest step = {step}")
    assert int(state.step) == 175000, f"step is {int(state.step)}, expected 175000"
    assert step == 175000, f"resolved step is {step}, expected 175000"

    for site in lm.site_names:
        V, U = state.components.site(site)
        spec = next(s for s in lm.sites if s.name == site)
        assert V.shape == (spec.d_in, spec.C), (site, V.shape)
        assert U.shape == (spec.C, spec.d_out), (site, U.shape)
        assert jnp.all(jnp.isfinite(V)) and jnp.all(jnp.isfinite(U)), f"{site}: non-finite V/U"
    print(f"  components shapes + finiteness OK for {list(lm.site_names)}")

    src = state.adversaries[SOURCE_STATE_KEY].sources["layers.18.mlp.gate_proj"]
    assert jnp.all((src >= 0.0) & (src <= 1.0)), "source values out of [0,1]"
    print(f"  sources in [0,1]; gate source shape {src.shape}")

    b, t = 1, 8
    dummy = {s.name: jnp.zeros((b, t, s.d_in), COMPUTE_DT) for s in lm.sites}
    ci = cast_floating(state.ci_fn, COMPUTE_DT)(dummy)
    for site in lm.site_names:
        assert jnp.all(jnp.isfinite(ci.lower[site])), f"{site}: non-finite CI"
    print("  CI fn forward on dummy batch finite for all sites")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_src = Path(
        "/mnt/data/artifacts/mechanisms/param-decomp/jax_runs/"
        "jax-l18-C49k-200k/preserved_ckpts/175000"
    )
    parser.add_argument(
        "--src", type=Path, default=default_src, help="OLD orbax ckpt dir (step dir)"
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="NEW run dir; checkpoint written to <dst>/ckpts/175000",
    )
    parser.add_argument("--run-id", type=str, default=None, help="mint p-<8hex> if omitted")
    args = parser.parse_args()

    src_ckpt: Path = args.src.resolve()
    assert src_ckpt.exists(), f"src not found: {src_ckpt}"
    src_run_dir = src_ckpt.parent.parent
    step = int(src_ckpt.name)
    assert step == 175000, f"this migration targets step 175000, got {step}"

    dst_run_dir: Path = args.dst.resolve()
    run_id = args.run_id or f"p-{secrets.token_hex(4)}"
    print(f"src       : {src_ckpt}")
    print(f"dst       : {dst_run_dir}")
    print(f"run_id    : {run_id}")

    mesh = dp_mesh()
    assert mesh.devices.size == 1, f"run on a single device (got {mesh.devices.size})"

    print("building abstract current reference (eval_shape, no host alloc) ...", flush=True)
    abstract_reference, cfg, lm = _build_reference(src_run_dir, run_id, mesh, abstract=True)

    print("restoring OLD checkpoint (single-device CPU) ...", flush=True)
    old = _restore_old_tree(src_ckpt)

    print("remapping to current layout ...", flush=True)
    new_state = _build_new_state(old, abstract_reference)
    del old

    dst_run_dir.mkdir(parents=True, exist_ok=True)
    _write_dst_config(src_run_dir, dst_run_dir, run_id)

    print(f"saving migrated checkpoint to {dst_run_dir / 'ckpts' / str(step)} ...", flush=True)
    manager = make_checkpoint_manager(dst_run_dir / "ckpts", cfg.cadence.keep_last)
    save_state(manager, step, new_state)
    del new_state

    print("verifying restore onto a fresh abstract reference ...", flush=True)
    fresh_reference, _, _ = _build_reference(src_run_dir, run_id, mesh, abstract=True)
    _verify(dst_run_dir, fresh_reference, lm, cfg)
    print("MIGRATION OK", flush=True)


if __name__ == "__main__":
    main()
