"""Forward-only loader for p-594db290: partial-restore ONLY `components` (V/U) + `ci_fn`
from the orbax checkpoint, skipping the optimizer states and PersistentPGD adversary
(sources + Adam moments) that a forward pass never touches. Fits on one GPU.

Unlike the p-1e7e8e36 loader this needs no layout remap: the checkpoint's pytree
(per-site `out_ws`/`out_bs` tuples) matches this branch's `CIFn` exactly, so the
templates from the init fns restore directly.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax import random
from jaxtyping import Array, Int

from param_decomp.ci_fn import CIFn
from param_decomp.components import DecompVU
from param_decomp.lm import DecomposedModel
from param_decomp.sharding import hsdp_mesh
from param_decomp.targets.llama8b_sharding import init_ci_fn_placed, init_decomp_vu_placed
from param_decomp.train import COMPUTE_DT, TrainState, cast_floating
from param_decomp_lab.experiments.lm.config import load_run_dir_config
from param_decomp_lab.experiments.lm.load_run import (
    HarvestForward,
    LoadedJaxRun,
    _u_norms,
    build_target,
)


def open_forward_only(run_dir: Path, step: int) -> LoadedJaxRun:
    cfg = load_run_dir_config(run_dir)
    mesh = hsdp_mesh()
    lm, vocab_size = build_target(cfg, mesh)

    init_key = random.PRNGKey(cfg.pd.seed)
    ci_key = random.fold_in(init_key, 1)
    components_tmpl = init_decomp_vu_placed(lm.sites, init_key, mesh)
    ci_fn_tmpl = init_ci_fn_placed(cfg.ci_fn, lm.sites, ci_key, mesh)

    repl = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    to_sds = lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=repl)
    item = {
        "components": jax.tree.map(to_sds, components_tmpl),
        "ci_fn": jax.tree.map(to_sds, ci_fn_tmpl),
    }
    restore_args = jax.tree.map(
        lambda sds: ocp.ArrayRestoreArgs(restore_type=jax.Array, sharding=repl, shape=sds.shape),
        item,
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )
    handler = ocp.PyTreeCheckpointHandler()
    ckpt_path = (run_dir / "ckpts" / str(step) / "default").resolve()
    restored = handler.restore(
        ckpt_path,
        args=ocp.args.PyTreeRestore(item=item, restore_args=restore_args, partial_restore=True),
    )
    components: DecompVU = restored["components"]
    ci_fn: CIFn = restored["ci_fn"]

    site_names = lm.site_names
    u_norms = _u_norms(components, site_names)

    @eqx.filter_jit
    def forward(
        model: DecomposedModel, components: DecompVU, ci_fn: CIFn, token_ids: Int[Array, "B T"]
    ):
        clean_output = model.clean_output(token_ids)
        taps = model.read_activations(token_ids, ci_fn.input_names)
        site_inputs = model.read_activations(token_ids, site_names)
        components_bf16 = cast_floating(components, COMPUTE_DT)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        lower_ci = ci_fn_bf16(taps, remat=False).lower
        component_acts = {}
        for site in site_names:
            V = components_bf16.site(site)[0]
            acts = site_inputs[site].astype(COMPUTE_DT) @ V
            component_acts[site] = acts.astype(jnp.float32) * u_norms[site]
        return (
            {site: lower_ci[site].astype(jnp.float32) for site in site_names},
            component_acts,
            jax.nn.softmax(clean_output.astype(jnp.float32), axis=-1),
        )

    state = TrainState(
        components=components,
        ci_fn=ci_fn,
        components_opt_state=None,
        ci_fn_opt_state=None,
        adversaries={},
        step=jnp.zeros((), jnp.int32),
    )
    return LoadedJaxRun(
        run_id=run_dir.name,
        step=step,
        lm=lm,
        config=cfg,
        vocab_size=vocab_size,
        _state=state,
        _forward=forward,
    )


__all__ = ["open_forward_only", "HarvestForward"]
