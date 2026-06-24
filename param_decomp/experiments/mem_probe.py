"""AOT memory probe for the train step — no data, no HF weights, no execution.

Lowers + compiles `make_train_step`'s jitted step at a chosen topology and prints the
per-device memory analysis. The TrainState and frozen target enter the probe as
ABSTRACT sharded avals: shapes come from `jax.jit(init, ...).lower(key).out_info` (the
init is never RUN — only its output shapes are read), and the GSPMD shardings are the
documented placement rules from `llama8b_sharding.py` re-attached per leaf. That keeps
the probe allocation-free: a 27-site C=49152 state is ~360 GB and OOMs any eager init
(the `jit__identity_fn` ~128 GiB blow-up replicating V/U at 64 GPU), but lowering on its
sharded avals costs nothing and yields the partitioned per-device estimate.

The compiled step's per-device peak HBM is `temp + argument + output - alias`.

Run under SLURM like the trainer (one task per GPU); the mesh is all visible devices:

  srun --ntasks-per-node=8 ... python -m param_decomp.experiments.mem_probe \
    --first_layer 18 --last_layer 26 --C 49152 \
    --sites_per_chunk 9 --recon_coeff 1.5 --per_gpu_batch 2

With `--xla_dump_to` set, XLA writes the buffer-assignment table naming the largest
allocations.
"""

import argparse
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.adversary import init_persistent_sources, init_sources_adam_state
from param_decomp.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    build_ci_fn,
)
from param_decomp.components import init_decomp_vu
from param_decomp.configs import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
    UniformKSubsetRoutingConfig,
)
from param_decomp.experiments.llama8b_real import (
    _random_decomposed_lm,
)
from param_decomp.recon import build_loss_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.sharding import init_distributed
from param_decomp.targets.llama8b import (
    llama31_8b_config,
    llama_site_specs,
    mlp_family_site_cs,
)
from param_decomp.targets.llama8b_sharding import dp_mesh
from param_decomp.train import TrainState, make_train_step


def _attach(leaves_tree: Any, shardings_tree: Any) -> Any:
    """Re-attach a same-structure tree of `NamedSharding`s onto a tree of abstract
    `ShapeDtypeStruct` leaves."""
    return jax.tree.map(
        lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
        leaves_tree,
        shardings_tree,
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )


def _place_state(typed: TrainState, mesh: Mesh) -> TrainState:
    """Attach the trainer's GSPMD shardings to the typed abstract `TrainState`, reusing each
    param owner's OWN `.shardings` (so the probe sees the exact production placement, incl.
    the chunkwise CI fn's Megatron layout). The Adam mu/nu mirror their params' shardings;
    every other leaf (sources/SCScope, scalars/1-D, opt counts) replicates."""
    comp_shardings = typed.components.shardings(mesh)
    ci_shardings = typed.ci_fn.shardings(mesh)
    components = _attach(typed.components, comp_shardings)
    ci_fn = _attach(typed.ci_fn, ci_shardings)
    components_opt_state = _place_adam_state(typed.components_opt_state, comp_shardings, mesh)
    ci_fn_opt_state = _place_adam_state(typed.ci_fn_opt_state, ci_shardings, mesh)
    return eqx.tree_at(
        lambda s: (s.components, s.ci_fn, s.components_opt_state, s.ci_fn_opt_state,
                   s.sources, s.sources_opt_state, s.step),
        typed,
        (components, ci_fn, components_opt_state, ci_fn_opt_state,
         _abstract_replicated(typed.sources, mesh),
         _abstract_replicated(typed.sources_opt_state, mesh),
         _abstract_replicated(typed.step, mesh)),
    )  # fmt: skip


def _place_adam_state(opt_state: Any, param_shardings: Any, mesh: Mesh) -> Any:
    """Place an optax AdamW state: the `mu`/`nu` param-shaped subtrees mirror their params'
    shardings; every scalar (the `count`s, the global-norm clip's `EmptyState`) replicates.
    The param-shaped subtrees are exactly the members of `opt_state` whose pytree structure
    equals `param_shardings`."""
    param_treedef = jax.tree.structure(
        param_shardings, is_leaf=lambda x: isinstance(x, NamedSharding)
    )

    def is_param_subtree(member: Any) -> bool:
        return (
            jax.tree.structure(member, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct))
            == param_treedef
        )

    return jax.tree.map(
        lambda m: _attach(m, param_shardings)
        if is_param_subtree(m)
        else _abstract_replicated(m, mesh),
        opt_state,
        is_leaf=is_param_subtree,
    )


def _typed_state_struct(
    lm: Any,
    opt_vu: Any,
    opt_ci: Any,
    ci_arch: ChunkwiseTransformerCIArch,
    src_leading: tuple[int, int],
    key: Any,
) -> TrainState:
    """The `TrainState` pytree STRUCTURE (typed, sharding-less) via the UNSHARDED init
    constructors under `eval_shape` — allocates nothing. `_place_state` then attaches the
    GSPMD shardings per leaf, reproducing the trainer's sharded init abstractly."""
    components = init_decomp_vu(lm.sites, key)
    ci_fn = build_ci_fn(ci_arch, lm.sites, random.fold_in(key, 1))
    sources = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), src_leading, random.fold_in(key, 2)
    )
    return TrainState(
        components=components, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(components, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources={"PersistentPGDReconLoss": sources},
        sources_opt_state={"PersistentPGDReconLoss": init_sources_adam_state(sources)},
        step=jnp.zeros((), jnp.int32),
    )  # fmt: skip


def _abstract_replicated(value: Any, mesh: Mesh) -> Any:
    repl = NamedSharding(mesh, P())
    return jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=repl),
        value,
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_gpu_batch", type=int, default=4)
    ap.add_argument("--no_remat", action="store_true",
                    help="disable the recon-forward rematerialization (memory A/B)")  # fmt: skip
    ap.add_argument("--first_layer", type=int, default=18)
    ap.add_argument("--last_layer", type=int, default=18)
    ap.add_argument("--C", type=int, default=24576)
    ap.add_argument("--sites_per_chunk", type=int, default=3)
    ap.add_argument("--recon_coeff", type=float, default=0.5)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dp", type=int, default=None,
                    help="distributed world size (nodes×8); omit for single-device")  # fmt: skip
    args = ap.parse_args()
    init_distributed(args.dp)
    mesh = dp_mesh()
    ndev = mesh.devices.size
    is0 = jax.process_index() == 0

    cfg = llama31_8b_config()
    sites = llama_site_specs(cfg, mlp_family_site_cs(args.first_layer, args.last_layer, args.C))
    seq = args.seq
    gbatch = args.per_gpu_batch * ndev
    abstract_key = jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=NamedSharding(mesh, P()))
    # `lm` is built abstractly (no allocation) — it carries its STATIC config (sites,
    # first_layer, eps) concretely; the frozen weight leaves stay abstract for the lowering.
    lm = eqx.filter_eval_shape(_random_decomposed_lm, cfg, sites, abstract_key)

    opt_vu = optax.chain(
        optax.clip_by_global_norm(0.01),
        optax.adamw(1.5e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0),
    )
    opt_ci = optax.adamw(5e-5, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)

    loss_metrics = (
        FaithfulnessLossConfig(coeff=1e5),
        ImportanceMinimalityLossConfig(
            coeff=5e-6, pnorm=2.0, beta=0.2,
            p_anneal_start_frac=0.0, p_anneal_final_p=0.4, p_anneal_end_frac=1.0,
        ),
        ChunkwiseSubsetReconLossConfig(
            routing=UniformKSubsetRoutingConfig(), coeff=args.recon_coeff,
            sites_per_chunk=args.sites_per_chunk, n_samples=1,
        ),
        PersistentPGDReconLossConfig(
            coeff=0.5, scope=SCScope(),
            optimizer=AdamPGDConfig(
                beta1=0.5, beta2=0.99,
                lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
            ),
            n_warmup_steps=2,
        ),
    )  # fmt: skip
    loss_terms = build_loss_terms(loss_metrics, lm.site_names)

    ci_arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{args.first_layer}",), output_sites=lm.site_names),),
        input_dim=cfg.n_embd,
        d_model=4096,
        n_blocks=4,
        n_heads=64,
        mlp_hidden=16384,
    )
    key_repl = jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=NamedSharding(mesh, P()))

    typed = jax.eval_shape(
        lambda k: _typed_state_struct(lm, opt_vu, opt_ci, ci_arch, (1, seq), k), key_repl
    )
    state_in = _place_state(typed, mesh)

    # The model IS the frozen target; replicate its abstract weight leaves for the lowering.
    model_in = _abstract_replicated(lm, mesh)
    resid_in = jax.ShapeDtypeStruct(
        (gbatch, seq, cfg.n_embd), jnp.bfloat16, sharding=NamedSharding(mesh, P("dp"))
    )
    key_in = jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=NamedSharding(mesh, P()))

    step_fn = make_train_step(
        lm=lm,
        loss_terms=loss_terms,
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=not args.no_remat, mesh=mesh,
    )  # fmt: skip

    # `eqx.filter_jit.lower` treats a bare `ShapeDtypeStruct` as a STATIC leaf (its `is_array`
    # filter excludes it), so it can't be lowered abstractly the way `jax.jit.lower` can. Wrap
    # the (already-jitted) step in a plain `jax.jit` whose `.lower` converts the abstract avals
    # to tracers — which the inner `filter_jit` then sees as dynamic arrays. The model's static
    # config (`sites`/`first_layer`/`eps`) rides through unchanged on `model_in`.
    lower_wrapper: Any = jax.jit(step_fn)
    compiled = lower_wrapper.lower(model_in, state_in, resid_in, key_in).compile()
    if is0:
        ma = compiled.memory_analysis()
        assert ma is not None, "backend returned no memory analysis"
        gib = 1024**3
        peak = (
            ma.temp_size_in_bytes
            + ma.argument_size_in_bytes
            + ma.output_size_in_bytes
            - ma.alias_size_in_bytes
        )
        print(
            f"[mem] ndev={ndev} bl={args.per_gpu_batch} B={gbatch} "
            f"sites_per_chunk={args.sites_per_chunk} remat={not args.no_remat} | "
            f"temp {ma.temp_size_in_bytes / gib:.1f} GiB | "
            f"args {ma.argument_size_in_bytes / gib:.1f} GiB | "
            f"out {ma.output_size_in_bytes / gib:.1f} GiB | "
            f"alias {ma.alias_size_in_bytes / gib:.1f} GiB | "
            f"PEAK/device {peak / gib:.1f} GiB",
            flush=True,
        )
        print("[mem] PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
