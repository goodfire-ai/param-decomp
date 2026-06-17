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

  srun --ntasks-per-node=8 ... python -m jax_single_pool.experiments.mem_probe \
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

from jax_single_pool.adversary import init_persistent_sources, init_sources_adam_state
from jax_single_pool.ci_fn import CIArch, init_ci_fn
from jax_single_pool.experiments.llama8b_real import (
    _random_target,  # pyright: ignore[reportPrivateUsage]
)
from jax_single_pool.llama8b import (
    init_decomp_vu,
    llama31_8b_config,
    llama_decomposed_lm,
    llama_site_specs,
    mlp_family_site_cs,
)
from jax_single_pool.llama8b_sharding import dp_mesh
from jax_single_pool.recon import build_recon_terms
from jax_single_pool.sharding import init_distributed
from jax_single_pool.train import TrainState, make_train_step
from param_decomp_config.losses import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
)
from param_decomp_config.routing import UniformKSubsetRoutingConfig
from param_decomp_config.schedule import ScheduleConfig


def _place_state(typed: TrainState, mesh: Mesh) -> TrainState:
    """Attach the trainer's GSPMD shardings to the typed abstract `TrainState`,
    re-deriving the `llama8b_sharding.py` placement per leaf from its field group +
    shape (so the Adam mu/nu mirror their params):

      components V `(d_in, C)` -> P(None, 'dp'); U `(C, d_out)` -> P('dp', None)
      ci_fn 2-D weight with `dp`-divisible last axis -> P(None, 'dp'); else replicate
      sources (SCScope), all scalars/1-D, opt counts -> replicate
    """
    n = mesh.devices.size
    repl = NamedSharding(mesh, P())
    model_dims = {4096, 14336}  # Llama-8B d_model / d_mlp; the C axis is neither

    def place(path: tuple[Any, ...], leaf: Any) -> Any:
        group = str(getattr(path[0], "key", getattr(path[0], "name", getattr(path[0], "idx", ""))))
        if leaf.ndim == 2 and group.startswith("components"):
            # V `(d_in, C)` shards axis 1, U `(C, d_out)` shards axis 0; the C axis is
            # the one NOT a model dim (V/U Adam mu/nu share their param's layout).
            c_axis = 0 if leaf.shape[0] not in model_dims else 1
            spec = ["dp" if ax == c_axis else None for ax in range(2)]
            sharding = NamedSharding(mesh, P(*spec))
        elif leaf.ndim == 2 and group.startswith("ci_fn") and leaf.shape[-1] % n == 0:
            sharding = NamedSharding(mesh, P(None, "dp"))
        else:
            sharding = repl
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)

    return jax.tree_util.tree_map_with_path(
        place, typed, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct)
    )


def _typed_state_struct(
    lm: Any, opt_vu: Any, opt_ci: Any, ci_arch: CIArch, src_leading: tuple[int, int], key: Any
) -> TrainState:
    """The `TrainState` pytree STRUCTURE (typed, sharding-less) via the UNSHARDED init
    constructors under `eval_shape` — allocates nothing. `_place_state` then attaches the
    GSPMD shardings per leaf, reproducing the trainer's sharded init abstractly."""
    components = init_decomp_vu(lm.sites, key)
    ci_fn = init_ci_fn(ci_arch, lm.sites, random.fold_in(key, 1))
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
    args = ap.parse_args()
    init_distributed()
    mesh = dp_mesh()
    ndev = mesh.devices.size
    is0 = jax.process_index() == 0

    cfg = llama31_8b_config()
    sites = llama_site_specs(cfg, mlp_family_site_cs(args.first_layer, args.last_layer, args.C))
    seq = 2048
    gbatch = args.per_gpu_batch * ndev
    lm = llama_decomposed_lm(cfg, sites)

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
    loss_spec = build_recon_terms(
        loss_metrics, lm.site_names, n_mask_samples=1, sampling="continuous"
    )

    ci_arch = CIArch(4096, 4, 64, 16384)
    key_repl = jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=NamedSharding(mesh, P()))

    typed = jax.eval_shape(
        lambda k: _typed_state_struct(lm, opt_vu, opt_ci, ci_arch, (1, seq), k), key_repl
    )
    state_in = _place_state(typed, mesh)

    target_shapes = (
        jax.jit(_random_target, static_argnums=(0, 1))
        .lower(cfg, args.first_layer, key_repl)
        .out_info
    )
    target_in = _abstract_replicated(target_shapes, mesh)
    resid_in = jax.ShapeDtypeStruct(
        (gbatch, seq, cfg.n_embd), jnp.bfloat16, sharding=NamedSharding(mesh, P("dp"))
    )
    key_in = jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=NamedSharding(mesh, P()))

    step_fn = make_train_step(
        lm=lm,
        loss_spec=loss_spec,
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=not args.no_remat, mesh=mesh,
    )  # fmt: skip

    compiled = step_fn.lower(state_in, target_in, resid_in, key_in).compile()
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
