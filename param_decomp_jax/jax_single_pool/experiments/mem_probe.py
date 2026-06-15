"""AOT memory probe for the train step — no data, no HF weights, no execution.

Compiles `jit_step` at the smoke topology (8 GPU, B=32, L18, C=24576) and prints the
per-device memory analysis; with `--xla_dump_to` set, XLA writes the
buffer-assignment table that names the largest allocations (debugging the smoke-v4
107 GiB OOM). Run under SLURM like the trainer:

  XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_as_text" srun ... mem_probe.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jax_single_pool.adversary import init_sources_adam_state
from jax_single_pool.ci_fn import CIArch

# the smoke's _random_target lives in the runner
from jax_single_pool.experiments.llama8b_real import (
    _random_target,  # pyright: ignore[reportPrivateUsage]
)
from jax_single_pool.llama8b import (
    llama31_8b_config,
    llama_decomposed_lm,
    llama_site_specs,
    mlp_family_site_cs,
)
from jax_single_pool.llama8b_sharding import (
    dp_mesh,
    init_ci_fn_sharded,
    init_decomp_vu_sharded,
    init_sources_sharded,
    replicate_target,
)
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


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--per_gpu_batch", type=int, default=4)
    ap.add_argument("--no_remat", action="store_true",
                    help="disable the recon-forward rematerialization (memory A/B)")  # fmt: skip
    ap.add_argument("--first_layer", type=int, default=18)
    ap.add_argument("--last_layer", type=int, default=18)
    ap.add_argument("--C", type=int, default=24576)
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

    target = replicate_target(_random_target(cfg, args.first_layer, random.PRNGKey(0)), mesh)
    vu = init_decomp_vu_sharded(lm.sites, random.PRNGKey(1), mesh)
    ci_fn = init_ci_fn_sharded(CIArch(4096, 4, 64, 16384), lm.sites, random.PRNGKey(2), mesh)
    src = init_sources_sharded(
        lm.site_names, tuple(s.C for s in lm.sites), seq, random.PRNGKey(3), mesh
    )
    opt_vu = optax.chain(
        optax.clip_by_global_norm(0.01),
        optax.adamw(1.5e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0),
    )
    opt_ci = optax.adamw(5e-5, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources={"PersistentPGDReconLoss": src},
        sources_opt_state={"PersistentPGDReconLoss": init_sources_adam_state(src)},
        step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    loss_spec = build_recon_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6, pnorm=2.0, beta=0.2,
                p_anneal_start_frac=0.0, p_anneal_final_p=0.4, p_anneal_end_frac=1.0,
            ),
            ChunkwiseSubsetReconLossConfig(routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=3, n_samples=1),
            PersistentPGDReconLossConfig(
                coeff=0.5,
                scope=SCScope(),
                optimizer=AdamPGDConfig(
                    beta1=0.5, beta2=0.99,
                    lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
                ),
                n_warmup_steps=2,
            ),
        ),
        lm.site_names,
        n_mask_samples=1,
        sampling="continuous",
    )  # fmt: skip
    step_fn = make_train_step(
        lm=lm,
        loss_spec=loss_spec,
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=not args.no_remat, mesh=mesh,
    )  # fmt: skip

    resid = jax.device_put(
        jnp.zeros((gbatch, seq, cfg.n_embd), jnp.bfloat16), NamedSharding(mesh, P("dp"))
    )
    lowered = step_fn.lower(state, target, resid, random.PRNGKey(7))
    compiled = lowered.compile()
    if is0:
        ma = compiled.memory_analysis()
        assert ma is not None, "backend returned no memory analysis"
        gib = 1024**3
        print(
            f"[mem] ndev={ndev} bl={args.per_gpu_batch} remat={not args.no_remat} | "
            f"temp {ma.temp_size_in_bytes / gib:.1f} GiB | "
            f"args {ma.argument_size_in_bytes / gib:.1f} GiB | "
            f"out {ma.output_size_in_bytes / gib:.1f} GiB | "
            f"alias {ma.alias_size_in_bytes / gib:.1f} GiB",
            flush=True,
        )
        print("[mem] PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
