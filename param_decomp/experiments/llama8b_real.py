"""Run the full Llama-3.1-8B single-pool VPD step (generic trainer) — measure tok/s/GPU.

NB: no MFU/FLOP estimate here on purpose — a hardcoded forward-count × an "achievable"
peak gave a misleadingly high MFU. tok/s/GPU (tokens ÷ wall-clock) is the trustworthy,
assumption-free throughput number; compare that directly across frameworks.

The full SPEC-compliant step on the REAL 8B model: residual-start suffix from
`--first_layer`, MLP (gate/up/down) decomposed on layers `[first_layer, last_layer]`
(3N sites), weight-delta, shared-transformer CI fn, 4 losses + persistent-PGD Adam
adversary, fp32 masters + bf16 compute, p-anneal + LR schedules, faithfulness warmup.
GSPMD-sharded: frozen suffix replicated, V/U + CI + Adam sharded over `dp`, batch
sharded over `dp`, PGD source replicated (grad reduced by the global-mean loss).

Schedules anneal over `--total_steps` (the production horizon), not the benched
`--steps`, so short benches measure start-of-training semantics.

Usage (single B200, random weights, fast smoke, 12 layers):
  python -m param_decomp.experiments.llama8b_real --per_gpu_batch 1 --steps 6 \
      --C 2048 --faith_warmup 0

Real HF weights + the residual-start prefix harvest:
  python -m param_decomp.experiments.llama8b_real --real_weights --first_layer 20 \
      --last_layer 31 --C 8192 --per_gpu_batch 1 --steps 6 --shard

Multi-GPU under SLURM (1 task/GPU): pass `--dp <world_size>` so init_distributed brings
up the mesh; omit it for a single-device run.
"""

import argparse
import time

import equinox as eqx
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    build_ci_fn,
)
from param_decomp.components import SiteSpec, init_decomp_vu
from param_decomp.configs import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
    UniformKSubsetRoutingConfig,
)
from param_decomp.recon import build_loss_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.sharding import init_distributed
from param_decomp.targets.llama8b import (
    DT,
    FrozenAttn,
    LlamaDecomposedModel,
    SuffixLayer,
    build_decomposed_lm,
    first_decomposed_layer,
    llama31_8b_config,
    llama_site_specs,
    load_decomposed_lm_from_hf,
    make_real_target_residual,
    mlp_family_site_cs,
)
from param_decomp.targets.llama8b_sharding import (
    dp_mesh,
    init_ci_fn_placed,
    init_decomp_vu_placed,
    init_sources_sharded,
    place_target,
    shard_batch,
)
from param_decomp.train import TrainState, make_faith_warmup_step, make_train_step
from vendored_jax.llama import LlamaConfig, llama3_inv_freq


def _random_decomposed_lm(
    cfg: LlamaConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> LlamaDecomposedModel:
    """A random `LlamaDecomposedModel` (random frozen suffix + the decomposition `sites`) —
    the no-HF-weights bench analog of `load_decomposed_lm_from_hf`."""
    first_layer = first_decomposed_layer(tuple(s.name for s in sites))
    ks = iter(random.split(key, 4096))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def n(shape: tuple[int, ...], s: float | None = None) -> jax.Array:
        return (random.normal(next(ks), shape) * (s or d**-0.5)).astype(DT)

    def fattn():
        return FrozenAttn(
            n((qd, d)), n((kvd, d)), n((kvd, d)), n((d, qd)),
            cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_rep,
        )  # fmt: skip

    def suffix_layer():
        return SuffixLayer(
            ln1=jnp.ones((d,), DT), ln2=jnp.ones((d,), DT), attn=fattn(),
            Wg=n((di, d)), Wu=n((di, d)), Wd=n((d, di)),
        )  # fmt: skip

    return build_decomposed_lm(
        layers=[suffix_layer() for _ in range(cfg.n_layer - first_layer)],
        norm=jnp.ones((d,), DT),
        lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=llama3_inv_freq(cfg),
        cfg=cfg,
        sites=sites,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_gpu_batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--C", type=int, default=8192)
    ap.add_argument("--first_layer", type=int, default=20)
    ap.add_argument("--last_layer", type=int, default=31)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n_warmup", type=int, default=2)
    ap.add_argument("--total_steps", type=int, default=100_000,
                    help="schedule horizon (p anneal, LR decay) — production 100k")  # fmt: skip
    ap.add_argument("--faith_warmup", type=int, default=400,
                    help="faithfulness-warmup steps before the bench loop (SPEC S21); 0 to skip")  # fmt: skip
    ap.add_argument("--real_weights", action="store_true")
    ap.add_argument("--shard", action="store_true", help="jit + C-shard V/U/CI/Adam + batch")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dp", type=int, default=None,
                    help="distributed world size (nodes×8); omit for single-device")  # fmt: skip
    args = ap.parse_args()
    assert args.steps > 0, "--steps must be positive"

    first, last = args.first_layer, args.last_layer
    assert 0 <= first <= last < 32, f"bad layer range {first}..{last}"

    distributed = init_distributed(args.dp)
    mesh = dp_mesh()
    ndev = mesh.devices.size
    is0 = jax.process_index() == 0
    gbatch = args.per_gpu_batch * ndev

    cfg = llama31_8b_config()
    sites = llama_site_specs(cfg, mlp_family_site_cs(first, last, args.C))
    site_names = tuple(s.name for s in sites)
    n_layers = last - first + 1
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first}",), output_sites=site_names),),
        input_dim=cfg.n_embd,
        d_model=4096,
        n_blocks=4,
        n_heads=64,
        mlp_hidden=16384,
    )
    if is0:
        print(
            f"[p0] LLAMA8B single-pool PD | {ndev} GPU | gbatch={gbatch} seq={args.seq} "
            f"layers={first}..{last} ({n_layers}L, {len(sites)} sites) "
            f"C={args.C} n_warmup={args.n_warmup} faith_warmup={args.faith_warmup} "
            f"mode={'shard' if args.shard else 'replicated'} "
            f"weights={'HF' if args.real_weights else 'random'}"
        )

    idx_global = random.randint(random.PRNGKey(42), (gbatch, args.seq), 0, cfg.vocab_size)
    if args.real_weights:
        if is0:
            print("[p0] loading HF suffix + harvesting residual via prefix forward...")
        lm = load_decomposed_lm_from_hf(args.model_name, cfg, sites)
        resid_global = make_real_target_residual(
            args.model_name, cfg, first, idx_global, chunk=args.per_gpu_batch
        )
    else:
        lm = _random_decomposed_lm(cfg, sites, random.PRNGKey(0))
        resid_global = (
            random.normal(random.PRNGKey(7), (gbatch, args.seq, cfg.n_embd)) * 0.5
        ).astype(DT)

    # Production optimizers (SPEC S19/S20): AdamW wd=0, cosine→0.1×; V/U clipped at 0.01.
    sched_vu = optax.cosine_decay_schedule(1.5e-4, args.total_steps, alpha=0.1)
    sched_ci = optax.cosine_decay_schedule(5.0e-5, args.total_steps, alpha=0.1)
    opt_vu = optax.chain(
        optax.clip_by_global_norm(0.01),
        optax.adamw(sched_vu, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0),
    )
    opt_ci = optax.adamw(sched_ci, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0)

    lm = place_target(lm, mesh)
    site_Cs = tuple(s.C for s in lm.sites)
    # fp32 masters; source init U[0,1] (SPEC S15), trailing channel = the weight-delta source.
    if args.shard:
        vu = init_decomp_vu_placed(lm.sites, random.PRNGKey(1), mesh)
        ci_fn = init_ci_fn_placed(arch, lm.sites, random.PRNGKey(2), mesh)
        src = init_sources_sharded(
            lm.site_names, site_Cs, args.seq, SCScope(), 1, random.PRNGKey(3), mesh
        )
    else:
        repl = NamedSharding(mesh, P())
        put = lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a  # noqa: E731
        vu = jax.tree.map(put, init_decomp_vu(lm.sites, random.PRNGKey(1)))
        ci_fn = jax.tree.map(put, build_ci_fn(arch, lm.sites, random.PRNGKey(2)))
        src = {
            k: jax.device_put(v, repl)
            for k, v in init_persistent_sources(
                lm.site_names, site_Cs, (1, args.seq), random.PRNGKey(3)
            ).items()
        }
    resid = shard_batch(resid_global, mesh)

    # ── faithfulness warmup (SPEC S21): V/U only, before the main loop ──
    if args.faith_warmup > 0:
        wopt = optax.adamw(1.0e-3, weight_decay=0.0)
        wstate = wopt.init(eqx.filter(vu, eqx.is_array))
        wstep = make_faith_warmup_step(wopt)
        t0 = time.time()
        wloss: jax.Array | None = None
        for _ in range(args.faith_warmup):
            vu, wstate, wloss = wstep(lm, vu, wstate)
        assert wloss is not None
        jax.block_until_ready(wloss)
        if is0:
            print(
                f"[p0] faith warmup: {args.faith_warmup} steps in {time.time() - t0:.1f}s, "
                f"final faith {float(wloss):.3e}"
            )

    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=0.5,
        scope=SCScope(),
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
        ),
        n_warmup_steps=args.n_warmup,
    )
    assert ppgd_cfg.coeff is not None
    state = TrainState(
        components=vu,
        ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        adversaries={
            ppgd_cfg.type: PersistentAdversary(
                sources=src,
                opt_state=init_sources_adam_state(src),
                state_key=ppgd_cfg.type,
                coeff=ppgd_cfg.coeff,
                adam=ppgd_cfg.optimizer,
                start_frac=ppgd_cfg.start_frac,
                n_warmup=ppgd_cfg.n_warmup_steps,
            )
        },
        step=jnp.zeros((), jnp.int32),
    )
    loss_terms = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=2.0,
                beta=0.2,
                p_anneal_start_frac=0.0,
                p_anneal_final_p=0.4,
                p_anneal_end_frac=1.0,
            ),
            ChunkwiseSubsetReconLossConfig(
                routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=3, n_samples=1
            ),
            ppgd_cfg,
        ),
        lm.site_names,
    )
    step = make_train_step(
        lm=lm,
        loss_terms=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=args.total_steps,
        remat_recon_forwards=True,
        mesh=mesh if args.shard else None,
    )

    m: dict[str, jax.Array] = {}
    for _ in range(2):
        state, m = step(lm, state, resid, random.PRNGKey(7))
        jax.block_until_ready((state.adversaries, m["total"]))

    per = []
    for s in range(args.steps):
        t = time.time()
        state, m = step(lm, state, resid, random.PRNGKey(1000 + s))
        jax.block_until_ready((state.adversaries, m["total"]))
        per.append(time.time() - t)
    blocked = sum(per) / len(per)

    t = time.time()
    for s in range(args.steps):
        state, m = step(lm, state, resid, random.PRNGKey(2000 + s))
    dispatch = (time.time() - t) / args.steps
    jax.block_until_ready((state.adversaries, m["total"]))

    peak_gb = max(
        d.memory_stats()["peak_bytes_in_use"] / 1e9
        for d in jax.local_devices()
        if d.memory_stats() is not None
    )

    if is0:
        toks = gbatch * args.seq
        print(
            f"[p0] blocked {blocked * 1e3:.1f} ms/step | dispatch {dispatch * 1e3:.1f} ms/step "
            f"| peak {peak_gb:.1f} GB/dev"
        )
        print(
            f"[p0] {toks / blocked:,.0f} tok/s | {toks / blocked / ndev:,.0f} tok/s/GPU "
            f"| final loss {float(m['total']):.4f}"
        )
        print(
            f"[p0]   losses: faith {float(m['faith']):.4e} imp {float(m['imp']):.4f} "
            f"stoch {float(m['loss/ChunkwiseSubsetReconLoss']):.4e} "
            f"ppgd {float(m['loss/PersistentPGDReconLoss']):.4e} "
            f"(p={float(m['p_imp']):.2f} src_lr={float(m['src_lr']):.2e})"
        )
        print(f"[p0] LLAMA8B ({ndev} GPU, {n_layers}L): OK")

    if distributed:
        jax.experimental.multihost_utils.sync_global_devices("llama8b_done")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
