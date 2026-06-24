"""Device-count invariance of the generic trainer (SPEC D4), on the tiny Llama target.

Runs the SAME fixed global batch + seed through the full step twice on this host:
once single-layout (mesh=None — everything on device 0), once GSPMD batch-sharded over
ALL visible devices — and asserts the per-step metric trajectories match up to
floating-point reassociation (rel ≤ 1e-4; cross-shard reduction order differs, so
bit-exactness is not achievable for the batch-reduced terms). That is the
SPMD-correctness contract: sharding layout must be semantically invisible.

Simulated multi-device CPU run:

  XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    python -m param_decomp.experiments.invariance_check --steps 3

(JAX's counter-based RNG is value-deterministic for a fixed key regardless of
sharding, so the stochastic terms draw identical values — only summation order
differs across layouts.)
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random

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
from param_decomp.recon import build_loss_spec
from param_decomp.schedule import ScheduleConfig
from param_decomp.sharding import dp_mesh, shard_batch
from param_decomp.targets.llama8b import (
    llama_site_specs,
    mlp_family_site_cs,
)
from param_decomp.tests.test_llama8b import _tiny_cfg, _tiny_decomposed_lm
from param_decomp.train import TrainState, make_train_step


def _run(steps: int, sharded: bool) -> list[dict[str, float]]:
    cfg = _tiny_cfg()
    C, seq, gbatch = 8, 16, 8
    sites = llama_site_specs(cfg, mlp_family_site_cs(3, 6, C))
    lm = _tiny_decomposed_lm(cfg, sites, random.PRNGKey(0))
    vu = init_decomp_vu(sites, random.PRNGKey(1))
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=("resid.3",), output_sites=lm.site_names),),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        mlp_hidden=32,
    )
    ci_fn = build_ci_fn(arch, lm.sites, random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    src = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), (1, seq), random.PRNGKey(3)
    )
    resid = random.normal(random.PRNGKey(4), (gbatch, seq, cfg.n_embd)) * 0.5

    mesh = dp_mesh() if sharded else None
    if mesh is not None:
        resid = shard_batch(resid, mesh, batch_axis=0)

    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources={"PersistentPGDReconLoss": src},
        sources_opt_state={"PersistentPGDReconLoss": init_sources_adam_state(src)},
        step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    loss_spec = build_loss_spec(
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
    )  # fmt: skip
    step = make_train_step(
        lm=lm,
        loss_spec=loss_spec,
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=True, mesh=mesh,
    )  # fmt: skip

    out = []
    for i in range(steps):
        state, m = step(lm, state, resid, random.PRNGKey(100 + i))
        out.append({k: float(v) for k, v in m.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    n_dev = len(jax.devices())
    print(f"devices: {n_dev}")
    single = _run(args.steps, sharded=False)
    sharded = _run(args.steps, sharded=True)

    # ABS floor: the per-leaf grad-norm diagnostics include tiny norms (~1e-4) whose
    # cross-shard reduction suffers cancellation — relative error there can graze 1e-4
    # while every loss term sits at ~1e-6 (observed: one ci-fn wv leaf, abs err ~1e-8).
    # The floor is far below any semantically meaningful metric's scale.
    REL, ABS = 1e-4, 1e-7
    ok = True
    worst = 0.0
    for i, (a, b) in enumerate(zip(single, sharded, strict=True)):
        for k in a:
            err = abs(a[k] - b[k])
            worst = max(worst, err / (abs(a[k]) + 1e-30))
            if err > REL * abs(a[k]) + ABS:
                ok = False
                print(f"step {i} {k}: single {a[k]!r} vs sharded({n_dev}) {b[k]!r} err {err:.2e}")
    assert ok, "trajectory diverged across shardings — SPMD correctness broken (SPEC D4)"
    print(
        f"OK: {args.steps}-step trajectory matches 1-layout vs {n_dev}-device GSPMD "
        f"(worst rel {worst:.2e}; tol rel {REL:.0e} + abs {ABS:.0e}; reassociation-only)"
    )


if __name__ == "__main__":
    main()
