"""Device-count invariance of the generic trainer (SPEC D4), on the tiny Llama target.

Runs the SAME fixed global batch + seed through the full step twice on this host:
once single-layout (mesh=None — everything on device 0), once GSPMD batch-sharded over
ALL visible devices — and asserts the per-step metric trajectories match up to
floating-point reassociation (rel ≤ 1e-4; cross-shard reduction order differs, so
bit-exactness is not achievable for the batch-reduced terms). That is the
SPMD-correctness contract: sharding layout must be semantically invisible.

Simulated multi-device CPU run:

  XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    python -m param_decomp.targets.invariance_check --steps 3

(JAX's counter-based RNG is value-deterministic for a fixed key regardless of
sharding, so the stochastic terms draw identical values — only summation order
differs across layouts.)
"""

import argparse
import contextlib
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import random

from param_decomp.core.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    MHACIAttention,
    build_ci_fn,
    resolve_ci_placement,
)
from param_decomp.core.components import init_component_stacks
from param_decomp.core.configs import (
    AdamPGDConfig,
    FaithfulnessLossConfig,
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
    NonlinearityLocalityLossConfig,
    PersistentPGDReconLossConfig,
    StochasticReconSubsetLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.init_placed import init_ci_fn_placed, init_component_stacks_placed
from param_decomp.core.losses import EmaFrequency, resolve_frequency
from param_decomp.core.model import PlacedModel
from param_decomp.core.objective import build_objective
from param_decomp.core.placement import from_config
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.sharding import hsdp_mesh, place_target, shard_batch
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_train_step,
)
from param_decomp.targets.glu_transformer import (
    glu_site_specs,
    mlp_family_site_cs,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer_taps import resid_tap_key


def _run(
    steps: int, topology: tuple[int, int, int] | None, sharding: str, census: bool
) -> list[dict[str, float]]:
    cfg = tiny_glu_cfg()
    C, seq, gbatch = 8, 16, 8
    sites = glu_site_specs(cfg, mlp_family_site_cs(3, 6, C))
    model = tiny_glu_decomposed_lm(cfg, sites, random.PRNGKey(0))
    half = len(model.site_names) // 2
    assert len(model.site_names) % 2 == 0, model.site_names
    arch = ChunkwiseTransformerCIArch(
        chunks=(
            Chunk(input_taps=(resid_tap_key(3),), output_sites=model.site_names[:half]),
            Chunk(input_taps=(resid_tap_key(3),), output_sites=model.site_names[half:]),
        ),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    tokens = random.randint(random.PRNGKey(4), (gbatch, seq), 0, cfg.vocab_size)

    mesh = None if topology is None else hsdp_mesh(*topology)
    rules = None if mesh is None else from_config(sharding, mesh, model.sites)
    if mesh is None:
        placed = PlacedModel(model=model, placement=None)
        vu = init_component_stacks(sites, random.PRNGKey(1))
        ci_fn = build_ci_fn(arch, model.sites, random.PRNGKey(2))
        src = init_persistent_sources(
            model.site_names,
            tuple(s.C for s in model.sites),
            (1, seq),
            jnp.float32,
            random.PRNGKey(3),
        )
    else:
        assert rules is not None
        placed = place_target(model, rules)
        vu = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
        ci_fn = init_ci_fn_placed(arch, placed.sites, random.PRNGKey(2), mesh, rules)
        src = init_persistent_sources(
            placed.site_names,
            tuple(s.C for s in placed.sites),
            (1, seq),
            jnp.float32,
            random.PRNGKey(3),
        )
        tokens = shard_batch(tokens, mesh, batch_axis=0)

    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=0.5,
        source_shape="sc",
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(
                max_val=0.01,
                points=(Knot(at=0.0, frac=0.0), Knot(at=0.025, frac=1.0), Knot(at=1.0, frac=1.0)),
            ),
        ),
        n_warmup_steps=2,
    )
    assert ppgd_cfg.coeff is not None
    imp_cfg = ImportanceMinimalityLossConfig(
        coeff=5e-6,
        pnorm=ScheduleConfig(max_val=2.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.2))),
        frequency=FrequencyMinimalityConfig(
            coeff=1e-6, reference_datapoint_count=128, ema_halflife_steps=8.0
        ),
    )
    freq_role = resolve_frequency(imp_cfg.frequency)
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={
                ppgd_cfg.type: PersistentAdversary(
                    sources=src,
                    opt_state=init_sources_adam_state(src),
                    state_key=ppgd_cfg.type,
                    adam=ppgd_cfg.optimizer,
                    n_warmup=ppgd_cfg.n_warmup_steps,
                )
            },
            freq_ema=freq_role.initial_state(model.sites)
            if isinstance(freq_role, EmaFrequency)
            else None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    loss_terms = build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            imp_cfg,
            StochasticReconSubsetLossConfig(
                routing=UniformKSubsetRoutingConfig(), coeff=0.5, n_mask_samples=1
            ),
            NonlinearityLocalityLossConfig(
                coeff=0.05,
                relative_threshold=ScheduleConfig.constant(4.0),
                unit_kind_coefficients={"neuron": 1.0},
            ),
            ppgd_cfg,
        ),
        placed.site_names,
    )
    step = make_train_step(
        model_static=placed,
        substrate=ForwardSubstrate.of(
            placed,
            remat_recon_forwards=True,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=resolve_ci_placement(arch, rules),
        ),
        objective=loss_terms,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        faithfulness=faithfulness_loss_for(model),
    )

    out = []
    with jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext():
        if census:
            assert mesh is not None and mesh.shape["replicate"] > 1, (
                "--census asserts CROSS-REPLICATE collective placement; at replicate=1 "
                "it can only pass vacuously — run it on a replicate>1 --mesh"
            )
            from param_decomp.core.tools.hlo_census import collective_census

            lowered = cast(Any, step).lower(placed, state, tokens, random.PRNGKey(100))
            hlo = lowered.compile().compiled.as_text()
            stride = mesh.shape["fsdp"] * mesh.shape["tp"]
            result = collective_census(hlo, replica_stride=stride, n_devices=mesh.devices.size)
            print(
                f"census: in_loop_cross_replicate={result.in_loop_cross_replicate} "
                f"entry_cross_replicate_reductions={result.exit_reductions}"
            )
            for key in sorted(result.counts):
                print(f"  {key}: {result.counts[key]}")
            # In-loop cross-replicate collectives may only be the sanctioned smalls
            # (replicated-persisted leaves' and batch-shared sources' whole-batch
            # sums); V/U and CI matrix grads must defer to entry reductions.
            assert all(size <= 2**20 for size in result.in_loop_cross_replicate_bytes), (
                result.in_loop_cross_replicate_bytes
            )
            assert result.exit_reductions > 0, result.counts
        for i in range(steps):
            state, m = step(placed, state, tokens, random.PRNGKey(100 + i))
            out.append({k: float(v) for k, v in m.items()})
    return out


def check_device_count_invariance(
    steps: int,
    topology: tuple[int, int, int],
    sharding: str,
    *,
    census: bool,
    rel: float = 5e-4,
) -> float:
    """Run the full step single-layout and topology-sharded and ASSERT the metric
    trajectories match (SPEC D4); returns the worst relative error. `census` also
    asserts the sharded step's cross-replicate collective placement. `rel` is sized
    for the default (1, n, 1) arm; replicate>1 topologies reduce in more orders and
    need a step-count-matched widening (drift grows ~5-10x per step)."""
    single = _run(steps, topology=None, sharding=sharding, census=False)
    sharded = _run(steps, topology=topology, sharding=sharding, census=census)

    # Reduction order differs across topologies, so every f32 reduction carries
    # epsilon-scale divergence, and the per-leaf grad-norm diagnostics amplify it
    # step-over-step through optimizer state — at their 1e-6..1e-3 magnitudes an abs
    # 1e-7 floor is unpromisable. Genuine loss scalars need no floor and keep the
    # tight one.
    diagnostic_abs, scalar_abs = 2e-5, 1e-7

    def abs_floor(key: str) -> float:
        return diagnostic_abs if key.startswith("grad_norms/") else scalar_abs

    n_dev = len(jax.devices())
    failures: list[str] = []
    worst = 0.0
    for i, (a, b) in enumerate(zip(single, sharded, strict=True)):
        for k in a:
            err = abs(a[k] - b[k])
            worst = max(worst, err / (abs(a[k]) + 1e-30))
            if err > rel * abs(a[k]) + abs_floor(k):
                failures.append(
                    f"step {i} {k}: single {a[k]!r} vs sharded({n_dev}) {b[k]!r} err {err:.2e}"
                )
    assert not failures, "trajectory diverged across shardings (SPEC D4):\n" + "\n".join(failures)
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument(
        "--mesh",
        default=None,
        help="replicate x fsdp x tp, e.g. 2x2x2 (default: 1 x n_devices x 1)",
    )
    ap.add_argument("--sharding", default="ddp", help="placement preset for the sharded arm")
    ap.add_argument(
        "--census",
        action="store_true",
        help="assert + print the sharded step's collective census (cross-replicate placement)",
    )
    args = ap.parse_args()

    n_dev = len(jax.devices())
    topology = tuple(int(v) for v in args.mesh.split("x")) if args.mesh else (1, n_dev, 1)
    assert len(topology) == 3
    print(f"devices: {n_dev} mesh: {topology} sharding: {args.sharding!r}")
    worst = check_device_count_invariance(args.steps, topology, args.sharding, census=args.census)
    print(
        f"OK: {args.steps}-step trajectory matches 1-layout vs {n_dev}-device GSPMD "
        f"(worst rel {worst:.2e}; reassociation-only)"
    )


if __name__ == "__main__":
    main()
