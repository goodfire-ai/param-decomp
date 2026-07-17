"""Compile-quality probe for the placement rules table (PLACEMENT_DESIGN.md validation).

For each {owner, zero1, ddp} × {adamw, muon-stacked} cell: place a tiny llama target's
ComponentStacks by the preset on a REAL 2×2 (replicate × fsdp) sim mesh (so owner and
zero1 genuinely differ), build the real train step, lower + compile, and count collective
ops in the optimized HLO. Structural answers for: does owner-partitioned muon compile
without per-iteration NS collectives; does zero1's ÷N→÷fsdp story survive the rules-driven
derivation; is ddp collective-minimal.

Run: XLA_FLAGS="--xla_force_host_platform_device_count=4" python -m \
    param_decomp.experiments.placement_compile_probe
"""

import os
import re

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from param_decomp.ci_fn import Chunk, ChunkwiseTransformerCIArch, MHACIAttention
from param_decomp.configs import (
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MuonOptimizerConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.placement import preset
from param_decomp.recon import build_loss_terms
from param_decomp.run import _ensure_global
from param_decomp.run_state import (
    _optimizer_with_clip,
    stacked_muon_dimension_numbers,
)
from param_decomp.schedule import ScheduleConfig
from param_decomp.targets.glu_transformer import glu_site_specs, mlp_family_site_cs
from param_decomp.targets.glu_transformer_sharding import (
    init_ci_fn_placed,
    init_component_stacks_placed,
)
from param_decomp.tests.test_llama8b import _tiny_cfg, _tiny_decomposed_lm
from param_decomp.train import Decomposition, TrainingItem, TrainState, make_train_step

COLLECTIVES = ("all-gather", "all-reduce", "reduce-scatter", "collective-permute", "all-to-all")


def count_collectives(hlo_text: str) -> dict[str, int]:
    return {op: len(re.findall(rf"\b{op}(?:-start)?\b", hlo_text)) for op in COLLECTIVES}


def build_cell(layout: str, opt_name: str, mesh: Mesh):
    rules = preset(layout, mesh)
    cfg = _tiny_cfg()
    C, seq = 8, 16
    sites = glu_site_specs(cfg, mlp_family_site_cs(3, 4, C))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))

    by_layer: dict[int, list[str]] = {}
    for name in lm.site_names:
        by_layer.setdefault(int(name.split(".")[1]), []).append(name)
    ci_arch = ChunkwiseTransformerCIArch(
        chunks=tuple(
            Chunk(input_taps=(f"resid.{layer}",), output_sites=tuple(names))
            for layer, names in sorted(by_layer.items())
        ),
        input_dim=cfg.n_embd, d_model=16, n_blocks=2, attention=MHACIAttention(n_heads=2),
        ffn_hidden=32, ffn_kind="gelu", learned_norm_scale=False,
    )  # fmt: skip
    vu = init_component_stacks_placed(lm.sites, jax.random.PRNGKey(3), rules)
    ci_fn = init_ci_fn_placed(ci_arch, lm.sites, jax.random.PRNGKey(4), mesh)
    if opt_name == "muon":
        muon_cfg = MuonOptimizerConfig(
            type="muon",
            lr_schedule=ScheduleConfig(start_val=1e-3),
            consistent_rms=0.2,
            grad_clip_norm=0.01,
            impl="stacked",
        )
        opt_vu = _optimizer_with_clip(
            muon_cfg, lambda c: jnp.asarray(1e-3), stacked_muon_dimension_numbers, mesh=mesh
        )
    else:
        opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            step=jnp.zeros((), jnp.int32),
        ),
    )  # fmt: skip
    state = _ensure_global(state, mesh)

    loss_terms = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
            ),
            ChunkwiseSubsetReconLossConfig(
                routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=3, n_samples=1
            ),
        ),
        lm.site_names,
    )
    step = make_train_step(
        lm=lm, losses=loss_terms,
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100, remat_recon_forwards=True, remat_ci_fn=False, mesh=mesh,
    )  # fmt: skip
    tokens = jax.device_put(
        jax.random.randint(jax.random.PRNGKey(9), (4, seq), 0, cfg.vocab_size),
        NamedSharding(mesh, PartitionSpec(("replicate", "fsdp"))),
    )
    return lm, state, step, tokens


def main() -> None:
    assert jax.device_count() == 4, jax.device_count()
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2, 1), ("replicate", "fsdp", "tp"))
    print(f"mesh: replicate=2, fsdp=2, tp=1 ({jax.device_count()} sim devices)\n")
    for layout in ("owner", "zero1", "ddp"):
        for opt_name in ("adamw", "muon"):
            with mesh:
                lm, state, step, tokens = build_cell(layout, opt_name, mesh)
                # eqx filter_jit wraps jit; the checker cannot see .lower — runtime attr
                lowered = step.lower(  # pyright: ignore[reportFunctionMemberAccess]
                    lm, state, tokens, jax.random.PRNGKey(2)
                )
                compiled = lowered.compile()
                # eqx filter_jit wraps Lowered/Compiled; unwrap to the jax object
                inner = getattr(compiled, "compiled", compiled)
                hlo = inner.as_text()
            counts = count_collectives(hlo)
            total = sum(counts.values())
            detail = "  ".join(f"{k}:{v}" for k, v in counts.items() if v)
            print(
                f"{layout:>6} x {opt_name:<6} compile OK   collectives total {total:>4}   {detail}"
            )


if __name__ == "__main__":
    main()
