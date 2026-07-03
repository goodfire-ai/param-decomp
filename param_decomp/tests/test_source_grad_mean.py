"""Replicated-source-leaf grad is the global MEAN, not N× it (SPEC R-7, D1/S16/D3).

Torch AVG-reduces shared-source grads explicitly (`reduce_source_grads(op=AVG)`,
`persistent_pgd_state.py`); JAX gets the same AVG *implicitly*: GSPMD autodiff of a
global-mean loss (`kl_per_position` normalizes by the GLOBAL B·T) over a REPLICATED
source leaf (`init_sources_sharded` -> `P()`) produces a mean cotangent. This is the
exact spot of the historical 3-pool SUM bug (project_3pool_ppgd_source_reduce_bug);
nothing previously pinned the AVG.

The contract this guards: the per-device gradient flowing into the replicated `sc`-scope
source is the GLOBAL mean. If GSPMD instead emitted a bare all-reduce(SUM) on the
source cotangent (the bug), the N-device source grad would be N× the single-device grad.
We compute the source-leaf grad of the adversarial ascent objective
(`kl_per_position(masked_output(...), clean_output)`, the same loss the persistent
ascent backprops, SPEC S12'/S14') on the SAME fixed global batch + seed at 1 layout
(`mesh=None`) and at N≥2 simulated devices, and assert they match to rel ≤ 1e-4.

Run the multi-device leg via the simulated-device env (matches the validation stack):

  XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    python -m pytest param_decomp/tests/test_source_grad_mean.py

At the default single-device count the multi-device leg is skipped (the SUM-vs-MEAN
distinction is only observable with >1 device); the test is then a no-op assertion that
the single-layout grad is finite.

Finding (4 sim CPU devices, tiny Llama MLP target): every site's per-device source grad
matches the single-layout grad with magnitude ratio 1.0000 (a SUM bug would give 4.0),
max abs err ~1e-10 against a ~1e-4 grad scale. The implicit GSPMD all-reduce on the
replicated source cotangent is therefore the global MEAN (already-normalized partials
summed), matching torch's explicit `reduce_source_grads(op=AVG)`.
"""

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.adversary import init_persistent_sources, source_masks
from param_decomp.ci_fn import Chunk, ChunkwiseTransformerCIArch, build_ci_fn
from param_decomp.components import init_decomp_vu
from param_decomp.losses import kl_per_position
from param_decomp.sharding import hsdp_mesh, shard_batch
from param_decomp.targets.llama8b import (
    llama_site_specs,
    mlp_family_site_cs,
)
from param_decomp.tests.test_llama8b import _tiny_cfg, _tiny_decomposed_lm
from param_decomp.train import COMPUTE_DT, cast_floating


def _source_grad(sharded: bool) -> dict[str, jax.Array]:
    """Grad of the route-all adversarial KL objective w.r.t. the persistent `sc`-scope
    sources, with components/CI frozen (SPEC §4.5) — the leaf whose cross-device
    reduction we are pinning. Returns one fp32 grad array per site."""
    cfg = _tiny_cfg()
    C, seq, gbatch = 8, 16, 8
    sites = llama_site_specs(cfg, mlp_family_site_cs(3, 6, C))
    lm = _tiny_decomposed_lm(cfg, sites, random.PRNGKey(0))
    vu = init_decomp_vu(sites, random.PRNGKey(1))
    first_block = min(int(name.split(".")[1]) for name in lm.site_names)
    ci_arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=lm.site_names),),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        mlp_hidden=32,
    )
    ci_fn = build_ci_fn(ci_arch, lm.sites, random.PRNGKey(2))
    src = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), (1, seq), jnp.float32, random.PRNGKey(3)
    )
    resid = random.randint(random.PRNGKey(4), (gbatch, seq), 0, cfg.vocab_size)

    mesh = hsdp_mesh() if sharded else None
    if mesh is not None:
        resid = shard_batch(resid, mesh, batch_axis=0)
        # The source leaf is REPLICATED across `dp` — exactly `init_sources_sharded`'s
        # placement (`P()`); this is where the SUM-vs-MEAN reduction lands.
        src = {name: jax.device_put(v, NamedSharding(mesh, P())) for name, v in src.items()}

    components_bf16 = cast_floating(vu, COMPUTE_DT)
    ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
    taps = lm.read_activations(resid, ci_fn.input_names)
    ci_lower = ci_fn_bf16(taps, remat=False).lower
    clean_output = jax.lax.stop_gradient(lm.clean_output(resid))

    def source_loss(sources: dict[str, jax.Array]) -> jax.Array:
        masks, delta_masks = source_masks(ci_lower, sources, lm.site_names)
        masked = lm.masked_output(
            lm.prepare_compute_weights(components_bf16),
            resid,
            masks,
            delta_masks,
            None,
            lm.site_names,
            True,
            remat=False,
        )
        return kl_per_position(masked, clean_output)

    grad_fn = jax.jit(jax.grad(source_loss))
    if mesh is not None:
        repl = NamedSharding(mesh, P())
        grad_fn = jax.jit(jax.grad(source_loss), out_shardings={name: repl for name in src})
    return grad_fn(src)


def test_source_leaf_grad_is_global_mean_not_sum():
    n_dev = len(jax.devices())
    single = _source_grad(sharded=False)
    for name, g in single.items():
        assert jnp.all(jnp.isfinite(g)), name

    if n_dev == 1:
        return  # SUM vs MEAN (N× the mean) is only observable with >1 device

    sharded = _source_grad(sharded=True)
    # Combined abs+rel as in `experiments/invariance_check.py`: the cross-shard
    # reduction order differs (bf16 masked forward), so a few tiny grad entries with
    # cancellation graze a pure-relative 1e-4 while their ABSOLUTE error stays ~1e-10,
    # orders of magnitude below the ~1e-4 grad scale — reassociation noise, not a SUM.
    REL, ABS = 1e-4, 1e-6
    for name in single:
        a = single[name]
        b = sharded[name]
        max_abs_err = float(jnp.max(jnp.abs(a - b)))
        max_allowed = REL * float(jnp.max(jnp.abs(a))) + ABS
        assert max_abs_err <= max_allowed, (
            f"site {name}: source grad differs across 1 vs {n_dev} devices "
            f"(max abs err {max_abs_err:.2e} > {max_allowed:.2e}); per-device grad is "
            f"not the global MEAN (an all-reduce(SUM) bug would give ~{n_dev}× the mean — R-7)"
        )
        # The load-bearing MEAN-vs-SUM discriminator, immune to per-element cancellation:
        # an all-reduce(SUM) bug makes the per-device grad ~n_dev× the single-layout one,
        # so the magnitude ratio would be ~n_dev, not ~1.
        ratio = float(jnp.sum(jnp.abs(b)) / jnp.sum(jnp.abs(a)))
        assert abs(ratio - 1.0) < 1e-3, (
            f"site {name}: sharded/single grad magnitude ratio {ratio:.4f} (expected ~1.0); "
            f"~{n_dev} would mean the replicated source leaf got an all-reduce(SUM), not MEAN (R-7)"
        )
