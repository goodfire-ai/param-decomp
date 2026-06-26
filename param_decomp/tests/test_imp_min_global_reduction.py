"""Imp-min global-sum-inside-log2 is device-count invariant (SPEC S7/S8/D2).

The eval-path imp-min value reuses the SAME `importance_minimality_terms` as the
train path (there is one impl in JAX — `losses.py`). Its correctness hinges on the
per-component sums being the EXACT global-batch sums, formed BEFORE the `log2`
(Jensen: mean-of-per-shard-log ≠ log-of-global-sum). Under GSPMD, `jnp.sum` over the
batch-sharded `(b, t)` axes IS that global reduction — XLA inserts the cross-shard
all-reduce inside the graph.

This guards that the value is invisible to sharding layout: compute the terms on the
same fixed global CI tensors once single-layout (mesh=None, all on device 0) and once
batch-sharded over every visible device, and assert they match to rel ≤ 1e-4
(cross-shard reduction order differs, so bit-exactness is not achievable). Run under
the simulated multi-device CPU env to exercise n > 1:

  XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    python -m pytest param_decomp/tests/test_imp_min_global_reduction.py
"""

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.losses import importance_minimality_terms
from param_decomp.sharding import hsdp_mesh, shard_batch


def _global_ci_upper() -> dict[str, jax.Array]:
    """Two heterogeneous-C sites; batch B divisible by any visible device count."""
    mesh = hsdp_mesh()
    n = mesh.devices.size
    B, T = 8 * n, 16
    return {
        "layers.0.mlp.gate_proj": random.uniform(random.PRNGKey(0), (B, T, 12)),
        "layers.1.self_attn.q_proj": random.uniform(random.PRNGKey(1), (B, T, 5)),
    }


def test_imp_min_global_reduction_invariant_to_device_count():
    pnorm = jnp.asarray(2.0)
    eps = 1e-12
    ci_upper = _global_ci_upper()
    sample = next(iter(ci_upper.values()))
    n_positions = sample.shape[0] * sample.shape[1]

    lp_single, freq_single = importance_minimality_terms(
        ci_upper, pnorm, eps, reference_token_count=n_positions
    )

    mesh = hsdp_mesh()

    @jax.jit
    def sharded_terms(ci: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        ci = {
            site: jax.lax.with_sharding_constraint(v, NamedSharding(mesh, P(("replicate", "fsdp"), None, None)))
            for site, v in ci.items()
        }
        return importance_minimality_terms(ci, pnorm, eps, reference_token_count=n_positions)

    ci_sharded = {site: shard_batch(v, mesh, batch_axis=0) for site, v in ci_upper.items()}
    lp_sharded, freq_sharded = sharded_terms(ci_sharded)

    # freq is the Jensen-sensitive term (global f_c INSIDE log2); lp is linear.
    for name, single, sharded in (
        ("lp", lp_single, lp_sharded),
        ("freq", freq_single, freq_sharded),
    ):
        single_f, sharded_f = float(single), float(sharded)
        rel = abs(single_f - sharded_f) / (abs(single_f) + 1e-30)
        assert rel <= 1e-4, (
            f"imp-min {name} diverged across shardings (n={mesh.devices.size}): "
            f"single {single_f!r} vs sharded {sharded_f!r} rel {rel:.2e} — "
            "global per-component sum not formed before log2 (SPEC S8/D2)"
        )
