"""De-risk: a C-sharded matmul (collective) inside lax.cond inside lax.scan under a mesh.

Mirrors the recon structure: a stack of layers, each either frozen (x@W) or decomposed
(x@V@U with the contracted C axis sharded over 'dp' -> an all-reduce). The decomposed
branch's collective sits inside a cond whose predicate is REPLICATED (same per-layer flag
on every device). Validates: compiles, runs, and matches the unrolled python reference."""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

L, d, C, B = 6, 8, 16, 2  # layers, model dim, C (sharded over 4), batch


def main() -> None:
    mesh = Mesh(np.array(jax.devices()), ("dp",))
    k = jax.random.PRNGKey(0)
    W = jax.random.normal(jax.random.fold_in(k, 1), (L, d, d))
    V = jax.random.normal(jax.random.fold_in(k, 2), (L, d, C))
    U = jax.random.normal(jax.random.fold_in(k, 3), (L, C, d))
    flags = jnp.array([False, True, True, False, True, False])
    x0 = jax.random.normal(jax.random.fold_in(k, 4), (B, d))

    def frozen_fn(x, W_l, V_l, U_l):
        return x @ W_l.T

    def decomp_fn(x, W_l, V_l, U_l):
        return (x @ V_l) @ U_l  # x@V_l: (B,C) C-sharded; @U_l: contracts sharded C -> all-reduce

    def body(x, layer):
        W_l, V_l, U_l, f_l = layer
        y = jax.lax.cond(f_l, decomp_fn, frozen_fn, x, W_l, V_l, U_l)
        return x + y, None

    def model(x, W, V, U, flags):
        out, _ = jax.lax.scan(body, x, (W, V, U, flags))
        return out

    def ref(x, W, V, U, flags):
        for i in range(L):
            y = (x @ V[i]) @ U[i] if bool(flags[i]) else x @ W[i].T
            x = x + y
        return x

    repl = NamedSharding(mesh, P())
    shardings = (
        repl,                                   # x
        repl,                                   # W (L,d,d) replicated
        NamedSharding(mesh, P(None, None, "dp")),  # V (L,d,C) shard C
        NamedSharding(mesh, P(None, "dp", None)),  # U (L,C,d) shard C
        repl,                                   # flags
    )
    jitted = jax.jit(model, in_shardings=shardings, out_shardings=repl)

    print("compiling scan+cond+collective ...", flush=True)
    compiled = jitted.lower(x0, W, V, U, flags).compile()
    print("compiled OK", flush=True)
    got = jitted(x0, W, V, U, flags)
    want = ref(x0, W, V, U, flags)
    maxerr = float(jnp.max(jnp.abs(got - want)))
    print(f"ran OK | max abs err vs unrolled reference = {maxerr:.2e}", flush=True)
    assert maxerr < 1e-4, maxerr
    print("PROBE PASS: collective inside cond inside scan works + matches reference", flush=True)


if __name__ == "__main__":
    main()
