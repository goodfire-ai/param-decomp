"""Causal importance of every p-594db290 subcomponent at the '=' position of the 10000
'a+b=' prompts (a, b in 1..100). Each prompt tokenizes to exactly [BOS, a, +, b, =]
(every number 1..100 is a single Llama-3.1 token), so '=' is the final position.

Writes per-site CI at the '=' position, shape (10000, C), row order a-outer/b-inner
(row i -> a = i//100 + 1, b = i%100 + 1). Also writes the pre-mask component activation
|U_c|*(x@V) at '=' for the same rows (arith_eq_act.npz).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from forward_loader import open_forward_only

RUN_DIR = Path("/mnt/data/artifacts/mechanisms/param-decomp/runs/p-594db290")
STEP = 400000
BATCH = 500
HERE = Path(__file__).parent


def main() -> None:
    tokens = np.load(HERE / "arith_tokens.npy")
    assert tokens.shape == (10000, 5), tokens.shape

    run = open_forward_only(RUN_DIR, STEP)
    sites = run.layer_activation_sizes
    print(f"run {run.run_id} step {run.step}; sites:", flush=True)
    for name, c in sites:
        print(f"  {name}: C={c}", flush=True)

    eq_ci = {name: np.zeros((tokens.shape[0], c), dtype=np.float32) for name, c in sites}
    eq_act = {name: np.zeros((tokens.shape[0], c), dtype=np.float32) for name, c in sites}
    for start in range(0, tokens.shape[0], BATCH):
        chunk = tokens[start : start + BATCH]
        fwd = run.forward(jnp.asarray(chunk))
        for name, _ in sites:
            eq_ci[name][start : start + BATCH] = np.asarray(fwd.lower_leaky_ci[name][:, -1, :])
            eq_act[name][start : start + BATCH] = np.asarray(fwd.component_acts[name][:, -1, :])
        print(f"  {start + chunk.shape[0]}/{tokens.shape[0]} prompts", flush=True)

    np.savez(HERE / "arith_eq_ci.npz", **eq_ci)
    np.savez(HERE / "arith_eq_act.npz", **eq_act)
    print(f"saved {HERE / 'arith_eq_ci.npz'}", flush=True)
    for name, _ in sites:
        m = eq_ci[name].max(axis=0)
        n_over = int((m > 0.5).sum())
        print(
            f"  {name}: {n_over} components with max CI > 0.5 (global max {m.max():.4g})",
            flush=True,
        )


if __name__ == "__main__":
    main()
