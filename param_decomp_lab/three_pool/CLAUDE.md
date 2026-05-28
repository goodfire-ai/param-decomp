# three_pool

3-pool training strategy for SPD on large frozen targets. Splits GPUs into
three rank-disjoint, wall-clock-parallel pools: **CI** (replicated CI fn, DP
across batch), **Layerwise/LW** (V/U sharded by site, block-DDP within group),
**PPGD** (stateless full V/U replica + persistent sources, DP across batch).
A dedicated unsharded CI pool is what makes a *global shared transformer* CI fn
physically realizable. See `DESIGN.md` for the per-step dependency graph and the
pipelining tricks.

## Files

| File | What it covers |
|---|---|
| `config.py` | `ThreePoolConfig` — serializable topology (which ranks form each pool, LW block groups). Pairs with a regular `PDConfig`. Topology integrity asserts live in `validate_topology`. |
| `layout.py` | `World` (declarative topology + process groups), `ThreePoolLayout` (this rank's view + the six cross-pool comm methods), `BatchEdge` (cross-pool batch-slice geometry). |
| `optimize.py` | `ThreePoolTrainer` / `optimize_three_pool` — composition root; mirrors `param_decomp.optimize.optimize`. PDConfig compatibility asserts in `_validate_pd_config_for_three_pool`. |
| `step_ci.py`, `step_layerwise.py`, `step_ppgd.py` | The three per-pool step functions. Each calls the layout's cross-pool exchanges. |
| `reductions.py` | Per-pool SUM/MAX reductions for logging; the `1/n_per_block` LW collapse trick. |
| `eval_step.py` | Cross-pool eval pass; builds `MetricContext` on PPGD, ships CIOutputs CI→PPGD. |
| `checkpoint.py` | Gather full state dict to rank 0. |

## Batch-split routing (the cross-pool wrinkle)

CI/LW/PPGD each shard the global batch on their own axis, so CI values (and the
grads coming back) must be routed across pools along the batch dim. The
constraint (`config.validate_topology` + `optimize._validate_pd_config_*`):

- Each arity divides the global batch: `B % N_ci`, `B % N_per_block`, `B % N_ppgd`.
- Each cross-pool edge (CI↔LW, CI↔PPGD) is **cross-divisible**: one arity
  divides the other, in *either* direction.

Cross-divisibility keeps every CI↔downstream overlap a whole, aligned sub-slice.
`BatchEdge` (in `layout.py`) is the single source of truth for the geometry and
handles **both** fan directions uniformly:

- **CI coarse** (`N_ci ≤ N_down`): one CI rank fans a sub-slice out to
  `fanout = N_down/N_ci` downstream ranks; grads stitch back fanout-to-one.
- **CI fine / inverted** (`N_ci > N_down`): one downstream rank gathers CI from
  `fanout = N_ci/N_down` CI ranks (concat) and scatters grads to those same K.

`BatchEdge.{ci_slices_for_down_slice, down_slices_for_ci_slice}` give the rank
pairings; `{overlap_within_ci, overlap_within_down}` give the matching sub-slice
on each side. The exchange methods in `ThreePoolLayout` are written once against
this API and never branch on regime. Pairs where neither arity divides the other
(ragged overlaps) are out of scope and rejected by the validator.

`PendingCiRecv` holds one packet (CI-coarse) or `fanout` packets (CI-fine, one
per source CI rank) and stitches them into the downstream rank's full `[b_down]`
CI tensor on `wait_and_unpack()`.

## Equivalence harness

`scripts/hetero_topology_equiv/` compares 3-pool loss curves to a single-pool
reference (same seed/losses/coeffs/steps/batch). `compare.py` reads `train/loss/*`
from `metrics.jsonl` and reports per-term 3p/1p ratios. Use a relaxation-
requiring topology (e.g. `threepool_inverted.yaml`: `N_ci=4, N_per_block=2,
N_ppgd=2`) to exercise the inverted path, and a current-legal one
(`threepool_legal.yaml`: `N_ci=2, N_per_block=4`) to guard against regression.
`faith` is ~bit-stable; `stoch`/`imp`/`ppgd` track within the ~1-5%
non-determinism band.

## Gotchas

- Rank 0 must be the LW block-0 leader (reductions ship CI/PPGD losses to rank 0).
- No distributed-aware checkpoint resumption yet (`save_every` stays None).
- All cross-pool p2p goes through `world.cross_pool_p2p_group`, structurally
  separate from the default communicator (barriers) to avoid NCCL wedging.
