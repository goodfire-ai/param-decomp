"""Harvest the UNREDUCED per-token CI for p-13c22bca and accumulate a per-component CI
histogram — the input to a genuine CI-density heatmap (x = component, y = per-token CI).

Unlike render_mean_ci.py (which means CI over tokens, collapsing each component to one
point), this keeps the full distribution: for each component we count how many of its
per-token CI values fall in each log-spaced CI band. Streamed over the same eval batches
the in-loop slow eval would use. Saves per-component (C, N_BINS) counts + per-component sum
(for the density ordering) at each token threshold.

CI bin 0 is an underflow bin (CI < CI_FLOOR, including exact 0 = inactive); bins 1..NY are
log-spaced in [CI_FLOOR, 1], with the top bin including CI = 1.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from param_decomp.checkpoint import make_checkpoint_manager, restore_step
from param_decomp.data import BatchSchedule, ShardServer, scan_shards
from param_decomp.run_state import build_optimizers, init_train_state
from param_decomp.sharding import dp_mesh
from param_decomp.train import COMPUTE_DT, cast_floating
from param_decomp_lab.experiments.lm.config import build_from_schema
from param_decomp_lab.experiments.lm.load_run import build_target

RUN_DIR = Path("/mnt/data/artifacts/mechanisms/param-decomp/jax_runs/p-13c22bca")
STEP = 200000
OUT_DIR = Path("/mnt/home/oli/claude-slack/data/workspaces/3229052cd398/mean_ci_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NY = 90
CI_FLOOR = 1e-9
Y_EDGES = jnp.logspace(np.log10(CI_FLOOR), 0.0, NY + 1)
N_BINS = NY + 1  # bin 0 = underflow (< CI_FLOOR, ie exact 0); bins 1..NY = log bands over [1e-9, 1]

raw = yaml.safe_load((RUN_DIR / "experiment_config.yaml").read_text())
launcher = yaml.safe_load((RUN_DIR / "config.yaml").read_text())
raw["run_id"] = launcher["run_id"]
raw["run_name"] = launcher["run_name"]
raw["out_dir"] = launcher["out_dir"]
raw["target"]["weights_dtype"] = "bfloat16"

cfg = build_from_schema(raw)
assert cfg.eval is not None
assert cfg.data is not None
print(f"run {cfg.run.run_id} | sites={[s.name for s in cfg.target.sites]}", flush=True)

mesh = dp_mesh()
print(f"devices: {jax.devices()}", flush=True)
lm, frozen, prefix, prefix_residual_fn, vocab = build_target(cfg, mesh)
print("target built", flush=True)

opt_vu, opt_ci, _ = build_optimizers(cfg.pd)
init_key, src_key = jax.random.split(jax.random.PRNGKey(cfg.pd.seed))
reference = init_train_state(cfg.pd, lm, cfg.ci_fn, cfg.data, opt_vu, opt_ci, init_key, src_key, mesh)
manager = make_checkpoint_manager(RUN_DIR / "ckpts", cfg.cadence.keep_last_n_checkpoints)
state = restore_step(manager, reference, STEP)
print(f"restored step {STEP}", flush=True)

eval = cfg.eval
eval_schedule = BatchSchedule(scan_shards(cfg.data.dir), eval.batch_size, cfg.pd.seed + 1)
eval_server = ShardServer(eval_schedule, cfg.data.seq_len, 0, 1)
eval_pass_index = STEP // eval.every
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))


def token_batch(batch_index: int) -> jax.Array:
    local = eval_server.local_batch(batch_index)
    return jax.make_array_from_process_local_data(sharding, local, (eval.batch_size, local.shape[1]))


TOKEN_THRESHOLDS = [131072, 2097152, 33554432]
seq_len = cfg.data.seq_len
tokens_per_full_batch = eval.batch_size * seq_len
n_full_batches = -(-TOKEN_THRESHOLDS[-1] // tokens_per_full_batch)

ROWS_PER_FWD = 4
harvest = jax.jit(lambda p, toks: prefix_residual_fn(p, toks))
ci_fn_bf16 = cast_floating(state.ci_fn, COMPUTE_DT)
site_names = lm.site_names


@jax.jit
def hist_step(ci_fn, frozen, residual):
    site_inputs = {s: x.astype(COMPUTE_DT) for s, x in lm.site_inputs(frozen, residual).items()}
    lower = ci_fn(site_inputs).lower
    counts = {}
    sums = {}
    for s in site_names:
        v = lower[s].astype(jnp.float32).reshape(-1, lower[s].shape[-1])
        idx = jnp.clip(jnp.digitize(v, Y_EDGES), 0, NY)
        counts[s] = jnp.stack([(idx == b).sum(axis=0) for b in range(N_BINS)], axis=1).astype(jnp.int32)
        sums[s] = v.sum(0)
    n_positions = lower[site_names[0]].reshape(-1, lower[site_names[0]].shape[-1]).shape[0]
    return counts, sums, jnp.asarray(n_positions, jnp.int32)


def save_snapshot(counts: dict[str, np.ndarray], sums: dict[str, np.ndarray], n_tokens: int) -> None:
    label = f"{n_tokens // 1000}k" if n_tokens < 1_000_000 else f"{n_tokens / 1_000_000:.1f}M"
    np.savez(
        OUT_DIR / f"ci_hist_{label}.npz",
        y_edges=np.asarray(Y_EDGES),
        ci_floor=CI_FLOOR,
        n_tokens=n_tokens,
        **{f"counts__{s.replace('.', '_')}": counts[s] for s in site_names},
        **{f"sum__{s.replace('.', '_')}": sums[s] for s in site_names},
    )
    print(f"[{label}] n_positions={n_tokens}", flush=True)
    for s in site_names:
        c = counts[s]
        active_frac = 1.0 - c[:, 0].sum() / c.sum()
        print(f"    {s}: total_obs={c.sum()} active_frac={active_frac:.4f}", flush=True)


count_acc = {s: np.zeros((0, N_BINS), np.int64) for s in site_names}
sum_acc = {s: np.zeros(0, np.float64) for s in site_names}
total_positions = 0
threshold_idx = 0
base_index = eval_pass_index * eval.n_steps
diagnosed = False
for b in range(n_full_batches):
    residual = harvest(prefix, token_batch(base_index + b))
    for r0 in range(0, eval.batch_size, ROWS_PER_FWD):
        counts, sums, n_pos = hist_step(ci_fn_bf16, frozen, residual[r0 : r0 + ROWS_PER_FWD])
        total_positions += int(n_pos)
        for s in site_names:
            c = np.asarray(counts[s])
            sm = np.asarray(sums[s])
            count_acc[s] = c.astype(np.int64) if count_acc[s].size == 0 else count_acc[s] + c
            sum_acc[s] = sm.astype(np.float64) if sum_acc[s].size == 0 else sum_acc[s] + sm
        if not diagnosed:
            diagnosed = True
            s0 = site_names[0]
            col = count_acc[s0].sum(0)
            print(f"DIAG {s0} after {total_positions} positions, bin distribution (bin0=underflow):", flush=True)
            print("    edges:", np.round(np.asarray(Y_EDGES), 5), flush=True)
            print("    bin_totals:", col, flush=True)
            print(f"    active_frac={1.0 - col[0] / col.sum():.4f}", flush=True)
        while threshold_idx < len(TOKEN_THRESHOLDS) and total_positions >= TOKEN_THRESHOLDS[threshold_idx]:
            save_snapshot(count_acc, sum_acc, TOKEN_THRESHOLDS[threshold_idx])
            threshold_idx += 1

print("wrote ci-hist npz to", OUT_DIR, flush=True)
