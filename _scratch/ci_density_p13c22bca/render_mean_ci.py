"""Regenerate the slow-eval log-mean-CI figure for a finished JAX run.

Mirrors the in-loop slow tier (`param_decomp_lab/experiments/lm/run.py::eval_fn`):
restore the checkpoint, harvest the residual from the frozen prefix over the SAME eval
batch the in-loop slow eval would use at `step`, accumulate per-site mean CI, and render
`plot_mean_component_cis_both_scales` (linear + log y). The run dir uses the older layout
(launcher meta in config.yaml, full schema in experiment_config.yaml), so the schema is
read from experiment_config.yaml with the run-identity fields merged in.
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
from param_decomp.slow_eval import _grid_dims, _render_figure, plt
from param_decomp.train import COMPUTE_DT, cast_floating
from param_decomp_lab.experiments.lm.config import build_from_schema
from param_decomp_lab.experiments.lm.load_run import build_target

RUN_DIR = Path("/mnt/data/artifacts/mechanisms/param-decomp/jax_runs/p-13c22bca")
STEP = 200000
OUT_DIR = Path("/mnt/home/oli/claude-slack/data/workspaces/3229052cd398/mean_ci_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

raw = yaml.safe_load((RUN_DIR / "experiment_config.yaml").read_text())
launcher = yaml.safe_load((RUN_DIR / "config.yaml").read_text())
raw["run_id"] = launcher["run_id"]
raw["run_name"] = launcher["run_name"]
raw["out_dir"] = launcher["out_dir"]
# The config records upstream's fp32 weights_dtype, but the JAX frozen target ran (and is
# consumed) in bf16 — the bf16-only loader ignores this field. Declare bf16 so the
# explicit-dtype guard (issue #727) doesn't block opening this finished run.
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
reference = init_train_state(
    cfg.pd, lm, cfg.ci_fn, cfg.data, opt_vu, opt_ci, init_key, src_key, mesh
)
manager = make_checkpoint_manager(RUN_DIR / "ckpts", cfg.cadence.keep_last_n_checkpoints)
state = restore_step(manager, reference, STEP)
print(f"restored step {STEP}", flush=True)

# The exact eval batch the in-loop slow eval would use at STEP: the eval stream seed is
# pd.seed + 1, the batch index is eval_pass_index * n_steps + j (j=0).
eval = cfg.eval
eval_schedule = BatchSchedule(scan_shards(cfg.data.dir), eval.batch_size, cfg.pd.seed + 1)
eval_server = ShardServer(eval_schedule, cfg.data.seq_len, 0, 1)
eval_pass_index = STEP // eval.every
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))


def token_batch(batch_index: int) -> jax.Array:
    local = eval_server.local_batch(batch_index)
    return jax.make_array_from_process_local_data(sharding, local, (eval.batch_size, local.shape[1]))


# Token sweep: accumulate the per-component mean CI incrementally over consecutive eval
# batches and snapshot the running curve at each threshold — an honest march toward the
# infinite-data limit. The 131072-token point is exactly the in-loop slow-eval scale
# (eval.batch_size=64 * seq_len=2048 over one pass). Consecutive batch indices are further
# independent draws from the same eval stream (seed pd.seed+1).
TOKEN_THRESHOLDS = [8192, 32768, 131072, 524288, 2097152, 8388608, 33554432]
seq_len = cfg.data.seq_len
tokens_per_full_batch = eval.batch_size * seq_len
n_full_batches = -(-TOKEN_THRESHOLDS[-1] // tokens_per_full_batch)

# The CI-fn logits are (N*T, ΣC) = (N*2048, 147456); run the CI forward on 4-row sub-slices
# so the bf16 logits (~2.4 GB) fit alongside the restored train state on one B200. The
# residual is harvested for the whole 64-row batch at once (~1 GB bf16) then sub-sliced.
ROWS_PER_FWD = 4
harvest = jax.jit(lambda p, toks: prefix_residual_fn(p, toks))

# The slow-eval step casts the CI fn to fp32, but that fp32 readout is broken on the B200
# (cuDNN attention is bf16-only; the fp32 (N*T, ΣC) projection blows TMEM/HBM). The trained
# model — and every consumer (harvest, load_run.forward) — reads CI in bf16 (COMPUTE_DT), so
# we compute the lower-leaky CI in bf16 here. At plot resolution this is indistinguishable
# from the intended fp32 mean-CI curve.
ci_fn_bf16 = cast_floating(state.ci_fn, COMPUTE_DT)
site_names = lm.site_names


@jax.jit
def mean_ci_step(ci_fn, frozen, residual):
    site_inputs = {s: x.astype(COMPUTE_DT) for s, x in lm.site_inputs(frozen, residual).items()}
    lower = ci_fn(site_inputs).lower
    sums = {s: lower[s].astype(jnp.float32).reshape(-1, lower[s].shape[-1]).sum(0) for s in site_names}
    n_positions = lower[site_names[0]].reshape(-1, lower[site_names[0]].shape[-1]).shape[0]
    return sums, jnp.asarray(n_positions, jnp.int32)


def plot_mean_cis_with_token_count(mean_cis: dict[str, np.ndarray], n_tokens: int) -> tuple[bytes, bytes]:
    """`plot_mean_component_cis_both_scales` with an `n_tokens` suptitle on each figure."""
    sorted_data = {name: np.sort(v)[::-1] for name, v in mean_cis.items()}
    n_rows, n_cols = _grid_dims(len(sorted_data))
    images: list[bytes] = []
    for log_y in (False, True):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows), squeeze=False)
        flat_axes = axs.T.ravel()
        for ax in flat_axes[len(sorted_data) :]:
            ax.set_visible(False)
        for ax, (name, sorted_components) in zip(flat_axes, sorted_data.items(), strict=False):
            if log_y:
                ax.set_yscale("log")
            ax.scatter(range(len(sorted_components)), sorted_components, marker="x", s=10)
            ax.set_xlabel("Component")
            ax.set_ylabel("mean CI")
            ax.set_title(name, fontsize=10)
        fig.suptitle(f"mean CI per component — {n_tokens:,} tokens", fontsize=12)
        fig.tight_layout()
        images.append(_render_figure(fig))
    return images[0], images[1]


def render_snapshot(mean_cis: dict[str, np.ndarray], n_tokens: int) -> None:
    label = f"{n_tokens // 1000}k" if n_tokens < 1_000_000 else f"{n_tokens / 1_000_000:.1f}M"
    linear_png, log_png = plot_mean_cis_with_token_count(mean_cis, n_tokens)
    (OUT_DIR / f"ci_mean_per_component_{label}.png").write_bytes(linear_png)
    (OUT_DIR / f"ci_mean_per_component_log_{label}.png").write_bytes(log_png)
    np.savez(
        OUT_DIR / f"mean_cis_{label}.npz",
        **{s.replace(".", "_"): v for s, v in mean_cis.items()},
    )
    print(f"[{label}] n_positions={n_tokens}", flush=True)
    for s, v in mean_cis.items():
        print(f"    {s}: C={v.size} max={v.max():.4f} median={np.median(v):.2e} >0.5: {(v > 0.5).sum()}", flush=True)


ci_sums = {s: np.zeros(0) for s in site_names}
total_positions = 0
threshold_idx = 0
base_index = eval_pass_index * eval.n_steps
for b in range(n_full_batches):
    residual = harvest(prefix, token_batch(base_index + b))
    for r0 in range(0, eval.batch_size, ROWS_PER_FWD):
        sums, n_pos = mean_ci_step(ci_fn_bf16, frozen, residual[r0 : r0 + ROWS_PER_FWD])
        total_positions += int(n_pos)
        for s in site_names:
            v = np.asarray(sums[s])
            ci_sums[s] = v if ci_sums[s].size == 0 else ci_sums[s] + v
        while threshold_idx < len(TOKEN_THRESHOLDS) and total_positions >= TOKEN_THRESHOLDS[threshold_idx]:
            render_snapshot({s: ci_sums[s] / total_positions for s in site_names}, TOKEN_THRESHOLDS[threshold_idx])
            threshold_idx += 1

print("wrote sweep figures + npz to", OUT_DIR, flush=True)
