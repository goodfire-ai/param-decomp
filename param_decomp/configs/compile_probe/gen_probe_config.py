"""Generate a small full-model-faithful compile-probe config.

DERIVED from the production 32L config (llama8b_full32L_seq512_b128_dp64.yaml) rather
than hand-mirroring the schema — so it stays valid as the schema moves. Keeps the
production algorithm (chunkwise-recon + persistent-PGD + faith + imp-min), CI-fn arch,
per-kind C structure, and tp; scales down to the LAST `n_layers` (the live decomposed
span the scan walks), one recon chunk, 10 steps, 2 faith-warmup steps, eval never fires.

Per-arm isolation: `PARAM_DECOMP_OUT_DIR` in the rank env points the run dir AND both
persistent caches (xla_compilation_cache / xla_autotune_cache, siblings of runs/) at
`compile_probe_scratch/<arm>`, so arms never share a cache. `JAX_LOG_COMPILES=1` prints
per-jit compile durations into the slurm log.

Usage:
    python gen_probe_config.py <n_layers> <dp> <arm_name> <out.yaml> [k=v ...]

Trailing `k=v` pairs are extra `runtime.compiler_options` (e.g.
`xla_gpu_autotune_level=0`); specifying any restates the full default set (a yaml
`compiler_options` block REPLACES the default dict).
"""

import sys
from pathlib import Path

import yaml

BASE_CONFIG = Path(__file__).parent.parent / "llama8b_full32L_seq512_b128_dp64.yaml"
SCRATCH = "/mnt/data/artifacts/mechanisms/param-decomp/compile_probe_scratch"
N_TOTAL_LAYERS = 32

# mirror of RuntimeConfig.compiler_options defaults, restated because a yaml
# compiler_options block replaces the whole default dict
DEFAULT_COMPILER_OPTIONS = {
    "xla_gpu_enable_latency_hiding_scheduler": True,
    "xla_gpu_enable_triton_gemm": False,
    "xla_gpu_enable_command_buffer": "",
    "xla_gpu_enable_highest_priority_async_stream": True,
    "xla_gpu_all_reduce_combine_threshold_bytes": 1073741824,
    "xla_gpu_all_gather_combine_threshold_bytes": 1073741824,
    "xla_gpu_reduce_scatter_combine_threshold_bytes": 134217728,
    "xla_gpu_enable_pipelined_all_gather": True,
    "xla_gpu_enable_pipelined_reduce_scatter": True,
    "xla_gpu_enable_pipelined_all_reduce": True,
    "xla_gpu_enable_while_loop_double_buffering": True,
    "xla_gpu_enable_all_gather_combine_by_dim": False,
    "xla_gpu_enable_reduce_scatter_combine_by_dim": False,
}


def main(n_layers: int, dp: int, arm: str, out_path: str, extra_opts: dict[str, object]) -> None:
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg["run_name"] = f"compileprobe-{arm}"
    cfg.pop("wandb", None)

    keep_layers = {str(i) for i in range(N_TOTAL_LAYERS - n_layers, N_TOTAL_LAYERS)}
    targets = [
        t
        for t in cfg["pd"]["decomposition_targets"]
        if t["module_pattern"].split(".")[2] in keep_layers
    ]
    assert len(targets) == 7 * n_layers, len(targets)
    cfg["pd"]["decomposition_targets"] = targets
    cfg["pd"]["steps"] = 10
    cfg["pd"]["batch_size"] = dp
    cfg["pd"]["faithfulness_warmup_steps"] = 2
    for lm in cfg["pd"]["loss_metrics"]:
        if lm["type"] == "ChunkwiseSubsetReconLoss":
            lm["sites_per_chunk"] = 7 * n_layers

    cfg["cadence"] = {"keep_last_n_checkpoints": 1, "save_every": 100000, "train_log_every": 1}
    cfg["eval"]["batch_size"] = dp
    cfg["eval"]["every"] = 100000
    cfg["eval"]["slow_every"] = 1000000
    cfg["eval"]["slow_on_first_step"] = False

    cfg["runtime"]["dp"] = dp
    if extra_opts:
        cfg["runtime"]["compiler_options"] = DEFAULT_COMPILER_OPTIONS | extra_opts
    cfg["runtime"]["launch_env"] = {
        "env": {"JAX_LOG_COMPILES": "1", "PARAM_DECOMP_OUT_DIR": f"{SCRATCH}/{arm}"}
    }

    Path(out_path).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out_path}: arm={arm} {len(targets)} sites dp={dp} extra={extra_opts}")


if __name__ == "__main__":
    extra = {}
    for kv in sys.argv[5:]:
        k, v = kv.split("=", 1)
        extra[k] = int(v) if v.isdigit() else v
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], extra)
