"""Recompile smoke: a 3-step CPU TMS run through the generic engine compiles the jitted
train step exactly ONCE. A second `jit(step)` compile means something in the loop's step
inputs (state / batch / key) changed pytree structure, shape, or dtype between steps —
the structure-instability regression class the 2026-07-10 compile audit hunted statically,
pinned here dynamically. Rides on `profile.log_compiles` (so the toggle is exercised
end-to-end) and also asserts the one-shot `train/perf/compile_s` scalar lands in
`metrics.jsonl` exactly once.
"""

import dataclasses
import json
import logging
import re
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml
from jax.sharding import Mesh

from param_decomp_lab.experiments.tms.config import TMSExperimentConfig
from param_decomp_lab.experiments.tms.run import build_tms_built_run, run_tms_decomposition

# jax 0.10 emits one `Compiling jit(<fn name>)` line per XLA compile when
# `jax_log_compiles` is on (`jax._src.interpreters.pxla`, promoted to WARNING); equinox's
# `filter_jit` preserves the wrapped fn's name, so the train step logs as `jit(step)`.
_COMPILING = re.compile(r"Compiling jit\((\w+)\)")


def test_three_step_toy_run_compiles_jit_step_exactly_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    raw = yaml.safe_load((Path(__file__).parent / "configs" / "tms_5-5_SMOKE.yaml").read_text())
    raw["pd"]["steps"] = 3
    raw["pd"]["faithfulness_warmup_steps"] = 2
    raw["target"]["pretrain"]["steps"] = 20
    raw["cadence"] = {"train_log_every": 1, "save_every": 10, "keep_last_n_checkpoints": 1}
    raw["runtime"]["launch_env"] = {"profile": {"log_compiles": True}}

    built = build_tms_built_run(TMSExperimentConfig(**raw), "p-00c0ffee")
    built = dataclasses.replace(built, run=dataclasses.replace(built.run, out_dir=tmp_path))

    # One-device mesh, not `hsdp_mesh()` over all devices: the recompile property is
    # device-count-independent, and the toy's d_in=5 doesn't tile a
    # `--xla_force_host_platform_device_count` sim mesh.
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1, 1), axis_names=("replicate", "fsdp", "tp")
    )
    caplog.set_level(logging.WARNING, logger="jax")
    try:
        run_tms_decomposition(built, raw, mesh)
    finally:
        # the engine flips these process-globals on for the run; don't leak into other tests
        jax.config.update("jax_log_compiles", False)
        jax.config.update("jax_explain_cache_misses", False)

    compiled = [
        m.group(1) for m in (_COMPILING.search(r.getMessage()) for r in caplog.records) if m
    ]
    assert compiled.count("step") == 1, (
        f"jit(step) compiled {compiled.count('step')}x over a 3-step run (expected exactly 1"
        f" — step inputs changed structure between steps). All compiles: {compiled}"
    )

    metrics = [
        json.loads(line) for line in (built.run.run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    compile_s = [m["train/perf/compile_s"] for m in metrics if "train/perf/compile_s" in m]
    assert len(compile_s) == 1 and compile_s[0] > 0, (
        f"expected exactly one positive train/perf/compile_s record, got {compile_s}"
    )
