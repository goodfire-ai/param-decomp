"""Local PD experiment runner"""

import json
import subprocess
import sys
from typing import Any

import fire
import yaml

from param_decomp.log import logger
from param_decomp.registry import EXPERIMENT_REGISTRY
from param_decomp.settings import REPO_ROOT


def _parse_override(spec: str) -> tuple[list[str], object]:
    """Parse a single 'a.b.c=value' override into (path, parsed_value)."""
    assert "=" in spec, f"--override entry must be 'key=value', got {spec!r}"
    key, _, raw = spec.partition("=")
    key = key.strip()
    assert key, f"empty key in override {spec!r}"
    value = yaml.safe_load(raw)
    return key.split("."), value


def _set_at_path(d: dict[str, Any], path: list[str], value: object) -> None:
    cursor: Any = d
    for part in path[:-1]:
        assert isinstance(cursor, dict), f"override path {path} traverses non-dict at {part!r}"
        cursor = cursor.setdefault(part, {})
    assert isinstance(cursor, dict)
    cursor[path[-1]] = value


def _build_overridden_config_json(config_path: str, overrides: list[str]) -> str:
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    for spec in overrides:
        path, value = _parse_override(spec)
        _set_at_path(data, path, value)
    return "json:" + json.dumps(data)


def main(
    experiment: str,
    cpu: bool = False,
    dp: int | None = None,
    override: str | list[str] | None = None,
) -> None:
    """Run a single PD experiment locally.

    Args:
        experiment: Experiment name from registry (e.g., 'tms_5-2', 'resid_mlp1')
        cpu: Run on CPU instead of GPU
        dp: Number of GPUs for single-node data parallelism (requires 2+)
        override: One or more `key=value` overrides applied on top of the experiment YAML.
            Values are parsed as YAML so `null`, `true`, `false`, `1.0` and lists work.
            Dots in `key` traverse nested dicts (e.g. `lr_schedule.start_val=0.0001`).
            Pass repeatedly: `--override steps=10 --override profile_memory=true`.

    Examples:
        pd-local tms_5-2           # Single GPU (default)
        pd-local tms_5-2 --cpu     # CPU only
        pd-local tms_5-2 --dp 4    # 4 GPUs on single node
        pd-local pile_llama_simple_mlp-4L --override steps=50 --override profile_memory=true
    """
    if experiment not in EXPERIMENT_REGISTRY:
        available = ", ".join(sorted(EXPERIMENT_REGISTRY.keys()))
        raise ValueError(f"Unknown experiment '{experiment}'. Available: {available}")

    if dp is not None and dp < 2:
        raise ValueError("--dp must be at least 2 for data parallelism")

    if cpu and dp is not None:
        raise ValueError("Cannot use both --cpu and --dp")

    overrides: list[str] = []
    if override is not None:
        overrides = [override] if isinstance(override, str) else list(override)

    exp_config = EXPERIMENT_REGISTRY[experiment]
    script_path = REPO_ROOT / exp_config.decomp_script
    config_path = REPO_ROOT / exp_config.config_path

    logger.info(f"Running experiment: {experiment}")
    logger.info(f"Config: {exp_config.config_path}")
    if overrides:
        logger.info(f"Overrides: {overrides}")

    config_args: list[str]
    if overrides:
        config_json = _build_overridden_config_json(str(config_path), overrides)
        config_args = ["--config_json", config_json]
    else:
        config_args = [str(config_path)]

    if dp is not None:
        # Multi-GPU: use torchrun
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(dp),
            str(script_path),
            *config_args,
        ]
    else:
        # Single GPU or CPU
        cmd = [
            sys.executable,
            str(script_path),
            *config_args,
        ]

    if cpu:
        env_prefix = "CUDA_VISIBLE_DEVICES="
        logger.info(f"Running: {env_prefix} {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env={"CUDA_VISIBLE_DEVICES": ""})
    else:
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
