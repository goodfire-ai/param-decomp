"""Local PD experiment runner"""

import subprocess
import sys

import fire

from param_decomp.log import logger
from param_decomp.registry import EXPERIMENT_REGISTRY
from param_decomp.settings import REPO_ROOT


def main(
    experiment: str,
    cpu: bool = False,
    dp: int | None = None,
) -> None:
    """Run a single PD experiment locally.

    Args:
        experiment: Experiment name from registry (e.g., 'tms_5-2', 'resid_mlp1')
        cpu: Run on CPU instead of GPU
        dp: Number of GPUs for single-node data parallelism (requires 2+)

    Examples:
        pd-local tms_5-2           # Single GPU (default)
        pd-local tms_5-2 --cpu     # CPU only
        pd-local tms_5-2 --dp 4    # 4 GPUs on single node
    """
    if experiment not in EXPERIMENT_REGISTRY:
        available = ", ".join(sorted(EXPERIMENT_REGISTRY.keys()))
        raise ValueError(f"Unknown experiment '{experiment}'. Available: {available}")

    if dp is not None and dp < 2:
        raise ValueError("--dp must be at least 2 for data parallelism")

    if cpu and dp is not None:
        raise ValueError("Cannot use both --cpu and --dp")

    exp_config = EXPERIMENT_REGISTRY[experiment]
    script_path = REPO_ROOT / exp_config.decomp_script
    config_path = REPO_ROOT / exp_config.config_path

    logger.info(f"Running experiment: {experiment}")
    logger.info(f"Config: {exp_config.config_path}")

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
            str(config_path),
        ]
    else:
        # Single GPU or CPU
        cmd = [
            sys.executable,
            str(script_path),
            str(config_path),
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
